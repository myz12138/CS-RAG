
import json
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from tqdm import tqdm
import os
from pathlib import Path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_BASE_URL"] = "https://hf-mirror.com"
import torch

import torch.nn.functional as F

from utils.remote_emb import EmbeddingClient

from transformers import AutoTokenizer, AutoModelForSequenceClassification



EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

EMB_DEVICE = os.getenv("EMB_DEVICE", "cpu")

enc = EmbeddingClient(EMB_MODEL, EMB_DEVICE)
DATASET = os.getenv("DATASET", "2wiki")

TOP_N_LLM = int(os.getenv("TOP_N_LLM", "7" if DATASET == "musique" else "5"))
RELATION_VARIANTS = int(os.getenv("RELATION_VARIANTS", "4" if DATASET == "musique" else "3"))
TOP_K_ENTITY = int(os.getenv("TOP_K_ENTITY", "3"))
RELATION_K = int(os.getenv("RELATION_K", "10" if DATASET == "musique" else "8"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LENGTH = int(
    os.getenv("PHASE1_RERANK_MAX_LENGTH", os.getenv("RERANK_MAX_LENGTH", "128"))
)
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "32"))
RERANK_FP16 = os.getenv("RERANK_FP16", "1") == "1"
INCLUDE_QUESTION_IN_RERANK = os.getenv("INCLUDE_QUESTION_IN_RERANK", "1") == "1"

NEFF_THRESHOLD = float(os.getenv("NEFF_THRESHOLD", {"musique": "1.5", "2wiki": "1.6", "hotpotqa": "2.0"}[DATASET]))

_DEFAULTS = {
    "2wiki": {
        "QUERY_JSON": "planner_queries/2wiki_data/query_graph_2wiki.json",
        "OUTPUT_JSON": "planner_queries/2wiki_data/phase1_evidence_v8_2wiki.json",
        "RAW_DATA_JSON": "dataset/2wikimultihopqa.json",
        "KG_PATH": "KGs/KG_2wiki",
    },
    "hotpotqa": {
        "QUERY_JSON": "planner_queries/hotpotqa_data/query_graph_hotpotqa.json",
        "OUTPUT_JSON": "planner_queries/hotpotqa_data/phase1_evidence_v8_hotpotqa.json",
        "RAW_DATA_JSON": "dataset/hotpotqa.json",
        "KG_PATH": "KGs/KG_hotpotqa",
    },
    "musique": {
        "QUERY_JSON": "planner_queries/musique_data/query_graph_musique.json",
        "OUTPUT_JSON": "planner_queries/musique_data/phase1_evidence_v8_musique.json",
        "RAW_DATA_JSON": "dataset/musique.json",
        "KG_PATH": "KGs/KG_musique",
    },
}

if DATASET not in _DEFAULTS:
    raise ValueError(f"Unsupported DATASET={DATASET}. Choose from: {list(_DEFAULTS.keys())}")

QUERY_JSON = os.getenv("QUERY_JSON", _DEFAULTS[DATASET]["QUERY_JSON"])
OUTPUT_JSON = os.getenv("OUTPUT_JSON", _DEFAULTS[DATASET]["OUTPUT_JSON"])
RAW_DATA_JSON = os.getenv("RAW_DATA_JSON", _DEFAULTS[DATASET]["RAW_DATA_JSON"])
KG_PATH = os.getenv("KG_PATH", _DEFAULTS[DATASET]["KG_PATH"])
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "1000"))

print("DATASET:", DATASET)
print("TOP_N_LLM:", TOP_N_LLM)
print("TOP_K_ENTITY:", TOP_K_ENTITY)
print("RELATION_K:", RELATION_K)
print("EMB_MODEL:", EMB_MODEL)
print("EMB_DEVICE:", EMB_DEVICE)
print("KG_PATH:", KG_PATH)
print("QUERY_JSON:", QUERY_JSON)
print("OUTPUT_JSON:", OUTPUT_JSON)
print("MAX_SAMPLES:", MAX_SAMPLES)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_raw_dataset_id2qa(dataset, path):
    """Load raw questions, answers, and candidate passages by question ID."""
    if not path or not os.path.exists(path):
        return {}

    data = _load_json(path)
    if not isinstance(data, list):
        return {}

    id2qa = {}
    for ex in data:
        if not isinstance(ex, dict):
            continue
        q = ex.get("question")
        ex_id = ex.get("id") or ex.get("_id") or ex.get("qid")
        if ex_id is None:
            continue
        ans = ex.get("answer")
        if dataset == "musique":
            ans = ex.get("answer")
        id2qa[str(ex_id)] = {"question": q, "answer": ans}
        if dataset == "musique":
            id2qa[str(ex_id)]["context_passages"] = ex.get("paragraphs", [])
            id2qa[str(ex_id)]["context_titles"] = [str(p.get("title") or "").strip() for p in ex.get("paragraphs", [])]
        if dataset in ("2wiki", "hotpotqa"):
            context = ex.get("context") or ex.get("contexts") or []
            id2qa[str(ex_id)]["context_titles"] = [
                str(c[0] if isinstance(c, list) else c.get("title", "")).strip()
                for c in context if isinstance(c, (list, dict)) and c
            ]
    return id2qa

def normalize_query_items(dataset, query_items, id2qa=None):
    """Normalize saved query plans and fill missing fields from raw data."""
    id2qa = id2qa or {}
    out = []
    for it in query_items:
        if not isinstance(it, dict):
            continue

        qid = it.get("id") or it.get("_id") or it.get("qid")
        if qid is None:
            continue
        qid = str(qid)

        q = it.get("question") or (id2qa.get(qid, {}) or {}).get("question")
        ans = it.get("ground_truth_answer")
        if ans is None:
            ans = it.get("answer")
        if ans is None:
            ans = (id2qa.get(qid, {}) or {}).get("answer")

        qp = it.get("query_plan")
        if qp is None:
            qp = it.get("query_triples")
        if qp is None:
            qp = it.get("triples")
        if qp is None:
            qg = it.get("query_graph") or {}
            if isinstance(qg, dict):
                qp = qg.get("query_plan") or qg.get("triples")

        if not isinstance(qp, list):
            raise ValueError(f"Item {qid} has no query_plan/triples list. Provide query_graph output as input.")

        out.append(
            {
                "id": qid,
                "question": q,
                "ground_truth_answer": ans,
                "query_plan": qp,
                "context_titles": (id2qa.get(qid, {}) or {}).get("context_titles"),
                "context_passages": (id2qa.get(qid, {}) or {}).get("context_passages"),
            }
        )
    return out


def load_kg_from_jsonl(out_dir: str):
    out_dir = Path(out_dir)
    entities = []
    triples = []
    title2entities = {}
    title2triples = {}

    with (out_dir / "entities.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entities.append(json.loads(line))

    with (out_dir / "triples.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triples.append(json.loads(line))

    with (out_dir / "title2entities.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                title2entities[obj["title"]] = obj["entity_ids"]

    with (out_dir / "title2triples.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                title2triples[obj["title"]] = obj["triple_idxs"]

    return {
        "entities": entities,
        "triples": triples,
        "title2entities": title2entities,
        "title2triples": title2triples,
    }


def load_kg(kg_path):
    """Load a KG from a serialized file or JSONL directory."""
    if isinstance(kg_path, str) and kg_path.endswith(".pt"):
        import torch
        return torch.load(kg_path)

    if isinstance(kg_path, str) and (kg_path.endswith(".json") or kg_path.endswith(".jsonl")) and os.path.isfile(kg_path):
        kg = _load_json(kg_path)
        if isinstance(kg, dict) and "entities" in kg and "triples" in kg:
            return kg

    try:
        return load_kg_from_jsonl(kg_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load KG from {kg_path}. "
            f"Provide a .pt KG or ensure read_hotpotqa.load_kg_from_jsonl is available. Error: {e}"
        )

def is_var(x):
    return isinstance(x, str) and x.startswith("?")

def ent_name(e):
    if isinstance(e, dict):
        if e.get("name"):
            return e["name"]
        if e.get("surface"):
            return e["surface"]
    return str(e)

def parse_triple(tr, entities):
    if isinstance(tr, dict):
        h = tr.get("head", tr.get("h"))
        r = tr.get("relation", tr.get("r", ""))
        t = tr.get("tail", tr.get("t"))
    else:
        h, r, t = tr

    h_id = h if isinstance(h, int) and 0 <= h < len(entities) else None
    t_id = t if isinstance(t, int) and 0 <= t < len(entities) else None
    h_name = ent_name(entities[h_id]) if h_id is not None else str(h)
    t_name = ent_name(entities[t_id]) if t_id is not None else str(t)
    return h_id, str(r), t_id, h_name, t_name

def build_kg_indices(kg):
    entities = kg["entities"]
    triples_raw = kg["triples"]

    triple_info = []
    out_index = defaultdict(list)
    in_index = defaultdict(list)

    for i, tr in enumerate(triples_raw):
        h_id, r, t_id, h_name, t_name = parse_triple(tr, entities)
        triple_info.append(
            {
                "head_id": h_id,
                "tail_id": t_id,
                "relation": r,
                "head": h_name,
                "tail": t_name,
                "title": tr.get("title") if isinstance(tr, dict) else None,
                "evidence": tr.get("evidence") if isinstance(tr, dict) else None,
            }
        )
        if isinstance(h_id, int):
            out_index[h_id].append(i)
        if isinstance(t_id, int):
            in_index[t_id].append(i)

    entity_names = [ent_name(e) for e in entities]
    return entities, entity_names, triple_info, out_index, in_index

def str_sim(a, b):
    qa = set((a or "").lower().split())
    kb = set((b or "").lower().split())
    return len(qa & kb) / len(qa) if qa else 0.0

def text_sim(a, b):
    """Cosine similarity between embeddings of two texts, with a simple in-process cache."""
    if not a or not b:
        return 0.0
    a = str(a)
    b = str(b)

    if not hasattr(text_sim, "_cache"):
        text_sim._cache = {}
        text_sim._max_cache = 50000

    cache = text_sim._cache

    def get_vec(x):
        v = cache.get(x)
        if v is not None:
            return v
        vec = enc.encode_one(x)
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)
        vec = F.normalize(vec.unsqueeze(0), p=2, dim=-1).squeeze(0)

        if len(cache) >= text_sim._max_cache:
            try:
                cache.pop(next(iter(cache)))
            except Exception:
                cache.clear()

        cache[x] = vec
        return vec

    v1 = get_vec(a)
    v2 = get_vec(b)
    return float(torch.dot(v1, v2).item())

def has_token_overlap(a, b):
    ta = [t for t in (a or "").lower().split() if t]
    tb = [t for t in (b or "").lower().split() if t]
    return len(set(ta) & set(tb)) > 0

def map_query_entities(query_triples, entity_names, allowed_entity_ids=None):
    """Match explicit query entities to KG entities by token overlap."""
    ents = set()
    for qt in query_triples:
        h = qt.get("head")
        t = qt.get("tail")
        if not is_var(h):
            ents.add(h)
        if not is_var(t):
            ents.add(t)

    mapping = {}
    for qe in ents:
        scored = []
        for idx, name in enumerate(entity_names):
            if allowed_entity_ids is not None and idx not in allowed_entity_ids:
                continue
            if not has_token_overlap(qe, name):
                continue
            s = str_sim(qe, name)
            scored.append((s, idx))
        if not scored:
            continue
        scored.sort(reverse=True)
        mapping[qe] = [idx for s, idx in scored[:TOP_K_ENTITY]]

    return list(ents), mapping

def map_surface_to_entity_ids(surface, entity_names):
    surface = str(surface)
    scored = []
    for idx, name in enumerate(entity_names):
        if not has_token_overlap(surface, name):
            continue
        s = str_sim(surface, name)
        scored.append((s, idx))
    if not scored:
        return []
    scored.sort(reverse=True)
    return [idx for s, idx in scored[:TOP_K_ENTITY]]

def get_1hop_triples(eid, out_index, in_index):
    return list(set(out_index.get(eid, []) + in_index.get(eid, [])))

def _clean_relation_variants(qt_effective):
    q_rel = str(qt_effective.get("relation", "") or "").strip()
    variants= qt_effective.get("relation_variants", [])
    if not isinstance(variants, list):
        variants = []
    variants_clean = [str(v).strip() for v in variants if isinstance(v, str) and str(v).strip()]
    if q_rel:
        if q_rel in variants_clean:
            variants_clean.remove(q_rel)
        variants_clean.insert(0, q_rel)
    variants_clean = list(dict.fromkeys(variants_clean))
    if not variants_clean:
        raise ValueError("The constraint has no valid relation phrase. Regenerate the query plan.")
    return variants_clean[:max(1, RELATION_VARIANTS)]

def relation_score(qt_effective, kg_relation):
    variants_clean = _clean_relation_variants(qt_effective)
    if not variants_clean:
        return 0.0
    kg_rel = str(kg_relation or "")
    if not kg_rel:
        return 0.0
    return float(max(text_sim(v, kg_rel) for v in variants_clean))

def top_by_relation(qt_effective, triple_info, cand_ids, k=8):
    scored = []
    score_map = {}
    for tid in cand_ids:
        kg_rel = str(triple_info[tid]["relation"] or "")
        s = relation_score(qt_effective, kg_rel)
        score_map[int(tid)] = float(s)
        scored.append((s, tid))
    scored.sort(reverse=True)
    top_pairs = scored[:k]
    top =[tid for s, tid in top_pairs]
    top_score_map = {int(tid): float(score_map.get(int(tid), 0.0)) for tid in top}
    return top, top_score_map



def build_reranker():
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise ImportError(
            "transformers is required for reranker. Please install: pip install transformers"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
    mdl.eval()
    mdl.to(device)
    if device.startswith("cuda") and RERANK_FP16:
        mdl.half()
    return {"tokenizer": tok, "model": mdl, "device": device}


def _rerank_score_pairs(reranker, pairs):
    """pairs: List[Tuple[str, str]] -> List[float]"""
    tok = reranker["tokenizer"]
    mdl = reranker["model"]
    device = reranker["device"]
    scores = []
    bs = max(1, int(RERANK_BATCH))
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            qs = [p[0] for p in batch]
            ps = [p[1] for p in batch]
            encd = tok(
                qs,
                ps,
                padding=True,
                truncation=True,
                max_length=RERANK_MAX_LENGTH,
                return_tensors="pt",
            )
            encd = {k: v.to(device) for k, v in encd.items()}
            out = mdl(**encd)
            logits = out.logits
            if logits.ndim == 2 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            logits = logits.float().detach().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            scores.extend([float(x) for x in logits])
    return scores


def rerank_select_topn(reranker, question, query_triple, candidates, top_n=TOP_N_LLM):
    """Rerank candidate triples using the question and current constraint."""
    if not candidates:
        return [], [], {}

    qh = str(query_triple.get("head", "") or "").strip()
    qt = str(query_triple.get("tail", "") or "").strip()
    qr = str(query_triple.get("relation", "") or "").strip()
    q_triple_str = " ".join([x for x in [qh, qr, qt] if x]).strip()

    q_prefix = (str(question or "").strip() + " ") if INCLUDE_QUESTION_IN_RERANK else ""
    query_text = ('question: '+q_prefix +' UnknownTriple: '+ q_triple_str).strip()

    pairs = []
    for c in candidates:
        ch = str(c.get("head", "") or "").strip()
        cr = str(c.get("relation", "") or "").strip()
        ct = str(c.get("tail", "") or "").strip()
        cand_str = " ".join([x for x in [ch, cr, ct] if x]).strip()
        passage = (f" {cand_str}").strip()
        pairs.append((query_text, passage))

    scores = _rerank_score_pairs(reranker, pairs)
    
    scored = [(float(scores[i]), i) for i in range(len(scores))]
    if not scored:
        return [], [], {}
    scored.sort(reverse=True)
    keep = scored[: max(1, int(top_n))]
    selected_indices = [idx for s, idx in keep]
    selected_triple_ids = [candidates[idx]["triple_id"] for idx in selected_indices]
    score_map = {int(candidates[i]["triple_id"]): float(scores[i]) for i in range(len(candidates))}
    return selected_triple_ids, selected_indices, score_map


def process_item(item, entities, entity_names, triple_info, out_index, in_index, client):
    question = item["question"]
    qtriples = item["query_plan"]
    allowed_tids = None
    allowed_entities = None
    if DATASET in ("2wiki", "hotpotqa", "musique"):
        titles = item.get("context_titles")
        if titles is None:
            raw = load_raw_dataset_id2qa(DATASET, RAW_DATA_JSON)
            titles = raw.get(str(item["id"]), {}).get("context_titles")
        if titles is None:
            raise ValueError(f"Missing candidate passages for query {item['id']}")
        titles = set(titles)
        allowed_tids = {i for i, tr in enumerate(triple_info) if tr.get("title") in titles}
        if DATASET == "musique":
            passages = item.get("context_passages")
            if passages is None:
                raw = load_raw_dataset_id2qa(DATASET, RAW_DATA_JSON)
                passages = raw.get(str(item["id"]), {}).get("context_passages")
            if passages is None:
                raise ValueError(f"Missing MuSiQue paragraphs for query {item['id']}")
            by_title = defaultdict(list)
            for passage in passages:
                text = " ".join(re.findall(r"\w+", str(passage.get("paragraph_text") or "").casefold()))
                by_title[str(passage.get("title") or "").strip()].append(f" {text} ")
            allowed_tids = {tid for tid in allowed_tids if triple_info[tid].get("evidence") and any(
                " " + " ".join(re.findall(r"\w+", str(triple_info[tid]["evidence"]).casefold())) + " " in text
                for text in by_title[triple_info[tid]["title"]])}
        allowed_entities = set()
        for tid in allowed_tids:
            allowed_entities.update([triple_info[tid]["head_id"], triple_info[tid]["tail_id"]])

    known_entities, emap = map_query_entities(qtriples, entity_names, allowed_entities)
    support = defaultdict(list)
    rel_score_selected_map = defaultdict(dict)
    rerank_score_selected_map = defaultdict(dict)
    var_candidates_per_triple = defaultdict(lambda: defaultdict(set))
    var_triple_entity = defaultdict(lambda: defaultdict(dict))
    triples_debug = {}
    dependencies = {}
    candidate_bindings = defaultdict(lambda: defaultdict(list))
    passed = set()
    var_final = {}

    def run_retrieval(qi, qt_effective, anchor_nodes, stage_tag, fixed_ids=None):
        cand_ids_all = set()
        for nid in anchor_nodes:
            cand_ids_all.update(get_1hop_triples(nid, out_index, in_index))
        if allowed_tids is not None:
            cand_ids_all &= allowed_tids
        if fixed_ids is not None:
            hb, tb = fixed_ids
            cand_ids_all = {tid for tid in cand_ids_all if
                (triple_info[tid]["head_id"], triple_info[tid]["tail_id"]) in ((hb, tb), (tb, hb))}
        cand_ids_top, rel_scores = top_by_relation(qt_effective, triple_info, sorted(cand_ids_all), k=RELATION_K)
        candidates = [{"triple_id": tid, "head": triple_info[tid]["head"],
                       "relation": triple_info[tid]["relation"], "tail": triple_info[tid]["tail"]}
                      for tid in cand_ids_top]
        tids, indices, scores = rerank_select_topn(client, question, qt_effective, candidates, top_n=TOP_N_LLM)
        for tid in tids:
            if tid not in support[qi]:
                support[qi].append(tid)
            rel_score_selected_map[qi][tid] = max(rel_score_selected_map[qi].get(tid, float("-inf")), rel_scores[tid])
            rerank_score_selected_map[qi][tid] = max(rerank_score_selected_map[qi].get(tid, float("-inf")), scores[tid])
        debug = triples_debug.setdefault(qi, {"query_triple_index": qi, "stage": stage_tag,
            "anchors": [], "candidate_ids_topk": [], "llm_selected_triple_ids": [],
            "triple_type_initial": "double_var" if all(is_var(qtriples[qi].get(k)) for k in ("head", "tail")) else "single_var_with_anchor"})
        debug["stage"] = stage_tag
        debug.setdefault("anchors", [])
        debug.setdefault("candidate_ids_topk", [])
        debug.setdefault("triple_type_initial", "double_var")
        for nid in anchor_nodes:
            record = {"entity_id": nid, "entity_name": entity_names[nid]}
            if record not in debug["anchors"]:
                debug["anchors"].append(record)
        debug["candidate_ids_topk"] = sorted(set(debug["candidate_ids_topk"]) | set(cand_ids_top))
        return tids

    def extract_var_candidates(qi, qt_effective, anchors, tids, bound=None):
        bound = bound or {}
        anchor_set = set(anchors)
        for tid in tids:
            tr = triple_info[tid]
            opposite = set()
            if tr["head_id"] in anchor_set:
                opposite.add(tr["tail_id"])
            if tr["tail_id"] in anchor_set:
                opposite.add(tr["head_id"])
            opposite = {eid for eid in opposite if isinstance(eid, int)}
            for side in ("head", "tail"):
                v = qtriples[qi].get(side)
                if not is_var(v):
                    continue
                eids = {bound[v]} if v in bound else opposite
                var_triple_entity[v][qi].setdefault(tid, set()).update(eids)
            unbound = {qtriples[qi].get(side) for side in ("head", "tail")
                       if is_var(qtriples[qi].get(side)) and qtriples[qi].get(side) not in bound}
            bindings = [dict(bound, **{v: eid}) for v in unbound for eid in opposite] if unbound else [dict(bound)]
            for assignment in bindings:
                if assignment not in candidate_bindings[qi][tid]:
                    candidate_bindings[qi][tid].append(assignment)

    def check_pool(qi):
        tids = sorted(support[qi], key=lambda tid: (-rerank_score_selected_map[qi][tid], tid))[:max(1, TOP_N_LLM)]
        support[qi] = tids
        variables = {qt for qt in (qtriples[qi].get("head"), qtriples[qi].get("tail")) if is_var(qt)}
        for v in variables:
            maps = var_triple_entity[v].get(qi, {})
            var_triple_entity[v][qi] = {tid: maps[tid] for tid in tids if tid in maps}
            var_candidates_per_triple[v][qi] = set().union(*var_triple_entity[v][qi].values()) if var_triple_entity[v][qi] else set()
        logits = [rerank_score_selected_map[qi][tid] for tid in tids]
        if logits:
            weights = [math.exp(x - max(logits)) for x in logits]
            total = sum(weights)
            neff = total * total / sum(w * w for w in weights)
        else:
            neff = float("inf")
        resolved = bool(tids) and neff <= NEFF_THRESHOLD and all(var_candidates_per_triple[v][qi] for v in variables)
        if resolved:
            passed.add(qi)
        triples_debug.setdefault(qi, {"query_triple_index": qi})["neff"] = neff if math.isfinite(neff) else None
        triples_debug[qi]["checked"] = True
        triples_debug[qi]["llm_selected_triple_ids"] = tids

    def consolidate():
        active = set(passed)
        pools = {qi: list(support[qi]) for qi in active}
        bindings = {qi: {tid: list(candidate_bindings[qi][tid]) for tid in pools[qi]} for qi in active}
        while True:
            per_var = defaultdict(dict)
            for qi in active:
                variables = {v for v in (qtriples[qi].get("head"), qtriples[qi].get("tail")) if is_var(v)}
                for v in variables:
                    per_var[v][qi] = {binding[v] for tid in pools[qi] for binding in bindings[qi][tid]}
            final = {}
            rejected = set()
            for v, per_qi in per_var.items():
                sets = list(per_qi.values())
                inter = set.intersection(*sets)
                if not inter:
                    rejected.update(per_qi)
                    continue
                ranked = []
                for eid in inter:
                    score = sum(max(rerank_score_selected_map[qi][tid] for tid in pools[qi]
                        if any(binding[v] == eid for binding in bindings[qi][tid])) for qi in per_qi)
                    ranked.append((score, eid))
                ranked.sort(reverse=True)
                final[v] = {"union_ids": sorted(set.union(*sets)),
                    "intersection_ids": sorted(inter) if len(sets) > 1 else None,
                    "primary_ids": [eid for score, eid in ranked],
                    "primary_ranked": [{"entity_id": eid, "entity_name": entity_names[eid], "score": score} for score, eid in ranked]}
            for qi in active:
                for v, sources in dependencies.get(qi, {}).items():
                    if not (sources & active):
                        rejected.add(qi)
            if rejected:
                active -= rejected
                pools = {qi: list(support[qi]) for qi in active}
                bindings = {qi: {tid: list(candidate_bindings[qi][tid]) for tid in pools[qi]} for qi in active}
                continue
            changed = False
            for qi in active:
                kept = []
                for tid in pools[qi]:
                    assignments = [binding for binding in bindings[qi][tid]
                        if all(eid in final[v]["primary_ids"] for v, eid in binding.items())]
                    if assignments != bindings[qi][tid]:
                        changed = True
                        bindings[qi][tid] = assignments
                    if assignments:
                        kept.append(tid)
                if kept != pools[qi]:
                    pools[qi] = kept
                    changed = True
                if not kept:
                    rejected.add(qi)
            if rejected:
                active -= rejected
                pools = {qi: list(support[qi]) for qi in active}
                bindings = {qi: {tid: list(candidate_bindings[qi][tid]) for tid in pools[qi]} for qi in active}
                continue
            if not changed:
                break
        passed.intersection_update(active)
        for qi in list(support):
            support[qi] = pools[qi] if qi in active else []
        for qi in active:
            candidate_bindings[qi] = {tid: bindings[qi][tid] for tid in pools[qi]}
            for v in per_var:
                if qi in per_var[v]:
                    var_triple_entity[v][qi] = {tid: {binding[v] for binding in bindings[qi][tid]} for tid in pools[qi]}
        for v, per_qi in var_candidates_per_triple.items():
            for qi in list(per_qi):
                if qi not in active:
                    del per_qi[qi]
                else:
                    per_qi[qi] = set().union(*(var_triple_entity[v][qi].get(tid, set()) for tid in pools[qi]))
        for qi, qt in enumerate(qtriples):
            debug = triples_debug.setdefault(qi, {"query_triple_index": qi, "checked": False})
            debug["status"] = "resolved" if qi in active else "unresolved"
            debug["llm_selected_triple_ids"] = support.get(qi, [])
        return final

    pending = []
    for qi, qt in enumerate(qtriples):
        hv, tv = is_var(qt.get("head")), is_var(qt.get("tail"))
        if hv and tv:
            pending.append(qi)
            continue
        if not (hv or tv):
            continue
        anchors = emap.get(qt.get("tail") if hv else qt.get("head"), [])
        tids = run_retrieval(qi, qt, anchors, "first_pass")
        extract_var_candidates(qi, qt, anchors, tids)
        check_pool(qi)
    var_final = consolidate()

    while pending:
        deferred = []
        for qi in pending:
            qt = qtriples[qi]
            h, t = qt.get("head"), qt.get("tail")
            hs = var_final.get(h, {}).get("primary_ids", [])
            ts = var_final.get(t, {}).get("primary_ids", [])
            if not hs and not ts:
                deferred.append(qi)
                continue
            bound_vars = [v for v, candidates in ((h, hs), (t, ts)) if candidates]
            dependencies[qi] = {v: {j for j in passed if j in var_candidates_per_triple[v]} for v in bound_vars}
            for hb in hs or [None]:
                for tb in ts or [None]:
                    if h == t and hb != tb:
                        continue
                    bound = {v: eid for v, eid in ((h, hb), (t, tb)) if eid is not None}
                    effective = dict(qt)
                    effective["head"] = ent_name(entities[hb]) if hb is not None else h
                    effective["tail"] = ent_name(entities[tb]) if tb is not None else t
                    anchors = [hb] if hb is not None else [tb]
                    fixed = (hb, tb) if hb is not None and tb is not None else None
                    tids = run_retrieval(qi, effective, anchors, "second_pass", fixed)
                    extract_var_candidates(qi, effective, anchors, tids, bound)
            check_pool(qi)
        if deferred == pending:
            break
        pending = deferred
        var_final = consolidate()

    evidence_ids = sorted({tid for tids in support.values() for tid in tids})
    idx_map = {tid: i for i, tid in enumerate(evidence_ids)}
    evidence_triples = []
    for tid in evidence_ids:
        tr = triple_info[tid]
        evidence_triples.append({"kg_triple_id": tid, "head_entity_id": tr["head_id"],
            "tail_entity_id": tr["tail_id"], "head": tr["head"], "relation": tr["relation"],
            "tail": tr["tail"], "title": tr.get("title"), "evidence": tr.get("evidence")})
    support_list = [{"query_triple_index": qi, "evidence_indices": [idx_map[tid] for tid in tids],
        "rerank_scores": {str(tid): rerank_score_selected_map[qi][tid] for tid in tids}}
        for qi, tids in support.items() if tids]
    debug_info = {"known_entities": known_entities,
        "entity_map": [{"query_entity": qe, "kg_entity_ids": ids,
            "kg_entity_names": [entity_names[eid] for eid in ids]} for qe, ids in emap.items()],
        "variable_candidates": {v: {key: info[key] for key in ("union_ids", "intersection_ids", "primary_ranked")}
            for v, info in var_final.items()},
        "variable_candidates_per_triple": {v: {str(qi): sorted(eids) for qi, eids in per_qi.items()}
            for v, per_qi in var_candidates_per_triple.items() if per_qi},
        "triples_debug": [triples_debug[qi] for qi in sorted(triples_debug)]}
    return {"id": item["id"], "question": question, "ground_truth_answer": item.get("ground_truth_answer"),
        "query_plan": qtriples, "evidence_triples": evidence_triples, "support": support_list, "debug": debug_info}


def main_1():
    kg = load_kg(KG_PATH)
    entities, entity_names, triple_info, out_index, in_index = build_kg_indices(kg)

    raw_id2qa = load_raw_dataset_id2qa(DATASET, RAW_DATA_JSON)
    query_items = _load_json(QUERY_JSON)
    if not isinstance(query_items, list):
        raise ValueError("QUERY_JSON must be a JSON list (query_graph builder output).")
    if MAX_SAMPLES > 0:
        query_items = query_items[:MAX_SAMPLES]

    query_items = normalize_query_items(DATASET, query_items, id2qa=raw_id2qa)

    client = build_reranker()

    out_dir = os.path.dirname(OUTPUT_JSON)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, item in enumerate(tqdm(query_items, desc=f"Phase-1 v6 universal ({DATASET})")):
            res = process_item(item, entities, entity_names, triple_info, out_index, in_index, client)
            if i > 0:
                f.write(",\n")
            json.dump(res, f, ensure_ascii=False, indent=2)
            f.flush()
        f.write("\n]")

    print("Saved:", OUTPUT_JSON)


if __name__ == "__main__":
    main_1()
