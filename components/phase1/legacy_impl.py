
import json
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

#try:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# except Exception as _e:
#     AutoTokenizer = None
#     AutoModelForSequenceClassification = None



EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

EMB_DEVICE = os.getenv("EMB_DEVICE", "cpu")

enc = EmbeddingClient(EMB_MODEL, EMB_DEVICE)
DATASET = os.getenv("DATASET", "2wiki")

TOP_N_LLM = int(os.getenv("TOP_N_LLM", "3"))
TOP_K_ENTITY = int(os.getenv("TOP_K_ENTITY", "3"))
RELATION_K = int(os.getenv("RELATION_K", "8"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "128"))
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "32"))
RERANK_FP16 = os.getenv("RERANK_FP16", "1") == "1"
INCLUDE_QUESTION_IN_RERANK = os.getenv("INCLUDE_QUESTION_IN_RERANK", "1") == "1"

NEFF_THRESHOLD = float(os.getenv("NEFF_THRESHOLD", "1.5"))

_DEFAULTS = {
    "2wiki": {
        "QUERY_JSON": "planned_queries/2wiki_data/query_graph_v8_2wiki.json",
        "OUTPUT_JSON": "planned_queries/2wiki_data/phase1_evidence_v8_2wiki.json",
        "RAW_DATA_JSON": "dataset/2wikimultihopqa.json",
        "KG_PATH": "artifacts/kg_perturbed/missing_edges_10/KGs/KG_2wiki",
    },
    "hotpotqa": {
        "QUERY_JSON": "planned_queries/hotpotqa_data/query_graph_v8_hotpotqa.json",
        "OUTPUT_JSON": "planned_queries/hotpotqa_data/phase1_evidence_v8_hotpotqa.json",
        "RAW_DATA_JSON": "dataset/hotpotqa.json",
        "KG_PATH": "KGs/KG_hotpotqa",
    },
    "musique": {
        "QUERY_JSON": "planned_queries/musique_data/query_graph_v8_musique.json",
        "OUTPUT_JSON": "planned_queries/musique_data/phase1_evidence_v8_musique.json",
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
    """
    Optional: load raw dataset to fill missing fields (question/answer) if query_graph lacks them.
    Returns: id(str) -> {"question":..., "answer":...}
    """
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
    return id2qa

def normalize_query_items(dataset, query_items, id2qa=None):
    """
    Normalize query_graph items to the schema expected by process_item:
      {"id", "question", "ground_truth_answer", "query_plan"}
    Accepts several common key variants.
    """
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
    """
    Supports:
      - .pt (torch.save)
      - json / dict with keys {"entities","triples"}
      - jsonl directory name/path via read_hotpotqa.load_kg_from_jsonl
    """
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
            }
        )
        if isinstance(h_id, int):
            out_index[h_id].append(i)
        if isinstance(t_id, int):
            in_index[t_id].append(i)

    entity_names = [ent_name(e) for e in entities]
    return entities, entity_names, triple_info, out_index, in_index

def str_sim(a, b):
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

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

def map_query_entities(query_triples, entity_names):
    """
    Explicit (non-variable) entities in query -> top-k KG entity ids (token-overlap filtered).
    """
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
    q_rel = str(qt_effective.get("relation", "") or "")
    variants= qt_effective.get("relation_variants", [])
    if not isinstance(variants, list):
        variants = []
    variants_clean = [str(v).strip() for v in variants if isinstance(v, str) and str(v).strip()]
    if q_rel:
        if q_rel in variants_clean:
            variants_clean.remove(q_rel)
        variants_clean.insert(0, q_rel)
    return variants_clean

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
    """
    Rerank Stage-1 top-k candidate triples.

    Following your requirement, the reranker input uses a concatenation of:
      - the (possibly unresolved) query triple expression
      - the candidate triple expression
    The original question is also prepended (configurable).
    """
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

    known_entities, emap = map_query_entities(qtriples, entity_names)

    triple_meta = []
    for qi, qt in enumerate(qtriples):
        h = qt.get("head")
        t = qt.get("tail")
        is_hv = is_var(h)
        is_tv = is_var(t)

        anchor_nodes = []
        if not is_hv and h in emap:
            anchor_nodes.extend(emap[h])
        if not is_tv and t in emap:
            for eid in emap[t]:
                if eid not in anchor_nodes:
                    anchor_nodes.append(eid)

        has_anchor = len(anchor_nodes) > 0
        if is_hv and is_tv:
            ttype = "double_var"
        elif (is_hv ^ is_tv) and has_anchor:
            ttype = "single_var_with_anchor"
        elif (not is_hv and not is_tv) and has_anchor:
            ttype = "no_var_with_anchor"
        else:
            ttype = "no_anchor"

        triple_meta.append({"index": qi, "anchor_nodes_initial": anchor_nodes, "triple_type_initial": ttype})

    support = defaultdict(list)
    rel_score_selected_map = defaultdict(dict)
    rerank_score_selected_map = defaultdict(dict)

    var_candidates_per_triple = defaultdict(lambda: defaultdict(set))
    var_triple_entity = defaultdict(lambda: defaultdict(dict))

    triples_debug = {}

    def run_retrieval(qi, qt_effective, anchor_nodes, stage_tag, triple_type_initial):
        cand_ids_all = set()
        for nid in anchor_nodes or []:
            cand_ids_all.update(get_1hop_triples(nid, out_index, in_index))
        cand_ids_all = sorted(cand_ids_all)

        cand_ids_top, top_score_map = top_by_relation(qt_effective, triple_info, cand_ids_all, k=RELATION_K) if cand_ids_all else ([], {})

        selected_tids, selected_indices, rerank_score_map = ([], [], {})
        if cand_ids_top:
            llm_candidates = [
                {"triple_id": tid, "head": triple_info[tid]["head"], "relation": triple_info[tid]["relation"], "tail": triple_info[tid]["tail"]}
                for tid in cand_ids_top
            ]
            selected_tids, selected_indices, rerank_score_map = rerank_select_topn(
                client, question, qt_effective, llm_candidates, top_n=TOP_N_LLM
            )
        
        if selected_tids:
            for tid in selected_tids:
                if tid not in support[qi]:
                    support[qi].append(tid)
                rel_score_selected_map[qi][int(tid)] = float(top_score_map.get(int(tid), 0.0))
                rerank_score_selected_map[qi][int(tid)] = float(rerank_score_map.get(int(tid), 0.0))

        triples_debug[qi] = {
            "query_triple_index": int(qi),
            "stage": stage_tag,
            "triple_type_initial": triple_type_initial,
            "anchors": [{"entity_id": int(nid), "entity_name": entity_names[nid]} for nid in (anchor_nodes or [])],
            "candidate_ids_topk": [int(tid) for tid in cand_ids_top],
            "llm_selected_triple_ids": [int(tid) for tid in selected_tids],
            "llm_selected_indices_in_topk": selected_indices,
        }
        return [int(tid) for tid in selected_tids]

    def extract_var_candidates_from_selected(qi, qt, anchor_nodes, selected_tids):
        h = qt.get("head")
        t = qt.get("tail")
        is_hv = is_var(h)
        is_tv = is_var(t)

        if not (is_hv ^ is_tv):
            return

        var_name = h if is_hv else t
        anchor_set = set(anchor_nodes or [])

        for tid in selected_tids:
            info = triple_info[tid]
            hi = info["head_id"]
            ti = info["tail_id"]
            eids_for_this_tid = set()

            if isinstance(hi, int) and hi in anchor_set and isinstance(ti, int) and ti not in anchor_set:
                eids_for_this_tid.add(int(ti))
            if isinstance(ti, int) and ti in anchor_set and isinstance(hi, int) and hi not in anchor_set:
                eids_for_this_tid.add(int(hi))

            if not eids_for_this_tid:
                continue

            var_candidates_per_triple[var_name][qi].update(eids_for_this_tid)
            if tid not in var_triple_entity[var_name][qi]:
                var_triple_entity[var_name][qi][tid] = set()
            var_triple_entity[var_name][qi][tid].update(eids_for_this_tid)

    def rank_var_candidates(v, cand_eids, per_triple_sets):
        def best_score_for_eid_on_qi(eid, qi):
            best = 0.0
            tid_map = var_triple_entity.get(v, {}).get(qi, {})
            for tid, eids in tid_map.items():
                if eid in eids:
                    s = float(rerank_score_selected_map.get(qi, {}).get(int(tid), 0.0))
                    if s > best:
                        best = s
            return best

        scored = []
        qids = sorted(per_triple_sets.keys())
        for eid in cand_eids:
            if not qids:
                s = 0.0
            elif len(qids) == 1:
                s = best_score_for_eid_on_qi(eid, qids[0])
            else:
                s = 0.0
                for qi in qids:
                    if eid in per_triple_sets[qi]:
                        s += best_score_for_eid_on_qi(eid, qi)
            scored.append((float(s), int(eid)))
        scored.sort(reverse=True)
        return scored

    def compute_final_var_candidate_sets():
        final = {}
        for v, per_triple in var_candidates_per_triple.items():
            union_set = set()
            for s in per_triple.values():
                union_set.update(set(s))

            inter_set = None
            if len(per_triple) > 1:
                sets = [set(s) for s in per_triple.values()]
                inter_set = set.intersection(*sets) if sets else set()

            primary = set(inter_set) if (inter_set is not None and len(inter_set) > 0) else set(union_set)
            ranked = rank_var_candidates(v, primary, per_triple)

            final[v] = {
                "union_ids": sorted([int(eid) for eid in union_set]),
                "intersection_ids": sorted([int(eid) for eid in inter_set]) if inter_set is not None else None,
                "primary_ranked": [{"entity_id": int(eid), "entity_name": entity_names[eid], "score": float(s)} for s, eid in ranked],
                "primary_ids": [int(eid) for s, eid in ranked],
            }
        return final

    def _neff_from_logits(logits):
        """Effective number of candidates (Neff) using linear-normalized logits (NO softmax / exp)."""
        if not logits:
            return float("inf")
        xs = [float(x) for x in logits]
        mn = min(xs)
        if mn <= 0.0:
            ws = [x - mn + 1e-6 for x in xs]
        else:
            ws = xs
        z = sum(ws)
        if z <= 0.0:
            return float("inf")
        ps = [w / z for w in ws]
        return 1.0 / sum(p * p for p in ps)

    def apply_neff_gating(var_final_dict):
        """If a variable is not solvable (Neff too large), drop it entirely (no candidates output)."""
        to_drop = []
        for v, info in (var_final_dict or {}).items():
            ranked = info.get("primary_ranked", []) or []
            ranked = ranked[: max(1, int(TOP_N_LLM))]
            logits = [float(x.get("score")) for x in ranked]
            neff = _neff_from_logits(logits)
            if (not ranked) or (neff > float(NEFF_THRESHOLD)):
                to_drop.append(v)
            else:
                info["primary_ranked"] = ranked
                info["primary_ids"] = [int(x["entity_id"]) for x in ranked]
        for v in to_drop:
            var_final_dict.pop(v, None)
            var_candidates_per_triple.pop(v, None)
            var_triple_entity.pop(v, None)
        return var_final_dict

    for meta in triple_meta:
        qi = meta["index"]
        qt = qtriples[qi]
        anchor_nodes = meta["anchor_nodes_initial"]
        ttype = meta["triple_type_initial"]

        if not anchor_nodes:
            continue
        if ttype == "double_var":
            continue

        selected_tids = run_retrieval(qi, qt, anchor_nodes, "first_pass", ttype)

        if ttype == "single_var_with_anchor" and selected_tids:
            extract_var_candidates_from_selected(qi, qt, anchor_nodes, selected_tids)

    var_final = compute_final_var_candidate_sets()
    var_final = apply_neff_gating(var_final)

    var_candidate_anchors = defaultdict(dict)
    for v, info in var_final.items():
        for eid in info.get("primary_ids", []):
            anchor_ids = set([int(eid)])
            surf = ent_name(entities[int(eid)]) if 0 <= int(eid) < len(entities) else str(eid)
            for mid in map_surface_to_entity_ids(surf, entity_names):
                anchor_ids.add(int(mid))
            var_candidate_anchors[v][int(eid)] = sorted(anchor_ids)

    for meta in triple_meta:
        qi = meta["index"]
        if meta["triple_type_initial"] != "double_var":
            continue

        qt = qtriples[qi]
        h = qt.get("head")
        t = qt.get("tail")

        h_var = h if is_var(h) else None
        t_var = t if is_var(t) else None

        h_cands = var_final.get(h_var, {}).get("primary_ids", []) if h_var else []
        t_cands = var_final.get(t_var, {}).get("primary_ids", []) if t_var else []

        if not h_cands and not t_cands:
            continue

        if h_cands and not t_cands:
            for hb in h_cands:
                anchor_nodes = var_candidate_anchors.get(h_var, {}).get(int(hb), [])
                if not anchor_nodes:
                    continue
                new_qt = {
                    "head": ent_name(entities[int(hb)]),
                    "relation": qt.get("relation", ""),
                    "relation_variants": qt.get("relation_variants", []),
                    "tail": t,
                }
                selected_tids = run_retrieval(qi, new_qt, anchor_nodes, "second_pass", meta["triple_type_initial"])
                if selected_tids:
                    extract_var_candidates_from_selected(qi, new_qt, anchor_nodes, selected_tids)

        elif t_cands and not h_cands:
            for tb in t_cands:
                anchor_nodes = var_candidate_anchors.get(t_var, {}).get(int(tb), [])
                if not anchor_nodes:
                    continue
                new_qt = {
                    "head": h,
                    "relation": qt.get("relation", ""),
                    "relation_variants": qt.get("relation_variants", []),
                    "tail": ent_name(entities[int(tb)]),
                }
                selected_tids = run_retrieval(qi, new_qt, anchor_nodes, "second_pass", meta["triple_type_initial"])
                if selected_tids:
                    extract_var_candidates_from_selected(qi, new_qt, anchor_nodes, selected_tids)

        else:
            for hb in h_cands:
                for tb in t_cands:
                    anchor_nodes = []
                    anchor_nodes.extend(var_candidate_anchors.get(h_var, {}).get(int(hb), []))
                    for x in var_candidate_anchors.get(t_var, {}).get(int(tb), []):
                        if x not in anchor_nodes:
                            anchor_nodes.append(x)
                    if not anchor_nodes:
                        continue
                    new_qt = {
                        "head": ent_name(entities[int(hb)]),
                        "relation": qt.get("relation", ""),
                        "relation_variants": qt.get("relation_variants", []),
                        "tail": ent_name(entities[int(tb)]),
                    }
                    run_retrieval(qi, new_qt, anchor_nodes, "second_pass", meta["triple_type_initial"])

    var_final = compute_final_var_candidate_sets()
    var_final = apply_neff_gating(var_final)

    evidence_ids = set()
    for tids in support.values():
        evidence_ids.update([int(t) for t in tids])

    evidence_triples = []
    idx_map = {}
    for k, tid in enumerate(sorted(evidence_ids)):
        info = triple_info[tid]
        evidence_triples.append(
            {
                "kg_triple_id": int(tid),
                "head_entity_id": int(info["head_id"]) if isinstance(info["head_id"], int) else None,
                "tail_entity_id": int(info["tail_id"]) if isinstance(info["tail_id"], int) else None,
                "head": info["head"],
                "relation": info["relation"],
                "tail": info["tail"],
            }
        )
        idx_map[int(tid)] = int(k)

    support_list = []
    for qi, tids in support.items():
        ev_idx = [idx_map[int(tid)] for tid in tids if int(tid) in idx_map]
        if not ev_idx:
            continue
        support_list.append({"query_triple_index": int(qi), "evidence_indices": ev_idx})

    debug_info = {
        "known_entities": known_entities,
        "entity_map": [
            {
                "query_entity": qe,
                "kg_entity_ids": [int(eid) for eid in emap[qe]],
                "kg_entity_names": [entity_names[eid] for eid in emap[qe]],
            }
            for qe in emap
        ],
        "variable_candidates": {
            v: {
                "union_ids": info["union_ids"],
                "intersection_ids": info["intersection_ids"],
                "primary_ranked": info["primary_ranked"],
            }
            for v, info in var_final.items()
        },
        "variable_candidates_per_triple": {
            v: {str(qi): sorted([int(eid) for eid in eids]) for qi, eids in per_triple.items()}
            for v, per_triple in var_candidates_per_triple.items()
        },
        "triples_debug": [triples_debug[qi] for qi in sorted(triples_debug.keys())],
    }

    return {
        "id": item["id"],
        "question": item["question"],
        "ground_truth_answer": item.get("ground_truth_answer"),
        "query_plan": item["query_plan"],
        "evidence_triples": evidence_triples,
        "support": support_list,
        "debug": debug_info,
    }


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
