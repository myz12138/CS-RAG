import json
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_BASE_URL"] = "https://hf-mirror.com"
import re
from tqdm import tqdm
import torch
import torch.nn.functional as F

from utils.remote_emb import EmbeddingClient

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception as _e:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


DATASET = os.getenv("DATASET", "musique")

DEFAULTS = {
    "2wiki": {
        "PHASE1_JSON": "planned_queries/2wiki_data/phase1_evidence_v8_2wiki.json",
        "DATA_JSON": "dataset/2wikimultihopqa.json",
        "OUTPUT_JSON": "planned_queries/2wiki_data/phase2_evidence_v8_2wiki.json",
        "KG_PATH": "KGs/KG_2wiki",
    },
    "hotpotqa": {
        "PHASE1_JSON": "planned_queries/hotpotqa_data/phase1_evidence_v8_hotpotqa.json",
        "DATA_JSON": "dataset/hotpotqa.json",
        "OUTPUT_JSON": "planned_queries/hotpotqa_data/phase2_evidence_v8_hotpotqa.json",
        "KG_PATH": "KGs/KG_hotpotqa",
    },
    "musique": {
        "PHASE1_JSON": "planned_queries/musique_data/phase1_evidence_v8_musique.json",
        "DATA_JSON": "dataset/musique.json",
        "OUTPUT_JSON": "planned_queries/musique_data/phase2_evidence_v8_musique.json",
        "KG_PATH": "KGs/KG_musique",
    },
}

if DATASET not in DEFAULTS:
    raise ValueError(f"Unsupported DATASET={DATASET}. Choose from: {list(DEFAULTS.keys())}")

PHASE1_JSON = os.getenv("PHASE1_JSON", DEFAULTS[DATASET]["PHASE1_JSON"])
DATA_JSON = os.getenv("DATA_JSON", DEFAULTS[DATASET]["DATA_JSON"])
OUTPUT_JSON = os.getenv("OUTPUT_JSON", DEFAULTS[DATASET]["OUTPUT_JSON"])
KG_PATH = os.getenv("KG_PATH", DEFAULTS[DATASET]["KG_PATH"])
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "1000"))
unres_top_similiar = int(os.getenv("UNRES_TOP_SIMILAR", "20"))
TOPK_UNRESOLVED = int(os.getenv("TOPK_UNRESOLVED", "3"))
print("TOPK_UNRESOLVED:",TOPK_UNRESOLVED)
print("MAX_SAMPLES:", MAX_SAMPLES)
print("UNRES_TOP_SIMILAR:", unres_top_similiar)
print("DATASET:", DATASET)
print("KG_PATH:", KG_PATH)
print("PHASE1_JSON:", PHASE1_JSON)
print("DATA_JSON:", DATA_JSON)
print("OUTPUT_JSON:", OUTPUT_JSON)

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DEVICE = os.getenv("EMB_DEVICE", "cpu")
enc = EmbeddingClient(EMB_MODEL, EMB_DEVICE)
print("EMB_MODEL:", EMB_MODEL)
print("EMB_DEVICE:", EMB_DEVICE)

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "256"))
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "32"))
RERANK_FP16 = os.getenv("RERANK_FP16", "1") == "1"
print("RERANK_MODEL:", RERANK_MODEL)

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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



def is_resolved_triple(debug,qt):
    h = (qt.get("head") or "").strip()
    t = (qt.get("tail") or "").strip()
    if get_candidate_names(debug, h)!=[] and get_candidate_names(debug, t)!=[]:
        return True
    else:
        return False



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


def is_var(x):
    return isinstance(x, str) and x.startswith("?")


def dedup_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if not x:
            continue
        x = str(x)
        low = x.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(x)
    return out


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
def load_list_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("DATA_JSON must be a JSON list (or a dict with a 'data' list).")
    return data


def load_id2ex(dataset, path):
    data = load_list_json(path)
    id2ex = {}
    for ex in data:
        if not isinstance(ex, dict):
            continue
        ex_id = ex.get("_id") or ex.get("id") or ex.get("qid")
        if ex_id is None and dataset == "musique":
            ex_id = ex.get("id")
        if ex_id is None:
            continue
        id2ex[str(ex_id)] = ex
    return id2ex


def iter_context_units_2wiki_like(ex):
    """Yield (doc_idx, title, [sentences]) for 2wiki/hotpotqa."""
    ctx = ex.get("context") or ex.get("contexts") or []
    if not isinstance(ctx, list):
        return
    for doc_idx, item in enumerate(ctx):
        title = ""
        sents = []
        if isinstance(item, list) and len(item) >= 2:
            title = str(item[0] or "").strip()
            if isinstance(item[1], list):
                sents = [str(x or "").strip() for x in item[1] if str(x or "").strip()]
            else:
                blob = str(item[1] or "").strip()
                sents = [blob] if blob else []
        elif isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            ss = item.get("sentences") or item.get("sents") or item.get("text") or []
            if isinstance(ss, list):
                sents = [str(x or "").strip() for x in ss if str(x or "").strip()]
            else:
                blob = str(ss or "").strip()
                sents = [blob] if blob else []
        else:
            continue
        if not title and not sents:
            continue
        yield doc_idx, title, sents


def iter_paragraphs_musique(ex):
    """Yield (para_idx, title, paragraph_text) for musique."""
    for p in ex.get("paragraphs", []) or []:
        if not isinstance(p, dict):
            continue
        title = str(p.get("title") or "").strip()
        para_idx = p.get("idx", None)
        txt = str(p.get("paragraph_text") or "").strip()
        if para_idx is None or not txt:
            continue
        yield int(para_idx), title, txt


def get_var_candidate_names(debug, var_name):
    if not (isinstance(var_name, str) and var_name.startswith("?")):
        return []
    vc = (debug or {}).get("variable_candidates", {}) or {}
    info = vc.get(var_name, {}) if isinstance(vc, dict) else {}
    ranked = info.get("primary_ranked", []) or []
    names = []
    for d in ranked:
        if isinstance(d, dict):
            nm = (d.get("entity_name") or "").strip()
            if nm:
                names.append(nm)
    return dedup_keep_order(names)


def get_known_entity_candidate_names(debug, surface):
    """Use Phase-1 entity_map (+ surface itself)."""
    if not (isinstance(surface, str) and surface.strip()):
        return []
    surface = surface.strip()
    names = [surface]
    em = (debug or {}).get("entity_map", []) or []
    if isinstance(em, list):
        for rec in em:
            if not isinstance(rec, dict):
                continue
            qe = (rec.get("query_entity") or "").strip()
            if qe == surface:
                for nm in (rec.get("kg_entity_names") or []):
                    if isinstance(nm, str) and nm.strip():
                        names.append(nm.strip())
                break
    return dedup_keep_order(names)


def get_candidate_names(debug, x):
    if is_var(x):
        return get_var_candidate_names(debug, x)
    return get_known_entity_candidate_names(debug, x)


def build_query_texts(qt, debug):
    """
    Build q_texts for sentence/paragraph embedding retrieval.

    Key design for unresolved triples:
      - use candidate entity names (from Phase-1 debug) for variables
      - never insert variable placeholders into queries
      - queries are simple concatenations of known parts
    """
    h = (qt.get("head") or "").strip()
    t = (qt.get("tail") or "").strip()
    r = (qt.get("relation") or "").strip()

    rels = []
    if r:
        rels.append(r)
    rv = qt.get("relation_variants", [])
    if isinstance(rv, list):
        for x in rv:
            if isinstance(x, str) and x.strip():
                rels.append(x.strip())
    rels = dedup_keep_order(rels)

    head_opts = get_candidate_names(debug, h) if h else []
    tail_opts = get_candidate_names(debug, t) if t else []

    head_iter = head_opts if head_opts else [None]
    tail_iter = tail_opts if tail_opts else [None]

    qs = []

    def add_forward(rp: str):
        rp = (rp or "").strip()
        if not rp:
            return
        for hh in head_iter:
            hh = (hh or "").strip() if isinstance(hh, str) else ""
            for tt in tail_iter:
                tt = (tt or "").strip() if isinstance(tt, str) else ""
                if hh and tt:
                    qs.append(f"{hh} {rp} {tt}")
                elif hh and not tt:
                    qs.append(f"{hh} {rp} which entity?")
                elif tt and not hh:
                    qs.append(f"which entity {rp} {tt}")
                else:
                    qs.append(rp)

    for rp in rels:
        add_forward(rp)

    qs = [" ".join(q.split()) for q in qs if q and str(q).strip()]
    return dedup_keep_order(qs)

def keyword_groups_for_sentence_filter(qt, debug):
    """
    Intersection-style constraints (2wiki/hotpotqa):
      - known surface always contributes a group
      - variable contributes only if solved (has candidates)
      - AND across groups; OR within group
    """
    h = (qt.get("head") or "").strip()
    t = (qt.get("tail") or "").strip()

    groups = []

    if h:
        if is_var(h):
            c = get_var_candidate_names(debug, h)
            if c:
                groups.append(c)
        else:
            c = get_known_entity_candidate_names(debug, h)
            if c:
                groups.append(c)

    if t:
        if is_var(t):
            c = get_var_candidate_names(debug, t)
            if c:
                groups.append(c)
        else:
            c = get_known_entity_candidate_names(debug, t)
            if c:
                groups.append(c)

    return [g for g in groups if g]


def sentence_satisfies_groups(text, groups):
    if not groups:
        return True
    tl = (text or "").lower()
    if not tl:
        return False
    for g in groups:
        ok = False
        for kw in g:
            if kw and kw.lower() in tl:
                ok = True
                break
        if not ok:
            return False
    return True


def any_keyword_match(text, keywords):
    """OR-style coarse filter (musique)."""
    if not keywords:
        return True
    tl = (text or "").lower()
    for kw in keywords:
        if kw and kw.lower() in tl:
            return True
    return False


def extract_union_keywords(qt, debug):
    kws = []
    h = (qt.get("head") or "").strip()
    t = (qt.get("tail") or "").strip()
    if h:
        kws += get_candidate_names(debug, h)
    if t:
        kws += get_candidate_names(debug, t)
    return dedup_keep_order([k for k in kws if k and str(k).strip()])





FOCUS_MAX_PER_VAR = int(os.getenv("FOCUS_MAX_PER_VAR", "5"))

def extract_focus_entities(debug, query_triples, max_per_var=FOCUS_MAX_PER_VAR):
    """Return (known_entities, solved_entities, focus_entities)."""
    known = []
    if isinstance(query_triples, list):
        for qt in query_triples:
            if not isinstance(qt, dict):
                continue
            for side in ("head", "tail"):
                x = (qt.get(side) or "").strip()
                if x and (not is_var(x)):
                    known += get_known_entity_candidate_names(debug, x)

    solved = []
    vc = (debug or {}).get("variable_candidates", {}) or {}
    if isinstance(vc, dict):
        for _, info in vc.items():
            if not isinstance(info, dict):
                continue
            ranked = info.get("primary_ranked", []) or []
            for d in ranked[: max(1, int(max_per_var))]:
                if isinstance(d, dict):
                    nm = (d.get("entity_name") or "").strip()
                    if nm:
                        solved.append(nm)

    known = dedup_keep_order(known)
    solved = dedup_keep_order(solved)
    focus = dedup_keep_order(known + solved)
    return known, solved, focus


def extract_query_relations(query_triples):
    """Collect relation strings and variants from the decomposed query triples."""
    rels = []
    if not isinstance(query_triples, list):
        return rels
    for qt in query_triples:
        if not isinstance(qt, dict):
            continue
        r = (qt.get("relation") or "").strip()
        if r:
            rels.append(r)
        rv = qt.get("relation_variants", [])
        if isinstance(rv, list):
            for x in rv:
                if isinstance(x, str) and x.strip():
                    rels.append(x.strip())
    return dedup_keep_order(rels)


def build_phase1_support_map(phase1_item):
    """Map query_triple_index -> list of KG evidence triples (from Phase-1 output)."""
    ev_triples = phase1_item.get("evidence_triples") or []
    support = phase1_item.get("support") or []
    mp = {}
    if not isinstance(ev_triples, list) or not isinstance(support, list):
        return mp
    for s in support:
        if not isinstance(s, dict):
            continue
        qi = s.get("query_triple_index", None)
        idxs = s.get("evidence_indices", []) or []
        try:
            qi = int(qi)
        except Exception:
            continue
        out = []
        if isinstance(idxs, list):
            for ix in idxs:
                try:
                    j = int(ix)
                except Exception:
                    continue
                if 0 <= j < len(ev_triples) and isinstance(ev_triples[j], dict):
                    out.append(ev_triples[j])
        if out:
            mp[qi] = out
    return mp

def build_rerank_query(ex, qt, debug):
    question = str(ex.get("question") or "").strip()
    h = (qt.get("head") or "").strip()
    r = (qt.get("relation") or "").strip()
    t = (qt.get("tail") or "").strip()
    expr = f"{h} {r} {t}".strip()
    return expr, question, expr


def retrieve_2wiki_like(ex, qt, debug, topk, reranker,status):
    """Clause-level retrieval (2wiki/hotpotqa):

    1) Keyword-group filtering on (title + clause)
    2) Embedding coarse scoring (cheap) and per-document truncation
    3) Cross-encoder reranking per triple (one triple -> one rerank batch)
    """
    topk = max(1, int(topk))
    q_texts = build_query_texts(qt, debug)
    groups = keyword_groups_for_sentence_filter(qt, debug)
    union_keywords = extract_union_keywords(qt, debug)
    rerank_query, question, expr = build_rerank_query(ex, qt, debug)
    
    all_cands = []
    for doc_idx, title, sents in iter_context_units_2wiki_like(ex):
        local = []
        for si, sent in enumerate(sents):
           
            filter_text = f"{title} {sent}".strip()
    
            coarse=  max(text_sim(q, filter_text) for q in q_texts)
            all_cands.append({
                        "title": title,
                        "paragraph_idx": int(doc_idx),
                        "sent_idx": int(si),
                        "evidence": sent,
                        "context": filter_text,
                        "similarity": coarse,
                        "sentence_similarity": coarse,
                    })
    #         if not q_texts:
    #             coarse = 0.0001
    #         else:
    #             coarse = max(text_sim(q, filter_text) for q in q_texts)
    #         sent_key = si#int(si) * 1000 + int(ci)
    #         local.append(
    #             (
    #                 float(coarse),
    #                 {
    #                     "title": title,
    #                     "paragraph_idx": int(doc_idx),
    #                     "sent_idx": int(sent_key),
    #                     "evidence": sent,
    #                     "context": filter_text,
    #                     "similarity": float(coarse),
    #                     "sentence_similarity": float(coarse),
    #                 },
    #             )
    #         )

    #     if local:
    #         local.sort(key=lambda x: x[0], reverse=True)
    #         all_cands.extend([x[1] for x in local])

    # if not all_cands:
    #     return [], union_keywords
    all_cands.sort(key=lambda d: float(d["similarity"]), reverse=True)
    all_cands=all_cands[:unres_top_similiar]

    
    pairs = []
    for c in all_cands:
        ctx = str(c.get("context") or "").strip()
        title=str(c.get("title") or "").strip()
        passage = f"{ctx}".strip()
        pairs.append((rerank_query, passage))

    scores = _rerank_score_pairs(reranker, pairs)
    for c, sc in zip(all_cands, scores):
        c["similarity"] = float(sc)
        c["sentence_similarity"] = float(sc)

    all_cands = [c for c in all_cands]

    all_cands.sort(key=lambda d: float(d["similarity"]), reverse=True)
    return all_cands[:topk], union_keywords

def retrieve_musique(ex, qt, debug, topk, reranker):
    """Paragraph-level retrieval (musique) with reranking."""
    topk = max(1, int(topk))
    q_texts = build_query_texts(qt, debug)
    keywords = extract_union_keywords(qt, debug)
    rerank_query, question, expr = build_rerank_query(ex, qt, debug)

    cands = []
    for para_idx, title, txt in iter_paragraphs_musique(ex):
        filter_text = f"{title} {txt}".strip()
        # if not any_keyword_match(filter_text, keywords):
        #     continue
        coarse = text_sim(filter_text, rerank_query)
        cands.append({
                        "title": title,
                        "paragraph_idx": int(para_idx),
                        "sent_idx": 0,
                        "context": filter_text,
                        "similarity": 0.0,
                        "sentence_similarity": 0.0,
                    })
        
        # if not q_texts:
        #     coarse = 0.0001
        # else:
        #     coarse = max(text_sim(q, filter_text) for q in q_texts)
        # cands.append(
        #     (
        #         float(coarse),
        #         {
        #             "title": title,
        #             "paragraph_idx": int(para_idx),
        #             "sent_idx": 0,
        #             "evidence": "",
        #             "context": txt,
        #             "similarity": float(coarse),
        #             "sentence_similarity": float(coarse),
        #         },
        #     )
        # )

    if not cands:
        return [], keywords

    #cands.sort(key=lambda x: x[0], reverse=True)
    cands.sort(key=lambda d: float(d["similarity"]), reverse=True)
    cands = cands[:unres_top_similiar]


    pairs = []
    for c in cands:
        title = str(c.get("title") or "").strip()
        txt = str(c.get("context") or "").strip()
        passage = f"{txt}".strip()
        pairs.append((rerank_query, passage))

    scores = _rerank_score_pairs(reranker, pairs)
    for c, sc in zip(cands, scores):
        c["similarity"] = float(sc)
        c["sentence_similarity"] = float(sc)
    pool = [c for c in cands if float(c["similarity"])]
    pool.sort(key=lambda d: float(d["similarity"]), reverse=True)

    return pool[:topk], keywords

import re

def _simple_norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff\s]+", " ", s)
    return s

def tri_in_t_i(t_i, tri, require_rel= False) -> bool:
    """
    t_i: sentence string
    tri: dict with string fields tri['h'], tri['r'], tri['t'] (or similar)
    return True iff (h and t) appear in t_i; relation optional.
    """
    sent = _simple_norm(t_i)
    h = _simple_norm(tri.get("head", ""))
    r = _simple_norm(tri.get("relation", ""))
    t = _simple_norm(tri.get("tail", ""))

    if not h or not t:
        return False

    ok = (h in sent) and (t in sent)
    if not ok:
        return False

    if require_rel:
        return bool(r) and (r in sent)
    return True
def main_2():

    kg = load_kg(KG_PATH)
    id2ex = load_id2ex(DATASET, DATA_JSON)

    reranker = build_reranker()

    with open(PHASE1_JSON, "r", encoding="utf-8") as f:
        phase1_items = json.load(f)
    if not isinstance(phase1_items, list):
        raise ValueError("PHASE1_JSON must be a JSON list.")
    if MAX_SAMPLES > 0:
        phase1_items = phase1_items[:MAX_SAMPLES]

    results = []
    for item in tqdm(phase1_items, desc=f"Phase-2 universal ({DATASET})"):
        qid = str(item.get("id"))
        query_triples = item.get("query_plan") or item.get("query_triples") or []
        debug = item.get("debug", {}) or {}
        ex = id2ex.get(qid)

        support_map = build_phase1_support_map(item)

        focus_known_entities, focus_solved_entities, focus_entities = extract_focus_entities(debug, query_triples)
        query_relations = extract_query_relations(query_triples)

        phase2_triples = []
        for qi, qt in enumerate(query_triples):
            candidate_evidences = []
            entity_keywords = []
            kg_triples = []
            status = "unresolved"

            if ex is not None:
                status = "resolved" if is_resolved_triple(debug, qt) else "unresolved"

                if status == "resolved":
                    kg_triples = support_map.get(int(qi), []) or []
                    for tri in kg_triples:
                        title=kg['triples'][int(tri['kg_triple_id'])]['title']
                        #evidence=kg['triples'][int(tri['kg_triple_id'])]['evidence']
                        #print(evi)
                        t_list = []
                        if DATASET in ("2wiki", "hotpotqa"):
                            for _, ctx_title, sents in iter_context_units_2wiki_like(ex):
                                if ctx_title == title:
                                    t_list = sents
                                    break
                        else:
                            
                            for _, para_title, para_txt in iter_paragraphs_musique(ex):
                                if para_title == title and para_txt:
                                    t_list.append(para_txt)
                        for t_i in t_list:
                            if tri_in_t_i(t_i,tri):
                                candidate_evidences.append({'title':title,'context':t_i})
                                break
                    #entity_keywords = extract_union_keywords(qt, debug)
                    #candidate_evidences = []
                else:
                    per_topk = TOPK_UNRESOLVED
                    if DATASET in ("2wiki", "hotpotqa"):
                        candidate_evidences, entity_keywords = retrieve_2wiki_like(ex, qt, debug, per_topk, reranker,status)
                    else:
                        candidate_evidences, entity_keywords = retrieve_musique(ex, qt, debug, per_topk, reranker)

            phase2_triples.append(
                {
                    "query_triple_index": int(qi),
                    "query_triple": qt,
                    "triple_status": status,
                    "entity_keywords": entity_keywords,
                    "candidate_evidences": candidate_evidences,
                    "kg_triples": kg_triples,
                }
            )


        results.append(
            {
                "id": qid,
                "question": item.get("question", ""),
                "ground_truth_answer": item.get("ground_truth_answer", item.get("answer", "")),
                "triples_evidence": phase2_triples,
              
            }
        )

    out_dir = os.path.dirname(OUTPUT_JSON)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main_2()
