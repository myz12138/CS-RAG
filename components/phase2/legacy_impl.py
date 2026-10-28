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
        "PHASE1_JSON": "planner_queries/2wiki_data/phase1_evidence_v8_2wiki.json",
        "DATA_JSON": "dataset/2wikimultihopqa.json",
        "OUTPUT_JSON": "planner_queries/2wiki_data/phase2_evidence_v8_2wiki.json",
        "KG_PATH": "KGs/KG_2wiki",
    },
    "hotpotqa": {
        "PHASE1_JSON": "planner_queries/hotpotqa_data/phase1_evidence_v8_hotpotqa.json",
        "DATA_JSON": "dataset/hotpotqa.json",
        "OUTPUT_JSON": "planner_queries/hotpotqa_data/phase2_evidence_v8_hotpotqa.json",
        "KG_PATH": "KGs/KG_hotpotqa",
    },
    "musique": {
        "PHASE1_JSON": "planner_queries/musique_data/phase1_evidence_v8_musique.json",
        "DATA_JSON": "dataset/musique.json",
        "OUTPUT_JSON": "planner_queries/musique_data/phase2_evidence_v8_musique.json",
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
unres_top_similiar = int(os.getenv("UNRES_TOP_SIMILAR", "8"))
TOPK_UNRESOLVED = int(os.getenv("TOPK_UNRESOLVED", "5"))
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

_TEXT_INDEX_RECALL = None
_TEXT_INDEX_RECALL_INITIALIZED = False
_TEXT_INDEX_CANDIDATES_INITIALIZED = False
_TEXT_INDEX_CANDIDATES = None
_TEXT_INDEX_RECALL_STATS = {
    "status": "not_configured",
    "index_dir": os.getenv("TEXT_INDEX_DIR", ""),
    "candidate_file": os.getenv("TEXT_INDEX_CANDIDATES", ""),
    "query_count": 0,
    "retrieved_chunk_count": 0,
    "selected_candidate_count": 0,
    "fallback_count": 0,
    "error": None,
}

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LENGTH = int(
    os.getenv("PHASE2_RERANK_MAX_LENGTH", os.getenv("RERANK_MAX_LENGTH", "256"))
)
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



def is_resolved_triple(debug, qt, qi=None):
    records = (debug or {}).get("triples_debug", []) or []
    if qi is not None:
        record = next((r for r in records if r.get("query_triple_index") == qi), None)
        if record is None or "status" not in record:
            raise ValueError("Phase-1 status is missing. Run phase1 again.")
        return record["status"] == "resolved"
    h = (qt.get("head") or "").strip()
    t = (qt.get("tail") or "").strip()
    return bool(get_candidate_names(debug, h) and get_candidate_names(debug, t))


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


def _runtime_model_label(value):
    value = str(value or "").strip()
    candidate = Path(value)
    if candidate.is_dir():
        return candidate.name
    return value


def _get_text_index_recall():
    """Load the text index on first use."""

    global _TEXT_INDEX_RECALL_INITIALIZED, _TEXT_INDEX_RECALL
    if _TEXT_INDEX_RECALL_INITIALIZED:
        return _TEXT_INDEX_RECALL
    _TEXT_INDEX_RECALL_INITIALIZED = True

    raw_dir = os.getenv("TEXT_INDEX_DIR", "").strip()
    if not raw_dir:
        return None
    try:
        from backend.app.indexing.text_recall import TextIndexRecall

        reader = TextIndexRecall.from_environment(Path(raw_dir))
        index_model = _runtime_model_label(reader.manifest.get("embedding_model"))
        runtime_model = _runtime_model_label(EMB_MODEL)
        if index_model and runtime_model and index_model != runtime_model:
            raise RuntimeError(
                "文本索引 embedding 模型与查询 embedding 模型不一致"
                f"（索引={index_model}，查询={runtime_model}）"
            )
        _TEXT_INDEX_RECALL = reader
        _TEXT_INDEX_RECALL_STATS["status"] = "ready"
        _TEXT_INDEX_RECALL_STATS["backend"] = str(
            reader.manifest.get("backend") or "local_file"
        )
        _TEXT_INDEX_RECALL_STATS["embedding_model"] = index_model
    except Exception as exc:
        _TEXT_INDEX_RECALL_STATS["status"] = "fallback"
        _TEXT_INDEX_RECALL_STATS["error"] = str(exc)
        print("TEXT_INDEX_RECALL unavailable; use reference full scan:", exc)
    return _TEXT_INDEX_RECALL


def _get_text_index_candidates():
    """Load the query-workspace sidecar mapping index aliases to sentence positions."""

    global _TEXT_INDEX_CANDIDATES_INITIALIZED, _TEXT_INDEX_CANDIDATES
    if _TEXT_INDEX_CANDIDATES_INITIALIZED:
        return _TEXT_INDEX_CANDIDATES
    _TEXT_INDEX_CANDIDATES_INITIALIZED = True

    raw_path = os.getenv("TEXT_INDEX_CANDIDATES", "").strip()
    if not raw_path:
        return None
    try:
        with open(raw_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        units = payload.get("units") if isinstance(payload, dict) else None
        if not isinstance(units, list):
            raise ValueError("TEXT_INDEX_CANDIDATES.units 不是列表")

        by_entry_id = {}
        for unit in units:
            if not isinstance(unit, dict):
                continue
            if unit.get("kind") != "context_sentence":
                continue
            entry_id = str(unit.get("entry_id") or "").strip()
            if not entry_id:
                continue
            try:
                location = (int(unit["doc_idx"]), int(unit["sent_idx"]))
            except (KeyError, TypeError, ValueError):
                continue
            by_entry_id.setdefault(entry_id, []).append(location)
        if not by_entry_id:
            raise ValueError("TEXT_INDEX_CANDIDATES 没有有效的句子映射")
        _TEXT_INDEX_CANDIDATES = by_entry_id
        _TEXT_INDEX_RECALL_STATS["candidate_unit_count"] = sum(
            len(locations) for locations in by_entry_id.values()
        )
    except Exception as exc:
        _TEXT_INDEX_RECALL_STATS["status"] = "fallback"
        _TEXT_INDEX_RECALL_STATS["error"] = str(exc)
        print("TEXT_INDEX_CANDIDATES unavailable; use reference full scan:", exc)
    return _TEXT_INDEX_CANDIDATES


def _indexed_context_locations(q_texts):
    """Return indexed sentence positions, or None when unavailable."""

    reader = _get_text_index_recall()
    by_entry_id = _get_text_index_candidates()
    if reader is None or by_entry_id is None:
        _TEXT_INDEX_RECALL_STATS["fallback_count"] += 1
        return None
    if not q_texts:
        _TEXT_INDEX_RECALL_STATS["fallback_count"] += 1
        return None

    try:
        multiplier = max(1, int(os.getenv("TEXT_INDEX_RECALL_MULTIPLIER", "4")))
    except (TypeError, ValueError):
        multiplier = 4
    recall_top_k = max(unres_top_similiar * multiplier, 50)
    selected = set()
    retrieved_chunks = 0
    try:
        for q_text in dedup_keep_order(q_texts):
            query_vector = enc.encode_one(q_text)
            hits = reader.search(query_vector, top_k=recall_top_k)
            retrieved_chunks += len(hits)
            for hit in hits:
                aliases = getattr(hit, "chunk_ids", ()) or (getattr(hit, "chunk_id", ""),)
                for entry_id in aliases:
                    for location in by_entry_id.get(str(entry_id).strip(), ()):
                        selected.add(location)
    except Exception as exc:
        _TEXT_INDEX_RECALL_STATS["status"] = "fallback"
        _TEXT_INDEX_RECALL_STATS["error"] = str(exc)
        _TEXT_INDEX_RECALL_STATS["fallback_count"] += 1
        print("TEXT_INDEX_RECALL query failed; use reference full scan:", exc)
        return None

    _TEXT_INDEX_RECALL_STATS["query_count"] += 1
    _TEXT_INDEX_RECALL_STATS["retrieved_chunk_count"] += retrieved_chunks
    _TEXT_INDEX_RECALL_STATS["selected_candidate_count"] += len(selected)
    if not selected:
        _TEXT_INDEX_RECALL_STATS["fallback_count"] += 1
        return None
    return selected


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
    """Build retrieval queries from constraints and candidate entity names."""
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
    """Group known and bound entity names for sentence filtering."""
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
                    tr = dict(ev_triples[j])
                    scores = s.get("rerank_scores", {})
                    tr["rerank_score"] = float(scores.get(str(tr["kg_triple_id"]), 0.0))
                    out.append(tr)
        if out:
            mp[qi] = out
    return mp

def build_rerank_query(ex, qt, debug):
    question = str(ex.get("question") or "").strip()
    h = (qt.get("head") or "").strip()
    r = (qt.get("relation") or "").strip()
    t = (qt.get("tail") or "").strip()
    expr = f"{h} {r} {t}".strip()
    query = f"question: {question} UnknownTriple: {expr}"
    return query, question, expr


def retrieve_2wiki_like(ex, qt, debug, topk, reranker, status):
    expr = " ".join(str(qt.get(k) or "") for k in ("head", "relation", "tail"))
    rerank_query, question, expr = build_rerank_query(ex, qt, debug)
    passages = {}
    for doc_idx, title, sents in iter_context_units_2wiki_like(ex):
        context = f"{title} {' '.join(sents)}".strip()
        if title not in passages:
            passages[title] = {"title": title, "paragraph_idx": doc_idx, "passage_id": title,
                "context": context, "similarity": text_sim(expr, context)}
    cands = sorted(passages.values(), key=lambda ev: -ev["similarity"])[:unres_top_similiar]
    scores = _rerank_score_pairs(reranker, [(rerank_query, ev["context"]) for ev in cands])
    for ev, score in zip(cands, scores):
        ev["similarity"] = float(score)
        ev["sentence_similarity"] = float(score)
    cands.sort(key=lambda ev: -ev["similarity"])
    return cands[:max(1, int(topk))], extract_union_keywords(qt, debug)


def retrieve_musique(ex, qt, debug, topk, reranker):
    rerank_query, question, expr = build_rerank_query(ex, qt, debug)
    cands = []
    for para_idx, title, txt in iter_paragraphs_musique(ex):
        context = f"{title} {txt}".strip()
        cands.append({"title": title, "paragraph_idx": para_idx, "passage_id": str(para_idx),
            "context": context, "similarity": text_sim(expr, context)})
    cands.sort(key=lambda ev: -ev["similarity"])
    cands = cands[:unres_top_similiar]
    scores = _rerank_score_pairs(reranker, [(rerank_query, ev["context"]) for ev in cands])
    for ev, score in zip(cands, scores):
        ev["similarity"] = float(score)
        ev["sentence_similarity"] = float(score)
    cands.sort(key=lambda ev: -ev["similarity"])
    return cands[:max(1, int(topk))], extract_union_keywords(qt, debug)


import re

def _simple_norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff\s]+", " ", s)
    return s

def tri_in_t_i(t_i, tri, require_rel= False) -> bool:
    """Check whether a sentence contains both endpoints and optionally the relation."""
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


def _write_text_index_recall_stats():
    raw_path = os.getenv("TEXT_INDEX_RECALL_STATS", "").strip()
    if not raw_path:
        return
    target = Path(raw_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    temporary.write_text(
        json.dumps(_TEXT_INDEX_RECALL_STATS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)


# Previous full-passage implementation retained for rollback.
# def collect_source_passages(ex, kg_triples):
#     passages = {}
#     if DATASET == "musique":
#         contexts = list(iter_paragraphs_musique(ex))
#     else:
#         contexts = [(idx, title, " ".join(sents)) for idx, title, sents in iter_context_units_2wiki_like(ex)]
#     for tri in kg_triples:
#         title = tri.get("title")
#         evidence = " ".join(re.findall(r"\w+", str(tri.get("evidence") or "").casefold()))
#         for doc_idx, ctx_title, source in contexts:
#             if ctx_title != title:
#                 continue
#             if DATASET == "musique":
#                 text = " ".join(re.findall(r"\w+", source.casefold()))
#                 if not evidence or f" {evidence} " not in f" {text} ":
#                     continue
#             key = str(doc_idx) if DATASET == "musique" else title
#             score = float(tri["rerank_score"])
#             record = {"title": title, "paragraph_idx": doc_idx, "passage_id": key,
#                 "context": f"{title} {source}".strip(), "similarity": score, "sentence_similarity": score}
#             if key not in passages or score > passages[key]["similarity"]:
#                 passages[key] = record
#     return sorted(passages.values(), key=lambda ev: -ev["similarity"])

def collect_source_passages(ex, kg_triples):
    passages = {}
    if DATASET == "musique":
        contexts = list(iter_paragraphs_musique(ex))
    else:
        contexts = [(idx, title, "") for idx, title, sents in iter_context_units_2wiki_like(ex)]
    for tri in kg_triples:
        title = tri.get("title")
        evidence = str(tri.get("evidence") or "").strip()
        if not evidence:
            continue
        evidence_key = " ".join(evidence.casefold().split())
        for doc_idx, ctx_title, source in contexts:
            if ctx_title != title:
                continue
            if DATASET == "musique":
                quote = " ".join(re.findall(r"\w+", evidence.casefold()))
                text = " ".join(re.findall(r"\w+", source.casefold()))
                if not quote or f" {quote} " not in f" {text} ":
                    continue
            passage_id = str(doc_idx) if DATASET == "musique" else title
            key = (passage_id, evidence_key)
            score = float(tri["rerank_score"])
            record = {"title": title, "paragraph_idx": doc_idx, "passage_id": passage_id,
                "context": evidence, "similarity": score, "sentence_similarity": score}
            if key not in passages or score > passages[key]["similarity"]:
                passages[key] = record
    return sorted(passages.values(), key=lambda ev: -ev["similarity"])


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
                status = "resolved" if is_resolved_triple(debug, qt, qi) else "unresolved"

                if status == "resolved":
                    kg_triples = support_map.get(int(qi), []) or []
                    candidate_evidences = collect_source_passages(ex, kg_triples)
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
                    "sufficiency_checked": next((r.get("checked", False) for r in debug.get("triples_debug", []) if r.get("query_triple_index") == qi), False),
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
    _write_text_index_recall_stats()


if __name__ == "__main__":
    main_2()
