import json
import os
import re
import string
from tqdm import tqdm
from openai import OpenAI


DATASET = os.getenv("DATASET", "2wiki")

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULTS = {
    "2wiki": {
        "DATA_JSON": "dataset/2wikimultihopqa.json",
        "PHASE2_JSON": "planner_queries/2wiki_data/phase2_evidence_v8_2wiki.json",
        "PHASE1_JSON": "planner_queries/2wiki_data/phase1_evidence_v8_2wiki.json",
        "OUTPUT_JSON": "planner_queries/2wiki_data/qa_results_with_recall_v8_2wiki.json",
    },
    "hotpotqa": {
        "DATA_JSON": "dataset/hotpotqa.json",
        "PHASE2_JSON": "planner_queries/hotpotqa_data/phase2_evidence_v8_hotpotqa.json",
        "PHASE1_JSON": "planner_queries/hotpotqa_data/phase1_evidence_v8_hotpotqa.json",
        "OUTPUT_JSON": "planner_queries/hotpotqa_data/qa_results_with_recall_v8_hotpotqa.json",
    },
    "musique": {
        "DATA_JSON": "dataset/musique.json",
        "PHASE2_JSON": "planner_queries/musique_data/phase2_evidence_v8_musique.json",
        "PHASE1_JSON": "planner_queries/musique_data/phase1_evidence_v8_musique.json",
        "OUTPUT_JSON": "planner_queries/musique_data/qa_results_with_recall_v8_musique.json",
    },
}

if DATASET not in DEFAULTS:
    raise ValueError(f"Unsupported DATASET={DATASET}. Choose from: {list(DEFAULTS.keys())}")

DATA_JSON = os.getenv("DATA_JSON", DEFAULTS[DATASET]["DATA_JSON"])
PHASE1_JSON = os.getenv("PHASE1_JSON", DEFAULTS[DATASET]["PHASE1_JSON"])
PHASE2_JSON = os.getenv("PHASE2_JSON", DEFAULTS[DATASET]["PHASE2_JSON"])
OUTPUT_JSON = os.getenv("OUTPUT_JSON", DEFAULTS[DATASET]["OUTPUT_JSON"])
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "1000"))
GT_SET_FILE_2WIKI = os.getenv(
    "GT_SET_FILE_2WIKI",
    "",
)



def build_client():
    kwargs = {"api_key": API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)


def normalize_whitespace(text):
    return " ".join((text or "").split())


def normalize_answer(s):
    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def exact_match_score(prediction, ground_truth):
    if not ground_truth:
        return 0.0
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    if not ground_truth:
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1

    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def ensure_answer_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


def best_em_f1(prediction, ground_truth_set):
    best_em = 0.0
    best_f1 = 0.0
    best_gt = ""
    for gt in ensure_answer_list(ground_truth_set):
        em = exact_match_score(prediction, gt)
        f1 = f1_score(prediction, gt)
        if (em > best_em) or (em == best_em and f1 > best_f1):
            best_em = em
            best_f1 = f1
            best_gt = gt
    return best_em, best_f1, best_gt


def load_2wiki_gt_set_map(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # with open(path, "r", encoding="utf-8") as f:
    if not isinstance(data, list):
        raise ValueError("GT_SET_FILE_2WIKI must be a JSON list.")

    id2set = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("id", "")).strip()
        if not qid:
            continue
        ans_set = ensure_answer_list(row.get("ground_truth_answer_set", []))
        id2set[qid] = ans_set
    return id2set


def _format_triple(t):
    if isinstance(t, dict):
        h = str(t.get("head", "") or "").strip()
        r = str(t.get("relation", "") or "").strip()
        ta = str(t.get("tail", "") or "").strip()
        parts = [x for x in [h, r, ta] if x]
        return " | ".join(parts) if parts else ""
    if isinstance(t, (list, tuple)) and len(t) >= 3:
        h = str(t[0] or "").strip()
        r = str(t[1] or "").strip()
        ta = str(t[2] or "").strip()
        parts = [x for x in [h, r, ta] if x]
        return " | ".join(parts) if parts else ""
    return str(t or "").strip()



def build_prompt_from_phase2_item(item):
    """Build a QA prompt from constraint-aligned text evidence."""
    question = str(item.get("question", "") or "").strip()
    triples_evidence = item["triples_evidence"]
    intro = (
        "You are a strict multi hop inference assistant that can complete multi hop inference based on given information. "
        "Now, you need to conduct rigorous and rational multi-step thinking based on the given search evidence to answer this query. "
        "You only need to use the given information to answer the query. "
        "We perform Atomic Constraint Planning and obtain multiple atomic constraints. "
        "For each atomic constraint, several pieces of relevant evidence are retrieved. "
        "You should use the provided evidence to resolve the variables in the atomic constraints step by step, and derive the final answer. "
        "You can only answer the final answer with short and appropriate phrases (such as names, numbers, or short noun phrases that conform to the query format). "
        "Do not include explanations, sentences, or any other words. "
        "Here is the information you can obtain:"
    )
    ev_lines = []
    for te in triples_evidence:
        qi = int(te["query_triple_index"])
        qt = _format_triple(te["query_triple"]).replace(" | ", " — ")
        ev_lines.append(f"τ{qi}: {qt}")
        for i, ev in enumerate(te["candidate_evidences"] or [], start=1):
            ctx = normalize_whitespace(ev.get("context")).strip()
            ev_lines.append(f"  Evidence {i}: {ctx}")
        ev_lines.append("")
    ev_block = "\n".join(ev_lines).strip()
    body = (
        f"Query: {question}\n\n"
        "The atomic constraint plan is as follows. Each atomic constraint contains an approximate relation expression "
        "and a variable that should be resolved using the retrieved evidence. "
        f"For each atomic constraint, the retrieved evidence is provided below:\n\n{ev_block}\n\n"
        "Based on the above information, answer the query.\n\n"
        f"Query: {question} Answer:"
    )
    return intro + "\n\n" + body


def compute_resolved_triple_ratio(item):
    triples = item.get("triples_evidence", []) or []
    if not isinstance(triples, list) or not triples:
        return 0.0
    triples = [te for te in triples if te.get("sufficiency_checked", False)]
    if not triples:
        return 0.0
    resolved = 0
    for te in triples:
        st = str((te or {}).get("triple_status", "")).strip().lower()
        if st == "resolved":
            resolved += 1
    return float(resolved) / float(len(triples))


def compute_recall_at_5(item, ex):
    if DATASET == "musique":
        gold = {str(p["idx"]) for p in ex.get("paragraphs", []) if p.get("is_supporting")}
    else:
        gold = {str(fact[0]).strip() for fact in ex.get("supporting_facts", []) if isinstance(fact, (list, tuple)) and fact}
    scores = {}
    for te in item.get("triples_evidence", []):
        for ev in te.get("candidate_evidences", []):
            if DATASET == "musique":
                key = ev.get("paragraph_idx")
                if key is None:
                    raise ValueError("MuSiQue paragraph idx is missing. Run phase2 again.")
                key = str(key)
            else:
                key = str(ev.get("title") or "").strip()
            if not key:
                continue
            score = ev.get("similarity")
            if score is None:
                raise ValueError("Evidence score is missing. Run phase1 and phase2 again.")
            scores[key] = max(scores.get(key, float("-inf")), float(score))
    ranked = sorted(scores, key=lambda key: (-scores[key], key))[:5]
    return len(set(ranked) & gold) / len(gold) if gold else 0.0


def is_prediction_supported_by_evidence(prediction, item):
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return 0.0
    pred_tokens = set(pred_norm.split())
    if not pred_tokens:
        return 0.0

    triples = item.get("triples_evidence", []) or []
    if not isinstance(triples, list):
        return 0.0

    for te in triples:
        evs = (te or {}).get("candidate_evidences", []) or []
        if not isinstance(evs, list):
            continue
        for ev in evs:
            if not isinstance(ev, dict):
                continue
            txt = ev.get("context") or ev.get("evidence") or ""
            txt_norm = normalize_answer(txt)
            if not txt_norm:
                continue
            ev_tokens = set(txt_norm.split())
            if pred_tokens.issubset(ev_tokens):
                return 1.0
    return 0.0


def main_ev():
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    print("DATASET:", DATASET)
    print("PHASE1_JSON:", PHASE1_JSON)
    print("PHASE2_JSON:", PHASE2_JSON)
    print("DATA_JSON:", DATA_JSON)
    print("OUTPUT_JSON:", OUTPUT_JSON)
    print("MAX_SAMPLES:", MAX_SAMPLES)
    print("OPENAI_MODEL:", MODEL_NAME)
    if DATASET == "2wiki":
        print("GT_SET_FILE_2WIKI:", GT_SET_FILE_2WIKI)

    client = build_client()
    with open(PHASE1_JSON, "r", encoding="utf-8") as f:
        _items_1 = json.load(f)
    with open(PHASE2_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("PHASE2_JSON must be a JSON list.")
    if MAX_SAMPLES > 0:
        items = items[:MAX_SAMPLES]

    gt_set_map_2wiki = {}
    if DATASET == "2wiki" and os.path.isfile(GT_SET_FILE_2WIKI):
        gt_set_map_2wiki = load_2wiki_gt_set_map(GT_SET_FILE_2WIKI)

    raw_by_id = {}
    if DATASET in ("2wiki", "hotpotqa", "musique"):
        with open(DATA_JSON, "r", encoding="utf-8") as f:
            raw_by_id = {str(ex.get("_id") or ex.get("id")): ex for ex in json.load(f)}

    outputs = []
    total_recall = 0.0
    total_em = 0.0
    total_f1 = 0.0
    n = 0

    for item in tqdm(items, desc=f"Evaluation universal ({DATASET})"):
        question = item.get("question", "")
        qid = str(item.get("id", ""))
        if DATASET == "2wiki":
            gold_set = ensure_answer_list(gt_set_map_2wiki.get(qid, []))
            if not gold_set:
                fallback = item.get("ground_truth_answer", item.get("answer", ""))
                gold_set = ensure_answer_list(fallback)
            gold_ans = gold_set[0] if gold_set else ""
        else:
            gold_ans = item.get("ground_truth_answer", item.get("answer", ""))
            gold_set = ensure_answer_list(gold_ans)
       
       
        prompt = build_prompt_from_phase2_item(item)

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        pred = (resp.choices[0].message.content or "").strip()

        if DATASET == "2wiki":
            em, f1, best_gt = best_em_f1(pred, gold_set)
        else:
            em = exact_match_score(pred, gold_ans)
            f1 = f1_score(pred, gold_ans)
            best_gt = gold_ans
        resolved_ratio = compute_resolved_triple_ratio(item)
        answer_supported = is_prediction_supported_by_evidence(pred, item)
        unsupported_rate = 1.0 - answer_supported

        total_em += em
        total_f1 += f1
        n += 1

        out_item = {
            "id": item.get("id"),
            "question": question,
            "ground_truth_answer": gold_ans,
            "model_input": prompt,
            "model_answer": pred,
            "em": em,
            "f1": f1,
            "resolved_triple_ratio": resolved_ratio,
            "answer_supported_by_evidence": answer_supported,
            "unsupported_answer_flag": unsupported_rate,
        }
        if DATASET == "2wiki":
            out_item["ground_truth_answer_set"] = gold_set
            out_item["best_matched_ground_truth"] = best_gt
        if DATASET in ("2wiki", "hotpotqa", "musique"):
            if qid not in raw_by_id:
                raise ValueError(f"Missing gold evidence for query {qid}")
            out_item["recall@5"] = compute_recall_at_5(item, raw_by_id[qid])
            total_recall += out_item["recall@5"]
        outputs.append(out_item)

    overall_em = total_em / n if n else 0.0
    overall_f1 = total_f1 / n if n else 0.0

    print(f"Overall EM: {overall_em:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")
    if DATASET in ("2wiki", "hotpotqa", "musique"):
        print(f"Overall Recall@5: {total_recall / n if n else 0.0:.4f}")

    out_dir = os.path.dirname(OUTPUT_JSON)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main_ev()
