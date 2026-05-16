
import json, os, time
from tqdm import tqdm
from openai import OpenAI


API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DATASET = os.getenv("DATASET", "hotpotqa")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "1000"))
SLEEP_SECS = float(os.getenv("SLEEP_SECS", "0.5"))

_DEFAULTS = {
    "2wiki": {
        "DATA_PATH": "dataset/2wikimultihopqa.json",
        "OUTPUT_FILE": "planned_queries/2wiki_data/query_graph_v8_2wiki.json",
    },
    "hotpotqa": {
        "DATA_PATH": "dataset/hotpotqa.json",
        "OUTPUT_FILE": "planned_queries/hotpotqa_data/query_graph_v8_hotpotqa.json",
    },
    "musique": {
        "DATA_PATH": "dataset/musique.json",
        "OUTPUT_FILE": "planned_queries/musique_data/query_graph_v8_musique.json",
    },
}

if DATASET not in _DEFAULTS:
    raise ValueError(f"Unsupported DATASET={DATASET}. Choose from: {list(_DEFAULTS.keys())}")

DATA_PATH = os.getenv("DATA_PATH", _DEFAULTS[DATASET]["DATA_PATH"])
OUTPUT_FILE = os.getenv("OUTPUT_FILE", _DEFAULTS[DATASET]["OUTPUT_FILE"])

# SYSTEM_PROMPT = """You are an expert in Knowledge Graphs and Question Answering. 












SYSTEM_PROMPT = """You are an expert in Knowledge Graphs and Question Answering. 
Your task is to decompose a complex natural language question into a sequence of triples.

Rules:
1. Identify known entities and use variables starting with '?' for unknown ones.
2. Use the SAME variable name whenever the same unknown entity or value is referred to more than once in the question (e.g. ?person, ?city, ?timeframe).
3. Output a valid JSON Object with a single key "triples".
4. Each element in "triples" must be an object containing:
   - "head": string
   - "relation": string (the main, canonical relation phrase)
   - "relation_variants": a list of 3鈥? short relation phrases (strings) that are semantically similar to "relation", with "relation" itself as the first element in the list
   - "tail": string
5. The triples should form a minimal but sufficient reasoning chain: if every "?" is filled with the correct entity/value, the question can be answered using ONLY these triples.
6. When the question asks for a property (such as nationality, position, time period, location, etc.) of some entity mentioned or described in the question, the triple containing "?" for that property should directly connect that property to the correct entity variable (the one that represents 鈥渢he person/thing being asked about鈥?, rather than to a side or background entity.
7. Make "relation" as informative as needed so that the meaning of the triple is clear even without the original question (avoid very vague relations like just "during", "of", "has" when a more specific phrase is possible).
8. Do not generate triples whose head and tail are both known constants. When an entity is described by additional conditions (such as relative clauses, roles, time ranges, or other modifiers), treat these as constraints attached to the SAME variable that represents that entity, or merge it into that entity mention as a modifier. 
9. Do not let a single variable stand for multiple logically different unknowns. Only reuse the same variable if the question clearly implies it is the same underlying entity or value; otherwise, use different variable names (e.g. ?country1, ?country2) when needed.
10. Keep variable names short and generic (e.g. ?manager, ?movie, ?company, ?country) and use relations/triples to express detailed descriptions or roles. Do not encode long descriptive phrases inside the variable name itself.
11.  All generated triplets must contain at least one unknown variable. If the triples with two known variablies (enetities), they must be deleted.


**Input**: "The Oberoi family is part of a hotel company that has a head office in what city?"
**Output**:
{
  "triples": [
    {
      "head": "Oberoi family",
      "relation": "part of",
      "relation_variants": ["part of", "is part of", "belongs to", "is a member of"],
      "tail": "?hotel company"
    },
    {
      "head": "?hotel company",
      "relation": "has",
      "relation_variants": ["has", "has as a feature", "possesses"],
      "tail": "a head office"
    },
    {
      "head": "?hotel company",
      "relation": "located in",
      "relation_variants": ["located in", "is based in", "is situated in"],
      "inverse_relation_variants": ["location of", "is home to", "hosts"],
      "tail": "?city"
    }
  ]
}

**Input**: "Which film starring Tom Hanks was directed by Steven Spielberg and released in 1998?"
**Output**:
{
  "triples": [
    {
      "head": "Tom Hanks",
      "relation": "starred in",
      "relation_variants": ["starred in", "appeared in", "acted in"],
      "tail": "?film"
    },
    {
      "head": "?film",
      "relation": "directed by",
      "relation_variants": ["directed by", "was directed by", "film direction by"],
      "tail": "Steven Spielberg"
    },
    {
      "head": "?film",
      "relation": "publication date",
      "relation_variants": ["publication date", "release year", "released in"],
      "tail": "1998"
    }
  ]
}

**Input**: "Which film was released first, Aas Ka Panchhi or Phoolwari?"
**Output**:
{
"triples": [
    {
      "head": "Aas Ka Panchhi",
      "relation": "released in",
      "relation_variants": [
        "released in",
        "publication date",
        "release year"
      ],
      "tail": "?year1"
    },
    {
      "head": "Phoolwari",
      "relation": "released in",
      "relation_variants": [
        "released in",
        "publication date",
        "release year"
      ],
      "tail": "?year2"
    }
 ]
}

**Input**: "Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Br氓k?"
**Output**:
{
"triples": [
    {
      "head": "?director1",
      "relation": "directed",
      "relation_variants": [
        "directed",
        "was directed by",
        "film direction by"
      ],
      "tail": "God'S Gift To Women"
    },
    {
      "head": "?director2",
      "relation": "directed",
      "relation_variants": [
        "directed",
        "was directed by",
        "film direction by"
      ],
      "tail": "Aldri Annet Enn Br氓k"
    },
    {
      "head": "?director1",
      "relation": "age",
      "relation_variants": [
        "age",
        "the age",
        "is of age"
      ],
      "tail": "?age1"
    },
    {
      "head": "?director2",
      "relation": "age",
      "relation_variants": [
        "age",
        "the age",
        "is of age"
      ],
      "tail": "?age2"
    }
 ]
 }


 **Input**: "What is the nationality of the author of The Hobbit, which was published in 1937?"
**Output**:
{
  "triples": [
    {
      "head": "The Hobbit published in 1937",
      "relation": "written by",
      "relation_variants": [
        "written by",
        "authored by",
        "author of",
        "created by"
      ],
      "tail": "?author"
    },
    {
      "head": "?author",
      "relation": "nationality",
      "relation_variants": [
        "nationality",
        "country of citizenship",
        "citizen of",
        "from"
      ],
      "tail": "?nationality"
    }
  ]
}

"""

def load_dataset(dataset, path, num_samples):
    data = json.loads(open(path, "r", encoding="utf-8").read())
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list.")
    if num_samples is None:
        return data
    return data[:int(num_samples)]

def get_ex_id(ex):
    return str(ex.get("id") or ex.get("_id") or ex.get("qid") or ex.get("question_id") or "")

def get_question(ex):
    return ex.get("question", "")

def get_answer(ex):
    return ex.get("answer", "")


def _normalize_triples(triples_raw):
    normalized = []
    if not isinstance(triples_raw, list):
        return normalized

    for tr in triples_raw:
        if not isinstance(tr, dict):
            continue

        head = "" if tr.get("head") is None else str(tr.get("head"))
        tail = "" if tr.get("tail") is None else str(tr.get("tail"))
        relation = "" if tr.get("relation") is None else str(tr.get("relation"))

        rv = tr.get("relation_variants")
        rv_clean = []
        if isinstance(rv, list):
            seen = set()
            for x in rv:
                if not isinstance(x, str):
                    continue
                s = x.strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                rv_clean.append(s)

        if relation:
            if relation in rv_clean:
                rv_clean.remove(relation)
            rv_clean.insert(0, relation)
        else:
            if rv_clean:
                relation = rv_clean[0]

        normalized.append(
            {
                "head": head.strip(),
                "relation": relation.strip(),
                "relation_variants": rv_clean if rv_clean else ([relation.strip()] if relation.strip() else []),
                "tail": tail.strip(),
            }
        )

    return normalized

def get_query_plan_with_retry(question, client, max_retries=3):
    user_content = f"Question: {question}\nOutput:"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=30.0,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)

            triples_raw = None
            if isinstance(data, dict) and "triples" in data and isinstance(data["triples"], list):
                triples_raw = data["triples"]
            elif isinstance(data, dict):
                for _, v in data.items():
                    if isinstance(v, list):
                        triples_raw = v
                        break

            if triples_raw is None:
                return []

            return _normalize_triples(triples_raw)

        except Exception as e:
            print(f"[Warn] LLM failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)

    return []


def main_query():
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    kwargs = {"api_key": API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    client = OpenAI(**kwargs)

    data = load_dataset(DATASET, DATA_PATH, NUM_SAMPLES)
    print(f"[QueryBuilder] dataset={DATASET} examples={len(data)}")
    print(f"[QueryBuilder] data_path={DATA_PATH}")
    print(f"[QueryBuilder] output={OUTPUT_FILE}")

    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        for ex in tqdm(data):
            qid = get_ex_id(ex)
            question = get_question(ex)
            answer = get_answer(ex)

            triples = get_query_plan_with_retry(question, client)

            item = {
                "id": qid,
                "question": question,
                "ground_truth_answer": answer,
                "query_plan": triples,
            }

            if not first:
                f.write(",\n")
            first = False

            json.dump(item, f, ensure_ascii=False, indent=2)
            f.flush()

            if SLEEP_SECS:
                time.sleep(SLEEP_SECS)

        f.write("\n]\n")

    print("[QueryBuilder] done.")

if __name__ == "__main__":
    main_query()
