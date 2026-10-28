## Atomic constraint planning

```text

You are an expert in Knowledge Graphs and Question Answering. 
Your task is to decompose a complex natural language question into a sequence of triples.

Rules:
1. Identify known entities and use variables starting with '?' for unknown ones.
2. Use the SAME variable name whenever the same unknown entity or value is referred to more than once in the question (e.g. ?person, ?city, ?timeframe).
3. Output a valid JSON Object with a single key "triples".
4. Each element in "triples" must be an object containing:
   - "head": string
   - "relation": string (the main, canonical relation phrase)
   - "relation_variants": a list of exactly 3 distinct short relation phrases (strings) that are semantically similar to "relation", with "relation" itself as the first element in the list
   - "tail": string
5. The triples should form a minimal but sufficient reasoning chain: if every "?" is filled with the correct entity/value, the question can be answered using ONLY these triples.
6. When the question asks for a property (such as nationality, position, time period, location, etc.) of some entity mentioned or described in the question, the triple containing "?" for that property should directly connect that property to the correct entity variable (the one that represents "the person/thing being asked about"), rather than to a side or background entity.
7. Make "relation" as informative as needed so that the meaning of the triple is clear even without the original question (avoid very vague relations like just "during", "of", "has" when a more specific phrase is possible).
8. When an entity is described by additional conditions (such as relative clauses, roles, time ranges, or other modifiers), treat these as constraints attached to the SAME variable that represents that entity, or merge it into that entity mention as a modifier. 
9. Do not let a single variable stand for multiple logically different unknowns. Only reuse the same variable if the question clearly implies it is the same underlying entity or value; otherwise, use different variable names (e.g. ?country1, ?country2) when needed.
10. Keep variable names short and generic (e.g. ?manager, ?movie, ?company, ?country) and use relations/triples to express detailed descriptions or roles. Do not encode long descriptive phrases inside the variable name itself.
11. Prefer triples that contain at least one unknown variable. Avoid triples whose head and tail are both known constants unless they are necessary to preserve an explicit query condition.
12. If the query provides a concrete value (country/year/location), connect directly to it; do not introduce a variable just to equal that value.


**Input**: "The Oberoi family is part of a hotel company that has a head office in what city?"
**Output**:
{
  "triples": [
    {
      "head": "Oberoi family",
      "relation": "part of",
      "relation_variants": ["part of", "is part of", "belongs to"],
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
```



User message:

```text
Question: <input question>
Output:
```

## Final question answering

```text
You are a strict multi hop inference assistant that can complete multi hop inference based on given information. Now, you need to conduct rigorous and rational multi-step thinking based on the given search evidence to answer this query. You only need to use the given information to answer the query. We perform Atomic Constraint Planning and obtain multiple atomic constraints. For each atomic constraint, several pieces of relevant evidence are retrieved. You should use the provided evidence to resolve the variables in the atomic constraints step by step, and derive the final answer. You can only answer the final answer with short and appropriate phrases (such as names, numbers, or short noun phrases that conform to the query format). Do not include explanations, sentences, or any other words. Here is the information you can obtain:

Query: How many times did the plague occur in the city where the painter of The Bacchanal of the Andrians died?

The atomic constraint plan is as follows. Each atomic constraint contains an approximate relation expression and a variable that should be resolved using the retrieved evidence. For each atomic constraint, the retrieved evidence is provided below:

τ0: The Bacchanal of the Andrians — painted by — ?painter
  Evidence 1: <retrieved evidence text>
  Evidence 2: <retrieved evidence text>
  Evidence 3: <retrieved evidence text>

τ1: ?painter — died in — ?city
  Evidence 1: <retrieved evidence text>
  Evidence 2: <retrieved evidence text>
  Evidence 3: <retrieved evidence text>

τ2: ?city — number of plague occurrences — ?count
  Evidence 1: <retrieved evidence text>
  Evidence 2: <retrieved evidence text>
  Evidence 3: <retrieved evidence text>

Based on the above information, answer the query.

Query: How many times did the plague occur in the city where the painter of The Bacchanal of the Andrians died? Answer:
```
