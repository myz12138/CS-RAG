#!/usr/bin/env python3
"""
Offline KG perturbation for robustness experiments.

This script supports issue-driven perturbation:
- Choose issue category or combinations in YAML.
- Sweep ratios.
- Choose injection scope: global KG or query-related subgraph.
- Expand scenarios by combine_mode (cartesian / aligned).
"""

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


def _load_config(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text)
        if not isinstance(cfg, dict):
            raise ValueError("Config root must be a mapping.")
        return cfg
    except Exception:
        cfg = json.loads(text)
        if not isinstance(cfg, dict):
            raise ValueError("Config root must be a mapping.")
        return cfg


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_kg_dir(kg_dir: Path) -> Dict:
    return {
        "entities": _read_jsonl(kg_dir / "entities.jsonl"),
        "triples": _read_jsonl(kg_dir / "triples.jsonl"),
    }


def _get_head_tail(tr: Dict) -> Optional[Tuple[int, int]]:
    if not isinstance(tr, dict):
        return None
    h = tr.get("h", tr.get("head"))
    t = tr.get("t", tr.get("tail"))
    try:
        return int(h), int(t)
    except Exception:
        return None


def _set_head_tail(tr: Dict, h: int, t: int) -> None:
    if "h" in tr or "head" not in tr:
        tr["h"] = int(h)
    if "t" in tr or "tail" not in tr:
        tr["t"] = int(t)
    if "head" in tr:
        tr["head"] = int(h)
    if "tail" in tr:
        tr["tail"] = int(t)


def _get_relation(tr: Dict) -> str:
    r = tr.get("r", tr.get("relation", ""))
    return str(r or "")


def _set_relation(tr: Dict, r: str) -> None:
    if "r" in tr or "relation" not in tr:
        tr["r"] = str(r)
    if "relation" in tr:
        tr["relation"] = str(r)


def _entity_name(entities: List[Dict], idx: int) -> str:
    if 0 <= idx < len(entities):
        e = entities[idx]
        if isinstance(e, dict):
            if e.get("name") is not None:
                return str(e.get("name"))
            if e.get("surface") is not None:
                return str(e.get("surface"))
        return str(e)
    return f"entity_{idx}"


def _entity_type(entities: List[Dict], idx: int) -> str:
    if 0 <= idx < len(entities) and isinstance(entities[idx], dict):
        t = entities[idx].get("type")
        if t is not None:
            return str(t).strip()
    return ""


def _sanitize_entities(entities: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for i, e in enumerate(entities):
        e2 = dict(e) if isinstance(e, dict) else {"name": str(e)}
        e2["id"] = int(i)
        out.append(e2)
    return out


def _sanitize_triples(triples: List[Dict], n_entities: int) -> List[Dict]:
    out: List[Dict] = []
    for tr in triples:
        if not isinstance(tr, dict):
            continue
        ht = _get_head_tail(tr)
        if ht is None:
            continue
        h, t = ht
        if h < 0 or t < 0 or h >= n_entities or t >= n_entities:
            continue
        tr2 = dict(tr)
        _set_head_tail(tr2, h, t)
        out.append(tr2)
    return out


def _copy_kg_fast(base_kg: Dict) -> Dict:
    """
    Faster than deepcopy for this KG shape:
    - entities/triples are flat list[dict] rows
    - we only need per-row dict copy
    """
    entities = [dict(e) if isinstance(e, dict) else e for e in base_kg["entities"]]
    triples = [dict(tr) if isinstance(tr, dict) else tr for tr in base_kg["triples"]]
    return {"entities": entities, "triples": triples}


def _rebuild_title_indexes(kg: Dict) -> Tuple[List[Dict], List[Dict]]:
    title2entities = defaultdict(set)
    title2triples = defaultdict(list)

    for idx, tr in enumerate(kg["triples"]):
        title = str(tr.get("title") or "").strip()
        if not title:
            title = "__untitled__"
            tr["title"] = title
        title2triples[title].append(int(idx))
        ht = _get_head_tail(tr)
        if ht is None:
            continue
        h, t = ht
        title2entities[title].add(int(h))
        title2entities[title].add(int(t))

    t2e_rows: List[Dict] = []
    t2t_rows: List[Dict] = []
    for title in sorted(title2triples.keys()):
        t2e_rows.append({"title": title, "entity_ids": sorted(title2entities.get(title, set()))})
        t2t_rows.append({"title": title, "triple_idxs": sorted(title2triples[title])})
    return t2e_rows, t2t_rows


def _write_kg_dir(out_dir: Path, kg: Dict, meta: Dict, reuse_title_indexes_from: Optional[Path] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "entities.jsonl", kg["entities"])
    _write_jsonl(out_dir / "triples.jsonl", kg["triples"])

    reused = False
    if isinstance(reuse_title_indexes_from, Path):
        src_t2e = reuse_title_indexes_from / "title2entities.jsonl"
        src_t2t = reuse_title_indexes_from / "title2triples.jsonl"
        if src_t2e.exists() and src_t2t.exists():
            shutil.copyfile(src_t2e, out_dir / "title2entities.jsonl")
            shutil.copyfile(src_t2t, out_dir / "title2triples.jsonl")
            reused = True

    if not reused:
        t2e_rows, t2t_rows = _rebuild_title_indexes(kg)
        _write_jsonl(out_dir / "title2entities.jsonl", t2e_rows)
        _write_jsonl(out_dir / "title2triples.jsonl", t2t_rows)

    meta["title_index_reused"] = bool(reused)
    (out_dir / "perturb_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _ratio_to_count(base: int, ratio: float) -> int:
    if base <= 0 or ratio <= 0:
        return 0
    n = int(round(float(base) * float(ratio)))
    if n <= 0:
        n = 1
    return min(base, n)


def _normalize_ratio(x) -> float:
    r = float(x)
    if r > 1.0:
        r = r / 100.0
    if r < 0.0 or r > 1.0:
        raise ValueError(f"Ratio must be in [0,1] or [0,100], got: {x}")
    return r


def _ratio_suffix(ratio: float) -> str:
    pct = float(ratio) * 100.0
    rounded = round(pct)
    if abs(pct - rounded) < 1e-9:
        return str(int(rounded))
    s = f"{pct:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _scenario_seed(base_seed: int, scenario_name: str) -> int:
    acc = 0
    for i, ch in enumerate(scenario_name):
        acc += (i + 1) * ord(ch)
    return int(base_seed + acc)


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _normalize_text(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> List[str]:
    s = _normalize_text(s)
    return re.findall(r"[a-z0-9]+", s)


def _is_var(x) -> bool:
    return isinstance(x, str) and x.startswith("?")


def _replace_first_case_insensitive(text: str, old: str, new: str) -> Tuple[str, bool]:
    if not text or not old:
        return text, False
    pat = re.compile(re.escape(old), flags=re.IGNORECASE)
    out, n = pat.subn(new, text, count=1)
    return out, bool(n > 0)


def _load_query_items(path: Path, max_samples: int) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Query JSON must be list: {path}")
    if max_samples > 0:
        data = data[:max_samples]
    return data


def _collect_query_terms(query_items: Sequence[Dict]) -> Dict:
    entities: Set[str] = set()
    relations: Set[str] = set()

    for item in query_items:
        if not isinstance(item, dict):
            continue
        qps = item.get("query_plan") or item.get("query_triples") or item.get("triples") or []
        if not isinstance(qps, list):
            continue
        for qt in qps:
            if not isinstance(qt, dict):
                continue
            h = qt.get("head")
            t = qt.get("tail")
            if isinstance(h, str) and h.strip() and not _is_var(h):
                entities.add(h.strip())
            if isinstance(t, str) and t.strip() and not _is_var(t):
                entities.add(t.strip())

            r = qt.get("relation")
            if isinstance(r, str) and r.strip():
                relations.add(_normalize_text(r))
            rv = qt.get("relation_variants", [])
            if isinstance(rv, list):
                for x in rv:
                    if isinstance(x, str) and x.strip():
                        relations.add(_normalize_text(x))

    return {"entities": sorted(entities), "relations": sorted(relations)}


def _build_entity_match_index(entities: Sequence[Dict]) -> Dict:
    exact = defaultdict(list)
    token_index = defaultdict(set)
    names_lower: List[str] = []

    for i, _ in enumerate(entities):
        nm = _entity_name(list(entities), i)
        low = _normalize_text(nm)
        names_lower.append(low)
        if low:
            exact[low].append(int(i))
        for tk in set(_tokenize(low)):
            token_index[tk].add(int(i))

    return {"exact": exact, "token_index": token_index, "names_lower": names_lower}


def _match_query_entity_to_ids(
    query_entity: str,
    match_index: Dict,
    topk: int,
    min_score: float,
) -> List[int]:
    q = _normalize_text(query_entity)
    if not q:
        return []

    exact = match_index["exact"]
    if q in exact:
        return [int(x) for x in exact[q][: max(1, int(topk))]]

    token_index = match_index["token_index"]
    names_lower = match_index["names_lower"]
    cands: Set[int] = set()
    for tk in set(_tokenize(q)):
        cands.update(token_index.get(tk, set()))
    if not cands:
        return []

    scored: List[Tuple[float, int]] = []
    for idx in cands:
        s = SequenceMatcher(None, q, names_lower[idx]).ratio()
        if s >= float(min_score):
            scored.append((float(s), int(idx)))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[: max(1, int(topk))]]


def _resolve_query_seed_ids_for_kg(kg: Dict, query_ctx: Dict) -> Set[int]:
    topk = int(query_ctx.get("entity_match_topk", 3))
    min_score = float(query_ctx.get("entity_match_min_score", 0.65))
    query_entities = query_ctx.get("query_entities", []) or []
    if not query_entities:
        return set()

    match_index = _build_entity_match_index(kg["entities"])
    seeds: Set[int] = set()
    for qe in query_entities:
        ids = _match_query_entity_to_ids(qe, match_index, topk=topk, min_score=min_score)
        for i in ids:
            if 0 <= i < len(kg["entities"]):
                seeds.add(int(i))
    return seeds


def _build_subgraph_scope(kg: Dict, seed_ids: Set[int], hops: int) -> Dict:
    n_entities = len(kg["entities"])
    n_triples = len(kg["triples"])
    entity2triples: List[List[int]] = [[] for _ in range(n_entities)]

    for tid, tr in enumerate(kg["triples"]):
        ht = _get_head_tail(tr)
        if ht is None:
            continue
        h, t = ht
        if 0 <= h < n_entities:
            entity2triples[h].append(int(tid))
        if 0 <= t < n_entities:
            entity2triples[t].append(int(tid))

    valid_seeds = {int(x) for x in seed_ids if 0 <= int(x) < n_entities}
    visited = set(valid_seeds)
    frontier = set(valid_seeds)
    depth = {int(x): 0 for x in valid_seeds}
    triple_ids: Set[int] = set()

    hops = max(0, int(hops))
    for _ in range(hops):
        if not frontier:
            break
        nxt: Set[int] = set()
        for eid in frontier:
            for tid in entity2triples[eid]:
                if 0 <= tid < n_triples:
                    triple_ids.add(int(tid))
                ht = _get_head_tail(kg["triples"][tid])
                if ht is None:
                    continue
                h, t = ht
                for nb in (h, t):
                    if not (0 <= nb < n_entities):
                        continue
                    if nb not in depth:
                        depth[nb] = int(depth[eid]) + 1
                    if nb not in visited:
                        nxt.add(int(nb))
        visited.update(nxt)
        frontier = nxt

    return {
        "entity_ids": sorted([int(x) for x in visited]),
        "triple_ids": sorted([int(x) for x in triple_ids]),
        "depth": {int(k): int(v) for k, v in depth.items()},
    }


def _build_scope_for_issue(kg: Dict, scope: str, query_ctx: Optional[Dict]) -> Dict:
    scope = str(scope or "global").strip()
    n_entities = len(kg["entities"])
    n_triples = len(kg["triples"])

    if scope == "global":
        return {
            "scope": "global",
            "entity_ids": list(range(n_entities)),
            "triple_ids": list(range(n_triples)),
            "depth": {},
            "query_relations": set(query_ctx.get("query_relations", [])) if isinstance(query_ctx, dict) else set(),
            "matched_seed_ids": [],
        }

    if scope != "query_subgraph":
        raise ValueError(f"Unknown scope: {scope}. Choose from ['global', 'query_subgraph'].")

    if not isinstance(query_ctx, dict):
        raise ValueError("scope=query_subgraph requires query context in config.")

    seeds = _resolve_query_seed_ids_for_kg(kg, query_ctx)
    hops = int(query_ctx.get("hops", 2))
    sub = _build_subgraph_scope(kg, seeds, hops=hops)

    if not sub["triple_ids"] or not sub["entity_ids"]:
        return {
            "scope": "global",
            "entity_ids": list(range(n_entities)),
            "triple_ids": list(range(n_triples)),
            "depth": {},
            "query_relations": set(query_ctx.get("query_relations", [])),
            "matched_seed_ids": [],
            "scope_fallback_reason": "query_subgraph_empty",
        }

    return {
        "scope": "query_subgraph",
        "entity_ids": sub["entity_ids"],
        "triple_ids": sub["triple_ids"],
        "depth": sub["depth"],
        "query_relations": set(query_ctx.get("query_relations", [])),
        "matched_seed_ids": sorted([int(x) for x in seeds]),
    }


def _drop_edges_from_pool(kg: Dict, ratio: float, rng: random.Random, triple_pool: Sequence[int]) -> int:
    triples = kg["triples"]
    valid_pool = [int(i) for i in triple_pool if 0 <= int(i) < len(triples)]
    n_drop = _ratio_to_count(len(valid_pool), ratio)
    if n_drop <= 0:
        return 0
    drop_idx = set(rng.sample(valid_pool, n_drop))
    kg["triples"] = [tr for i, tr in enumerate(triples) if i not in drop_idx]
    return int(n_drop)


def _relation_vocab(kg: Dict) -> List[str]:
    rels = []
    seen = set()
    for tr in kg["triples"]:
        r = _get_relation(tr).strip()
        if not r:
            continue
        low = r.lower()
        if low in seen:
            continue
        seen.add(low)
        rels.append(r)
    return rels


def _update_evidence_relation(kg: Dict, tr: Dict, old_relation: str, new_relation: str) -> bool:
    ev = str(tr.get("evidence") or "")
    if ev:
        ev2, replaced = _replace_first_case_insensitive(ev, old_relation, new_relation)
        if replaced:
            tr["evidence"] = ev2
            return True
    ht = _get_head_tail(tr)
    if ht is not None:
        h, t = ht
        tr["evidence"] = f"{_entity_name(kg['entities'], h)} {new_relation} {_entity_name(kg['entities'], t)}"
    return False


def _update_evidence_endpoint(kg: Dict, tr: Dict, old_eid: int, new_eid: int) -> bool:
    ev = str(tr.get("evidence") or "")
    old_name = _entity_name(kg["entities"], old_eid)
    new_name = _entity_name(kg["entities"], new_eid)
    if ev and old_name:
        ev2, replaced = _replace_first_case_insensitive(ev, old_name, new_name)
        if replaced:
            tr["evidence"] = ev2
            return True
    return False


def _strip_qualifier_text(text: str, cfg: Dict) -> str:
    if not text:
        return text
    out = str(text)

    if bool(cfg.get("strip_parenthetical", True)):
        out = re.sub(r"\([^)]*\)", "", out)
        out = re.sub(r"\[[^\]]*\]", "", out)

    if bool(cfg.get("strip_temporal_tokens", True)):
        out = re.sub(r"\b(1[0-9]{3}|20[0-9]{2})\b", "", out)
        out = re.sub(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            "",
            out,
            flags=re.IGNORECASE,
        )

    if bool(cfg.get("strip_trailing_comma_clause", True)):
        parts = [p.strip() for p in out.split(",") if p.strip()]
        if len(parts) >= 2:
            out = ", ".join(parts[:-1]).strip()

    out = re.sub(r"\s+", " ", out).strip()
    return out


def _relation_similarity(
    old_norm: str,
    old_tokens: Set[str],
    cand_norm: str,
    cand_tokens: Set[str],
    use_sequence_matcher: bool,
) -> float:
    if not old_norm or not cand_norm:
        return 0.0
    j = 0.0
    if old_tokens and cand_tokens:
        j = len(old_tokens & cand_tokens) / max(1, len(old_tokens | cand_tokens))
    if not use_sequence_matcher:
        return float(j)
    seq = SequenceMatcher(None, old_norm, cand_norm).ratio()
    return float(max(j, seq))


def _build_relation_vocab_records(relation_vocab: Sequence[str]) -> List[Dict]:
    recs: List[Dict] = []
    for r in relation_vocab:
        raw = str(r).strip()
        if not raw:
            continue
        norm = _normalize_text(raw)
        if not norm:
            continue
        recs.append({"raw": raw, "norm": norm, "tokens": set(_tokenize(norm))})
    return recs


def _auto_flip_relation(
    old_rel: str,
    relation_vocab_records: Sequence[Dict],
    issue_cfg: Dict,
    rng: random.Random,
    cache: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Tuple[str, str]:
    old_raw = str(old_rel or "").strip()
    old_norm = _normalize_text(old_raw)
    if not old_norm:
        return old_raw, "default_fallback"

    if isinstance(cache, dict) and old_norm in cache:
        return cache[old_norm]

    use_phrase_pairs = bool(issue_cfg.get("use_phrase_pairs", False))
    enable_vocab_fallback = bool(issue_cfg.get("enable_vocab_fallback", False))

    flip_map_cfg = issue_cfg.get("relation_flip_map", {}) or {}
    flip_map = {str(k).strip().lower(): str(v).strip() for k, v in flip_map_cfg.items() if str(k).strip()}
    mapped = flip_map.get(old_norm, "")
    if mapped and _normalize_text(mapped) != old_norm:
        ans = (mapped, "mapped_or_pattern")
        if isinstance(cache, dict):
            cache[old_norm] = ans
        return ans

    if use_phrase_pairs:
        default_pairs = [
            ["born", "died"],
            ["start", "end"],
            ["before", "after"],
            ["after", "before"],
            ["parent", "child"],
            ["father", "child"],
            ["mother", "child"],
            ["spouse", "sibling"],
            ["winner", "loser"],
            ["win", "lose"],
            ["contain", "located in"],
            ["located in", "contains"],
            ["employer", "employee"],
            ["member", "opponent"],
            ["director", "actor"],
        ]
        phrase_pairs = issue_cfg.get("auto_flip_pairs", default_pairs)
        pairs: List[Tuple[str, str]] = []
        if isinstance(phrase_pairs, list):
            for row in phrase_pairs:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    a = str(row[0]).strip().lower()
                    b = str(row[1]).strip().lower()
                    if a and b:
                        pairs.append((a, b))
                        pairs.append((b, a))

        old_low = old_raw.lower()
        for src, dst in pairs:
            if src in old_low:
                cand = re.sub(re.escape(src), dst, old_raw, count=1, flags=re.IGNORECASE).strip()
                if cand and _normalize_text(cand) != old_norm:
                    ans = (cand, "mapped_or_pattern")
                    if isinstance(cache, dict):
                        cache[old_norm] = ans
                    return ans

    if enable_vocab_fallback:
        rel_records: Sequence[Dict] = relation_vocab_records
        max_candidates = int(issue_cfg.get("auto_flip_max_vocab_candidates", 512))
        if max_candidates > 0 and len(relation_vocab_records) > max_candidates:
            keep_head = int(issue_cfg.get("auto_flip_keep_head", 32))
            keep_head = max(0, min(keep_head, max_candidates))
            chosen: List[Dict] = []
            if keep_head > 0:
                chosen.extend(list(relation_vocab_records[:keep_head]))
            need = max_candidates - len(chosen)
            rest_start = keep_head
            rest_len = max(0, len(relation_vocab_records) - rest_start)
            if need > 0 and rest_len > 0:
                sampled_idx = rng.sample(range(rest_start, len(relation_vocab_records)), min(need, rest_len))
                chosen.extend([relation_vocab_records[i] for i in sampled_idx])
            rel_records = chosen

        if rel_records:
            max_sim = float(issue_cfg.get("auto_flip_max_similarity", 0.35))
            use_seq = bool(issue_cfg.get("auto_flip_use_sequence_matcher", False))
            old_tokens = set(_tokenize(old_norm))
            scored: List[Tuple[float, str]] = []
            for rec in rel_records:
                cand = str(rec.get("raw", "")).strip()
                cand_norm = str(rec.get("norm", "")).strip()
                if not cand or not cand_norm or cand_norm == old_norm:
                    continue
                sim = _relation_similarity(
                    old_norm=old_norm,
                    old_tokens=old_tokens,
                    cand_norm=cand_norm,
                    cand_tokens=rec.get("tokens", set()),
                    use_sequence_matcher=use_seq,
                )
                scored.append((sim, cand))
            low = [r for s, r in scored if s <= max_sim]
            if low:
                ans = (str(rng.choice(low)), "vocab_based")
                if isinstance(cache, dict):
                    cache[old_norm] = ans
                return ans
            scored.sort(key=lambda x: x[0])
            if scored:
                ans = (str(scored[0][1]), "vocab_based")
                if isinstance(cache, dict):
                    cache[old_norm] = ans
                return ans

    default_rel = str(issue_cfg.get("fallback_default_relation", "is unrelated to")).strip()
    ans = (default_rel or old_raw, "default_fallback")
    if isinstance(cache, dict):
        cache[old_norm] = ans
    return ans


def _issue_semantic_flip(
    kg: Dict,
    ratio: float,
    rng: random.Random,
    scope_info: Dict,
    issue_cfg: Dict,
    noise_cfg: Dict,
) -> Dict:
    pool = [int(i) for i in scope_info["triple_ids"] if 0 <= int(i) < len(kg["triples"])]
    n_change = _ratio_to_count(len(pool), ratio)
    if n_change <= 0:
        return {"pool_size": len(pool), "changed": 0}

    enable_vocab_fallback = bool(issue_cfg.get("enable_vocab_fallback", False))
    relation_vocab_records: List[Dict] = []
    if enable_vocab_fallback:
        relation_vocab = _relation_vocab(kg)
        relation_vocab_records = _build_relation_vocab_records(relation_vocab)

    selected = rng.sample(pool, n_change)
    changed = 0
    evidence_replaced = 0
    strategy_stats = {"mapped_or_pattern": 0, "vocab_based": 0, "default_fallback": 0}
    relation_flip_cache: Dict[str, Tuple[str, str]] = {}
    for tid in selected:
        tr = kg["triples"][tid]
        old_r = _get_relation(tr)
        old_l = _normalize_text(old_r)
        new_r, strategy = _auto_flip_relation(
            old_rel=old_r,
            relation_vocab_records=relation_vocab_records,
            issue_cfg=issue_cfg,
            rng=rng,
            cache=relation_flip_cache,
        )
        new_l = _normalize_text(new_r)

        if not new_r or new_l == old_l:
            continue

        strategy_stats[strategy if strategy in strategy_stats else "default_fallback"] += 1

        _set_relation(tr, new_r)
        if _update_evidence_relation(kg, tr, old_r, new_r):
            evidence_replaced += 1
        changed += 1

    return {
        "pool_size": len(pool),
        "selected": int(n_change),
        "changed": int(changed),
        "evidence_relation_replaced": int(evidence_replaced),
        "flip_strategy_stats": strategy_stats,
        "flip_cache_size": int(len(relation_flip_cache)),
    }


def _issue_mis_bound_relation(
    kg: Dict,
    ratio: float,
    rng: random.Random,
    scope_info: Dict,
    issue_cfg: Dict,
    noise_cfg: Dict,
) -> Dict:
    pool = [int(i) for i in scope_info["triple_ids"] if 0 <= int(i) < len(kg["triples"])]
    entity_pool = [int(i) for i in scope_info["entity_ids"] if 0 <= int(i) < len(kg["entities"])]
    if not entity_pool:
        entity_pool = list(range(len(kg["entities"])))

    n_change = _ratio_to_count(len(pool), ratio)
    if n_change <= 0:
        return {"pool_size": len(pool), "changed": 0}

    endpoint = str(issue_cfg.get("endpoint", "tail")).strip().lower()
    same_type_only = bool(issue_cfg.get("same_type_only", True))

    type_to_ids = defaultdict(list)
    for eid in entity_pool:
        t = _entity_type(kg["entities"], eid).lower()
        type_to_ids[t].append(int(eid))

    selected = rng.sample(pool, n_change)
    changed = 0
    evidence_entity_replaced = 0

    for tid in selected:
        tr = kg["triples"][tid]
        ht = _get_head_tail(tr)
        if ht is None:
            continue
        h, t = ht

        side = endpoint
        if side not in {"head", "tail"}:
            side = "head" if rng.random() < 0.5 else "tail"

        old_eid = h if side == "head" else t
        other_eid = t if side == "head" else h

        cands = list(entity_pool)
        if same_type_only:
            ty = _entity_type(kg["entities"], old_eid).lower()
            if ty in type_to_ids and len(type_to_ids[ty]) >= 2:
                cands = list(type_to_ids[ty])

        cands = [x for x in cands if x != old_eid and x != other_eid]
        if not cands:
            continue
        new_eid = int(rng.choice(cands))

        if side == "head":
            _set_head_tail(tr, new_eid, t)
        else:
            _set_head_tail(tr, h, new_eid)

        if _update_evidence_endpoint(kg, tr, old_eid, new_eid):
            evidence_entity_replaced += 1
        changed += 1

    return {
        "pool_size": len(pool),
        "selected": int(n_change),
        "changed": int(changed),
        "evidence_entity_replaced": int(evidence_entity_replaced),
    }


def _issue_over_generalized_relation(
    kg: Dict,
    ratio: float,
    rng: random.Random,
    scope_info: Dict,
    issue_cfg: Dict,
    noise_cfg: Dict,
) -> Dict:
    pool = [int(i) for i in scope_info["triple_ids"] if 0 <= int(i) < len(kg["triples"])]
    n_ops = _ratio_to_count(len(pool), ratio)
    if n_ops <= 0:
        return {"pool_size": len(pool), "changed": 0, "added": 0}

    mode = str(issue_cfg.get("mode", "replace")).strip().lower()
    generic_rel = issue_cfg.get("generic_relations") or [
        "related to",
        "associated with",
        "connected to",
        "linked with",
    ]
    generic_rel = [str(x).strip() for x in generic_rel if str(x).strip()]
    if not generic_rel:
        generic_rel = ["related to"]

    selected = rng.sample(pool, n_ops)
    replaced = 0
    added = 0
    evidence_replaced = 0

    if mode == "add":
        for tid in selected:
            if not (0 <= tid < len(kg["triples"])):
                continue
            old = kg["triples"][tid]
            new_tr = dict(old)
            old_r = _get_relation(new_tr)
            new_r = rng.choice(generic_rel)
            _set_relation(new_tr, new_r)
            _update_evidence_relation(kg, new_tr, old_r, new_r)
            kg["triples"].append(new_tr)
            added += 1
    else:
        for tid in selected:
            tr = kg["triples"][tid]
            old_r = _get_relation(tr)
            new_r = rng.choice(generic_rel)
            if old_r.strip().lower() == new_r.strip().lower():
                continue
            _set_relation(tr, new_r)
            if _update_evidence_relation(kg, tr, old_r, new_r):
                evidence_replaced += 1
            replaced += 1

    return {
        "pool_size": len(pool),
        "selected": int(n_ops),
        "mode": mode,
        "changed": int(replaced),
        "added": int(added),
        "evidence_relation_replaced": int(evidence_replaced),
    }


def _issue_missing_bridge_edge(
    kg: Dict,
    ratio: float,
    rng: random.Random,
    scope_info: Dict,
    issue_cfg: Dict,
    noise_cfg: Dict,
) -> Dict:
    pool = [int(i) for i in scope_info["triple_ids"] if 0 <= int(i) < len(kg["triples"])]
    if not pool:
        return {"pool_size": 0, "bridge_pool_size": 0, "dropped_edges": 0}

    depth = scope_info.get("depth", {}) or {}
    query_rel = {str(x).strip().lower() for x in scope_info.get("query_relations", set()) if str(x).strip()}
    prefer_query_rel = bool(issue_cfg.get("prefer_query_relations", True))
    prefer_depth_bridge = bool(issue_cfg.get("prefer_depth_bridge", True))

    bridge_pool: List[int] = []
    for tid in pool:
        tr = kg["triples"][tid]
        ht = _get_head_tail(tr)
        if ht is None:
            continue
        h, t = ht
        ok = True

        if prefer_depth_bridge and depth:
            dh = depth.get(h, None)
            dt = depth.get(t, None)
            if dh is None or dt is None:
                ok = False
            elif abs(int(dh) - int(dt)) < 1:
                ok = False

        if ok and prefer_query_rel and query_rel:
            rel = _get_relation(tr).strip().lower()
            if rel not in query_rel:
                fuzzy = False
                for qr in query_rel:
                    if rel in qr or qr in rel:
                        fuzzy = True
                        break
                if not fuzzy:
                    ok = False

        if ok:
            bridge_pool.append(int(tid))

    candidates = bridge_pool if bridge_pool else pool
    dropped = _drop_edges_from_pool(kg, ratio, rng, candidates)
    return {
        "pool_size": len(pool),
        "bridge_pool_size": len(bridge_pool),
        "selected_pool_size": len(candidates),
        "dropped_edges": int(dropped),
    }


def _issue_dropped_qualifier(
    kg: Dict,
    ratio: float,
    rng: random.Random,
    scope_info: Dict,
    issue_cfg: Dict,
    noise_cfg: Dict,
) -> Dict:
    pool = [int(i) for i in scope_info["triple_ids"] if 0 <= int(i) < len(kg["triples"])]
    n_change = _ratio_to_count(len(pool), ratio)
    if n_change <= 0:
        return {"pool_size": len(pool), "triples_changed": 0, "keys_removed": 0, "evidence_shortened": 0}

    qualifier_keys = issue_cfg.get(
        "qualifier_keys",
        [
            "qualifier",
            "qualifiers",
            "time",
            "date",
            "year",
            "start_time",
            "end_time",
            "point_in_time",
            "location",
        ],
    )
    qualifier_keys = [str(k).strip() for k in qualifier_keys if str(k).strip()]

    selected = rng.sample(pool, n_change)
    triples_changed = 0
    keys_removed = 0
    evidence_shortened = 0

    for tid in selected:
        tr = kg["triples"][tid]
        changed = False

        for k in qualifier_keys:
            if k in tr:
                tr.pop(k, None)
                keys_removed += 1
                changed = True

        old_ev = str(tr.get("evidence") or "")
        new_ev = _strip_qualifier_text(old_ev, issue_cfg)
        if new_ev and new_ev != old_ev:
            tr["evidence"] = new_ev
            evidence_shortened += 1
            changed = True

        if changed:
            triples_changed += 1

    return {
        "pool_size": len(pool),
        "selected": int(n_change),
        "triples_changed": int(triples_changed),
        "keys_removed": int(keys_removed),
        "evidence_shortened": int(evidence_shortened),
    }


_ISSUE_HANDLERS = {
    "semantic_flip": _issue_semantic_flip,
    "mis_bound_relation": _issue_mis_bound_relation,
    "over_generalized_relation": _issue_over_generalized_relation,
    "missing_bridge_edge": _issue_missing_bridge_edge,
    "dropped_qualifier": _issue_dropped_qualifier,
}


def _normalize_scopes(scopes_raw) -> List[str]:
    if isinstance(scopes_raw, str):
        scopes_raw = [scopes_raw]
    if not isinstance(scopes_raw, list) or not scopes_raw:
        scopes_raw = ["global"]

    out = []
    seen = set()
    for x in scopes_raw:
        s = str(x).strip()
        if not s:
            continue
        if s not in {"global", "query_subgraph"}:
            raise ValueError(f"Unknown scope '{s}'. Allowed: global, query_subgraph")
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if not out:
        out = ["global"]
    return out


def _normalize_issue_types(issue_types_raw) -> List[str]:
    if isinstance(issue_types_raw, str):
        issue_types_raw = [issue_types_raw]
    if not isinstance(issue_types_raw, list):
        raise ValueError("issue_types must be a list of issue names.")
    out = []
    seen = set()
    for x in issue_types_raw:
        it = str(x).strip()
        if not it:
            continue
        if it not in _ISSUE_HANDLERS:
            raise ValueError(f"Unknown issue type '{it}'. Available: {sorted(_ISSUE_HANDLERS.keys())}")
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _load_ratios(raw) -> List[float]:
    if isinstance(raw, (int, float)):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raw = [0.1]
    return [_normalize_ratio(x) for x in raw]


def _build_scenarios_from_issue_sets(config: Dict) -> List[Dict]:
    run_cfg = config.get("run", {}) or {}
    issue_sets_raw = run_cfg.get("issue_sets", None)
    inline_issue_types = run_cfg.get("issue_types", None)

    issue_sets: List[Dict] = []
    if isinstance(issue_sets_raw, list) and issue_sets_raw:
        for i, row in enumerate(issue_sets_raw):
            if isinstance(row, str):
                issue_sets.append({"name": str(row), "issue_types": [str(row)]})
            elif isinstance(row, dict):
                nm = str(row.get("name", "")).strip() or f"issue_set_{i}"
                issue_sets.append({"name": nm, **row})
    elif inline_issue_types:
        issue_sets = [{"name": "inline_issue_types", "issue_types": inline_issue_types}]
    else:
        return []

    global_ratios = _load_ratios(run_cfg.get("ratios", [0.1]))
    global_scopes = _normalize_scopes(run_cfg.get("scopes", ["global"]))
    global_combine_mode = str(run_cfg.get("combine_mode", "cartesian")).strip().lower()
    if global_combine_mode not in {"cartesian", "aligned"}:
        raise ValueError("run.combine_mode must be 'cartesian' or 'aligned'.")
    include_clean_copy = bool(run_cfg.get("include_clean_copy", False))

    scenarios: List[Dict] = []
    if include_clean_copy:
        scenarios.append(
            {
                "name": "clean_copy",
                "scenario_type": "issue_set",
                "issue_set_name": "clean_copy",
                "issue_types": [],
                "ratio": 0.0,
                "scope": "global",
            }
        )

    for row in issue_sets:
        set_name = str(row.get("name", "")).strip() or "issue_set"
        issue_types = _normalize_issue_types(row.get("issue_types", []))
        set_ratios = _load_ratios(row.get("ratios", global_ratios))
        set_scopes = _normalize_scopes(row.get("scopes", global_scopes))
        set_combine_mode = str(row.get("combine_mode", global_combine_mode)).strip().lower()
        if set_combine_mode not in {"cartesian", "aligned"}:
            raise ValueError(f"issue_set '{set_name}' combine_mode must be 'cartesian' or 'aligned'.")
        per_set_issue_params = row.get("issue_params", {}) if isinstance(row.get("issue_params"), dict) else {}

        if not issue_types:
            for scope in set_scopes[:1]:
                scenarios.append(
                    {
                        "name": set_name,
                        "scenario_type": "issue_set",
                        "issue_set_name": set_name,
                        "issue_types": [],
                        "ratio": 0.0,
                        "scope": scope,
                        "issue_params": per_set_issue_params,
                    }
                )
            continue

        pairs: List[Tuple[float, str]] = []
        if set_combine_mode == "aligned":
            if len(set_ratios) != len(set_scopes):
                raise ValueError(
                    f"issue_set '{set_name}' uses combine_mode=aligned, but ratios({len(set_ratios)}) "
                    f"!= scopes({len(set_scopes)})."
                )
            pairs = [(float(r), s) for r, s in zip(set_ratios, set_scopes)]
        else:
            pairs = [(float(r), s) for r in set_ratios for s in set_scopes]

        for ratio, scope in pairs:
            scope_tag = "qsub" if scope == "query_subgraph" else "global"
            sc_name = f"{set_name}_{_ratio_suffix(ratio)}_{scope_tag}"
            scenarios.append(
                {
                    "name": sc_name,
                    "scenario_type": "issue_set",
                    "issue_set_name": set_name,
                    "issue_types": issue_types,
                    "ratio": float(ratio),
                    "scope": scope,
                    "combine_mode": set_combine_mode,
                    "issue_params": per_set_issue_params,
                }
            )
    return scenarios


def _filter_scenarios_by_tokens(scenarios: List[Dict], wanted_tokens: List[str]) -> List[Dict]:
    if not wanted_tokens:
        return scenarios
    wanted = {str(x).strip() for x in wanted_tokens if str(x).strip()}
    out: List[Dict] = []
    for s in scenarios:
        nm = str(s.get("name", "")).strip()
        st = str(s.get("scenario_type", "")).strip()
        issue_set_name = str(s.get("issue_set_name", "")).strip()
        issues = [str(x).strip() for x in (s.get("issue_types") or [])]
        if nm in wanted or st in wanted or issue_set_name in wanted or any(i in wanted for i in issues):
            out.append(s)
    return out


def _scenario_changes_graph_structure(scenario: Dict, config: Dict) -> bool:
    ratio = float(scenario.get("ratio", 0.0) or 0.0)
    if ratio <= 0:
        return False

    issue_types = [str(x).strip() for x in (scenario.get("issue_types", []) or []) if str(x).strip()]
    if not issue_types:
        return False

    global_issue_cfg = config.get("issue_params", {}) or {}
    scenario_issue_cfg = scenario.get("issue_params", {}) or {}

    for issue in issue_types:
        if issue in {"missing_bridge_edge", "mis_bound_relation"}:
            return True
        if issue == "over_generalized_relation":
            merged = {}
            if isinstance(global_issue_cfg.get(issue), dict):
                merged.update(global_issue_cfg.get(issue))
            if isinstance(scenario_issue_cfg.get(issue), dict):
                merged.update(scenario_issue_cfg.get(issue))
            mode = str(merged.get("mode", "replace")).strip().lower()
            if mode == "add":
                return True
        if issue not in {"semantic_flip", "dropped_qualifier", "over_generalized_relation"}:
            return True
    return False


def _apply_scenario_issue_set(
    base_kg: Dict,
    scenario: Dict,
    config: Dict,
    noise_cfg: Dict,
    query_ctx: Optional[Dict],
    seed: int,
) -> Tuple[Dict, Dict]:
    rng = random.Random(seed)
    kg = _copy_kg_fast(base_kg)

    ratio = float(scenario.get("ratio", 0.0) or 0.0)
    scope = str(scenario.get("scope", "global"))
    issue_types = [str(x).strip() for x in (scenario.get("issue_types", []) or []) if str(x).strip()]

    global_issue_cfg = config.get("issue_params", {}) or {}
    scenario_issue_cfg = scenario.get("issue_params", {}) or {}
    run_cfg = config.get("run", {}) or {}
    final_sanitize = bool(run_cfg.get("final_sanitize", False))

    stats = {
        "scope": scope,
        "ratio": float(ratio),
        "issue_types": issue_types,
        "issue_stats": {},
        "scope_debug": {},
    }

    for issue in issue_types:
        if issue not in _ISSUE_HANDLERS:
            raise ValueError(f"Unknown issue type in scenario: {issue}")

        scope_info = _build_scope_for_issue(kg, scope, query_ctx=query_ctx)
        if scope_info.get("scope_fallback_reason"):
            stats["scope_debug"]["fallback_reason"] = scope_info["scope_fallback_reason"]

        merged_issue_cfg = {}
        if isinstance(global_issue_cfg.get(issue), dict):
            merged_issue_cfg.update(global_issue_cfg.get(issue))
        if isinstance(scenario_issue_cfg.get(issue), dict):
            merged_issue_cfg.update(scenario_issue_cfg.get(issue))

        handler = _ISSUE_HANDLERS[issue]
        issue_stats = handler(
            kg=kg,
            ratio=ratio,
            rng=rng,
            scope_info=scope_info,
            issue_cfg=merged_issue_cfg,
            noise_cfg=noise_cfg,
        )
        stats["issue_stats"][issue] = issue_stats

    if final_sanitize:
        kg["entities"] = _sanitize_entities(kg["entities"])
        kg["triples"] = _sanitize_triples(kg["triples"], len(kg["entities"]))
    return kg, stats


def _resolve_dataset(config: Dict, dataset_override: Optional[str]) -> Tuple[str, Dict]:
    run_cfg = config.get("run", {}) or {}
    dataset = dataset_override or run_cfg.get("dataset")
    if not dataset:
        raise ValueError("Dataset not specified. Set run.dataset or pass --dataset.")
    dataset = str(dataset)
    datasets = config.get("datasets", {}) or {}
    ds_cfg = datasets.get(dataset)
    if not isinstance(ds_cfg, dict):
        raise ValueError(f"Dataset '{dataset}' not found in config.datasets.")
    return dataset, ds_cfg


def _build_query_context_if_needed(config: Dict, ds_cfg: Dict, scenarios: List[Dict], base_kg: Dict) -> Optional[Dict]:
    need_query_subgraph = any(str(s.get("scope", "global")) == "query_subgraph" for s in scenarios)
    if not need_query_subgraph:
        return None

    run_cfg = config.get("run", {}) or {}
    qcfg = run_cfg.get("query_subgraph", {}) or {}

    query_json_val = str(qcfg.get("query_json", "")).strip() or str(ds_cfg.get("query_json", "")).strip()
    if not query_json_val:
        raise ValueError(
            "query_subgraph scope requested but query_json is missing. "
            "Set run.query_subgraph.query_json or datasets.<dataset>.query_json."
        )
    query_json = Path(query_json_val)
    if not query_json.exists():
        raise FileNotFoundError(f"query_json not found: {query_json}")

    max_samples = _safe_int(qcfg.get("max_samples", 1000), 1000)
    hops = _safe_int(qcfg.get("hops", 2), 2)
    topk = _safe_int(qcfg.get("entity_match_topk", 3), 3)
    min_score = float(qcfg.get("entity_match_min_score", 0.65))

    query_items = _load_query_items(query_json, max_samples=max_samples)
    terms = _collect_query_terms(query_items)

    ctx = {
        "query_json": str(query_json),
        "query_entities": terms["entities"],
        "query_relations": terms["relations"],
        "max_samples": int(max_samples),
        "hops": int(hops),
        "entity_match_topk": int(topk),
        "entity_match_min_score": float(min_score),
    }

    seed_preview = _resolve_query_seed_ids_for_kg(base_kg, ctx)
    ctx["seed_preview_count_on_base_kg"] = int(len(seed_preview))
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline KG perturbation for robustness experiments.")
    parser.add_argument("--config", default="configs/kg_perturb.yaml", help="Path to YAML/JSON config.")
    parser.add_argument("--dataset", default=None, help="Override dataset key in config.")
    parser.add_argument("--scenario", action="append", default=[], help="Run only selected scenario(s).")
    parser.add_argument("--list", action="store_true", help="List scenarios and exit.")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    run_cfg = config.get("run", {}) or {}
    cli_wanted = [str(x).strip() for x in (args.scenario or []) if str(x).strip()]

    scenarios = _build_scenarios_from_issue_sets(config)
    wanted = list(cli_wanted)
    if not wanted:
        selected = run_cfg.get("selected_scenarios", [])
        if isinstance(selected, list):
            wanted = [str(x).strip() for x in selected if str(x).strip()]
    scenarios = _filter_scenarios_by_tokens(scenarios, wanted)
    mode = "issue_sets"

    if args.list:
        print(f"[KG-Perturb] mode={mode}")
        print("Scenarios:")
        for s in scenarios:
            print(f" - {s.get('name')}")
        return

    if not scenarios:
        raise ValueError("No scenarios selected. Check run.issue_sets / run.selected_scenarios / --scenario.")

    dataset, ds_cfg = _resolve_dataset(config, args.dataset)
    kg_dir = Path(str(ds_cfg.get("kg_dir", "")).strip())
    if not kg_dir.exists():
        raise FileNotFoundError(f"KG directory not found: {kg_dir}")

    base_seed = _safe_int(config.get("seed", 42), 42)
    output_root = Path(str(config.get("output_root", "artifacts/kg_perturbed")))
    noise_cfg = config.get("noise", {}) or {}

    base_kg = _load_kg_dir(kg_dir)
    base_kg["entities"] = _sanitize_entities(base_kg["entities"])
    base_kg["triples"] = _sanitize_triples(base_kg["triples"], len(base_kg["entities"]))

    query_ctx = _build_query_context_if_needed(config, ds_cfg, scenarios, base_kg)

    print(f"[KG-Perturb] mode={mode}")
    print(f"[KG-Perturb] dataset={dataset}")
    print(f"[KG-Perturb] input={kg_dir}")
    print(f"[KG-Perturb] entities={len(base_kg['entities'])} triples={len(base_kg['triples'])}")
    print(f"[KG-Perturb] scenarios={len(scenarios)}")
    print(
        "[KG-Perturb] expansion:"
        f" combine_mode={str((run_cfg.get('combine_mode', 'cartesian'))).strip().lower()},"
        f" ratios={run_cfg.get('ratios', [0.1])},"
        f" scopes={run_cfg.get('scopes', ['global'])}"
    )
    if query_ctx is not None:
        print(
            "[KG-Perturb] query_subgraph context:"
            f" query_json={query_ctx['query_json']},"
            f" query_entities={len(query_ctx['query_entities'])},"
            f" query_relations={len(query_ctx['query_relations'])},"
            f" base_seed_matches={query_ctx['seed_preview_count_on_base_kg']}"
        )

    for sc in scenarios:
        sc_name = str(sc.get("name", "scenario"))
        seed = _scenario_seed(base_seed, sc_name)

        perturbed_kg, stats = _apply_scenario_issue_set(
            base_kg=base_kg,
            scenario=sc,
            config=config,
            noise_cfg=noise_cfg,
            query_ctx=query_ctx,
            seed=seed,
        )

        out_dir = output_root / sc_name / kg_dir.name
        meta = {
            "dataset": dataset,
            "mode": mode,
            "scenario": sc_name,
            "seed": seed,
            "input_kg_dir": str(kg_dir),
            "output_kg_dir": str(out_dir),
            "counts_before": {"entities": len(base_kg["entities"]), "triples": len(base_kg["triples"])},
            "counts_after": {"entities": len(perturbed_kg["entities"]), "triples": len(perturbed_kg["triples"])},
            "scenario_config": sc,
            "query_context": query_ctx if query_ctx is not None else {},
            "stats": stats,
        }
        needs_reindex = _scenario_changes_graph_structure(sc, config=config)
        _write_kg_dir(
            out_dir,
            perturbed_kg,
            meta,
            reuse_title_indexes_from=None if needs_reindex else kg_dir,
        )

        print(
            f"[KG-Perturb] scenario={sc_name}"
            f" -> entities={len(perturbed_kg['entities'])}, triples={len(perturbed_kg['triples'])}"
            f", out={out_dir}, title_index_reused={str(bool(meta.get('title_index_reused', False))).lower()}"
        )

    print("[KG-Perturb] done.")


if __name__ == "__main__":
    main()
