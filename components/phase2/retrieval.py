"""Phase-2 retrieval and matching helpers (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    any_keyword_match,
    build_phase1_support_map,
    build_query_texts,
    build_rerank_query,
    extract_focus_entities,
    extract_query_relations,
    extract_union_keywords,
    keyword_groups_for_sentence_filter,
    retrieve_2wiki_like,
    retrieve_musique,
    sentence_satisfies_groups,
    tri_in_t_i,
)

__all__ = [
    "any_keyword_match",
    "build_phase1_support_map",
    "build_query_texts",
    "build_rerank_query",
    "extract_focus_entities",
    "extract_query_relations",
    "extract_union_keywords",
    "keyword_groups_for_sentence_filter",
    "retrieve_2wiki_like",
    "retrieve_musique",
    "sentence_satisfies_groups",
    "tri_in_t_i",
]

