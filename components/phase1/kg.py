"""Phase-1 KG utilities (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    build_kg_indices,
    ent_name,
    get_1hop_triples,
    is_var,
    load_kg,
    map_query_entities,
    map_surface_to_entity_ids,
    parse_triple,
)

__all__ = [
    "build_kg_indices",
    "ent_name",
    "get_1hop_triples",
    "is_var",
    "load_kg",
    "map_query_entities",
    "map_surface_to_entity_ids",
    "parse_triple",
]

