/** Display-only graph projection. Never changes stored triples or retrieval. */
export type GraphTriple = Record<string, unknown>;
export type GraphSelection = { kind: "node"; name: string; id: string } | { kind: "edge"; triple: GraphTriple };
export interface GraphNode { id: string; label: string; x: number; y: number; degree: number }
export interface GraphLink { id: string; from: GraphNode; to: GraphNode; triple: GraphTriple }
export interface GraphLayout { nodes: GraphNode[]; links: GraphLink[]; width: number; height: number }

export function graphEndpoint(triple: GraphTriple, side: "head" | "tail"): { id: string; label: string } {
  const entity = triple[side + "_entity"] as { id?: unknown; label?: unknown } | undefined;
  return { id: String(entity?.id ?? triple[side]), label: String(entity?.label ?? triple[side]) };
}

export function tripleKey(triple: GraphTriple): string {
  return String(triple.triple_id ?? triple.triple_hash ?? JSON.stringify([triple.head, triple.relation, triple.tail]));
}

export function filterGraph(
  triples: GraphTriple[],
  options: { search?: string; source?: string; relation?: string; evidenceOnly?: boolean; neighbor?: string },
): GraphTriple[] {
  const search = (options.search ?? "").trim().toLowerCase();
  return triples.filter(triple =>
    (!search || [triple.head, triple.relation, triple.tail].join(" ").toLowerCase().includes(search)) &&
    (!options.source || String(triple.source_id ?? "") === options.source) &&
    (!options.relation || String(triple.relation) === options.relation) &&
    (!options.evidenceOnly || String(triple.evidence ?? "").trim().length > 0) &&
    (!options.neighbor || graphEndpoint(triple, "head").id === options.neighbor || graphEndpoint(triple, "tail").id === options.neighbor),
  );
}

export function relatedTriples(triples: GraphTriple[], name: string): GraphTriple[] {
  return triples.filter(triple => graphEndpoint(triple, "head").id === name || graphEndpoint(triple, "tail").id === name);
}

/** Restore the content-labelled radial overview; positions carry no semantic meaning. */
export function layoutGraph(triples: GraphTriple[]): GraphLayout {
  const adjacency = new Map<string, Set<string>>();
  const labels = new Map<string, string>();
  for (const triple of triples) {
    const from = graphEndpoint(triple, "head"), to = graphEndpoint(triple, "tail");
    labels.set(from.id, from.label); labels.set(to.id, to.label);
    if (!adjacency.has(from.id)) adjacency.set(from.id, new Set());
    if (!adjacency.has(to.id)) adjacency.set(to.id, new Set());
    adjacency.get(from.id)!.add(to.id);
    adjacency.get(to.id)!.add(from.id);
  }
  const nodes: GraphNode[] = [...labels].map(([id, label], index) => {
    const angle = index * Math.PI * 2 / labels.size - Math.PI / 2;
    const radius = index % 3 === 0 ? 138 : 197;
    return {
      id, label, degree: adjacency.get(id)!.size,
      x: 460 + Math.cos(angle) * radius * 1.65,
      y: 263 + Math.sin(angle) * radius,
    };
  });
  const lookup = new Map(nodes.map(node => [node.id, node]));
  return {
    nodes,
    links: triples.map(triple => ({ id: tripleKey(triple), from: lookup.get(graphEndpoint(triple, "head").id)!, to: lookup.get(graphEndpoint(triple, "tail").id)!, triple })),
    width: 920, height: 530,
  };
}

export function edgePath(from: GraphNode, to: GraphNode): string {
  return from.id === to.id
    ? `M${from.x - 5} ${from.y - 6} C${from.x - 48} ${from.y - 70},${from.x + 48} ${from.y - 70},${from.x + 5} ${from.y - 6}`
    : `M${from.x} ${from.y} L${to.x} ${to.y}`;
}
