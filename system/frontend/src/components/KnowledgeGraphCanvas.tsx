import { useEffect, useId, useMemo, useRef, useState } from "react";
import { edgePath, graphEndpoint, layoutGraph, relatedTriples, tripleKey, type GraphSelection, type GraphTriple } from "../utils/knowledgeGraph";

export function KnowledgeGraphCanvas({ triples, selection, onSelect }: {
  triples: GraphTriple[]; selection: GraphSelection | null; onSelect: (selection: GraphSelection | null) => void;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{ x: number; y: number; originX: number; originY: number } | null>(null);
  const [camera, setCamera] = useState({ zoom: 1, x: 0, y: 0 });
  const [hover, setHover] = useState<string | null>(null);
  const [labels, setLabels] = useState(true);
  const [viewport, setViewport] = useState({ width: 1200, height: 700 });
  const layout = useMemo(() => layoutGraph(triples), [triples]);
  const marker = "arrow-" + useId().replace(/:/g, "");
  const fit = Math.min(Math.max(100, viewport.width - 72) / layout.width, Math.max(100, viewport.height - 100) / layout.height);
  const offsetX = (viewport.width - layout.width * fit) / 2, offsetY = (viewport.height - layout.height * fit) / 2;
  const focusName = hover ?? (selection?.kind === "node" ? selection.id : null);
  const neighbors = useMemo(() => focusName
    ? new Set(relatedTriples(triples, focusName).flatMap(triple => [graphEndpoint(triple, "head").id, graphEndpoint(triple, "tail").id]))
    : null, [triples, focusName]);
  useEffect(() => { setCamera({ zoom: 1, x: 0, y: 0 }); setHover(null); }, [layout]);
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setViewport(current => Math.abs(current.width - width) > 1 || Math.abs(current.height - height) > 1 ? { width, height } : current);
    });
    observer.observe(svg);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      const matrix = svg.getScreenCTM();
      if (!matrix) return;
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
      const view = svg.viewBox.baseVal;
      const x = point.x - view.width / 2, y = point.y - view.height / 2;
      setCamera(current => {
        const zoom = Math.max(.45, Math.min(4, current.zoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12)));
        const ratio = zoom / current.zoom;
        return { zoom, x: x - (x - current.x) * ratio, y: y - (y - current.y) * ratio };
      });
    };
    svg.addEventListener("wheel", wheel, { passive: false });
    return () => svg.removeEventListener("wheel", wheel);
  }, []);
  const zoomBy = (factor: number) => setCamera(current => ({ ...current, zoom: Math.max(.45, Math.min(4, current.zoom * factor)) }));
  return <div className="kg-canvas">
    <svg ref={svgRef} viewBox={`0 0 ${viewport.width} ${viewport.height}`} role="group" aria-label="知识图谱画布"
      onPointerDown={event => {
        if (event.button !== 0) return;
        drag.current = { x: event.clientX, y: event.clientY, originX: camera.x, originY: camera.y };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={event => {
        if (!drag.current) return;
        const matrix = event.currentTarget.getScreenCTM(), start = drag.current;
        if (matrix) setCamera(current => ({ ...current, x: start.originX + (event.clientX - start.x) / matrix.a, y: start.originY + (event.clientY - start.y) / matrix.d }));
      }}
      onPointerUp={event => { drag.current = null; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
      onPointerCancel={() => { drag.current = null; }}>
      <defs><marker id={marker} markerWidth="8" markerHeight="8" refX="15" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L8 4 L0 8" fill="#7fa8b1"/></marker></defs>
      <g transform={`translate(${camera.x + viewport.width / 2} ${camera.y + viewport.height / 2}) scale(${camera.zoom}) translate(${-viewport.width / 2} ${-viewport.height / 2}) translate(${offsetX} ${offsetY}) scale(${fit})`}>
        {layout.links.map(link => {
          const active = selection?.kind === "edge" && tripleKey(selection.triple) === link.id;
          const connected = !focusName || link.from.id === focusName || link.to.id === focusName;
          return <g className={"kg-edge" + (active ? " selected" : "") + (!connected ? " faded" : "")} key={link.id}
            role="button" tabIndex={0} aria-pressed={active} aria-label={[link.from.label, link.triple.relation, link.to.label].join("，")}
            onPointerDown={event => event.stopPropagation()}
            onClick={() => onSelect(active ? null : { kind: "edge", triple: link.triple })}
            onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect({ kind: "edge", triple: link.triple }); } }}>
            <title>{`${link.from.label} → ${String(link.triple.relation)} → ${link.to.label}`}</title>
            <path className="kg-edge-hit" d={edgePath(link.from, link.to)}/>
            <path className="kg-edge-line" d={edgePath(link.from, link.to)} markerEnd={`url(#${marker})`}/>
            <text x={(link.from.x + link.to.x) / 2} y={(link.from.y + link.to.y) / 2 - 10}>{String(link.triple.relation)}</text>
          </g>;
        })}
        {layout.nodes.map(node => {
          const active = selection?.kind === "node" && selection.id === node.id;
          const highlighted = active || (selection?.kind === "edge" && [graphEndpoint(selection.triple, "head").id, graphEndpoint(selection.triple, "tail").id].includes(node.id));
          return <g key={node.id} className={"kg-node" + (highlighted ? " selected" : "") + (neighbors && !neighbors.has(node.id) ? " faded" : "")}
            transform={`translate(${node.x} ${node.y})`} role="button" tabIndex={0} aria-pressed={active} aria-label={"实体：" + node.label}
            onPointerDown={event => event.stopPropagation()} onMouseEnter={() => setHover(node.id)} onMouseLeave={() => setHover(null)}
            onClick={() => onSelect(active ? null : { kind: "node", name: node.label, id: node.id })}
            onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect({ kind: "node", name: node.label, id: node.id }); } }}>
            <title>{`${node.label} · ${node.degree} 个相邻实体`}</title>
            <circle className="kg-node-hit" r={20 / Math.max(.65, fit)}/><circle r={Math.min(12, 5 + Math.sqrt(node.degree) * 2) / Math.max(.65, fit)}/>
            {(labels || highlighted || hover === node.id) && <text y={27 / Math.max(.65, fit)} textAnchor="middle" style={{ fontSize: 14 / Math.max(.65, fit) }}>{node.label.length > 18 ? node.label.slice(0, 17) + "…" : node.label}</text>}
          </g>;
        })}
      </g>
    </svg>
    <div className="kg-canvas-options"><label><input type="checkbox" checked={labels} onChange={event => setLabels(event.target.checked)}/>实体名称</label></div>
    <div className="kg-camera-controls" aria-label="图谱画布操作">
      <button aria-label="缩小图谱" onClick={() => zoomBy(1 / 1.2)} disabled={camera.zoom <= .45}>−</button>
      <output aria-label="图谱缩放比例">{Math.round(camera.zoom * 100)}%</output>
      <button aria-label="放大图谱" onClick={() => zoomBy(1.2)} disabled={camera.zoom >= 4}>＋</button>
      <button onClick={() => setCamera({ zoom: 1, x: 0, y: 0 })}>适配画布</button>
    </div>
  </div>;
}
