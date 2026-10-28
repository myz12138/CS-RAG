import { useEffect, useMemo, useRef, useState } from "react";
import { getVersionContents } from "../api/client";
import type { Source, VersionContents } from "../types";
import { graphEndpoint, filterGraph, relatedTriples, tripleKey, type GraphSelection, type GraphTriple } from "../utils/knowledgeGraph";
import { Icon } from "./Icon";
import { KnowledgeGraphCanvas } from "./KnowledgeGraphCanvas";

export function KnowledgeGraphExplorer({ projectId, versionId, contents, sources, loading, onOpenLibrary }: {
  projectId: string; versionId: string; contents: VersionContents | null; sources: Source[];
  loading: boolean; onOpenLibrary: (kind: "text" | "kg") => void;
}) {
  const root = useRef<HTMLElement | null>(null);
  const detailRef = useRef<HTMLDivElement | null>(null);
  const generation = useRef(0);
  const [search, setSearch] = useState("");
  const [source, setSource] = useState("");
  const [relation, setRelation] = useState("");
  const [evidenceOnly, setEvidenceOnly] = useState(false);
  const [neighbor, setNeighbor] = useState<{ id: string; label: string } | null>(null);
  const [selection, setSelection] = useState<GraphSelection | null>(null);
  const [scope, setScope] = useState("preview");
  const [loaded, setLoaded] = useState<GraphTriple[]>([]);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState("");
  const [full, setFull] = useState(false);
  const [detailLimit, setDetailLimit] = useState(6);
  const initial = contents?.items ?? [];
  const total = contents?.version_total ?? contents?.total ?? 0;
  const triples = scope === "preview" || loaded.length === 0 ? initial : loaded;
  const visible = useMemo(() => filterGraph(triples, { search, source, relation, evidenceOnly, neighbor: neighbor?.id }), [triples, search, source, relation, evidenceOnly, neighbor]);
  const entities = useMemo(() => new Map(visible.flatMap(triple => {
    const head = graphEndpoint(triple, "head"), tail = graphEndpoint(triple, "tail");
    return [[head.id, head.label], [tail.id, tail.label]] as Array<[string, string]>;
  })), [visible]);
  const relations = useMemo(() => [...new Set(triples.map(triple => String(triple.relation)))].sort(), [triples]);
  const sourceNames = new Map(sources.map(item => [item.source_id, item.original_name]));
  const selectedRelations = selection?.kind === "node" ? relatedTriples(triples, selection.id) : [];
  useEffect(() => () => { generation.current++; }, []);
  useEffect(() => {
    const update = () => setFull(document.fullscreenElement === root.current);
    document.addEventListener("fullscreenchange", update);
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);
  useEffect(() => {
    setSelection(null); setDetailLimit(6);
  }, [search, source, relation, evidenceOnly, neighbor, scope]);
  useEffect(() => {
    setDetailLimit(6);
    if (selection && !full) detailRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selection, full]);

  async function loadMore() {
    if (fetching || !versionId || !contents) return;
    const token = ++generation.current;
    setFetching(true); setError("");
    const accumulated = [...(loaded.length ? loaded : initial)];
    const target = Math.min(total, accumulated.length + 1000);
    try {
      while (accumulated.length < target) {
        const response = await getVersionContents(projectId, versionId, "kg", { offset: accumulated.length, limit: Math.min(100, target - accumulated.length) });
        if (token !== generation.current) return;
        if (response.items.length === 0) break;
        accumulated.push(...response.items);
        setLoaded([...accumulated]);
      }
    } catch (cause) {
      if (token === generation.current) setError(cause instanceof Error ? cause.message : "图谱加载失败");
    } finally {
      if (token === generation.current) setFetching(false);
    }
  }
  function changeScope(value: string) {
    setScope(value);
    if (value === "loaded" && loaded.length < total) void loadMore();
  }
  function reset() {
    setSearch(""); setSource(""); setRelation(""); setEvidenceOnly(false); setNeighbor(null); setSelection(null);
  }
  async function fullscreen() {
    try {
      if (document.fullscreenElement === root.current) await document.exitFullscreen();
      else await root.current?.requestFullscreen();
    } catch { setError("当前浏览器不支持全屏，可最大化窗口查看。"); }
  }
  const sourceLabel = (triple: GraphTriple) => sourceNames.get(String(triple.source_id)) ?? "当前版本来源";
  return <section ref={root} className="kg-explorer" aria-label="知识图谱探索">
    <header className="kg-section-heading">
      <div className="kg-heading-title"><Icon name="graph" size={22}/><h2>知识图谱</h2></div>
      <button className="secondary-button" onClick={() => onOpenLibrary("kg")} disabled={!versionId}>浏览图谱条目<Icon name="arrow" size={16}/></button>
    </header>
    <div className="kg-tools">
      <div className="kg-search-row">
        <label className="kg-search"><Icon name="search" size={18}/><input aria-label="搜索已加载图谱" placeholder="搜索实体或关系" value={search} onChange={event => setSearch(event.target.value)}/>{search && <button aria-label="清除图谱搜索" onClick={() => setSearch("")}><Icon name="close" size={16}/></button>}</label>
        <label className="kg-scope-select"><span>浏览范围</span><select aria-label="图谱浏览范围" value={scope} onChange={event => changeScope(event.target.value)} disabled={fetching || !versionId}><option value="preview">概览</option><option value="loaded">完整图谱</option></select></label>
        <div className="kg-view-actions"><button className="secondary-button" onClick={reset}>清除筛选</button><button className="secondary-button" onClick={() => void fullscreen()}>{full ? "退出全屏" : "全屏"}</button></div>
      </div>
      <div className="kg-filter-row">
        <label><span>来源</span><select aria-label="筛选图谱来源" value={source} onChange={event => setSource(event.target.value)}><option value="">全部来源</option>{sources.map(item => <option key={item.source_id} value={item.source_id}>{item.original_name}</option>)}</select></label>
        <label><span>关系</span><select aria-label="筛选图谱关系" value={relation} onChange={event => setRelation(event.target.value)}><option value="">全部关系</option>{relations.map(name => <option value={name} key={name}>{name}</option>)}</select></label>
        <label className="kg-evidence-filter"><input type="checkbox" checked={evidenceOnly} onChange={event => setEvidenceOnly(event.target.checked)}/>仅看有原文依据</label>
      </div>
    </div>
    <div className="kg-scope-bar" role="status">
      <span>显示 <strong>{entities.size}</strong> 个实体 · <strong>{visible.length}</strong> 条关系</span>
      <span>{fetching ? "正在加载" : "已加载"} <strong>{triples.length}</strong> / {total} 条关系</span>
      {scope === "loaded" && triples.length < total && !fetching && <button className="link-button" onClick={() => void loadMore()}>继续加载</button>}
      {neighbor && <button className="kg-focus-chip" onClick={() => setNeighbor(null)}>邻域：{neighbor.label}<Icon name="close" size={14}/></button>}
    </div>
    {error && <div className="kg-error" role="alert">{error}</div>}
    <div className="kg-graph-body">
      {!!visible.length && <label className="kg-entity-picker"><select aria-label="选择图谱实体" value={selection?.kind === "node" ? selection.id : ""} onChange={event => {
        const id = event.target.value;
        setSelection(id ? { kind: "node", id, name: entities.get(id) ?? id } : null);
      }}><option value="">选择实体查看关联</option>{[...entities].map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>}
      {loading ? <div className="kg-empty" role="status"><span className="loading-orbit"/><strong>正在加载知识图谱</strong></div>
        : visible.length ? <KnowledgeGraphCanvas triples={visible} selection={selection} onSelect={setSelection}/>
        : <div className="kg-empty"><Icon name="graph" size={42}/><h3>{total ? "没有符合筛选的关系" : "当前版本尚无图谱"}</h3>{total ? <button className="secondary-button" onClick={reset}>清除筛选</button> : <button className="secondary-button" onClick={() => onOpenLibrary("text")}>管理资料</button>}</div>}
    </div>
    {selection && <div ref={detailRef} className="kg-inline-detail" aria-label="图谱选择详情">
      <div className="kg-detail-heading"><h3>{selection.kind === "node" ? selection.name : "关系与依据"}</h3><button className="icon-button" aria-label="收起图谱详情" onClick={() => setSelection(null)}><Icon name="close" size={18}/></button></div>
      {selection.kind === "node" ? <>
        <div className="kg-node-actions"><span>已加载图谱中有 {selectedRelations.length} 条关联</span><button className="secondary-button" onClick={() => { setSearch(""); setRelation(""); setSource(""); setEvidenceOnly(false); setNeighbor({ id: selection.id, label: selection.name }); }}>只看一跳邻域</button></div>
        <div className="kg-related-list">{selectedRelations.slice(0, detailLimit).map(triple => <button key={tripleKey(triple)} onClick={() => setSelection({ kind: "edge", triple })}><span>{String(triple.head)}</span><b>{String(triple.relation)}</b><span>{String(triple.tail)}</span><Icon name="arrow" size={16}/></button>)}</div>
        {selectedRelations.length > detailLimit && <button className="link-button" onClick={() => setDetailLimit(value => value + 12)}>更多关联关系</button>}
      </> : <>
        <div className="kg-selected-triple"><span>{String(selection.triple.head)}</span><b>{String(selection.triple.relation)}</b><span>{String(selection.triple.tail)}</span></div>
        <div className="kg-evidence-source">来源：{sourceLabel(selection.triple)}</div>
        <blockquote>{String(selection.triple.evidence || "该关系未附单独的原文依据。")}</blockquote>
        <details><summary>来源标识与完整字段</summary><pre>{JSON.stringify(selection.triple, null, 2)}</pre></details>
      </>}
    </div>}
  </section>;
}
