import { useEffect, useState } from "react";
import type { HealthData, CapabilitiesData, ModelConfig, ModelConfigUpdate, BuildConfig, BuildConfigUpdate, QueryConfig, QueryConfigUpdate } from "../types";
import { StatusLine } from "../components/Common";
export default function SettingsView({ health, capabilities, modelConfig, buildConfig, queryConfig, checking, checkMessage, saving, saveMessage, buildSaving, buildSaveMessage, querySaving, querySaveMessage, onModelCheck, onModelSave, onBuildSave, onQuerySave }: { health: HealthData | null; capabilities: CapabilitiesData | null; modelConfig: ModelConfig | null; buildConfig: BuildConfig | null; queryConfig: QueryConfig | null; checking: boolean; checkMessage: string; saving: boolean; saveMessage: string; buildSaving: boolean; buildSaveMessage: string; querySaving: boolean; querySaveMessage: string; onModelCheck: () => void; onModelSave: (update: ModelConfigUpdate) => void; onBuildSave: (update: BuildConfigUpdate) => void; onQuerySave: (update: QueryConfigUpdate) => void }) {
  const [tab, setTab] = useState("models");
  if (!capabilities) {
    return <div className="module-placeholder panel"><div className="empty-icon">⚙</div><h2>正在读取服务能力</h2><p>连接后显示本地或远程部署信息。</p></div>;
  }
  const runtime = capabilities.algorithm.runtime;
  return (
    <div className="settings-layout">
      <section className="panel settings-hero"><div className="settings-orb">◎</div><div><h2>{health?.deployment_mode === "remote" ? "远程 CS-RAG 服务" : "本地 CS-RAG 服务"}</h2><p>前端通过统一 API 连接，算法和模型只由后端执行。</p></div><span className={"health-pill " + (health?.status === "ok" ? "connected" : "disconnected")}>{health?.status === "ok" ? "在线" : "离线"}</span></section>
      <section className="settings-grid">
        <div className="section-tabs settings-tabs" role="tablist" aria-label="配置分类">{[{ id: "models", title: "模型与连接" }, { id: "build", title: "构建与索引" }, { id: "query", title: "检索与问答" }, { id: "runtime", title: "服务与能力" }].map(item => <button role="tab" aria-selected={tab === item.id} className={tab === item.id ? "active" : ""} onClick={() => setTab(item.id)} key={item.id}>{item.title}</button>)}</div>
        <div hidden={tab !== "models"}>
        <ModelConfigForm config={modelConfig} capabilities={capabilities} checking={checking} checkMessage={checkMessage} saving={saving} saveMessage={saveMessage} onModelCheck={onModelCheck} onModelSave={onModelSave} />
        </div><div hidden={tab !== "build"}>
        <BuildConfigForm config={buildConfig} saving={buildSaving} saveMessage={buildSaveMessage} onSave={onBuildSave} />
        </div><div hidden={tab !== "query"}>
        <QueryConfigForm config={queryConfig} saving={querySaving} saveMessage={querySaveMessage} onSave={onQuerySave} />
        </div><div hidden={tab !== "runtime"}>
        <div className="panel settings-card"><div className="panel-header compact-header"><div><h2>算法基线</h2><p>固定提交与运行依赖</p></div><span className={"health-pill " + (runtime.available ? "connected" : "disconnected")}>{runtime.available ? "可运行" : "待安装"}</span></div><div className="setting-value"><span>参考提交</span><code>{capabilities.algorithm.reference_commit.slice(0, 12)}</code></div><div className="setting-value"><span>适配状态</span><code>{capabilities.algorithm.adapter_status === "subprocess_runner_ready_dependency_gated" ? "已接入，运行前检查依赖" : "适配状态待确认"}</code></div>{runtime.missing.length > 0 && <div className="missing-deps">缺少：{runtime.missing.join("、")}</div>}</div>
        </div>
      </section>
      <section className="panel settings-card" hidden={tab !== "runtime"}><div className="panel-header compact-header"><div><h2>输入边界</h2><p>新版本只从文本开始，知识图谱由系统自动抽取</p></div></div><div className="format-groups"><div><span>可上传文本</span><strong>{capabilities.input_formats.text.join(" · ")}</strong></div><div><span>知识图谱</span><strong>{capabilities.input_formats.knowledge_graph.length > 0 ? capabilities.input_formats.knowledge_graph.join(" · ") : "由文本自动生成"}</strong></div></div></section>
    </div>
  );
}

function ModelConfigForm({ config, capabilities, checking, checkMessage, saving, saveMessage, onModelCheck, onModelSave }: { config: ModelConfig | null; capabilities: CapabilitiesData; checking: boolean; checkMessage: string; saving: boolean; saveMessage: string; onModelCheck: () => void; onModelSave: (update: ModelConfigUpdate) => void }) {
  const [form, setForm] = useState<ModelConfigUpdate>({
    base_url: "",
    model_name: "gpt-4o-mini",
    timeout_seconds: 90,
    embedding_base_url: "",
    embedding_model: "BAAI/bge-small-zh-v1.5",
    reranker_model: "BAAI/bge-reranker-v2-m3",
    api_key: "",
    embedding_api_key: "",
  });

  useEffect(() => {
    if (config) {
      setForm({
        base_url: config.base_url,
        model_name: config.model_name,
        timeout_seconds: config.timeout_seconds,
        embedding_base_url: config.embedding_base_url,
        embedding_model: config.embedding_model,
        reranker_model: config.reranker_model,
        api_key: "",
        embedding_api_key: "",
      });
    }
  }, [config]);

  const setField = (field: keyof ModelConfigUpdate, value: string | number) => setForm((current) => ({ ...current, [field]: value }));
  return <div className="panel settings-card model-config-card">
    <div className="panel-header compact-header"><div><h2>模型网关</h2><p>模型由前端配置，后端立即生效</p></div></div>
    <StatusLine label="LLM" value={capabilities.model_gateway.llm ? "已配置" : "未配置"} active={capabilities.model_gateway.llm} />
    <StatusLine label="Embedding" value={capabilities.model_gateway.embedding ? (capabilities.model_gateway.embedding_remote ? "远程" : "本地") : "未配置"} active={capabilities.model_gateway.embedding} />
    <StatusLine label="Reranker" value={capabilities.model_gateway.reranker ? "已配置" : "未配置"} active={capabilities.model_gateway.reranker} />
    <div className="model-form-grid">
      <label>LLM 服务地址<input className="text-input" value={form.base_url} onChange={(event) => setField("base_url", event.target.value)} placeholder="例如：https://api.example.com/v1" /></label>
      <label>LLM 模型（KG 抽取 / 查询规划 / 答案）<input className="text-input" value={form.model_name} onChange={(event) => setField("model_name", event.target.value)} /></label>
      <label>请求超时（秒）<input className="text-input" type="number" min="1" max="3600" step="1" value={form.timeout_seconds} onChange={(event) => setField("timeout_seconds", Number(event.target.value) || 1)} /></label>
      <label>LLM Key<input className="text-input" type="password" value={form.api_key} onChange={(event) => setField("api_key", event.target.value)} placeholder={config?.api_key_configured ? "已配置，留空保持不变" : "请输入服务 Key"} /></label>
      <label>查询 Embedding 地址（可选）<input className="text-input" value={form.embedding_base_url} onChange={(event) => setField("embedding_base_url", event.target.value)} placeholder="留空使用本地模型" /></label>
      <label>Embedding Key（可选）<input className="text-input" type="password" value={form.embedding_api_key} onChange={(event) => setField("embedding_api_key", event.target.value)} placeholder={config?.embedding_api_key_configured ? "已配置，留空保持不变" : "远程 Embedding 才需要"} /></label>
      <label>Embedding 模型（本地切分 / 索引）<input className="text-input" value={form.embedding_model} onChange={(event) => setField("embedding_model", event.target.value)} /></label>
      <label>Reranker 模型（查询阶段）<input className="text-input" value={form.reranker_model} onChange={(event) => setField("reranker_model", event.target.value)} /></label>
    </div>
    <div className="model-config-note">文本语义切分与文本索引固定使用本地 Embedding；填写远程地址只影响查询阶段的原文召回。</div>
    <div className="model-form-actions"><button className="primary-button compact" onClick={() => onModelSave({ ...form, api_key: form.api_key?.trim() || undefined, embedding_api_key: form.embedding_api_key?.trim() || undefined })} disabled={saving}>{saving ? "保存中…" : "保存模型配置"}</button><button className="secondary-button" onClick={onModelCheck} disabled={checking}>{checking ? "检查中…" : "测试 LLM 连接"}</button>{saveMessage && <span>{saveMessage}</span>}{checkMessage && <span>{checkMessage}</span>}</div>
  </div>;
}

function BuildConfigForm({ config, saving, saveMessage, onSave }: { config: BuildConfig | null; saving: boolean; saveMessage: string; onSave: (update: BuildConfigUpdate) => void }) {
  const [form, setForm] = useState<BuildConfigUpdate>({
    max_tokens: 512,
    min_tokens: 80,
    parent_max_tokens: 1536,
    include_heading: true,
    semantic_breakpoint_percentile: 85,
    tokenizer_model: "",
    embedding_device: "cpu",
    text_index_batch_size: 32,
  });

  useEffect(() => {
    if (!config) return;
    setForm({
      max_tokens: config.max_tokens,
      min_tokens: config.min_tokens,
      parent_max_tokens: config.parent_max_tokens,
      include_heading: config.include_heading,
      semantic_breakpoint_percentile: config.semantic_breakpoint_percentile,
      tokenizer_model: config.tokenizer_model || "",
      embedding_device: config.embedding_device || "cpu",
      text_index_batch_size: config.text_index_batch_size,
    });
  }, [config]);

  const setNumber = (field: "max_tokens" | "min_tokens" | "parent_max_tokens" | "semantic_breakpoint_percentile" | "text_index_batch_size", value: string) => {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      setForm((current) => ({ ...current, [field]: parsed }));
    }
  };
  const standardDevices = ["cpu", "cuda", "mps"];
  const deviceOptions = standardDevices.includes(form.embedding_device) ? standardDevices : [form.embedding_device, ...standardDevices];

  return <div className="panel settings-card model-config-card build-config-card">
    <div className="panel-header compact-header"><div><h2>构建与索引</h2><p>只影响后续新建版本，已发布版本保持不变</p></div><span className="status-chip">联合切分</span></div>
    <div className="setting-value"><span>当前 Embedding</span><code>{config?.embedding_model || "由模型网关配置"}</code></div>
    <div className="model-form-grid build-form-grid">
      <label>Chunk 最大 Token<input className="text-input" type="number" min="1" max="65536" value={form.max_tokens} onChange={(event) => setNumber("max_tokens", event.target.value)} /></label>
      <label>语义分段最小 Token<input className="text-input" type="number" min="0" max="65536" value={form.min_tokens} onChange={(event) => setNumber("min_tokens", event.target.value)} /></label>
      <label>父级上下文最大 Token<input className="text-input" type="number" min="1" max="131072" value={form.parent_max_tokens} onChange={(event) => setNumber("parent_max_tokens", event.target.value)} /></label>
      <label>语义断点敏感度（百分位）<input className="text-input" type="number" min="0" max="100" step="0.5" value={form.semantic_breakpoint_percentile} onChange={(event) => setNumber("semantic_breakpoint_percentile", event.target.value)} /></label>
      <label>索引批量写入数量<input className="text-input" type="number" min="1" max="4096" value={form.text_index_batch_size} onChange={(event) => setNumber("text_index_batch_size", event.target.value)} /></label>
      <label>Embedding 运行设备<select className="text-input" value={form.embedding_device} onChange={(event) => setForm((current) => ({ ...current, embedding_device: event.target.value }))}>{deviceOptions.map((device) => <option key={device} value={device}>{device === "cpu" ? "CPU" : device === "cuda" ? "CUDA" : device === "mps" ? "Apple MPS" : device}</option>)}</select></label>
      <label>精确 Tokenizer（可选）<input className="text-input" value={form.tokenizer_model} onChange={(event) => setForm((current) => ({ ...current, tokenizer_model: event.target.value }))} placeholder="留空使用近似 Token 计数" /></label>
      <label className="checkbox-field"><input type="checkbox" checked={form.include_heading} onChange={(event) => setForm((current) => ({ ...current, include_heading: event.target.checked }))} /><span>将章节标题带入 Chunk 上下文</span></label>
    </div>
    <div className="model-config-note">固定规则：无重叠、超限按完整句子截断、父块不做向量化；本地模型仅从已下载权重加载。</div>
    <div className="model-form-actions"><button className="primary-button compact" onClick={() => onSave({ ...form, tokenizer_model: form.tokenizer_model.trim() })} disabled={saving}>{saving ? "保存中…" : "保存构建参数"}</button>{saveMessage && <span>{saveMessage}</span>}</div>
  </div>;
}

function QueryConfigForm({ config, saving, saveMessage, onSave }: { config: QueryConfig | null; saving: boolean; saveMessage: string; onSave: (update: QueryConfigUpdate) => void }) {
  const [form, setForm] = useState<QueryConfigUpdate>({
    planner_max_retries: 3,
    top_n_llm: 3,
    top_k_entity: 3,
    relation_k: 8,
    neff_threshold: 1.5,
    phase1_rerank_max_length: 128,
    phase2_rerank_max_length: 256,
    rerank_batch: 32,
    rerank_fp16: false,
    include_question_in_rerank: true,
    unresolved_top_similar: 20,
    top_k_unresolved: 3,
    focus_max_per_variable: 5,
    text_index_recall_multiplier: 4,
    prompt_max_triples: 20,
    prompt_max_text_per_triple: 8,
  });

  useEffect(() => {
    if (config) setForm({ ...config });
  }, [config]);

  const setNumber = (field: "planner_max_retries" | "top_n_llm" | "top_k_entity" | "relation_k" | "neff_threshold" | "phase1_rerank_max_length" | "phase2_rerank_max_length" | "rerank_batch" | "unresolved_top_similar" | "top_k_unresolved" | "focus_max_per_variable" | "text_index_recall_multiplier" | "prompt_max_triples" | "prompt_max_text_per_triple", value: string) => {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) setForm((current) => ({ ...current, [field]: parsed }));
  };

  return <div className="panel settings-card model-config-card query-config-card">
    <div className="panel-header compact-header"><div><h2>检索与问答参数</h2><p>对应 CS-RAG 的查询规划、Phase1、Phase2 和答案生成；保存后影响后续查询</p></div><span className="status-chip">参考算法参数</span></div>
    <div className="query-config-group"><span className="eyebrow">查询规划</span><div className="model-form-grid">
      <label>规划失败最大重试<input className="text-input" type="number" min="1" max="10" value={form.planner_max_retries} onChange={(event) => setNumber("planner_max_retries", event.target.value)} /></label>
    </div></div>
    <div className="query-config-group"><span className="eyebrow">第一阶段 · 图检索与重排</span><div className="model-form-grid">
      <label>每条约束保留三元组数<input className="text-input" type="number" min="1" max="100" value={form.top_n_llm} onChange={(event) => setNumber("top_n_llm", event.target.value)} /></label>
      <label>已知实体候选数<input className="text-input" type="number" min="1" max="100" value={form.top_k_entity} onChange={(event) => setNumber("top_k_entity", event.target.value)} /></label>
      <label>关系粗排候选数<input className="text-input" type="number" min="1" max="1000" value={form.relation_k} onChange={(event) => setNumber("relation_k", event.target.value)} /></label>
      <label>Neff 充分性阈值<input className="text-input" type="number" min="0.1" max="1000" step="0.1" value={form.neff_threshold} onChange={(event) => setNumber("neff_threshold", event.target.value)} /></label>
      <label>Phase1 重排最大长度<input className="text-input" type="number" min="32" max="4096" value={form.phase1_rerank_max_length} onChange={(event) => setNumber("phase1_rerank_max_length", event.target.value)} /></label>
      <label>重排批量大小<input className="text-input" type="number" min="1" max="4096" value={form.rerank_batch} onChange={(event) => setNumber("rerank_batch", event.target.value)} /></label>
      <label className="checkbox-field"><input type="checkbox" checked={form.include_question_in_rerank} onChange={(event) => setForm((current) => ({ ...current, include_question_in_rerank: event.target.checked }))} /><span>重排输入包含原始问题</span></label>
      <label className="checkbox-field"><input type="checkbox" checked={form.rerank_fp16} onChange={(event) => setForm((current) => ({ ...current, rerank_fp16: event.target.checked }))} /><span>CUDA 下启用重排半精度</span></label>
    </div></div>
    <div className="query-config-group"><span className="eyebrow">第二阶段 · 原文恢复与答案</span><div className="model-form-grid">
      <label>Phase2 重排最大长度<input className="text-input" type="number" min="32" max="4096" value={form.phase2_rerank_max_length} onChange={(event) => setNumber("phase2_rerank_max_length", event.target.value)} /></label>
      <label>未解析文本粗召回数<input className="text-input" type="number" min="1" max="1000" value={form.unresolved_top_similar} onChange={(event) => setNumber("unresolved_top_similar", event.target.value)} /></label>
      <label>每条未解析约束保留数<input className="text-input" type="number" min="1" max="100" value={form.top_k_unresolved} onChange={(event) => setNumber("top_k_unresolved", event.target.value)} /></label>
      <label>每个变量聚焦实体数<input className="text-input" type="number" min="1" max="100" value={form.focus_max_per_variable} onChange={(event) => setNumber("focus_max_per_variable", event.target.value)} /></label>
      <label>文本索引召回倍数<input className="text-input" type="number" min="1" max="100" value={form.text_index_recall_multiplier} onChange={(event) => setNumber("text_index_recall_multiplier", event.target.value)} /></label>
      <label>答案最多纳入约束数<input className="text-input" type="number" min="1" max="1000" value={form.prompt_max_triples} onChange={(event) => setNumber("prompt_max_triples", event.target.value)} /></label>
      <label>每条约束最多纳入证据数<input className="text-input" type="number" min="1" max="100" value={form.prompt_max_text_per_triple} onChange={(event) => setNumber("prompt_max_text_per_triple", event.target.value)} /></label>
    </div></div>
    <div className="model-config-note">Neff 用于变量候选的充分性门控。模型在“模型与连接”中配置；保存参数后用于后续查询，历史运行保留各自的参数快照。</div>
    <div className="model-form-actions"><button className="primary-button compact" onClick={() => onSave({ ...form })} disabled={saving}>{saving ? "保存中…" : "保存检索参数"}</button>{saveMessage && <span>{saveMessage}</span>}</div>
  </div>;
}


