export type ConnectionStatus = "checking" | "connected" | "disconnected";

export interface HealthData {
  status: string;
  service: string;
  version: string;
  deployment_mode: "local" | "remote" | string;
  timestamp: string;
}

export interface CapabilitiesData {
  api_version: string;
  deployment_mode: string;
  input_formats: { text: string[]; knowledge_graph: string[] };
  build_config: BuildConfig;
  query_trace_levels: string[];
  query_config: QueryConfig;
  algorithm: {
    name: string;
    reference_commit: string;
    adapter_status: string;
    runtime: {
      available: boolean;
      missing: string[];
      dependencies: Record<string, boolean>;
      query_bridge: string;
    };
  };
  model_gateway: {
    llm: boolean;
    embedding: boolean;
    reranker: boolean;
    embedding_remote: boolean;
    configured_base_url: boolean;
    configured_api_key: boolean;
  };
}

export interface QueryConfig {
  planner_max_retries: number;
  top_n_llm: number;
  top_k_entity: number;
  relation_k: number;
  neff_threshold: number;
  phase1_rerank_max_length: number;
  phase2_rerank_max_length: number;
  rerank_batch: number;
  rerank_fp16: boolean;
  include_question_in_rerank: boolean;
  unresolved_top_similar: number;
  top_k_unresolved: number;
  focus_max_per_variable: number;
  text_index_recall_multiplier: number;
  prompt_max_triples: number;
  prompt_max_text_per_triple: number;
}

export type QueryConfigUpdate = QueryConfig;

export interface ModelCheckResult {
  available: boolean;
  model: string;
  response_preview: string;
  elapsed_ms: number;
}

export interface ModelConfig {
  base_url: string;
  model_name: string;
  timeout_seconds: number;
  embedding_base_url: string;
  embedding_model: string;
  reranker_model: string;
  api_key_configured: boolean;
  embedding_api_key_configured: boolean;
}

export interface ModelConfigUpdate {
  base_url: string;
  model_name: string;
  timeout_seconds: number;
  embedding_base_url: string;
  embedding_model: string;
  reranker_model: string;
  api_key?: string;
  embedding_api_key?: string;
}

export interface BuildConfig {
  strategy: "unified" | string;
  max_tokens: number;
  min_tokens: number;
  parent_max_tokens: number;
  include_heading: boolean;
  semantic_breakpoint_percentile: number;
  tokenizer_model: string;
  tokenizer_local_files_only: boolean;
  embedding_device: string;
  embedding_local_files_only: boolean;
  text_index_batch_size: number;
  overlap_tokens: 0;
  embedding_model?: string;
}

export interface BuildConfigUpdate {
  max_tokens: number;
  min_tokens: number;
  parent_max_tokens: number;
  include_heading: boolean;
  semantic_breakpoint_percentile: number;
  tokenizer_model: string;
  embedding_device: string;
  text_index_batch_size: number;
}

export interface Project {
  project_id: string;
  name: string;
  description: string;
  status: string;
  created_at: string;
  updated_at: string;
  current_version_id: string | null;
}

export interface VersionSummary {
  version_id: string;
  project_id: string;
  version_no: number;
  mode: string;
  status: string;
  text_status: string;
  kg_status: string;
  source_snapshot: string[];
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface VersionDetail {
  version: {
    version_id: string;
    project_id: string;
    version_no: number;
    mode: string;
    status: string;
    text_status: string;
    kg_status: string;
    source_snapshot: string[];
    error_code: string | null;
    error_message: string | null;
    created_at: string;
    updated_at: string;
  };
  counts: {
    text_entries: number;
    kg_triples: number;
    kg_entities: number;
    kg_conflicts: number;
  };
  conflicts: Array<Record<string, unknown>>;
  text_index?: {
    status?: string;
    vector_count?: number;
    parent_count?: number;
    dimension?: number;
    embedding_model?: string;
    storage_format?: string;
  };
}

export interface VersionContents {
  project_id: string;
  version_id: string;
  kind: "text" | "kg";
  total: number;
  version_total?: number;
  offset?: number;
  limit?: number;
  items: Array<Record<string, unknown>>;
}

export interface Source {
  source_id: string;
  project_id: string;
  original_name: string;
  stored_path: string;
  source_kind: "text" | "kg" | string;
  source_format: string;
  mime_type: string | null;
  checksum_sha256: string;
  size_bytes: number;
  parse_status: string;
  created_at: string;
  updated_at: string;
}

export interface BuildPreview {
  project_id: string;
  overall_status: string;
  mode: string;
  allow_graph_only: boolean;
  text_layer: { status: string; source_count: number; source_ids: string[] };
  kg_layer: { status: string; source_count: number; source_ids: string[] };
  expected_llm_calls: number | null;
  expected_llm_calls_basis: string;
  chunking: {
    strategy: string;
    retrieval_unit: string;
    token_limit: number;
    min_tokens: number;
    parent_token_limit: number;
    semantic_breakpoint_percentile: number;
    include_heading: boolean;
    overlap_tokens: number;
    hierarchy_rule: string;
    embedding_model: string | null;
    embedding_device: string;
    embedding_ready: boolean;
    embedding_local_only: boolean;
    tokenizer_model: string | null;
    tokenizer_local_only: boolean;
    index_batch_size: number;
    llm_call_unit: string;
  };
  source_snapshot: string[];
  blockers: Array<{ code: string; message: string }>;
  actions: string[];
}

export interface Task {
  task_id: string;
  task_type: string;
  status: string;
  progress: number;
  message: string;
  payload: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface QueryStart {
  query_id: string;
  task_id: string;
  version_id: string;
  status: string;
}

export interface QueryTraceItem {
  step: string;
  status: string;
  message: string;
  counts?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  output_ref?: string | null;
  input_ref?: string | null;
  reason_code?: string | null;
  warnings?: string[];
}

export interface QueryEvent {
  query_id: string;
  seq: number;
  step: string;
  status: string;
  message: string;
  constraint_id: string | null;
  variable_id: string | null;
  pass_index: number | null;
  branch_id: string | null;
  input_ref: string | null;
  output_ref: string | null;
  counts: Record<string, unknown>;
  metrics: Record<string, unknown>;
  reason_code: string | null;
  warnings: string[];
  started_at: string;
  finished_at: string | null;
  elapsed_ms: number | null;
}

export interface QueryRecord {
  query_id: string;
  project_id: string;
  version_id: string;
  question: string;
  trace_level: string;
  status: string;
  current_step: string | null;
  answer: string | null;
  warnings: string[];
  metadata: Record<string, unknown>;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  query_plan: Array<Record<string, unknown>>;
  evidence: Array<Record<string, unknown>>;
  trace: QueryTraceItem[];
  artifacts: Record<string, unknown>;
}

export interface QueryConstraintDetail {
  constraint_id: string;
  query_triple: Record<string, unknown> | null;
  phase1_debug: Array<Record<string, unknown>>;
  evidence: Array<Record<string, unknown>>;
}

export interface QueryEvidenceDetail {
  evidence_id: string;
  constraint: Record<string, unknown>;
  item: Record<string, unknown>;
}

export interface EvaluationSummary {
  project_id: string;
  version_id: string | null;
  sample_count: number;
  completed_count: number;
  failed_count: number;
  aggregate: Record<string, number>;
  metric_notes: Record<string, string>;
  recent: Array<{
    query_id: string;
    version_id: string;
    question: string;
    status: string;
    answer: string | null;
    metrics: Record<string, unknown>;
    error_code: string | null;
    updated_at: string;
  }>;
}

export interface ApiEnvelope<T> {
  data: T;
  meta: Record<string, unknown>;
  request_id: string | null;
}
