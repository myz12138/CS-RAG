import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

// Presentation fixtures only; no model calls or generated QA answers.
const compile = (source) => ts.transpileModule(source, { compilerOptions: {
  target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ES2020, jsx: ts.JsxEmit.ReactJSX,
} }).outputText;
const url = (source) => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const formatUrl = url(compile(readFileSync(new URL("../src/utils/format.ts", import.meta.url), "utf8")));
const messagesUrl = url(compile(readFileSync(new URL("../src/utils/messages.ts", import.meta.url), "utf8")));
const { constraintCheckSummary, evidenceSummary } = await import(formatUrl);
const require = createRequire(import.meta.url);
const source = compile(readFileSync(new URL("../src/pages/QueryView.tsx", import.meta.url), "utf8"))
  .replaceAll('"react/jsx-runtime"', JSON.stringify(pathToFileURL(require.resolve("react/jsx-runtime")).href))
  .replaceAll('"react"', JSON.stringify(pathToFileURL(require.resolve("react")).href))
  .replaceAll('"../utils/format"', JSON.stringify(formatUrl))
  .replaceAll('"../utils/messages"', JSON.stringify(messagesUrl));
const { default: QueryView } = await import(url(source));

test("Neff and constraint status are independent and missing checks are not invented", () => {
  assert.match(constraintCheckSummary({ checked: true, neff: 1.1, status: "unresolved" }), /Neff 1.1.*未解决/);
  assert.match(constraintCheckSummary({ checked: false, status: "unresolved" }), /未执行/);
  assert.match(constraintCheckSummary({ checked: true, neff: null }), /无有限值/);
  assert.match(constraintCheckSummary({}), /未记录/);
});

test("KG evidence and text fallback are distinguished even for empty output", () => {
  assert.match(evidenceSummary({ triple_status: "resolved", candidate_evidences: [] }), /KG evidence.*没有可用证据/);
  assert.match(evidenceSummary({ triple_status: "unresolved", candidate_evidences: [{}] }), /文本初筛与重排.*1 条/);
});

test("open constraint inspector renders current algorithm snapshot, not stale fetched detail", () => {
  const record = { query_id: "q", status: "running", current_step: "phase2_text_recovery", version_id: "v",
    warnings: [], query_plan: [{ head: "甲", relation: "关系", tail: "?x" }],
    evidence: [{ query_triple_index: 0, triple_status: "resolved", candidate_evidences: [] }],
    metadata: { phase1_debug: { triples_debug: [{ query_triple_index: 0, checked: true, neff: 1.2, status: "resolved" }] } },
  };
  const html = renderToStaticMarkup(createElement(QueryView, {
    projects: [{ project_id: "p", name: "展示测试", status: "ready" }], selectedProjectId: "p",
    versions: [{ version_id: "v", status: "ready", updated_at: "2026-10-28" }], selectedVersionId: "v",
    question: "", running: true, record, events: [], taskProgress: 0.74, taskMessage: "",
    selectedConstraintId: "T0", constraintDetail: null, constraintLoading: false, evidenceDetail: null,
    queryConfig: null, traceLevel: "detailed",
  }));
  assert.match(html, /Neff 1.2/);
  assert.match(html, /最终状态：已解决/);
  assert.match(html, /KG evidence 分支/);
  assert.doesNotMatch(html, /未保存 Neff|query-stage-substeps/);
});
