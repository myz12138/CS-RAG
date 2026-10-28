import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import ts from "typescript";

const compile = source => ts.transpileModule(source, { compilerOptions: {
  target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ES2020,
} }).outputText;
const url = source => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const messagesUrl = url(compile(readFileSync(new URL("../src/utils/messages.ts", import.meta.url), "utf8")));
const { errorMessage, statusLabel } = await import(messagesUrl);
const transportUrl = url(compile(readFileSync(new URL("../src/api/transport.ts", import.meta.url), "utf8"))
  .replaceAll('"../utils/messages"', JSON.stringify(messagesUrl)));
const { fetchData } = await import(transportUrl);

test("network and browser errors are Chinese; technical terms are unchanged", () => {
  for (const message of ["Failed to fetch", "Load failed", "NetworkError when attempting to fetch resource."])
    assert.match(errorMessage(new TypeError(message)), /无法连接服务/);
  assert.match(errorMessage(new Error("Request timed out")), /请求超时/);
  assert.match(errorMessage(new DOMException("aborted", "AbortError")), /取消或中断/);
  assert.equal(errorMessage("Embedding 模型 BAAI/bge-small-zh-v1.5 未就绪"), "Embedding 模型 BAAI/bge-small-zh-v1.5 未就绪");
  assert.equal(statusLabel("pending"), "等待中");
});

test("JSON and upload requests share localized transport failures", async t => {
  t.mock.method(globalThis, "fetch", async () => { throw new TypeError("Failed to fetch"); });
  await assert.rejects(fetchData("/api/test"), /无法连接服务/);
  await assert.rejects(fetchData("/api/upload", { method: "POST", body: new FormData() }), /无法连接服务/);
});

test("validation arrays, HTTP errors and malformed responses are readable", async t => {
  let response = new Response(JSON.stringify({ detail: [{ loc: ["body", "question"], type: "string_too_short", msg: "String should have at least 1 character", ctx: { min_length: 1 } }] }), { status: 422 });
  t.mock.method(globalThis, "fetch", async () => response);
  await assert.rejects(fetchData("/api/test"), /问题：文本长度至少为 1 个字符/);
  response = new Response("<html>Bad gateway</html>", { status: 502 });
  await assert.rejects(fetchData("/api/test"), /上游服务响应异常/);
  response = new Response("<html>Wrong address</html>", { status: 200 });
  await assert.rejects(fetchData("/api/test"), /数据格式异常/);
  response = new Response(JSON.stringify({ wrong: true }));
  await assert.rejects(fetchData("/api/test"), /缺少 data 字段/);
});

test("success payloads, including English evidence and algorithm symbols, are untouched", async t => {
  const data = { answer: "Alice", relation: "part of", variable: "?x", model: "BAAI/bge-small-zh-v1.5", neff: 1.25 };
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify({ data })));
  assert.deepEqual(await fetchData("/api/test"), data);
});

test("visible brand and page title no longer contain Studio", () => {
  const shell = readFileSync(new URL("../src/components/WorkspaceShell.tsx", import.meta.url), "utf8");
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  assert.doesNotMatch(shell, />Studio</);
  assert.match(html, /<title>CS-RAG<\/title>/);
});
