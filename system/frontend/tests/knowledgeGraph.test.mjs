import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import ts from "typescript";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

// Compile in memory: no smoke data, temporary files, dependency installation or LLM substitution.
const source = readFileSync(new URL("../src/utils/knowledgeGraph.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ES2020 } }).outputText;
const graphModule = "data:text/javascript;base64," + Buffer.from(compiled).toString("base64");
const { layoutGraph, filterGraph, relatedTriples, edgePath, tripleKey } = await import(graphModule);
const require = createRequire(import.meta.url);
const canvasSource = readFileSync(new URL("../src/components/KnowledgeGraphCanvas.tsx", import.meta.url), "utf8");
const canvasCompiled = ts.transpileModule(canvasSource, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ES2020, jsx: ts.JsxEmit.ReactJSX } }).outputText
  .replaceAll('"react/jsx-runtime"', JSON.stringify(pathToFileURL(require.resolve("react/jsx-runtime")).href))
  .replaceAll('"react"', JSON.stringify(pathToFileURL(require.resolve("react")).href))
  .replaceAll('"../utils/knowledgeGraph"', JSON.stringify(graphModule));
const { KnowledgeGraphCanvas } = await import("data:text/javascript;base64," + Buffer.from(canvasCompiled).toString("base64"));
const triples = [
  { triple_id: "1", head: "甲", relation: "关联", tail: "乙", source_id: "a", evidence: "实际依据" },
  { triple_id: "2", head: "甲", relation: "关联", tail: "丙", source_id: "b", evidence: "" },
  { triple_id: "3", head: "丁", relation: "属于", tail: "丁", source_id: "a", evidence: "自环" },
];

test("projection preserves every input edge and does not mutate knowledge data", () => {
  const before = JSON.stringify(triples);
  const layout = layoutGraph(triples);
  assert.equal(layout.nodes.length, 4);
  assert.deepEqual(layout.links.map(link => link.id), ["1", "2", "3"]);
  assert.equal(layout.links[0].triple, triples[0]);
  assert.equal(JSON.stringify(triples), before);
});
test("layout is deterministic with finite coordinates, including isolated components and self loops", () => {
  assert.deepEqual(layoutGraph(triples), layoutGraph(triples));
  for (const node of layoutGraph(triples).nodes) assert.ok(Number.isFinite(node.x) && Number.isFinite(node.y));
  assert.deepEqual(layoutGraph([]).nodes, []);
  assert.match(edgePath({ id: "丁", x: 2, y: 3 }, { id: "丁", x: 2, y: 3 }), /C/);
});
test("source, relation, evidence and search filters intersect without inferring facts", () => {
  assert.deepEqual(filterGraph(triples, { source: "a", relation: "关联", evidenceOnly: true, search: "  甲 " }), [triples[0]]);
  assert.deepEqual(filterGraph(triples, { search: "不存在" }), []);
  assert.equal(filterGraph(triples, {}).length, 3);
});
test("one-hop neighborhood includes incoming/outgoing edges and preserves identifiers", () => {
  assert.deepEqual(relatedTriples(triples, "乙"), [triples[0]]);
  assert.deepEqual(filterGraph(triples, { neighbor: "甲" }), triples.slice(0, 2));
  assert.equal(tripleKey(triples[0]), "1");
  assert.notEqual(tripleKey({ head: "甲|乙", relation: "丙", tail: "丁" }), tripleKey({ head: "甲", relation: "乙|丙", tail: "丁" }));
});

test("graph identity follows backend entity IDs without merging raw facts", () => {
  const records = [
    { triple_id: "a", head: "Alpha", relation: "r", tail: "乙", head_entity: { id: "entity-1", label: "Alpha" } },
    { triple_id: "b", head: "ALPHA", relation: "s", tail: "丙", head_entity: { id: "entity-1", label: "Alpha" } },
  ];
  assert.equal(layoutGraph(records).nodes.length, 3);
  assert.equal(relatedTriples(records, "entity-1").length, 2);
  assert.equal(records[1].head, "ALPHA");
});

test("overview restores the original radial positions without changing graph identities", () => {
  const layout = layoutGraph(triples);
  assert.equal(layout.width, 920);
  assert.equal(layout.height, 530);
  assert.deepEqual(layout.nodes.map(node => node.id), ["甲", "乙", "丙", "丁"]);
  assert.equal(layout.nodes[0].x, 460);
  assert.equal(layout.nodes[0].y, 125);
});

test("canvas renders entity and relation names by default even above 100 entities", () => {
  const records = Array.from({ length: 60 }, (_, index) => ({ triple_id: String(index), head: `头实体${index}`, relation: `具体关系${index}`, tail: `尾实体${index}` }));
  const html = renderToStaticMarkup(createElement(KnowledgeGraphCanvas, { triples: records, selection: null, onSelect() {} }));
  assert.equal((html.match(/<text\b/g) ?? []).length, 180);
  for (let index = 0; index < records.length; index++) {
    assert.ok(html.includes(`>头实体${index}</text>`));
    assert.ok(html.includes(`>尾实体${index}</text>`));
    assert.ok(html.includes(`>具体关系${index}</text>`));
  }
});
