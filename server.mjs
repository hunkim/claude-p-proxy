#!/usr/bin/env node
// claude-p-proxy — local OpenAI / Anthropic-compatible API server backed by `claude -p`
// Personal use only: calls the Claude Code CLI logged in with your Claude Pro/Max subscription.
// Supports: text, streaming, image input, function calling (tool use). No dependencies (Node 18+).
// Run: node server.mjs

import http from "node:http";
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";

// Find the claude CLI: CLAUDE_BIN → CLI installed by npm install → claude on PATH
const HERE = dirname(fileURLToPath(import.meta.url));
function findClaude() {
  if (process.env.CLAUDE_BIN) return process.env.CLAUDE_BIN;
  const local = join(HERE, "node_modules", ".bin", process.platform === "win32" ? "claude.cmd" : "claude");
  return existsSync(local) ? local : "claude";
}

// --port 9000 (or PORT=9000). If not set explicitly and 8787 is busy, move on to the next free port
const argPort = process.argv.includes("--port") ? process.argv[process.argv.indexOf("--port") + 1] : undefined;
const CFG = {
  host: process.env.HOST || "127.0.0.1",           // local access only (do not expose)
  port: Number(argPort || process.env.PORT || 8787),
  portExplicit: Boolean(argPort || process.env.PORT),
  apiKey: process.env.LOCAL_API_KEY || "",          // if set, check Bearer / x-api-key
  claudeBin: findClaude(),
  defaultModel: process.env.DEFAULT_MODEL || "sonnet",
  maxConcurrent: Number(process.env.MAX_CONCURRENT || 2),
  timeoutMs: Number(process.env.TIMEOUT_MS || 300_000),
  maxImageBytes: Number(process.env.MAX_IMAGE_BYTES || 10 * 1024 * 1024),
  defaultSystem: process.env.DEFAULT_SYSTEM ||
    "You are a helpful assistant. Answer the user's latest message directly.",
};

// Empty working directory: keeps project context (CLAUDE.md etc.) out
const WORKDIR = mkdtempSync(join(tmpdir(), "claude-proxy-"));
const TAG_OPEN = "<tool_calls>", TAG_CLOSE = "</tool_calls>";
const shortId = () => randomUUID().replace(/-/g, "").slice(0, 24);

// ---------- Model name mapping ----------
function mapModel(name) {
  if (!name) return CFG.defaultModel;
  const n = String(name).toLowerCase();
  if (/^claude-.*\d/.test(n)) return name;                 // full model IDs (claude-opus-5-5 etc.) as is
  for (const alias of ["opus", "sonnet", "haiku", "fable"]) if (n.includes(alias)) return alias;
  return CFG.defaultModel;                                  // gpt-4o etc. → default model
}

// =====================================================================
// 1. Input normalization: OpenAI / Anthropic messages → internal format
//    turn = { role: "user"|"assistant", parts: [...] }
//    part = {kind:"text",text} | {kind:"image",url|media_type+data}
//         | {kind:"tool_call",id,name,args} | {kind:"tool_result",id,name,parts}
// =====================================================================
function imageFromUrl(url) {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url || "");
  return m ? { kind: "image", media_type: m[1], data: m[2] } : { kind: "image", url };
}

function oaParts(content) {
  if (content == null) return [];
  if (typeof content === "string") return [{ kind: "text", text: content }];
  return content.flatMap((p) => {
    if (typeof p === "string") return [{ kind: "text", text: p }];
    if (p.type === "text" || p.type === "input_text") return [{ kind: "text", text: p.text ?? "" }];
    if (p.type === "image_url") return [imageFromUrl(typeof p.image_url === "string" ? p.image_url : p.image_url?.url)];
    if (p.type === "input_image") return [imageFromUrl(p.image_url)];
    return [];
  });
}

function normalizeOpenAI(messages = []) {
  const system = [], turns = [], callNames = {};
  for (const m of messages) {
    if (m.role === "system" || m.role === "developer") { system.push(oaParts(m.content).map((p) => p.text).join("\n")); continue; }
    if (m.role === "assistant") {
      const parts = oaParts(m.content);
      for (const tc of m.tool_calls || []) {
        callNames[tc.id] = tc.function?.name;
        parts.push({ kind: "tool_call", id: tc.id, name: tc.function?.name, args: safeJson(tc.function?.arguments, {}) });
      }
      if (m.function_call) parts.push({ kind: "tool_call", id: "legacy", name: m.function_call.name, args: safeJson(m.function_call.arguments, {}) });
      turns.push({ role: "assistant", parts });
    } else if (m.role === "tool" || m.role === "function") {
      turns.push({ role: "user", parts: [{ kind: "tool_result", id: m.tool_call_id || "legacy", name: m.name || callNames[m.tool_call_id], parts: oaParts(m.content) }] });
    } else {
      turns.push({ role: "user", parts: oaParts(m.content) });
    }
  }
  return { system: system.join("\n\n"), turns };
}

function anParts(content) {
  if (content == null) return [];
  if (typeof content === "string") return [{ kind: "text", text: content }];
  return content.flatMap((b) => {
    if (b.type === "text") return [{ kind: "text", text: b.text ?? "" }];
    if (b.type === "image") {
      const s = b.source || {};
      if (s.type === "base64") return [{ kind: "image", media_type: s.media_type, data: s.data }];
      if (s.type === "url") return [{ kind: "image", url: s.url }];
      return [];
    }
    if (b.type === "tool_use") return [{ kind: "tool_call", id: b.id, name: b.name, args: b.input ?? {} }];
    if (b.type === "tool_result") return [{ kind: "tool_result", id: b.tool_use_id, is_error: b.is_error, parts: anParts(b.content) }];
    return [];
  });
}

function normalizeAnthropic(body) {
  const system = typeof body.system === "string" ? body.system : anParts(body.system).map((p) => p.text).join("\n");
  const turns = (body.messages || []).map((m) => ({ role: m.role === "assistant" ? "assistant" : "user", parts: anParts(m.content) }));
  return { system, turns };
}

// Tool definitions → [{name, description, schema}]
function normalizeTools(body) {
  const list = body.tools || (body.functions || []).map((f) => ({ type: "function", function: f }));
  return list.map((t) => t.type === "function" && t.function
    ? { name: t.function.name, description: t.function.description || "", schema: t.function.parameters || { type: "object", properties: {} } }
    : { name: t.name, description: t.description || "", schema: t.input_schema || { type: "object", properties: {} } })
    .filter((t) => t.name);
}

// tool_choice → {mode:"auto"|"none"|"required"|"specific", name?}
function normalizeChoice(tc) {
  if (!tc) return { mode: "auto" };
  if (typeof tc === "string") return { mode: tc === "required" ? "required" : tc === "none" ? "none" : "auto" };
  if (tc.type === "function") return { mode: "specific", name: tc.function?.name };
  if (tc.type === "tool") return { mode: "specific", name: tc.name };
  if (tc.type === "any") return { mode: "required" };
  if (tc.type === "none") return { mode: "none" };
  return { mode: "auto" };
}

// =====================================================================
// 2. Internal format → claude -p input (text + image blocks)
// =====================================================================
async function resolveImage(p) {
  if (p.data) return p;
  const r = await fetch(p.url);
  if (!r.ok) throw new Error(`image download failed (${r.status}): ${p.url}`);
  const buf = Buffer.from(await r.arrayBuffer());
  if (buf.length > CFG.maxImageBytes) throw new Error(`image too large: ${p.url}`);
  const media_type = (r.headers.get("content-type") || "image/png").split(";")[0];
  return { kind: "image", media_type, data: buf.toString("base64") };
}

async function buildInput(turns) {
  const images = [];
  const render = async (parts) => {
    const out = [], calls = [];
    for (const p of parts) {
      if (p.kind === "text") out.push(p.text);
      else if (p.kind === "image") { images.push(await resolveImage(p)); out.push(`[image ${images.length}]`); }
      else if (p.kind === "tool_call") calls.push({ id: p.id, name: p.name, arguments: p.args });
      else if (p.kind === "tool_result") {
        const inner = await render(p.parts || []);
        out.push(`<tool_result id="${p.id}"${p.name ? ` name="${p.name}"` : ""}${p.is_error ? ' is_error="true"' : ""}>\n${inner}\n</tool_result>`);
      }
    }
    if (calls.length) out.push(`${TAG_OPEN}\n${JSON.stringify(calls)}\n${TAG_CLOSE}`);
    return out.filter(Boolean).join("\n");
  };

  // Merge consecutive turns with the same role (e.g. several tool results)
  const merged = [];
  for (const t of turns) {
    const prev = merged[merged.length - 1];
    if (prev && prev.role === t.role) prev.parts.push(...t.parts); else merged.push({ role: t.role, parts: [...t.parts] });
  }
  const rendered = [];
  for (const t of merged) rendered.push({ role: t.role, text: await render(t.parts) });

  let text;
  const last = rendered[rendered.length - 1];
  if (rendered.length === 1 && last.role === "user") text = last.text;
  else {
    const history = rendered.slice(0, -1).map((t) => `<${t.role}>\n${t.text}\n</${t.role}>`).join("\n");
    text = `<conversation_history>\n${history}\n</conversation_history>\n\n` +
      (last?.role === "user"
        ? `Continue the conversation. Reply only as the assistant to this latest message:\n\n${last.text}`
        : `Continue the assistant's last message:\n\n${last?.text ?? ""}`);
  }
  return { text, images };
}

// Tool-use instructions (appended to the system prompt)
function toolInstructions(tools, choice) {
  if (!tools.length || choice.mode === "none") return "";
  const defs = tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema }));
  let rule = "Call tools only when they are needed; otherwise answer normally without the block.";
  if (choice.mode === "required") rule = "You MUST call at least one tool in this reply.";
  if (choice.mode === "specific") rule = `You MUST call the tool "${choice.name}" in this reply.`;
  return `

# Tools
You can call these tools (JSON Schema for each tool's arguments):
${JSON.stringify(defs, null, 2)}

To call tools, end your reply with exactly this block and nothing after it:
${TAG_OPEN}
[{"name": "<tool name>", "arguments": { ... }}]
${TAG_CLOSE}

Rules:
- The block must contain a valid JSON array; several calls may go in one array.
- Use only the tools listed above, with arguments that match their schema.
- After the block, stop. Tool results will come back to you in <tool_result> tags.
- You can write a short sentence before the block, but never invent tool results yourself.
- ${rule}`;
}

// Extract the tool-call block from model output
function safeJson(s, fallback) {
  if (s && typeof s === "object") return s;
  try { return JSON.parse(s); } catch { return fallback; }
}
function parseToolCalls(text, tools) {
  const i = text.indexOf(TAG_OPEN);
  if (i < 0 || !tools.length) return { text, calls: [] };
  const j = text.indexOf(TAG_CLOSE, i);
  let raw = text.slice(i + TAG_OPEN.length, j < 0 ? undefined : j).trim()
    .replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  let arr = safeJson(raw, null);
  if (arr && !Array.isArray(arr)) arr = [arr];
  const names = new Set(tools.map((t) => t.name));
  const calls = (arr || [])
    .filter((c) => c && names.has(c.name))
    .map((c) => ({ name: c.name, args: safeJson(c.arguments ?? c.input ?? c.parameters, {}) ?? {} }));
  if (!calls.length) return { text, calls: [] };           // parse failed → treat as plain text
  return { text: text.slice(0, i).trim(), calls };
}

// Filter that hides text after "<tool_calls>" while streaming
function makeHoldback(enabled, emit) {
  let acc = "", sent = 0, found = false;
  return {
    push(delta) {
      acc += delta;
      if (!enabled) { emit(delta); sent = acc.length; return; }
      if (found) return;
      const idx = acc.indexOf(TAG_OPEN);
      if (idx >= 0) { found = true; if (idx > sent) emit(acc.slice(sent, idx)); sent = idx; return; }
      let k = Math.min(TAG_OPEN.length - 1, acc.length);    // hold back a tail that may be the start of the tag
      while (k > 0 && !TAG_OPEN.startsWith(acc.slice(-k))) k--;
      const safe = acc.length - k;
      if (safe > sent) { emit(acc.slice(sent, safe)); sent = safe; }
    },
    finish(hasCalls) {                                     // if it wasn't a tool call, flush the remaining text
      if (!hasCalls && acc.length > sent) emit(acc.slice(sent));
    },
    get text() { return acc; },
  };
}

// =====================================================================
// 3. Run claude -p
// =====================================================================
let running = 0;
const queue = [];
function acquire() {
  return new Promise((res) => { if (running < CFG.maxConcurrent) { running++; res(); } else queue.push(res); });
}
function release() { const next = queue.shift(); if (next) next(); else running--; }

/** Run claude -p. Text chunks go to onDelta. Returns { text, usage, model } */
async function runClaude({ input, system, model, effort, onDelta, signal }) {
  await acquire();
  try {
    if (signal?.aborted) throw new Error("client disconnected");   // client left while queued → don't run
    return await new Promise((resolve, reject) => {
      const args = [
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        "--model", model,
        "--system-prompt", system || CFG.defaultSystem,
        "--tools", "",                  // turn off all Claude Code built-in tools (plain chatbot)
        "--strict-mcp-config",          // don't load MCP servers
        "--no-session-persistence",
      ];
      if (effort) args.push("--effort", effort);
      const env = { ...process.env };
      delete env.ANTHROPIC_API_KEY;     // force the subscription login (no API billing)
      const child = spawn(CFG.claudeBin, args, { cwd: WORKDIR, env, stdio: ["pipe", "pipe", "pipe"] });

      let buf = "", streamed = "", finalText = null, usage = {}, usedModel = model, stderr = "", errMsg = null;
      const timer = setTimeout(() => child.kill("SIGTERM"), CFG.timeoutMs);
      signal?.addEventListener("abort", () => child.kill("SIGTERM"));

      child.stdout.on("data", (chunk) => {
        buf += chunk;
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
          if (!line) continue;
          let ev; try { ev = JSON.parse(line); } catch { continue; }
          if (ev.type === "stream_event") {
            const e = ev.event;
            if (e?.type === "content_block_delta" && e.delta?.type === "text_delta") { streamed += e.delta.text; onDelta?.(e.delta.text); }
            if (e?.type === "message_start" && e.message?.model) usedModel = e.message.model;
          } else if (ev.type === "system" && ev.model) {
            usedModel = ev.model;
          } else if (ev.type === "result") {
            if (ev.is_error || ev.subtype !== "success") errMsg = ev.result || ev.subtype || "claude error";
            else finalText = ev.result ?? "";
            usage = ev.usage || {};
          }
        }
      });
      child.stderr.on("data", (d) => { stderr += d; });
      child.stdin.on("error", () => {});
      child.on("error", (e) => { clearTimeout(timer); reject(new Error(`failed to run claude: ${e.message}`)); });
      child.on("close", (code) => {
        clearTimeout(timer);
        if (errMsg) return reject(new Error(errMsg));
        if (finalText == null && !streamed) {
          return reject(new Error(`claude exited with code ${code}: ${stderr.trim().slice(0, 500) || "no output (check login / usage limits)"}`));
        }
        if (!streamed && finalText) onDelta?.(finalText);   // no chunks → send it all at once
        resolve({ text: streamed || finalText || "", usage, model: usedModel });
      });

      // One user message: image blocks + text block
      const content = [];
      input.images.forEach((img, n) => {
        content.push({ type: "text", text: `[image ${n + 1}]` });
        content.push({ type: "image", source: { type: "base64", media_type: img.media_type, data: img.data } });
      });
      content.push({ type: "text", text: input.text || "(empty)" });
      child.stdin.end(JSON.stringify({ type: "user", message: { role: "user", content } }) + "\n");
    });
  } finally { release(); }
}

// Shared setup: normalized conversation + tools → runClaude arguments
async function prepare(norm, body) {
  const tools = normalizeTools(body);
  const choice = normalizeChoice(body.tool_choice ?? body.function_call);
  const activeTools = choice.mode === "none" ? [] : tools;
  const input = await buildInput(norm.turns);
  const system = (norm.system || CFG.defaultSystem) + toolInstructions(activeTools, choice);
  return { input, system, model: mapModel(body.model), effort: effortFrom(body), tools: activeTools };
}

// ---------- HTTP helpers ----------
function send(res, status, obj) {
  res.writeHead(status, { "content-type": "application/json", "access-control-allow-origin": "*" });
  res.end(JSON.stringify(obj));
}
function sse(res) {
  res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive", "access-control-allow-origin": "*" });
}
async function readJson(req) {
  let body = ""; for await (const c of req) body += c;
  try { return body ? JSON.parse(body) : {}; }
  catch (e) { throw Object.assign(new Error(`invalid JSON body: ${e.message}`), { status: 400 }); }
}
function authorized(req) {
  if (!CFG.apiKey) return true;
  const bearer = (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
  return bearer === CFG.apiKey || req.headers["x-api-key"] === CFG.apiKey;
}
function effortFrom(body) {
  const e = body.reasoning_effort || body.effort || body.output_config?.effort;
  if (["low", "medium", "high", "xhigh", "max"].includes(e)) return e;
  if (e === "minimal") return "low";
  return undefined;
}

// =====================================================================
// 4. OpenAI: POST /v1/chat/completions
// =====================================================================
function oaUsage(u = {}) {
  const p = (u.input_tokens || 0) + (u.cache_read_input_tokens || 0) + (u.cache_creation_input_tokens || 0);
  const c = u.output_tokens || 0;
  return { prompt_tokens: p, completion_tokens: c, total_tokens: p + c };
}
const oaToolCalls = (calls) => calls.map((c, index) => ({
  index, id: `call_${shortId()}`, type: "function", function: { name: c.name, arguments: JSON.stringify(c.args) },
}));

async function openaiChat(req, res, body, signal) {
  const p = await prepare(normalizeOpenAI(body.messages), body);
  const id = `chatcmpl-${randomUUID()}`, created = Math.floor(Date.now() / 1000), model = body.model || p.model;

  if (body.stream) {
    sse(res);
    const chunk = (delta, finish = null) => res.write(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`);
    chunk({ role: "assistant", content: "" });
    try {
      const hb = makeHoldback(p.tools.length > 0, (t) => chunk({ content: t }));
      const r = await runClaude({ ...p, signal, onDelta: (t) => hb.push(t) });
      const { calls } = parseToolCalls(r.text, p.tools);
      hb.finish(calls.length > 0);
      if (calls.length) {
        oaToolCalls(calls).forEach((tc) => chunk({ tool_calls: [tc] }));
        chunk({}, "tool_calls");
      } else chunk({}, "stop");
      if (body.stream_options?.include_usage) res.write(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model, choices: [], usage: oaUsage(r.usage) })}\n\n`);
    } catch (e) {
      res.write(`data: ${JSON.stringify({ error: { message: e.message, type: "server_error" } })}\n\n`);
    }
    res.write("data: [DONE]\n\n"); res.end();
    return;
  }

  const r = await runClaude({ ...p, signal });
  const { text, calls } = parseToolCalls(r.text, p.tools);
  const message = { role: "assistant", content: calls.length ? (text || null) : text };
  if (calls.length) message.tool_calls = oaToolCalls(calls).map(({ index, ...tc }) => tc);
  send(res, 200, {
    id, object: "chat.completion", created, model,
    choices: [{ index: 0, message, finish_reason: calls.length ? "tool_calls" : "stop" }],
    usage: oaUsage(r.usage),
  });
}

// =====================================================================
// 5. Anthropic: POST /v1/messages
// =====================================================================
async function anthropicMessages(req, res, body, signal) {
  const p = await prepare(normalizeAnthropic(body), body);
  const id = `msg_${shortId()}`, model = body.model || p.model;

  if (body.stream) {
    sse(res);
    const ev = (type, data) => res.write(`event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`);
    ev("message_start", { message: { id, type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } });
    let idx = -1, textOpen = false;
    const emitText = (t) => {
      if (!t) return;
      if (!textOpen) { idx++; textOpen = true; ev("content_block_start", { index: idx, content_block: { type: "text", text: "" } }); }
      ev("content_block_delta", { index: idx, delta: { type: "text_delta", text: t } });
    };
    try {
      const hb = makeHoldback(p.tools.length > 0, emitText);
      const r = await runClaude({ ...p, signal, onDelta: (t) => hb.push(t) });
      const { calls } = parseToolCalls(r.text, p.tools);
      hb.finish(calls.length > 0);
      if (textOpen) ev("content_block_stop", { index: idx });
      for (const c of calls) {
        idx++;
        ev("content_block_start", { index: idx, content_block: { type: "tool_use", id: `toolu_${shortId()}`, name: c.name, input: {} } });
        ev("content_block_delta", { index: idx, delta: { type: "input_json_delta", partial_json: JSON.stringify(c.args) } });
        ev("content_block_stop", { index: idx });
      }
      ev("message_delta", { delta: { stop_reason: calls.length ? "tool_use" : "end_turn", stop_sequence: null }, usage: { output_tokens: r.usage.output_tokens || 0 } });
      ev("message_stop", {});
    } catch (e) {
      ev("error", { error: { type: "api_error", message: e.message } });
    }
    res.end();
    return;
  }

  const r = await runClaude({ ...p, signal });
  const { text, calls } = parseToolCalls(r.text, p.tools);
  const content = [];
  if (text) content.push({ type: "text", text });
  for (const c of calls) content.push({ type: "tool_use", id: `toolu_${shortId()}`, name: c.name, input: c.args });
  send(res, 200, {
    id, type: "message", role: "assistant", model, content,
    stop_reason: calls.length ? "tool_use" : "end_turn", stop_sequence: null,
    usage: { input_tokens: r.usage.input_tokens || 0, output_tokens: r.usage.output_tokens || 0 },
  });
}

// =====================================================================
// 6. Routing
// =====================================================================
const MODELS = ["sonnet", "opus", "haiku", "fable"];
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  if (req.method === "OPTIONS") {
    res.writeHead(204, { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "GET,POST,OPTIONS" });
    return res.end();
  }
  if (url.pathname === "/health") return send(res, 200, { ok: true, running, queued: queue.length });
  if (!authorized(req)) return send(res, 401, { error: { type: "authentication_error", message: "invalid api key" } });

  const ac = new AbortController();
  res.on("close", () => { if (!res.writableEnded) ac.abort(); });

  try {
    if (req.method === "GET" && url.pathname === "/v1/models") {
      return send(res, 200, { object: "list", data: MODELS.map((id) => ({ id, object: "model", type: "model", owned_by: "anthropic", display_name: id })) });
    }
    if (req.method === "POST" && url.pathname === "/v1/chat/completions") return await openaiChat(req, res, await readJson(req), ac.signal);
    if (req.method === "POST" && url.pathname === "/v1/messages") return await anthropicMessages(req, res, await readJson(req), ac.signal);
    send(res, 404, { error: { type: "not_found_error", message: `${req.method} ${url.pathname}` } });
  } catch (e) {
    console.error("[error]", e.message);
    if (!res.headersSent) send(res, e.status || 500, { error: { type: e.status === 400 ? "invalid_request_error" : "api_error", message: e.message } });
    else res.end();
  }
});

// Check the CLI before starting
const check = spawnSync(CFG.claudeBin, ["--version"], { encoding: "utf8", shell: process.platform === "win32" });
if (check.error || check.status !== 0) {
  console.error(`claude CLI not found (${CFG.claudeBin}).
  Run 'npm install' in this folder, or install it with 'npm i -g @anthropic-ai/claude-code'.
  Then run 'npx claude auth login' once to sign in with your Pro/Max subscription.`);
  process.exit(1);
}
console.log(`claude CLI: ${check.stdout.trim()} (${CFG.claudeBin})`);

let port = CFG.port;
server.on("error", (e) => {
  if (e.code === "EADDRINUSE" && !CFG.portExplicit && port < CFG.port + 10) {
    console.log(`port ${port} is in use → trying ${port + 1}`);
    return server.listen(++port, CFG.host);
  }
  console.error(e.code === "EADDRINUSE" ? `port ${port} is already in use. Pick another one with --port.` : e.message);
  process.exit(1);
});
server.listen(port, CFG.host, () => {
  const base = `http://${CFG.host}:${port}`;
  console.log(`claude-p-proxy → ${base}`);
  console.log(`  OpenAI    : base_url ${base}/v1   (POST /v1/chat/completions)`);
  console.log(`  Anthropic : base_url ${base}      (POST /v1/messages)`);
  console.log(`  api_key   : ${CFG.apiKey ? "the LOCAL_API_KEY value" : "any value (e.g. local)"}`);
  console.log(`  models    : ${MODELS.join(", ")}  (default ${CFG.defaultModel}, ${CFG.maxConcurrent} concurrent)`);
  console.log(`
Example:
  curl ${base}/v1/chat/completions -H 'content-type: application/json' \\
    -d '{"model":"sonnet","messages":[{"role":"user","content":"Hello"}]}'`);
});
