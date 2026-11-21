# claude-p-proxy

> **For personal use.** Built for using my own Claude Pro/Max subscription, on my own machine, by myself.

A local server that wraps `claude -p` (Claude Code CLI) in OpenAI- and Anthropic-compatible APIs.
Point any SDK's `base_url` at it and it just works. Supports text, streaming, image input, and function calling.
For the ChatGPT/Codex version, see [codex-p-proxy](https://github.com/hunkim/codex-p-proxy).

## Install & run

```bash
git clone https://github.com/hunkim/claude-p-proxy && cd claude-p-proxy
npm install                 # also installs the Claude Code CLI
npx claude auth login       # first time only: sign in with your Pro/Max account
npm start                   # http://127.0.0.1:8787
```

If you already have `claude` installed globally, `node server.mjs` works without `npm install`.

Change the port with `npm start -- --port 9000`. If you don't set one and 8787 is busy, it moves on to 8788, 8789, … automatically. The startup screen prints the actual `base_url` to use.

## Usage

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="local")
print(c.chat.completions.create(model="sonnet",
      messages=[{"role": "user", "content": "Hello"}]).choices[0].message.content)
```

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:8787", api_key="local")
print(c.messages.create(model="opus", max_tokens=1024,
      messages=[{"role": "user", "content": "Hello"}]).content[0].text)
```

For chat UIs like Open WebUI or LibreChat, use `http://127.0.0.1:8787/v1` as the OpenAI-compatible URL.

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible
- `POST /v1/messages` — Anthropic-compatible
- `GET /v1/models`, `GET /health`

A model name containing `opus`/`sonnet`/`haiku`/`fable` uses that model, a full ID like `claude-sonnet-5-5` is passed as is,
and anything else (like `gpt-4o`) goes to `DEFAULT_MODEL`. `reasoning_effort` (low to max) is passed on as `--effort`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8787` | `--port` takes precedence over `PORT` |
| `DEFAULT_MODEL` | `sonnet` | |
| `MAX_CONCURRENT` | `2` | Number of `claude` processes run at once |
| `LOCAL_API_KEY` | (none) | If set, checks `Authorization: Bearer` or `x-api-key` |
| `TIMEOUT_MS` | `300000` | Time limit per request |
| `DEFAULT_SYSTEM` | A short instruction | Used when there's no system message |
| `MAX_IMAGE_BYTES` | 10MB | Max size for URL images |
| `CLAUDE_BIN` | auto | CLI path (default: the CLI in `node_modules` → `claude` on PATH) |

## How it works

- Each request runs `claude -p` in an empty temp folder, with Claude Code's built-in tools, MCP, and session saving turned off.
- The conversation history is merged into one text prompt, and images are attached as base64 image blocks via `--input-format stream-json`. The server downloads URL images itself.
- Function calling: tool definitions go into the system prompt, the model is asked to answer with a `<tool_calls>[...]</tool_calls>` JSON block, and the server extracts it and returns it as `tool_calls` / `tool_use`. Arguments aren't guaranteed to match the schema, so validate them where it matters.
- `ANTHROPIC_API_KEY` is removed from the child process so only your subscription is used.

## Limitations

- A few seconds until the first token (process startup)
- temperature, max_tokens, stop, etc. are ignored; PDF and audio input aren't supported
- Usage counts against the same subscription limits as the Claude app and Claude Code
