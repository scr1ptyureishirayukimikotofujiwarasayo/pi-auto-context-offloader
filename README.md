# pi-auto-context-offloader

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pi Package](https://img.shields.io/badge/pi-package-blue)](https://pi.dev/packages)

A [Pi coding-agent](https://github.com/earendil-works/pi) extension that keeps the
live conversation under a **user-configurable token budget** by offloading the
oldest turns to a plain-text file on disk and replacing them in-context with
compact markers. The model can recall or search the offloaded content through
tools, and explicitly offload whatever it deems unnecessary.

## Why

- **Lower cost** — offloaded tokens are never re-sent to the provider until the
  model actively recalls them, so token spend is bounded by the limit instead of
  by conversation length.
- **Better fidelity than compaction** — the full raw text is preserved on disk
  (no lossy summary); the model retrieves exactly what it needs, when it needs it.
- **Model-driven control** — the LLM can proactively push content out of context
  and pull it back on demand.

## Install

As a Pi package (npm or git):

```bash
pi install npm:pi-auto-context-offloader
# or
pi install git:github.com/scr1ptyureishirayukimikotofujiwarasayo/pi-auto-context-offloader
```

For a single invocation without installing:

```bash
pi -e git:github.com/scr1ptyureishirayukimikotofujiwarasayo/pi-auto-context-offloader
```

From a local checkout:

```bash
pi --extension ./index.ts
```

Or auto-load globally by placing this folder under
`~/.pi/agent/extensions/pi-auto-context-offloader/`, then `/reload`.

## Usage

| Command | Effect |
|---------|--------|
| `/ctxlimit <n>` | Set a custom token limit between **10 and 100000** (e.g. `/ctxlimit 45000`, `/ctxlimit 50k`). Omit `<n>` to view the current value. Persisted per project. |
| `/ctxstatus` | Show current usage vs limit and offload statistics. |
| `/clearcontext` | Delete the offloaded files for **this conversation** (`.pi/offload/<sessionId>/`). Use `/clearcontext all` to wipe every conversation's offload folder. |

The default limit is **33,000 tokens**.

## Tools the model can call

| Tool | Purpose |
|------|---------|
| `ctx_status` | Report full-session vs effective request tokens, cache stability, and offload statistics. |
| `ctx_recall` | Read back offloaded blocks (by `count`, `blockId`, `query`, `order`, and paginated `offset`). |
| `ctx_search` | Keyword-search the offload file and return compact matching excerpts with block IDs. |
| `ctx_offload` | Append a note/summary to the offload file, or schedule an oldest-message trim on the next LLM call. |

## How it works

- The extension hooks Pi's `context_with_system` event, which fires before every
  LLM call with the complete message array including the system prompt and tool
  definitions (`getCurrentSystemMessage()`).
- **Hysteresis & prompt-cache pinning**: when calibrated request tokens exceed
  the limit, the oldest turns are offloaded down to a low-water target (~55% of
  available conversation budget) and the `[system head, offload marker]` prefix
  is pinned across subsequent turns until the growing tail reaches the limit
  again. This preserves provider prompt caches across consecutive calls instead
  of invalidating the prefix every turn.
- **Usage calibration**: `message_end` compares estimated request tokens against
  provider-reported input tokens (`input + cacheRead + cacheWrite`) so budget
  enforcement adapts to tokenizer differences and system-prompt extensions.
- **Turn-boundary safety & lossless storage**: cuts never split
  `assistant(toolCall)` from its `toolResult` messages. Offloaded turns are
  saved in ~2k-token deduplicated chunks to `.pi/offload/<sessionId>/offload.log`
  without truncating tool outputs, while `ctx_search` returns bounded excerpts
  and `ctx_recall` paginates large blocks safely.
- The limit is stored in `.pi/ctx-manager.json` (per working directory).

### Files created at runtime

| Path | Contents |
|------|----------|
| `.pi/ctx-manager.json` | Persisted token limit for the project. |
| `.pi/offload/<sessionId>/offload.log` | Append-only offloaded content — **one folder per conversation**. |

Both live under the current working directory and are git-ignored by default.

## Development

```bash
npm install
npm run typecheck
npm test
```

The extension is a single TypeScript file (`index.ts`) loaded by Pi via
[jiti](https://github.com/unjs/jiti) — no build step is required.

Host-provided packages (`@earendil-works/pi-coding-agent`,
`@earendil-works/pi-agent-core`, `@earendil-works/pi-ai`, `typebox`) are declared
as `peerDependencies` and supplied by Pi at runtime.

## License

[MIT](LICENSE)
