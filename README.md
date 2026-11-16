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
| `/ctxlimit <n>` | Set the token limit (e.g. `/ctxlimit 33000`). Omit `<n>` to view the current value. Persisted per project. |
| `/ctxstatus` | Show current usage vs limit and offload statistics. |

The default limit is **33,000 tokens**.

## Tools the model can call

| Tool | Purpose |
|------|---------|
| `ctx_status` | Report usage vs limit and offload statistics. |
| `ctx_recall` | Read back the latest offloaded blocks (newest first, optional substring filter). |
| `ctx_search` | Keyword-search the offload file. |
| `ctx_offload` | Append a note/summary to the offload file, or request oldest-message offload. |

## How it works

- The extension hooks Pi's `context` event, which fires before every LLM call.
- When projected usage exceeds the limit, the oldest messages are serialized with
  `serializeConversation` and appended to `.pi/ctx-offload-<sessionId>.log`,
  then replaced in-context by a single compact marker message.
- Offloading is **incremental**: only messages not yet written this session are
  appended (tracked via `offloadedUpTo`), and identical blocks are de-duplicated,
  so the file does not grow quadratically or repeat the same prefix per call.
- The marker tells the model how many messages were saved and how to recover
  them with `ctx_recall` / `ctx_search`.
- The limit is stored in `.pi/ctx-manager.json` (per working directory).

### Files created at runtime

| Path | Contents |
|------|----------|
| `.pi/ctx-manager.json` | Persisted token limit for the project. |
| `.pi/ctx-offload-<sessionId>.log` | Append-only offloaded content — **one file per conversation**. |

Both live under the current working directory and are git-ignored by default.

## Development

```bash
npm install
npm run typecheck
```

The extension is a single TypeScript file (`index.ts`) loaded by Pi via
[jiti](https://github.com/unjs/jiti) — no build step is required.

Host-provided packages (`@earendil-works/pi-coding-agent`,
`@earendil-works/pi-agent-core`, `@earendil-works/pi-ai`, `typebox`) are declared
as `peerDependencies` and supplied by Pi at runtime.

## License

[MIT](LICENSE)
