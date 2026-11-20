# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-20

### Added

- Complete request accounting through Pi's `context_with_system` event,
  including the current system prompt, dynamic sections, and tool definitions.
- Provider-usage calibration based on `input + cacheRead + cacheWrite`.
- Prompt-cache pinning and hysteresis so the system head and offload marker stay
  stable between cuts.
- Lossless, turn-aligned offload chunks and bounded, paginated retrieval.
- Regression tests for cache stability, tool-call boundaries, calibration,
  deduplication, retrieval, manual trimming, and chunk sizing.

### Changed

- Preserve assistant tool calls together with their tool results at cut points.
- Keep full tool outputs in per-session offload logs instead of truncating them.
- Expose host packages as peer dependencies and add package test/typecheck
  scripts.

## [0.2.0] - 2026-11-16

### Added

- `/clearcontext` command — deletes this conversation's offload folder
  (`.pi/offload/<sessionId>/`); `/clearcontext all` wipes every conversation's
  offload folder. Resets in-memory block counters so a fresh dump starts at block 1.
- Offload content is now stored **one folder per conversation** at
  `.pi/offload/<sessionId>/offload.log` instead of a flat file per session.

### Changed

- **One offload file per conversation**: the path is resolved at `session_start`
  from `ctx.sessionManager.getSessionId()`, writing to
  `.pi/ctx-offload-<sessionId>.log` instead of a single shared `ctx-offload.log`.
  All `ctx_*` tools read the same per-session file.
- `/ctxlimit` now accepts a **custom limit between 10 and 100000** tokens
  (validated on set and clamped when loading persisted state), and accepts
  shorthand like `50k` / `100 k` / `33,000`.
- `ctx_status` now reports the **effective request size** (`overhead + retained
  tail`) and warns when the fixed prompt+tools overhead alone exceeds the limit
  (an unreachable target), suggesting a larger `/ctxlimit`.

### Fixed

- `/ctxlimit` autocomplete no longer hijacks a typed custom value: it now
  filters by the typed prefix and returns `null` once a valid custom number
  is entered, so Enter submits exactly what was typed instead of a preset.
- Duplicate/quadratic growth: because the `context` hook is non-destructive and
  fires before every LLM call, the same prefix was re-appended repeatedly (seen
  up to 45x for one block). Offloading is now **incremental** (`offloadedUpTo`)
  and identical blocks are de-duplicated via a rebuilt block index.

## [0.1.0] - 2026-11-16

### Added

- `context` hook that auto-offloads the oldest conversation turns to
  `.pi/ctx-offload.log` when the live context exceeds the configured limit.
- `/ctxlimit <n>` command to set or view the token limit (persisted per project
  in `.pi/ctx-manager.json`).
- `/ctxstatus` command to show usage vs limit and offload statistics.
- `ctx_status` tool — report budget usage and offload stats.
- `ctx_recall` tool — read back offloaded blocks (newest first, optional filter).
- `ctx_search` tool — keyword search across the offload file.
- `ctx_offload` tool — append a note/summary to the offload file, or request
  oldest-message offload.
