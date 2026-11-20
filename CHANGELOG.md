# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-20

### Added

- Full-request budgeting through `context_with_system` and `getCurrentSystemMessage()`
  (accounts for system prompt, dynamic sections, and tool schemas directly instead
  of relying on stale post-offload `getContextUsage()`).
- Prompt-cache preservation via hysteresis (`~55%` low-water target of available
  conversation budget) and pinned `[head, marker]` prefixes across turns.
- Provider-usage calibration in `message_end` (`input + cacheRead + cacheWrite`)
  plus cache-hit and estimated-savings metrics in `ctx_status` / `/ctxstatus`.
- Lossless turn-aligned ~2k-token chunking (`serializeMessagesLossless` +
  SHA-1 deduplication), bounded `ctx_search` excerpts, and `blockId` / `order` /
  `offset` pagination in `ctx_recall`.
- Functional `ctx_offload` count/auto trimming and `session_tree` /
  `session_compact` / `model_select` state resets.
- Regression test suite (`npm test` -> `test/index.test.ts`).

### Changed

- Cut-point selection never splits `assistant(toolCall)` from its `toolResult`
  messages, preventing orphaned-tool-result errors.
- Bumped package version to `0.3.0` and updated `package-lock.json` and `README.md`.

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
