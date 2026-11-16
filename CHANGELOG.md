# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
