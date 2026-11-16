/**
 * Auto Context Manager
 *
 * Keeps the live conversation under a user-configurable token budget by
 * offloading the oldest turns to a plain-text file on disk and replacing them
 * in-context with compact markers. The model can recall or search the offloaded
 * content through tools, and explicitly offload whatever it deems unnecessary.
 *
 * Why a file instead of in-memory summaries:
 *   - Cost: offloaded tokens are never re-sent to the provider until the model
 *     actively recalls them, so token spend is bounded by the limit, not by
 *     conversation length.
 *   - Fidelity: the full raw text is preserved on disk, so nothing is lost to
 *     a lossy summary. The model retrieves exactly what it needs.
 *
 * Controls:
 *   /ctxlimit <n>        set the token limit (10–100000; accepts 20k/50k), or print current
 *   /ctxstatus           print current usage vs limit and offload stats
 *
 * Tools exposed to the model:
 *   ctx_recall           read back a range / the latest offloaded content
 *   ctx_search           keyword search the offload file
 *   ctx_offload          explicitly offload older conversation content now
 *   ctx_status           report budget usage and offload stats
 */

import { type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { estimateTokens, serializeConversation, convertToLlm } from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { Message } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import * as fs from "node:fs";
import * as path from "node:path";

// ---------------------------------------------------------------------------
// Configuration / persistence
// ---------------------------------------------------------------------------

const DEFAULT_LIMIT = 33_000;
// Accepted range for a custom /ctxlimit value (tokens).
const MIN_LIMIT = 10;
const MAX_LIMIT = 100_000;

interface State {
  limit: number;
}

function statePath(cwd: string): string {
  // Per-project state so each project has its own offload file and limit.
  return path.join(cwd, ".pi", "ctx-manager.json");
}

function offloadPath(cwd: string): string {
  // One offload file per conversation. `currentOffloadFile` is resolved at
  // session start from the session id; the cwd-based path is only a fallback
  // for the rare case a tool runs before session_start has fired.
  return currentOffloadFile || path.join(cwd, ".pi", "ctx-offload.log");
}

function loadState(cwd: string): State {
  try {
    const raw = fs.readFileSync(statePath(cwd), "utf8");
    const parsed = JSON.parse(raw) as Partial<State>;
    const limit =
      typeof parsed.limit === "number" && parsed.limit > 0
        ? Math.min(MAX_LIMIT, Math.max(MIN_LIMIT, Math.floor(parsed.limit)))
        : DEFAULT_LIMIT;
    return { limit };
  } catch {
    return { limit: DEFAULT_LIMIT };
  }
}

function saveState(cwd: string, state: State): void {
  try {
    fs.mkdirSync(path.dirname(statePath(cwd)), { recursive: true });
    fs.writeFileSync(statePath(cwd), JSON.stringify(state, null, 2), "utf8");
  } catch (err) {
    // Non-fatal: state just won't persist across restarts.
    console.error("[ctx-manager] failed to save state:", err);
  }
}

// ---------------------------------------------------------------------------
// Offload file format
// ---------------------------------------------------------------------------
// The offload file is append-only. Each block starts with a marker line
// `## OFFLOAD <id> <timestamp> <tokens>` followed by the serialized messages,
// and ends with a blank line. `ctx_search` scans these blocks.

let nextOffloadId = 1;
// Dedupe index: serialized block body -> block id. The `context` hook fires
// before *every* LLM call and is non-destructive (the session history is never
// mutated), so the same leading messages get offered for offload on each call.
// Without this index we would append the identical block over and over.
let offloadIndex = new Map<string, number>();
// Absolute path of the offload file for the current conversation, set on
// session_start from the session id so each conversation gets its own file.
let currentOffloadFile = "";

function appendOffload(cwd: string, messages: AgentMessage[], tokens: number): { id: number; start: number } {
  const text = serializeConversation(convertToLlm(messages));
  const key = text.trim();
  const existing = offloadIndex.get(key);
  if (existing !== undefined) {
    // Already offloaded this exact content (e.g. a repeat LLM call in the same
    // turn). Reuse the block instead of appending a duplicate.
    return { id: existing, start: 0 };
  }
  fs.mkdirSync(path.dirname(offloadPath(cwd)), { recursive: true });
  const id = nextOffloadId++;
  const ts = new Date().toISOString();
  const block = `\n## OFFLOAD ${id} ${ts} ${tokens}\n${text}\n`;
  fs.appendFileSync(offloadPath(cwd), block, "utf8");
  offloadIndex.set(key, id);
  return { id, start: block.length };
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  let cwd = process.cwd();
  let state: State = { limit: DEFAULT_LIMIT };
  // What the last `context` hook actually sent (post-offload) vs the raw session,
  // so ctx_status can report a truthful number instead of the untruncated total.
  let lastRequest: { overhead: number; request: number; session: number; kept: number } | null = null;
  // How many leading messages have already been written to the offload file this
  // session. The `context` hook sees the full history every LLM call, so this
  // lets us write only the not-yet-offloaded delta instead of re-snapshotting.
  let offloadedUpTo = 0;

  pi.on("session_start", async (_event, ctx) => {
    cwd = ctx.cwd;
    state = loadState(cwd);
    // Per-conversation offload file: <cwd>/.pi/ctx-offload-<sessionId>.log
    let sid = "default";
    try {
      sid = ctx.sessionManager.getSessionId();
    } catch {
      /* ephemeral session: fall back to default */
    }
    const safe = String(sid).replace(/[^A-Za-z0-9._-]/g, "-");
    currentOffloadFile = path.join(cwd, ".pi", `ctx-offload-${safe}.log`);
    nextOffloadId = 1;
    // Rebuild the dedupe index from the existing file so we never append a block
    // whose content is already present, and continue ids after the highest seen.
    offloadIndex = new Map();
    for (const b of readOffloadBlocks(cwd)) {
      offloadIndex.set(b.text, b.id);
      if (b.id >= nextOffloadId) nextOffloadId = b.id + 1;
    }
    lastRequest = null;
    offloadedUpTo = 0;
  });

  // -------------------------------------------------------------------------
  // Auto-offload: run before each LLM call; if the conversation is over budget,
  // serialize the oldest turns to the offload file and replace them with markers.
  // -------------------------------------------------------------------------
  pi.on("context", async (event, ctx) => {
    const messages = event.messages;
    const usage = ctx.getContextUsage();
    const currentTokens = usage?.tokens ?? estimateTotal(messages);

    if (currentTokens <= state.limit) {
      const msgTokens = estimateTotal(messages);
      lastRequest = {
        overhead: Math.max(0, currentTokens - msgTokens),
        request: currentTokens,
        session: currentTokens,
        kept: msgTokens,
      };
      return; // within budget, nothing to do
    }

    // Decide how many leading messages to offload. We keep the most recent
    // messages that fit within the budget (minus a safety margin), preserving
    // the tail of the conversation so the model stays coherent.
    const margin = Math.floor(state.limit * 0.1);
    const targetTotal = state.limit - margin;

    // Tokens that are NOT part of `messages` (system prompt, tool schemas, etc.).
    // `currentTokens` counts the whole context, so we must reserve this fixed
    // overhead before budgeting the message tail — otherwise, with a large
    // system prompt, the message budget would look bottomless and nothing would
    // ever be offloaded despite the context being over the limit.
    const messageTokens = estimateTotal(messages);
    const overhead = Math.max(0, currentTokens - messageTokens);

    // Budget available for the retained message tail. When the fixed overhead
    // alone already exceeds the limit, the target is unreachable — fall back to
    // keeping a small coherent tail (~15% of the limit) so growth stays bounded
    // without wiping the conversation on every turn.
    const unreachable = overhead >= state.limit;
    const keepBudget = unreachable
      ? Math.floor(state.limit * 0.15)
      : Math.max(0, targetTotal - overhead);

    // Walk from the newest message backwards to find the earliest index we keep.
    let keepFrom = messages.length;
    let acc = 0;
    for (let i = messages.length - 1; i >= 0; i--) {
      const t = estimateTokens(messages[i]);
      if (acc + t > keepBudget && i < messages.length - 1) {
        break;
      }
      acc += t;
      keepFrom = i;
    }

    // We must never offload the most recent user message (the current request).
    // Keep at least the last message.
    if (keepFrom >= messages.length) {
      keepFrom = messages.length - 1;
    }

    // Only the messages between what we already offloaded this session and the
    // new keep-point need to be written. The `context` hook is non-destructive
    // (the session history is never mutated), so the same leading messages
    // reappear on every LLM call; tracking `offloadedUpTo` keeps the file
    // incremental instead of re-snapshotting the whole prefix each time.
    if (keepFrom > offloadedUpTo) {
      const delta = messages.slice(offloadedUpTo, keepFrom);
      const contentMessages = delta.filter(isContentBearing);
      const offloadTokens = delta.reduce((sum, m) => sum + estimateTokens(m), 0);
      if (contentMessages.length > 0) {
        appendOffload(cwd, contentMessages, offloadTokens);
      }
      offloadedUpTo = keepFrom;
    }

    // Build a compact marker message that stands in for the offloaded history.
    const marker: AgentMessage = {
      role: "user",
      content: [
        {
          type: "text",
          text:
            `[Earlier conversation offloaded to the context file to stay within your ` +
            `${state.limit}-token budget. ${offloadedUpTo} messages saved. ` +
            `Use ctx_recall to read them back, or ctx_search to find specific details.]`,
        },
      ],
      timestamp: Date.now(),
    } as AgentMessage;

    const retained = messages.slice(keepFrom);
    const retainedTokens = retained.reduce((sum, m) => sum + estimateTokens(m), 0);
    const markerTokens = estimateTokens(marker);
    lastRequest = {
      overhead,
      request: overhead + retainedTokens + markerTokens,
      session: currentTokens,
      kept: retainedTokens + markerTokens,
    };

    return { messages: [marker, ...retained] };
  });

  // -------------------------------------------------------------------------
  // Tool: ctx_status — report budget usage and offload stats
  // -------------------------------------------------------------------------
  pi.registerTool({
    name: "ctx_status",
    label: "Context Status",
    description:
      "Report the current context usage, the configured token limit, and how much has been offloaded to the context file.",
    parameters: Type.Object({}),
    async execute(_id, _params, _signal, _onUpdate, ctx) {
      const usage = ctx.getContextUsage();
      const offloaded = readOffloadStats(cwd);
      const tokens = usage?.tokens ?? null;
      const percent = tokens != null ? Math.round((tokens / state.limit) * 100) : null;
      const lines: string[] = [
        `Context limit: ${state.limit} tokens`,
        `Session usage: ${tokens ?? "unknown"} tokens${percent != null ? ` (${percent}% of limit)` : ""}`,
      ];
      if (lastRequest) {
        const reqPct = Math.round((lastRequest.request / state.limit) * 100);
        lines.push(
          `Last request sent: ~${lastRequest.request} tokens (${reqPct}% of limit) ` +
            `= ~${lastRequest.overhead} fixed (prompt+tools) + ~${lastRequest.kept} retained tail`,
        );
        if (lastRequest.overhead >= state.limit) {
          lines.push(
            `WARNING: prompt+tools alone (~${lastRequest.overhead}) exceed the limit; ` +
              `no offload can bring a request under budget. Raise it (e.g. /ctxlimit ${lastRequest.overhead + 20000}).`,
          );
        }
      }
      lines.push(`Offloaded: ${offloaded.blocks} blocks, ~${offloaded.tokens} tokens in ${offloadPath(cwd)}`);
      const text = lines.join("\n");
      return {
        content: [{ type: "text", text }],
        details: {
          usageTokens: tokens,
          percent,
          offloadedBlocks: offloaded.blocks,
          offloadedTokens: offloaded.tokens,
          lastRequestTokens: lastRequest?.request ?? null,
          overheadTokens: lastRequest?.overhead ?? null,
        },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: ctx_recall — read back offloaded content
  // -------------------------------------------------------------------------
  pi.registerTool({
    name: "ctx_recall",
    label: "Recall Context",
    description:
      "Read back conversation content that was previously offloaded to the context file. " +
      "Use this to recover earlier context, decisions, or details the model no longer has in its live context.",
    parameters: Type.Object({
      count: Type.Optional(
        Type.Integer({
          minimum: 1,
          maximum: 50,
          description: "Number of trailing offload blocks to return, newest first (default 1).",
        })
      ),
      query: Type.Optional(
        Type.String({ description: "Optional substring to filter blocks by (case-insensitive)." })
      ),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      const blocks = readOffloadBlocks(cwd);
      let selected = blocks;
      if (params.query) {
        const q = params.query.toLowerCase();
        selected = blocks.filter((b) => b.text.toLowerCase().includes(q));
      }
      const count = params.count ?? 1;
      const picked = selected.slice(-count).reverse();
      if (picked.length === 0) {
        return {
          content: [{ type: "text", text: "No offloaded content found" + (params.query ? " matching that query" : "") + "." }],
          details: { blocks: 0 },
        };
      }
      const out = picked.map((b) => b.text).join("\n\n");
      return { content: [{ type: "text", text: out }], details: { blocks: picked.length } };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: ctx_search — keyword search the offload file
  // -------------------------------------------------------------------------
  pi.registerTool({
    name: "ctx_search",
    label: "Search Context",
    description:
      "Search the offloaded conversation history for a keyword or phrase. Returns matching blocks from the context file.",
    parameters: Type.Object({
      query: Type.String({ description: "Keyword or phrase to search for (case-insensitive)." }),
      limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      const blocks = readOffloadBlocks(cwd);
      const q = params.query.toLowerCase();
      const matches = blocks.filter((b) => b.text.toLowerCase().includes(q));
      const picked = matches.slice(-(params.limit ?? 5)).reverse();
      if (picked.length === 0) {
        return {
          content: [{ type: "text", text: `No matches for "${params.query}" in offloaded context.` }],
          details: { matches: 0 },
        };
      }
      const out = picked.map((b) => b.text).join("\n\n");
      return { content: [{ type: "text", text: out }], details: { matches: picked.length } };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: ctx_offload — explicitly offload content now
  // -------------------------------------------------------------------------
  pi.registerTool({
    name: "ctx_offload",
    label: "Offload Context",
    description:
      "Offload content out of the live conversation to reduce context size. " +
      "Two modes: (1) pass `content` to append a note/summary you no longer need in-context to the offload file; " +
      "(2) omit `content` to request that the oldest messages be offloaded (this happens automatically on the next turn when over budget).",
    parameters: Type.Object({
      content: Type.Optional(
        Type.String({ description: "Text to append to the offload file (e.g. a summary or detail you can drop from live context)." })
      ),
      count: Type.Optional(
        Type.Integer({
          minimum: 1,
          maximum: 200,
          description: "Number of oldest messages to offload (default: enough to drop below the limit).",
        })
      ),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      const usage = ctx.getContextUsage();
      const currentTokens = usage?.tokens ?? 0;
      const excess = currentTokens - state.limit;

      const parts: string[] = [];

      if (params.content && params.content.trim()) {
        const content = params.content.trim();
        const block = `\n## OFFLOAD ${nextOffloadId++} ${new Date().toISOString()} ${estimateTokens({ role: "user", content, timestamp: Date.now() } as AgentMessage)}\n${content}\n`;
        fs.mkdirSync(path.dirname(offloadPath(cwd)), { recursive: true });
        fs.appendFileSync(offloadPath(cwd), block, "utf8");
        parts.push(`Appended your note (${content.length} chars) to ${offloadPath(cwd)}.`);
      }

      parts.push(
        `Current usage: ${currentTokens} tokens, limit ${state.limit}.` +
          (excess > 0
            ? ` ${excess} tokens over budget — oldest messages will be auto-offloaded on the next turn.`
            : ` Within budget.`)
      );

      return {
        content: [{ type: "text", text: parts.join("\n") }],
        details: { currentTokens, limit: state.limit, excess },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Commands
  // -------------------------------------------------------------------------
  pi.registerCommand("ctxlimit", {
    description: `Set or view the conversation token limit (${MIN_LIMIT}–${MAX_LIMIT})`,
    getArgumentCompletions: (prefix: string) => {
      const presets = ["10000", "15000", "20000", "33000", "50000", "75000", "100000"];
      const p = (prefix ?? "").trim().toLowerCase();
      if (!p) return presets.map((v) => ({ value: v, label: v }));
      // If the typed text is already a complete, valid custom limit that is not
      // one of the presets, stop suggesting — otherwise Enter would replace the
      // user's custom number with the highlighted preset.
      const normalized = p.replace(/[\s_,]/g, "").replace(/k$/, "000");
      const n = Number(normalized);
      const isPreset = presets.includes(String(n));
      if (Number.isFinite(n) && n >= MIN_LIMIT && n <= MAX_LIMIT && !isPreset) return null;
      const filtered = presets.filter((v) => v.startsWith(p));
      return filtered.length > 0 ? filtered.map((v) => ({ value: v, label: v })) : null;
    },
    handler: async (args, ctx) => {
      const trimmed = args?.trim();
      if (!trimmed) {
        ctx.ui.notify(
          `Context limit: ${state.limit} tokens (allowed ${MIN_LIMIT}–${MAX_LIMIT}; type a custom number)`,
          "info"
        );
        return;
      }
      // Accept "10000", "10k", "100 k", "100_000" and similar.
      const normalized = trimmed.toLowerCase().replace(/[\s_,]/g, "").replace(/k$/, "000");
      const n = Number(normalized);
      if (!Number.isFinite(n) || n < MIN_LIMIT || n > MAX_LIMIT) {
        ctx.ui.notify(
          `Invalid limit "${trimmed}". Enter a number between ${MIN_LIMIT} and ${MAX_LIMIT} (e.g. 20000, 50k, 100000).`,
          "error"
        );
        return;
      }
      state.limit = Math.floor(n);
      saveState(cwd, state);
      ctx.ui.notify(`Context limit set to ${state.limit} tokens`, "info");
    },
  });

  pi.registerCommand("ctxstatus", {
    description: "Show context usage, limit, and offload stats",
    handler: async (_args, ctx) => {
      const usage = ctx.getContextUsage();
      const stats = readOffloadStats(cwd);
      const tokens = usage?.tokens ?? null;
      const percent = tokens != null ? Math.round((tokens / state.limit) * 100) : null;
      ctx.ui.notify(
        `Limit ${state.limit} | used ${tokens ?? "?"}${percent != null ? ` (${percent}%)` : ""} | ` +
          `offloaded ${stats.blocks} blocks (~${stats.tokens} tok)`,
        "info"
      );
    },
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isContentBearing(m: AgentMessage): boolean {
  const role = (m as { role?: string }).role;
  if (role === "toolResult") return true;
  if (role === "user" || role === "assistant") {
    const c = (m as { content?: unknown }).content;
    if (typeof c === "string") return c.trim().length > 0;
    if (Array.isArray(c)) return c.length > 0;
    return false;
  }
  // Custom agent messages (e.g. bash execution, custom) — let convertToLlm decide.
  return true;
}

function estimateTotal(messages: AgentMessage[]): number {
  return messages.reduce((sum, m) => sum + estimateTokens(m), 0);
}

interface OffloadBlock {
  id: number;
  tokens: number;
  text: string;
}

function readOffloadBlocks(cwd: string): OffloadBlock[] {
  try {
    const raw = fs.readFileSync(offloadPath(cwd), "utf8");
    const blocks: OffloadBlock[] = [];
    // Split on block markers.
    const parts = raw.split(/\n## OFFLOAD (\d+) [^\n]+ (\d+)\n/);
    // parts[0] is any preamble; then triples: [id, tokens, body, id, tokens, body...]
    for (let i = 1; i + 2 < parts.length; i += 3) {
      const id = Number(parts[i]);
      const tokens = Number(parts[i + 1]);
      const text = parts[i + 2].trim();
      if (text) blocks.push({ id, tokens, text });
    }
    return blocks;
  } catch {
    return [];
  }
}

function readOffloadStats(cwd: string): { blocks: number; tokens: number } {
  const blocks = readOffloadBlocks(cwd);
  return { blocks: blocks.length, tokens: blocks.reduce((s, b) => s + b.tokens, 0) };
}
