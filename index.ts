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
 *   /ctxlimit <n>        set the token limit (any time), or print current value
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

interface State {
  limit: number;
}

function statePath(cwd: string): string {
  // Per-project state so each project has its own offload file and limit.
  return path.join(cwd, ".pi", "ctx-manager.json");
}

function offloadPath(cwd: string): string {
  return path.join(cwd, ".pi", "ctx-offload.log");
}

function loadState(cwd: string): State {
  try {
    const raw = fs.readFileSync(statePath(cwd), "utf8");
    const parsed = JSON.parse(raw) as Partial<State>;
    return {
      limit: typeof parsed.limit === "number" && parsed.limit > 0 ? parsed.limit : DEFAULT_LIMIT,
    };
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

function appendOffload(cwd: string, messages: AgentMessage[], tokens: number): { id: number; start: number } {
  fs.mkdirSync(path.dirname(offloadPath(cwd)), { recursive: true });
  const id = nextOffloadId++;
  const text = serializeConversation(convertToLlm(messages));
  const ts = new Date().toISOString();
  const block = `\n## OFFLOAD ${id} ${ts} ${tokens}\n${text}\n`;
  fs.appendFileSync(offloadPath(cwd), block, "utf8");
  return { id, start: block.length };
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  let cwd = process.cwd();
  let state: State = { limit: DEFAULT_LIMIT };

  pi.on("session_start", async (_event, ctx) => {
    cwd = ctx.cwd;
    state = loadState(cwd);
    nextOffloadId = 1;
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
      return; // within budget, nothing to do
    }

    // Decide how many leading messages to offload. We keep the most recent
    // messages that fit within the budget (minus a safety margin), preserving
    // the tail of the conversation so the model stays coherent.
    const margin = Math.floor(state.limit * 0.1);
    const keepBudget = state.limit - margin;

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

    const toOffload = messages.slice(0, keepFrom);
    if (toOffload.length === 0) {
      return;
    }

    const offloadTokens = toOffload.reduce((sum, m) => sum + estimateTokens(m), 0);

    // Only offload content-bearing messages (skip empty/custom-only noise).
    const contentMessages = toOffload.filter(isContentBearing);

    // Write to disk (always, even if nothing content-bearing, to record the drop).
    appendOffload(cwd, contentMessages, offloadTokens);

    // Build a compact marker message that stands in for the offloaded history.
    const marker: Message = {
      role: "user",
      content: [
        {
          type: "text",
          text:
            `[Earlier conversation offloaded to context file to stay within your ` +
            `${state.limit}-token budget. ${contentMessages.length} messages (~${offloadTokens} tokens) ` +
            `were saved. Use ctx_recall to read them back, or ctx_search to find specific details.]`,
        },
      ],
      timestamp: Date.now(),
    };

    // Replace the offloaded prefix with the marker.
    const newMessages: AgentMessage[] = [marker as AgentMessage, ...messages.slice(keepFrom)];

    return { messages: newMessages };
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
      const text =
        `Context limit: ${state.limit} tokens\n` +
        `Current usage: ${tokens ?? "unknown"} tokens${percent != null ? ` (${percent}% of limit)` : ""}\n` +
        `Offloaded blocks: ${offloaded.blocks}, ~${offloaded.tokens} tokens in ${offloadPath(cwd)}`;
      return { content: [{ type: "text", text }], details: { usageTokens: tokens, percent, offloadedBlocks: offloaded.blocks, offloadedTokens: offloaded.tokens } };
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
    description: "Set or view the conversation token limit",
    getArgumentCompletions: () => [
      { value: "10000", label: "10000" },
      { value: "20000", label: "20000" },
      { value: "33000", label: "33000" },
      { value: "50000", label: "50000" },
      { value: "100000", label: "100000" },
    ],
    handler: async (args, ctx) => {
      const trimmed = args?.trim();
      if (!trimmed) {
        ctx.ui.notify(`Context limit: ${state.limit} tokens`, "info");
        return;
      }
      const n = Number(trimmed);
      if (!Number.isFinite(n) || n < 1000) {
        ctx.ui.notify(`Invalid limit "${trimmed}". Use a number >= 1000.`, "error");
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
