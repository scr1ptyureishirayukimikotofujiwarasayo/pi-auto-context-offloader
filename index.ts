/**
 * Auto Context Manager
 *
 * Keeps the live conversation under a user-configurable token budget by
 * offloading the oldest turns to a plain-text file on disk and replacing them
 * in-context with a compact, cache-stable marker.
 *
 * Cost-saving architecture:
 *   - Uses `context_with_system` + `getCurrentSystemMessage()` to budget the
 *     exact request (system prompt + tools + messages) without falling back to
 *     the previous turn's post-offload usage (eliminates full-history sawtooth
 *     re-sends).
 *   - Hysteresis + pinned cut point: cuts down to a low-water target (~55% of
 *     available conversation room) and freezes the exact `[head, marker]` prefix
 *     until the growing tail reaches the limit again. This preserves provider
 *     prompt caches across consecutive turns.
 *   - Provider-usage calibration: compares estimated request tokens against
 *     actual input tokens (`input + cacheRead + cacheWrite`) reported on
 *     `message_end`, absorbing tokenizer drift and `before_agent_start` system
 *     prompt additions.
 *   - Turn-boundary safety: never splits `assistant(toolCall)` from its
 *     `toolResult` messages.
 *   - Lossless chunked storage & bounded retrieval: saves full tool results in
 *     ~2k-token deduplicated blocks; `ctx_search` returns targeted snippets and
 *     `ctx_recall` paginates within a safe character cap.
 */

import {
  type ExtensionAPI,
  estimateTokens,
} from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { getCurrentSystemMessage, type SystemMessage } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

// ---------------------------------------------------------------------------
// Configuration / persistence
// ---------------------------------------------------------------------------

const DEFAULT_LIMIT = 33_000;
const MIN_LIMIT = 10;
const MAX_LIMIT = 100_000;

/** Target fraction of *available conversation budget* (`limit - head`) kept on a cut. */
const LOW_WATER_CONVERSATION_RATIO = 0.55;
/** Fallback tail fraction when system+tools alone exceed `limit`. */
const UNREACHABLE_TAIL_RATIO = 0.15;
/** Target size per offloaded block in `offload.log` so search/recall stay granular. */
const CHUNK_TARGET_TOKENS = 2_000;
/** Token reserve accounted for the compact offload marker message. */
const MARKER_RESERVE_TOKENS = 60;

interface State {
  limit: number;
}

function statePath(cwd: string): string {
  return path.join(cwd, ".pi", "ctx-manager.json");
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
    console.error("[ctx-manager] failed to save state:", err);
  }
}

// ---------------------------------------------------------------------------
// Message classification & lossless serialization
// ---------------------------------------------------------------------------

export function isCutPointMessage(message: AgentMessage): boolean {
  switch (message.role) {
    case "user":
    case "assistant":
    case "bashExecution":
    case "custom":
    case "branchSummary":
    case "compactionSummary":
      return true;
    case "toolResult":
    case "system":
      return false;
    default:
      return false;
  }
}

export function isTurnStartMessage(message: AgentMessage): boolean {
  switch (message.role) {
    case "user":
    case "bashExecution":
    case "custom":
    case "branchSummary":
    case "compactionSummary":
      return true;
    case "assistant":
    case "toolResult":
    case "system":
      return false;
    default:
      return false;
  }
}

export function isContentBearing(m: AgentMessage): boolean {
  if (m.role === "system") return false;
  if (m.role === "toolResult") return true;
  if (m.role === "user" || m.role === "assistant") {
    const c = m.content;
    if (typeof c === "string") return c.trim().length > 0;
    if (Array.isArray(c)) return c.length > 0;
    return false;
  }
  return true;
}

function extractTextAndImageSummary(
  content: string | ReadonlyArray<{ type: string; text?: string; mimeType?: string }>,
): string {
  if (typeof content === "string") return content;
  const parts: string[] = [];
  for (const block of content) {
    if (block.type === "text" && block.text) {
      parts.push(block.text);
    } else if (block.type === "image") {
      parts.push(`[image${block.mimeType ? `: ${block.mimeType}` : ""}]`);
    }
  }
  return parts.join("\n");
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "undefined";
  } catch {
    return "[unserializable]";
  }
}

/**
 * Lossless message serializer for `offload.log`. Unlike `serializeConversation`,
 * tool outputs are never truncated so exact file contents, errors, and IDs can
 * always be retrieved via `ctx_search` and `ctx_recall`.
 */
export function serializeMessagesLossless(messages: readonly AgentMessage[]): string {
  const parts: string[] = [];

  for (const msg of messages) {
    if (msg.role === "system") continue;

    if (msg.role === "user") {
      const text = extractTextAndImageSummary(msg.content);
      if (text) parts.push(`[User]: ${text}`);
      continue;
    }

    if (msg.role === "assistant") {
      const textParts: string[] = [];
      const thinkingParts: string[] = [];
      const toolCalls: string[] = [];

      for (const block of msg.content) {
        if (block.type === "text" && block.text) {
          textParts.push(block.text);
        } else if (block.type === "thinking" && block.thinking) {
          thinkingParts.push(block.thinking);
        } else if (block.type === "toolCall") {
          toolCalls.push(`${block.name}(${safeJson(block.arguments)})`);
        }
      }

      if (thinkingParts.length > 0) {
        parts.push(`[Assistant thinking]: ${thinkingParts.join("\n")}`);
      }
      if (textParts.length > 0) {
        parts.push(`[Assistant]: ${textParts.join("\n")}`);
      }
      if (toolCalls.length > 0) {
        parts.push(`[Assistant tool calls]: ${toolCalls.join("; ")}`);
      }
      continue;
    }

    if (msg.role === "toolResult") {
      const text = extractTextAndImageSummary(msg.content);
      const errTag = msg.isError ? ", ERROR" : "";
      parts.push(`[Tool result (${msg.toolName}${errTag})]:\n${text || "(empty)"}`);
      continue;
    }

    if (msg.role === "bashExecution") {
      const b = msg;
      const meta: string[] = [];
      if (b.exitCode !== undefined) meta.push(`exit ${b.exitCode}`);
      if (b.cancelled) meta.push("cancelled");
      const metaStr = meta.length > 0 ? ` (${meta.join(", ")})` : "";
      parts.push(`[Bash: ${b.command}${metaStr}]:\n${b.output}`);
      continue;
    }

    if (msg.role === "custom") {
      const c = msg;
      const text = extractTextAndImageSummary(c.content);
      if (text) parts.push(`[Custom (${c.customType})]: ${text}`);
      continue;
    }

    if (msg.role === "branchSummary") {
      parts.push(`[Branch summary]: ${msg.summary}`);
      continue;
    }

    if (msg.role === "compactionSummary") {
      parts.push(`[Compaction summary]: ${msg.summary}`);
      continue;
    }
  }

  return parts.join("\n\n");
}

/**
 * Split a sequence of messages into turn-aligned chunks of approximately
 * `targetTokens` so each block in `offload.log` is compact and searchable.
 */
export function chunkMessagesForOffload(
  messages: readonly AgentMessage[],
  targetTokens = CHUNK_TARGET_TOKENS,
): AgentMessage[][] {
  const contentOnly = messages.filter(isContentBearing);
  if (contentOnly.length === 0) return [];

  // Group into logical turns first (each starting at a turn-start message).
  const turns: AgentMessage[][] = [];
  let currentTurn: AgentMessage[] = [];
  for (const msg of contentOnly) {
    if (isTurnStartMessage(msg) && currentTurn.length > 0) {
      turns.push(currentTurn);
      currentTurn = [];
    }
    currentTurn.push(msg);
  }
  if (currentTurn.length > 0) turns.push(currentTurn);

  const chunks: AgentMessage[][] = [];
  let accChunk: AgentMessage[] = [];
  let accTokens = 0;

  const flushChunk = () => {
    if (accChunk.length > 0) {
      chunks.push(accChunk);
      accChunk = [];
      accTokens = 0;
    }
  };

  for (const turn of turns) {
    const turnTokens = estimateTotal(turn);
    if (accChunk.length > 0 && accTokens + turnTokens > targetTokens) {
      flushChunk();
    }

    if (turnTokens <= targetTokens) {
      accChunk.push(...turn);
      accTokens += turnTokens;
      continue;
    }

    // Single turn exceeds targetTokens: split at message boundaries inside the turn.
    flushChunk();
    for (const msg of turn) {
      const mt = estimateTokens(msg);
      if (accChunk.length > 0 && accTokens + mt > targetTokens) {
        flushChunk();
      }
      accChunk.push(msg);
      accTokens += mt;
    }
    flushChunk();
  }

  flushChunk();
  return chunks;
}

// ---------------------------------------------------------------------------
// Search snippet extraction & recall bounding
// ---------------------------------------------------------------------------

export interface OffloadBlock {
  id: number;
  tokens: number;
  text: string;
}

export function parseOffloadBlocks(raw: string): OffloadBlock[] {
  const blocks: OffloadBlock[] = [];
  const parts = raw.split(/(?:^|\r?\n)## OFFLOAD (\d+) [^\r\n]+ (\d+)\r?\n/);
  for (let i = 1; i + 2 < parts.length; i += 3) {
    const id = Number(parts[i]);
    const tokens = Number(parts[i + 1]);
    const text = parts[i + 2].trim();
    if (text) blocks.push({ id, tokens, text });
  }
  return blocks;
}

interface MatchWindow {
  start: number;
  end: number;
}

/**
 * Extract compact, merged excerpts around case-insensitive matches in `text`
 * instead of dumping an entire multi-kilobyte block into live context.
 */
export function extractSearchSnippets(
  text: string,
  query: string,
  maxMatchesPerBlock = 6,
  radiusChars = 220,
): { snippets: string[]; totalMatches: number } {
  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  if (!lowerQuery) return { snippets: [], totalMatches: 0 };

  const windows: MatchWindow[] = [];
  let totalMatches = 0;
  let pos = 0;

  while (pos < lowerText.length) {
    const idx = lowerText.indexOf(lowerQuery, pos);
    if (idx === -1) break;
    totalMatches++;
    if (windows.length < maxMatchesPerBlock) {
      let start = Math.max(0, idx - radiusChars);
      let end = Math.min(text.length, idx + query.length + radiusChars);

      // Snap outward to nearby newline if close so snippets read cleanly.
      const prevNl = text.lastIndexOf("\n", idx);
      if (prevNl !== -1 && idx - prevNl <= radiusChars) {
        const twoLinesUp = text.lastIndexOf("\n", Math.max(0, prevNl - 1));
        start = twoLinesUp !== -1 && idx - twoLinesUp <= radiusChars + 120 ? twoLinesUp + 1 : prevNl + 1;
      }
      const nextNl = text.indexOf("\n", idx + query.length);
      if (nextNl !== -1 && nextNl - (idx + query.length) <= radiusChars) {
        const twoLinesDown = text.indexOf("\n", nextNl + 1);
        end = twoLinesDown !== -1 && twoLinesDown - idx <= radiusChars + 160 ? twoLinesDown : nextNl;
      }

      windows.push({ start, end });
    }
    pos = idx + Math.max(1, lowerQuery.length);
  }

  if (windows.length === 0) return { snippets: [], totalMatches: 0 };

  // Merge overlapping or closely adjacent windows.
  const merged: MatchWindow[] = [windows[0]];
  for (let i = 1; i < windows.length; i++) {
    const prev = merged[merged.length - 1];
    const curr = windows[i];
    if (curr.start <= prev.end + 48) {
      prev.end = Math.max(prev.end, curr.end);
    } else {
      merged.push(curr);
    }
  }

  const snippets = merged.map((w) => {
    const prefix = w.start > 0 ? "..." : "";
    const suffix = w.end < text.length ? "..." : "";
    return `${prefix}${text.slice(w.start, w.end).trim()}${suffix}`;
  });

  return { snippets, totalMatches };
}

// ---------------------------------------------------------------------------
// Budget & cut-point selection
// ---------------------------------------------------------------------------

export function estimateTotal(messages: readonly AgentMessage[]): number {
  let sum = 0;
  for (const m of messages) sum += estimateTokens(m);
  return sum;
}

export function estimateRequestTokens(
  head: SystemMessage | undefined,
  marker: AgentMessage | undefined,
  tail: readonly AgentMessage[],
): number {
  let total = 0;
  if (head) total += estimateTokens(head as AgentMessage);
  if (marker) total += estimateTokens(marker);
  total += estimateTotal(tail);
  return total;
}

function getMessageTimestamp(msg: AgentMessage): number {
  const ts = (msg as { timestamp?: unknown }).timestamp;
  return typeof ts === "number" ? ts : 0;
}

function systemMessagesEqual(a: SystemMessage | undefined, b: SystemMessage | undefined): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return safeJson(a) === safeJson(b);
}

export interface CutOptions {
  messages: readonly AgentMessage[];
  limit: number;
  calibrationRatio: number;
  minCutIndex?: number;
  forceCut?: boolean;
}

/**
 * Choose a valid cut point in `messages` (`1 <= cut < messages.length`) that brings
 * the retained tail down to the low-water target while never cutting before a
 * `toolResult` (so `assistant(toolCall)` + its `toolResult`s stay together).
 *
 * Returns `null` if no valid content-bearing prefix can be offloaded.
 */
export function selectCutPoint(opts: CutOptions): number | null {
  const { messages, limit, calibrationRatio, minCutIndex = 0 } = opts;
  if (messages.length <= 1) return null;

  const validCuts: number[] = [];
  for (let i = 1; i < messages.length; i++) {
    if (isCutPointMessage(messages[i])) {
      validCuts.push(i);
    }
  }
  if (validCuts.length === 0) return null;

  const fullHead = getCurrentSystemMessage(messages);
  const rawHeadTokens = fullHead ? estimateTokens(fullHead as AgentMessage) : 0;
  const effectiveHead = Math.round(rawHeadTokens * calibrationRatio);

  const conversationBudget = Math.max(0, limit - effectiveHead);
  const targetEffectiveConv =
    conversationBudget <= 0
      ? Math.floor(limit * UNREACHABLE_TAIL_RATIO)
      : Math.floor(conversationBudget * LOW_WATER_CONVERSATION_RATIO);

  const rawTailBudget = Math.max(0, Math.floor(targetEffectiveConv / calibrationRatio) - MARKER_RESERVE_TOKENS);

  // Walk backwards from newest to oldest to find earliest index fitting `rawTailBudget`.
  let acc = 0;
  let desiredFrom = messages.length - 1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const t = estimateTokens(messages[i]);
    if (acc + t > rawTailBudget && i < messages.length - 1) {
      break;
    }
    acc += t;
    desiredFrom = i;
  }

  let cut = validCuts.find((c) => c >= desiredFrom) ?? validCuts[validCuts.length - 1];

  if (minCutIndex > 0) {
    const floored = validCuts.find((c) => c >= minCutIndex) ?? validCuts[validCuts.length - 1];
    if (floored > cut) cut = floored;
  }

  // Ensure the prefix actually contains offloadable conversation messages.
  const hasOffloadableContent = messages.slice(0, cut).some(isContentBearing);
  if (!hasOffloadableContent) return null;

  return cut;
}

// ---------------------------------------------------------------------------
// Hashing / offload file helpers
// ---------------------------------------------------------------------------

function hashText(text: string): string {
  return crypto.createHash("sha1").update(text).digest("hex");
}

interface PinnedCut {
  index: number;
  anchorRole: string;
  anchorTimestamp: number;
  head: SystemMessage | undefined;
  marker: AgentMessage;
}

interface PendingAudit {
  estimatedTokens: number;
  pinnedHit: boolean;
}

interface RequestStats {
  overhead: number;
  request: number;
  calibratedRequest: number;
  session: number;
  kept: number;
  cachePinned: boolean;
  pinnedHits: number;
}

interface UsageStats {
  lastInputTokens: number;
  lastCacheReadTokens: number;
  totalCacheReadTokens: number;
  totalInputTokens: number;
  totalSavedTokensEstimate: number;
}

type ManualOffloadRequest =
  | { mode: "auto" }
  | { mode: "count"; count: number };

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  let cwd = process.cwd();
  let state: State = { limit: DEFAULT_LIMIT };
  let currentOffloadFile = "";
  let nextOffloadId = 1;
  let offloadIndex = new Map<string, number>();

  // Session-local runtime state for hysteresis, cache pinning, and calibration.
  let offloadedUpTo = 0;
  let pinnedCut: PinnedCut | null = null;
  let pinnedHitCount = 0;
  let calibrationRatio = 1.0;
  let pendingAudit: PendingAudit | null = null;
  let pendingManualOffload: ManualOffloadRequest | null = null;
  let lastRequest: RequestStats | null = null;
  let usageStats: UsageStats = {
    lastInputTokens: 0,
    lastCacheReadTokens: 0,
    totalCacheReadTokens: 0,
    totalInputTokens: 0,
    totalSavedTokensEstimate: 0,
  };

  function getOffloadFile(dir: string): string {
    return currentOffloadFile || path.join(dir, ".pi", "offload", "default", "offload.log");
  }

  function readOffloadBlocks(dir: string): OffloadBlock[] {
    try {
      const raw = fs.readFileSync(getOffloadFile(dir), "utf8");
      return parseOffloadBlocks(raw);
    } catch {
      return [];
    }
  }

  function readOffloadStats(dir: string): { blocks: number; tokens: number } {
    const blocks = readOffloadBlocks(dir);
    return { blocks: blocks.length, tokens: blocks.reduce((s, b) => s + b.tokens, 0) };
  }

  function appendSerializedBlock(dir: string, text: string, tokens: number): { id: number; appended: boolean } {
    const key = hashText(text.trim());
    const existing = offloadIndex.get(key);
    if (existing !== undefined) {
      return { id: existing, appended: false };
    }
    const file = getOffloadFile(dir);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const id = nextOffloadId++;
    const ts = new Date().toISOString();
    const block = `\n## OFFLOAD ${id} ${ts} ${tokens}\n${text.trim()}\n`;
    fs.appendFileSync(file, block, "utf8");
    offloadIndex.set(key, id);
    return { id, appended: true };
  }

  function rebuildOffloadIndex(dir: string): void {
    nextOffloadId = 1;
    offloadIndex = new Map();
    for (const b of readOffloadBlocks(dir)) {
      offloadIndex.set(hashText(b.text.trim()), b.id);
      if (b.id >= nextOffloadId) nextOffloadId = b.id + 1;
    }
  }

  function resetRuntimeCutState(): void {
    offloadedUpTo = 0;
    pinnedCut = null;
    pinnedHitCount = 0;
    calibrationRatio = 1.0;
    pendingAudit = null;
    pendingManualOffload = null;
    lastRequest = null;
  }

  // -------------------------------------------------------------------------
  // Lifecycle hooks
  // -------------------------------------------------------------------------

  pi.on("session_start", async (_event, ctx) => {
    cwd = ctx.cwd;
    state = loadState(cwd);
    let sid = "default";
    try {
      sid = ctx.sessionManager.getSessionId();
    } catch {
      /* ephemeral session */
    }
    const safe = String(sid).replace(/[^A-Za-z0-9._-]/g, "-");
    currentOffloadFile = path.join(cwd, ".pi", "offload", safe, "offload.log");
    rebuildOffloadIndex(cwd);
    resetRuntimeCutState();
    usageStats = {
      lastInputTokens: 0,
      lastCacheReadTokens: 0,
      totalCacheReadTokens: 0,
      totalInputTokens: 0,
      totalSavedTokensEstimate: 0,
    };
  });

  pi.on("session_tree", async () => {
    resetRuntimeCutState();
  });

  pi.on("session_compact", async () => {
    resetRuntimeCutState();
  });

  pi.on("model_select", async () => {
    calibrationRatio = 1.0;
    pendingAudit = null;
  });

  // Calibrate token estimator & track actual cache efficiency from provider usage.
  pi.on("message_end", async (event) => {
    const msg = event.message;
    if (msg.role !== "assistant") return;
    if (msg.stopReason === "aborted" || msg.stopReason === "error") return;

    const usage = msg.usage;
    if (!usage) return;
    const actualInput = usage.input + usage.cacheRead + usage.cacheWrite;
    if (actualInput <= 0) return;

    usageStats.lastInputTokens = actualInput;
    usageStats.lastCacheReadTokens = usage.cacheRead;
    usageStats.totalCacheReadTokens += usage.cacheRead;
    usageStats.totalInputTokens += actualInput;

    if (lastRequest && lastRequest.session > lastRequest.request) {
      usageStats.totalSavedTokensEstimate += lastRequest.session - lastRequest.request;
    }

    if (pendingAudit && pendingAudit.estimatedTokens > 0) {
      const sample = actualInput / pendingAudit.estimatedTokens;
      const clamped = Math.min(1.8, Math.max(0.85, sample));
      calibrationRatio = calibrationRatio === 1.0 ? clamped : calibrationRatio * 0.6 + clamped * 0.4;
      pendingAudit = null;
    }
  });

  // -------------------------------------------------------------------------
  // Core budget enforcement: `context_with_system`
  // -------------------------------------------------------------------------
  pi.on("context_with_system", async (event) => {
    const messages = event.messages;
    if (messages.length === 0) return;

    const rawSessionTokens = estimateTotal(messages);
    const fullHead = getCurrentSystemMessage(messages);
    const fullHeadTokens = fullHead ? estimateTokens(fullHead as AgentMessage) : 0;

    // 1. Validate existing pinned cut (if any).
    if (pinnedCut !== null) {
      const anchor = messages[pinnedCut.index];
      const anchorValid =
        pinnedCut.index > 0 &&
        pinnedCut.index < messages.length &&
        anchor !== undefined &&
        anchor.role === pinnedCut.anchorRole &&
        getMessageTimestamp(anchor) === pinnedCut.anchorTimestamp &&
        isCutPointMessage(anchor);

      if (!anchorValid) {
        pinnedCut = null;
        pinnedHitCount = 0;
        offloadedUpTo = 0;
      }
    }

    // 2. Fast cache-preserving path when pinnedCut is active and still under budget.
    if (pinnedCut !== null && pendingManualOffload === null) {
      const currentHead = getCurrentSystemMessage(messages.slice(0, pinnedCut.index));
      if (!systemMessagesEqual(pinnedCut.head, currentHead)) {
        pinnedCut.head = currentHead;
      }

      const tail = messages.slice(pinnedCut.index);
      const rawReq = estimateRequestTokens(pinnedCut.head, pinnedCut.marker, tail);
      const calibratedReq = Math.round(rawReq * calibrationRatio);

      if (calibratedReq <= state.limit) {
        pinnedHitCount++;
        const headTok = pinnedCut.head ? estimateTokens(pinnedCut.head as AgentMessage) : 0;
        lastRequest = {
          overhead: headTok,
          request: rawReq,
          calibratedRequest: calibratedReq,
          session: rawSessionTokens,
          kept: rawReq - headTok,
          cachePinned: true,
          pinnedHits: pinnedHitCount,
        };
        pendingAudit = { estimatedTokens: rawReq, pinnedHit: true };

        const out = pinnedCut.head
          ? [pinnedCut.head as AgentMessage, pinnedCut.marker, ...tail]
          : [pinnedCut.marker, ...tail];
        return { messages: out };
      }
    }

    // 3. Pass-through path when never cut yet and full transcript fits in budget.
    if (pinnedCut === null && pendingManualOffload === null) {
      const calibratedSession = Math.round(rawSessionTokens * calibrationRatio);
      if (calibratedSession <= state.limit) {
        lastRequest = {
          overhead: fullHeadTokens,
          request: rawSessionTokens,
          calibratedRequest: calibratedSession,
          session: rawSessionTokens,
          kept: Math.max(0, rawSessionTokens - fullHeadTokens),
          cachePinned: false,
          pinnedHits: 0,
        };
        pendingAudit = { estimatedTokens: rawSessionTokens, pinnedHit: false };
        return;
      }
    }

    // 4. Compute a new cut point down to the low-water target.
    let minCutIndex = pinnedCut?.index ?? 0;
    if (pendingManualOffload?.mode === "count") {
      minCutIndex = (pinnedCut?.index ?? 0) + pendingManualOffload.count;
    } else if (pinnedCut !== null && pendingManualOffload === null) {
      // Previous pinnedCut exceeded limit: ensure we advance at least 1 valid cut if possible.
      minCutIndex = pinnedCut.index + 1;
    }
    const manualRequested = pendingManualOffload !== null;
    pendingManualOffload = null;

    const cut = selectCutPoint({
      messages,
      limit: state.limit,
      calibrationRatio,
      minCutIndex,
      forceCut: manualRequested,
    });

    if (cut === null || (pinnedCut !== null && cut === pinnedCut.index)) {
      // Cannot advance cut point further without splitting active toolCall/toolResult pairs.
      if (pinnedCut !== null) {
        pinnedHitCount++;
        const tail = messages.slice(pinnedCut.index);
        const rawReq = estimateRequestTokens(pinnedCut.head, pinnedCut.marker, tail);
        const headTok = pinnedCut.head ? estimateTokens(pinnedCut.head as AgentMessage) : 0;
        lastRequest = {
          overhead: headTok,
          request: rawReq,
          calibratedRequest: Math.round(rawReq * calibrationRatio),
          session: rawSessionTokens,
          kept: rawReq - headTok,
          cachePinned: true,
          pinnedHits: pinnedHitCount,
        };
        pendingAudit = { estimatedTokens: rawReq, pinnedHit: true };
        const out = pinnedCut.head
          ? [pinnedCut.head as AgentMessage, pinnedCut.marker, ...tail]
          : [pinnedCut.marker, ...tail];
        return { messages: out };
      }

      lastRequest = {
        overhead: fullHeadTokens,
        request: rawSessionTokens,
        calibratedRequest: Math.round(rawSessionTokens * calibrationRatio),
        session: rawSessionTokens,
        kept: Math.max(0, rawSessionTokens - fullHeadTokens),
        cachePinned: false,
        pinnedHits: 0,
      };
      pendingAudit = { estimatedTokens: rawSessionTokens, pinnedHit: false };
      return;
    }

    // 5. Offload only the newly dropped prefix in deterministic, turn-aligned chunks.
    const scanFrom = offloadedUpTo > 0 && offloadedUpTo <= cut ? offloadedUpTo : 0;
    if (cut > scanFrom) {
      const delta = messages.slice(scanFrom, cut);
      const chunks = chunkMessagesForOffload(delta, CHUNK_TARGET_TOKENS);
      for (const chunk of chunks) {
        const serialized = serializeMessagesLossless(chunk);
        if (serialized.trim()) {
          appendSerializedBlock(cwd, serialized, estimateTotal(chunk));
        }
      }
    }
    offloadedUpTo = cut;

    // 6. Freeze the new `[head, marker]` prefix so subsequent turns hit provider cache.
    const head = getCurrentSystemMessage(messages.slice(0, cut));
    const stats = readOffloadStats(cwd);
    const anchor = messages[cut];
    const marker: AgentMessage = {
      role: "user",
      content: [
        {
          type: "text",
          text:
            `[Earlier conversation offloaded to ${path.basename(getOffloadFile(cwd))} ` +
            `(${stats.blocks} blocks, ~${stats.tokens} tokens) to stay within your ${state.limit}-token budget. ` +
            `Use ctx_search(query) for targeted excerpts or ctx_recall(blockId/count/offset) to read blocks back.]`,
        },
      ],
      timestamp: getMessageTimestamp(anchor),
    } as AgentMessage;

    pinnedCut = {
      index: cut,
      anchorRole: anchor.role,
      anchorTimestamp: getMessageTimestamp(anchor),
      head,
      marker,
    };
    pinnedHitCount = 0;

    const tail = messages.slice(cut);
    const rawReq = estimateRequestTokens(head, marker, tail);
    const headTok = head ? estimateTokens(head as AgentMessage) : 0;

    lastRequest = {
      overhead: headTok,
      request: rawReq,
      calibratedRequest: Math.round(rawReq * calibrationRatio),
      session: rawSessionTokens,
      kept: rawReq - headTok,
      cachePinned: true,
      pinnedHits: 0,
    };
    pendingAudit = { estimatedTokens: rawReq, pinnedHit: false };

    const out = head ? [head as AgentMessage, marker, ...tail] : [marker, ...tail];
    return { messages: out };
  });

  // -------------------------------------------------------------------------
  // Tool: ctx_status — report budget usage, cache stability, and offload stats
  // -------------------------------------------------------------------------
  pi.registerTool({
    name: "ctx_status",
    label: "Context Status",
    description:
      "Report the current context usage, the configured token limit, cache stability, and how much has been offloaded to the context file.",
    parameters: Type.Object({}),
    async execute(_id, _params, _signal, _onUpdate, ctx) {
      const usage = ctx.getContextUsage();
      const offloaded = readOffloadStats(cwd);
      const tokens = lastRequest?.session ?? usage?.tokens ?? null;
      const percent = tokens != null ? Math.round((tokens / state.limit) * 100) : null;
      const lines: string[] = [
        `Context limit: ${state.limit} tokens`,
        `Full session size: ~${tokens ?? "unknown"} tokens${percent != null ? ` (${percent}% of limit)` : ""}`,
      ];
      if (lastRequest) {
        const reqPct = Math.round((lastRequest.calibratedRequest / state.limit) * 100);
        lines.push(
          `Last request sent: ~${lastRequest.calibratedRequest} effective tokens (${reqPct}% of limit) ` +
            `[raw ~${lastRequest.request} = ~${lastRequest.overhead} system+tools + ~${lastRequest.kept} tail` +
            `${calibrationRatio !== 1 ? `, calibration ${calibrationRatio.toFixed(2)}x` : ""}]`,
        );
        if (lastRequest.cachePinned) {
          lines.push(`Prompt cache prefix: pinned (${lastRequest.pinnedHits} consecutive cache-preserving call(s))`);
        }
        if (Math.round(lastRequest.overhead * calibrationRatio) >= state.limit) {
          lines.push(
            `WARNING: system+tools alone (~${Math.round(lastRequest.overhead * calibrationRatio)}) exceed the limit; ` +
              `raise it (e.g. /ctxlimit ${Math.round(lastRequest.overhead * calibrationRatio) + 20000}).`,
          );
        }
      }
      if (usageStats.lastInputTokens > 0) {
        const cachePct = Math.round((usageStats.lastCacheReadTokens / usageStats.lastInputTokens) * 100);
        lines.push(
          `Last provider usage: ${usageStats.lastInputTokens} input tokens (${usageStats.lastCacheReadTokens} cached, ${cachePct}% cache hit)`,
        );
      }
      if (usageStats.totalSavedTokensEstimate > 0) {
        lines.push(`Estimated input tokens saved by offloading so far: ~${usageStats.totalSavedTokensEstimate}`);
      }
      lines.push(`Offloaded: ${offloaded.blocks} blocks, ~${offloaded.tokens} tokens in ${getOffloadFile(cwd)}`);
      const text = lines.join("\n");
      return {
        content: [{ type: "text", text }],
        details: {
          usageTokens: tokens,
          percent,
          offloadedBlocks: offloaded.blocks,
          offloadedTokens: offloaded.tokens,
          lastRequestTokens: lastRequest?.calibratedRequest ?? null,
          overheadTokens: lastRequest?.overhead ?? null,
          calibrationRatio,
          pinnedHits: lastRequest?.pinnedHits ?? 0,
        },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: ctx_recall — paginated / targeted block recall
  // -------------------------------------------------------------------------
  pi.registerTool({
    name: "ctx_recall",
    label: "Recall Context",
    description:
      "Read back conversation content that was previously offloaded to the context file. " +
      "Supports recalling the latest blocks (`count`), filtering (`query`), inspecting a specific block (`blockId`), " +
      "and paginating large results (`offset`) without blowing the context budget.",
    parameters: Type.Object({
      count: Type.Optional(
        Type.Integer({
          minimum: 1,
          maximum: 50,
          description: "Number of offload blocks to return (default 1 when blockId is not set).",
        }),
      ),
      blockId: Type.Optional(
        Type.Integer({
          minimum: 1,
          description: "Specific OFFLOAD block ID to read (e.g. discovered via ctx_search).",
        }),
      ),
      query: Type.Optional(
        Type.String({ description: "Optional substring to filter blocks by (case-insensitive)." }),
      ),
      order: Type.Optional(
        Type.Union([Type.Literal("newest"), Type.Literal("oldest")], {
          description: "Block ordering when selecting by count (default 'newest').",
        }),
      ),
      offset: Type.Optional(
        Type.Integer({
          minimum: 0,
          description: "Character offset into the formatted output for reading large blocks in pages (default 0).",
        }),
      ),
    }),
    async execute(_id, params, _signal, _onUpdate, _ctx) {
      const blocks = readOffloadBlocks(cwd);
      let picked: OffloadBlock[];

      if (params.blockId !== undefined) {
        const found = blocks.find((b) => b.id === params.blockId);
        picked = found ? [found] : [];
      } else {
        let selected = blocks;
        if (params.query) {
          const q = params.query.toLowerCase();
          selected = blocks.filter((b) => b.text.toLowerCase().includes(q));
        }
        const count = params.count ?? 1;
        picked =
          params.order === "oldest"
            ? selected.slice(0, count)
            : selected.slice(-count).reverse();
      }

      if (picked.length === 0) {
        return {
          content: [
            {
              type: "text",
              text:
                params.blockId !== undefined
                  ? `Offload block #${params.blockId} not found.`
                  : "No offloaded content found" + (params.query ? " matching that query" : "") + ".",
            },
          ],
          details: { blocks: 0 },
        };
      }

      const fullOut = picked
        .map((b) => `=== OFFLOAD #${b.id} (~${b.tokens} tokens) ===\n${b.text}`)
        .join("\n\n");

      const maxChars = Math.min(24_000, Math.max(4_000, Math.floor(state.limit * 0.35) * 4));
      const offset = Math.min(params.offset ?? 0, fullOut.length);
      const end = Math.min(fullOut.length, offset + maxChars);
      let text = fullOut.slice(offset, end);

      if (offset > 0 || end < fullOut.length) {
        text += `\n\n[Showing chars ${offset}-${end} of ${fullOut.length}. Pass offset=${end} to read more.]`;
      }

      return {
        content: [{ type: "text", text }],
        details: { blocks: picked.length, totalChars: fullOut.length, offset, end },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: ctx_search — snippet-based keyword search across offloaded blocks
  // -------------------------------------------------------------------------
  pi.registerTool({
    name: "ctx_search",
    label: "Search Context",
    description:
      "Search the offloaded conversation history for a keyword or phrase. Returns compact matching excerpts " +
      "with block IDs instead of dumping entire blocks; use ctx_recall(blockId=...) if you need the full block.",
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

      const maxChars = Math.min(12_000, Math.max(3_000, Math.floor(state.limit * 0.25) * 4));
      const sections: string[] = [];
      let totalChars = 0;
      let truncated = false;

      for (const b of picked) {
        const { snippets, totalMatches } = extractSearchSnippets(b.text, params.query);
        const section =
          `--- Block #${b.id} (~${b.tokens} tokens, ${totalMatches} match${totalMatches === 1 ? "" : "es"}) ---\n` +
          snippets.join("\n...\n");

        if (sections.length > 0 && totalChars + section.length > maxChars) {
          truncated = true;
          break;
        }
        const clipped =
          section.length > maxChars
            ? section.slice(0, maxChars) + "\n... [excerpt truncated]"
            : section;
        sections.push(clipped);
        totalChars += clipped.length;
      }

      const footer =
        `\n\n[${matches.length} block(s) matched` +
        (truncated ? `, showing top ${sections.length} within budget` : "") +
        `. Use ctx_recall({ blockId: <id> }) to read a full block.]`;

      return {
        content: [{ type: "text", text: sections.join("\n\n") + footer }],
        details: { matches: matches.length, shownBlocks: sections.length },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: ctx_offload — explicitly offload content or advance cut point
  // -------------------------------------------------------------------------
  pi.registerTool({
    name: "ctx_offload",
    label: "Offload Context",
    description:
      "Offload content out of the live conversation to reduce context size. " +
      "Two modes: (1) pass `content` to append a note/summary to the offload file; " +
      "(2) omit `content` (optionally pass `count`) to trim the oldest messages down to the low-water target on the next LLM call.",
    parameters: Type.Object({
      content: Type.Optional(
        Type.String({ description: "Text to append to the offload file (e.g. a summary or detail you can drop from live context)." }),
      ),
      count: Type.Optional(
        Type.Integer({
          minimum: 1,
          maximum: 200,
          description: "Minimum number of oldest messages to offload on the next LLM call.",
        }),
      ),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      const usage = ctx.getContextUsage();
      const currentTokens = lastRequest?.calibratedRequest ?? usage?.tokens ?? 0;
      const parts: string[] = [];

      if (params.content && params.content.trim()) {
        const content = params.content.trim();
        const est = estimateTokens({ role: "user", content, timestamp: Date.now() } as AgentMessage);
        const res = appendSerializedBlock(cwd, content, est);
        parts.push(
          res.appended
            ? `Saved note as block #${res.id} (${content.length} chars) to ${getOffloadFile(cwd)}.`
            : `Note already present as block #${res.id} in ${getOffloadFile(cwd)}.`,
        );
      }

      if (!params.content || params.count !== undefined) {
        pendingManualOffload =
          params.count !== undefined ? { mode: "count", count: params.count } : { mode: "auto" };
        parts.push(
          params.count !== undefined
            ? `Scheduled offload of at least ${params.count} oldest messages on the next LLM call.`
            : `Scheduled trim down to low-water target on the next LLM call.`,
        );
      }

      parts.push(`Effective request usage: ~${currentTokens} / ${state.limit} tokens.`);

      return {
        content: [{ type: "text", text: parts.join("\n") }],
        details: { currentTokens, limit: state.limit, scheduled: pendingManualOffload !== null },
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
          "info",
        );
        return;
      }
      const normalized = trimmed.toLowerCase().replace(/[\s_,]/g, "").replace(/k$/, "000");
      const n = Number(normalized);
      if (!Number.isFinite(n) || n < MIN_LIMIT || n > MAX_LIMIT) {
        ctx.ui.notify(
          `Invalid limit "${trimmed}". Enter a number between ${MIN_LIMIT} and ${MAX_LIMIT} (e.g. 20000, 50k, 100000).`,
          "error",
        );
        return;
      }
      state.limit = Math.floor(n);
      saveState(cwd, state);
      ctx.ui.notify(`Context limit set to ${state.limit} tokens`, "info");
    },
  });

  pi.registerCommand("ctxstatus", {
    description: "Show context usage, limit, cache status, and offload stats",
    handler: async (_args, ctx) => {
      const usage = ctx.getContextUsage();
      const stats = readOffloadStats(cwd);
      const tokens = lastRequest?.calibratedRequest ?? usage?.tokens ?? null;
      const percent = tokens != null ? Math.round((tokens / state.limit) * 100) : null;
      const cacheInfo =
        lastRequest?.cachePinned ? ` | cache pinned (${lastRequest.pinnedHits} hits)` : "";
      ctx.ui.notify(
        `Limit ${state.limit} | req ~${tokens ?? "?"}${percent != null ? ` (${percent}%)` : ""}${cacheInfo} | ` +
          `offloaded ${stats.blocks} blocks (~${stats.tokens} tok)`,
        "info",
      );
    },
  });

  pi.registerCommand("clearcontext", {
    description: "Delete offloaded context files (this conversation, or `all`)",
    getArgumentCompletions: (prefix: string) => {
      const items = [
        { value: "all", label: "all — every conversation" },
        { value: "current", label: "current — this conversation only" },
      ];
      const p = (prefix ?? "").trim().toLowerCase();
      if (!p) return items;
      const f = items.filter((i) => i.value.startsWith(p));
      return f.length > 0 ? f : null;
    },
    handler: async (args, ctx) => {
      const scope = (args ?? "").trim().toLowerCase();
      const root = path.join(cwd, ".pi", "offload");
      const resetAll = () => {
        offloadIndex = new Map();
        nextOffloadId = 1;
        resetRuntimeCutState();
      };

      if (scope === "all") {
        let removed = 0;
        try {
          for (const name of fs.readdirSync(root)) {
            const p = path.join(root, name);
            if (fs.statSync(p).isDirectory()) {
              fs.rmSync(p, { recursive: true, force: true });
              removed++;
            }
          }
        } catch {
          /* no offload dir yet */
        }
        resetAll();
        ctx.ui.notify(`Cleared ${removed} conversation offload folder(s).`, "info");
        return;
      }

      const dir = path.dirname(getOffloadFile(cwd));
      let existed = false;
      try {
        existed = fs.existsSync(dir);
        fs.rmSync(dir, { recursive: true, force: true });
      } catch {
        /* ignore */
      }
      resetAll();
      ctx.ui.notify(
        existed
          ? `Cleared offloaded context for this conversation (${path.basename(dir)}).`
          : `No offloaded context to clear for this conversation (${path.basename(dir)}).`,
        "info",
      );
    },
  });
}
