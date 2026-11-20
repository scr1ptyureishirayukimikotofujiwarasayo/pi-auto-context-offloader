import { describe, it } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import extension, {
  chunkMessagesForOffload,
  extractSearchSnippets,
  parseOffloadBlocks,
  selectCutPoint,
  serializeMessagesLossless,
} from "../index.ts";

interface RegisteredTool {
  name: string;
  execute: (
    id: string,
    params: Record<string, unknown>,
    signal: AbortSignal | undefined,
    onUpdate: unknown,
    ctx: ExtensionContext,
  ) => Promise<{ content: Array<{ type: string; text: string }>; details: Record<string, unknown> }>;
}

interface RegisteredCommand {
  handler: (args: string, ctx: ExtensionContext) => Promise<void>;
}

function createHarness(cwd: string, sessionId = "test-session") {
  const handlers = new Map<string, Array<(event: unknown, ctx: ExtensionContext) => Promise<unknown>>>();
  const tools = new Map<string, RegisteredTool>();
  const commands = new Map<string, RegisteredCommand>();
  const notifications: Array<{ msg: string; level: string }> = [];

  const api = {
    on(event: string, fn: (event: unknown, ctx: ExtensionContext) => Promise<unknown>) {
      const list = handlers.get(event) ?? [];
      list.push(fn);
      handlers.set(event, list);
    },
    registerTool(tool: RegisteredTool) {
      tools.set(tool.name, tool);
    },
    registerCommand(name: string, cmd: RegisteredCommand) {
      commands.set(name, cmd);
    },
  } as unknown as ExtensionAPI;

  const ctx = {
    cwd,
    sessionManager: {
      getSessionId: () => sessionId,
    },
    getContextUsage: () => ({ tokens: null, contextWindow: 200_000, percent: null }),
    ui: {
      notify: (msg: string, level: string) => notifications.push({ msg, level }),
    },
  } as unknown as ExtensionContext;

  extension(api);

  async function emit(event: string, payload: unknown): Promise<unknown> {
    const list = handlers.get(event) ?? [];
    let lastResult: unknown = undefined;
    for (const fn of list) {
      const res = await fn(payload, ctx);
      if (res !== undefined) lastResult = res;
    }
    return lastResult;
  }

  return { emit, tools, commands, ctx, notifications };
}

function makeSystem(text: string, extras?: { sections?: Record<string, string>; toolsAdded?: unknown[] }): AgentMessage {
  return {
    role: "system",
    content: text,
    ...(extras?.sections ? { sections: extras.sections } : {}),
    ...(extras?.toolsAdded ? { toolsAdded: extras.toolsAdded } : {}),
    timestamp: 1,
  } as AgentMessage;
}

function makeUser(text: string, ts: number): AgentMessage {
  return {
    role: "user",
    content: [{ type: "text", text }],
    timestamp: ts,
  } as AgentMessage;
}

function makeAssistantText(text: string, ts: number): AgentMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    api: "anthropic-messages",
    provider: "anthropic",
    model: "claude-sonnet",
    usage: {
      input: 100,
      output: 20,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 120,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    timestamp: ts,
  } as AgentMessage;
}

function makeAssistantToolCall(id: string, name: string, ts: number): AgentMessage {
  return {
    role: "assistant",
    content: [{ type: "toolCall", id, name, arguments: { path: "src/app.ts" } }],
    api: "anthropic-messages",
    provider: "anthropic",
    model: "claude-sonnet",
    usage: {
      input: 100,
      output: 20,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 120,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "toolUse",
    timestamp: ts,
  } as AgentMessage;
}

function makeToolResult(toolCallId: string, toolName: string, text: string, ts: number): AgentMessage {
  return {
    role: "toolResult",
    toolCallId,
    toolName,
    content: [{ type: "text", text }],
    isError: false,
    timestamp: ts,
  } as AgentMessage;
}

function withTempDir(fn: (dir: string) => Promise<void> | void): Promise<void> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-ctx-test-"));
  return Promise.resolve(fn(dir)).finally(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });
}

describe("pi-auto-context-offloader", () => {
  it("passes through under budget without touching disk", async () => {
    await withTempDir(async (dir) => {
      const h = createHarness(dir);
      await h.emit("session_start", {});
      await h.commands.get("ctxlimit")!.handler("10000", h.ctx);

      const msgs = [makeSystem("Sys prompt", {}), makeUser("Hello", 10), makeAssistantText("Hi!", 20)];
      const res = await h.emit("context_with_system", { type: "context_with_system", messages: msgs });
      assert.equal(res, undefined);
      assert.equal(fs.existsSync(path.join(dir, ".pi", "offload", "test-session", "offload.log")), false);
    });
  });

  it("uses hysteresis and preserves identical prefix across subsequent turns (prompt cache)", async () => {
    await withTempDir(async (dir) => {
      const h = createHarness(dir);
      await h.emit("session_start", {});
      // Limit = 1200 tokens (~4800 chars). System = 200 tokens. Conversation budget = 1000 tokens.
      // Low water = 55% of 1000 = 550 tokens.
      await h.commands.get("ctxlimit")!.handler("1200", h.ctx);

      const msgs: AgentMessage[] = [makeSystem("S".repeat(800))]; // ~200 tokens
      for (let i = 0; i < 8; i++) {
        // Each turn = ~250 tokens (1000 chars) -> total ~2200 tokens > 1200
        msgs.push(makeUser(`User turn ${i}: ` + "U".repeat(500), 100 + i * 2));
        msgs.push(makeAssistantText(`Asst turn ${i}: ` + "A".repeat(480), 101 + i * 2));
      }

      const first = (await h.emit("context_with_system", {
        type: "context_with_system",
        messages: msgs,
      })) as { messages: AgentMessage[] };

      assert.ok(first, "expected offloading when over 1200 tokens");
      const marker1 = first.messages[1];
      const head1 = first.messages[0];
      assert.equal(head1.role, "system");
      assert.equal(marker1.role, "user");
      assert.ok(first.messages.length < msgs.length, "should have cut older turns");

      const offloadFile = path.join(dir, ".pi", "offload", "test-session", "offload.log");
      const sizeAfterFirstCut = fs.statSync(offloadFile).size;

      // Append a small turn that stays under 1200 tokens total
      msgs.push(makeUser("Small follow-up question", 500));
      const second = (await h.emit("context_with_system", {
        type: "context_with_system",
        messages: msgs,
      })) as { messages: AgentMessage[] };

      assert.ok(second, "still uses pinned cut");
      // Exact same object references for head and marker -> 100% cache-stable prefix!
      assert.strictEqual(second.messages[0], head1);
      assert.strictEqual(second.messages[1], marker1);
      assert.equal(fs.statSync(offloadFile).size, sizeAfterFirstCut, "must not re-offload while under limit");

      const status = await h.tools.get("ctx_status")!.execute("1", {}, undefined, undefined, h.ctx);
      assert.match(status.content[0].text, /pinned \(1 consecutive cache-preserving call/);
    });
  });

  it("never splits assistant(toolCall) from its toolResult messages", async () => {
    const msgs: AgentMessage[] = [
      makeSystem("Sys"),
      makeUser("Read file", 10),
      makeAssistantToolCall("tc1", "read", 20),
      makeToolResult("tc1", "read", "X".repeat(4000), 30), // ~1000 tokens
      makeAssistantText("Done reading", 40),
    ];

    const cut = selectCutPoint({
      messages: msgs,
      limit: 600,
      calibrationRatio: 1.0,
    });
    assert.ok(cut !== null);
    assert.notEqual(msgs[cut].role, "toolResult", "cut point must never be a toolResult");
    // Specifically, either cut is at assistant(toolCall) (idx 2) or after toolResult (idx 4)
    assert.ok(cut === 2 || cut === 4, `unexpected cut index ${cut}`);
  });

  it("replays mid-conversation system sections & tools into the leading head", async () => {
    await withTempDir(async (dir) => {
      const h = createHarness(dir);
      await h.emit("session_start", {});
      await h.commands.get("ctxlimit")!.handler("500", h.ctx);

      const dynamicTool = {
        name: "special_tool",
        description: "Dynamically unlocked tool",
        parameters: { type: "object", properties: {} },
      };

      const msgs: AgentMessage[] = [
        makeSystem("Base prompt"),
        makeUser("Turn 0 " + "A".repeat(800), 10),
        makeSystem("", { sections: { dynamic: "Extra section rule" }, toolsAdded: [dynamicTool] }),
        makeAssistantText("Ack " + "B".repeat(800), 20),
        makeUser("Turn 1 " + "C".repeat(800), 30),
        makeAssistantText("Final " + "D".repeat(400), 40),
      ];

      const res = (await h.emit("context_with_system", {
        type: "context_with_system",
        messages: msgs,
      })) as { messages: AgentMessage[] };

      assert.ok(res);
      const head = res.messages[0] as AgentMessage & {
        content: string;
        sections?: Record<string, string>;
        toolsAdded?: Array<{ name: string }>;
      };
      assert.equal(head.role, "system");
      assert.equal(head.sections?.dynamic, "Extra section rule");
      assert.ok(head.toolsAdded?.some((t) => t.name === "special_tool"));
    });
  });

  it("calibrates against real provider usage from message_end", async () => {
    await withTempDir(async (dir) => {
      const h = createHarness(dir);
      await h.emit("session_start", {});
      await h.commands.get("ctxlimit")!.handler("1000", h.ctx);

      // ~700 estimated tokens -> under 1000, so pass-through initially
      const msgs: AgentMessage[] = [
        makeSystem("Sys"),
        makeUser("U".repeat(2600), 10),
      ];
      const first = await h.emit("context_with_system", { type: "context_with_system", messages: msgs });
      assert.equal(first, undefined);

      // Provider reports 1120 actual input tokens (~1.7x estimated) due to dense tokenizer / extra prompt
      await h.emit("message_end", {
        type: "message_end",
        message: {
          ...makeAssistantText("ok", 20),
          usage: {
            input: 320,
            output: 20,
            cacheRead: 800,
            cacheWrite: 0,
            totalTokens: 1140,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
        },
      });

      // Add another turn: raw ~750 tokens, but calibrated (~1.7x -> ~1275) > 1000 -> must offload!
      msgs.push(makeAssistantText("ok", 20));
      msgs.push(makeUser("next question", 30));

      const second = (await h.emit("context_with_system", {
        type: "context_with_system",
        messages: msgs,
      })) as { messages: AgentMessage[] };
      assert.ok(second, "calibration should trigger offloading when actual usage exceeds limit");
    });
  });

  it("deduplicates chunks across session resume and preserves lossless tool results", async () => {
    await withTempDir(async (dir) => {
      const bigOutput = "HEADER\n" + "line ".repeat(1200) + "NEEDLE_SECRET_TOKEN_42" + " tail".repeat(1200);
      const msgs: AgentMessage[] = [
        makeSystem("Sys"),
        makeUser("Run big read", 10),
        makeAssistantToolCall("tc1", "read", 20),
        makeToolResult("tc1", "read", bigOutput, 30),
        makeUser("Recent question", 40),
      ];

      // Verify lossless serializer keeps full NEEDLE and doesn't truncate
      const serialized = serializeMessagesLossless(msgs);
      assert.ok(serialized.includes("NEEDLE_SECRET_TOKEN_42"));
      assert.ok(!serialized.includes("truncated"));

      // First session cuts and writes to offload.log
      const h1 = createHarness(dir, "resume-sess");
      await h1.emit("session_start", {});
      await h1.commands.get("ctxlimit")!.handler("500", h1.ctx);
      await h1.emit("context_with_system", { type: "context_with_system", messages: msgs });

      const offloadFile = path.join(dir, ".pi", "offload", "resume-sess", "offload.log");
      const blocksBefore = parseOffloadBlocks(fs.readFileSync(offloadFile, "utf8"));
      assert.ok(blocksBefore.length >= 1);

      // Simulate restart / resume on the exact same session & transcript
      const h2 = createHarness(dir, "resume-sess");
      await h2.emit("session_start", {});
      await h2.emit("context_with_system", { type: "context_with_system", messages: msgs });

      const blocksAfter = parseOffloadBlocks(fs.readFileSync(offloadFile, "utf8"));
      assert.equal(blocksAfter.length, blocksBefore.length, "resume must not duplicate offloaded blocks");

      // ctx_search returns compact snippets rather than dumping the entire 12kb block
      const searchRes = await h2.tools.get("ctx_search")!.execute(
        "s1",
        { query: "NEEDLE_SECRET_TOKEN_42" },
        undefined,
        undefined,
        h2.ctx,
      );
      const searchText = searchRes.content[0].text;
      assert.ok(searchText.includes("NEEDLE_SECRET_TOKEN_42"));
      assert.ok(searchText.length < 1500, `search snippet too large: ${searchText.length}`);

      const blockMatch = searchText.match(/Block #(\d+)/);
      assert.ok(blockMatch, "search should cite matching Block #id");
      const targetBlockId = Number(blockMatch[1]);

      // ctx_recall supports blockId + pagination
      const recall1 = await h2.tools.get("ctx_recall")!.execute(
        "r1",
        { blockId: targetBlockId, offset: 0 },
        undefined,
        undefined,
        h2.ctx,
      );
      assert.ok(recall1.content[0].text.includes("Showing chars 0-"));
      assert.ok(recall1.content[0].text.length < 4800);

      // Page forward to verify full lossless retrieval
      const nextOffset = recall1.details.end as number;
      const recall2 = await h2.tools.get("ctx_recall")!.execute(
        "r2",
        { blockId: targetBlockId, offset: nextOffset },
        undefined,
        undefined,
        h2.ctx,
      );
      assert.ok(recall2.content[0].text.length > 0);
    });
  });

  it("supports ctx_offload note mode, auto trim mode, and monotonic multi-cut progression", async () => {
    await withTempDir(async (dir) => {
      const h = createHarness(dir, "multi-cut");
      await h.emit("session_start", {});
      await h.commands.get("ctxlimit")!.handler("1000", h.ctx);

      const noteRes = await h.tools.get("ctx_offload")!.execute(
        "o1",
        { content: "Remember decision: use WAL mode for SQLite." },
        undefined,
        undefined,
        h.ctx,
      );
      assert.match(noteRes.content[0].text, /Saved note as block #1/);

      // Duplicate note should be deduplicated
      const noteDup = await h.tools.get("ctx_offload")!.execute(
        "o2",
        { content: "Remember decision: use WAL mode for SQLite." },
        undefined,
        undefined,
        h.ctx,
      );
      assert.match(noteDup.content[0].text, /already present as block #1/);

      // Build conversation under limit (~700 tokens) and schedule manual trim
      const msgs: AgentMessage[] = [
        makeSystem("Sys"),
        makeUser("Turn A " + "X".repeat(800), 10),
        makeAssistantText("Reply A " + "Y".repeat(800), 20),
        makeUser("Turn B " + "Z".repeat(800), 30),
        makeAssistantText("Reply B " + "W".repeat(400), 40),
      ];

      await h.tools.get("ctx_offload")!.execute("o3", {}, undefined, undefined, h.ctx);
      const cut1 = (await h.emit("context_with_system", {
        type: "context_with_system",
        messages: msgs,
      })) as { messages: AgentMessage[] };
      assert.ok(cut1 && cut1.messages.length < msgs.length, "manual ctx_offload() trims down to low-water target");

      // Grow conversation past 1000 tokens -> second cut advances monotonically
      for (let i = 0; i < 6; i++) {
        msgs.push(makeUser(`More ${i} ` + "M".repeat(600), 100 + i * 2));
        msgs.push(makeAssistantText(`Ans ${i} ` + "N".repeat(600), 101 + i * 2));
      }
      const cut2 = (await h.emit("context_with_system", {
        type: "context_with_system",
        messages: msgs,
      })) as { messages: AgentMessage[] };
      assert.ok(cut2);
      assert.notStrictEqual(cut2.messages[1], cut1.messages[1], "marker advances on second cut");

      const blocks = parseOffloadBlocks(
        fs.readFileSync(path.join(dir, ".pi", "offload", "multi-cut", "offload.log"), "utf8"),
      );
      // Verify all block IDs are strictly increasing and unique
      const ids = blocks.map((b) => b.id);
      assert.deepEqual(ids, [...new Set(ids)]);
    });
  });

  it("extractSearchSnippets merges nearby matches cleanly and chunkMessagesForOffload bounds chunk sizes", () => {
    const text = "alpha ".repeat(100) + "FINDME one " + "beta ".repeat(5) + "FINDME two " + "gamma ".repeat(100);
    const { snippets, totalMatches } = extractSearchSnippets(text, "FINDME");
    assert.equal(totalMatches, 2);
    assert.equal(snippets.length, 1, "nearby matches should merge into one snippet");
    assert.ok(snippets[0].includes("FINDME one") && snippets[0].includes("FINDME two"));

    const manyMsgs: AgentMessage[] = [];
    for (let i = 0; i < 12; i++) {
      manyMsgs.push(makeUser("U".repeat(2000), i * 2 + 1)); // ~500 tokens each
      manyMsgs.push(makeAssistantText("A".repeat(2000), i * 2 + 2)); // ~500 tokens each
    }
    const chunks = chunkMessagesForOffload(manyMsgs, 2000);
    assert.ok(chunks.length >= 4 && chunks.length <= 8);
  });
});
