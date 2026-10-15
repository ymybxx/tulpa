import { describe, expect, it } from "vitest";
import { chatMessage, type ChatMessage } from "../shared/chat";
import { dayLabel, mergeMessages, originLabel, snippet, stateHint, stateLabel, timelineRows } from "./model";

const at = (iso: string) => new Date(iso);
const message = (sequence: number, role: ChatMessage["role"], createdAt: string, content = `m${sequence}`): ChatMessage =>
  ({ id: `id-${sequence}`, sequence, role, content, status: role === "user" ? "initialized" : "completed", createdAt });

describe("chat timeline", () => {
  const now = at("2026-09-26T10:30:00");

  it("names days the way people say them", () => {
    expect(dayLabel(at("2026-09-26T10:12:00"), now)).toBe("今天 10:12");
    expect(dayLabel(at("2026-09-25T23:41:00"), now)).toBe("昨天 23:41");
    expect(dayLabel(at("2026-09-02T08:05:00"), now)).toBe("9月2日 08:05");
    expect(dayLabel(at("2025-12-31T21:00:00"), now)).toBe("2025年12月31日 21:00");
  });

  it("starts a new stretch at a new day or after a long pause", () => {
    const rows = timelineRows([
      message(1, "assistant", "2026-09-25T23:41:00"),
      message(2, "user", "2026-09-25T23:43:00"),
      message(3, "assistant", "2026-09-25T23:43:30"),
      message(4, "assistant", "2026-09-26T09:00:00"),
      message(5, "user", "2026-09-26T10:12:00"),
    ], now);
    expect(rows.map((row) => row.kind === "divider" ? row.label : `${row.message.sequence}`))
      .toEqual(["昨天 23:41", "1", "2", "3", "今天 09:00", "4", "今天 10:12", "5"]);
  });

  it("marks where a 新对话 starts, even right after the last message", () => {
    const rows = timelineRows([
      { ...message(1, "user", "2026-09-26T10:10:00"), sessionId: "a" },
      { ...message(2, "assistant", "2026-09-26T10:10:20"), sessionId: "a" },
      { ...message(3, "user", "2026-09-26T10:12:00"), sessionId: "b" },
    ], now);
    expect(rows.map((row) => row.kind === "divider" ? row.label : `${row.message.sequence}`))
      .toEqual(["今天 10:10", "1", "2", "新对话 · 今天 10:12", "3"]);
    expect(chatMessage({ messageId: "m", sequence: 1, role: "user", content: "嗨", status: "completed",
      createdAt: "2026-09-26T10:00:00Z", sessionId: "s-1" })?.sessionId).toBe("s-1");
  });

  it("keeps an interrupted answer's unsaid rest", () => {
    const parsed = chatMessage({ messageId: "m", sequence: 1, role: "assistant", content: "今天天气不错，", status: "interrupted",
      createdAt: "2026-09-26T10:00:00Z", unsaid: "我们出去走走吧。" });
    expect(parsed?.unsaid).toBe("我们出去走走吧。");
  });

  it("marks what the user said to the pet, and only that", () => {
    const base = { messageId: "m", sequence: 1, role: "user", content: "嗨", status: "completed", createdAt: "2026-09-26T10:00:00Z" };
    expect(chatMessage({ ...base, inputMode: "spoken" })?.spoken).toBe(true);
    expect(chatMessage({ ...base, inputMode: "typed" })?.spoken).toBeUndefined();
    expect(chatMessage(base)?.spoken).toBeUndefined();
  });

  it("says why TA spoke up from what was recorded", () => {
    expect(originLabel({ kind: "sense", appName: "Visual Studio Code" }, "阿澄")).toBe("看到你在用 Visual Studio Code");
    expect(originLabel({ kind: "sense" }, "阿澄")).toBe("看到听到了新东西");
    expect(originLabel({ kind: "self" }, "阿澄")).toBe("阿澄自己想说的");
    expect(chatMessage({ messageId: "m", sequence: 1, role: "assistant", content: "嗨", status: "completed",
      createdAt: "2026-09-26T10:00:00Z", origin: { kind: "made-up" } })?.origin).toBeUndefined();
    expect(chatMessage({ messageId: "m", sequence: 1, role: "assistant", content: "嗨", status: "completed",
      createdAt: "2026-09-26T10:00:00Z", origin: { kind: "task" } })?.origin).toEqual({ kind: "task" });
  });

  it("refreshes a message that changed and keeps the order", () => {
    const thinking = { ...message(3, "assistant", "2026-09-26T10:00:00", ""), status: "generating" as const };
    const merged = mergeMessages([message(1, "user", "2026-09-26T09:00:00"), thinking],
      [{ ...thinking, content: "想好了", status: "completed" }, message(2, "user", "2026-09-26T09:30:00")]);
    expect(merged.map((item) => `${item.sequence}:${item.content}`)).toEqual(["1:m1", "2:m2", "3:想好了"]);
  });

  it("says what TA is doing and cuts a result around the match", () => {
    expect(stateLabel("listening")).toBe("在听你说");
    expect(stateLabel("speaking")).toBe("在说话");
    expect(stateLabel(null)).toBe("陪伴中");
    // At rest the word follows 主动陪伴, so it never claims a switched-off feature.
    expect(stateLabel(null, false)).toBe("等你来聊");
    expect(stateLabel("speaking", false)).toBe("在说话");
    expect(stateHint(null, false)).toContain("TA 不会自己开口");
    expect(snippet("就是这个报错我好像见过", "报错", 2)).toEqual({ pre: "…这个", hit: "报错", post: "我好…" });
    expect(snippet("Build failed", "build")).toEqual({ pre: "", hit: "Build", post: " failed" });
  });
});


describe("replies cut off before a word", () => {
  it("leave no empty bubble behind", () => {
    const base = { sequence: 1, createdAt: "2026-09-28T13:27:18Z", role: "assistant" as const };
    const rows = timelineRows([
      { ...base, id: "a", content: "", status: "interrupted" },
      { ...base, id: "b", sequence: 2, content: "说了一半", status: "interrupted" },
    ] as unknown as ChatMessage[], new Date("2026-09-28T14:00:00Z"));
    expect(rows.filter((row) => row.kind === "message").map((row) => row.key)).toEqual(["b"]);
  });
});
