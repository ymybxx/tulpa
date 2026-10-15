import { describe, expect, it } from "vitest";

import { activeBackground, parseUsageRows, summarizeUsage, tokenText, usageGroupOf } from "./usage";

const row = (call_type: string, purpose = "", values: Partial<Record<string, number>> = {}) => ({
  call_type, purpose, today_calls: 0, today_tokens: 0, week_calls: 0, week_tokens: 0,
  hour_calls: 0, hour_tokens: 0, recent_calls: 0, ...values,
});

describe("token usage", () => {
  it("files each call under what it was for", () => {
    expect(usageGroupOf("experimental.channel.s", "")).toBe("reply");
    expect(usageGroupOf("experimental.channel.s", "proactive")).toBe("proactive");
    expect(usageGroupOf("experimental.jev.decision", "proactive")).toBe("proactive");
    expect(usageGroupOf("experimental.thought.generate", "")).toBe("inner");
    expect(usageGroupOf("environment", "")).toBe("senses");
    expect(usageGroupOf("memory.dynamic", "")).toBe("memory");
    expect(usageGroupOf("speech.facial.jev", "")).toBe("facial");
    expect(usageGroupOf("plugin.select", "")).toBe("other");
  });

  it("adds up today and names what still runs in the background", () => {
    const summary = summarizeUsage(parseUsageRows({ items: [
      row("experimental.thought.generate", "", { today_calls: 30, today_tokens: 90_000, recent_calls: 4, hour_calls: 12, hour_tokens: 36_000 }),
      row("environment", "", { today_calls: 5, today_tokens: 20_000 }),
      row("chat", "", { today_calls: 2, today_tokens: 3_000, recent_calls: 1 }),
      { call_type: 7 },
    ] }));
    expect(summary.today).toEqual({ calls: 37, tokens: 113_000 });
    expect(summary.groups.find((item) => item.group === "inner")!.hour).toEqual({ calls: 12, tokens: 36_000 });
    // Replying to the person is not background use.
    expect(activeBackground(summary)).toEqual(["内心"]);
  });

  it("reads token counts the way people do", () => {
    expect(tokenText(0)).toBe("0");
    expect(tokenText(42)).toBe("约 42");
    expect(tokenText(3_240)).toBe("约 3200");
    expect(tokenText(32_400)).toBe("约 3.2 万");
    expect(tokenText(1_234_000)).toBe("约 123 万");
  });
});
