import { describe, expect, it } from "vitest";
import { logTypeName, parsedText, responseContent } from "./llm-logs";

describe("readable LLM logs", () => {
  it("decodes JSON strings while preserving real multiline strings and malformed responses", () => {
    expect(parsedText('{"a":"第一行\\n第二行","data":{"ok":true}}')).toEqual({
      a: "第一行\n第二行",
      data: { ok: true },
    });
    expect(parsedText("第一行\n第二行")).toBe("第一行\n第二行");
    expect(parsedText("{invalid")).toBe("{invalid");
  });
  it("uses actual streamed text and ordinary response message content", () => {
    expect(
      responseContent({ text: '{"a":"完整返回"}', raw: "event stream" }),
    ).toEqual({ a: "完整返回" });
    expect(
      responseContent({
        body: { choices: [{ message: { content: "hello\nworld" } }] },
      }),
    ).toBe("hello\nworld");
    expect(responseContent({ body: { error: { message: "failed" } } })).toEqual(
      { error: { message: "failed" } },
    );
  });
  it("keeps new call types visible rather than dropping them", () => {
    expect(logTypeName("plugin.select")).toBe("直播选题");
    expect(logTypeName("future.kind")).toBe("future.kind");
  });
});

describe("log type names", () => {
  it("names every call type the service writes today, never a code", () => {
    for (const type of [
      "experimental.jev.decision", "experimental.jev.observation", "experimental.jev.thought",
      "experimental.channel.s", "experimental.channel.r", "experimental.channel.w", "experimental.channel.t",
      "experimental.web.result", "experimental.recall.summary", "experimental.result.summary",
      "experimental.material.browse", "experimental.thought.generate", "speech.facial.jev",
    ]) expect(logTypeName(type)).not.toContain(".");
  });
});
