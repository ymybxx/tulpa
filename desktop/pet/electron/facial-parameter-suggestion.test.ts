import { describe, expect, it, vi } from "vitest";
import { completionAnswer, extractJson, suggestFacialParameter, validateSuggestion } from "./facial-parameter-suggestion";
import type { FacialParameterSample } from "../src/shared/facial-test";

const sample: FacialParameterSample = {
  parameter: { id: "ParamCustom", name: "自定义眉形", minimum: -1, defaultValue: 0, maximum: 1 },
  frames: [
    { kind: "default", value: 0, image: "data:image/jpeg;base64,AAAA" },
    { kind: "target", value: -1, image: "data:image/jpeg;base64,BBBB" },
    { kind: "target", value: 1, image: "data:image/jpeg;base64,CCCC" },
  ],
};
const route = { id: "reasoning", name: "Reasoning", baseUrl: "https://example.com/v1", model: "vision", apiKey: "secret" };
const reply = (content: unknown) => new Response(JSON.stringify({ choices: [{ message: { content: typeof content === "string" ? content : JSON.stringify(content) } }] }));
const bodyOf = (fetcher: ReturnType<typeof vi.fn>, call: number) => JSON.parse(fetcher.mock.calls[call]![1].body as string);

describe("facial parameter suggestions", () => {
  it("sends calibrated frames with level mapping and returns a validated draft", async () => {
    const suggestion = { parameterId: "ParamCustom", confidence: "clear", label: "眉尾", description: "说出这句话时眉尾怎样变化？",
      levels: ["眉尾下压", "眉尾平直", "眉尾上挑"], note: "可看到眉尾改变" };
    const fetcher = vi.fn(async () => reply(suggestion));
    const result = await suggestFacialParameter(sample, "score", 3, route, new AbortController().signal, fetcher);
    expect(result).toEqual({ ...suggestion, mode: "score" });
    const [url] = fetcher.mock.calls[0]!;
    expect(url).toBe("https://example.com/v1/chat/completions");
    const content = bodyOf(fetcher, 0).messages[1].content;
    expect(content.filter((item: { type: string }) => item.type === "image_url")
      .map((item: { image_url: { url: string } }) => item.image_url.url)).toEqual(sample.frames.map(frame => frame.image));
    expect(content[0].text).toContain("共 3 档");
    expect(content[0].text).toContain("第1档=-1，第2档=0，第3档=1");
    const captions = content.filter((item: { type: string; text?: string }) => item.text?.startsWith("截图")).map((item: { text: string }) => item.text);
    expect(captions).toEqual(["截图 1：原生值 0（默认姿态，对应第2档）", "截图 2：原生值 -1（对应第1档）", "截图 3：原生值 1（对应第3档）"]);
  });

  it("asks for questions JEV can answer from the spoken text alone", async () => {
    const fetcher = vi.fn(async () => reply({ parameterId: "ParamCustom", confidence: "uncertain", note: "" }));
    await suggestFacialParameter(sample, "binary", 2, route, new AbortController().signal, fetcher);
    const [system, user] = bodyOf(fetcher, 0).messages;
    expect(system.content).toContain("看不到模型画面");
    expect(system.content).toContain("画面样子 + 适用语境");
    expect(user.content[0].text).toContain("不要问「是否出现问号」这类画面内容");
  });

  it("retries when the description restates the value mapping", async () => {
    const levels = ["头转向画面左侧，回避或看向左边时", "头正对前方，平常说话时", "头转向画面右侧，看向右边时"];
    const withValues = { parameterId: "ParamCustom", confidence: "clear", label: "头部转向",
      description: "默认正对前方，负值向左转，正值向右转。说这句话时角色看向哪边？", levels, note: "" };
    const clean = { ...withValues, description: "说这句话时角色是否在回避对方或看向某一侧？" };
    const fetcher = vi.fn().mockResolvedValueOnce(reply(withValues)).mockResolvedValueOnce(reply(clean));
    const result = await suggestFacialParameter(sample, "score", 3, route, new AbortController().signal, fetcher);
    expect(result.description).toBe(clean.description);
    expect(bodyOf(fetcher, 1).messages.at(-1).content).toContain("description 不应解释数值或方向（出现了「负值」）");
  });

  it("supports yes/no wording and rejects output for another parameter or wrong level count", () => {
    const binary = { parameterId: "ParamCustom", confidence: "clear", label: "眉尾", description: "是否挑眉？",
      falseDescription: "不挑眉", trueDescription: "挑眉", note: "" };
    expect(validateSuggestion(binary, "ParamCustom", "binary", 2)).toEqual({ ...binary, mode: "binary" });
    expect(() => validateSuggestion(binary, "Other", "binary", 2)).toThrow("parameterId");
    expect(() => validateSuggestion({ ...binary, levels: ["低", "高"] }, "ParamCustom", "score", 3)).toThrow("需要 3 项，实际 2 项");
    expect(validateSuggestion({ parameterId: "ParamCustom", confidence: "uncertain", note: "差异太小" },
      "ParamCustom", "score", 3)).toEqual({ parameterId: "ParamCustom", confidence: "uncertain",
        label: "", description: "", mode: "score", note: "差异太小" });
  });

  it("normalizes common model formatting slips instead of failing", () => {
    const result = validateSuggestion({ confidence: "Clear", label: " 眉尾 ", description: "眉尾怎样？",
      levels: ["第1档：眉尾下压", { description: "眉尾平直" }, "3. 眉尾上挑"] }, "ParamCustom", "score", 3);
    expect(result).toMatchObject({ confidence: "clear", label: "眉尾", levels: ["眉尾下压", "眉尾平直", "眉尾上挑"], note: "" });
    expect(validateSuggestion({ parameterId: "ParamCustom", confidence: "clear", label: "挑眉", description: "是否挑眉？",
      levels: ["眉尾平直", "眉尾上挑"] }, "ParamCustom", "binary", 3))
      .toMatchObject({ falseDescription: "眉尾平直", trueDescription: "眉尾上挑" });
    expect(extractJson('好的，结果如下：\n```json\n{"a":1}\n```')).toEqual({ a: 1 });
    expect(extractJson('结果：{"a":1} 以上')).toEqual({ a: 1 });
  });

  it("explains exactly which levels are duplicated so the repair turn can fix them", async () => {
    const duplicated = { parameterId: "ParamCustom", confidence: "clear", label: "眉尾", description: "眉尾怎样？",
      levels: ["眉尾平直", "眉尾平直", "眉尾上挑"], note: "" };
    const fixed = { ...duplicated, levels: ["眉尾平直", "眉尾略微上挑", "眉尾上挑"] };
    const fetcher = vi.fn().mockResolvedValueOnce(reply(duplicated)).mockResolvedValueOnce(reply(fixed));
    const result = await suggestFacialParameter(sample, "score", 3, route, new AbortController().signal, fetcher);
    expect(result.levels).toEqual(fixed.levels);
    const correction = bodyOf(fetcher, 1).messages.at(-1).content;
    expect(correction).toContain("第 1 档与第 2 档都是「眉尾平直」");
    expect(correction).toContain("程度词");
  });

  it("repairs a binary reply that omits both required descriptions", async () => {
    const incomplete = { parameterId: "ParamCustom", confidence: "clear", label: "挑眉",
      description: "说这句话时是否挑眉？", note: "最小值眉尾平直，最大值上挑" };
    const complete = { ...incomplete, falseDescription: "眉尾平直", trueDescription: "眉尾上挑" };
    const fetcher = vi.fn().mockResolvedValueOnce(reply(incomplete)).mockResolvedValueOnce(reply(complete));
    const result = await suggestFacialParameter(sample, "binary", 3, route, new AbortController().signal, fetcher);
    expect(result).toEqual({ ...complete, mode: "binary" });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(bodyOf(fetcher, 1).messages.at(-1).content).toContain("选择式缺少 falseDescription 和 trueDescription");
  });

  it("retries unparsable output and falls back when response_format is unsupported", async () => {
    const complete = { parameterId: "ParamCustom", confidence: "clear", label: "挑眉", description: "是否挑眉？",
      falseDescription: "眉尾平直", trueDescription: "眉尾上挑", note: "" };
    const fetcher = vi.fn()
      .mockResolvedValueOnce(new Response("unsupported response_format", { status: 400 }))
      .mockResolvedValueOnce(reply("我看到眉尾上挑了"))
      .mockResolvedValueOnce(reply(complete));
    const result = await suggestFacialParameter(sample, "binary", 2, route, new AbortController().signal, fetcher);
    expect(result.trueDescription).toBe("眉尾上挑");
    expect(bodyOf(fetcher, 0).response_format).toEqual({ type: "json_object" });
    expect(bodyOf(fetcher, 1).response_format).toBeUndefined();
  });

  it("reports an actionable error if every attempt stays invalid", async () => {
    const incomplete = { parameterId: "ParamCustom", confidence: "clear", label: "挑眉",
      description: "是否挑眉？", note: "" };
    const fetcher = vi.fn(async () => reply(incomplete));
    await expect(suggestFacialParameter(sample, "binary", 3, route, new AbortController().signal, fetcher))
      .rejects.toThrow(/选择式缺少 falseDescription 和 trueDescription.*手动填写/u);
    expect(fetcher).toHaveBeenCalledTimes(3);
  });
});

describe("completion answers", () => {
  const signal = new AbortController().signal;

  it("joins a streamed answer and skips the model's thinking", async () => {
    const events = [
      { choices: [{ delta: { reasoning_content: "想一想……" } }] },
      { choices: [{ delta: { content: "{\"a\":" } }] },
      { choices: [{ delta: { content: "1}" }, finish_reason: "stop" }] },
    ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") + "data: [DONE]\n\n";
    // Split mid-line, as a network delivers it.
    const bytes = new TextEncoder().encode(events);
    const body = new ReadableStream({ start(controller) {
      controller.enqueue(bytes.slice(0, 37)); controller.enqueue(bytes.slice(37)); controller.close();
    } });
    const answer = await completionAnswer(new Response(body, { headers: { "content-type": "text/event-stream" } }), signal);
    expect(answer).toEqual({ content: "{\"a\":1}", finishReason: "stop" });
  });

  it("reads a provider that ignores stream and answers with JSON", async () => {
    const answer = await completionAnswer(new Response(JSON.stringify({ choices: [{ finish_reason: "length", message: { content: "x" } }] }),
      { headers: { "content-type": "application/json" } }), signal);
    expect(answer).toEqual({ content: "x", finishReason: "length" });
  });

  it("names an error sent inside the stream", async () => {
    const response = new Response('data: {"error":{"message":"quota exceeded"}}\n\n', { headers: { "content-type": "text/event-stream" } });
    await expect(completionAnswer(response, signal)).rejects.toThrow("quota exceeded");
  });
});
