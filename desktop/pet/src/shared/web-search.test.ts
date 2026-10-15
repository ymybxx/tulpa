import { describe, expect, it } from "vitest";
import { webSearchSupported } from "./web-search";

describe("webSearchSupported", () => {
  it("knows the services that search for real", () => {
    expect(webSearchSupported("https://ark.cn-beijing.volces.com/api/v3")).toBe(true);
    expect(webSearchSupported("https://dashscope.aliyuncs.com/compatible-mode/v1")).toBe(true);
    expect(webSearchSupported("https://dashscope-intl.aliyuncs.com/compatible-mode/v1")).toBe(true);
    expect(webSearchSupported("https://api.openai.com/v1")).toBe(false);
    expect(webSearchSupported("https://api.deepseek.com")).toBe(false);
    expect(webSearchSupported("https://evil.example/ark.cn-beijing.volces.com")).toBe(false);
    expect(webSearchSupported("not a url")).toBe(false);
  });
});
