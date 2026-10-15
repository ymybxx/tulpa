import { describe, expect, it } from "vitest";

import { JEV_ENDPOINT } from "../src/shared/decision-model";
import { checkAiService, checkJevService } from "./ai-service-check";

const request = { baseUrl: "https://example.com/v1/", model: "qwen3.7-flash", apiKey: " sk-1 " };

describe("checkAiService", () => {
  it("sends a one-token completion to the normalized endpoint", async () => {
    let seen: { url: string; init?: RequestInit } | null = null;
    const result = await checkAiService(request, async (input, init) => {
      seen = { url: String(input), init };
      return Response.json({ choices: [] });
    });

    expect(result).toEqual({ ok: true });
    expect(seen!.url).toBe("https://example.com/v1/chat/completions");
    expect((seen!.init!.headers as Record<string, string>).Authorization).toBe("Bearer sk-1");
    expect(JSON.parse(seen!.init!.body as string)).toMatchObject({ model: "qwen3.7-flash", max_tokens: 1 });
  });

  it("explains common failures in plain words", async () => {
    await expect(checkAiService(request, async () => new Response("", { status: 401 })))
      .resolves.toEqual({ ok: false, message: "Key 不对或没有权限" });
    await expect(checkAiService(request, async () => new Response("", { status: 404 })))
      .resolves.toEqual({ ok: false, message: "找不到这个接口地址或模型" });
    await expect(checkAiService(request, async () => { throw new TypeError("fetch failed"); }))
      .resolves.toEqual({ ok: false, message: "连不上这个服务，请检查网络或接口地址" });
  });

  it("rejects malformed requests before any network call", async () => {
    const never = async () => { throw new Error("should not fetch"); };
    await expect(checkAiService({ ...request, apiKey: "" }, never)).rejects.toThrow("API Key");
    await expect(checkAiService({ ...request, baseUrl: "ftp://x" }, never)).rejects.toThrow("HTTP");
    await expect(checkAiService({ ...request, baseUrl: "https://u:p@x.com" }, never)).rejects.toThrow("HTTP");
  });
});

describe("checkJevService", () => {
  it("asks Jev one tiny question with the key and reports a bad key plainly", async () => {
    const calls: Array<[string, RequestInit | undefined]> = [];
    const fetcher = async (url: string | URL | Request, init?: RequestInit) => {
      calls.push([String(url), init]);
      return new Response("{}", { status: calls.length === 1 ? 200 : 401 });
    };
    expect(await checkJevService({ apiKey: " jev-key " }, fetcher)).toEqual({ ok: true });
    expect(calls[0]![0]).toBe(JEV_ENDPOINT);
    expect((calls[0]![1]!.headers as Record<string, string>).Authorization).toBe("Bearer jev-key");
    expect(JSON.parse(String(calls[0]![1]!.body))).toMatchObject({ model: "jev-1.13.0", questions: { ready: { type: "choice" } } });
    expect(await checkJevService({ apiKey: "bad" }, fetcher)).toEqual({ ok: false, message: "Key 不对或没有权限" });
    await expect(checkJevService({ apiKey: " " }, fetcher)).rejects.toThrow("Jev API Key");
  });
});
