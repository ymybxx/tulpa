import { describe, expect, it, vi } from "vitest";
import { configureLlmLogs, loggedFetch } from "./llm-logs";

describe("desktop LLM recording", () => {
  it("returns the original response and records full JSON without authorization headers", async () => {
    const save = vi.fn(
      async () => new Response('{"recorded":true}', { status: 200 }),
    );
    vi.stubGlobal("fetch", save);
    configureLlmLogs(async () => ({
      apiUrl: "http://localhost:1",
      apiKey: "internal-fixture",
    }));
    const response = new Response(
      '{"choices":[{"message":{"content":"line 1\\nline 2"}}]}',
    );
    const provider = vi.fn(async () => response);
    const call = loggedFetch("desktop.texture", provider);
    try {
      const actual = await call(
        "https://model.invalid/chat/completions?token=hidden",
        {
          method: "POST",
          headers: { Authorization: "Bearer private-fixture" },
          body: JSON.stringify({
            model: "test",
            messages: [{ role: "user", content: "line 1\nline 2" }],
          }),
        },
      );
      expect(actual).toBe(response);
      await expect(actual.json()).resolves.toHaveProperty("choices");
      await vi.waitFor(() => expect(save).toHaveBeenCalledOnce());
      const logged = JSON.parse(
        (save.mock.calls[0] as unknown as [string, RequestInit])[1]
          .body as string,
      );
      expect(logged.request.messages[0].content).toBe("line 1\nline 2");
      expect(logged.response.body.choices[0].message.content).toBe(
        "line 1\nline 2",
      );
      expect(JSON.stringify(logged)).not.toContain("private-fixture");
      expect(logged.request_url).not.toContain("hidden");
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("caps only the log while returning the complete provider body", async () => {
    const save = vi.fn(async () => new Response('{"recorded":true}'));
    vi.stubGlobal("fetch", save);
    configureLlmLogs(async () => ({
      apiUrl: "http://localhost:1",
      apiKey: "fixture",
    }));
    const body = "a".repeat(17 * 1024 * 1024);
    const call = loggedFetch("desktop.vision", async () => new Response(body));
    try {
      const response = await call("https://model.invalid/chat/completions", {
        method: "POST",
        body: '{"model":"test"}',
      });
      expect((await response.text()).length).toBe(body.length);
      await vi.waitFor(() => expect(save).toHaveBeenCalledOnce());
      const log = JSON.parse(
        (save.mock.calls[0] as unknown as [string, RequestInit])[1]
          .body as string,
      );
      expect(log.response.truncated).toBe(true);
      expect(log.response.raw.length).toBe(16 * 1024 * 1024);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("does not fail the provider response when the local log service is unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("offline");
      }),
    );
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    configureLlmLogs(async () => ({
      apiUrl: "http://localhost:1",
      apiKey: "fixture",
    }));
    try {
      const response = await loggedFetch(
        "desktop.vision",
        async () => new Response("ok"),
      )("https://model.invalid/chat/completions", {
        method: "POST",
        body: '{"model":"test"}',
      });
      expect(await response.text()).toBe("ok");
      await vi.waitFor(() => expect(warning).toHaveBeenCalledOnce());
    } finally {
      vi.unstubAllGlobals();
      warning.mockRestore();
    }
  });
});
