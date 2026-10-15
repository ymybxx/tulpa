import { afterEach, describe, expect, it, vi } from "vitest";
import type { DesktopRuntimeStatus } from "../src/shared/contracts";
import { PetChatService } from "./pet-chat-service";

const id = "00000000-0000-0000-0000-000000000001";
const status = (voiceEnabled: boolean): DesktopRuntimeStatus => ({ phase: "ready", apiUrl: "http://127.0.0.1:8129", voiceEnabled, playbackEnabled: voiceEnabled } as DesktopRuntimeStatus);
const json = (data: object, code = 200): Response => new Response(JSON.stringify(data), { status: code });
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
describe("quick chat submission", () => {
  it("uses the native owner without changing microphone settings, and deduplicates submission", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(json({ fishId: id })).mockResolvedValueOnce(json({ ok: true, messageId: id, answerExpected: true }));
    vi.stubGlobal("fetch", fetch);
    const service = new PetChatService({ status: () => status(true), apiKey: async () => "local-test-key", onText: vi.fn() });
    expect(await service.submit(id, "hello")).toEqual({ ok: true });
    expect(await service.submit(id, "hello")).toEqual({ ok: true });
    expect(fetch).toHaveBeenCalledTimes(2);
    const [url, options] = fetch.mock.calls[1]!;
    expect(url).toContain("/v1/pet/chat");
    expect(JSON.parse(options.body)).toEqual({ requestId: id, fishId: id, content: "hello" });
  });
  it("still sends typed chat, as text, while the voice companion cannot open the microphone", async () => {
    const sse = "event: answer.delta\ndata: {\"content\":\"在呢\"}\n\nevent: answer.completed\ndata: {}\n\n";
    const fetch = vi.fn().mockResolvedValueOnce(json({ fishId: id })).mockResolvedValueOnce(json({ sessionId: id, sequence: 3, answerExpected: true })).mockResolvedValueOnce(new Response(sse));
    vi.stubGlobal("fetch", fetch);
    const onText = vi.fn();
    const service = new PetChatService({ status: () => ({ ...status(true), phase: "starting-companion" }), apiKey: async () => "test", onText });
    expect(await service.submit(id, "今天心情好吗")).toEqual({ ok: true });
    expect(fetch.mock.calls[1]![0]).toContain(`/v1/fishes/${id}/ask`);
    await vi.waitFor(() => expect(onText).toHaveBeenLastCalledWith("在呢", "completed", null));
    service.stop();
  });
  it("does not fall back or resend when native acceptance is uncertain", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(json({ fishId: id })).mockRejectedValueOnce(new Error("socket disconnected after write"));
    vi.stubGlobal("fetch", fetch);
    const service = new PetChatService({ status: () => status(true), apiKey: async () => "test", onText: vi.fn() });
    expect(await service.submit(id, "hello")).toMatchObject({ ok: false, retryable: false });
    await service.submit(id, "hello");
    expect(fetch).toHaveBeenCalledTimes(2);
  });
  it("streams a text-only answer without starting native voice or writing playback offsets", async () => {
    const sse = "event: answer.delta\ndata: {\"content\":\"你好🐟\"}\n\nevent: answer.completed\ndata: {}\n\n";
    const fetch = vi.fn().mockResolvedValueOnce(json({ fishId: id })).mockResolvedValueOnce(json({ sessionId: id, sequence: 21, answerExpected: true })).mockResolvedValueOnce(new Response(sse));
    vi.stubGlobal("fetch", fetch);
    const onText = vi.fn();
    const service = new PetChatService({ status: () => status(false), apiKey: async () => "test", onText });
    expect(await service.submit(id, "hello")).toEqual({ ok: true });
    await vi.waitFor(() => expect(onText).toHaveBeenLastCalledWith("你好🐟", "completed", null));
    expect(fetch.mock.calls[1]![0]).toContain(`/v1/fishes/${id}/ask`);
    expect(fetch.mock.calls[2]![0]).toContain("afterSequence=21");
    expect(fetch).toHaveBeenCalledTimes(3);
    service.stop();
  });
  it("tells the bubble which service let a typed reply down", async () => {
    const sse = "event: answer.failed\ndata: {\"content\":\"\",\"failure\":\"jev:timeout\"}\n\n";
    const fetch = vi.fn().mockResolvedValueOnce(json({ fishId: id })).mockResolvedValueOnce(json({ sessionId: id, sequence: 4, answerExpected: true })).mockResolvedValueOnce(new Response(sse));
    vi.stubGlobal("fetch", fetch);
    const onText = vi.fn();
    const service = new PetChatService({ status: () => status(false), apiKey: async () => "test", onText });
    expect(await service.submit(id, "hello")).toEqual({ ok: true });
    await vi.waitFor(() => expect(onText).toHaveBeenLastCalledWith("", "failed", { service: "jev", kind: "timeout" }));
    service.stop();
  });
});

describe("background replies in the chat bubble", () => {
  const pending = () => new Response(`event: companion.message\ndata: ${JSON.stringify({
    messageId: id, sessionId: id, sequence: 9, content: "任务完成🐟", consumedOffset: 0,
  })}\n\n`);

  it("displays a durable reply with both audio switches off and records text delivery", async () => {
    const fetch = vi.fn(async (url: string, options?: RequestInit) => {
      if (url.endsWith("/assistant")) return json({ fishId: id });
      if (url.endsWith("/sessions")) return json({ items: [{ sessionId: id }] });
      if (url.includes("/events?")) return pending();
      if (url.includes("/consumption?")) return new Response(null, { status: 204 });
      throw new Error(`Unexpected request: ${url} ${options?.method}`);
    });
    vi.stubGlobal("fetch", fetch);
    const onText = vi.fn();
    const service = new PetChatService({ status: () => status(false), apiKey: async () => "test", onText });
    try {
      service.syncRuntime();
      await vi.waitFor(() => expect(fetch.mock.calls.find(([url]) => url.includes("/consumption?"))).toBeDefined());
      expect(onText).toHaveBeenCalledTimes(1);
      expect(onText).toHaveBeenCalledWith("任务完成🐟", "completed");
      const receipt = fetch.mock.calls.find(([url]) => url.includes("/consumption?"))!;
      expect(JSON.parse(receipt[1]!.body as string)).toEqual({ consumedOffset: -1, interrupted: false });
      expect(fetch.mock.calls.every(([url]) => !url.includes("/pet/chat") && !url.includes("/configuration"))).toBe(true);
    } finally { service.stop(); }
  });

  it("leaves captions and playback to the existing Companion when voice is running", () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    const service = new PetChatService({ status: () => status(true), apiKey: async () => "test", onText: vi.fn() });
    service.syncRuntime();
    expect(fetch).not.toHaveBeenCalled();
    service.stop();
  });

  it("retries a failed receipt without displaying the same reply twice", async () => {
    let receipts = 0;
    const fetch = vi.fn(async (url: string) => {
      if (url.endsWith("/assistant")) return json({ fishId: id });
      if (url.endsWith("/sessions")) return json({ items: [{ sessionId: id }] });
      if (url.includes("/events?")) return pending();
      if (url.includes("/consumption?")) return new Response(null, { status: ++receipts === 1 ? 503 : 204 });
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetch);
    const onText = vi.fn();
    const service = new PetChatService({ status: () => status(false), apiKey: async () => "test", onText });
    try {
      service.syncRuntime();
      await vi.waitFor(() => expect(receipts).toBe(2), { timeout: 4_000 });
      expect(onText).toHaveBeenCalledTimes(1);
    } finally { service.stop(); }
  });

  it("renews a healthy stream without adding the transport-error backoff", async () => {
    const renewal = new AbortController();
    const originalTimeout = AbortSignal.timeout;
    const timeout = vi.spyOn(AbortSignal, "timeout").mockImplementation(ms => ms === 18_000 ? renewal.signal : originalTimeout(ms));
    let connections = 0;
    const fetch = vi.fn(async (url: string, options?: RequestInit) => {
      if (url.endsWith("/assistant")) return json({ fishId: id });
      if (url.endsWith("/sessions")) return json({ items: [{ sessionId: id }] });
      if (url.includes("/events?")) {
        if (++connections > 1) return pending();
        return new Response(new ReadableStream({
          start(controller) { options!.signal!.addEventListener("abort", () => controller.error(options!.signal!.reason), { once: true }); },
        }));
      }
      if (url.includes("/consumption?")) return new Response(null, { status: 204 });
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetch);
    const onText = vi.fn();
    const service = new PetChatService({ status: () => status(false), apiKey: async () => "test", onText });
    try {
      service.syncRuntime();
      await vi.waitFor(() => expect(connections).toBe(1));
      // Subsequent renewal timers are independent, as they are in production.
      timeout.mockImplementation(ms => originalTimeout(ms));
      renewal.abort(new DOMException("planned renewal", "TimeoutError"));
      await vi.waitFor(() => expect(onText).toHaveBeenCalledWith("任务完成🐟", "completed"), { timeout: 1_000 });
      expect(connections).toBe(2);
    } finally { service.stop(); }
  });

  it("does not show a reply from the previous pet after the binding changes", async () => {
    let bindingReads = 0;
    const fetch = vi.fn(async (url: string) => {
      if (url.endsWith("/assistant")) return json({ fishId: ++bindingReads === 1 ? id : "another-fish" });
      if (url.endsWith("/sessions")) return json({ items: [{ sessionId: id }] });
      if (url.includes("/events?")) return pending();
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetch);
    const onText = vi.fn();
    const service = new PetChatService({ status: () => status(false), apiKey: async () => "test", onText });
    try {
      service.syncRuntime();
      await vi.waitFor(() => expect(bindingReads).toBe(2));
      expect(onText).not.toHaveBeenCalled();
      expect(fetch.mock.calls.some(([url]) => url.includes("/consumption?"))).toBe(false);
    } finally { service.stop(); }
  });
});
