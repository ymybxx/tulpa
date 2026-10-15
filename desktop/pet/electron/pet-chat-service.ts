import type { DesktopRuntimeStatus } from "../src/shared/contracts";
import type { ChatAttachmentUpload } from "../src/shared/chat-attachments";
import { setTimeout as delay } from "node:timers/promises";
import { serviceFailure, type ServiceFailure } from "../src/shared/service-failures";

interface ChatAcceptance { ok?: boolean; error?: string; retryable?: boolean; messageId: string; sessionId: string; sequence: number; answerExpected: boolean }
interface SendResult { ok: boolean; error?: string; retryable?: boolean }
class SendError extends Error { constructor(message: string, readonly retryable: boolean) { super(message); } }
interface Options {
  status(): DesktopRuntimeStatus;
  apiKey(): Promise<string>;
  onText(text: string, status: string, failure?: ServiceFailure | null): void;
}

/** Main-process adapter: renderers get neither credentials nor a second TTS.
 * Native typing uses the same turn entry as speech; text-only mode uses /ask. */
export class PetChatService {
  private readonly submissions = new Map<string, { content: string; result: Promise<SendResult> }>();
  private static readonly ATTACHMENT_ID = /^[0-9a-f-]{36}$/iu;
  private stream: AbortController | null = null;
  private background: AbortController | null = null;
  private backgroundApi: string | null = null;
  private readonly backgroundCursors = new Map<string, number>();
  private readonly displayedMessages = new Set<string>();
  private busy = false;
  constructor(private readonly options: Options) {}

  submit(id: string, content: string, attachmentIds: readonly string[] = []): Promise<SendResult> {
    if (!/^[0-9a-f-]{36}$/iu.test(id) || !content.trim() || content.length > 32_000) return Promise.resolve({ ok: false, error: "请输入 1–32000 个字" });
    if (attachmentIds.length > 8 || attachmentIds.some((item) => !PetChatService.ATTACHMENT_ID.test(item))) {
      return Promise.resolve({ ok: false, error: "附件无效，请重新添加" });
    }
    const key = [content, ...attachmentIds].join("\u0000");
    const previous = this.submissions.get(id);
    if (previous) return previous.content === key ? previous.result : Promise.resolve({ ok: false, error: "发送标识已被使用" });
    if (this.busy) return Promise.resolve({ ok: false, error: "上一条文字正在发送", retryable: true });
    this.busy = true;
    const result = this.send(id, content, attachmentIds).catch((error: unknown): SendResult => ({ ok: false, error: error instanceof Error ? error.message : "发送状态未确认，请查看聊天记录", retryable: error instanceof SendError && error.retryable })).then(result => {
      if (!result.ok && result.retryable) this.submissions.delete(id);
      return result;
    }).finally(() => { this.busy = false; });
    this.submissions.set(id, { content: key, result });
    if (this.submissions.size > 64) this.submissions.delete(this.submissions.keys().next().value!);
    return result;
  }

  stop(): void {
    this.stopAnswer();
    this.background?.abort();
    this.background = null;
    this.backgroundApi = null;
  }

  /**
   * 「新对话」: TA stops carrying on the conversation so far; what it remembers stays.
   * The voice companion hears of it from the service; the text-only stream rebinds now.
   */
  async startNewConversation(): Promise<{ ok: true; sessionId: string } | { ok: false; error: string }> {
    const failed = { ok: false as const, error: "没能开始新对话，请再试一次" };
    let sessionId: unknown;
    try {
      const { api, headers, fishId } = await this.assistant();
      const response = await fetch(`${api}/v1/pet/session/new`, {
        method: "POST", headers, body: JSON.stringify({ fishId }), signal: AbortSignal.timeout(10_000),
      });
      if (!response.ok) return failed;
      sessionId = ((await response.json()) as { sessionId?: unknown }).sessionId;
    } catch (error) {
      return error instanceof Error && error.message.startsWith("请先") ? { ok: false, error: error.message } : failed;
    }
    if (typeof sessionId !== "string") return failed;
    this.stopAnswer();
    this.background?.abort();
    this.background = null;
    this.backgroundApi = null;
    this.syncRuntime();
    return { ok: true, sessionId };
  }

  /** Background Assistant replies use the same text bubble as ordinary chat.
   * A running Companion remains the sole owner of native playback/captions. */
  syncRuntime(): void {
    const status = this.options.status();
    const api = status.phase === "ready" && !status.voiceEnabled ? status.apiUrl : null;
    if (api === this.backgroundApi) return;
    this.background?.abort();
    this.background = null;
    this.backgroundApi = api;
    if (!api) return;
    const controller = this.background = new AbortController();
    void this.readBackground(api, controller.signal).catch(() => {
      // Shutdown aborts both the current stream and any reconnect delay.
    });
  }

  private stopAnswer(): void { this.stream?.abort(); this.stream = null; }

  private async readBackground(api: string, signal: AbortSignal): Promise<void> {
    while (!signal.aborted) {
      let renewal: AbortSignal | undefined;
      let reconnectDelay = 3_000;
      try {
        const headers = { Authorization: `Bearer ${await this.options.apiKey()}`, "Content-Type": "application/json" };
        const get = async (route: string): Promise<any> => {
          const response = await fetch(`${api}${route}`, { headers, signal: AbortSignal.any([signal, AbortSignal.timeout(5_000)]) });
          if (!response.ok) throw new Error("chat context unavailable");
          return response.json();
        };
        const context = async (): Promise<{ fishId: string; sessionId: string } | null> => {
          const assistant = await get("/v1/fishes/assistant");
          if (typeof assistant.fishId !== "string") return null;
          const sessions = await get(`/v1/fishes/${assistant.fishId}/sessions`);
          const sessionId = sessions.items?.[0]?.sessionId;
          return typeof sessionId === "string" ? { fishId: assistant.fishId, sessionId } : null;
        };
        const target = await context();
        if (target && !this.busy && !this.stream) {
          const { fishId, sessionId } = target;
          const key = `${api}:${fishId}:${sessionId}`;
          const route = `/v1/fishes/${fishId}/companion-messages`;
          const after = this.backgroundCursors.get(key) ?? 0;
          // Keep the same 18s context-refresh cadence, with no disconnected gap.
          renewal = AbortSignal.timeout(18_000);
          const streamSignal = AbortSignal.any([signal, renewal]);
          for await (const { event, data } of readChatEvents(`${api}${route}/events?sessionId=${sessionId}&afterSequence=${after}`, headers, streamSignal)) {
            if (event !== "companion.message" || data.sessionId !== sessionId) continue;
            const { messageId, content, sequence, consumedOffset } = data;
            if (typeof messageId !== "string" || typeof content !== "string" || !Number.isSafeInteger(sequence) || !Number.isInteger(consumedOffset)) continue;
            if (consumedOffset !== -1 && Number(consumedOffset) < Array.from(content).length) {
              const current = await context();
              if (signal.aborted || this.options.status().voiceEnabled || this.busy || this.stream || current?.fishId !== fishId || current?.sessionId !== sessionId) break;
              if (!this.displayedMessages.has(messageId)) {
                this.options.onText(content, "completed");
                this.displayedMessages.add(messageId);
                if (this.displayedMessages.size > 1_000) this.displayedMessages.delete(this.displayedMessages.values().next().value!);
              }
              const response = await fetch(`${api}${route}/${messageId}/consumption?sessionId=${sessionId}`, {
                method: "POST", headers, signal: AbortSignal.any([signal, AbortSignal.timeout(5_000)]),
                // Existing text-chat sentinel; never invent a played-audio offset.
                body: JSON.stringify({ consumedOffset: -1, interrupted: false }),
              });
              if (!response.ok) throw new Error("chat delivery receipt unavailable");
              await delay(6_000, undefined, { signal });
            }
            this.backgroundCursors.set(key, Math.max(this.backgroundCursors.get(key) ?? 0, Number(sequence)));
            if (this.backgroundCursors.size > 8) this.backgroundCursors.delete(this.backgroundCursors.keys().next().value!);
          }
          if (renewal.aborted) reconnectDelay = 0;
        }
      } catch (error) {
        // A planned renewal is healthy. Only real failures need retry backoff.
        if (renewal?.aborted && error === renewal.reason) reconnectDelay = 0;
      }
      if (reconnectDelay) await delay(reconnectDelay, undefined, { signal });
    }
  }

  /** Keeps a picture or text file on the local service until its message is sent. */
  async attach(upload: ChatAttachmentUpload): Promise<{ ok: true; id: string } | { ok: false; error: string }> {
    try {
      const { api, headers, fishId } = await this.assistant();
      const response = await fetch(`${api}/v1/fishes/${fishId}/attachments`, {
        method: "POST", headers, body: JSON.stringify(upload), signal: AbortSignal.timeout(30_000),
      });
      const data = await response.json().catch(() => ({})) as { attachmentId?: unknown; message?: unknown; detail?: unknown };
      if (!response.ok || typeof data.attachmentId !== "string") {
        const reason = typeof data.message === "string" ? data.message : typeof data.detail === "string" ? data.detail : "";
        return { ok: false, error: reason || "附件没有传上去，请再试一次" };
      }
      return { ok: true, id: data.attachmentId };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : "附件没有传上去，请再试一次" };
    }
  }

  /** A sent picture as a data URL, for showing it in the history. */
  async attachmentImage(id: string): Promise<string | null> {
    if (!PetChatService.ATTACHMENT_ID.test(id)) return null;
    try {
      const { api, headers, fishId } = await this.assistant();
      const response = await fetch(`${api}/v1/fishes/${fishId}/attachments/${id}`, { headers, signal: AbortSignal.timeout(15_000) });
      if (!response.ok) return null;
      const type = response.headers.get("content-type") ?? "image/jpeg";
      return `data:${type};base64,${Buffer.from(await response.arrayBuffer()).toString("base64")}`;
    } catch {
      return null;
    }
  }

  private async assistant(): Promise<{ api: string; headers: Record<string, string>; fishId: string }> {
    const status = this.options.status();
    if (!status.apiUrl) throw new Error("服务正在启动，请稍后再试");
    const headers = { Authorization: `Bearer ${await this.options.apiKey()}`, "Content-Type": "application/json" };
    const response = await fetch(`${status.apiUrl}/v1/fishes/assistant`, { headers, signal: AbortSignal.timeout(10_000) });
    const assistant = await response.json().catch(() => ({})) as { fishId?: unknown };
    if (!response.ok || typeof assistant.fishId !== "string") throw new Error("请先在设置中选择宠物");
    return { api: status.apiUrl, headers, fishId: assistant.fishId };
  }

  private async send(requestId: string, content: string, attachmentIds: readonly string[] = []): Promise<SendResult> {
    const status = this.options.status();
    // The API serves typed chat while the voice companion is still starting or
    // cannot open its audio (e.g. no microphone permission); only a ready
    // companion takes the message and speaks the reply.
    if (!status.apiUrl || !["starting-companion", "ready"].includes(status.phase)) throw new SendError("服务正在启动，请稍后再发送", true);
    const spoken = status.phase === "ready" && status.voiceEnabled;
    const api = status.apiUrl;
    const headers = { Authorization: `Bearer ${await this.options.apiKey()}`, "Content-Type": "application/json" };
    const request = async (route: string, body?: object): Promise<any> => {
      let response: Response;
      try {
        response = await fetch(`${api}${route}`, { headers, ...(body ? { method: "POST", body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(20_000) });
      } catch { throw new SendError(body ? "发送状态未确认，请先查看聊天记录；草稿已保留" : "暂时无法连接服务，草稿已保留", !body); }
      const data = await response.json();
      if (!response.ok) throw new SendError(typeof data.detail === "string" ? data.detail : `暂时无法发送 (${response.status})，草稿已保留`, !body || [400, 401, 403, 422].includes(response.status) || (route === "/v1/pet/chat" && response.status === 503));
      return data;
    };
    const assistant = await request("/v1/fishes/assistant");
    const fishId = assistant.fishId;
    if (typeof fishId !== "string") throw new SendError("请先在设置中选择宠物", true);
    // Never fall back after an uncertain native submission: it may already
    // have reached /ask. A second POST would create duplicate conversation.
    const attached = attachmentIds.length ? { attachmentIds } : {};
    const accepted: ChatAcceptance = spoken
      ? await request("/v1/pet/chat", { requestId, fishId, content, ...attached })
      : await request(`/v1/fishes/${fishId}/ask`, { content, ...attached });
    if (accepted.ok === false) return { ok: false, error: accepted.error ?? "未能发送，草稿已保留", retryable: accepted.retryable };
    this.stopAnswer();
    if (accepted.answerExpected && !(spoken && (status.playbackEnabled ?? status.voiceEnabled))) {
      this.stream = new AbortController();
      const controller = this.stream;
      const signal = this.stream.signal;
      this.options.onText("", "buffering");
      void this.readText(`${api}/v1/fishes/${fishId}/answer?sessionId=${accepted.sessionId}&afterSequence=${accepted.sequence}`, headers, signal)
        .catch(() => { if (!signal.aborted) this.options.onText("回答连接已断开，可在聊天记录中查看。", "failed"); })
        .finally(() => { if (this.stream === controller) this.stream = null; });
    }
    return { ok: true };
  }

  private async readText(url: string, headers: Record<string, string>, signal: AbortSignal): Promise<void> {
    let text = "";
    for await (const { event, data } of readChatEvents(url, headers, signal)) {
      if (event === "answer.snapshot") text = typeof data.content === "string" ? data.content : text;
      if (event === "answer.delta" && typeof data.content === "string") text += data.content;
      if (event === "answer.snapshot" || event === "answer.delta") this.options.onText(text, "text");
      if (["answer.completed", "answer.failed", "answer.interrupted"].includes(event ?? "")) {
        this.options.onText(text, event === "answer.failed" ? "failed" : "completed", serviceFailure(data.failure));
        return;
      }
    }
  }
}

async function* readChatEvents(url: string, headers: Record<string, string>, signal: AbortSignal): AsyncGenerator<{ event: string | undefined; data: Record<string, unknown> }> {
    const response = await fetch(url, { headers, signal });
    if (!response.ok || !response.body) throw new Error("answer unavailable");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (!signal.aborted) {
        const part = await reader.read();
        if (part.done) return;
        buffer = (buffer + decoder.decode(part.value, { stream: true })).replace(/\r\n/g, "\n");
        let end: number;
        while ((end = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, end); buffer = buffer.slice(end + 2);
          const lines = block.split("\n");
          const event = lines.find(line => line.startsWith("event:"))?.slice(6).trim();
          const raw = lines.filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
          if (!raw) continue;
          const data = JSON.parse(raw) as Record<string, unknown>;
          yield { event, data };
        }
      }
    } finally { await reader.cancel().catch(() => undefined); }
}
