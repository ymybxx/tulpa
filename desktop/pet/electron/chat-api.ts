import { chatMessage, type ChatMessage, type ChatPage } from "../src/shared/chat";

export interface ChatApiTarget {
  apiUrl: string;
  apiKey: string;
}

const PAGE_SIZE = 40;

/** The conversation with the current character, read from the local memory service. */
export class ChatHistoryApi {
  constructor(private readonly target: () => Promise<ChatApiTarget | null>) {}

  async page(before?: number): Promise<ChatPage> {
    const fishId = await this.assistant();
    const query = new URLSearchParams({ limit: String(PAGE_SIZE) });
    if (before !== undefined) query.set("before", String(before));
    const payload = await this.request<{ items?: unknown[]; hasBefore?: boolean }>("GET", `/v1/fishes/${encodeURIComponent(fishId)}/timeline?${query}`);
    return { fishId, items: messages(payload.items), hasBefore: payload.hasBefore === true };
  }

  async search(text: string): Promise<ChatMessage[]> {
    const query = text.trim();
    if (!query) return [];
    const fishId = await this.assistant();
    const payload = await this.request<{ items?: unknown[] }>("GET",
      `/v1/fishes/${encodeURIComponent(fishId)}/messages/search?${new URLSearchParams({ q: query.slice(0, 200) })}`);
    return messages(payload.items);
  }

  /** Stops the answer being spoken; the pet stops too when it is the one speaking. */
  async interrupt(messageId: string): Promise<void> {
    const fishId = await this.assistant();
    await this.request("POST", `/v1/fishes/${encodeURIComponent(fishId)}/answer/interrupt`, { messageId, consumedOffset: 0 });
  }

  async listen(): Promise<void> {
    await this.request("POST", "/v1/pet/listen");
  }

  async stopListening(): Promise<void> {
    await this.request("POST", "/v1/pet/listen/stop");
  }

  /** The owner confirmed the sentences TA just missed were theirs. */
  async addMissedOwnerVoice(): Promise<void> {
    await this.request("POST", "/v1/pet/owner-voice/add");
  }

  private async assistant(): Promise<string> {
    const payload = await this.request<{ fishId?: unknown }>("GET", "/v1/fishes/assistant");
    if (typeof payload.fishId !== "string") throw new Error("还没有选好要聊天的角色");
    return payload.fishId;
  }

  private async request<T = unknown>(method: string, path: string, body?: unknown): Promise<T> {
    const target = await this.target();
    if (!target) throw new Error("服务正在启动，请稍后再试");
    const response = await fetch(`${target.apiUrl}${path}`, {
      method,
      headers: { Authorization: `Bearer ${target.apiKey}`, ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(15_000),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => null) as { detail?: unknown; message?: unknown } | null;
      const text = typeof detail?.detail === "string" ? detail.detail : typeof detail?.message === "string" ? detail.message : "";
      throw new Error(text || `请求失败（${response.status}）`);
    }
    if (response.status === 204) return undefined as T;
    return await response.json() as T;
  }
}

function messages(items: unknown[] | undefined): ChatMessage[] {
  return (items ?? []).flatMap((item) => { const message = chatMessage(item); return message ? [message] : []; });
}
