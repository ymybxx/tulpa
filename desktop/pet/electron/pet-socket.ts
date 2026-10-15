import { EventEmitter } from "node:events";

import WebSocket, { type RawData } from "ws";

export interface PetSocketOptions {
  apiUrl: string;
  apiKey?: string;
}

export class PetSocket extends EventEmitter {
  private readonly url: string;
  private readonly apiKey?: string;
  private socket: WebSocket | null = null;
  private retryTimer: NodeJS.Timeout | null = null;
  private retryMs = 250;
  private stopped = true;

  constructor(options: PetSocketOptions) {
    super();
    this.url = websocketUrl(options.apiUrl, "/v1/pet/events");
    this.apiKey = options.apiKey;
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.socket?.terminate();
    this.socket = null;
    this.emit("connection", false);
  }

  private connect(): void {
    if (this.stopped || this.socket) return;
    const headers = this.apiKey
      ? { Authorization: `Bearer ${this.apiKey}` }
      : undefined;
    const socket = new WebSocket(this.url, { headers, maxPayload: 8 * 1024 * 1024 });
    this.socket = socket;

    socket.on("open", () => {
      if (socket !== this.socket) return;
      this.retryMs = 250;
      this.emit("connection", true);
    });
    socket.on("message", (data, isBinary) => this.forward(data, isBinary));
    socket.on("error", () => {
      // close triggers the bounded reconnect path below.
    });
    socket.on("close", () => {
      if (socket !== this.socket) return;
      this.socket = null;
      this.emit("connection", false);
      this.scheduleReconnect();
    });
  }

  private forward(data: RawData, isBinary: boolean): void {
    if (isBinary) {
      const buffer = Buffer.isBuffer(data)
        ? data
        : Buffer.concat(Array.isArray(data) ? data : [Buffer.from(data as ArrayBuffer)]);
      this.emit("audio", buffer);
      return;
    }
    try {
      const parsed: unknown = JSON.parse(data.toString());
      if (
        typeof parsed === "object" &&
        parsed !== null &&
        (parsed as { type?: unknown }).type !== "pet.ping"
      ) {
        this.emit("control", parsed);
      }
    } catch {
      // Ignore malformed server messages; the authenticated socket remains usable.
    }
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.retryTimer) return;
    const delay = this.retryMs;
    this.retryMs = Math.min(5_000, this.retryMs * 2);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }
}

export function websocketUrl(baseUrl: string, pathname: string): string {
  const url = new URL(baseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = pathname;
  url.search = "";
  url.hash = "";
  return url.toString();
}
