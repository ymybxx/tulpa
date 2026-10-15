import { randomBytes, timingSafeEqual } from "node:crypto";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { networkInterfaces } from "node:os";
import type { GatewayConfiguration, GatewayStatus } from "../src/shared/contracts";
import { readPrivateText, writePrivateText } from "./private-file";

export const GATEWAY_PATH = "/events";
const MAX_BODY_BYTES = 64 * 1024;
/** Events per minute from everyone holding the key together. */
const MAX_PER_MINUTE = 60;

/** What the local service said about one forwarded event. */
export interface ForwardResult { status: number; body: unknown }

export interface GatewayOptions {
  /** Where the key lives; only the current OS user can read it. */
  keyFile: string;
  /** Hands a parsed event to the local service, or throws when it is not running. */
  forward(event: unknown): Promise<ForwardResult>;
  now?: () => number;
}

/**
 * Lets other programs — scripts, shortcuts, smart-home automations — send TA a
 * message: `POST /events` with `Authorization: Bearer <key>` and a JSON body.
 * It listens on every network address; the key is what keeps others out.
 */
export class GatewayServer {
  private server: Server | null = null;
  private configuration: GatewayConfiguration = { enabled: false, port: 0 };
  private key: string | null = null;
  private keyPromise: Promise<string> | null = null;
  private error: string | null = null;
  private recent: number[] = [];
  private readonly now: () => number;

  constructor(private readonly options: GatewayOptions) {
    this.now = options.now ?? Date.now;
  }

  async configure(next: GatewayConfiguration): Promise<void> {
    const same = next.enabled === this.configuration.enabled && next.port === this.configuration.port;
    this.configuration = { ...next };
    if (same && (this.server || !next.enabled)) return;
    await this.close();
    this.error = null;
    if (!next.enabled) return;
    await this.currentKey();
    const server = createServer((request, response) => void this.handle(request, response));
    try {
      await new Promise<void>((resolve, reject) => {
        server.once("error", reject);
        server.listen(next.port, "0.0.0.0", () => { server.off("error", reject); resolve(); });
      });
      this.server = server;
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      this.error = code === "EADDRINUSE"
        ? `端口 ${next.port} 被别的程序占用了，换一个端口试试`
        : `没能开始接收：${(error as Error).message}`;
    }
  }

  async close(): Promise<void> {
    const server = this.server;
    this.server = null;
    if (!server) return;
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }

  async status(): Promise<GatewayStatus> {
    const port = this.configuration.port;
    return {
      enabled: this.configuration.enabled,
      listening: this.server !== null,
      port,
      addresses: [`http://127.0.0.1:${port}${GATEWAY_PATH}`,
        ...localAddresses().map((address) => `http://${address}:${port}${GATEWAY_PATH}`)],
      key: await this.currentKey(),
      error: this.error,
    };
  }

  /** A new key; whatever used the old one has to be given this one. */
  async resetKey(): Promise<GatewayStatus> {
    if (this.keyPromise) await this.keyPromise;
    const next = randomBytes(32).toString("base64url");
    await this.saveKey(next);
    this.key = next;
    return this.status();
  }

  private async currentKey(): Promise<string> {
    if (this.key) return this.key;
    const pending = this.keyPromise ??= this.loadOrCreateKey();
    try { return await pending; }
    finally { if (this.keyPromise === pending) this.keyPromise = null; }
  }

  private async loadOrCreateKey(): Promise<string> {
    try {
      const saved = (await readPrivateText(this.options.keyFile)).trim();
      if (/^[A-Za-z0-9_-]{32,}$/.test(saved)) return (this.key = saved);
      throw new Error("本地接收 Key 文件无效，请在设置中手动重置");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    const next = randomBytes(32).toString("base64url");
    await this.saveKey(next);
    return (this.key = next);
  }

  private async saveKey(key: string): Promise<void> {
    await writePrivateText(this.options.keyFile, `${key}\n`);
  }

  private async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const reply = (status: number, body: Record<string, unknown>): void => {
      if (response.headersSent) return;
      response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
      response.end(JSON.stringify(body));
    };
    try {
      const url = new URL(request.url ?? "/", "http://gateway");
      if (url.pathname !== GATEWAY_PATH) return reply(404, { error: `只接收 POST ${GATEWAY_PATH}` });
      if (request.method !== "POST") return reply(405, { error: `只接收 POST ${GATEWAY_PATH}` });
      if (!this.authorized(request.headers.authorization)) {
        return reply(401, { error: "钥匙不对：请求头要带 Authorization: Bearer <钥匙>" });
      }
      // Only senders with the key count, so strangers cannot crowd them out.
      if (!this.allowed()) return reply(429, { error: "发得太快了，过一会儿再发" });
      if (!String(request.headers["content-type"] ?? "").toLowerCase().startsWith("application/json")) {
        return reply(415, { error: "请用 JSON 发送（Content-Type: application/json）" });
      }
      const text = await readBody(request);
      if (text === null) return reply(413, { error: "内容太长了" });
      let event: unknown;
      try { event = JSON.parse(text); } catch { return reply(400, { error: "不是有效的 JSON" }); }
      let result: ForwardResult;
      try {
        result = await this.options.forward(event);
      } catch {
        return reply(503, { error: "TA 现在不在线，稍后再发" });
      }
      if (result.status === 202) return reply(202, result.body as Record<string, unknown>);
      if (result.status === 400 || result.status === 422) {
        const detail = (result.body as { error?: { message?: unknown } } | null)?.error?.message;
        return reply(400, {
          error: "内容不对：需要 text，可选 title、source、id、speak",
          ...(typeof detail === "string" ? { detail } : {}),
        });
      }
      if (result.status === 409) return reply(409, { error: "还没有可以转告的角色" });
      return reply(503, { error: "TA 现在不在线，稍后再发" });
    } catch {
      reply(500, { error: "出了点问题，没收到" });
    }
  }

  private allowed(): boolean {
    const now = this.now();
    this.recent = this.recent.filter((time) => now - time < 60_000);
    if (this.recent.length >= MAX_PER_MINUTE) return false;
    this.recent.push(now);
    return true;
  }

  private authorized(header: string | undefined): boolean {
    const [scheme, token] = (header ?? "").split(" ");
    if (scheme?.toLowerCase() !== "bearer" || !token || !this.key) return false;
    const given = Buffer.from(token);
    const expected = Buffer.from(this.key);
    return given.length === expected.length && timingSafeEqual(given, expected);
  }
}

async function readBody(request: IncomingMessage): Promise<string | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    size += (chunk as Buffer).length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks).toString("utf8");
}

/**
 * This Mac's addresses that other devices can reach: home and office networks
 * and Tailscale-style ones. A proxy's virtual card (198.18.x) or a link-local
 * address would only mislead whoever copies it.
 */
export function localAddresses(interfaces = networkInterfaces()): string[] {
  return Object.values(interfaces).flatMap((items) => (items ?? [])
    .filter((item) => item.family === "IPv4" && !item.internal && reachable(item.address))
    .map((item) => item.address));
}

function reachable(address: string): boolean {
  const [a, b] = address.split(".").map(Number) as [number, number];
  return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168)
    || (a === 100 && b >= 64 && b <= 127);
}
