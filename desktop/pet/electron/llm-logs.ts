import { randomUUID } from "node:crypto";
import type { LlmLogFilter } from "../src/shared/llm-logs";

type Connection = { apiUrl: string; apiKey: string };
let connection: (() => Promise<Connection>) | undefined;
export function configureLlmLogs(value: () => Promise<Connection>): void {
  connection = value;
}

async function api(path: string, body?: unknown): Promise<any> {
  if (!connection) throw new Error("本地日志服务尚未启动");
  const target = await connection();
  const response = await fetch(`${target.apiUrl}/internal/llm-logs${path}`, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${target.apiKey}`,
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(15_000),
  });
  if (!response.ok) throw new Error(`读取调用日志失败（${response.status}）`);
  return response.json();
}
export async function listLlmLogs(filter: LlmLogFilter = {}): Promise<any> {
  const query = new URLSearchParams();
  for (const key of ["call_type", "status", "before", "limit"] as const) {
    if (filter[key] !== undefined && filter[key] !== "")
      query.set(key, String(filter[key]));
  }
  return api(`?${query}`);
}
export const llmLogTypes = () => api("/types");
export function llmLogDetail(id: string): Promise<any> {
  if (!/^[0-9a-f-]{36}$/i.test(id)) throw new Error("调用记录 ID 无效");
  return api(`/${id}`);
}

/** Record the three desktop-side VLM clients through the same local log store. */
export function loggedFetch(
  kind: string,
  fetcher: (...args: any[]) => Promise<Response> = fetch,
): typeof fetch {
  return async (input, init) => {
    if (!connection || typeof init?.body !== "string")
      return fetcher(input, init);
    const started = Date.now();
    const url = new URL(input instanceof Request ? input.url : String(input));
    url.search = "";
    url.hash = "";
    url.username = "";
    url.password = "";
    let request: any;
    try {
      request = JSON.parse(init.body);
    } catch {
      request = init.body;
    }
    const base = {
      id: randomUUID(),
      call_type: kind,
      model: request?.model ?? "",
      request_url: url.href,
      started_at: new Date(started).toISOString().replace(/\.\d{3}Z$/, "Z"),
      request,
    };
    const record = (
      status: string,
      response: unknown,
      http_status: number | null,
      error: string | null,
    ) => {
      void api("", {
        ...base,
        status,
        response,
        http_status,
        error,
        ended_at: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
        duration_ms: Date.now() - started,
      }).catch(() => console.warn("未能保存一条模型调用日志"));
    };
    try {
      const response = await fetcher(input, init);
      let copy: Response;
      try {
        copy = response.clone();
      } catch {
        // A provider adapter can return an already-consumed response. Logging
        // must not replace it with a new error or change the caller's behavior.
        console.warn("无法复制模型返回用于记录日志");
        return response;
      }
      void captureBody(copy)
        .then(({ raw, truncated }) => {
          let body: unknown = raw;
          try {
            body = JSON.parse(raw);
          } catch {
            /* Preserve malformed replies. */
          }
          record(
            response.ok ? "completed" : "failed",
            { body, raw, truncated },
            response.status,
            null,
          );
        })
        .catch((error) =>
          record(
            init?.signal?.aborted ? "cancelled" : "failed",
            null,
            response.status,
            error?.name ?? "读取响应失败",
          ),
        );
      return response;
    } catch (error) {
      record(
        init?.signal?.aborted ? "cancelled" : "failed",
        null,
        null,
        error instanceof Error ? error.name : "请求失败",
      );
      throw error;
    }
  };
}

async function captureBody(
  response: Response,
): Promise<{ raw: string; truncated: boolean }> {
  if (!response.body) return { raw: "", truncated: false };
  const reader = response.body.getReader();
  const parts: string[] = [];
  const decoder = new TextDecoder();
  let bytes = 0;
  let truncated = false;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const remaining = 16 * 1024 * 1024 - bytes;
      parts.push(
        decoder.decode(value.subarray(0, remaining), { stream: true }),
      );
      bytes += value.length;
      if (value.length > remaining) {
        truncated = true;
        void reader.cancel().catch(() => undefined);
        break;
      }
    }
    parts.push(decoder.decode());
    return { raw: parts.join(""), truncated };
  } finally {
    reader.releaseLock();
  }
}

/** Token use by call type and purpose since `dayStart` (the local midnight). */
export async function llmUsage(dayStart: string): Promise<unknown> {
  return api(`/usage?${new URLSearchParams({ dayStart })}`);
}
