import type { AiServiceCheckRequest, AiServiceCheckResult } from "../src/shared/contracts";
import { JEV_DEFAULT_MODEL, JEV_ENDPOINT } from "../src/shared/decision-model";

type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

const TIMEOUT_MS = 15_000;

/**
 * Sends the smallest possible chat completion so onboarding can confirm the
 * key, address and model name before the memory runtime is configured.
 */
export async function checkAiService(
  request: unknown,
  fetcher: Fetcher = fetch,
): Promise<AiServiceCheckResult> {
  const { baseUrl, model, apiKey } = validateRequest(request);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetcher(`${baseUrl}/chat/completions`, {
      method: "POST",
      signal: controller.signal,
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }], max_tokens: 1 }),
    });
    if (response.ok) return { ok: true };
    return { ok: false, message: describeFailure(response.status) };
  } catch (error) {
    return {
      ok: false,
      message: controller.signal.aborted ? "连接超时，请检查网络或接口地址" : "连不上这个服务，请检查网络或接口地址",
    };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Asks Jev one trivial choice so onboarding can confirm the key before
 * 主动陪伴 is switched on. Same endpoint and body shape the runtime uses.
 */
export async function checkJevService(
  request: unknown,
  fetcher: Fetcher = fetch,
): Promise<AiServiceCheckResult> {
  const record = typeof request === "object" && request !== null ? request as Record<string, unknown> : {};
  const apiKey = typeof record.apiKey === "string" ? record.apiKey.trim() : "";
  const model = typeof record.model === "string" && record.model.trim() ? record.model.trim() : JEV_DEFAULT_MODEL;
  if (!apiKey) throw new Error("请填写 Jev API Key");
  return checkDecisionService({ endpoint: JEV_ENDPOINT, model, apiKey },
    { timeout: "连接 Jev 超时，请检查网络", unreachable: "连不上 Jev，请检查网络" }, fetcher);
}

/** One trivial question to the decision model (百炼 or Jev), as the runtime would ask it. */
export async function checkDecisionService(
  connection: { endpoint: string; model: string; apiKey: string },
  failure: { timeout: string; unreachable: string; denied?: string },
  fetcher: Fetcher = fetch,
): Promise<AiServiceCheckResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetcher(connection.endpoint, {
      method: "POST",
      signal: controller.signal,
      headers: { Authorization: `Bearer ${connection.apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: connection.model,
        state: "连接检查",
        questions: { ready: { type: "choice", instructions: "选 yes。", criteria: { yes: "可以", no: "不可以" } } },
      }),
    });
    if (response.ok) return { ok: true };
    const denied = response.status === 401 || response.status === 403;
    return { ok: false, message: denied && failure.denied ? failure.denied : describeFailure(response.status) };
  } catch {
    return { ok: false, message: controller.signal.aborted ? failure.timeout : failure.unreachable };
  } finally {
    clearTimeout(timer);
  }
}

function describeFailure(status: number): string {
  if (status === 401 || status === 403) return "Key 不对或没有权限";
  if (status === 404) return "找不到这个接口地址或模型";
  if (status === 429) return "请求太频繁或额度用完了";
  if (status >= 500) return `服务暂时不可用（${status}）`;
  return `服务返回了错误（${status}）`;
}

function validateRequest(value: unknown): AiServiceCheckRequest {
  if (typeof value !== "object" || value === null) throw new Error("AI 服务参数无效");
  const record = value as Record<string, unknown>;
  const apiKey = typeof record.apiKey === "string" ? record.apiKey.trim() : "";
  const model = typeof record.model === "string" ? record.model.trim() : "";
  if (!apiKey) throw new Error("请填写 API Key");
  if (!model) throw new Error("请填写模型名称");
  let url: URL;
  try {
    url = new URL(typeof record.baseUrl === "string" ? record.baseUrl.trim() : "");
  } catch {
    throw new Error("接口地址无效");
  }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) {
    throw new Error("接口地址必须是 HTTP(S) 地址");
  }
  return { baseUrl: url.href.replace(/\/$/, ""), model, apiKey };
}
