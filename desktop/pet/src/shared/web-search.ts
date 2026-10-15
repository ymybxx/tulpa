/**
 * Services whose own web search TA can use: 豆包 (火山方舟) and 通义千问
 * (DashScope). Others ignore the request and answer from what the model
 * already knows, which would pass for a search.
 */
export function webSearchSupported(baseUrl: string): boolean {
  let host: string;
  try { host = new URL(baseUrl).hostname; } catch { return false; }
  return /^ark\.[a-z0-9-]+\.volces\.com$/.test(host)
    || host === "dashscope.aliyuncs.com" || host === "dashscope-intl.aliyuncs.com";
}

export const WEB_SEARCH_SERVICES = "豆包和通义千问";
