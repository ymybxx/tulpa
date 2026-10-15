export interface LlmLogFilter {
  call_type?: string;
  status?: string;
  before?: number;
  limit?: number;
}
export interface LlmLogSummary {
  id: string;
  sequence: number;
  call_type: string;
  model: string;
  status: "running" | "completed" | "failed" | "cancelled";
  started_at: string;
  ended_at: string | null;
  duration_ms: number | null;
  http_status: number | null;
  error: string | null;
  source: string;
}
export interface LlmLogPage {
  items: LlmLogSummary[];
  next_cursor: number | null;
  dropped_records: number;
}
export interface LlmLogDetail extends LlmLogSummary {
  request_url: string;
  request: unknown;
  response: unknown;
}

export const LOG_TYPE_NAMES: Record<string, string> = {
  chat: "聊天与主动发言",
  reasoning: "辅助推理",
  "experimental.thought.generate": "后台想法生成",
  "experimental.jev.thought": "后台想法选择",
  "experimental.judge.thought": "后台想法选择（推理模型）",
  "experimental.judge.decision": "陪伴决策（推理模型）",
  "experimental.judge.observation": "影音观察筛选（推理模型）",
  "experimental.jev.decision": "陪伴决策",
  "experimental.jev.observation": "影音观察筛选",
  "experimental.channel.s": "回复：直接说",
  "experimental.channel.r": "回复：先回忆",
  "experimental.channel.w": "回复：先查资料",
  "experimental.channel.t": "回复：交给 Agent",
  "experimental.web.result": "联网搜索",
  "experimental.recall.summary": "回忆结果整理",
  "experimental.result.summary": "任务结果整理",
  "experimental.material.browse": "闲时找话题",
  "speech.facial.jev": "说话表情",
  "demo.facial.jev": "表情测试",
  environment: "环境感知",
  "plugin.speech": "插件发言",
  "plugin.select": "直播选题",
  "plugin.repetition": "直播重复判断",
  "plugin.research": "直播研究判断",
  "plugin.reasoning": "插件其他推理",
  "host.event": "事件发言",
  "memory.extract": "记忆提取",
  "memory.static": "静态记忆整理",
  "memory.dynamic": "动态记忆整理",
  "recall.plan": "记忆检索规划",
  "desktop.emotion": "模型表情分析",
  "desktop.texture": "AI 外观编辑",
  "desktop.parts": "识图分部位",
  "desktop.vision": "屏幕识别",
  other: "其他调用",
};
export const LOG_STATUS_NAMES = {
  running: "进行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};
export function logTypeName(value: string): string {
  return LOG_TYPE_NAMES[value] ?? value;
}

export function parsedText(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const text = value.trim();
  if (!text.startsWith("{") && !text.startsWith("[")) return value;
  try {
    return JSON.parse(text);
  } catch {
    return value;
  }
}
export function responseContent(value: unknown): unknown {
  if (!value || typeof value !== "object") return value;
  const response = value as Record<string, any>;
  if (typeof response.text === "string" && response.text)
    return parsedText(response.text);
  const body = response.body;
  const message = body?.choices?.[0]?.message;
  if (typeof message?.content === "string") return parsedText(message.content);
  return body ?? value;
}
