/**
 * Token use by what it was for, from the local service's call records. The
 * person sees purposes they can switch off, not call types.
 */
export type UsageGroup = "reply" | "proactive" | "inner" | "senses" | "memory" | "facial" | "other";

/** One row of /internal/llm-logs/usage: a call type and purpose over time windows. */
export interface UsageRow {
  call_type: string;
  purpose: string;
  today_calls: number;
  today_tokens: number;
  week_calls: number;
  week_tokens: number;
  hour_calls: number;
  hour_tokens: number;
  recent_calls: number;
}

export interface UsageTotals { calls: number; tokens: number }
export interface UsageGroupSummary {
  group: UsageGroup;
  label: string;
  /** Runs without the person asking for it. */
  background: boolean;
  today: UsageTotals;
  week: UsageTotals;
  hour: UsageTotals;
  /** Called in the last ten minutes. */
  active: boolean;
}
export interface UsageSummary { groups: UsageGroupSummary[]; today: UsageTotals }

export const USAGE_GROUPS: ReadonlyArray<{ group: UsageGroup; label: string; background: boolean; detail: string }> = [
  { group: "reply", label: "回复你", background: false, detail: "你说话或打字后，TA 想怎么回、说出来" },
  { group: "proactive", label: "主动找你聊", background: true, detail: "TA 自己决定开口，以及把查到的结果告诉你" },
  { group: "inner", label: "内心", background: true, detail: "闲下来自己想事情、找话题" },
  { group: "senses", label: "看屏幕和听声音", background: true, detail: "识别屏幕画面、总结电脑里的声音" },
  { group: "memory", label: "整理记忆", background: true, detail: "聊完后把值得记住的事整理下来" },
  { group: "facial", label: "说话表情", background: true, detail: "每句话配上表情和动作" },
  { group: "other", label: "其他", background: true, detail: "插件、换装和其他后台处理" },
];

const INNER = new Set(["experimental.thought.generate", "experimental.jev.thought", "experimental.judge.thought", "experimental.material.browse"]);
const SENSES = new Set(["environment", "desktop.vision", "experimental.jev.observation", "experimental.judge.observation"]);
const MEMORY = new Set(["memory.extract", "memory.static", "memory.dynamic", "recall.plan"]);
const FACIAL = new Set(["speech.facial.jev", "demo.facial.jev", "desktop.emotion"]);
const REPLY = new Set([
  "chat", "experimental.jev.decision", "experimental.judge.decision", "experimental.web.result",
  "experimental.recall.summary", "experimental.result.summary",
]);

export function usageGroupOf(callType: string, purpose: string): UsageGroup {
  if (purpose === "proactive") return "proactive";
  if (INNER.has(callType)) return "inner";
  if (SENSES.has(callType)) return "senses";
  if (MEMORY.has(callType)) return "memory";
  if (FACIAL.has(callType)) return "facial";
  if (REPLY.has(callType) || callType.startsWith("experimental.channel.")) return "reply";
  return "other";
}

export function summarizeUsage(rows: readonly UsageRow[]): UsageSummary {
  const zero = (): UsageTotals => ({ calls: 0, tokens: 0 });
  const groups = USAGE_GROUPS.map(({ group, label, background }): UsageGroupSummary => ({
    group, label, background, today: zero(), week: zero(), hour: zero(), active: false,
  }));
  const today = zero();
  for (const row of rows) {
    const summary = groups.find((item) => item.group === usageGroupOf(row.call_type, row.purpose))!;
    summary.today.calls += row.today_calls; summary.today.tokens += row.today_tokens;
    summary.week.calls += row.week_calls; summary.week.tokens += row.week_tokens;
    summary.hour.calls += row.hour_calls; summary.hour.tokens += row.hour_tokens;
    summary.active ||= row.recent_calls > 0;
    today.calls += row.today_calls; today.tokens += row.today_tokens;
  }
  return { groups, today };
}

/** "约 3.2 万" rather than a long number. */
export function tokenText(tokens: number, locale: "zh-CN" | "en" = "zh-CN"): string {
  if (locale === "en") return tokens <= 0 ? "0" : `≈ ${new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(tokens)}`;
  if (tokens <= 0) return "0";
  if (tokens < 10_000) return `约 ${Math.round(tokens / 100) * 100 || tokens}`;
  return `约 ${(tokens / 10_000).toFixed(tokens < 100_000 ? 1 : 0)} 万`;
}

/** Background purposes that called a model in the last ten minutes. */
export function activeBackground(summary: UsageSummary): string[] {
  return summary.groups.filter((item) => item.background && item.active).map((item) => item.label);
}

export function parseUsageRows(value: unknown): UsageRow[] {
  const items = (value as { items?: unknown })?.items;
  if (!Array.isArray(items)) return [];
  const number = (item: Record<string, unknown>, key: string): number =>
    typeof item[key] === "number" && Number.isFinite(item[key]) ? item[key] as number : 0;
  return items.flatMap((raw) => {
    const item = raw as Record<string, unknown>;
    if (typeof item?.call_type !== "string") return [];
    return [{
      call_type: item.call_type, purpose: typeof item.purpose === "string" ? item.purpose : "",
      today_calls: number(item, "today_calls"), today_tokens: number(item, "today_tokens"),
      week_calls: number(item, "week_calls"), week_tokens: number(item, "week_tokens"),
      hour_calls: number(item, "hour_calls"), hour_tokens: number(item, "hour_tokens"),
      recent_calls: number(item, "recent_calls"),
    }];
  });
}
