import type { SettingsSection } from "../settings-next/model";

/**
 * Which outside service let a turn down and how, as the local service names it
 * ("chat:timeout", see fish_memory.failures). Every place that tells the user
 * about a failure words it here, by service and kind; none decides on its own.
 */
export type FailedService = "jev" | "chat" | "reasoning" | "voice";
export type FailureKind = "timeout" | "connection" | "key" | "quota" | "rate_limit" | "rejected" | "error";
export interface ServiceFailure { service: FailedService; kind: FailureKind }

const SERVICES = new Set<FailedService>(["jev", "chat", "reasoning", "voice"]);
const KINDS = new Set<FailureKind>(["timeout", "connection", "key", "quota", "rate_limit", "rejected", "error"]);

/** Named as in 设置. */
const NAMES: Record<FailedService, string> = {
  jev: "决策模型", chat: "对话大脑", reasoning: "推理大脑", voice: "说话的声音",
};

const EN_NAMES: Record<FailedService, string> = {
  jev: "The decision model", chat: "The chat model", reasoning: "The reasoning model", voice: "The voice service",
};

/** Where the user fixes a key or a quota. */
const SECTIONS: Record<FailedService, SettingsSection> = { jev: "jev", chat: "ai", reasoning: "ai", voice: "ai" };

export function serviceFailure(value: unknown): ServiceFailure | null {
  if (typeof value !== "string") return null;
  const [service, kind, extra] = value.split(":");
  return extra === undefined && SERVICES.has(service as FailedService) && KINDS.has(kind as FailureKind)
    ? { service: service as FailedService, kind: kind as FailureKind } : null;
}

/** "陪伴大脑 Jev 没有": a space after a Latin name. */
function lead(service: FailedService): string {
  const name = NAMES[service];
  return /[A-Za-z0-9]$/.test(name) ? `${name} ` : name;
}

/** Why, in a few words: "连不上对话大脑". */
export function failureReason({ service, kind }: ServiceFailure, locale: "zh-CN" | "en" = "zh-CN"): string {
  if (locale === "en") {
    const name = EN_NAMES[service];
    const reasons: Record<FailureKind, string> = {
      timeout: `${name} did not respond in time`, connection: `Cannot connect to ${name}`,
      key: `${name} API key is invalid or expired`, quota: `${name} may have run out of credits`,
      rate_limit: `${name} is receiving too many requests`, rejected: `${name} rejected this message`,
      error: `${name} encountered an error`,
    };
    return reasons[kind];
  }
  switch (kind) {
    case "timeout": return `${lead(service)}没有及时回应`;
    case "connection": return `连不上${NAMES[service]}`;
    case "key": return `${lead(service)}的 Key 不对或已失效`;
    case "quota": return `${lead(service)}的额度可能用完了`;
    case "rate_limit": return `${lead(service)}说请求太频繁`;
    case "rejected": return `${lead(service)}不接受这句话的内容`;
    case "error": return `${lead(service)}出错了`;
  }
}

/** A key or quota problem stays until the user changes something in 设置. */
export function needsSettings(failure: ServiceFailure): boolean {
  return failure.kind === "key" || failure.kind === "quota";
}

export function settingsSectionFor(failure: ServiceFailure): SettingsSection {
  return SECTIONS[failure.service];
}

/**
 * What to tell the user about a reply that did not reach them: `name` is the
 * character's; `said` is whether any of it was said. A voice failure is a
 * reply shown but not heard.
 */
export function replyFailureNote(failure: ServiceFailure | null, name: string, { said }: { said: boolean }, locale: "zh-CN" | "en" = "zh-CN"): string {
  if (locale === "en") {
    const reason = failure ? failureReason(failure, locale) : "";
    if (failure?.service === "voice") return `Could not read this reply aloud: ${reason}`;
    if (said) return `This reply was interrupted${reason ? `: ${reason}` : ""}`;
    if (!failure) return `${name} could not reply. Please try again.`;
    const advice: Record<FailureKind, string> = {
      timeout: "Please try again.", connection: "Check your network or proxy.", key: "", quota: "",
      rate_limit: "Please wait a moment and try again.", rejected: "Try rephrasing your message.", error: "Please try again.",
    };
    return `${reason}. ${name} could not reply. ${advice[failure.kind]}`.trim();
  }
  if (failure?.service === "voice") return `这句没能念出来：${failureReason(failure)}`;
  if (said) return failure ? `这句没能说完：${failureReason(failure)}` : "这句没能说完";
  if (!failure) return `${name}这次没能回答，再说一遍试试`;
  const { service, kind } = failure;
  switch (kind) {
    case "timeout": return `${lead(service)}这次没有及时回应，${name}没能回答，再说一遍试试`;
    case "connection": return `连不上${NAMES[service]}，${name}没能回答。看看网络或代理`;
    case "key": return `${lead(service)}的 Key 不对或已失效，${name}没能回答`;
    case "quota": return `${lead(service)}的额度可能用完了，${name}没能回答`;
    case "rate_limit": return `${lead(service)}说请求太频繁，${name}没能回答，稍等一下再说`;
    case "rejected": return `${lead(service)}不接受这句话的内容，${name}没能回答，换个说法试试`;
    case "error": return `${lead(service)}这次出错了，${name}没能回答，再说一遍试试`;
  }
}
