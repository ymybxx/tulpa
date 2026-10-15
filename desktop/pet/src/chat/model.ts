import { t, getLocale } from "../i18n";
import type { ChatMessage, ChatOrigin } from "../shared/chat";
import type { PetState } from "../shared/contracts";

export type ChatRow =
  | { kind: "divider"; key: string; label: string }
  | { kind: "message"; key: string; message: ChatMessage; time: string };

/** A new divider after this long without a message, as well as at each new day. */
const GAP_MS = 30 * 60_000;

const clock = (date: Date) => date.toLocaleTimeString(getLocale(), { hour: "2-digit", minute: "2-digit", hour12: false });
const dayStart = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();

/** 「今天 10:12」「昨天 23:41」「9月24日 23:41」, with the year only when it differs. */
export function dayLabel(date: Date, now: Date): string {
  const days = Math.round((dayStart(now) - dayStart(date)) / 86_400_000);
  const day = days === 0 ? t("今天") : days === 1 ? t("昨天")
    : date.toLocaleDateString(getLocale(), { ...(date.getFullYear() !== now.getFullYear() ? { year: "numeric" as const } : {}), month: getLocale() === "en" ? "short" : "long", day: "numeric" });
  return `${day} ${clock(date)}`;
}

/**
 * The conversation as rows: a date line where a new stretch of talk starts,
 * marked 「新对话」 where TA started over, then the messages.
 */
export function timelineRows(messages: readonly ChatMessage[], now: Date): ChatRow[] {
  const rows: ChatRow[] = [];
  let previous: ChatMessage | null = null;
  for (const message of messages) {
    // A reply cut off before its first word said nothing; an empty
    // "说到这里被你打断了" bubble only confuses.
    if (message.role === "assistant" && message.status === "interrupted"
      && !message.content.trim() && !message.attachments?.length) continue;
    const at = new Date(message.createdAt);
    const before = previous ? new Date(previous.createdAt) : null;
    const fresh = Boolean(previous?.sessionId && message.sessionId && previous.sessionId !== message.sessionId);
    if (fresh) {
      rows.push({ kind: "divider", key: `s-${message.id}`, label: newConversationLabel(at, now) });
    } else if (!before || dayStart(before) !== dayStart(at) || at.getTime() - before.getTime() >= GAP_MS) {
      rows.push({ kind: "divider", key: `d-${message.id}`, label: dayLabel(at, now) });
    }
    rows.push({ kind: "message", key: message.id, message, time: clock(at) });
    previous = message;
  }
  return rows;
}

/** 「新对话 · 今天 22:50」: where TA started over without the talk before it. */
export function newConversationLabel(date: Date, now: Date): string {
  return t("新对话 · {{time}}", { time: dayLabel(date, now) });
}

/** Adds or refreshes messages by ID, oldest first. */
export function mergeMessages(current: readonly ChatMessage[], incoming: readonly ChatMessage[]): ChatMessage[] {
  const byId = new Map(current.map((message) => [message.id, message]));
  for (const message of incoming) byId.set(message.id, message);
  return [...byId.values()].sort((a, b) => a.sequence - b.sequence);
}

/** The small line above a message TA sent unasked, e.g. 「看到你在用 VS Code」. */
export function originLabel(origin: ChatOrigin, name: string): string {
  switch (origin.kind) {
    case "sense": return origin.appName ? t("看到你在用 {{app}}", { app: origin.appName }) : t("看到听到了新东西");
    case "event": return origin.appName ? t("{{app}} 有新动静", { app: origin.appName }) : t("有新动静");
    case "plugin": return t("插件提醒");
    case "task": return t("之前交代的事办好了");
    case "self": return t("{{name}}自己想说的", { name });
  }
}

/**
 * What the stage says TA is doing. At rest it follows 主动陪伴: "陪伴中" only
 * when TA may speak up by itself, so the word never contradicts that switch.
 */
export function stateLabel(state: PetState | null, companionOn = true): string {
  switch (state) {
    case "listening": case "acknowledging": return t("在听你说");
    case "thinking": return t("在想");
    case "speaking": return t("在说话");
    case "unbound": return t("还没醒来");
    default: return companionOn ? t("陪伴中") : t("等你来聊");
  }
}

/** The hover line under the stage label: what the label is about. */
export function stateHint(state: PetState | null, companionOn = true): string {
  if (state && state !== "standby" && state !== "unbound") return t("TA 此刻在做的事");
  if (state === "unbound") return t("TA 还在启动，稍等一下");
  return companionOn
    ? t("TA 此刻闲着；主动陪伴开着，看到有意思的会主动找你聊")
    : t("TA 此刻闲着；主动陪伴没开，TA 不会自己开口，找 TA 就行");
}

/** The part of a found message around the first match, for a result line. */
export function snippet(text: string, query: string, room = 24): { pre: string; hit: string; post: string } {
  const at = text.toLowerCase().indexOf(query.toLowerCase());
  if (at < 0 || !query) return { pre: text.slice(0, room * 2), hit: "", post: "" };
  const start = Math.max(0, at - room);
  const end = Math.min(text.length, at + query.length + room);
  return {
    pre: `${start > 0 ? "…" : ""}${text.slice(start, at)}`,
    hit: text.slice(at, at + query.length),
    post: `${text.slice(at + query.length, end)}${end < text.length ? "…" : ""}`,
  };
}

export const STARTERS = ["你能陪我做什么？", "介绍一下你自己", "随便聊聊今天"];
