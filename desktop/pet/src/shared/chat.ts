import type { ChatAttachmentUpload } from "./chat-attachments";
import type { SenseAction, SensesSnapshot } from "./pet-senses";
import { serviceFailure, type ServiceFailure } from "./service-failures";

/** One message of the conversation with the current character, as the chat page shows it. */
export interface ChatMessage {
  id: string;
  sequence: number;
  role: "user" | "assistant";
  /** What was actually said: an interrupted answer only up to where it stopped. */
  content: string;
  status: "initialized" | "generating" | "completed" | "interrupted" | "failed";
  createdAt: string;
  /** Set when TA spoke up without being asked: what prompted it. */
  origin?: ChatOrigin;
  /** An interrupted answer's rest: prepared but never said. */
  unsaid?: string;
  /** A user message said to the pet rather than typed; unknown for older ones. */
  spoken?: boolean;
  /** Pictures and files sent with a user message; its text names each one. */
  attachments?: ChatAttachment[];
  /** Which outside service let this reply down: failed, or shown but never heard. */
  failure?: ServiceFailure;
  /** The conversation it belongs to: the page marks where a 「新对话」 starts. */
  sessionId?: string;
}

export interface ChatAttachment { id: string; kind: "image" | "text"; name: string }

/**
 * Why TA spoke up: something seen or heard (with the app, when known), another
 * event, a plugin, a finished task, or TA's own idea. Recorded facts, never
 * a reason the model made up.
 */
export interface ChatOrigin { kind: "sense" | "event" | "plugin" | "task" | "self"; appName?: string }
const ORIGINS = new Set(["sense", "event", "plugin", "task", "self"]);

export interface ChatPage {
  fishId: string;
  items: ChatMessage[];
  hasBefore: boolean;
}

export interface ChatSendResult { ok: boolean; error?: string; retryable?: boolean }

/**
 * The main window's chat: the same conversation and the same sending path as
 * the pet's 「聊两句」, so the pet speaks the reply and both show it.
 */
export interface ChatApi {
  /** The newest messages, or those before `before` (a sequence), oldest first. */
  page(before?: number): Promise<ChatPage>;
  search(query: string): Promise<ChatMessage[]>;
  /** `attachmentIds` come from attach(); `content` already names each one. */
  send(requestId: string, content: string, attachmentIds?: string[]): Promise<ChatSendResult>;
  /** Keeps a picture or text file until the message carrying it is sent. */
  attach(upload: ChatAttachmentUpload): Promise<{ ok: true; id: string } | { ok: false; error: string }>;
  /** A sent picture, as a data URL to show in the history. */
  attachmentImage(id: string): Promise<string | null>;
  interrupt(messageId: string): Promise<void>;
  /** Start the pet listening, as if its wake word was heard. */
  listen(): Promise<void>;
  /** 「新对话」: TA stops carrying on the conversation so far; what it remembers stays. */
  newConversation(): Promise<{ ok: boolean; error?: string }>;
  /** A new conversation started here or from the pet's 「聊两句」. */
  onNewConversation(callback: (sessionId: string) => void): () => void;
  /** Stop listening: the conversation ends as when TA ends it. */
  stopListening(): Promise<void>;
  senses(): Promise<SensesSnapshot | null>;
  senseAction(action: Exclude<SenseAction, { type: "panel" }>): Promise<void>;
  /** The pet's latest state event, if one arrived yet. */
  petState(): Promise<unknown>;
  /** The pet's own events: its state and what it is saying right now. */
  onPetEvent(callback: (event: unknown) => void): () => void;
  onSenses(callback: (snapshot: SensesSnapshot | null) => void): () => void;
}

const STATUSES = new Set(["initialized", "generating", "completed", "interrupted", "failed"]);

/** Reads one message from the local service, or null when it is not one. */
export function chatMessage(value: unknown): ChatMessage | null {
  if (!value || typeof value !== "object") return null;
  const item = value as Record<string, unknown>;
  if (typeof item.messageId !== "string" || !Number.isSafeInteger(item.sequence)
    || (item.role !== "user" && item.role !== "assistant") || typeof item.content !== "string"
    || typeof item.status !== "string" || !STATUSES.has(item.status) || typeof item.createdAt !== "string") return null;
  const origin = item.origin as Record<string, unknown> | undefined;
  return { id: item.messageId, sequence: item.sequence as number, role: item.role, content: item.content,
    status: item.status as ChatMessage["status"], createdAt: item.createdAt,
    ...(typeof item.unsaid === "string" && item.unsaid ? { unsaid: item.unsaid } : {}),
    ...(item.inputMode === "spoken" ? { spoken: true } : {}),
    ...(typeof item.sessionId === "string" ? { sessionId: item.sessionId } : {}),
    ...(serviceFailure(item.failure) ? { failure: serviceFailure(item.failure)! } : {}),
    ...(Array.isArray(item.attachments) && item.attachments.length ? {
      attachments: item.attachments.flatMap((raw): ChatAttachment[] => {
        const entry = raw as Record<string, unknown>;
        return typeof entry.id === "string" && (entry.kind === "image" || entry.kind === "text") && typeof entry.name === "string"
          ? [{ id: entry.id, kind: entry.kind, name: entry.name }] : [];
      }),
    } : {}),
    ...(origin && typeof origin.kind === "string" && ORIGINS.has(origin.kind) ? {
      origin: { kind: origin.kind as ChatOrigin["kind"], ...(typeof origin.appName === "string" && origin.appName ? { appName: origin.appName } : {}) },
    } : {}) };
}
