import type { ChatAttachmentUpload } from "../src/shared/chat-attachments";

const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);

/** A chat box's upload, or null unless it is a well-formed picture or text file. */
export function chatAttachmentUpload(value: unknown): ChatAttachmentUpload | null {
  if (!value || typeof value !== "object") return null;
  const item = value as Record<string, unknown>;
  if (typeof item.name !== "string" || item.name.length > 200 || typeof item.mimeType !== "string" || item.mimeType.length > 100) return null;
  const size = (key: "width" | "height") =>
    typeof item[key] === "number" && Number.isInteger(item[key]) && (item[key] as number) > 0 && (item[key] as number) <= 20_000
      ? { [key]: item[key] as number } : {};
  if (item.kind === "image") {
    if (!IMAGE_TYPES.has(item.mimeType) || typeof item.data !== "string" || !item.data || item.data.length > 9_000_000) return null;
    return { kind: "image", name: item.name, mimeType: item.mimeType, data: item.data, ...size("width"), ...size("height") };
  }
  if (item.kind === "text") {
    if (typeof item.text !== "string" || !item.text.trim() || item.text.length > 100_000) return null;
    return { kind: "text", name: item.name, mimeType: item.mimeType, text: item.text };
  }
  return null;
}
