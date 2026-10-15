/**
 * Pictures and text files sent in chat. The message text names each one, so
 * History, Jev and memory see "[发来图片：a.png]"; only the reply to that
 * message receives the picture or the file's text itself.
 */

export interface ChatAttachmentUpload {
  kind: "image" | "text";
  name: string;
  mimeType: string;
  /** Base64 picture bytes, for an image. */
  data?: string;
  /** The file's text, for a text file. */
  text?: string;
  width?: number;
  height?: number;
}

/** An attachment chosen in a chat box, uploaded when the message is sent. */
export interface PendingAttachment {
  key: string;
  upload: ChatAttachmentUpload;
  /** A data URL shown in the chat box before sending. */
  preview?: string;
}

export const MAX_ATTACHMENTS = 8;
const MAX_TEXT_CHARACTERS = 100_000;
const MAX_IMAGE_SIDE = 1600;
const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);
const TEXT_EXTENSIONS = new Set([
  "txt", "md", "markdown", "csv", "tsv", "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf", "log",
  "xml", "html", "htm", "css", "js", "mjs", "cjs", "ts", "tsx", "jsx", "py", "rb", "go", "rs", "java", "kt",
  "swift", "c", "h", "cpp", "hpp", "cs", "php", "sh", "zsh", "bash", "sql", "vue", "svelte", "tex", "srt",
]);
const DOCUMENT_EXTENSIONS = new Set(["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "pages", "numbers", "key"]);

export type AttachmentKind = "image" | "text" | "document" | "other";

/** What a chosen file is, from its type and name. */
export function attachmentKind(file: { name: string; type: string }): AttachmentKind {
  const extension = file.name.includes(".") ? file.name.split(".").pop()!.toLowerCase() : "";
  if (IMAGE_TYPES.has(file.type)) return "image";
  if (DOCUMENT_EXTENSIONS.has(extension)) return "document";
  if (file.type.startsWith("text/") || TEXT_EXTENSIONS.has(extension) || file.type === "application/json") return "text";
  return "other";
}

/** How a message names what it carries, e.g. "[发来图片：a.png]". */
export function attachmentLabel(kind: "image" | "text", name: string): string {
  return `[${kind === "image" ? "发来图片" : "发来文件"}：${name}]`;
}

/** The words typed, followed by a name for each attachment. */
export function contentWithAttachments(text: string, attachments: readonly { kind: "image" | "text"; name: string }[]): string {
  return [text.trim(), ...attachments.map((item) => attachmentLabel(item.kind, item.name))].filter(Boolean).join(" ");
}

/** The typed words of a message, without the names of what it carried. */
export function contentWithoutAttachments(content: string, attachments: readonly { kind: "image" | "text"; name: string }[]): string {
  let text = content;
  for (const item of attachments) text = text.replace(attachmentLabel(item.kind, item.name), "");
  return text.trim();
}

/**
 * Reads a chosen file for sending, or says plainly why it cannot go.
 * Pictures are scaled to at most 1600 pixels a side.
 */
export async function readAttachment(
  file: File,
  { imagesAllowed, locale = "zh-CN" }: { imagesAllowed: boolean; locale?: "zh-CN" | "en" },
): Promise<PendingAttachment | { error: string }> {
  const copy = (zh: string, en: string) => locale === "en" ? en : zh;
  const kind = attachmentKind(file);
  const key = `${file.name}:${file.size}:${file.lastModified}:${Math.random()}`;
  if (kind === "image") {
    if (!imagesAllowed) return { error: copy("现在的对话大脑不能看图，换一个能看图的模型才能发图片", "Your chat model does not support images. Select one that does to send pictures.") };
    const image = await scaledImage(file);
    if (!image) return { error: copy(`「${file.name}」打不开，换一张图片试试`, `Cannot open “${file.name}”. Try another picture.`) };
    return {
      key,
      preview: image.dataUrl,
      upload: {
        kind: "image", name: file.name || "图片", mimeType: image.mimeType,
        data: image.dataUrl.slice(image.dataUrl.indexOf(",") + 1), width: image.width, height: image.height,
      },
    };
  }
  if (kind === "text") {
    const text = await file.text();
    if (!text.trim()) return { error: copy(`「${file.name}」是空的`, `“${file.name}” is empty.`) };
    if (text.length > MAX_TEXT_CHARACTERS) return { error: copy(`「${file.name}」超过 10 万字，发不了`, `“${file.name}” exceeds the 100,000-character limit.`) };
    return { key, upload: { kind: "text", name: file.name, mimeType: file.type || "text/plain", text } };
  }
  if (kind === "document") return { error: copy(`「${file.name}」这类文件（PDF、Word、Excel 等）暂不支持，可以复制里面的文字发过来`, `Documents such as “${file.name}” (PDF, Word, Excel) are not supported yet. Paste the text instead.`) };
  return { error: copy(`「${file.name}」暂不支持，能发图片和文本文件`, `“${file.name}” is not supported. You can send pictures and text files.`) };
}

async function scaledImage(file: File): Promise<{ dataUrl: string; mimeType: string; width: number; height: number } | null> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    return null;
  }
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  // PNG keeps screenshots and transparency sharp; photos go as JPEG.
  const mimeType = file.type === "image/png" && width * height <= 1_000_000 ? "image/png" : "image/jpeg";
  return { dataUrl: canvas.toDataURL(mimeType, 0.88), mimeType, width, height };
}
