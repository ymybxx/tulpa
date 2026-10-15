import { localizeStatic } from "../i18n/dom";
import { t, getLocale, subscribeLocale } from "../i18n";
import "./pet-chat.css";
import type { PetChatApi, PetChatState, Rect } from "../shared/pet-chat";
import { MAX_ATTACHMENTS, contentWithAttachments, readAttachment, type PendingAttachment } from "../shared/chat-attachments";
import { mountSenses } from "./pet-senses-view";
import { mountQuickPanel } from "./pet-quick-view";

declare global { interface Window { petChat: PetChatApi } }
const get = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const entry = get<HTMLButtonElement>("entry");
const home = get<HTMLButtonElement>("home");
const composer = get<HTMLFormElement>("composer");
const input = get<HTMLTextAreaElement>("input");
const send = get<HTMLButtonElement>("send");
const hint = get("hint");
const bubble = get("bubble");
const speech = get("speech");
const speechNote = get("speech-note");
const scroll = get("speech-scroll");
let editing = false;
let sending = false;
let composing = false;
let lastText = "";
let request: { id: string; content: string } | null = null;
let attachments: PendingAttachment[] = [];
let imagesAllowed = true;
const files = get<HTMLInputElement>("files");
const clearAttachments = get<HTMLButtonElement>("clear-attachments");
const IDLE_HINT = "Enter 发送 · ⇧ Enter 换行";
let latestState: PetChatState | null = null;
const renderSenses = mountSenses(window.petChat);
const renderQuick = mountQuickPanel(window.petChat);

const place = (element: HTMLElement, rect: Rect): void => {
  element.style.left = `${rect.x}px`; element.style.top = `${rect.y}px`;
  element.style.width = `${rect.width}px`; element.style.height = `${rect.height}px`;
};
/** Laid out at full size inside, then shown at `zoom` of it; CSS zoom scales the box too. */
const placeZoomed = (element: HTMLElement, rect: Rect, zoom: number): void => {
  element.style.zoom = String(zoom);
  place(element, { x: rect.x / zoom, y: rect.y / zoom, width: rect.width / zoom, height: rect.height / zoom });
};
function display(state: PetChatState): void {
  latestState = state;
  place(composer, state.layout.composer); place(bubble, state.layout.bubble);
  // The pills keep their full-size look and shrink as a whole beside a small TA.
  placeZoomed(entry, state.layout.entry, state.layout.hud); placeZoomed(home, state.layout.home, state.layout.hud);
  entry.hidden = !state.entryVisible || state.editing || state.layout.chatAside;
  home.hidden = entry.hidden;
  composer.hidden = !state.editing;
  bubble.hidden = !state.bubbleVisible || state.layout.chatAside;
  const label = state.bubbleStatus === "paused" ? t("暂停") : ["interrupted", "failed"].includes(state.bubbleStatus) ? t("已停下") : ["completed", "text"].includes(state.bubbleStatus) ? "" : t("正在说");
  get("bubble-label").textContent = label;
  // The heading only shows while there is something to say about the speech.
  bubble.dataset.voice = label === t("正在说") ? "speaking" : label ? "stopped" : "quiet";
  if (lastText !== state.bubbleText) {
    lastText = state.bubbleText;
    // Render as plain text. Model responses are never interpreted as markup.
    speech.textContent = lastText;
  }
  speechNote.hidden = !state.bubbleNote;
  speechNote.textContent = state.bubbleNote;
  // Reserve layout space once, but let a short utterance use a compact card.
  const chrome = label ? 47 : 25;
  const noteHeight = state.bubbleNote ? speechNote.offsetHeight + (state.bubbleText ? 4 : 0) : 0;
  const bubbleHeight = Math.min(state.layout.bubble.height, chrome + Math.min(72, speech.scrollHeight) + noteHeight);
  bubble.style.height = `${bubbleHeight}px`;
  // Above the composer the card keeps its lower edge by the input and grows upward.
  if (state.layout.bubble.y < state.layout.composer.y) bubble.style.top = `${state.layout.bubble.y + state.layout.bubble.height - bubbleHeight}px`;
  scroll.classList.toggle("overflowing", scroll.scrollHeight > scroll.clientHeight + 2);
  scroll.scrollTop = scroll.scrollHeight;
  if (state.editing && !editing) requestAnimationFrame(() => { input.focus(); input.setSelectionRange(input.value.length, input.value.length); });
  editing = state.editing;
  renderSenses(state);
  renderQuick(state);
}
window.petChat.onState(display);
entry.addEventListener("click", () => window.petChat.open());
home.addEventListener("click", () => window.petChat.openMain());
get("close").addEventListener("click", () => window.petChat.close());
get("history").addEventListener("click", () => window.petChat.openHistory());
const newChat = get<HTMLButtonElement>("new-chat");
let newChatNote: ReturnType<typeof setTimeout> | null = null;
newChat.addEventListener("click", () => {
  newChat.disabled = true;
  void window.petChat.newConversation().catch(() => ({ ok: false, error: "没能开始新对话，请再试一次" })).then((result) => {
    newChat.disabled = false;
    if (newChatNote) clearTimeout(newChatNote);
    hint.textContent = result.ok ? t("新对话开始了，TA 记得的事都还在") : t(result.error ?? "没能开始新对话，请再试一次");
    hint.title = hint.textContent;
    hint.classList.toggle("error", !result.ok);
    newChatNote = setTimeout(() => { newChatNote = null; showAttachments(); }, 4_000);
    input.focus();
  });
});
input.addEventListener("compositionstart", () => { composing = true; window.petChat.composing(true); });
input.addEventListener("compositionend", () => { composing = false; window.petChat.composing(false); });
const ready = (): boolean => Boolean(input.value.trim() || attachments.length);
const showAttachments = (): void => {
  hint.classList.remove("error");
  hint.textContent = attachments.length ? t("附件：{{names}}", { names: attachments.map((item) => item.upload.name).join(getLocale() === "en" ? ", " : "、") }) : t(IDLE_HINT);
  hint.title = hint.textContent;
  clearAttachments.hidden = !attachments.length;
  send.disabled = sending || !ready();
};
const addFiles = async (chosen: Iterable<File>): Promise<void> => {
  for (const file of chosen) {
    if (attachments.length >= MAX_ATTACHMENTS) { hint.textContent = t("一次最多发 {{count}} 个附件", { count: MAX_ATTACHMENTS }); hint.classList.add("error"); return; }
    const read = await readAttachment(file, { imagesAllowed, locale: getLocale() });
    if ("error" in read) { hint.textContent = read.error; hint.title = read.error; hint.classList.add("error"); return; }
    attachments = [...attachments, read];
    request = null;
    showAttachments();
  }
};
void window.petChat.imageInput().then((value) => { imagesAllowed = value?.chat !== "unsupported"; }).catch(() => undefined);
// A file let go anywhere else would replace the page with the file itself.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
get("attach").addEventListener("click", () => files.click());
files.addEventListener("change", () => { if (files.files) void addFiles(files.files); files.value = ""; });
clearAttachments.addEventListener("click", () => { attachments = []; request = null; showAttachments(); input.focus(); });
composer.addEventListener("dragover", (event) => { if (event.dataTransfer?.types.includes("Files")) { event.preventDefault(); composer.classList.add("dropping"); } });
composer.addEventListener("dragleave", () => composer.classList.remove("dropping"));
composer.addEventListener("drop", (event) => {
  event.preventDefault(); composer.classList.remove("dropping");
  if (event.dataTransfer?.files.length) void addFiles(event.dataTransfer.files);
});
input.addEventListener("paste", (event) => {
  const pasted = [...(event.clipboardData?.files ?? [])];
  if (pasted.length) { event.preventDefault(); void addFiles(pasted); }
});
input.addEventListener("input", () => { send.disabled = sending || !ready(); if (!attachments.length) { hint.textContent = t(IDLE_HINT); hint.classList.remove("error"); } });
input.addEventListener("keydown", event => {
  if (event.isComposing || composing || event.keyCode === 229) return;
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); composer.requestSubmit(); }
  if (event.key === "Escape") { event.preventDefault(); window.petChat.close(); }
});
composer.addEventListener("submit", event => {
  event.preventDefault();
  if (sending || composing || !ready()) return;
  const draft = input.value;
  const attached = attachments;
  const content = contentWithAttachments(draft, attached.map((item) => item.upload));
  if (!request || request.content !== content) request = { id: crypto.randomUUID(), content };
  const current = request;
  sending = true; send.disabled = true; hint.textContent = attached.length ? t("正在传附件…") : t("发送中…"); hint.classList.remove("error");
  void (async () => {
    const ids: string[] = [];
    for (const item of attached) {
      const uploaded = await window.petChat.attach(item.upload);
      if (!uploaded.ok) throw new Error(uploaded.error);
      ids.push(uploaded.id);
    }
    hint.textContent = t("发送中…");
    const result = await window.petChat.send(current.id, content, ids);
    if (!result.ok && result.retryable) request = null;
    if (!result.ok) throw new Error(result.error ?? t("未能发送，草稿已保留"));
    // The user may already be drafting another message during acceptance.
    if (input.value === draft) input.value = "";
    if (attachments === attached) attachments = [];
    request = null;
    clearAttachments.hidden = !attachments.length;
    hint.textContent = t("已发送");
  })().catch((error: unknown) => {
    request = null;
    hint.textContent = error instanceof Error ? error.message : t("暂时无法发送，草稿已保留");
    hint.classList.add("error");
  }).finally(() => { sending = false; send.disabled = !ready(); });
});
subscribeLocale(() => {
  if (latestState) display(latestState);
  if (!sending && !hint.classList.contains("error")) showAttachments();
});
window.petChat.ready();

localizeStatic(document);
