import type { ChatAttachmentUpload } from "./chat-attachments";
import type { SenseAction, SensePanel, SensesSnapshot } from "./pet-senses";
import type { QuickAction, QuickPanelView } from "./pet-quick";

export interface Rect { x: number; y: number; width: number; height: number }
/** The character on screen; `middle` is how wide the body is halfway down. */
export interface PetBody extends Rect { middle?: { y: number; left: number; right: number } }
export interface PetChatGeometry { bounds: PetBody | null; hovered: boolean }
export interface PetChatLayout {
  window: Rect;
  entry: Rect;
  /** Opens the main window, right under the chat entry. */
  home: Rect;
  composer: Rect;
  bubble: Rect;
  side: "left" | "right";
  /** Sense capsule above the pet's head, shown with the chat entry. */
  capsule: Rect;
  /** Persistent "still preparing" chip, just above the capsule. */
  status: Rect;
  /** Space reserved for the sense panel; its card grows from the capsule side. */
  panel: Rect;
  panelPlacement: "above" | "below";
  /**
   * While the sense panel is open and nowhere leaves room for the bubble and
   * the two pills beside the pet, they step aside instead of being covered.
   */
  chatAside: boolean;
  /** How far the pills and the capsule shrink beside a small TA (1 = full size). */
  hud: number;
}
export interface PetChatState {
  layout: PetChatLayout;
  senses: SensesSnapshot | null;
  sensePanel: SensePanel;
  senseNote: string;
  /** Speech models still preparing or failed; null once everything is ready. */
  readiness: { state: "preparing" | "failed"; text: string; detail: string } | null;
  entryVisible: boolean;
  editing: boolean;
  bubbleText: string;
  /** Why the reply did not reach the user, when it did not. */
  bubbleNote: string;
  bubbleStatus: string;
  bubbleVisible: boolean;
  /** The right-click quick panel, in this window's coordinates; null while closed. */
  quick: QuickPanelView | null;
}
export interface PetChatApi {
  getUiLocale(): Promise<import("./contracts").DesktopUiLocale>;
  onUiLocale(callback: (locale: import("./contracts").DesktopUiLocale) => void): () => void;
  ready(): void;
  open(): void;
  close(): void;
  /** The input method is picking characters: a click on its candidates is not a click away. */
  composing(active: boolean): void;
  openHistory(): void;
  send(requestId: string, content: string, attachmentIds?: string[]): Promise<{ ok: boolean; error?: string; retryable?: boolean }>;
  /** Keeps a picture or text file until the message carrying it is sent. */
  attach(upload: ChatAttachmentUpload): Promise<{ ok: true; id: string } | { ok: false; error: string }>;
  /** 「新对话」: TA stops carrying on the conversation so far; what it remembers stays. */
  newConversation(): Promise<{ ok: boolean; error?: string }>;
  /** Whether the chat model reads pictures; null while TA is still starting. */
  imageInput(): Promise<{ chat: "supported" | "unsupported" | "unknown" } | null>;
  onState(callback: (state: PetChatState) => void): () => void;
  /**
   * Where the pointer is over the sense capsule, in window coordinates; null
   * once it leaves. CSS :hover does not follow the pointer while tulpa is in
   * the background, so the page learns it from the main process.
   */
  onPointer(callback: (point: { x: number; y: number } | null) => void): () => void;
  senseAction(action: SenseAction): void;
  quickAction(action: QuickAction): void;
  /** The main window button beside the pet: back to wherever the user last was. */
  openMain(): void;
}

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(v, hi));
export function contains(r: Rect, x: number, y: number, margin = 0): boolean {
  return x >= r.x - margin && y >= r.y - margin && x < r.x + r.width + margin && y < r.y + r.height + margin;
}
export function validRect(value: unknown): value is Rect {
  if (!value || typeof value !== "object") return false;
  const r = value as Rect;
  return [r.x, r.y, r.width, r.height].every(Number.isFinite) && r.width > 0 && r.height > 0;
}

/** Screen coordinates are DIP, including negative origins on secondary monitors.
 * The pointer usually lands on the middle of the body, so the chat entry
 * appears right beside it there, and the composer opens on that same spot.
 * `extra` is another screen area the window must cover, such as the quick panel. */
export function layoutPetChat(pet: PetBody, work: Rect, extra: Rect | null = null, panelOpen = false): PetChatLayout {
  const pad = 16;
  const hud = hudScale(pet);
  const ENTRY_WIDTH = BASE_ENTRY_WIDTH * hud, ENTRY_HEIGHT = BASE_ENTRY_HEIGHT * hud;
  const ENTRY_GAP = BASE_ENTRY_GAP * hud, BODY_GAP = BASE_BODY_GAP * hud;
  const safe = { x: work.x + pad, y: work.y + pad, width: Math.max(48, work.width - pad * 2), height: Math.max(48, work.height - pad * 2) };
  const width = Math.min(316, safe.width);
  const height = Math.min(122, safe.height);
  const bubbleHeight = Math.min(144, Math.max(48, safe.height - height - 12));
  const middle = pet.middle ?? { y: pet.y + pet.height / 2, left: pet.x, right: pet.x + pet.width };
  const reach = {
    y: clamp(middle.y, pet.y, pet.y + pet.height),
    left: clamp(middle.left, pet.x, pet.x + pet.width),
    right: clamp(middle.right, pet.x, pet.x + pet.width),
  };
  const leftSpace = reach.left - BODY_GAP - safe.x;
  const rightSpace = safe.x + safe.width - reach.right - BODY_GAP;
  const side = leftSpace >= width || leftSpace >= rightSpace ? "left" : "right";
  const x = clamp(side === "left" ? reach.left - BODY_GAP - width : reach.right + BODY_GAP, safe.x, safe.x + safe.width - width);
  let y = clamp(reach.y - height / 2, safe.y, safe.y + safe.height - height);
  // On small displays / oversized pets, use free space above or below first.
  if (Math.max(leftSpace, rightSpace) < width) {
    const above = pet.y - safe.y - 12;
    const below = safe.y + safe.height - pet.y - pet.height - 12;
    if (below >= height) y = pet.y + pet.height + 12;
    else if (above >= height) y = pet.y - height - 12;
  }
  const composer = { x, y, width, height };
  // The entry sits on the composer's inner edge, so the input opens where the pointer already is.
  const entry = {
    x: side === "left" ? x + width - ENTRY_WIDTH : x,
    y: clamp(reach.y - ENTRY_HEIGHT / 2, y, y + height - ENTRY_HEIGHT),
    width: ENTRY_WIDTH, height: ENTRY_HEIGHT,
  };
  // The main window is the other thing people reach for, so its button sits
  // right under the chat entry; above it when the screen ends.
  const below = entry.y + ENTRY_HEIGHT + ENTRY_GAP;
  const home = {
    x: entry.x,
    y: below + ENTRY_HEIGHT <= safe.y + safe.height ? below : entry.y - ENTRY_GAP - ENTRY_HEIGHT,
    width: ENTRY_WIDTH, height: ENTRY_HEIGHT,
  };
  // What TA says sits above the input, toward the face; below it when the screen ends.
  const bubbleY = y - 12 - bubbleHeight >= safe.y ? y - 12 - bubbleHeight : Math.min(y + height + 12, safe.y + safe.height - bubbleHeight);
  let bubble = { x, y: bubbleY, width, height: bubbleHeight };
  const { capsule, panel, panelPlacement } = layoutSenses(pet, safe, hud);
  // A panel that opens downward can land on the bubble and the pills. Move
  // them under the panel, else to the other side of the pet, else aside.
  let column = [composer, bubble, entry, home];
  let chatAside = false;
  if (panelOpen && column.some((r) => overlaps(r, panel))) {
    const top = Math.min(...column.map((r) => r.y));
    const bottom = Math.max(...column.map((r) => r.y + r.height));
    const down = panel.y + panel.height + 12 - top;
    const otherX = side === "left" ? reach.right + BODY_GAP : reach.left - BODY_GAP - width;
    const across = column.map((r) => ({ ...r, x: r.x - x + otherX }));
    if (bottom + down <= safe.y + safe.height) {
      column = column.map((r) => ({ ...r, y: r.y + down }));
    } else if (otherX >= safe.x && otherX + width <= safe.x + safe.width && !across.some((r) => overlaps(r, panel))) {
      column = across;
    } else {
      chatAside = true;
    }
    [composer.x, composer.y] = [column[0]!.x, column[0]!.y];
    bubble = column[1]!;
    [entry.x, entry.y] = [column[2]!.x, column[2]!.y];
    [home.x, home.y] = [column[3]!.x, column[3]!.y];
  }
  const statusY = capsule.y - STATUS_CHIP.height - 6 >= safe.y
    ? capsule.y - STATUS_CHIP.height - 6
    : capsule.y + capsule.height + 6;
  const status = {
    x: clamp(pet.x + pet.width / 2 - STATUS_CHIP.width / 2, safe.x, safe.x + safe.width - STATUS_CHIP.width),
    y: statusY, ...STATUS_CHIP,
  };
  const parts = [composer, bubble, entry, home, capsule, panel, status, ...(extra ? [extra] : [])];
  const left = Math.min(...parts.map((r) => r.x)) - 12;
  const right = Math.max(...parts.map((r) => r.x + r.width)) + 12;
  const top = Math.min(...parts.map((r) => r.y)) - 12;
  const bottom = Math.max(...parts.map((r) => r.y + r.height)) + 12;
  const window = { x: Math.floor(left), y: Math.floor(top), width: Math.ceil(right - Math.floor(left)), height: Math.ceil(bottom - Math.floor(top)) };
  const local = (r: Rect): Rect => ({ ...r, x: r.x - window.x, y: r.y - window.y });
  return {
    window, composer: local(composer), entry: local(entry), home: local(home), bubble: local(bubble), side,
    capsule: local(capsule), panel: local(panel), panelPlacement, status: local(status), chatAside, hud,
  };
}

function overlaps(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
}

export const SENSE_CAPSULE = { width: 138, height: 36 };
/** The two labelled pills beside the pet, 聊两句 and 主窗口, at full size. */
const BASE_ENTRY_WIDTH = 92;
const BASE_ENTRY_HEIGHT = 36;
const BASE_ENTRY_GAP = 8;
const BASE_BODY_GAP = 10;
/** TA at least this tall gets full-size pills and capsule. */
const HUD_FULL_HEIGHT = 360;
/** Smaller than this and the labels would be hard to read. */
const HUD_MIN_SCALE = 0.75;

/** The pills and capsule shrink with a small TA, so they never dwarf it, down to a readable size. */
export function hudScale(pet: Rect): number {
  return clamp(pet.height / HUD_FULL_HEIGHT, HUD_MIN_SCALE, 1);
}
export const STATUS_CHIP = { width: 240, height: 28 };
export const SENSE_PANEL = { width: 304, height: 364 };

/** Centered over the head; the panel opens upward unless the head is near the top edge. */
function layoutSenses(pet: Rect, safe: Rect, hud = 1): Pick<PetChatLayout, "capsule" | "panel" | "panelPlacement"> {
  const center = pet.x + pet.width / 2;
  const size = { width: SENSE_CAPSULE.width * hud, height: SENSE_CAPSULE.height * hud };
  const gap = 6 * hud;
  const capsuleY = pet.y - size.height - gap >= safe.y ? pet.y - size.height - gap : clamp(pet.y + gap, safe.y, safe.y + safe.height - size.height);
  const capsule = {
    x: clamp(center - size.width / 2, safe.x, safe.x + safe.width - size.width),
    y: capsuleY, ...size,
  };
  const width = Math.min(SENSE_PANEL.width, safe.width);
  const height = Math.min(SENSE_PANEL.height, safe.height);
  const above = capsule.y - 8 - height >= safe.y;
  const panel = {
    x: clamp(center - width / 2, safe.x, safe.x + safe.width - width),
    y: above ? capsule.y - 8 - height : clamp(capsule.y + capsule.height + 8, safe.y, safe.y + safe.height - height),
    width, height,
  };
  return { capsule, panel, panelPlacement: above ? "above" : "below" };
}
