import type { Rect } from "./pet-chat";

/**
 * The quick panel that right-clicking the pet opens. Its switch grid is the
 * user's own: they choose which tiles show and in what order.
 */
export type QuickTileKey =
  | "speaker" | "microphone" | "screen" | "audio" | "inner" | "companion" | "bubbles" | "passthrough"
  | "chat" | "settings";

export const QUICK_TILES: Record<QuickTileKey, { title: string; kind: "switch" | "action" }> = {
  speaker: { title: "声音播放", kind: "switch" },
  microphone: { title: "麦克风收音", kind: "switch" },
  screen: { title: "屏幕感知", kind: "switch" },
  audio: { title: "电脑音频感知", kind: "switch" },
  inner: { title: "内心", kind: "switch" },
  companion: { title: "主动陪伴", kind: "switch" },
  bubbles: { title: "说话气泡", kind: "switch" },
  passthrough: { title: "鼠标穿透", kind: "switch" },
  chat: { title: "打开聊天", kind: "action" },
  settings: { title: "设置", kind: "action" },
};
export const QUICK_TILE_KEYS = Object.keys(QUICK_TILES) as QuickTileKey[];
export const DEFAULT_QUICK_TILES: QuickTileKey[] = ["speaker", "microphone", "screen", "companion", "bubbles", "passthrough"];

const isTile = (value: unknown): value is QuickTileKey => typeof value === "string" && value in QUICK_TILES;

/** A saved tile list, or the defaults when there is none. An empty list is a real choice. */
export function parseQuickTiles(value: unknown): QuickTileKey[] {
  if (!Array.isArray(value)) return [...DEFAULT_QUICK_TILES];
  return [...new Set(value.filter(isTile))];
}

export interface QuickLook { id: string; name: string }

/** What the panel shows; sense switches come from the shared senses snapshot. */
export interface QuickPanelView {
  rect: Rect;
  /** Which edge sits at the click; the panel grows away from it. */
  edge: "top" | "bottom";
  editing: boolean;
  tiles: QuickTileKey[];
  speaker: boolean;
  bubbles: boolean;
  character: string | null;
  petState: string | null;
  looks: QuickLook[];
  lookId: string | null;
}

export type QuickAction =
  | { type: "tile"; key: QuickTileKey }
  | { type: "tiles"; tiles: QuickTileKey[] }
  | { type: "edit"; editing: boolean }
  | { type: "look"; id: string }
  | { type: "compose" }
  | { type: "hide" }
  | { type: "main" }
  | { type: "quit" }
  | { type: "close" };

export function parseQuickAction(value: unknown): QuickAction | null {
  if (typeof value !== "object" || value === null) return null;
  const action = value as Record<string, unknown>;
  switch (action.type) {
    case "tile": return isTile(action.key) ? { type: "tile", key: action.key } : null;
    case "tiles": return Array.isArray(action.tiles) && action.tiles.length <= QUICK_TILE_KEYS.length && action.tiles.every(isTile)
      ? { type: "tiles", tiles: parseQuickTiles(action.tiles) } : null;
    case "edit": return typeof action.editing === "boolean" ? { type: "edit", editing: action.editing } : null;
    case "look": return typeof action.id === "string" && action.id.length > 0 && action.id.length <= 200 ? { type: "look", id: action.id } : null;
    case "compose": case "hide": case "main": case "quit": case "close": return { type: action.type };
    default: return null;
  }
}

export const QUICK_PANEL_WIDTH = 272;
/** Room for the tallest panel: editing with every tile placed on one side. */
export const QUICK_PANEL_HEIGHT = 480;

/** Next to the click, turned toward the free side when the screen edge is near. */
export function placeQuickPanel(point: { x: number; y: number }, work: Rect): { rect: Rect; edge: "top" | "bottom" } {
  const pad = 8;
  const width = Math.min(QUICK_PANEL_WIDTH, work.width - pad * 2);
  const height = Math.min(QUICK_PANEL_HEIGHT, work.height - pad * 2);
  const right = work.x + work.width - pad;
  const bottom = work.y + work.height - pad;
  const x = point.x + 4 + width <= right ? point.x + 4 : Math.max(work.x + pad, point.x - 4 - width);
  const below = bottom - point.y;
  const above = point.y - work.y - pad;
  const edge = below >= height || below >= above ? "top" : "bottom";
  const y = edge === "top" ? Math.min(point.y + 4, bottom - height) : Math.max(work.y + pad, point.y - 4 - height);
  return { rect: { x, y, width, height }, edge };
}
