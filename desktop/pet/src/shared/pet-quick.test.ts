import { describe, expect, it } from "vitest";
import { DEFAULT_QUICK_TILES, parseQuickAction, parseQuickTiles, placeQuickPanel, QUICK_PANEL_HEIGHT, QUICK_PANEL_WIDTH } from "./pet-quick";

describe("quick panel tiles", () => {
  it("starts from the defaults, keeps the user's order and an empty choice, and drops unknown or repeated tiles", () => {
    expect(parseQuickTiles(undefined)).toEqual(DEFAULT_QUICK_TILES);
    expect(parseQuickTiles(["inner", "speaker", "inner", "fly", 3])).toEqual(["inner", "speaker"]);
    expect(parseQuickTiles([])).toEqual([]);
  });

  it("accepts only well-formed actions", () => {
    expect(parseQuickAction({ type: "tile", key: "bubbles" })).toEqual({ type: "tile", key: "bubbles" });
    expect(parseQuickAction({ type: "tile", key: "rm -rf" })).toBeNull();
    expect(parseQuickAction({ type: "tiles", tiles: ["chat", "chat", "settings"] })).toEqual({ type: "tiles", tiles: ["chat", "settings"] });
    expect(parseQuickAction({ type: "tiles", tiles: ["chat", "nope"] })).toBeNull();
    expect(parseQuickAction({ type: "look", id: "" })).toBeNull();
    expect(parseQuickAction({ type: "edit", editing: "yes" })).toBeNull();
    expect(parseQuickAction({ type: "quit" })).toEqual({ type: "quit" });
    expect(parseQuickAction(null)).toBeNull();
  });
});

describe("quick panel placement", () => {
  const work = { x: 0, y: 25, width: 1440, height: 875 };
  it("opens right and down from the click when there is room", () => {
    const { rect, edge } = placeQuickPanel({ x: 400, y: 100 }, work);
    expect(edge).toBe("top");
    expect(rect).toEqual({ x: 404, y: 104, width: QUICK_PANEL_WIDTH, height: QUICK_PANEL_HEIGHT });
  });
  it("turns left near the right edge and grows upward near the bottom", () => {
    const { rect, edge } = placeQuickPanel({ x: 1400, y: 860 }, work);
    expect(edge).toBe("bottom");
    expect(rect.x + rect.width).toBe(1396);
    expect(rect.y + rect.height).toBe(856);
  });
  it("stays on a small or secondary screen", () => {
    const small = { x: -800, y: -200, width: 600, height: 400 };
    const { rect } = placeQuickPanel({ x: -300, y: 0 }, small);
    expect(rect.x).toBeGreaterThanOrEqual(small.x);
    expect(rect.y).toBeGreaterThanOrEqual(small.y);
    expect(rect.x + rect.width).toBeLessThanOrEqual(small.x + small.width);
    expect(rect.y + rect.height).toBeLessThanOrEqual(small.y + small.height);
  });
});
