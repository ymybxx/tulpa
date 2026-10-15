import { describe, expect, it, vi } from "vitest";

import { PetStage } from "./pet-stage";
import { placePet } from "./window-drag";

type Rect = { x: number; y: number; width: number; height: number };

function stageWindow(bounds: Rect) {
  let current = { ...bounds };
  const sent: unknown[] = [];
  return {
    sent,
    setBoundsCalls: [] as Rect[],
    isDestroyed: () => false,
    isVisible: () => true,
    getBounds: () => ({ ...current }),
    setBounds(next: Rect) { this.setBoundsCalls.push(next); current = { ...next }; },
    on: vi.fn(),
    removeListener: vi.fn(),
    webContents: { isDestroyed: () => false, send: (_channel: string, value: unknown) => { sent.push(value); } },
  };
}

const left = { x: 0, y: 25, width: 1512, height: 944 };
const right = { x: 1512, y: 0, width: 1920, height: 1080 };
const displayAt = (point: { x: number; y: number }) => point.x < 1512 ? left : right;

describe("TA's stage", () => {
  it("moves and scales the panel inside a still window, with the model's offset alongside", () => {
    const window = stageWindow(left);
    let offset = { x: 0, y: 0 };
    const stage = new PetStage(window as never, { x: 900, y: 200, width: 480, height: 720 }, displayAt, () => offset);
    const moved = vi.fn(), resized = vi.fn();
    stage.on("move", moved).on("resize", resized);

    offset = { x: 12, y: 0 };
    stage.setBounds({ x: 880, y: 170, width: 520, height: 780 });
    stage.setPosition(860, 170);

    expect(window.setBoundsCalls).toEqual([]);
    expect(window.sent).toEqual([
      { x: 880, y: 145, width: 520, height: 780, offset: { x: 12, y: 0 } },
      { x: 860, y: 145, width: 520, height: 780, offset: { x: 12, y: 0 } },
    ]);
    expect(moved).toHaveBeenCalledTimes(2);
    expect(resized).toHaveBeenCalledTimes(1);
    expect(stage.getBounds()).toEqual({ x: 860, y: 170, width: 520, height: 780 });
  });

  it("covers the display the panel is on, moving only when it crosses to another", () => {
    const window = stageWindow({ x: 0, y: 0, width: 10, height: 10 });
    const stage = new PetStage(window as never, { x: 900, y: 200, width: 480, height: 720 }, displayAt, () => ({ x: 0, y: 0 }));
    expect(window.setBoundsCalls).toEqual([left]);

    stage.setPosition(1400, 200);
    expect(window.setBoundsCalls).toEqual([left, right]);
    expect(window.sent.at(-1)).toMatchObject({ x: -112, y: 200 });
  });

  it("sends nothing when the panel does not change", () => {
    const window = stageWindow(left);
    const stage = new PetStage(window as never, { x: 900, y: 200, width: 480, height: 720 }, displayAt, () => ({ x: 0, y: 0 }));
    stage.setBounds({ x: 900.2, y: 199.8, width: 480, height: 720 });
    expect(window.sent).toEqual([]);
  });

  it.each([left, { x: 0, y: 25, width: 1024, height: 700 }])(
    "keeps the model reachable when its display is removed or shrinks to %j", (remaining) => {
      let currentDisplay = right;
      let offset = { x: 30, y: -20 };
      const window = stageWindow(right);
      const stage = new PetStage(window as never, { x: 2800, y: 300, width: 480, height: 720 },
        () => currentDisplay, () => offset);
      const model = { x: 140, y: 80, width: 200, height: 580 };
      currentDisplay = remaining;
      stage.refit((area) => {
        const bounds = stage.getBounds();
        const placement = placePet({ x: bounds.x + offset.x, y: bounds.y + offset.y }, bounds, model, area);
        offset = placement.offset;
        stage.setPosition(placement.window.x, placement.window.y);
      });
      const panel = stage.getBounds();
      const modelX = panel.x + model.x + offset.x;
      const modelY = panel.y + model.y + offset.y;
      expect(modelX).toBeLessThanOrEqual(remaining.x + remaining.width - 96);
      expect(modelY).toBeLessThanOrEqual(remaining.y + remaining.height - 116);
      expect(modelX + model.width).toBeGreaterThanOrEqual(remaining.x + 96);
      expect(modelY + model.height).toBeGreaterThanOrEqual(remaining.y + 116);
      expect(stage.getStageBounds()).toEqual(remaining);
      expect(window.sent.at(-1)).toEqual(stage.placement());
    },
  );
});
