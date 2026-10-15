import { describe, expect, it } from "vitest";

import {
  pointInsideInteractionControl,
  unlockControlPosition,
} from "../src/shared/interaction-lock";

describe("pet interaction lock", () => {
  it("only captures clicks inside the visible unlock control", () => {
    const bounds = { left: 420, top: 16, right: 464, bottom: 60 };
    expect(pointInsideInteractionControl({ x: 442, y: 38 }, bounds)).toBe(true);
    expect(pointInsideInteractionControl({ x: 419, y: 38 }, bounds)).toBe(false);
    expect(pointInsideInteractionControl({ x: 442, y: 61 }, bounds)).toBe(false);
  });

  it("places the control beside the model instead of at the window corner", () => {
    expect(unlockControlPosition(
      { x: 80, y: 110, width: 230, height: 500 },
      { width: 34, height: 34 },
      { width: 480, height: 720 },
    )).toEqual({ left: 318, top: 118 });
  });

  it("goes to the model's left when the right side is out of the window", () => {
    expect(unlockControlPosition(
      { x: 300, y: -20, width: 170, height: 700 },
      { width: 34, height: 34 },
      { width: 480, height: 720 },
    )).toEqual({ left: 258, top: 8 });
  });

  it("sits above a small model that leaves no room beside it, not over its face", () => {
    // A shrunken pet: the labelled 恢复点击 pill is wider than the space on either side.
    expect(unlockControlPosition(
      { x: 100, y: 200, width: 120, height: 260 },
      { width: 110, height: 34 },
      { width: 321, height: 482 },
    )).toEqual({ left: 105, top: 158 });
  });

  it("covers the model only when there is no room anywhere else", () => {
    expect(unlockControlPosition(
      { x: 10, y: 10, width: 300, height: 460 },
      { width: 110, height: 34 },
      { width: 321, height: 482 },
    )).toEqual({ left: 192, top: 18 });
  });
});
