import { describe, expect, it } from "vitest";

import {
  alignToSlot,
  preserveVisiblePixels,
  type RgbaImage,
} from "./image-processing";

function image(width: number, height: number, rgba: number[]): RgbaImage {
  return { width, height, data: new Uint8Array(rgba) };
}

describe("visible pixel protection", () => {
  it("keeps every visible original pixel and only accepts generated hidden pixels", () => {
    const original = image(2, 1, [255, 0, 0, 255, 0, 0, 0, 0]);
    const completed = image(2, 1, [0, 255, 0, 255, 0, 0, 255, 255]);

    expect([...preserveVisiblePixels(original, completed).data]).toEqual([
      255, 0, 0, 255,
      0, 0, 255, 255,
    ]);
  });
});

describe("constrained slot alignment", () => {
  it("returns scale and translation for a compatible silhouette", () => {
    const result = alignToSlot(
      { x: 10, y: 20, width: 100, height: 200, anchor: { x: 60, y: 100 } },
      { x: 0, y: 0, width: 110, height: 210, anchor: { x: 55, y: 105 } },
      { minScale: 0.8, maxScale: 1.2, aspectTolerance: 0.15 },
    );

    expect(result.scale).toBeCloseTo(1.05, 2);
    expect(result.translateX).toBeCloseTo(-8, 0);
    expect(result.translateY).toBeCloseTo(0, 0);
  });

  it("rejects a body shape outside the master tolerance", () => {
    expect(() => alignToSlot(
      { x: 0, y: 0, width: 200, height: 100, anchor: { x: 100, y: 50 } },
      { x: 0, y: 0, width: 100, height: 200, anchor: { x: 50, y: 100 } },
      { minScale: 0.8, maxScale: 1.2, aspectTolerance: 0.15 },
    )).toThrow(/轮廓/);
  });
});
