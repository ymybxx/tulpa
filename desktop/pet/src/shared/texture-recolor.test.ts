import { describe, expect, it } from "vitest";

import { parseHexColor, recolorPixels } from "./texture-recolor";

const pixel = (data: Uint8Array, index: number) => [...data.subarray(index * 4, index * 4 + 4)];

describe("texture recolor", () => {
  it("puts the piece's typical shade on the target color and keeps darker and lighter shades in order", () => {
    // Line art, three body shades (the middle one is the median), a highlight, one transparent pixel.
    const data = Uint8Array.from([
      20, 20, 40, 255,
      70, 80, 120, 255,
      80, 90, 130, 255,
      90, 100, 140, 255,
      200, 210, 240, 255,
      9, 9, 9, 0,
    ]);
    recolorPixels(data, null, parseHexColor("#E8A0B8")!);
    const median = pixel(data, 2);
    for (const [channel, value] of [0xe8, 0xa0, 0xb8].entries()) expect(Math.abs(median[channel]! - value)).toBeLessThanOrEqual(3);
    const brightness = (index: number) => pixel(data, index).slice(0, 3).reduce((sum, value) => sum + value, 0);
    expect(brightness(0)).toBeLessThan(brightness(1));
    expect(brightness(1)).toBeLessThan(brightness(2));
    expect(brightness(3)).toBeLessThan(brightness(4));
    // Pink, not grey: red leads blue leads green.
    const lines = pixel(data, 0);
    expect(lines[0]).toBeGreaterThan(lines[1]!);
    expect(pixel(data, 5)).toEqual([9, 9, 9, 0]);
    expect(pixel(data, 4)[3]).toBe(255);
  });

  it("leaves pixels outside the mask alone", () => {
    const data = Uint8Array.from([100, 150, 200, 255, 100, 150, 200, 255]);
    const mask = Uint8Array.from([255, 255, 255, 255, 255, 255, 255, 0]);
    recolorPixels(data, mask, parseHexColor("#202020")!);
    expect(pixel(data, 1)).toEqual([100, 150, 200, 255]);
    expect(pixel(data, 0)).not.toEqual([100, 150, 200, 255]);
  });

  it("reads hex colors with or without #", () => {
    expect(parseHexColor("#F6D9CC")).toEqual({ r: 0xf6, g: 0xd9, b: 0xcc });
    expect(parseHexColor("f6d9cc")).toEqual({ r: 0xf6, g: 0xd9, b: 0xcc });
    expect(parseHexColor("pink")).toBeNull();
  });
});
