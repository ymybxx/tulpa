import { describe, expect, it } from "vitest";

import {
  composeTexturePatchPixel,
  texturePatchDestination,
  texturePatchFramebufferRect,
} from "./texture-patch";

describe("texture patch geometry", () => {
  it("maps source bounds to the current LOD without expanding outside source edges", () => {
    expect(texturePatchDestination(
      { x: 2759, y: 11, width: 476, height: 1716 },
      { width: 4096, height: 8192 },
      { width: 1024, height: 2048 },
    )).toEqual({ left: 689, top: 2, right: 809, bottom: 432 });
  });

  it("keeps top-left PNG rows aligned with unflipped ImageBitmap texture storage", () => {
    expect(texturePatchFramebufferRect(
      { left: 689, top: 2, right: 809, bottom: 432 },
      2048,
    )).toEqual({ x: 689, y: 2, width: 120, height: 430 });
  });

  it("keeps the saved outfit regions on their CPU-composed LOD rows", () => {
    const source = { width: 4096, height: 8192 };
    const target = { width: 1024, height: 2048 };
    const regions = [
      { bounds: { x: 2759, y: 11, width: 476, height: 1716 }, y: 2, height: 430 },
      { bounds: { x: 3263, y: 1355, width: 404, height: 1564 }, y: 338, height: 392 },
      { bounds: { x: 3110, y: 4449, width: 361, height: 1742 }, y: 1112, height: 436 },
      { bounds: { x: 17, y: 6080, width: 1654, height: 501 }, y: 1520, height: 126 },
    ];

    for (const region of regions) {
      const destination = texturePatchDestination(region.bounds, source, target);
      expect(texturePatchFramebufferRect(destination, target.height)).toMatchObject({
        y: region.y,
        height: region.height,
      });
    }
  });
});

describe("composeTexturePatchPixel", () => {
  it("replaces RGB inside the mask while alpha may only decrease", () => {
    expect(composeTexturePatchPixel(
      [80, 40, 20, 100],
      [200, 160, 120, 180],
      255,
    )).toEqual([200, 160, 120, 100]);
  });

  it("keeps the current pixel outside the mask and permits transparent removal", () => {
    expect(composeTexturePatchPixel(
      [80, 40, 20, 100],
      [200, 160, 120, 0],
      0,
    )).toEqual([80, 40, 20, 100]);
    expect(composeTexturePatchPixel(
      [80, 40, 20, 100],
      [200, 160, 120, 0],
      255,
    )).toEqual([200, 160, 120, 0]);
  });
});
