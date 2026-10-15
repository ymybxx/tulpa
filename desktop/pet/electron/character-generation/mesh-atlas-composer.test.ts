import { describe, expect, it } from "vitest";

import type { MasterTemplateSlot } from "../../src/shared/character-generation";
import type { RgbaImage } from "./image-processing";
import { composeMeshAtlas } from "./mesh-atlas-composer";

const slot: MasterTemplateSlot = {
  id: "face",
  name: "Face",
  drawableId: "D_FACE",
  partId: "PART_FACE",
  group: "face",
  source: "visible",
  required: true,
  zIndex: 1,
  atlas: { texture: 0, x: 0, y: 0, width: 2, height: 2 },
  anchor: { x: 1, y: 1 },
  canvasBounds: { x: -1, y: -1, width: 2, height: 2 },
  mesh: {
    vertices: [-1, 1, 1, 1, 1, -1, -1, -1],
    uvs: [0, 1, 1, 1, 1, 0, 0, 0],
    indices: [0, 1, 2, 0, 2, 3],
  },
  silhouetteMask: "masks/face.png",
  completionPrompt: "complete face",
};

describe("composeMeshAtlas", () => {
  it("inverse-bakes the neutral canvas through Cubism UV triangles", () => {
    const source: RgbaImage = {
      width: 2,
      height: 2,
      data: new Uint8Array([
        255, 0, 0, 255, 0, 255, 0, 255,
        0, 0, 255, 255, 255, 255, 255, 255,
      ]),
    };
    const base: RgbaImage = { width: 2, height: 2, data: new Uint8Array(16) };
    const output = composeMeshAtlas(
      base,
      { width: 2, height: 2, originX: 1, originY: 1, pixelsPerUnit: 1 },
      [{ slot, image: source }],
    );

    expect(pixel(output, 0, 0)[3]).toBe(255);
    expect(pixel(output, 1, 1)[3]).toBe(255);
    expect([...output.data]).not.toEqual([...base.data]);
  });

  it("applies eye edits as color deltas while preserving original alpha and detail", () => {
    const reference = solid(2, 2, [100, 110, 120, 255]);
    const edited = solid(2, 2, [125, 90, 130, 10]);
    const base = solid(2, 2, [40, 80, 160, 77]);

    const output = composeMeshAtlas(
      base,
      { width: 2, height: 2, originX: 1, originY: 1, pixelsPerUnit: 1 },
      [{ slot, image: edited, reference }],
    );

    expect(pixel(output, 0, 0)).toEqual([65, 60, 170, 77]);
    expect(pixel(output, 1, 1)).toEqual([65, 60, 170, 77]);
  });
});

function solid(width: number, height: number, rgba: number[]): RgbaImage {
  return {
    width,
    height,
    data: new Uint8Array(Array(width * height).fill(rgba).flat()),
  };
}

function pixel(image: RgbaImage, x: number, y: number): number[] {
  const offset = (y * image.width + x) * 4;
  return [...image.data.slice(offset, offset + 4)];
}
