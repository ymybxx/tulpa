import { describe, expect, it } from "vitest";
import { PNG } from "pngjs";

import {
  composeAnnotatedModelReferencePng,
  composeModelLocatorPng,
} from "./model-locator";

describe("composeModelLocatorPng", () => {
  it("highlights only the exact projected mask while preserving full-model context", () => {
    const reference = new PNG({ width: 2, height: 1 });
    reference.data.set([100, 110, 120, 255, 200, 210, 220, 255]);
    const mask = new PNG({ width: 2, height: 1 });
    mask.data.set([255, 255, 255, 255, 0, 0, 0, 0]);

    const result = PNG.sync.read(composeModelLocatorPng(
      PNG.sync.write(reference),
      PNG.sync.write(mask),
      0,
      { x: 0, y: 0, width: 2, height: 1, canvasWidth: 2, canvasHeight: 1 },
    ));

    expect([...result.data.subarray(0, 4)]).not.toEqual([100, 110, 120, 255]);
    expect([...result.data.subarray(4, 8)]).toEqual([200, 210, 220, 255]);
  });

  it("rejects a stale locator mask with different dimensions", () => {
    const reference = new PNG({ width: 2, height: 1 });
    const mask = new PNG({ width: 1, height: 1 });
    expect(() => composeModelLocatorPng(
      PNG.sync.write(reference),
      PNG.sync.write(mask),
      0,
      { x: 0, y: 0, width: 2, height: 1, canvasWidth: 2, canvasHeight: 1 },
    ))
      .toThrow("定位蒙版尺寸");
  });

  it("draws matching A/B annotations for multiple exact model regions", () => {
    const reference = new PNG({ width: 120, height: 120 });
    for (let offset = 0; offset < reference.data.length; offset += 4) {
      reference.data.set([230, 230, 230, 255], offset);
    }
    const maskA = new PNG({ width: 20, height: 20 });
    const maskB = new PNG({ width: 20, height: 20 });
    maskA.data.fill(255);
    maskB.data.fill(255);

    const result = PNG.sync.read(composeAnnotatedModelReferencePng(
      PNG.sync.write(reference),
      [
        {
          key: "A",
          maskBytes: PNG.sync.write(maskA),
          regionIndex: 0,
          placement: { x: 15, y: 45, width: 20, height: 20, canvasWidth: 120, canvasHeight: 120 },
        },
        {
          key: "B",
          maskBytes: PNG.sync.write(maskB),
          regionIndex: 1,
          placement: { x: 85, y: 45, width: 20, height: 20, canvasWidth: 120, canvasHeight: 120 },
        },
      ],
    ));
    const pixel = (x: number, y: number) =>
      [...result.data.subarray((y * result.width + x) * 4, (y * result.width + x) * 4 + 4)];

    expect(pixel(20, 50)).not.toEqual([230, 230, 230, 255]);
    expect(pixel(90, 50)).not.toEqual([230, 230, 230, 255]);
    expect(pixel(60, 60)).toEqual([230, 230, 230, 255]);
    // Each region also receives a dark-backed letter badge outside its mask.
    expect([...result.data].filter((value, index) => index % 4 !== 3 && value < 30).length)
      .toBeGreaterThan(0);
  });
});
