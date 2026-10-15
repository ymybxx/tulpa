import { describe, expect, it } from "vitest";
import { cubismMaskBufferCount } from "./cubism-mask-buffers";

const masks = (count: number) => Array.from({ length: count }, (_, index) => Int32Array.of(index));

describe("Cubism mask buffer capacity", () => {
  it("retains one buffer for models within the single-texture limit", () => {
    expect(cubismMaskBufferCount([])).toBe(1);
    expect(cubismMaskBufferCount(masks(36))).toBe(1);
  });

  it("fits complex models and respects the lower capacity of each multi-texture buffer", () => {
    expect(cubismMaskBufferCount(masks(37))).toBe(2);
    expect(cubismMaskBufferCount(masks(42))).toBe(2);
    expect(cubismMaskBufferCount(masks(64))).toBe(2);
    expect(cubismMaskBufferCount(masks(65))).toBe(3);
  });

  it("shares repeated mask combinations without depending on ID order", () => {
    const shared = Array.from({ length: 80 }, (_, index) =>
      Int32Array.from(index % 2 ? [5, 9] : [9, 5]));
    expect(cubismMaskBufferCount([...masks(35), ...shared, Int32Array.of()])).toBe(1);
    expect(cubismMaskBufferCount([...masks(36), ...shared])).toBe(2);
  });

  it("sizes the independent drawable and offscreen managers without adding their counts", () => {
    expect(cubismMaskBufferCount(masks(20), masks(20))).toBe(1);
    expect(cubismMaskBufferCount(masks(5), masks(65))).toBe(3);
  });
});
