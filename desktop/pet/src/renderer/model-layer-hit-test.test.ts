import { describe, expect, it } from "vitest";
import { meshContainsPoint } from "./model-layer-hit-test";
describe("layer point selection", () => {
  it("picks actual triangles in either winding and rejects empty bounding-box corners", () => {
    const vertices = new Float32Array([0, 0, 1, 0, 0, 1]);
    for (const indices of [[0, 1, 2], [2, 1, 0]]) {
      expect(meshContainsPoint(vertices, indices, 0.2, 0.2)).toBe(true);
      expect(meshContainsPoint(vertices, indices, 0.5, 0.5)).toBe(true);
      expect(meshContainsPoint(vertices, indices, 0.9, 0.9)).toBe(false);
      expect(meshContainsPoint(vertices, indices, -0.1, 0.2)).toBe(false);
    }
  });
  it("ignores collapsed or malformed triangles", () => {
    expect(meshContainsPoint([0, 0, 0, 0, 0, 0], [0, 1, 2], 0, 0)).toBe(false);
    expect(meshContainsPoint([0, 0, 1, 0, 0, 1], [0, 1, 99], 0, 0)).toBe(false);
  });
});
