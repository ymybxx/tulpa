import { describe, expect, it } from "vitest";

import { characterFrame } from "./character-frame";

function image(width: number, height: number, opaque: (x: number, y: number) => boolean): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) if (opaque(x, y)) data[(y * width + x) * 4 + 3] = 255;
  }
  return data;
}

describe("characterFrame", () => {
  it("centers a symmetric body", () => {
    const frame = characterFrame(image(100, 100, (x, y) => y >= 10 && y < 90 && x >= 40 && x < 60), 100, 100, 0)!;
    expect(Math.abs(frame.left + frame.width / 2 - 50)).toBeLessThanOrEqual(1);
    expect(frame).toMatchObject({ top: 10, height: 80 });
  });

  it("keeps the body centered when a prop sticks out to one side", () => {
    // Body at x 60–80 for 80 rows; a staff reaching x 0 on 10 rows.
    const data = image(120, 100, (x, y) => (y >= 10 && y < 90 && x >= 60 && x < 80) || (y >= 50 && y < 60 && x < 60));
    const frame = characterFrame(data, 120, 100, 0)!;
    expect(Math.abs(frame.left + frame.width / 2 - 70)).toBeLessThanOrEqual(1);
    // Still wide enough to include the staff.
    expect(frame.left).toBeLessThanOrEqual(0);
    expect(frame.left + frame.width).toBeGreaterThanOrEqual(80);
  });

  it("returns null for an empty image", () => {
    expect(characterFrame(image(10, 10, () => false), 10, 10)).toBeNull();
  });
});
