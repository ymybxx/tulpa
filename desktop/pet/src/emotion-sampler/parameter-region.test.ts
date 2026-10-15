import { describe, expect, it } from "vitest";
import { fitRegion } from "./parameter-region";

describe("the enlarged part of a parameter screenshot", () => {
  it("frames a small change with room around it", () => {
    const region = fitRegion({ x: 600, y: 900, width: 60, height: 40 }, 1024, 1536);
    // At most a 5x enlargement, centered on the change.
    expect(region.width).toBeCloseTo(1024 / 5, 5);
    expect(region.x + region.width / 2).toBeCloseTo(630, 5);
    expect(region.y + region.height / 2).toBeCloseTo(920, 5);
  });

  it("keeps a wide change wide instead of cropping the whole body around it", () => {
    const region = fitRegion({ x: 250, y: 500, width: 580, height: 300 }, 1024, 1536);
    expect(region.width / region.height).toBeGreaterThan(1.5);
    expect(region.height).toBeLessThan(500);
  });

  it("stays inside the picture", () => {
    const region = fitRegion({ x: 0, y: 1300, width: 1024, height: 236 }, 1024, 1536);
    expect(region.x).toBeGreaterThanOrEqual(0);
    expect(region.x + region.width).toBeLessThanOrEqual(1024);
    expect(region.y + region.height).toBeLessThanOrEqual(1536);
  });
});
