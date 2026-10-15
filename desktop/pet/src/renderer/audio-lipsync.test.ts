import { describe, expect, it } from "vitest";

import { lipShapeForGeneration } from "./audio-lipsync";
import type { LipShape } from "./pet-renderer";

const OPEN_MOUTH: LipShape = {
  volume: 0.8,
  a: 0.8,
  e: 0,
  i: 0,
  o: 0,
  u: 0,
};

describe("lipShapeForGeneration", () => {
  it("drops a worklet result from a cancelled generation", () => {
    expect(
      lipShapeForGeneration("new", {
        generation: "old",
        shape: OPEN_MOUTH,
      }),
    ).toBeNull();
  });

  it("accepts only the current generation", () => {
    expect(
      lipShapeForGeneration("current", {
        generation: "current",
        shape: OPEN_MOUTH,
      }),
    ).toEqual(OPEN_MOUTH);
  });
});
