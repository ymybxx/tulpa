import { describe, expect, it } from "vitest";
import type { EmotionAsset } from "../../src/shared/emotion-matching";
import { stateValidationAsset } from "./validation-plan";

describe("persistent state sampling", () => {
  it("captures after the expression transient duration without dropping the authored fade", () => {
    const input = { kind: "expression", durationMs: 700, timestampsMs: [700] } as EmotionAsset;
    expect(stateValidationAsset(input).timestampsMs).toEqual([700, 2400]);
    expect(input.timestampsMs).toEqual([700]);
    expect(stateValidationAsset({ ...input, durationMs: 5000, timestampsMs: [5000] }).timestampsMs.at(-1)).toBe(5200);
  });
  it("keeps the original motion coverage and adds both sides of the first loop boundary", () => {
    const input = { kind: "motion", durationMs: 118_000, timestampsMs: [200, 29_500, 59_000, 117_983] } as EmotionAsset;
    const planned = stateValidationAsset(input).timestampsMs;
    expect(planned.slice(0, 4)).toEqual(input.timestampsMs);
    expect(planned.slice(-2)).toEqual([118_017, 118_200]);
    expect(planned.length).toBeLessThanOrEqual(16);
    expect(planned.at(-1)).toBeLessThan(120_000);
  });
});
