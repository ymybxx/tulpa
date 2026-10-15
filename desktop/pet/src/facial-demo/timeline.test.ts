import { describe, expect, it } from "vitest";
import { FaceTimeline } from "./timeline";
const NEUTRAL_FACE = { ParamEyeLOpen: 1, ParamEyeROpen: 1, ParamCheek: 0 };
describe("spoken segment face scheduling", () => {
  it("stages future values, applies only current, drops expired values", () => {
    const timeline = new FaceTimeline();
    const future = { ...NEUTRAL_FACE, ParamCheek: 0.8 };
    expect(timeline.receive(1, future)).toBeNull();
    expect(timeline.enter(0)).toBeNull();
    expect(timeline.receive(0, NEUTRAL_FACE)).toEqual(NEUTRAL_FACE);
    expect(timeline.enter(1)).toEqual(future);
    expect(timeline.receive(0, NEUTRAL_FACE)).toBeNull();
    expect(timeline.enter(0)).toBeNull();
    expect(timeline.expired).toBe(1);
  });
  it("expires results once their audio finishes even before the next segment", () => {
    const timeline = new FaceTimeline();
    timeline.enter(0); timeline.finish(0);
    expect(timeline.receive(0, NEUTRAL_FACE)).toBeNull();
    expect(timeline.expired).toBe(1);
    expect(timeline.receive(1, NEUTRAL_FACE)).toBeNull();
    expect(timeline.enter(1)).toEqual(NEUTRAL_FACE);
  });
});
