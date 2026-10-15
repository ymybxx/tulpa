import { describe, expect, it } from "vitest";
import type { CubismLook } from "@cubism/effect/cubismlook";
import type { CubismTargetPoint } from "@cubism/math/cubismtargetpoint";
import type { CubismModel } from "@cubism/model/cubismmodel";
import { LeadingLookUpdater } from "./leading-look";

describe("looking at a target, eyes first", () => {
  it("jumps the eyes ahead of the slower head, then rests where the head and eyes share the turn", () => {
    const seen = { head: 0, eyes: 0 };
    // The drag eases toward its target over about half a second.
    let target = 0, x = 0;
    const drag = { set: (value: number) => { target = value; }, update: (dt: number) => { x += (target - x) * (1 - Math.exp(-dt / 0.15)); },
      getX: () => x, getY: () => 0 } as unknown as CubismTargetPoint;
    const head = { updateParameters: (_model: unknown, value: number) => { seen.head = value; } } as unknown as CubismLook;
    const eyes = { updateParameters: (_model: unknown, value: number) => { seen.eyes = value; } } as unknown as CubismLook;
    const look = new LeadingLookUpdater(head, eyes, drag), model = {} as CubismModel;
    look.setTarget(0.5, 0);
    const trace: Array<{ head: number; eyes: number }> = [];
    for (let frame = 0; frame < 120; frame++) { look.onLateUpdate(model, 1 / 60); trace.push({ ...seen }); }
    // Past their resting turn while the head is still well short of its own.
    expect(trace[5]!.eyes).toBeGreaterThan(0.55);
    expect(trace[5]!.head).toBeLessThan(trace[5]!.eyes / 2);
    expect(trace.at(-1)!.head).toBeCloseTo(0.5, 2);
    expect(trace.at(-1)!.eyes).toBeCloseTo(0.5, 2);
  });
});
