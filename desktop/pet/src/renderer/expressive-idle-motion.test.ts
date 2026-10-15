import { describe, expect, it } from "vitest";
import { idlePresetSettings } from "../shared/natural-idle-settings";
import { ExpressiveIdleMotion, type IdlePoseEvent } from "./expressive-idle-motion";
const random = () => { let state = 54321; return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; }; };
describe("continuous expressive idle", () => {
  it("stays finite, continuous and bounded over ten minutes without retaining a timeline", () => {
    const motion = new ExpressiveIdleMotion(random());
    motion.configure(idlePresetSettings("active"));
    let previous = motion.update(1 / 60), maximumStep = 0, closed = 0;
    for (let frame = 0; frame < 36000; frame++) {
      const last = { ...previous };
      previous = motion.update(1 / 60);
      for (const key of ["x", "y", "z"] as const) {
        expect(Math.abs(previous[key])).toBeLessThanOrEqual(28);
        maximumStep = Math.max(maximumStep, Math.abs(previous[key] - last[key]));
      }
      expect(Object.values(previous).every(Number.isFinite)).toBe(true);
      if (previous.blink === 0)
        closed++;
    }
    expect(maximumStep).toBeLessThan(1.5);
    expect(closed).toBeGreaterThan(30);
    expect(motion.diagnostics().retainedKnots).toBeLessThanOrEqual(5);
  });
  it("separately controls large moves and prevents quick large reversals", () => {
    for (const probability of [0, 1]) {
      const events: IdlePoseEvent[] = [];
      const motion = new ExpressiveIdleMotion(random(), event => events.push(event));
      const config = idlePresetSettings("active");
      config.speed = 1.5;
      config.expressive!.largeMotionProbability = probability;
      config.expressive!.reverseInterval = 3;
      motion.configure(config);
      for (let i = 0; i < 18000; i++)
        motion.update(1 / 60);
      expect(events.some(e => e.large)).toBe(probability === 1);
      const previous = new Map<string, IdlePoseEvent>();
      for (const event of events) {
        if (!event.large || !event.axis)
          continue;
        const last = previous.get(event.axis);
        if (last && last.side !== event.side)
          expect(event.time - last.time).toBeGreaterThanOrEqual(4.5);
        previous.set(event.axis, event);
      }
    }
  });
  it("preserves independent eye timing when pose settings change", () => {
    const a = new ExpressiveIdleMotion(random()), b = new ExpressiveIdleMotion(random());
    const config = idlePresetSettings("active");
    a.configure(config);
    b.configure({ ...config, speed: 0.5, expressive: { ...config.expressive!, pauseProbability: 0.8, largeMotionProbability: 0 } });
    for (let i = 0; i < 3600; i++) {
      const x = a.update(1 / 60), y = b.update(1 / 60);
      expect([x.blink, x.aimX, x.aimY]).toEqual([y.blink, y.aimX, y.aimY]);
    }
  });
  it("runs at the same pace at 30 and 60 fps", () => {
    const a = new ExpressiveIdleMotion(random()), b = new ExpressiveIdleMotion(random());
    a.configure(idlePresetSettings("active"));
    b.configure(idlePresetSettings("active"));
    for (let i = 0; i < 1200; i++) {
      const x = a.update(1 / 30);
      b.update(1 / 60);
      const y = b.update(1 / 60);
      for (const key of ["x", "y", "z", "aimX", "aimY"] as const)
        expect(Math.abs(x[key] - y[key])).toBeLessThan(0.01);
    }
  });
  it("keeps head/body motion and blinking independent of gaze activity, including zero", () => {
    const a = new ExpressiveIdleMotion(random()), b = new ExpressiveIdleMotion(random());
    const config = idlePresetSettings("active");
    a.configure({ ...config, expressive: { ...config.expressive!, eyeActivity: 0 } });
    b.configure({ ...config, expressive: { ...config.expressive!, eyeActivity: 2 } });
    let closed = 0;
    for (let frame = 0; frame < 3600; frame++) {
      const x = a.update(1 / 60), y = b.update(1 / 60);
      for (const key of ["x", "y", "z", "lean", "posture", "yawBias", "blink"] as const)
        expect(x[key]).toBe(y[key]);
      expect(x.aimX).toBe(0);
      expect(x.aimY).toBe(0.035);
      if (x.blink === 0) closed++;
    }
    expect(closed).toBeGreaterThan(5);
    b.configure({ ...config, expressive: { ...config.expressive!, blink: false } });
    for (let frame = 0; frame < 600; frame++) expect(b.update(1 / 60).blink).toBe(1);
  });
});
