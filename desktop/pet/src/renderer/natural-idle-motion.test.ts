import { describe, expect, it, vi } from "vitest";
import { NaturalIdleMotion, type IdleHeadPose } from "./natural-idle-motion";

const settings = { enabled: true, intensity: 1, speed: 1 };

function seededRandom() {
  let seed = 12345;
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

function record(fps: number, seconds: number, intensity = 1): IdleHeadPose[] {
  const motion = new NaturalIdleMotion(seededRandom());
  motion.configure({ ...settings, intensity });
  return Array.from({ length: seconds * fps }, () => ({ ...motion.update(1 / fps, true) }));
}

describe("natural idle motion", () => {
  it("leaves existing models unchanged without opt-in, with no random work", () => {
    const random = vi.fn(() => 0.5);
    const motion = new NaturalIdleMotion(random);
    for (let frame = 0; frame < 600; frame++) {
      expect(motion.update(1 / 60, true)).toEqual({ x: 0, y: 0, z: 0 });
    }
    expect(random).not.toHaveBeenCalled();
  });

  it("visits varied multi-axis poses without jumps or a repeating fixed cycle", () => {
    const poses = record(60, 90, 2);
    for (const axis of ["x", "y", "z"] as const) {
      const values = poses.map((pose) => pose[axis]);
      expect(Math.max(...values) - Math.min(...values)).toBeGreaterThan(10);
      expect(Math.max(...values.map(Math.abs))).toBeLessThanOrEqual(30);
      const steps = values.slice(1).map((value, i) => Math.abs(value - values[i]!));
      // At maximum intensity even the occasional quick glance stays continuous.
      expect(Math.max(...steps)).toBeLessThan(2.5);
      const velocities = values.slice(1).map((value, i) => value - values[i]!);
      const accelerations = velocities.slice(1).map((value, i) => Math.abs(value - velocities[i]!));
      expect(Math.max(...accelerations)).toBeLessThan(0.25);
      expect(steps.some((step) => step < 0.001)).toBe(true);
      expect(values.slice(600, 1200)).not.toEqual(values.slice(1200, 1800));
    }
  });

  it("keeps the same tempo at 30 and 60 FPS", () => {
    const slow = record(30, 20);
    const fast = record(60, 20);
    for (let frame = 0; frame < slow.length; frame++) {
      for (const axis of ["x", "y", "z"] as const) {
        expect(Math.abs(slow[frame]![axis] - fast[frame * 2 + 1]![axis])).toBeLessThan(0.05);
      }
    }
  });

  it.each(["blocked", "disabled"])("fades out and resumes smoothly when %s", (reason) => {
    const motion = new NaturalIdleMotion(seededRandom());
    motion.configure(settings);
    let before = { x: 0, y: 0, z: 0 };
    for (let frame = 0; frame < 140; frame++) before = { ...motion.update(1 / 60, true) };
    if (reason === "disabled") motion.configure({ ...settings, enabled: false });
    const after = { ...motion.update(1 / 60, reason !== "blocked") };
    expect(Math.abs(after.x - before.x)).toBeLessThan(1);
    for (let frame = 0; frame < 240; frame++) motion.update(1 / 60, reason !== "blocked");
    expect(motion.update(1 / 60, reason !== "blocked")).toEqual({ x: 0, y: 0, z: 0 });
    motion.configure(settings);
    const resumed = motion.update(1 / 60, true);
    expect(Math.abs(resumed.x)).toBeLessThan(1);
  });
});
