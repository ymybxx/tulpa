import { describe, expect, it } from "vitest";
import { FacialTransition, transitionProfile } from "./facial-transition";

describe("C2 facial trajectories", () => {
  it("preserves position, velocity and acceleration on an in-flight reversal", () => {
    const move = new FacialTransition(0, -30, 30, transitionProfile("head"));
    move.setTarget(24); move.update(0.2);
    const before = move.snapshot();
    expect(before.velocity).toBeGreaterThan(0); expect(before.acceleration).toBeGreaterThan(0);
    move.setTarget(-24);
    expect(move.snapshot()).toEqual(before);
    move.update(0.00001);
    const after = move.snapshot();
    expect(after.position).toBeGreaterThan(before.position);
    expect(Math.abs(after.acceleration - before.acceleration)).toBeLessThan(60 * 90 * 0.00001 * 1.01);
    move.update(4); expect(move.snapshot()).toEqual({ position: -24, velocity: 0, acceleration: 0 });
  });

  it("bounds positions and derivatives through rapid endpoint retargeting", () => {
    const profile = transitionProfile("head"), move = new FacialTransition(0.5, 0, 1, profile);
    let previous = move.snapshot(), seed = 17;
    for (let i = 0; i < 2400; i++) {
      if (i % 47 === 0) {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        move.setTarget(i % 3 === 0 ? 0 : i % 3 === 1 ? 1 : seed / 0xffffffff);
      }
      move.update(1 / 240);
      const next = move.snapshot();
      expect(next.position).toBeGreaterThanOrEqual(0); expect(next.position).toBeLessThanOrEqual(1);
      expect(Math.abs(next.velocity)).toBeLessThanOrEqual(profile.velocity + 1e-6);
      expect(Math.abs(next.acceleration)).toBeLessThanOrEqual(profile.acceleration + 1e-6);
      expect(Math.abs(next.acceleration - previous.acceleration) * 240).toBeLessThanOrEqual(profile.jerk + 1e-4);
      previous = next;
    }
    move.setTarget(0.3); move.update(10);
    expect(move.snapshot()).toEqual({ position: 0.3, velocity: 0, acceleration: 0 });
  });

  it("is frame-rate independent and duplicate targets do not restart a motion", () => {
    const samples = [30, 60, 144].map(fps => {
      const move = new FacialTransition(0.2, 0, 1, transitionProfile("body"));
      move.setTarget(0.9);
      for (let i = 0; i < fps; i++) { move.setTarget(0.9); move.update(1 / fps); }
      return move.snapshot();
    });
    for (const next of samples.slice(1)) for (const key of ["position", "velocity", "acceleration"] as const) {
      expect(next[key]).toBeCloseTo(samples[0]![key], 10);
    }
  });

  it("is invariant to arbitrary native scales, asymmetric neutral values and reversed bindings", () => {
    const canonical = new FacialTransition(0.3, 0, 1, transitionProfile("head"));
    const native = new FacialTransition(620, -100, 2300, transitionProfile("head"));
    const reversed = new FacialTransition(1580, -100, 2300, transitionProfile("head"));
    for (const target of [0.9, 0.1, 0.55]) {
      canonical.setTarget(target); native.setTarget(-100 + 2400 * target); reversed.setTarget(2300 - 2400 * target);
      for (let i = 0; i < 70; i++) {
        canonical.update(1 / 120); native.update(1 / 120); reversed.update(1 / 120);
        expect((native.value + 100) / 2400).toBeCloseTo(canonical.value, 8);
        expect((2300 - reversed.value) / 2400).toBeCloseTo(canonical.value, 8);
      }
    }
  });

  it("lets eyes respond sooner than head/body and takes longer for larger moves", () => {
    const eyes = new FacialTransition(0, 0, 1, transitionProfile("eyes"));
    const head = new FacialTransition(0, 0, 1, transitionProfile("head"));
    const body = new FacialTransition(0, 0, 1, transitionProfile("body"));
    const small = new FacialTransition(0, 0, 1, transitionProfile("head"));
    for (const m of [eyes, head, body]) { m.setTarget(0.6); m.update(0.3); }
    small.setTarget(0.1); small.update(0.3);
    expect(eyes.value).toBeGreaterThan(head.value); expect(head.value).toBeGreaterThan(body.value);
    expect(small.value / 0.1).toBeGreaterThan(head.value / 0.6);
    expect(head.snapshot().velocity).toBeGreaterThan(0);
  });

  it("handles invalid time, fixed parameters and resuming after a long frame", () => {
    const fixed = new FacialTransition(10, 10, 10, transitionProfile("face"));
    fixed.setTarget(99); expect(fixed.update(1)).toBe(10);
    const move = new FacialTransition(0, -1, 1, transitionProfile("face"));
    move.setTarget(9);
    expect(move.update(NaN)).toBe(0); expect(move.update(-1)).toBe(0);
    move.setTarget(NaN); expect(move.update(10)).toBe(1);
    expect(move.snapshot()).toEqual({ position: 1, velocity: 0, acceleration: 0 });
  });
});
