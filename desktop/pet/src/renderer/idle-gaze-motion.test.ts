import { describe, expect, it } from "vitest";
import { IdleGazeMotion } from "./idle-gaze-motion";

const random = (seed: number) => () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 4294967296;
};
function sample(activity: number, seed: number) {
  const gaze = new IdleGazeMotion(random(seed));
  gaze.configure(activity);
  let previous = { aimX: 0, aimY: 0.035 }, moving = false, moves = 0, maxStep = 0, openSum = 0;
  const targets = [{ x: 0, y: 0.035 }];
  for (let frame = 0; frame < 10800; frame++) {
    const p = gaze.update(frame / 60);
    const distance = Math.hypot(p.aimX - previous.aimX, p.aimY - previous.aimY);
    const nowMoving = distance > 0.00001;
    if (nowMoving && !moving) moves++;
    if (!nowMoving && moving) targets.push({ x: p.aimX, y: p.aimY });
    moving = nowMoving;
    maxStep = Math.max(maxStep, distance);
    expect(p.open).toBeGreaterThanOrEqual(1);
    expect(p.open).toBeLessThanOrEqual(1.012);
    expect(Math.abs(p.aimX)).toBeLessThanOrEqual(1);
    expect(Math.abs(p.aimY)).toBeLessThanOrEqual(1);
    openSum += p.open;
    previous = { aimX: p.aimX, aimY: p.aimY };
  }
  return { moves, targets, maxStep, meanOpen: openSum / 10800 };
}

describe("idle gaze activity", () => {
  it("increases gaze changes with activity while keeping the ordinary eye pose stable", () => {
    const totals = [0, 0, 0];
    for (const seed of [1, 54321, 20260916]) {
      for (const [index, activity] of [0.5, 1, 2].entries()) {
        const measured = sample(activity, seed);
        totals[index]! += measured.moves;
        expect(measured.meanOpen).toBeLessThan(1.002);
        expect(measured.maxStep).toBeLessThan(0.18);
      }
    }
    expect(totals[1]).toBeGreaterThan(totals[0]! * 1.2);
    expect(totals[2]).toBeGreaterThan(totals[1]! * 1.3);
  });

  it("mixes small fixations with brief glances that return to the previous fixation", () => {
    const { targets } = sample(2, 54321);
    let glances = 0, smallMoves = 0;
    const directions: string[] = [];
    for (let i = 1; i < targets.length - 1; i++) {
      const p = targets[i]!, previous = targets[i - 1]!, next = targets[i + 1]!;
      if (Math.hypot(p.x, p.y - 0.035) > 0.4) {
        glances++;
        directions.push(p.x < -0.4 ? "left" : p.x > 0.4 ? "right" : p.y < 0 ? "down" : "up");
        expect(Math.hypot(next.x - previous.x, next.y - previous.y)).toBeLessThan(0.002);
      } else if (Math.hypot(p.x - previous.x, p.y - previous.y) < 0.35) smallMoves++;
    }
    expect(glances).toBeGreaterThan(5);
    expect(smallMoves).toBeGreaterThan(glances);
    for (let i = 0; i + 4 <= directions.length; i += 4)
      expect([...directions.slice(i, i + 4)].sort()).toEqual(["down", "left", "right", "up"]);
    for (let i = 1; i < directions.length; i++) expect(directions[i]).not.toBe(directions[i - 1]);
  });

  it("settles smoothly at zero and resumes without a position jump", () => {
    const gaze = new IdleGazeMotion(random(12345));
    gaze.configure(2);
    for (let frame = 0; frame < 240; frame++) gaze.update(frame / 60);
    const previous = { ...gaze.update(4) };
    gaze.configure(0);
    expect(gaze.update(4)).toMatchObject({ aimX: previous.aimX, aimY: previous.aimY });
    for (let frame = 241; frame < 1200; frame++) {
      const p = gaze.update(frame / 60);
      if (frame > 260) expect(p).toEqual({ aimX: 0, aimY: 0.035, open: 1, brow: 0 });
    }
    gaze.configure(2);
    expect(gaze.update(1199 / 60)).toEqual({ aimX: 0, aimY: 0.035, open: 1, brow: 0 });
    expect(gaze.update(21).aimX).not.toBe(0);
  });
});
