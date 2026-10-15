import { describe, expect, it } from "vitest";
import { facialConflictPlan, resolveFacialConflicts, type ConflictPart } from "./facial-conflicts";

const part = (id: string, group = "custom", min = 0, max = 1, neutral = 0): ConflictPart => ({ id, nativeId: `N${id}`, group, neutral, min, max });
const parts = [part("shush"), part("chin"), part("walkArm", "custom", -1, 1, 0), part("anger"), part("head", "head", -30, 30, 0), part("armA", "arms", -30, 30, 0)];
const effects = {
  // Two hand poses: each hides the resting arm (1, 2) and draws its own.
  Nshush: { shown: [10, 11], hidden: [1, 2], moved: [] },
  Nchin: { shown: [12], hidden: [1, 2], moved: [] },
  // Moves the resting arms and the pose arms alike.
  NwalkArm: { shown: [], hidden: [], moved: [1, 2, 10, 12] },
  // Only adds a mark.
  Nanger: { shown: [30], hidden: [], moved: [] },
  Nhead: { shown: [], hidden: [], moved: [10, 11, 30, 40] },
  NarmA: { shown: [], hidden: [], moved: [2, 11] },
};

describe("facial conflicts", () => {
  const plan = facialConflictPlan(parts, effects);

  it("pairs parts that swap the same drawables and holds movers of a swapped arm", () => {
    expect(plan.exclusive).toEqual([["shush", "chin"]]);
    expect(plan.holds).toEqual({ shush: ["walkArm", "armA"], chin: ["walkArm"] });
  });

  it("keeps the stronger of two excluding parts and rests what moves the kept pose", () => {
    const values = resolveFacialConflicts({ shush: 1, chin: 0.6, walkArm: -0.8, anger: 1, head: 20, armA: 12 }, parts, plan);
    expect(values).toEqual({ shush: 1, chin: 0, walkArm: 0, anger: 1, head: 20, armA: 0 });
  });

  it("lets everything move while no pose is on", () => {
    const answer = { shush: 0, chin: 0, walkArm: -0.8, anger: 1, head: 20, armA: 12 };
    expect(resolveFacialConflicts(answer, parts, plan)).toEqual(answer);
  });

  it("without effects, changes nothing", () => {
    const empty = facialConflictPlan(parts, undefined);
    expect(resolveFacialConflicts({ shush: 1, chin: 1 }, parts, empty)).toEqual({ shush: 1, chin: 1 });
  });
});
