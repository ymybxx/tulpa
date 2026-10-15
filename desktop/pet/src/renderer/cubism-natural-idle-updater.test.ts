import type { CubismModel } from "@cubism/model/cubismmodel";
import { CubismUpdateScheduler } from "@cubism/motion/cubismupdatescheduler";
import { CubismUpdateOrder, ICubismUpdater } from "@cubism/motion/icubismupdater";
import { describe, expect, it } from "vitest";
import { CubismNaturalIdleUpdater } from "./cubism-natural-idle-updater";

describe("Cubism natural idle integration", () => {
  it("feeds real head inputs before physics and preserves unrelated parameters", () => {
    const ids = ["ParamAngleX", "ParamAngleZ", "Param85", "ParamMouthOpenY", "ParamBodyAngleX1"];
    const base = [0, 0, 1, 0.7, 0];
    const values = [...base];
    const model = {
      getModel: () => ({ parameters: { ids } }),
      getParameterValueByIndex: (index: number) => values[index],
      setParameterValueByIndex: (index: number, value: number) => {
        expect(index).toBeGreaterThanOrEqual(0);
        values[index] = Math.min(5, Math.max(-5, value));
      },
    } as unknown as CubismModel;
    let available = true;
    const idle = new CubismNaturalIdleUpdater(model, () => available, () => 0.8);
    idle.configure({ enabled: true, intensity: 2, speed: 1 });
    class Physics extends ICubismUpdater {
      constructor() { super(CubismUpdateOrder.CubismUpdateOrder_Physics); }
      onLateUpdate(): void { values[4] = values[0]! / 2; }
    }
    const scheduler = new CubismUpdateScheduler();
    scheduler.addUpdatableList(new Physics());
    scheduler.addUpdatableList(idle);
    const advance = () => {
      values.splice(0, values.length, ...base);
      scheduler.onLateUpdate(model, 1 / 60);
    };
    for (let i = 0; i < 180; i++) advance();
    expect(Math.abs(values[4]!)).toBeGreaterThan(0.2);
    expect(values[0]).toBeLessThanOrEqual(5);
    expect(values.slice(2, 4)).toEqual([1, 0.7]);
    available = false;
    for (let i = 0; i < 240; i++) advance();
    expect(values).toEqual(base);
  });
});
