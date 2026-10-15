import { afterAll, beforeAll, describe, expect, it } from "vitest";

import type { CubismIdHandle } from "@cubism/id/cubismid";
import { CubismPose } from "@cubism/effect/cubismpose";
import { CubismFramework } from "@cubism/live2dcubismframework";
import type { CubismModel } from "@cubism/model/cubismmodel";
import { CubismExpressionMotion } from "@cubism/motion/cubismexpressionmotion";
import { CubismMotion } from "@cubism/motion/cubismmotion";
import { CubismUpdateScheduler } from "@cubism/motion/cubismupdatescheduler";
import { CubismEyeBlinkUpdater } from "@cubism/motion/cubismeyeblinkupdater";
import { CubismEyeBlink } from "@cubism/effect/cubismeyeblink";
import { CubismBehaviorExpressionUpdater } from "./cubism-behavior-expression-updater";

import {
  captureCubismPoseSelection,
  CubismBehaviorGroup,
  restoreCubismDefaultParameters,
  restoreCubismPoseSelection,
} from "./cubism-behavior-group";

beforeAll(() => {
  Object.assign(globalThis, {
    Live2DCubismCore: {
      Version: { csmGetVersion: () => 0x05030000 },
      Memory: { initializeAmountOfMemory: () => undefined },
      Logging: {
        csmSetLogFunction: () => undefined,
        csmGetLogFunction: () => null,
      },
    },
  });
  CubismFramework.startUp();
  CubismFramework.initialize();
});

afterAll(() => {
  CubismFramework.cleanUp();
});

describe("CubismBehaviorGroup with official managers", () => {
  it("keeps closed-eye expressions visible over automatic blink and restores blink on stop", () => {
    const model = new ParameterModel(["ParamEyeLOpen"], [1]);
    const eyeId = CubismFramework.getIdManager().getId("ParamEyeLOpen");
    const blink = CubismEyeBlink.create(); blink.setParameterIds([eyeId]);
    blink.setBlinkingInterval(1_000_000);
    const value = expression("ParamEyeLOpen", 0, "Multiply");
    const group = new CubismBehaviorGroup([], [value], false);
    let active: CubismBehaviorGroup | null = group;
    const scheduler = new CubismUpdateScheduler();
    // Register out of order to exercise the official scheduler ordering too.
    scheduler.addUpdatableList(new CubismBehaviorExpressionUpdater(() => active, () => null));
    scheduler.addUpdatableList(new CubismEyeBlinkUpdater(() => false, blink));

    for (let frame = 0; frame < 30; frame++) {
      model.set("ParamEyeLOpen", 1);
      scheduler.onLateUpdate(model.value as CubismModel, 1 / 60);
      expect(model.get("ParamEyeLOpen")).toBeCloseTo(0, 5);
    }
    active = null;
    scheduler.onLateUpdate(model.value as CubismModel, 1 / 60);
    expect(model.get("ParamEyeLOpen")).toBeCloseTo(1, 5);
    scheduler.release(); group.release(); value.release(); CubismEyeBlink.delete(blink);
  });

  it("applies the speaking face layer after active expression layers exactly once per frame", () => {
    const model = new ParameterModel(["ParamEyeLOpen"], [1]);
    const values = [expression("ParamEyeLOpen", 0.5, "Multiply"), expression("ParamEyeLOpen", 0.25, "Add")];
    const active = new CubismBehaviorGroup([], [values[0]!], false);
    const persistent = new CubismBehaviorGroup([], [values[1]!], false);
    const updater = new CubismBehaviorExpressionUpdater(() => active, () => persistent);
    updater.onLateUpdate(model.value as CubismModel, 1 / 60);
    expect(model.get("ParamEyeLOpen")).toBeCloseTo(0.75, 5);
    active.release(); persistent.release(); values.forEach((value) => value.release());
  });

  it("updates parallel motions first and expressions afterward in saved order", () => {
    const model = new ParameterModel(["ParamX"]);
    const motions = [motion("ParamX", 1), motion("ParamX", 2)];
    const expressions = [expression("ParamX", 3, "Overwrite"), expression("ParamX", 2, "Add")];
    const group = new CubismBehaviorGroup(motions, expressions, false);

    expect(group.managerCounts()).toEqual({ motions: 2, expressions: 2 });
    expect(group.updateMotions(model.value as CubismModel, 1 / 30)).toBe(true);
    // Both managers started on their first update and the second selected
    // motion deterministically wins the overlapping parameter.
    expect(model.get("ParamX")).toBeCloseTo(2, 5);
    expect(group.updateExpressions(model.value as CubismModel, 1 / 30)).toBe(true);
    expect(model.get("ParamX")).toBeCloseTo(5, 5);

    group.release();
    motions.forEach((value) => value.release());
    expressions.forEach((value) => value.release());
    expect(group.managerCounts()).toEqual({ motions: 0, expressions: 0 });
  });

  it("forces state motions to loop and restores authored flags on release", () => {
    const model = new ParameterModel(["ParamX"]);
    const value = motion("ParamX", 1);
    value.setFadeInTime(0.4);
    value.setFadeOutTime(0.8);
    expect(value.getLoop()).toBe(false);
    const group = new CubismBehaviorGroup([value], [], true);

    expect(value.getLoop()).toBe(true);
    expect(group.motionCycleDurationSeconds()).toBe(1);
    expect(group.fadeInDurationSeconds()).toBe(0.4);
    expect(group.fadeOutDurationSeconds()).toBe(0.8);
    group.updateMotions(model.value as CubismModel, 1.2);
    group.updateMotions(model.value as CubismModel, 1.2);
    expect(group.motionsFinished()).toBe(false);

    group.release();
    expect(value.getLoop()).toBe(false);
    value.release();
  });

  it("reports a non-looping group complete only after every motion finishes", () => {
    const model = new ParameterModel(["ParamX", "ParamY"]);
    const values = [motion("ParamX", 1, 0.2), motion("ParamY", 1, 0.5)];
    const group = new CubismBehaviorGroup(values, [], false);

    expect(group.motionCycleDurationSeconds()).toBe(0.5);

    group.updateMotions(model.value as CubismModel, 0.25);
    expect(group.motionsFinished()).toBe(false);
    group.updateMotions(model.value as CubismModel, 0.3);
    expect(group.motionsFinished()).toBe(false);
    group.updateMotions(model.value as CubismModel, 0.3);
    expect(group.motionsFinished()).toBe(true);

    group.release();
    values.forEach((value) => value.release());
  });

  it("restores authored defaults when no state candidate remains", () => {
    const model = new ParameterModel(["ParamX"], [0.25]);
    model.set("ParamX", 1.5);
    restoreCubismDefaultParameters(model.value as CubismModel);
    expect(model.get("ParamX")).toBe(0.25);
    expect(model.saved).toBe(true);
  });

  it("restores virtual pose selectors and lets official CubismPose fade back", () => {
    const model = new PoseModel(["PartArmRA", "PartArmRB"]);
    const poseBuffer = jsonBuffer({
      Type: "Live2D Pose",
      FadeInTime: 0.5,
      Groups: [[{ Id: "PartArmRA", Link: [] }, { Id: "PartArmRB", Link: [] }]],
    });
    const pose = CubismPose.create(poseBuffer, poseBuffer.byteLength);
    const modelValue = model.value as CubismModel;
    pose.updateParameters(modelValue, 0);
    const baseline = captureCubismPoseSelection(modelValue);

    model.setSelector("PartArmRA", 0);
    model.setSelector("PartArmRB", 1);
    pose.updateParameters(modelValue, 0.5);
    expect(model.opacity("PartArmRA")).toBe(0);
    expect(model.opacity("PartArmRB")).toBe(1);

    restoreCubismPoseSelection(modelValue, baseline);
    pose.updateParameters(modelValue, 0.25);
    expect(model.opacity("PartArmRA")).toBeGreaterThan(0);
    expect(model.opacity("PartArmRA")).toBeLessThan(1);
    expect(model.opacity("PartArmRB")).toBeGreaterThan(0);
    expect(model.opacity("PartArmRB")).toBeLessThan(1);
    pose.updateParameters(modelValue, 0.25);
    expect(model.opacity("PartArmRA")).toBe(1);
    expect(model.opacity("PartArmRB")).toBe(0);

    CubismPose.delete(pose);
  });

  it("primes a persistent expression to full weight before drawing", () => {
    const model = new ParameterModel(["ParamX"]);
    const value = expression("ParamX", 1, "Overwrite", 1);
    const group = new CubismBehaviorGroup([], [value], false);

    group.primeExpressions(model.value as CubismModel);

    expect(model.get("ParamX")).toBeCloseTo(1, 5);
    group.release();
    value.release();
  });
});

function motion(id: string, value: number, duration = 1): CubismMotion {
  const buffer = jsonBuffer({
    Version: 3,
    Meta: {
      Duration: duration,
      Fps: 30,
      Loop: false,
      AreBeziersRestricted: true,
      CurveCount: 1,
      TotalSegmentCount: 1,
      TotalPointCount: 2,
      UserDataCount: 0,
      TotalUserDataSize: 0,
    },
    Curves: [{
      Target: "Parameter",
      Id: id,
      FadeInTime: 0,
      FadeOutTime: 0,
      Segments: [0, value, 0, duration, value],
    }],
    UserData: [],
  });
  const result = CubismMotion.create(buffer, buffer.byteLength);
  result.setEffectIds([], []);
  return result;
}

function expression(
  id: string,
  value: number,
  blend: "Add" | "Multiply" | "Overwrite",
  fadeTime = 0,
): CubismExpressionMotion {
  const buffer = jsonBuffer({
    Type: "Live2D Expression",
    FadeInTime: fadeTime,
    FadeOutTime: fadeTime,
    Parameters: [{ Id: id, Value: value, Blend: blend }],
  });
  return CubismExpressionMotion.create(buffer, buffer.byteLength);
}

function jsonBuffer(value: unknown): ArrayBuffer {
  return new TextEncoder().encode(JSON.stringify(value)).buffer as ArrayBuffer;
}

class ParameterModel {
  private readonly ids: CubismIdHandle[];
  private readonly values: number[];
  private readonly defaults: number[];

  saved = false;

  constructor(names: string[], defaults = names.map(() => 0)) {
    this.ids = names.map((name) => CubismFramework.getIdManager().getId(name));
    this.defaults = [...defaults];
    this.values = [...defaults];
  }

  get value(): unknown {
    return {
      getParameterIndex: (id: CubismIdHandle) => this.ids.indexOf(id),
      getParameterCount: () => this.values.length,
      getParameterDefaultValue: (index: number) => this.defaults[index] ?? 0,
      getParameterValueByIndex: (index: number) => this.values[index] ?? 0,
      setParameterValueByIndex: (index: number, value: number, weight = 1) => {
        const previous = this.values[index] ?? 0;
        this.values[index] = previous + (value - previous) * weight;
      },
      getParameterValueById: (id: CubismIdHandle) => this.values[this.ids.indexOf(id)] ?? 0,
      setParameterValueById: (id: CubismIdHandle, value: number, weight = 1) => {
        const index = this.ids.indexOf(id);
        const previous = this.values[index] ?? 0;
        this.values[index] = previous + (value - previous) * weight;
      },
      addParameterValueById: (id: CubismIdHandle, value: number, weight = 1) => {
        const index = this.ids.indexOf(id);
        this.values[index] = (this.values[index] ?? 0) + value * weight;
      },
      multiplyParameterValueById: (id: CubismIdHandle, value: number, weight = 1) => {
        const index = this.ids.indexOf(id);
        this.values[index] = (this.values[index] ?? 0) * (1 + (value - 1) * weight);
      },
      isRepeat: () => false,
      getParameterRepeatValue: (_index: number, value: number) => value,
      setModelOapcity: () => undefined,
      saveParameters: () => { this.saved = true; },
    };
  }

  get(name: string): number {
    const id = CubismFramework.getIdManager().getId(name);
    return this.values[this.ids.indexOf(id)] ?? 0;
  }

  set(name: string, value: number): void {
    const id = CubismFramework.getIdManager().getId(name);
    this.values[this.ids.indexOf(id)] = value;
  }
}

class PoseModel {
  private readonly ids: CubismIdHandle[];
  private readonly selectors: number[];
  private readonly opacities: number[];

  constructor(names: string[]) {
    this.ids = names.map((name) => CubismFramework.getIdManager().getId(name));
    this.selectors = names.map(() => 0);
    this.opacities = names.map(() => 0);
  }

  get value(): unknown {
    return {
      getPartCount: () => this.ids.length,
      getPartId: (index: number) => this.ids[index],
      getPartIndex: (id: CubismIdHandle) => this.ids.indexOf(id),
      getPartOpacityByIndex: (index: number) => this.opacities[index] ?? 0,
      setPartOpacityByIndex: (index: number, value: number) => {
        this.opacities[index] = value;
      },
      getParameterIndex: (id: CubismIdHandle) => this.ids.indexOf(id),
      getParameterValueByIndex: (index: number) => this.selectors[index] ?? 0,
      setParameterValueByIndex: (index: number, value: number) => {
        this.selectors[index] = value;
      },
    };
  }

  opacity(name: string): number {
    return this.opacities[this.index(name)] ?? 0;
  }

  setSelector(name: string, value: number): void {
    this.selectors[this.index(name)] = value;
  }

  private index(name: string): number {
    return this.ids.indexOf(CubismFramework.getIdManager().getId(name));
  }
}
