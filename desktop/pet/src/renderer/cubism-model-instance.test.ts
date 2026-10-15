import { describe, expect, it, vi } from "vitest";
import { CubismUpdateScheduler } from "@cubism/motion/cubismupdatescheduler";
import { ICubismUpdater } from "@cubism/motion/icubismupdater";
import { CubismMatrix44 } from "@cubism/math/cubismmatrix44";

vi.hoisted(() => {
  Object.assign(globalThis, {
    Live2DCubismCore: {
      ColorBlendType_Normal: 0,
      ColorBlendType_AddGlow: 1,
      ColorBlendType_Add: 2,
      ColorBlendType_Darken: 3,
      ColorBlendType_Multiply: 4,
      ColorBlendType_ColorBurn: 5,
      ColorBlendType_LinearBurn: 6,
      ColorBlendType_Lighten: 7,
      ColorBlendType_Screen: 8,
      ColorBlendType_ColorDodge: 9,
      ColorBlendType_Overlay: 10,
      ColorBlendType_SoftLight: 11,
      ColorBlendType_HardLight: 12,
      ColorBlendType_LinearLight: 13,
      ColorBlendType_Hue: 14,
      ColorBlendType_Color: 15,
      ColorBlendType_AddCompatible: 16,
      ColorBlendType_MultiplyCompatible: 17,
    },
  });
});

import {
  CubismModelInstance,
  DRAWABLE_HIT_BOUNDS_REFRESH_SECONDS,
  DrawableHitBoundsRefreshCadence,
  facialDemoPhysicsOutputOnlyIds,
} from "./cubism-model-instance";

describe("pet viewport offset", () => {
  it("moves the drawn bounds and pointer projection together without resizing", () => {
    const projection = new CubismMatrix44();
    const instance = Object.create(CubismModelInstance.prototype) as CubismModelInstance;
    Object.assign(instance, {
      projection,
      _modelMatrix: new CubismMatrix44(),
      _model: squareMesh(),
      displayMetrics: { cssWidth: 1_000, cssHeight: 800 },
      drawableHitBounds: new Float32Array([0, 1, 0, 1]),
      viewOffsetX: 0,
      viewOffsetY: 0,
      viewZoom: 1,
      zoomPanX: 0,
      zoomPanY: 0,
    });
    instance.setViewOffset(50, -100);
    expect(instance.getBounds()?.x).toBeCloseTo(550);
    expect(instance.getBounds()?.y).toBeCloseTo(-100);
    expect(instance.getBounds()?.width).toBeCloseTo(500);
    expect(instance.getBounds()?.height).toBeCloseTo(400);
    expect(instance.getBounds()?.middle?.y).toBeCloseTo(100);
    expect(instance.getBounds()?.middle?.left).toBeCloseTo(550);
    expect(instance.getBounds()?.middle?.right).toBeCloseTo(1_050);
    expect(projection.invertTransformX(0.1)).toBeCloseTo(0);
    expect(projection.invertTransformY(0.25)).toBeCloseTo(0);
    instance.setViewOffset(0, 0);
    expect(instance.getBounds()?.x).toBeCloseTo(500);
    expect(instance.getBounds()?.y).toBeCloseTo(0);
  });

  it("leaves no pan behind when the zoom goes back to 1 in a smaller panel", () => {
    const make = (metrics: { cssWidth: number; cssHeight: number; pixelWidth: number; pixelHeight: number }) => {
      const instance = Object.create(CubismModelInstance.prototype) as CubismModelInstance;
      Object.assign(instance, {
        projection: new CubismMatrix44(),
        _modelMatrix: new CubismMatrix44(),
        _model: { ...squareMesh(), getCanvasWidth: () => 1 },
        packageValue: { manifest: { display: { scale: 1, x: 0.5, y: 0.5 } } },
        setRenderTargetSize: () => {},
        viewport: new Float32Array(4),
        displayMetrics: metrics,
        drawableHitBounds: new Float32Array([0.2, 0.9, 0.1, 0.6]),
        viewOffsetX: 0, viewOffsetY: 0, viewZoom: 1, zoomPanX: 0, zoomPanY: 0,
      });
      instance.resize(metrics as Parameters<CubismModelInstance["resize"]>[0]);
      return instance;
    };
    const large = { cssWidth: 1_440, cssHeight: 2_160, pixelWidth: 1_440, pixelHeight: 2_160 };
    const small = { cssWidth: 96, cssHeight: 144, pixelWidth: 96, pixelHeight: 144 };
    const pet = make(large);
    pet.setViewZoom(2.5);
    pet.resize(small as Parameters<CubismModelInstance["resize"]>[0]);
    pet.setViewZoom(1);
    const fresh = make(small).getBounds()!;
    const bounds = pet.getBounds()!;
    expect(bounds.x).toBeCloseTo(fresh.x);
    expect(bounds.y).toBeCloseTo(fresh.y);
    expect(bounds.height).toBeCloseTo(fresh.height);
  });

  it("keeps the model center steady while zooming and preserves it on viewport rebuild", () => {
    const projection = new CubismMatrix44();
    const instance = Object.create(CubismModelInstance.prototype) as CubismModelInstance;
    const metrics = { cssWidth: 1_000, cssHeight: 800, pixelWidth: 1_000, pixelHeight: 1_000 };
    Object.assign(instance, {
      projection,
      _modelMatrix: new CubismMatrix44(),
      _model: { ...squareMesh(), getCanvasWidth: () => 1 },
      packageValue: { manifest: { display: { scale: 1, x: 0.5, y: 0.5 } } },
      setRenderTargetSize: () => {},
      viewport: new Float32Array(4),
      displayMetrics: metrics,
      drawableHitBounds: new Float32Array([0, 1, 0, 1]),
      viewOffsetX: 0, viewOffsetY: 0, viewZoom: 1, zoomPanX: 0, zoomPanY: 0,
    });
    instance.setViewOffset(50, -100);
    const before = instance.getBounds()!;
    instance.setViewZoom(2);
    const zoomed = instance.getBounds()!;
    expect(zoomed.x + zoomed.width / 2).toBeCloseTo(before.x + before.width / 2);
    expect(zoomed.y + zoomed.height / 2).toBeCloseTo(before.y + before.height / 2);
    expect(zoomed.width).toBeCloseTo(before.width * 2);
    instance.resize(metrics as Parameters<CubismModelInstance["resize"]>[0]);
    expect(instance.getBounds()?.x).toBeCloseTo(zoomed.x);
    expect(instance.getBounds()?.y).toBeCloseTo(zoomed.y);
  });
});

/** One drawable: the unit square as two triangles. */
function squareMesh() {
  return { getDrawableVertices: () => new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), getDrawableVertexIndices: () => new Uint16Array([0, 1, 2, 0, 2, 3]) };
}

describe("the body beside a slanted prop", () => {
  // A narrow body (x 0.4–0.6) and a prop slanting from its hand out to x -1 below the middle.
  const meshes = [
    { vertices: [0.4, 0, 0.6, 0, 0.6, 1, 0.4, 1], indices: [0, 1, 2, 0, 2, 3] },
    { vertices: [0.4, 0.55, -1, 0, -1, 0.05, 0.4, 0.6], indices: [0, 1, 2, 0, 2, 3] },
  ];
  const instance = () => {
    const value = Object.create(CubismModelInstance.prototype) as CubismModelInstance;
    Object.assign(value, {
      projection: new CubismMatrix44(),
      _modelMatrix: new CubismMatrix44(),
      _model: {
        getCanvasHeight: () => 2,
        getDrawableVertices: (i: number) => new Float32Array(meshes[i]!.vertices),
        getDrawableVertexIndices: (i: number) => new Uint16Array(meshes[i]!.indices),
      },
      hitBoundsRefresh: { interactionNeedsRefresh: () => false },
      displayMetrics: { cssWidth: 1_000, cssHeight: 800 },
      drawableHitBounds: new Float32Array([0.4, 0.6, 0, 1, -1, 0.4, 0, 0.6]),
      viewOffsetX: 0, viewOffsetY: 0, viewZoom: 1, zoomPanX: 0, zoomPanY: 0,
    });
    return value;
  };

  it("measures the body's width where the meshes cross the middle band, not their boxes", () => {
    const model = instance();
    model.setViewOffset(10, 0);
    // The prop's box reaches x -1 (screen 10) across the band, but its mesh only reaches x ≈ -0.007 there.
    expect(model.getBounds()!.x).toBeCloseTo(10);
    expect(model.getBounds()!.middle!.y).toBeCloseTo(200);
    expect(model.getBounds()!.middle!.left).toBeCloseTo(506.4, 0);
    expect(model.getBounds()!.middle!.right).toBeCloseTo(810);
  });

  it("lets clicks through the empty corner of the prop's box", () => {
    const model = instance();
    const metrics = { cssWidth: 1_000, cssHeight: 800 } as Parameters<CubismModelInstance["containsDraggablePoint"]>[2];
    // Screen (300, 200) is model (-0.4, 0.5): inside the prop's box, far from the prop itself.
    expect(model.containsDraggablePoint(300, 200, metrics)).toBe(false);
    expect(model.containsDraggablePoint(750, 200, metrics)).toBe(true);
    // Model (-0.5, 0.2) lies on the prop.
    expect(model.containsDraggablePoint(250, 320, metrics)).toBe(true);
  });
});

describe("behavior transition parameter persistence", () => {
  it("returns unkeyed body parameters to their baseline after a motion-to-expression handoff", () => {
    const { instance, model } = transitionInstance();

    instance.update(1 / 60);
    expect(model.values[0]).toBeGreaterThan(0);
    expect(model.values[0]).toBeLessThan(12);
    expect(model.saved).toEqual([0, 0]);
    for (let i = 0; i < 90; i++) instance.update(1 / 60);
    expect(model.values).toEqual([0, 0]);
  });

  it("keeps new motion targets while returning parameters the new motion does not animate", () => {
    const { instance, model } = transitionInstance((values) => { values[1] = 3; });

    for (let i = 0; i < 90; i++) instance.update(1 / 60);

    expect(model.values).toEqual([0, 3]);
    expect(model.saved).toEqual([0, 3]);
  });

  it("captures an interrupted transition's displayed pose without late effects", () => {
    const { instance, model } = transitionInstance(undefined, (values) => { values[0]! += 100; });
    for (let i = 0; i < 30; i++) instance.update(1 / 60);
    expect(model.values[0]).toBeCloseTo(106, 5);

    const source = (instance as unknown as { captureTransitionSourceValues(): Float32Array })
      .captureTransitionSourceValues();
    expect(source[0]).toBeCloseTo(6, 5);
    expect(source[1]).toBeCloseTo(-3, 5);
    expect(model.saved).toEqual([0, 0]);
  });
});

function transitionInstance(
  motion?: (values: number[]) => void,
  lateEffect?: (values: number[]) => void,
) {
  const model = {
    values: [0, 0],
    saved: [0, 0],
    loadParameters() { this.values = [...this.saved]; },
    saveParameters() { this.saved = [...this.values]; },
    getParameterCount() { return this.values.length; },
    getParameterValueByIndex(index: number) { return this.values[index]!; },
    setParameterValueByIndex(index: number, value: number) { this.values[index] = value; },
    update() {},
  };
  // Exercise the real per-frame lifecycle without loading textures or a GL context.
  const instance = Object.create(CubismModelInstance.prototype) as CubismModelInstance;
  Object.assign(instance, {
    _model: model,
    released: false,
    activeBehavior: {
      elapsedSeconds: 0,
      loopMotions: true,
      layers: { updateMotions: () => { motion?.(model.values); return Boolean(motion); } },
    },
    parameterTransition: { from: new Float32Array([12, -6]), elapsedSeconds: 0, durationSeconds: 1 },
    scheduler: { onLateUpdate: () => lateEffect?.(model.values) },
    sampleState: { mode: "asset" },
  });
  return { instance, model };
}

describe("DrawableHitBoundsRefreshCadence", () => {
  it.each([30, 60])(
    "refreshes the first model update and then near ten hertz at %i FPS",
    (framesPerSecond) => {
      const cadence = new DrawableHitBoundsRefreshCadence();
      const frameSeconds = 1 / framesPerSecond;

      expect(cadence.afterModelUpdate(frameSeconds)).toBe(true);
      cadence.markRefreshed();

      const updatesBeforeDeadline = Math.ceil(
        DRAWABLE_HIT_BOUNDS_REFRESH_SECONDS / frameSeconds,
      ) - 1;
      for (let index = 0; index < updatesBeforeDeadline; index += 1) {
        expect(cadence.afterModelUpdate(frameSeconds)).toBe(false);
      }
      expect(cadence.afterModelUpdate(frameSeconds)).toBe(true);
    },
  );

  it("requires one fresh scan for the latest model revision before interaction", () => {
    const cadence = new DrawableHitBoundsRefreshCadence();

    expect(cadence.afterModelUpdate(1 / 60)).toBe(true);
    cadence.markRefreshed();
    expect(cadence.interactionNeedsRefresh()).toBe(false);

    expect(cadence.afterModelUpdate(1 / 60)).toBe(false);
    expect(cadence.interactionNeedsRefresh()).toBe(true);
    cadence.markRefreshed();

    expect(cadence.interactionNeedsRefresh()).toBe(false);
    expect(cadence.interactionNeedsRefresh()).toBe(false);
  });
});

describe("isolated facial demo composition", () => {
  it("continues the same motion and sway clocks through speech and fade-out, with one physics step per frame", () => {
    const { instance, model, physics } = facialDemoInstance();
    const scheduler = new CubismUpdateScheduler();
    let phase = 0;
    class Sway extends ICubismUpdater {
      constructor() { super(500); }
      onLateUpdate(_model: unknown, dt: number) { phase += dt; model.values[0] = Math.sin(phase * 2) * 3; }
    }
    class Physics extends ICubismUpdater {
      constructor() { super(600); }
      onLateUpdate() { physics.evaluate(); }
    }
    physics.evaluate.mockImplementation(() => { model.values[4] = model.values[0]! * 0.5; });
    const motion = { elapsedSeconds: 4, loopMotions: true, finishAfterOneCycle: false,
      layers: { updateMotions: () => { model.values[0] = 0; return true; }, hasMotions: () => true } };
    Object.assign(instance, { scheduler, activeBehavior: motion });
    scheduler.addUpdatableList(new Sway()); scheduler.addUpdatableList(new Physics());
    for (let i = 0; i < 60; i++) instance.update(1 / 60);
    const before = [...model.values];
    instance.beginFacialOverlay({ mouth: null, eyes: [], gazeParameterIds: [], neutralValues: { ParamAngleX: 0 } }, true);
    expect(model.values).toEqual(before);
    expect((instance as any).stopBehavior).not.toHaveBeenCalled();
    // Spoken parts vary a little each sentence; this one lands exactly on its step.
    const random = vi.spyOn(Math, "random").mockReturnValue(0.5);
    instance.setFacialDemoTargets({ ParamAngleX: 4 });
    for (let i = 0; i < 180; i++) instance.update(1 / 60);
    random.mockRestore();
    expect(model.values[0]).toBeCloseTo(Math.sin(phase * 2) * 3 + 4, 5);
    expect(model.values[4]).toBeCloseTo(model.values[0]! * 0.5, 5);
    expect(model.saved[0]).toBe(0); // No speech offsets persisted into the motion baseline.
    const speaking = [...model.values]; instance.endFacialOverlay();
    expect(model.values).toEqual(speaking);
    for (let i = 0; i < 60; i++) instance.update(1 / 60);
    expect(instance.hasFacialOverlay()).toBe(false);
    expect((instance as any).activeBehavior).toBe(motion);
    expect(motion.elapsedSeconds).toBeCloseTo(9);
    expect(model.values[0]).toBeCloseTo(Math.sin(phase * 2) * 3);
    expect(physics.evaluate).toHaveBeenCalledTimes(300);
    expect(physics.stabilization).not.toHaveBeenCalled();
  });
  it("retains physics input channels while excluding output-only channels", () => {
    const json = { PhysicsSettings: [
      { Input: [{ Source: { Target: "Parameter", Id: "ParamAngleX" } }], Output: [{ Destination: { Target: "Parameter", Id: "ParamBodyAngleX" } }] },
      { Output: [{ Destination: { Target: "Parameter", Id: "ParamAngleX" } }] },
    ] };
    expect([...facialDemoPhysicsOutputOnlyIds(new TextEncoder().encode(JSON.stringify(json)).buffer)]).toEqual(["ParamBodyAngleX"]);
  });

  it("feeds facial inputs into physics, preserves its outputs and the configured clothing, and bypasses normal drivers", () => {
    const { instance, model, physics, scheduler, updateLip } = facialDemoInstance();
    instance.beginFacialDemo();
    instance.setFacialDemoTargets({ ParamAngleX: 20, ParamMouthForm: 0.7, ParamCheek: 1 });
    instance.setFacialDemoSpeechLevel(0.8);
    instance.update(0.1);
    const snapshot = instance.facialDemoSnapshot();
    const input = physics.inputs[0]!;
    expect(input.ParamAngleX).toBeGreaterThan(0);
    expect(input.ParamMouthForm).toBeGreaterThan(0.1);
    expect(input.ParamMouthOpenY).toBeGreaterThan(0);
    expect(snapshot.values.ParamAngleX).toBe(input.ParamAngleX);
    expect(snapshot.values.ParamMouthForm).toBe(input.ParamMouthForm);
    expect(snapshot.values.AuthoredHeadOutput).toBe(input.ParamAngleX! * 0.5);
    expect(snapshot.values.ClothingSwitch).toBe(1);
    expect(snapshot.targets).not.toHaveProperty("ParamCheek");
    expect(scheduler.onLateUpdate).not.toHaveBeenCalled();
    expect(updateLip).not.toHaveBeenCalled();
    expect(model.values).toHaveLength(model.ids.length);
  });

  it("multiplies natural blinking by the expression eye opening before physics", () => {
    const { instance, physics } = facialDemoInstance();
    instance.beginFacialDemo();
    const internals = instance as unknown as { facialDemo: { blink: { updateParameters: (model: unknown, delta: number) => void } } };
    internals.facialDemo.blink.updateParameters = (value) => {
      const model = value as { values: number[] };
      model.values[1] = 0.25;
    };
    instance.setFacialDemoTargets({ ParamEyeLOpen: 0.4 });
    for (let frame = 0; frame < 20; frame++) instance.update(0.1);
    expect(instance.facialDemoSnapshot().values.ParamEyeLOpen).toBeCloseTo(0.1, 5);
    expect(physics.inputs.at(-1)!.ParamEyeLOpen).toBeCloseTo(0.1, 5);
  });

  it("ends in the editing baseline without retaining facial or speech inputs", () => {
    const { instance, model } = facialDemoInstance();
    instance.beginFacialDemo();
    instance.setFacialDemoTargets({ ParamAngleX: 20, ParamMouthForm: 0.7 });
    instance.setFacialDemoSpeechLevel(0.8);
    instance.update(0.1);
    instance.endFacialDemo();
    expect(instance.facialDemoSnapshot()).toEqual({ active: false, values: {}, targets: {} });
    expect(model.values).toEqual([0, 1, 0, 0.1, 0, 1]);
    expect(model.saved).toEqual(model.values);
  });

  it("applies the SDK blink to a manually bound reversed eyelid without multiplying raw native values", () => {
    const { instance, model, physics } = facialDemoInstance();
    model.ids[1] = "CustomEye";
    Object.assign(instance, { parameterIds: new Map([["CustomEye", { getString: () => "CustomEye" }]]) });
    instance.beginFacialDemo({ mouth: null, eyes: [{ id: "CustomEye", closed: 1 }] });
    const internals = instance as unknown as { facialDemo: { blink: { updateParameters: (model: { setParameterValueById(id: unknown, value: number): void }, delta: number) => void } } };
    internals.facialDemo.blink.updateParameters = value => value.setParameterValueById({ getString: () => "CustomEye" }, 0.25);
    instance.setFacialDemoTargets({ CustomEye: 0.2 });
    for (let frame = 0; frame < 20; frame++) instance.update(0.1);
    expect(instance.facialDemoSnapshot().values.CustomEye).toBeCloseTo(0.8, 5);
    expect(physics.inputs.at(-1)!.CustomEye).toBeCloseTo(0.8, 5);
    expect(model.values).toHaveLength(model.ids.length);
  });

  it("runs the real breath channel through physics without accepting JEV breath or output targets", () => {
    const { instance, model, physics } = facialDemoInstance();
    model.ids.push("ParamBreath", "ParamHairFront");
    model.defaults.push(0, 0); model.values.push(0, 0); model.saved.push(0, 0);
    Object.assign(instance, { facialDemoPhysicsOutputs: new Set(["AuthoredHeadOutput"]) });
    instance.beginFacialDemo();
    instance.setFacialDemoTargets({ ParamBreath: 1, ParamHairFront: 1, AuthoredHeadOutput: 25 });
    expect(instance.facialDemoSnapshot().targets).toEqual({});
    for (let frame = 0; frame < 16; frame++) instance.update(0.1);
    expect(physics.inputs.at(-1)!.ParamBreath).toBeCloseTo(1);
    expect(instance.facialDemoSnapshot().values.ParamBreath).toBeCloseTo(1);
    expect(instance.facialDemoSnapshot().values.ParamHairFront).toBe(0);
    for (let frame = 0; frame < 16; frame++) instance.update(0.1);
    expect(instance.facialDemoSnapshot().values.ParamBreath).toBeCloseTo(0);
    expect(physics.inputs.every(input => input.ParamAngleX === 0)).toBe(true);
  });

  it("leaves an explicitly configured breath appearance fixed", () => {
    const { instance, model, physics } = facialDemoInstance();
    model.ids.push("ParamBreath");
    model.defaults.push(0); model.values.push(0); model.saved.push(0);
    Object.assign(instance, { defaultAppearance: {
      parameters: () => new Set(["ParamBreath"]),
      apply: () => { model.values[5] = 1; model.values[6] = 0.4; },
    } });
    instance.beginFacialDemo();
    for (let frame = 0; frame < 16; frame++) instance.update(0.1);
    expect(physics.inputs.every(input => input.ParamBreath === 0.4)).toBe(true);
    expect(instance.facialDemoSnapshot().values.ParamBreath).toBe(0.4);
  });

  it("keeps TTS and automatic blinking from claiming physics-only mouth and eyelid outputs", () => {
    const { instance, model, physics } = facialDemoInstance();
    Object.assign(instance, { facialDemoPhysicsOutputs: new Set(["ParamMouthOpenY", "ParamEyeLOpen"]) });
    physics.evaluate.mockImplementation(() => {
      model.values[1] = 0.3;
      model.values[2] = 0.2;
    });
    instance.beginFacialDemo();
    instance.setFacialDemoTargets({ ParamMouthOpenY: 1, ParamEyeLOpen: 1 });
    instance.setFacialDemoSpeechLevel(1);
    for (let frame = 0; frame < 20; frame++) instance.update(0.1);
    const snapshot = instance.facialDemoSnapshot();
    expect(snapshot.targets).not.toHaveProperty("ParamMouthOpenY");
    expect(snapshot.targets).not.toHaveProperty("ParamEyeLOpen");
    expect(snapshot.values.ParamMouthOpenY).toBe(0.2);
    expect(snapshot.values.ParamEyeLOpen).toBe(0.3);
    const demo = (instance as unknown as { facialDemo: { eyeIndices: number[] } }).facialDemo;
    expect(demo.eyeIndices).toEqual([]);
  });
});

function facialDemoInstance() {
  const model = {
    ids: ["ParamAngleX", "ParamEyeLOpen", "ParamMouthOpenY", "ParamMouthForm", "AuthoredHeadOutput", "ClothingSwitch"],
    values: [20, 0.2, 0.5, 0.8, 10, 0],
    saved: [20, 0.2, 0.5, 0.8, 10, 0],
    defaults: [0, 1, 0, 0, 0, 0],
    loadParameters() { this.values = [...this.saved]; },
    saveParameters() { this.saved = [...this.values]; },
    getModel() { return { parameters: { ids: this.ids } }; },
    getParameterCount() { return this.values.length; },
    getParameterValueByIndex(index: number) { return this.values[index]!; },
    getParameterDefaultValue(index: number) { return this.defaults[index]!; },
    getParameterMinimumValue(index: number) { return index === 0 || index === 4 ? -30 : 0; },
    getParameterMaximumValue(index: number) { return index === 0 || index === 4 ? 30 : 1; },
    setParameterValueByIndex(index: number, value: number) { this.values[index] = value; },
    setParameterValueById(id: { getString(): string }, value: number) {
      const index = this.ids.indexOf(id.getString());
      if (index < 0) throw new Error("Virtual parameters must never be created");
      this.values[index] = value;
    },
    update: vi.fn(),
  };
  const physics = {
    inputs: [] as Array<Record<string, number>>,
    stabilization: vi.fn(() => { model.values[4] = model.values[0]! * 0.5; }),
    evaluate: vi.fn(() => {
      physics.inputs.push(Object.fromEntries(model.ids.map((id, index) => [id, model.values[index]!])));
      model.values[4] = model.values[0]! * 0.5;
      model.values[0] = -30;
      model.values[3] = 0;
      model.values[5] = 0;
    }),
  };
  const scheduler = { onLateUpdate: vi.fn() };
  const updateLip = vi.fn();
  const instance = Object.create(CubismModelInstance.prototype) as CubismModelInstance;
  Object.assign(instance, {
    _model: model,
    _physics: physics,
    released: false,
    facialDemo: null,
    packageValue: { capabilities: {} },
    eyeBlinkIds: [{ getString: () => "ParamEyeLOpen" }, { getString: () => "MissingEye" }],
    _motionManager: { stopAllMotions: vi.fn() },
    _expressionManager: { stopAllMotions: vi.fn() },
    stopBehavior: vi.fn(),
    stopPersistentExpressions: vi.fn(),
    defaultAppearance: { parameters: () => new Set(), apply: () => { model.values[5] = 1; model.values[3] = 0.1; } },
    updateDrawableHitBounds: vi.fn(),
    hitBoundsRefresh: { afterModelUpdate: () => false },
    scheduler,
    updateLip,
    lipSyncValues: new Map(),
    lipTarget: { volume: 0, a: 0, e: 0, i: 0, o: 0, u: 0 },
  });
  return { instance, model, physics, scheduler, updateLip };
}
