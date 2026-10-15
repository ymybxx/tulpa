import { describe, expect, it } from "vitest";
import { buildBindings, decisionParameter, fromNativeTargets, motionBindings, neutralTargets, overlayBindings, DecisionSides, DECISION_REST_MS, planDecisions, STANDARD_PARAMETERS, toNativeTargets } from "./parameter-bindings";
import type { NativeModelParameter } from "../shared/contracts";

const parameter = (id: string, minimum = -1, maximum = 1, defaultValue = 0): NativeModelParameter => ({ id, name: id, minimum, maximum, defaultValue });

describe("standard parameter bindings", () => {
  it("covers the standard parts once with complete scoring metadata and automatic owners", () => {
    expect(STANDARD_PARAMETERS).toHaveLength(29);
    expect(new Set(STANDARD_PARAMETERS.map(value => value.id)).size).toBe(29);
    // The VTube Studio face inputs, with brow angle and blush added for emotion.
    expect(STANDARD_PARAMETERS.filter(value => value.owner === "jev").map(value => value.id)).toEqual([
      "ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamEyeLOpen", "ParamEyeROpen", "ParamEyeBallX", "ParamEyeBallY",
      "ParamBrowLY", "ParamBrowRY", "ParamBrowLAngle", "ParamBrowRAngle", "ParamMouthForm", "ParamCheek",
      "ParamMouthX", "ParamMouthShrug", "ParamMouthPressLipOpen", "ParamCheekPuffC", "ParamTongueOut",
    ]);
    expect(STANDARD_PARAMETERS.filter(value => value.owner === "physics").map(value => value.id)).toEqual([
      "ParamHairFront", "ParamHairSide", "ParamHairBack", "ParamHairFluffy", "ParamBustX", "ParamBustY",
    ]);
    expect(STANDARD_PARAMETERS.filter(value => value.owner === "breath").map(value => value.id)).toEqual(["ParamBreath"]);
    // How far the mouth opens, and the shapes lip sync drives, follow the speech, never the decision model.
    expect(STANDARD_PARAMETERS.filter(value => value.owner === "speech").map(value => value.id)).toEqual([
      "ParamMouthOpenY", "ParamJawOpen", "ParamMouthPuckerWiden", "ParamMouthFunnel",
    ]);
    for (const definition of STANDARD_PARAMETERS) {
      expect(definition.anchors).toHaveLength(definition.levels.length);
      expect(definition.anchors[0]).toBe(definition.min);
      expect(definition.anchors.at(-1)).toBe(definition.max);
      expect(definition.neutral).toBeGreaterThanOrEqual(definition.min);
      expect(definition.neutral).toBeLessThanOrEqual(definition.max);
    }
  });

  it("keeps missing controls visible but never admits virtual parameter IDs", () => {
    const bindings = buildBindings([parameter("ParamMouthForm"), parameter("Unrelated")]);
    expect(bindings.all).toHaveLength(29);
    expect(bindings.active.map(value => value.id)).toEqual(["ParamMouthForm"]);
    expect(bindings.all.find(value => value.id === "ParamCheek")).toMatchObject({ supported: false, actualId: null });
    expect(toNativeTargets({ ParamCheek: 1, Unrelated: 1, ParamMouthForm: 0.5 }, bindings.all)).toEqual({ ParamMouthForm: 0.5 });
  });

  it("uses the verified cheek alias only when the native standard parameter is absent", () => {
    const alias = buildBindings([parameter("Param111", 0, 0.8)], "Param111");
    expect(toNativeTargets({ ParamCheek: 1 }, alias.active)).toEqual({ Param111: 0.8 });
    const native = buildBindings([parameter("ParamCheek", 0, 1), parameter("Param111", 0, 1)], "Param111");
    expect(toNativeTargets({ ParamCheek: 0.4 }, native.active)).toEqual({ ParamCheek: 0.4 });
    expect(buildBindings([], "Param111").active).toEqual([]);
  });

  it("clamps each left and right control to its own authored range and retains the native default", () => {
    const bindings = buildBindings([
      parameter("ParamEyeLOpen", 0, 0.8, 0.7), parameter("ParamEyeROpen", 0.2, 1.5, 1),
      parameter("ParamCheek", 0, 1, 0), parameter("ParamMouthForm", -0.5, 0.7, 0.1),
    ]);
    expect(neutralTargets(bindings.active)).toEqual({ ParamEyeLOpen: 0.7, ParamEyeROpen: 1, ParamMouthForm: 0.1, ParamCheek: 0 });
    expect(toNativeTargets({ ParamEyeLOpen: 0, ParamEyeROpen: 0, ParamCheek: 0.4, ParamMouthForm: -9 }, bindings.active)).toEqual({ ParamEyeLOpen: 0, ParamEyeROpen: 0.2, ParamMouthForm: -0.5, ParamCheek: 0.4 });
    expect(toNativeTargets({ ParamEyeLOpen: 2, ParamEyeROpen: 2 }, bindings.active)).toEqual({ ParamEyeLOpen: 0.8, ParamEyeROpen: 1.5 });
  });

  it("prevents JEV from taking physics outputs, hair, bust or breathing channels", () => {
    const bindings = buildBindings([
      parameter("ParamAngleX", -30, 30), parameter("ParamAngleZ", -30, 30),
      parameter("ParamHairFront"), parameter("ParamBustY"), parameter("ParamBreath", 0, 1),
    ], null, ["ParamAngleZ"]);
    expect(bindings.active.map(value => value.id)).toEqual(["ParamAngleX"]);
    expect(bindings.all.find(value => value.id === "ParamAngleZ")).toMatchObject({ supported: true, owner: "physics", controllable: false });
    expect(toNativeTargets({ ParamAngleX: 12, ParamAngleZ: 10, ParamHairFront: 1, ParamBustY: 1, ParamBreath: 1 }, bindings.all)).toEqual({ ParamAngleX: 12 });
  });

  it("does not send fixed or incompatible native ranges to JEV", () => {
    const bindings = buildBindings([parameter("ParamEyeLOpen", 0.5, 0.5, 0.5), parameter("ParamMouthForm", 3, 4, 3)]);
    expect(bindings.active).toEqual([]);
    expect(bindings.all.find(value => value.id === "ParamEyeLOpen")).toMatchObject({ supported: true, controllable: false });
    expect(bindings.all.find(value => value.id === "ParamMouthForm")).toMatchObject({ supported: false });
  });

  it("calibrates a missing standard control to an asymmetric reversed native range and back", () => {
    const bindings = buildBindings([parameter("CustomBrow", 20, 80, 60)], null, [], {
      ParamBrowLY: [{ parameterId: "CustomBrow", lowerValue: 80, neutralValue: 60, upperValue: 20 }],
    });
    expect(bindings.active.map(value => value.id)).toEqual(["ParamBrowLY"]);
    expect(neutralTargets(bindings.active)).toEqual({ ParamBrowLY: 0 });
    expect(toNativeTargets({ ParamBrowLY: -0.5 }, bindings.active)).toEqual({ CustomBrow: 70 });
    expect(toNativeTargets({ ParamBrowLY: 0.5 }, bindings.active)).toEqual({ CustomBrow: 40 });
    expect(toNativeTargets({ ParamBrowLY: 9 }, bindings.active)).toEqual({ CustomBrow: 20 });
    expect(fromNativeTargets({ CustomBrow: 70 }, bindings.active)).toEqual({ ParamBrowLY: -0.5 });
    expect(fromNativeTargets({ CustomBrow: 40 }, bindings.active)).toEqual({ ParamBrowLY: 0.5 });
  });

  it("drives every parameter a part lists, each through its own curve, and skips one that is gone", () => {
    const bindings = buildBindings([parameter("Smile", -1, 1, 0), parameter("Squint", 0, 1, 0), parameter("EyeX", -1, 1, 0), parameter("EyeX2", -1, 1, 0)], null, [], {
      ParamMouthForm: [
        { parameterId: "Smile", lowerValue: -1, neutralValue: 0, upperValue: 1 },
        { parameterId: "Squint", lowerValue: 0, neutralValue: 0, upperValue: 1 },
        { parameterId: "Missing", lowerValue: -1, neutralValue: 0, upperValue: 1 },
      ],
      ParamEyeBallX: [
        { parameterId: "EyeX", lowerValue: -1, neutralValue: 0, upperValue: 1 },
        { parameterId: "EyeX2", lowerValue: 1, neutralValue: 0, upperValue: -1 },
      ],
    });
    const smile = bindings.active.find(value => value.id === "ParamMouthForm")!;
    expect(smile.outputs.map(output => output.parameterId)).toEqual(["Smile", "Squint"]);
    expect(smile).toMatchObject({ actualId: "Smile", controllable: true });
    expect(toNativeTargets({ ParamMouthForm: 0.5, ParamEyeBallX: 0.4 }, bindings.active)).toEqual({ Smile: 0.5, Squint: 0.5, EyeX: 0.4, EyeX2: -0.4 });
    expect(overlayBindings(bindings.active).gazeParameterIds).toEqual(["EyeX", "EyeX2"]);
    expect(motionBindings(bindings.active).transitions).toMatchObject({ Smile: { group: "mouth" }, Squint: { group: "mouth" } });
    const gone = buildBindings([parameter("Other")], null, [], { ParamMouthForm: [{ parameterId: "Missing", lowerValue: -1, neutralValue: 0, upperValue: 1 }] });
    expect(gone.all.find(value => value.id === "ParamMouthForm")).toMatchObject({ controllable: false, outputs: [] });
  });

  it("follows the model's own VTS wiring before names, and a choice made here before both", () => {
    const settings = [{ input: "MouthSmile", inputLower: 0, inputUpper: 1, outputLower: -1, outputUpper: 1,
      clampInput: false, clampOutput: false, parameterId: "Smile" }];
    const native = [parameter("ParamMouthForm"), parameter("Smile"), parameter("Other")];
    const vts = buildBindings(native, null, [], {}, {}, {}, [], {}, settings).all.find(value => value.id === "ParamMouthForm")!;
    expect(vts).toMatchObject({ fromVts: true, controllable: true, actualId: "Smile", reason: "按模型自带的 VTube Studio 设置" });
    expect(vts.manualBindings).toBeUndefined();
    expect(toNativeTargets({ ParamMouthForm: 1 }, [vts])).toEqual({ Smile: 1 });
    const chosen = buildBindings(native, null, [], { ParamMouthForm: [{ parameterId: "Other", lowerValue: -1, neutralValue: 0, upperValue: 1 }] },
      {}, {}, [], {}, settings).all.find(value => value.id === "ParamMouthForm")!;
    expect(chosen.outputs.map(output => output.parameterId)).toEqual(["Other"]);
    expect(buildBindings(native).all.find(value => value.id === "ParamMouthForm")!.actualId).toBe("ParamMouthForm");
  });

  it("combines shared left/right inputs and keeps saturation finite", () => {
    const eye = { parameterId: "Eyes", lowerValue: 0, neutralValue: 1, upperValue: 1 };
    const bindings = buildBindings([parameter("Eyes", 0, 1, 1)], null, [], { ParamEyeLOpen: [eye], ParamEyeROpen: [eye] });
    expect(toNativeTargets({ ParamEyeLOpen: 0.2, ParamEyeROpen: 0.6 }, bindings.active)).toEqual({ Eyes: 0.4 });
    expect(toNativeTargets({ ParamEyeLOpen: 2, ParamEyeROpen: 2 }, bindings.active)).toEqual({ Eyes: 1 });
    expect(fromNativeTargets({ Eyes: 1 }, bindings.active)).toEqual({ ParamEyeLOpen: 1, ParamEyeROpen: 1 });
    expect(motionBindings(bindings.active)).toMatchObject({ mouth: null, eyes: [{ id: "Eyes", closed: 0 }] });
  });

  it("tells the voice-led motion how far each native parameter moves per degree of head pitch or unit of gaze", () => {
    const native = [parameter("ParamAngleY", -30, 30), parameter("HeadY", 0, 100, 50), parameter("ParamEyeBallX"), parameter("ParamMouthForm")];
    const own = motionBindings(buildBindings(native).active).transitions!;
    expect(own.ParamAngleY?.follow).toEqual({ part: "headPitch", up: 1, down: 1 });
    expect(own.ParamEyeBallX?.follow).toEqual({ part: "eyeX", up: 1, down: 1 });
    expect(own.ParamMouthForm?.follow).toBeUndefined();
    // An author's wiring that reaches further one way than the other is followed each way.
    const manual = { parameterId: "HeadY", lowerValue: 0, neutralValue: 50, upperValue: 80 };
    const mapped = motionBindings(buildBindings(native, null, [], { ParamAngleY: [manual] }).active).transitions!;
    expect(mapped.HeadY?.follow?.up).toBeCloseTo(30 / 30);
    expect(mapped.HeadY?.follow?.down).toBeCloseTo(50 / 30);
  });

  it("uses explicit overrides without changing automatic ownership or admitting stale parameters", () => {
    const manual = { parameterId: "Custom", lowerValue: -1, neutralValue: 0, upperValue: 1 };
    const native = [parameter("Custom"), parameter("ParamMouthForm")];
    expect(toNativeTargets({ ParamMouthForm: 0.5 }, buildBindings(native, null, [], { ParamMouthForm: [manual] }).active)).toEqual({ Custom: 0.5 });
    expect(toNativeTargets({ ParamMouthForm: 0.5 }, buildBindings(native).active)).toEqual({ ParamMouthForm: 0.5 });
    for (const mapping of [
      { ...manual, parameterId: "Missing" }, { ...manual, upperValue: 10 },
      { ...manual, parameterId: "ParamBreath" },
    ]) {
      expect(buildBindings([...native, parameter("ParamBreath")], null, [], { ParamMouthForm: [mapping] }).active).toEqual([]);
    }
    expect(buildBindings(native, null, ["Custom"], { ParamMouthForm: [manual] }).active).toEqual([]);
    expect(buildBindings(native, null, [], { ParamHairFront: [manual] }).active.map(value => value.id)).toEqual(["ParamMouthForm"]);
  });

  it("exports calibrated eye animation values only to the local renderer; the mouth's opening is speech's", () => {
    const bindings = buildBindings([parameter("Open", -10, 10), parameter("Eye", 20, 80)], null, [], {
      ParamMouthOpenY: [{ parameterId: "Open", lowerValue: 10, neutralValue: 10, upperValue: -10 }],
      ParamEyeLOpen: [{ parameterId: "Eye", lowerValue: 80, neutralValue: 30, upperValue: 20 }],
    });
    expect(bindings.active.map(value => value.id)).toEqual(["ParamEyeLOpen"]);
    expect(motionBindings(bindings.active)).toMatchObject({ mouth: null, eyes: [{ id: "Eye", closed: 80 }],
      transitions: { Eye: { group: "eyes", minimum: 20, maximum: 80 } } });
  });

  it("excludes disabled generation while retaining local speech and the original automatic owners", () => {
    const resolved = buildBindings([parameter("ParamEyeLOpen", 0, 2, 1), parameter("ParamCheek", 0, 1), parameter("ParamMouthOpenY", 0, 1), parameter("ParamBreath", 0, 1)], null, [], {}, {
      ParamCheek: { enabled: false, multiplier: 3 },
    });
    expect(resolved.active.map(value => value.id)).toEqual(["ParamEyeLOpen"]);
    expect(toNativeTargets({ ParamEyeLOpen: 1.2, ParamCheek: 1, ParamMouthOpenY: 1 }, resolved.all)).toEqual({ ParamEyeLOpen: 1.2 });
    expect(overlayBindings(resolved.all)).toMatchObject({ neutralValues: { ParamEyeLOpen: 1 }, mouth: null });
    expect(overlayBindings(resolved.all).neutralValues).not.toHaveProperty("ParamCheek");
    expect(resolved.all.find(value => value.id === "ParamMouthOpenY")).toMatchObject({ owner: "speech", controllable: false });
    expect(resolved.all.find(value => value.id === "ParamBreath")).toMatchObject({ owner: "breath", controllable: false });
  });

  it("keeps a switched-off head pitch for the voice's intonation only", () => {
    const resolved = buildBindings([parameter("ParamAngleY", -30, 30), parameter("ParamBrowLY")], null, [], {}, {
      ParamAngleY: { enabled: false, multiplier: 1 }, ParamBrowLY: { enabled: false, multiplier: 1 },
    });
    expect(resolved.active).toEqual([]);
    expect(toNativeTargets({ ParamAngleY: -20 }, resolved.all)).toEqual({});
    const overlay = overlayBindings(resolved.all);
    expect(overlay.neutralValues).toEqual({ ParamAngleY: 0 });
    expect(overlay.transitions?.ParamAngleY?.follow).toEqual({ part: "headPitch", up: 1, down: 1 });
    expect(overlay.transitions).not.toHaveProperty("ParamBrowLY");
  });

  it("scales deviations from neutral before native calibration and clamps without moving neutral", () => {
    const resolved = buildBindings([parameter("ParamEyeLOpen", 0, 2, 1), parameter("CustomBrow", 20, 80, 60)], null, [], {
      ParamBrowLY: [{ parameterId: "CustomBrow", lowerValue: 80, neutralValue: 60, upperValue: 20 }],
    }, { ParamEyeLOpen: { enabled: true, multiplier: 2 }, ParamBrowLY: { enabled: true, multiplier: 2 } });
    expect(toNativeTargets({ ParamEyeLOpen: .75, ParamBrowLY: .25 }, resolved.all)).toEqual({ ParamEyeLOpen: .5, CustomBrow: 40 });
    expect(toNativeTargets({ ParamEyeLOpen: 1.8, ParamBrowLY: 1 }, resolved.all)).toEqual({ ParamEyeLOpen: 2, CustomBrow: 20 });
    expect(toNativeTargets({ ParamEyeLOpen: 1, ParamBrowLY: 0 }, resolved.all)).toEqual({ ParamEyeLOpen: 1, CustomBrow: 60 });
    expect(motionBindings(resolved.all).eyes).toEqual([{ id: "ParamEyeLOpen", closed: 0 }]);
    expect(toNativeTargets({ ParamEyeLOpen: .2 }, resolved.all.map(value => ({ ...value, multiplier: 0 })))).toEqual({ ParamEyeLOpen: 1 });
  });

  it("sends editable standard descriptions and a real custom native parameter to JEV", () => {
    const custom = { id: "CustomEar", label: "耳朵", description: "耳朵竖起程度",
      levels: ["耳朵垂下", "耳朵自然", "耳朵竖起"] as [string, string, string], enabled: true, multiplier: 2 };
    const resolved = buildBindings([
      parameter("ParamAngleY", -30, 30), parameter("CustomEar", -1, 1),
    ], null, [], {}, {}, { ParamAngleY: "头部俯仰，由当前句决定。" }, [custom]);
    expect(resolved.active.map(value => value.id)).toEqual(["ParamAngleY", "CustomEar"]);
    expect(decisionParameter(resolved.active[0]!)).toMatchObject({ description: "头部俯仰，由当前句决定。" });
    expect(decisionParameter(resolved.active[1]!)).toMatchObject({ id: "CustomEar", description: custom.description,
      levels: custom.levels, minimum: -1, maximum: 1 });
    expect(toNativeTargets({ CustomEar: 0.5 }, resolved.active)).toEqual({ CustomEar: 1 });
    expect(buildBindings([parameter("CustomEar")], null, ["CustomEar"], {}, {}, {}, [custom]).active).toEqual([]);
    expect(buildBindings([parameter("CustomEar")], null, [], {}, {}, {}, [{ ...custom, enabled: false }]).active).toEqual([]);
  });

  it("uses configurable score gradients or binary endpoints on the same real parameter", () => {
    const native = [parameter("ParamCheek", 0, 1, 0), parameter("CustomEar", -1, 1, 0)];
    const score = buildBindings(native, null, [], {}, {}, {}, [], {
      ParamCheek: { mode: "score", levels: ["无", "浅", "较深", "很深"] },
    }).active[0]!;
    expect(decisionParameter(score)).toMatchObject({ mode: "score", levels: ["无", "浅", "较深", "很深"],
      anchors: [0, 1 / 3, 2 / 3, 1] });
    const selected = buildBindings(native, null, [], {}, {}, {}, [{ id: "CustomEar", label: "耳朵", description: "耳朵是否竖起",
      levels: ["垂", "普通", "竖"], mode: "binary", falseDescription: "不竖起", trueDescription: "竖起",
      enabled: true, multiplier: 1 }], {
      ParamCheek: { mode: "binary", falseDescription: "没有红晕", trueDescription: "显示红晕" },
    }).active;
    expect(selected.map(decisionParameter)).toEqual([
      expect.objectContaining({ id: "ParamCheek", mode: "binary", falseDescription: "没有红晕", trueDescription: "显示红晕" }),
      expect.objectContaining({ id: "CustomEar", mode: "binary", falseDescription: "不竖起", trueDescription: "竖起" }),
    ]);
    expect(toNativeTargets({ ParamCheek: 1, CustomEar: -1 }, selected)).toEqual({ ParamCheek: 1, CustomEar: -1 });
  });
});

describe("removed parts", () => {
  it("keeps a deleted part out of Jev's hands until it is added back", async () => {
    const { validateParameterSettings } = await import("../shared/facial-bindings");
    const settings = validateParameterSettings({ ParamCheek: { enabled: true, multiplier: 1.2, removed: true } });
    expect(settings.ParamCheek).toEqual({ enabled: false, multiplier: 1.2, removed: true });
    const bindings = buildBindings([parameter("ParamCheek", 0, 1)], null, [], {}, settings);
    const cheek = bindings.all.find(binding => binding.id === "ParamCheek")!;
    expect(cheek.removed).toBe(true);
    expect(bindings.active.some(binding => binding.id === "ParamCheek")).toBe(false);
    expect(() => validateParameterSettings({ ParamCheek: { enabled: true, multiplier: 1, hidden: true } })).toThrow();
  });
});

describe("the questions Jev answers", () => {
  const model = [
    parameter("ParamEyeLOpen", 0, 1.2, 1), parameter("ParamEyeROpen", 0, 1.2, 1),
    parameter("ParamBrowLY"), parameter("ParamAngleX", -30, 30), parameter("ParamEyeBallX"),
    parameter("ParamArmLA", -10, 10), parameter("ParamArmRA", -10, 10),
  ];

  it("asks mirrored face parts once and gives both sides the answer", () => {
    const { active } = buildBindings(model);
    const plan = planDecisions(active);
    const eyes = plan.parameters.find(value => value.id === "ParamEyeOpen")!;
    expect(eyes.description).toContain("双眼");
    expect(eyes.levels!.every(level => !level.includes("左眼"))).toBe(true);
    // One brow alone and each arm stay their own question.
    expect(plan.parameters.map(value => value.id)).toEqual(
      ["ParamAngleX", "ParamEyeOpen", "ParamEyeBallX", "ParamBrowLY", "ParamArmLA", "ParamArmRA"]
        .filter(id => plan.parameters.some(value => value.id === id)));
    expect(plan.parameters.some(value => value.id === "ParamEyeLOpen" || value.id === "ParamEyeROpen")).toBe(false);
    const answers = Object.fromEntries(plan.parameters.map(value => [value.id, value.defaultValue]));
    const values = plan.expand({ ...answers, ParamEyeOpen: 0.3 })!;
    expect(values.ParamEyeLOpen).toBe(0.3);
    expect(values.ParamEyeROpen).toBe(0.3);
    expect(Object.keys(values).sort()).toEqual(active.map(binding => binding.id).sort());
  });

  it("asks a reworded side as written, so the pair splits again", () => {
    const { active } = buildBindings(model, undefined, undefined, undefined, undefined, { ParamEyeLOpen: "只在眨眼时闭上左眼" });
    const plan = planDecisions(active);
    expect(plan.parameters.map(value => value.id)).toEqual(expect.arrayContaining(["ParamEyeLOpen", "ParamEyeROpen"]));
    expect(plan.parameters.some(value => value.id === "ParamEyeOpen")).toBe(false);
  });

  it("asks how far to turn and picks one side per axis, afresh after a rest", () => {
    const { active } = buildBindings(model);
    const draws = [0.9, 0.1];
    const plan = planDecisions(active, new DecisionSides(() => draws.shift()!));
    const head = plan.parameters.find(value => value.id === "ParamAngleX")!;
    expect(head).toMatchObject({ minimum: 0, defaultValue: 0 });
    expect(head.levels![0]).toContain("正对");
    const answers = Object.fromEntries(plan.parameters.map(value => [value.id, value.defaultValue]));
    const first = plan.expand({ ...answers, ParamAngleX: 25, ParamEyeBallX: 0.5 })!;
    // Head and gaze turn the same way.
    expect(Math.sign(first.ParamAngleX!)).toBe(Math.sign(first.ParamEyeBallX!));
    expect(Math.abs(first.ParamAngleX!)).toBe(25);
    const held = plan.expand({ ...answers, ParamAngleX: 12 })!;
    expect(Math.sign(held.ParamAngleX!)).toBe(Math.sign(first.ParamAngleX!));
    expect(plan.expand({ ...answers })!.ParamAngleX).toBe(0);
    const next = plan.expand({ ...answers, ParamAngleX: 25 })!;
    expect(Math.sign(next.ParamAngleX!)).toBe(-Math.sign(first.ParamAngleX!));
  });

  it("counts a quiet gap between replies as a rest, so the head does not always lean one way", () => {
    const { active } = buildBindings(model);
    let now = 0;
    const draws = [0.9, 0.1];
    const plan = planDecisions(active, new DecisionSides(() => draws.shift()!, () => now));
    const answers = Object.fromEntries(plan.parameters.map(value => [value.id, value.defaultValue]));
    const first = plan.expand({ ...answers, ParamAngleZ: 15 })!;
    now += 2_000;
    // Still the same gesture a moment later.
    expect(Math.sign(plan.expand({ ...answers, ParamAngleZ: 15 })!.ParamAngleZ!)).toBe(Math.sign(first.ParamAngleZ!));
    now += DECISION_REST_MS + 1;
    // Jev never answered zero, yet the next reply picks its side afresh.
    expect(Math.sign(plan.expand({ ...answers, ParamAngleZ: 15 })!.ParamAngleZ!)).toBe(-Math.sign(first.ParamAngleZ!));
  });

  it("refuses answers missing a question", () => {
    const plan = planDecisions(buildBindings(model).active);
    expect(plan.expand({ ParamEyeOpen: 1 })).toBeNull();
  });
});

