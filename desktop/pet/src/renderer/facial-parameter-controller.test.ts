import { describe, expect, it } from "vitest";
import type { NativeModelParameter } from "../shared/contracts";
import { FacialParameterController, type SpeechDynamics } from "./facial-parameter-controller";

const parameters: NativeModelParameter[] = [
  { id: "ParamAngleX", name: "Head", minimum: -30, maximum: 30, defaultValue: 0 },
  { id: "ParamEyeLOpen", name: "Eye", minimum: 0, maximum: 2, defaultValue: 1 },
  { id: "ParamMouthOpenY", name: "Mouth", minimum: 0, maximum: 1, defaultValue: 0 },
  { id: "ParamMouthForm", name: "Smile", minimum: -1, maximum: 1, defaultValue: 0 },
];
const create = () => new FacialParameterController(parameters, { ParamAngleX: 5, ParamEyeLOpen: 0.8 });

describe("FacialParameterController", () => {
  it("admits only real model parameters and clamps absolute targets to their authored range", () => {
    const controller = create();
    controller.setTargets({ ParamAngleX: 100, ParamEyeLOpen: -1, ParamCheek: 1, ParamMouthForm: NaN });
    expect(controller.snapshot().targets).toEqual({ ParamAngleX: 30, ParamEyeLOpen: 0 });
    expect(controller.snapshot().values).toEqual({ ParamAngleX: 5, ParamEyeLOpen: 0.8 });
    expect(controller.update(1).ParamAngleX).toBeCloseTo(30, 2);
  });

  it("reaches the same pose at equal elapsed time across frame rates", () => {
    const values = [30, 60, 144].map(fps => {
      const controller = create();
      controller.setTargets({ ParamAngleX: -20, ParamEyeLOpen: 0.3 });
      for (let frame = 0; frame < fps; frame++) controller.update(1 / fps);
      return controller.snapshot().values;
    });
    for (const value of values.slice(1)) {
      expect(value.ParamAngleX).toBeCloseTo(values[0]!.ParamAngleX!, 10);
      expect(value.ParamEyeLOpen).toBeCloseTo(values[0]!.ParamEyeLOpen!, 10);
    }
  });

  it("keeps both the current position and travel direction when a target interrupts a transition", () => {
    const controller = create();
    controller.setTargets({ ParamAngleX: 25 });
    const before = controller.update(0.05).ParamAngleX!;
    controller.setTargets({ ParamAngleX: -25 });
    expect(controller.snapshot().values.ParamAngleX).toBe(before);
    expect(controller.update(0.001).ParamAngleX).toBeGreaterThan(before);
    expect(controller.update(2).ParamAngleX).toBeCloseTo(-25, 2);
  });

  it("smoothly restores removed targets to the captured editing appearance", () => {
    const controller = create();
    controller.setTargets({ ParamAngleX: 25, ParamEyeLOpen: 0.1 });
    const before = controller.update(1);
    controller.setTargets({});
    expect(controller.snapshot().values).toEqual(before);
    expect(controller.snapshot().targets).toEqual({});
    const middle = controller.update(0.1);
    expect(middle.ParamAngleX).toBeGreaterThan(5);
    expect(middle.ParamAngleX).toBeLessThan(before.ParamAngleX!);
    const settled = controller.update(1);
    expect(settled.ParamAngleX).toBeCloseTo(5, 3);
    expect(settled.ParamEyeLOpen).toBeCloseTo(0.8, 3);
  });

  it("combines speech with expression openness while preserving mouth form", () => {
    const controller = create();
    controller.setTargets({ ParamMouthOpenY: 0.25, ParamMouthForm: -0.6 });
    controller.setSpeechLevel(0.8);
    const speaking = controller.update(1);
    expect(speaking.ParamMouthOpenY).toBeCloseTo(0.8);
    expect(speaking.ParamMouthForm).toBeCloseTo(-0.6);
    controller.setSpeechLevel(0);
    const silent = controller.update(1);
    expect(silent.ParamMouthOpenY).toBeCloseTo(0.25);
    expect(silent.ParamMouthForm).toBeCloseTo(-0.6);
  });

  it("never creates a speech channel for models without the standard mouth parameter", () => {
    const controller = new FacialParameterController(parameters.slice(0, 2), {});
    controller.setSpeechLevel(1);
    expect(controller.update(1)).toEqual({});
  });

  it("keeps TTS on the mapped, reversed mouth without changing expression smoothing", () => {
    const controller = new FacialParameterController([
      ...parameters, { id: "CustomMouth", name: "嘴", minimum: 20, maximum: 80, defaultValue: 80 },
    ], {}, { id: "CustomMouth", closed: 80, open: 20 });
    controller.setTargets({ CustomMouth: 68 });
    controller.setSpeechLevel(0.8);
    expect(controller.update(1).CustomMouth).toBeCloseTo(32);
    expect(controller.snapshot().targets.CustomMouth).toBeCloseTo(32);
    expect(controller.snapshot().values).not.toHaveProperty("ParamMouthOpenY");
    controller.setSpeechLevel(0);
    expect(controller.update(1).CustomMouth).toBeCloseTo(68);
  });

  it("carries semantic head timing through a custom native ID and calibrated range", () => {
    const normal = new FacialParameterController(parameters, { ParamAngleX: 0 });
    const custom = new FacialParameterController([{ id: 'Custom', name: '', minimum: 20, maximum: 80, defaultValue: 50 }],
      { Custom: 50 }, null, { Custom: { group: 'head', minimum: 20, maximum: 80 } });
    normal.setTargets({ ParamAngleX: 24 }); custom.setTargets({ Custom: 26 }); // reversed mapping
    for (let i = 0; i < 35; i++) {
      const a = normal.update(1 / 60), b = custom.update(1 / 60);
      expect(b.Custom).toBeCloseTo(50 - a.ParamAngleX!, 8);
    }
    const before = custom.kinematics(); custom.setTargets({ Custom: 74 });
    expect(custom.kinematics()).toEqual(before);
    expect(custom.update(3).Custom).toBe(74);
  });

  it("waves a swinging part a few times around its step, then lets it rest", () => {
    const arm = { id: "ParamArmLA", name: "Arm", minimum: -10, maximum: 10, defaultValue: 0 };
    const controller = new FacialParameterController([arm], { ParamArmLA: 0 }, null,
      { ParamArmLA: { group: "arms", minimum: -10, maximum: 10, motion: "swing" } });
    controller.setTargets({ ParamArmLA: 8 });
    const trace: number[] = [];
    for (let frame = 0; frame < 60 * 8; frame++) trace.push(controller.update(1 / 60).ParamArmLA!);
    // Turning points: up near the step, back toward rest, several times.
    const peaks = trace.filter((value, index) => index > 0 && index < trace.length - 1
      && value > trace[index - 1]! && value >= trace[index + 1]! && value > 7);
    expect(peaks.length).toBeGreaterThanOrEqual(3);
    expect(trace.at(-1)).toBeCloseTo(0, 1);
    // The same step on the next sentence waves again.
    controller.setTargets({ ParamArmLA: 8 });
    for (let frame = 0; frame < 30; frame++) controller.update(1 / 60);
    expect(controller.snapshot().values.ParamArmLA).toBeGreaterThan(4);
  });

  it("reaches a pulsing part's step, holds briefly and comes back", () => {
    const head = { id: "ParamAngleY", name: "Nod", minimum: -30, maximum: 30, defaultValue: 0 };
    const controller = new FacialParameterController([head], { ParamAngleY: 0 }, null,
      { ParamAngleY: { group: "head", minimum: -30, maximum: 30, motion: "pulse" } });
    controller.setTargets({ ParamAngleY: -20 });
    let lowest = 0;
    for (let frame = 0; frame < 60 * 4; frame++) lowest = Math.min(lowest, controller.update(1 / 60).ParamAngleY!);
    expect(lowest).toBeLessThan(-18);
    expect(controller.snapshot().values.ParamAngleY).toBeCloseTo(0, 1);
  });

  it("holds other parts at their step for the whole sentence", () => {
    const controller = create();
    controller.setTargets({ ParamAngleX: 20 });
    for (let frame = 0; frame < 60 * 4; frame++) controller.update(1 / 60);
    expect(controller.snapshot().values.ParamAngleX).toBeCloseTo(20, 3);
  });
});

describe("expression variation", () => {
  const face = [
    { id: "ParamEyeLOpen", name: "左眼", minimum: 0, maximum: 1, defaultValue: 1 },
    { id: "ParamEyeROpen", name: "右眼", minimum: 0, maximum: 1, defaultValue: 1 },
    { id: "ParamAngleY", name: "头", minimum: -30, maximum: 30, defaultValue: 0 },
  ];
  const settle = (controller: FacialParameterController, seconds = 3) => {
    let values = {} as Readonly<Record<string, number>>;
    for (let frame = 0; frame < seconds * 60; frame += 1) values = controller.update(1 / 60);
    return values;
  };

  it("changes the face first, then turns the head, then lets the body follow, each family a share further or shorter", () => {
    // Draws, family by family: eyes scale and wait, head scale and wait, body scale and wait.
    const draws = [1, 0, 0, 0, 0.5, 0];
    const parts = [...face, { id: "ParamBodyAngleX", name: "身体", minimum: -10, maximum: 10, defaultValue: 0 }];
    const controller = new FacialParameterController(parts, {}, null, undefined,
      { amplitude: 0.2, onset: { head: 0.15, body: 0.3 }, onsetSpread: 0.08, random: () => draws.shift() ?? 0.5 });
    controller.setTargets({ ParamEyeLOpen: 0.5, ParamEyeROpen: 0.5, ParamAngleY: -10, ParamBodyAngleX: -5 });
    const atFace = controller.update(0.1);
    expect(atFace.ParamEyeLOpen).toBeLessThan(1);
    expect(atFace.ParamAngleY).toBe(0);
    expect(atFace.ParamBodyAngleX).toBe(0);
    const atHead = controller.update(0.15);
    expect(atHead.ParamAngleY).toBeLessThan(0);
    expect(atHead.ParamBodyAngleX).toBe(0);
    expect(controller.update(0.15).ParamBodyAngleX).toBeLessThan(0);
    const values = settle(controller, 4);
    // Both eyes 20% further from rest; the head 20% shorter; the body as asked.
    expect(values.ParamEyeLOpen).toBeCloseTo(0.4, 2);
    expect(values.ParamEyeROpen).toBeCloseTo(0.4, 2);
    expect(values.ParamAngleY).toBeCloseTo(-8, 1);
    expect(values.ParamBodyAngleX).toBeCloseTo(-5, 1);
    expect(controller.settled()).toBe(false);
    controller.setTargets({});
    settle(controller, 4);
    expect(controller.settled()).toBe(true);
  });

  it("does not restart a part when the same step is asked again", () => {
    let calls = 0;
    const controller = new FacialParameterController(face, {}, null, undefined,
      { amplitude: 0.2, onset: { head: 0, body: 0 }, onsetSpread: 0, random: () => { calls += 1; return 1; } });
    controller.setTargets({ ParamAngleY: -10 });
    settle(controller);
    const drawn = calls;
    controller.setTargets({ ParamAngleY: -10 });
    expect(calls).toBe(drawn);
    expect(settle(controller).ParamAngleY).toBeCloseTo(-12, 1);
  });

  it("keeps the exact step and timing without variation, as previews need", () => {
    const controller = new FacialParameterController(face, {}, null);
    controller.setTargets({ ParamAngleY: -10 });
    expect(controller.update(0.1).ParamAngleY).toBeLessThan(0);
    expect(settle(controller).ParamAngleY).toBeCloseTo(-10, 1);
  });
});

describe("expressions following the voice within a sentence", () => {
  const face = [
    { id: "ParamEyeLOpen", name: "左眼", minimum: 0, maximum: 1, defaultValue: 1 },
    { id: "ParamBrowLY", name: "左眉", minimum: -1, maximum: 1, defaultValue: 0 },
    { id: "ParamAngleY", name: "头", minimum: -30, maximum: 30, defaultValue: 0 },
    { id: "ParamEyeBallX", name: "眼珠", minimum: -1, maximum: 1, defaultValue: 0 },
  ];
  const dynamics: SpeechDynamics = { floor: { face: 0.6, head: 0.85, body: 0.85 }, pause: 0.18, blinkGap: 1.5,
    intonation: { head: 0.5, headLimit: 3, brow: 0.04, browLimit: 0.25 }, glance: { start: 0, pause: 0, gap: 2 } };
  const transitions = {
    ParamEyeLOpen: { group: "eyes", minimum: 0, maximum: 1 },
    ParamBrowLY: { group: "brows", minimum: -1, maximum: 1, follow: { part: "brow" as const, up: 1, down: 1 } },
    ParamAngleY: { group: "head", minimum: -30, maximum: 30, follow: { part: "headPitch" as const, up: 1, down: 1 } },
    ParamEyeBallX: { group: "gaze", minimum: -1, maximum: 1, follow: { part: "eyeX" as const, up: 1, down: 1 } },
  };
  const create = (options: Partial<SpeechDynamics> = {}) => new FacialParameterController(face, { ParamEyeLOpen: 1, ParamBrowLY: 0, ParamAngleY: 0 }, null, transitions,
    undefined, { ...dynamics, ...options }, [{ id: "ParamEyeLOpen", closed: 0 }]);
  const run = (controller: FacialParameterController, level: number, seconds: number, each?: (values: Readonly<Record<string, number>>) => void, pitch = 0) => {
    let values = {} as Readonly<Record<string, number>>;
    for (let frame = 0; frame < Math.round(seconds * 60); frame += 1) {
      controller.hearVoice(level, pitch);
      values = controller.update(1 / 60);
      each?.(values);
    }
    return values;
  };

  it("shows the whole step before the voice starts and on loud words, and eases back on quiet ones", () => {
    const controller = create();
    controller.setTargets({ ParamBrowLY: -1, ParamAngleY: -10 });
    expect(run(controller, 0, 2).ParamBrowLY).toBeCloseTo(-1, 2);
    expect(run(controller, 0.8, 1).ParamBrowLY).toBeCloseTo(-1, 1);
    const quiet = run(controller, 0.15, 1.5);
    expect(quiet.ParamBrowLY).toBeGreaterThan(-0.8);
    expect(quiet.ParamBrowLY).toBeLessThan(-0.55);
    // The head eases back much less than the face.
    expect(quiet.ParamAngleY).toBeLessThan(-8);
  });

  it("does not nod on loud words: at a steady pitch the head holds still however the loudness jumps", () => {
    const controller = create(), head: number[] = [];
    run(controller, 0.2, 2, undefined, 220);
    for (const level of [0.9, 0.2, 0.9, 0.2, 0.9]) run(controller, level, 0.3, values => head.push(values.ParamAngleY!), 220);
    expect(Math.max(...head.map(Math.abs))).toBeLessThan(0.2);
  });

  it("blinks once when the voice pauses, and not when it is silent from the start", () => {
    const silent = create(), eye: number[] = [];
    run(silent, 0, 1, values => eye.push(values.ParamEyeLOpen!));
    expect(Math.min(...eye)).toBeCloseTo(1, 3);
    const controller = create();
    run(controller, 0.6, 1);
    const paused: number[] = [];
    run(controller, 0, 1.5, values => paused.push(values.ParamEyeLOpen!));
    expect(Math.min(...paused)).toBeLessThan(0.1);
    expect(paused.at(-1)).toBeCloseTo(1, 3);
    expect(paused.filter((value, index) => value < 0.5 && !(paused[index - 1]! < 0.5)).length).toBe(1);
    // Pauses close together blink once; spaced out, each does.
    const blinks = (gap: number) => {
      const speaker = create(), series: number[] = [];
      for (let i = 0; i < 4; i += 1) { run(speaker, 0.6, gap); run(speaker, 0, 0.4, values => series.push(values.ParamEyeLOpen!)); }
      return series.filter((value, index) => value < 0.5 && !(series[index - 1]! < 0.5)).length;
    };
    expect(blinks(0.3)).toBe(2);
    expect(blinks(1.4)).toBe(4);
  });

  it("lifts the head and brows as the voice rises above its own pitch and lowers the head as it falls", () => {
    for (const base of [110, 260]) {
      const controller = create();
      run(controller, 0.5, 3, undefined, base);
      const level = run(controller, 0.5, 0.1, undefined, base);
      expect(Math.abs(level.ParamAngleY!)).toBeLessThan(0.3);
      // Four semitones up, then four down: the same for a low voice and a high one.
      const up = run(controller, 0.5, 1, undefined, base * 2 ** (4 / 12));
      expect(up.ParamAngleY).toBeGreaterThan(1.2);
      expect(up.ParamAngleY).toBeLessThanOrEqual(3);
      expect(up.ParamBrowLY).toBeGreaterThan(0.1);
      run(controller, 0.5, 2, undefined, base);
      const down = run(controller, 0.5, 1, undefined, base * 2 ** (-4 / 12));
      expect(down.ParamAngleY).toBeLessThan(-1.2);
      expect(down.ParamBrowLY).toBeCloseTo(0, 3);
      // In a pause it settles back.
      expect(Math.abs(run(controller, 0, 2).ParamAngleY!)).toBeLessThan(0.1);
    }
  });

  it("glances away and back as a turn starts, and keeps glances apart", () => {
    const always = create({ glance: { start: 1, pause: 1, gap: 2 }, random: () => 0.9 });
    run(always, 0, 1.5);
    const eyes: number[] = [];
    run(always, 0.6, 1.5, values => eyes.push(values.ParamEyeBallX!));
    expect(Math.max(...eyes)).toBeGreaterThan(0.4);
    expect(eyes.at(-1)).toBeCloseTo(0, 3);
    // A pause right after does not glance again within the gap.
    const again: number[] = [];
    run(always, 0, 0.4, values => again.push(values.ParamEyeBallX!));
    expect(Math.max(...again.map(Math.abs))).toBeLessThan(0.01);
    const never = create({ glance: { start: 0, pause: 0, gap: 2 } }), still: number[] = [];
    run(never, 0, 1.5); run(never, 0.6, 2, values => still.push(values.ParamEyeBallX!));
    expect(Math.max(...still.map(Math.abs))).toBeLessThan(1e-6);
  });

  it("leaves expressions exact without dynamics", () => {
    const controller = new FacialParameterController(face, {}, null, transitions);
    controller.setTargets({ ParamBrowLY: -1 });
    expect(run(controller, 0.15, 3).ParamBrowLY).toBeCloseTo(-1, 2);
  });
});
