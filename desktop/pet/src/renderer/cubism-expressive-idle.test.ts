import { describe, expect, it } from "vitest";
import type { CubismModel } from "@cubism/model/cubismmodel";
import type { CubismPhysics } from "@cubism/physics/cubismphysics";
import { CubismExpressiveIdle } from "./cubism-expressive-idle";
import { idlePresetSettings } from "../shared/natural-idle-settings";
import { resolveIdleRig } from "./idle-parameter-binding";
import { motionParameterIds } from "./idle-parameter-ownership";
function setup() {
  const ids = ["ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamAngleX1", "ParamAngleY1", "ParamBodyAngleX1", "ParamBodyAngleY1", "ParamBodyAngleZ1", "ParamEyeLOpen", "ParamEyeROpen", "ParamEyeBallX", "ParamEyeBallY", "ParamMouthOpenY", "Param83", "Param85"];
  const base = ids.map(id => id.includes("Eye") && id.endsWith("Open") ? 1 : id === "ParamMouthOpenY" ? 0.4 : id === "Param85" ? 1 : 0);
  const values = new Float32Array(base);
  const min = ids.map(id => id.includes("EyeBall") ? -1 : id.includes("Angle") ? -30 : 0);
  const max = ids.map(id => id.includes("EyeBall") ? 1 : id.includes("Angle") ? 30 : id.endsWith("Open") ? 2 : 1);
  const model = { getParameterCount: () => ids.length, getModel: () => ({ parameters: { ids, values, defaultValues: base, minimumValues: min, maximumValues: max } }),
    getParameterValueByIndex: (i: number) => values[i], setParameterValueByIndex: (i: number, v: number) => { expect(i).toBeGreaterThanOrEqual(0); values[i] = Math.max(min[i]!, Math.min(max[i]!, v)); } } as unknown as CubismModel;
  const outputs = ["ParamAngleX1", "ParamAngleY1", "ParamBodyAngleX1", "ParamBodyAngleY1", "ParamBodyAngleZ1"].map(id => ({ destination: { id: { getString: () => id } }, weight: 100, reflect: id === "ParamBodyAngleX1" }));
  const physics = { _physicsRig: { outputs } } as unknown as CubismPhysics;
  let state = 12345;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
  const controller = new CubismExpressiveIdle(model, physics, random);
  const set = (id: string, value: number) => { values[ids.indexOf(id)] = value; };
  const get = (id: string) => values[ids.indexOf(id)]!;
  const tick = (author: () => void = () => { }) => {
    values.set(base);
    author();
    for (const stage of controller.stages.filter(s => s.getExecutionOrder() < 400))
      stage.onLateUpdate(model, 1 / 60);
    set("ParamAngleX", get("ParamAngleX") + 5);
    set("ParamEyeBallX", 0.9);
    controller.stages.find(s => s.getExecutionOrder() === 550.5)!.onLateUpdate(model, 1 / 60);
    for (const output of outputs) {
      const id = output.destination.id.getString(), weight = output.weight / 100;
      set(id, get(id) * (1 - weight) + get("ParamAngleX") * 0.3 * weight);
    }
    controller.stages.find(s => s.getExecutionOrder() === 650)!.onLateUpdate(model, 1 / 60);
  };
  return { model, physics, controller, outputs, tick, get, set };
}
describe("expressive idle parameter ownership", () => {
  it("keeps head, torso and blinking running during a hand-only action", () => {
    const s = setup();
    s.controller.configure(idlePresetSettings("active"));
    s.controller.setOwnership(new Set(["Param83"]), new Set(), new Set(["Param85"]));
    const head: number[] = [], eyes: number[] = [];
    for (let i = 0; i < 1200; i++) {
      s.tick(() => s.set("Param83", 1));
      head.push(s.get("ParamAngleX"));
      eyes.push(s.get("ParamEyeLOpen"));
      expect(s.get("Param83")).toBe(1);
      expect(s.get("Param85")).toBe(1);
      expect(s.get("ParamMouthOpenY")).toBeCloseTo(0.4);
    }
    expect(Math.max(...head) - Math.min(...head)).toBeGreaterThan(10);
    expect(Math.min(...eyes)).toBeLessThan(0.01);
    expect(Math.max(...eyes)).toBeGreaterThan(1);
  });
  it("yields keyed channels to a body/eye motion and restores physics on disable", () => {
    const s = setup();
    s.controller.configure(idlePresetSettings("active"));
    for (let i = 0; i < 120; i++)
      s.tick();
    const time = (s.controller.diagnostics() as {
      time: number;
    }).time;
    s.controller.setOwnership(new Set(["ParamAngleX1", "ParamBodyAngleX1", "ParamEyeLOpen"]), new Set(), new Set());
    for (let i = 0; i < 300; i++)
      s.tick(() => { s.set("ParamAngleX1", 12); s.set("ParamBodyAngleX1", 7); s.set("ParamEyeLOpen", 0.3); });
    expect(s.get("ParamAngleX1")).toBeCloseTo(12);
    expect(s.get("ParamBodyAngleX1")).toBeCloseTo(7);
    expect(s.get("ParamEyeLOpen")).toBeCloseTo(0.3);
    expect((s.controller.diagnostics() as {
      time: number;
    }).time).toBeGreaterThan(time + 4);
    s.controller.setOwnership(new Set(), new Set(), new Set());
    s.controller.configure({ enabled: false, intensity: 1, speed: 1 });
    let previous = s.get("ParamBodyAngleX1"), maxStep = 0;
    for (let i = 0; i < 300; i++) {
      s.tick();
      maxStep = Math.max(maxStep, Math.abs(s.get("ParamBodyAngleX1") - previous));
      previous = s.get("ParamBodyAngleX1");
    }
    expect(maxStep).toBeLessThan(2);
    expect(s.outputs.every(o => o.weight === 100)).toBe(true);
    expect(s.controller.isActive()).toBe(false);
  });
  it("lets a half-open expression blink without overriding its eye pose", () => {
    const s = setup();
    s.controller.configure(idlePresetSettings("active"));
    s.controller.setOwnership(new Set(), new Set(["ParamEyeLOpen"]), new Set());
    const eyes: number[] = [];
    for (let i = 0; i < 1200; i++) {
      s.tick(() => s.set("ParamEyeLOpen", 0.5));
      eyes.push(s.get("ParamEyeLOpen"));
    }
    expect(Math.min(...eyes)).toBeLessThan(0.01);
    expect(Math.max(...eyes)).toBeLessThanOrEqual(0.5);
  });
  it("uses real custom mappings and rejects missing or duplicate targets", () => {
    const s = setup();
    const settings = idlePresetSettings("active").expressive!;
    settings.bindings = { bodyX: "Param83", bodyY: "Param83", headX: "missing", eyeX: null };
    const rig = resolveIdleRig(s.model, s.physics, settings);
    expect(rig.channels.bodyX?.id).toBe("Param83");
    expect(rig.channels.bodyY).toBeUndefined();
    expect(rig.channels.headX).toBeUndefined();
    expect(rig.channels.eyeX).toBeUndefined();
  });
  it("recognizes motion-owned eye effects without blocking unrelated channels", () => {
    const bytes = new TextEncoder().encode(JSON.stringify({ Curves: [{ Target: "Parameter", Id: "Param83" }, { Target: "Model", Id: "EyeBlink" }] }));
    expect([...motionParameterIds(bytes.buffer, ["ParamEyeLOpen", "ParamEyeROpen"], [])]).toEqual(["Param83", "ParamEyeLOpen", "ParamEyeROpen"]);
  });
  it("keeps normal eye openness at high activity and still follows the pointer at zero", () => {
    const s = setup(), settings = idlePresetSettings("active");
    settings.expressive!.eyeActivity = 2;
    settings.expressive!.blink = false;
    s.controller.configure(settings);
    for (let frame = 0; frame < 3600; frame++) {
      s.tick();
      expect(s.get("ParamEyeLOpen")).toBeGreaterThanOrEqual(0.999);
      expect(s.get("ParamEyeROpen")).toBeGreaterThanOrEqual(0.999);
      expect(s.get("ParamEyeLOpen")).toBeLessThan(1.013);
    }
    settings.expressive!.eyeActivity = 0;
    settings.intensity = 0;
    s.controller.configure(settings);
    for (let frame = 0; frame < 300; frame++) s.tick();
    const neutral = s.get("ParamEyeBallX");
    for (let frame = 0; frame < 60; frame++) {
      s.controller.setGaze(1, 0, "pointer");
      s.tick();
    }
    expect(s.get("ParamEyeBallX") - neutral).toBeGreaterThan(0.2);
  });
  it("looks at the pointer eyes first: the head follows and the eyes ease back as it arrives", () => {
    const s = setup(), settings = idlePresetSettings("active");
    settings.intensity = 0;
    Object.assign(settings.expressive!, { bodyIntensity: 0, eyeActivity: 0, blink: false });
    s.controller.configure(settings);
    // The pointer is already being followed when it jumps across.
    for (let frame = 0; frame < 300; frame++) { s.controller.setGaze(0, 0, "pointer"); s.tick(); }
    const head0 = s.get("ParamAngleX"), eye0 = s.get("ParamEyeBallX");
    const head: number[] = [], eyes: number[] = [];
    for (let frame = 0; frame < 90; frame++) {
      s.controller.setGaze(1, 0, "pointer");
      s.tick();
      head.push(s.get("ParamAngleX") - head0); eyes.push(s.get("ParamEyeBallX") - eye0);
    }
    const settledHead = head.at(-1)!, peakEyes = Math.max(...eyes);
    expect(settledHead).toBeGreaterThan(3);
    // A tenth of a second in, the eyes are most of the way there and the head has barely started.
    expect(eyes[6]! / peakEyes).toBeGreaterThan(0.8);
    expect(head[6]! / settledHead).toBeLessThan(0.45);
    expect(eyes.at(-1)!).toBeLessThan(peakEyes * 0.97);
    expect(eyes.at(-1)!).toBeGreaterThan(0.1);
  });
  it("moves the final eye-ball parameters independently while the head stays still", () => {
    const s = setup(), settings = idlePresetSettings("active");
    settings.intensity = 0;
    Object.assign(settings.expressive!, { bodyIntensity: 0, eyeActivity: 2, blink: false });
    s.controller.configure(settings);
    for (let frame = 0; frame < 300; frame++) s.tick();
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, offCenterFrames = 0;
    for (let frame = 0; frame < 3600; frame++) {
      s.tick();
      const x = s.get("ParamEyeBallX");
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, s.get("ParamEyeBallY")); maxY = Math.max(maxY, s.get("ParamEyeBallY"));
      if (Math.hypot(x, s.get("ParamEyeBallY")) > 0.35) offCenterFrames++;
      expect(Math.abs(s.get("ParamAngleX"))).toBeLessThan(0.001);
      expect(s.get("ParamEyeLOpen")).toBeGreaterThanOrEqual(0.999);
    }
    expect(minX).toBeLessThan(-0.4);
    expect(maxX).toBeGreaterThan(0.4);
    expect(minY).toBeLessThan(-0.35);
    expect(maxY).toBeGreaterThan(0.4);
    // A glance must remain visible long enough to read, rather than flash by.
    expect(offCenterFrames).toBeGreaterThan(300);
    s.controller.setOwnership(new Set(["ParamEyeBallX"]), new Set(), new Set());
    for (let frame = 0; frame < 180; frame++) s.tick(() => s.set("ParamEyeBallX", 0.2));
    expect(s.get("ParamEyeBallX")).toBeCloseTo(0.2);
  });
});
