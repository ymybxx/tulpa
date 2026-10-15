import type { CubismModel } from "@cubism/model/cubismmodel";
import { describe, expect, it } from "vitest";
import { CubismFacialOverlay } from "./cubism-facial-overlay";

function fixture() {
  const parameters = [
    { id: "Head", name: "", minimum: -30, maximum: 30, defaultValue: 0 },
    { id: "Eyes", name: "", minimum: 20, maximum: 80, defaultValue: 40 },
    { id: "Mouth", name: "", minimum: 20, maximum: 80, defaultValue: 80 },
    { id: "Gaze", name: "", minimum: -1, maximum: 1, defaultValue: 0 },
  ];
  const values = [0, 40, 80, 0];
  const model = { getParameterValueByIndex: (i: number) => values[i]!, setParameterValueByIndex: (i: number, value: number) => { values[i] = value; } } as unknown as CubismModel;
  const bindings = { neutralValues: { Head: 0, Eyes: 40, Mouth: 80, Gaze: 0 }, gazeParameterIds: ["Gaze"],
    mouth: { id: "Mouth", closed: 80, open: 20 }, eyes: [{ id: "Eyes", closed: 80 }] };
  return { values, model, layer: new CubismFacialOverlay(parameters, bindings, null) };
}

describe("continuous facial overlay", () => {
  it("keeps audio mouth opening written after the expression pass, including reversed bindings", () => {
    const { layer, values, model } = fixture();
    layer.controller.setTargets({ Mouth: 80 }); // Expression asks for a closed mouth.
    layer.onLateUpdate(model, 1 / 60);
    values[2] = 32; // Production updateLip runs AFTER onLateUpdate.
    layer.finishFrame(model, new Map([["Mouth", { value: 32, neutral: 0 }]]));
    expect(values[2]).toBe(32);
  });

  it("combines late vowel shape with expression without accumulating or erasing either", () => {
    const parameters = [
      { id: "Open", name: "", minimum: 0, maximum: 1, defaultValue: 0 },
      { id: "Form", name: "", minimum: -1, maximum: 1, defaultValue: 1 },
    ];
    const values = [0, 0];
    const model = { getParameterValueByIndex: (i: number) => values[i]!,
      setParameterValueByIndex: (i: number, value: number) => { values[i] = value; } } as unknown as CubismModel;
    const layer = new CubismFacialOverlay(parameters, { neutralValues: { Open: 0, Form: 1 },
      mouth: { id: "Open", closed: 0, open: 1 }, eyes: [], gazeParameterIds: [] }, null);
    layer.controller.setTargets({ Open: 0, Form: 0.6 });
    const late = new Map([["Open", { value: 0, neutral: 0 }], ["Form", { value: 0, neutral: 0 }]]);
    for (let frame = 0; frame < 180; frame++) {
      values.splice(0, 2, 0, 0);
      layer.onLateUpdate(model, 1 / 60);
      values.splice(0, 2, frame % 2 ? 0.8 : 0.2, 0.1);
      late.get("Open")!.value = values[0]!;
      late.get("Form")!.value = values[1]!;
      values[1] = 1; // A default appearance preset overwrites the phoneme.
      layer.finishFrame(model, late);
      expect(values[0]).toBe(frame % 2 ? 0.8 : 0.2);
      if (frame > 120) expect(values[1]).toBeCloseTo(0.7, 5);
    }
    layer.release();
    for (let frame = 0; frame < 30; frame++) {
      values.splice(0, 2, 0, 0);
      layer.onLateUpdate(model, 1 / 60);
      values.splice(0, 2, 0.4, -0.2);
      late.get("Open")!.value = values[0]!;
      late.get("Form")!.value = values[1]!;
      layer.finishFrame(model, late);
    }
    expect(layer.finished).toBe(true);
    expect(values).toEqual([0.4, -0.2]);
  });

  it("cannot invert a lowered mouth corner when appearance overwrites the phoneme", () => {
    const parameter = { id: "Form", name: "", minimum: -1, maximum: 1, defaultValue: 1 };
    let value = 1;
    const model = { getParameterValueByIndex: () => value,
      setParameterValueByIndex: (_: number, next: number) => { value = next; } } as unknown as CubismModel;
    const layer = new CubismFacialOverlay([parameter], { neutralValues: { Form: 1 }, mouth: null, eyes: [], gazeParameterIds: [] }, null);
    layer.controller.setTargets({ Form: -0.5 });
    const late = new Map([["Form", { value: 0.1, neutral: 0 }]]);
    for (let frame = 0; frame < 180; frame++) {
      value = 1;
      layer.onLateUpdate(model, 1 / 60);
      value = 1; // defaultAppearance runs after the captured audio value 0.1.
      layer.finishFrame(model, late);
    }
    expect(value).toBeCloseTo(-0.4, 5);
    layer.release();
    let previous = value;
    for (let frame = 0; frame < 20; frame++) {
      value = 1;
      layer.onLateUpdate(model, 1 / 60);
      value = 1;
      layer.finishFrame(model, late);
      expect(value).toBeGreaterThanOrEqual(previous);
      expect(value - previous).toBeLessThan(0.13);
      previous = value;
    }
    expect(value).toBe(1);
  });

  it("preserves the moving baseline and cannot accumulate its offsets across frames", () => {
    const { layer, values, model } = fixture();
    layer.controller.setTargets({ Head: 6 });
    for (let i = 0; i < 240; i++) {
      const base = 4 * Math.sin(i / 30); values[0] = base;
      layer.onLateUpdate(model, 1 / 60); layer.finishFrame(model);
      if (i > 120) expect(values[0]! - base).toBeCloseTo(6, 5);
    }
    const before = values[0]; layer.release();
    expect(values[0]).toBe(before); // No synchronous reset at end of speech.
    for (let i = 0; i < 18; i++) { values[0] = 3; layer.onLateUpdate(model, 1 / 60); layer.finishFrame(model); }
    expect(layer.finished).toBe(true);
    expect(values[0]).toBe(3);
  });

  it("retains blinks and TTS on inverted manual bindings, including late gaze compensation", () => {
    const { layer, values, model } = fixture();
    layer.controller.setTargets({ Eyes: 60, Gaze: 0.2 });
    layer.controller.setSpeechLevel(0.8);
    for (let i = 0; i < 120; i++) {
      values.splice(0, 4, 0, 40, 80, 0);
      layer.onLateUpdate(model, 1 / 60);
      values[3] = -0.3; // Existing idle computes gaze after physics.
      layer.finishFrame(model);
    }
    expect(values[1]).toBeCloseTo(60);
    expect(values[2]).toBeCloseTo(32);
    expect(values[3]).toBeCloseTo(-0.1);
    values[1] = 80; // A full blink must stay closed.
    layer.onLateUpdate(model, 1 / 60); layer.finishFrame(model);
    expect(values[1]).toBe(80);
  });

  it("resumes an interrupted fade continuously and respects model limits", () => {
    const { layer, values, model } = fixture();
    layer.controller.setTargets({ Head: 100 });
    layer.onLateUpdate(model, 1); layer.finishFrame(model);
    expect(values[0]).toBeLessThanOrEqual(30);
    layer.release(); values[0] = 0; layer.onLateUpdate(model, 0.15); layer.finishFrame(model);
    const halfway = values[0]; layer.resume();
    values[0] = 0; layer.onLateUpdate(model, 0); layer.finishFrame(model);
    expect(values[0]).toBe(halfway);
    expect(layer.finished).toBe(false);
  });
});
