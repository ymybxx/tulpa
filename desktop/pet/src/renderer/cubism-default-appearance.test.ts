import type { CubismModel } from "@cubism/model/cubismmodel";
import type { CubismExpressionMotion } from "@cubism/motion/cubismexpressionmotion";
import { ExpressionBlendType } from "@cubism/motion/cubismexpressionmotion";
import { describe, expect, it } from "vitest";
import { CubismDefaultAppearance } from "./cubism-default-appearance";

function expression(id: string, value: number, blendType = ExpressionBlendType.Additive): CubismExpressionMotion {
  return { getExpressionParameters: () => [{ parameterId: { getString: () => id }, value, blendType }] } as unknown as CubismExpressionMotion;
}

function fixture() {
  const ids = ["Watermark", "Hat", "ParamAngleX"];
  const defaults = [0, 1, 0];
  const values = [...defaults];
  const model = {
    getModel: () => ({ parameters: { ids } }),
    getParameterDefaultValue: (index: number) => defaults[index],
    getParameterMinimumValue: () => 0,
    getParameterMaximumValue: () => 1,
    setParameterValueByIndex: (index: number, value: number) => { values[index] = value; },
  } as unknown as CubismModel;
  return { model, values, defaults, layer: new CubismDefaultAppearance(model) };
}

describe("saved default appearance", () => {
  it("prioritizes explicit native values, clamps ranges, and ignores unknown IDs", () => {
    const { model, values, layer } = fixture();
    layer.configure([expression("Watermark", 0.2), expression("Hat", -1)], {
      Watermark: 2, ParamAngleX: 0.6, Missing: 1, Hat: NaN,
    });
    for (let i = 0; i < 600; i++) {
      values[0] = 0; values[2] = 0.9;
      layer.apply(model);
    }
    expect(values).toEqual([1, 0, 0.6]);
    expect([...layer.parameters()]).toEqual(["Watermark", "Hat", "ParamAngleX"]);
    layer.configure([expression("Watermark", 0.2)], { ParamAngleX: -10 });
    layer.apply(model);
    expect(values[0]).toBe(0.2);
    expect(values[2]).toBe(0);
    layer.configure([]);
    expect(layer.affects(new Set(["ParamAngleX"])) ).toBe(false);
  });

  it("applies multiple switches immediately and without additive drift", () => {
    const { model, values, layer } = fixture();
    layer.configure([expression("Watermark", 1), expression("Hat", -1)]);
    for (let i = 0; i < 600; i++) layer.onLateUpdate(model);
    expect(values).toEqual([1, 0, 0]);
  });

  it("holds selected channels against motions without freezing unrelated motion", () => {
    const { model, values, layer } = fixture();
    layer.configure([expression("Watermark", 1)]);
    values[0] = 0;
    values[2] = 0.4;
    layer.apply(model);
    expect(values).toEqual([1, 1, 0.4]);
    expect(layer.affects(new Set(["ParamAngleX"]))).toBe(false);
  });

  it("uses authored defaults for multiply and lets the last overlapping switch win", () => {
    const { model, values, layer } = fixture();
    layer.configure([expression("Hat", 0.4, ExpressionBlendType.Multiply), expression("Watermark", 1), expression("Watermark", 0.5)]);
    layer.apply(model);
    expect(values).toEqual([0.5, 0.4, 0]);
  });

  it("releases cleared switches and never creates missing model parameters", () => {
    const { model, values, defaults, layer } = fixture();
    layer.configure([expression("Watermark", 1), expression("Missing", 1)]);
    layer.apply(model);
    layer.configure([]);
    values.splice(0, values.length, ...defaults);
    layer.apply(model);
    expect(values).toEqual(defaults);
  });
});
