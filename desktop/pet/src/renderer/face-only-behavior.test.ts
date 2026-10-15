import { describe, expect, it } from "vitest";

import type { CubismExpressionMotion } from "@cubism/motion/cubismexpressionmotion";
import type { CubismMotion } from "@cubism/motion/cubismmotion";
import { CubismMotionCurveTarget } from "@cubism/motion/cubismmotioninternal";

import {
  isFaceOnlyExpression,
  isFaceOnlyMotion,
  isFaceParameterGroup,
} from "./face-only-behavior";

const descriptors = [
  { id: "ParamEyeLOpen", name: "左眼", groupId: "ParamGroupEyes" },
  { id: "ParamMouthOpenY", name: "嘴巴", groupId: "ParamGroupMouth" },
  { id: "ParamAngleX", name: "角度 X", groupId: "ParamGroupFace" },
  { id: "ParamBodyAngleX", name: "身体角度 X", groupId: "ParamGroupBody" },
];

function id(value: string): { getString: () => string } {
  return { getString: () => value };
}

function motion(
  curves: Array<{ type: number; id: string }>,
): CubismMotion {
  return {
    _motionData: {
      curveCount: curves.length,
      curves: curves.map((curve) => ({ type: curve.type, id: id(curve.id) })),
    },
  } as unknown as CubismMotion;
}

function expression(ids: string[]): CubismExpressionMotion {
  return {
    getExpressionParameters: () => ids.map((value) => ({ parameterId: id(value) })),
  } as unknown as CubismExpressionMotion;
}

describe("face-only behavior inspection", () => {
  it("accepts only explicit facial CDI groups", () => {
    expect(isFaceParameterGroup("ParamGroupFace")).toBe(true);
    expect(isFaceParameterGroup("ParamGroupEyes")).toBe(true);
    expect(isFaceParameterGroup("ParamGroup_Mouth")).toBe(true);
    expect(isFaceParameterGroup("ParamGroupBody")).toBe(false);
    expect(isFaceParameterGroup(null)).toBe(false);
  });

  it("accepts a motion whose every curve targets a known face group", () => {
    const value = motion([
      {
        type: CubismMotionCurveTarget.CubismMotionCurveTarget_Parameter,
        id: "ParamEyeLOpen",
      },
      {
        type: CubismMotionCurveTarget.CubismMotionCurveTarget_Parameter,
        id: "ParamMouthOpenY",
      },
    ]);
    expect(isFaceOnlyMotion(value, descriptors)).toBe(true);
  });

  it("rejects mixed, non-parameter, unknown, and empty motions", () => {
    expect(isFaceOnlyMotion(
      motion([
        { type: CubismMotionCurveTarget.CubismMotionCurveTarget_Parameter, id: "ParamEyeLOpen" },
        { type: CubismMotionCurveTarget.CubismMotionCurveTarget_Parameter, id: "ParamBodyAngleX" },
      ]),
      descriptors,
    )).toBe(false);
    expect(isFaceOnlyMotion(
      motion([{ type: CubismMotionCurveTarget.CubismMotionCurveTarget_Model, id: "ParamEyeLOpen" }]),
      descriptors,
    )).toBe(false);
    expect(isFaceOnlyMotion(
      motion([{ type: CubismMotionCurveTarget.CubismMotionCurveTarget_Parameter, id: "ParamUnknown" }]),
      descriptors,
    )).toBe(false);
    expect(isFaceOnlyMotion(motion([]), descriptors)).toBe(false);
  });

  it("accepts facial expressions and rejects unknown or mixed expressions", () => {
    expect(isFaceOnlyExpression(expression(["ParamEyeLOpen", "ParamAngleX"]), descriptors)).toBe(true);
    expect(isFaceOnlyExpression(expression(["ParamEyeLOpen", "ParamBodyAngleX"]), descriptors)).toBe(false);
    expect(isFaceOnlyExpression(expression(["ParamUnknown"]), descriptors)).toBe(false);
    expect(isFaceOnlyExpression(expression([]), descriptors)).toBe(false);
    expect(isFaceOnlyExpression({} as CubismExpressionMotion, descriptors)).toBe(false);
  });

  it("fails closed for ambiguous CDI descriptors", () => {
    const duplicate = [...descriptors, descriptors[0]!];
    expect(isFaceOnlyExpression(expression(["ParamEyeLOpen"]), duplicate)).toBe(false);
  });
});

