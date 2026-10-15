import type { CubismExpressionMotion } from "@cubism/motion/cubismexpressionmotion";
import { describe, expect, it } from "vitest";
import { expressionBlocksNaturalIdle, naturalIdleAffectedParameters } from "./natural-idle-conflicts";

function expression(ids: string[]): CubismExpressionMotion {
  return {
    getExpressionParameters: () => ids.map((id) => ({ parameterId: { getString: () => id } })),
  } as unknown as CubismExpressionMotion;
}

describe("natural idle expression conflicts", () => {
  it("allows 小兔猪's watermark and unrelated appearance or face parameters", () => {
    const affected = naturalIdleAffectedParameters([]);
    expect(expressionBlocksNaturalIdle(expression(["Param159"]), affected)).toBe(false);
    expect(expressionBlocksNaturalIdle(expression(["ParamHat", "ParamCheek"]), affected)).toBe(false);
    expect(expressionBlocksNaturalIdle(expression([]), affected)).toBe(false);
  });

  it("yields to expressions targeting driven head parameters without relying on CDI groups", () => {
    const affected = naturalIdleAffectedParameters([]);
    expect(expressionBlocksNaturalIdle(expression(["ParamAngleX"]), affected)).toBe(true);
    expect(expressionBlocksNaturalIdle(expression(["Param159", "ParamAngleZ"]), affected)).toBe(true);
  });

  it("protects custom physics outputs transitively and terminates with cyclic rigs", () => {
    const affected = naturalIdleAffectedParameters([
      { inputs: ["CustomBody"], outputs: ["CustomShoulder"] },
      { inputs: ["ParamAngleY", "OtherInput"], outputs: ["CustomBody"] },
      { inputs: ["CustomShoulder"], outputs: ["CustomBody"] },
      { inputs: ["ParamUnrelated"], outputs: ["ParamHat"] },
    ]);
    expect(expressionBlocksNaturalIdle(expression(["CustomShoulder"]), affected)).toBe(true);
    expect(expressionBlocksNaturalIdle(expression(["ParamHat"]), affected)).toBe(false);
    expect(affected.has("OtherInput")).toBe(false);
  });

  it("still yields when an expression cannot be inspected", () => {
    expect(expressionBlocksNaturalIdle({} as CubismExpressionMotion, naturalIdleAffectedParameters([]))).toBe(true);
  });
});
