import type { ACubismMotion } from "@cubism/motion/acubismmotion";
import type { CubismExpressionMotion } from "@cubism/motion/cubismexpressionmotion";
export const UNKNOWN_IDLE_OWNERSHIP = "*";
export function expressionParameterIds(expression: ACubismMotion): ReadonlySet<string> {
  const getter = (expression as CubismExpressionMotion).getExpressionParameters;
  if (typeof getter !== "function")
    return new Set([UNKNOWN_IDLE_OWNERSHIP]);
  try {
    return new Set(getter.call(expression).map(value => value.parameterId.getString()));
  }
  catch {
    return new Set([UNKNOWN_IDLE_OWNERSHIP]);
  }
}
/** Parsed once with the asset, including SDK model-level eye/lip curves. */
export function motionParameterIds(buffer: ArrayBuffer, eyes: readonly string[], lips: readonly string[]): ReadonlySet<string> {
  try {
    const json = JSON.parse(new TextDecoder().decode(buffer));
    if (!Array.isArray(json.Curves))
      return new Set([UNKNOWN_IDLE_OWNERSHIP]);
    const result = new Set<string>();
    for (const curve of json.Curves) {
      if (curve.Target === "Parameter" && typeof curve.Id === "string")
        result.add(curve.Id);
      if (curve.Target === "Model" && curve.Id === "EyeBlink")
        for (const id of eyes)
          result.add(id);
      if (curve.Target === "Model" && curve.Id === "LipSync")
        for (const id of lips)
          result.add(id);
    }
    return result;
  }
  catch {
    return new Set([UNKNOWN_IDLE_OWNERSHIP]);
  }
}
export function mergeParameterIds(sets: readonly ReadonlySet<string>[]): ReadonlySet<string> {
  const result = new Set<string>();
  for (const set of sets)
    for (const id of set)
      result.add(id);
  return result;
}
