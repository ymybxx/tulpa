import type { CubismExpressionMotion } from "@cubism/motion/cubismexpressionmotion";
import type { ACubismMotion } from "@cubism/motion/acubismmotion";
import {
  CubismMotionCurveTarget,
  type CubismMotionCurve,
} from "@cubism/motion/cubismmotioninternal";
import type { CubismMotion } from "@cubism/motion/cubismmotion";

import type { ModelParameterDescriptor } from "../shared/lipsync-adapter";

/**
 * Groups that are safe to keep alive after a speaking motion has finished.
 *
 * GroupId is authored in a model's CDI file, so it is intentionally treated
 * as an opaque value everywhere else.  We only accept this small, explicit
 * allow-list here.  A missing or unfamiliar group is therefore a safe
 * negative rather than an optimistic face classification.
 */
const FACE_GROUP_IDS = new Set([
  "face",
  "faces",
  "eye",
  "eyes",
  "eyeball",
  "eyeballs",
  "brow",
  "brows",
  "mouth",
  "lip",
  "lips",
  "paramgroupface",
  "paramgroupeye",
  "paramgroupeyes",
  "paramgroupeyeball",
  "paramgroupeyeballs",
  "paramgroupbrow",
  "paramgroupbrows",
  "paramgroupmouth",
  "paramgrouplip",
  "paramgrouplips",
]);

/**
 * Return whether a CDI parameter group is one of the explicitly supported
 * facial groups.  Null/blank/unknown values return false by design.
 */
export function isFaceParameterGroup(groupId: string | null | undefined): boolean {
  if (typeof groupId !== "string") return false;
  const normalized = groupId.trim().toLowerCase().replace(/[^a-z0-9]/g, "");
  return normalized.length > 0 && FACE_GROUP_IDS.has(normalized);
}

/**
 * Build the lookup used by both Motion and Expression inspection.
 * Duplicate IDs are rejected: an ambiguous CDI description must not turn an
 * unknown target into a false positive.
 */
function parameterGroups(
  descriptors: readonly ModelParameterDescriptor[],
): Map<string, string | null> | null {
  const result = new Map<string, string | null>();
  for (const descriptor of descriptors) {
    if (!descriptor || typeof descriptor.id !== "string" || !descriptor.id.trim()) {
      return null;
    }
    if (result.has(descriptor.id)) return null;
    result.set(descriptor.id, descriptor.groupId);
  }
  return result;
}

function parameterId(value: unknown): string | null {
  if (typeof value === "string") return value.trim() || null;
  if (!value || typeof value !== "object") return null;
  const getString = (value as { getString?: unknown }).getString;
  if (typeof getString !== "function") return null;
  try {
    const result = (getString as () => unknown).call(value);
    return typeof result === "string" && result.trim() ? result.trim() : null;
  } catch {
    return null;
  }
}

function allFaceParameters(
  ids: readonly unknown[],
  descriptors: readonly ModelParameterDescriptor[],
): boolean {
  if (ids.length === 0) return false;
  const groups = parameterGroups(descriptors);
  if (!groups) return false;
  for (const value of ids) {
    const id = parameterId(value);
    if (!id || !groups.has(id) || !isFaceParameterGroup(groups.get(id))) return false;
  }
  return true;
}

/**
 * Conservatively classify an official Cubism Motion as face-only.
 *
 * Every curve must target a model Parameter (not model opacity/part opacity),
 * every target ID must be present in CDI, and every target's group must be in
 * the explicit facial allow-list.  Any malformed, unknown, or mixed curve
 * makes this return false.
 */
export function isFaceOnlyMotion(
  motion: CubismMotion | null | undefined,
  descriptors: readonly ModelParameterDescriptor[],
): boolean {
  if (!motion) return false;
  try {
    const data = motion._motionData;
    if (!data || !Number.isInteger(data.curveCount) || data.curveCount <= 0) return false;
    const curves = data.curves as readonly CubismMotionCurve[] | undefined;
    if (!curves || curves.length < data.curveCount) return false;
    const ids: unknown[] = [];
    for (let index = 0; index < data.curveCount; index += 1) {
      const curve = curves[index];
      if (!curve || curve.type !== CubismMotionCurveTarget.CubismMotionCurveTarget_Parameter) {
        return false;
      }
      ids.push(curve.id);
    }
    return allFaceParameters(ids, descriptors);
  } catch {
    return false;
  }
}

interface ExpressionParameterLike {
  parameterId?: unknown;
}

/**
 * Conservatively classify an Expression as face-only.  The public
 * CubismExpressionMotion getter is used when available; the ACubismMotion
 * type is accepted because model instances store expressions by that base
 * type.  Expressions without an inspectable parameter list are unknown.
 */
export function isFaceOnlyExpression(
  expression: CubismExpressionMotion | ACubismMotion | null | undefined,
  descriptors: readonly ModelParameterDescriptor[],
): boolean {
  if (!expression) return false;
  try {
    const getter = (expression as unknown as {
      getExpressionParameters?: () => readonly ExpressionParameterLike[];
    }).getExpressionParameters;
    if (typeof getter !== "function") return false;
    const parameters = getter.call(expression);
    if (!Array.isArray(parameters) || parameters.length === 0) return false;
    return allFaceParameters(
      parameters.map((item) => item?.parameterId),
      descriptors,
    );
  } catch {
    return false;
  }
}
