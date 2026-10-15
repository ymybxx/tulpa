import type { CubismModel } from "@cubism/model/cubismmodel";
import type { ACubismMotion } from "@cubism/motion/acubismmotion";
import { CubismExpressionMotionManager } from "@cubism/motion/cubismexpressionmotionmanager";
import type { CubismMotion } from "@cubism/motion/cubismmotion";
import { CubismMotionManager } from "@cubism/motion/cubismmotionmanager";

export interface CubismPoseSelectionSnapshot {
  parameterIndices: Int32Array;
  values: Float32Array;
}

/**
 * A deterministic behavior layer set. Cubism's managers intentionally fade an
 * existing entry when another starts, so every selected file owns a dedicated
 * manager. Arrays are updated in manifest order.
 */
export class CubismBehaviorGroup {
  private readonly motionManagers: CubismMotionManager[] = [];
  private readonly expressionManagers: CubismExpressionMotionManager[] = [];
  private readonly motionLoops: Array<{ motion: CubismMotion; original: boolean }> = [];
  private readonly expressionMotions: ACubismMotion[] = [];
  private released = false;

  constructor(
    motions: readonly CubismMotion[],
    expressions: readonly ACubismMotion[],
    readonly loopMotions: boolean,
  ) {
    try {
      for (const motion of motions) {
        const original = motion.getLoop();
        motion.setLoop(loopMotions);
        const manager = new CubismMotionManager();
        manager.startMotionPriority(motion, false, 3);
        this.motionLoops.push({ motion, original });
        this.motionManagers.push(manager);
      }
      for (const expression of expressions) {
        const manager = new CubismExpressionMotionManager();
        manager.startMotion(expression, false);
        this.expressionMotions.push(expression);
        this.expressionManagers.push(manager);
      }
    } catch (error) {
      this.release();
      throw error;
    }
  }

  updateMotions(model: CubismModel, deltaSeconds: number): boolean {
    let updated = false;
    for (const manager of this.motionManagers) {
      updated = manager.updateMotion(model, deltaSeconds) || updated;
    }
    return updated;
  }

  updateExpressions(model: CubismModel, deltaSeconds: number): boolean {
    let updated = false;
    for (const manager of this.expressionManagers) {
      updated = manager.updateMotion(model, deltaSeconds) || updated;
    }
    return updated;
  }

  /** Advance a persistent expression layer to full weight before its first drawn frame. */
  primeExpressions(model: CubismModel): void {
    if (this.expressionManagers.length === 0) return;
    this.updateExpressions(model, 0);
    const duration = this.fadeInDurationSeconds();
    if (duration > 0) this.updateExpressions(model, duration);
  }

  hasMotions(): boolean {
    return this.motionManagers.length > 0;
  }

  motionsFinished(): boolean {
    return this.motionManagers.length > 0 &&
      this.motionManagers.every((manager) => manager.isFinished());
  }

  motionCycleDurationSeconds(): number | null {
    if (!this.motionLoops.length) return null;
    const durations = this.motionLoops.map(({ motion }) => motion.getLoopDuration());
    if (durations.some((value) => !Number.isFinite(value) || value <= 0)) return 8;
    return Math.max(...durations);
  }

  fadeInDurationSeconds(): number {
    return maxMotionDuration([
      ...this.motionLoops.map(({ motion }) => motion.getFadeInTime()),
      ...this.expressionMotions.map((motion) => motion.getFadeInTime()),
    ]);
  }

  fadeOutDurationSeconds(): number {
    return maxMotionDuration([
      ...this.motionLoops.map(({ motion }) => motion.getFadeOutTime()),
      ...this.expressionMotions.map((motion) => motion.getFadeOutTime()),
    ]);
  }

  managerCounts(): { motions: number; expressions: number } {
    return { motions: this.motionManagers.length, expressions: this.expressionManagers.length };
  }

  release(): void {
    if (this.released) return;
    this.released = true;
    for (const manager of this.motionManagers) {
      manager.stopAllMotions();
      manager.release();
    }
    for (const manager of this.expressionManagers) {
      manager.stopAllMotions();
      manager.release();
    }
    for (const { motion, original } of this.motionLoops) motion.setLoop(original);
    this.motionManagers.length = 0;
    this.expressionManagers.length = 0;
    this.motionLoops.length = 0;
    this.expressionMotions.length = 0;
  }
}

function maxMotionDuration(values: readonly number[]): number {
  const valid = values.filter((value) => Number.isFinite(value) && value > 0);
  return valid.length > 0 ? Math.max(...valid) : 0;
}

/** Clears the motion baseline so an empty state really is the authored model pose. */
export function restoreCubismDefaultParameters(model: CubismModel): void {
  for (let index = 0; index < model.getParameterCount(); index += 1) {
    model.setParameterValueByIndex(index, model.getParameterDefaultValue(index), 1);
  }
  model.saveParameters();
}

/**
 * Capture the virtual selector parameters used by CubismPose.
 *
 * A pose part uses its Part ID as a parameter ID. CubismModel creates those
 * selectors outside the Core parameter array, so saveParameters() and ordinary
 * parameter snapshots intentionally do not include them.
 */
export function captureCubismPoseSelection(
  model: CubismModel,
): CubismPoseSelectionSnapshot {
  const parameterIndices: number[] = [];
  const values: number[] = [];
  const seen = new Set<number>();
  for (let partIndex = 0; partIndex < model.getPartCount(); partIndex += 1) {
    const parameterIndex = model.getParameterIndex(model.getPartId(partIndex));
    if (seen.has(parameterIndex)) continue;
    seen.add(parameterIndex);
    const value = model.getParameterValueByIndex(parameterIndex);
    // Non-pose Parts receive a null virtual value from the Framework. Keep
    // only selectors initialized by CubismPose or backed by a real parameter.
    if (!Number.isFinite(value)) continue;
    parameterIndices.push(parameterIndex);
    values.push(value);
  }
  return {
    parameterIndices: Int32Array.from(parameterIndices),
    values: Float32Array.from(values),
  };
}

/** Restore selector targets; CubismPose performs the actual opacity crossfade. */
export function restoreCubismPoseSelection(
  model: CubismModel,
  snapshot: CubismPoseSelectionSnapshot,
): void {
  const count = Math.min(snapshot.parameterIndices.length, snapshot.values.length);
  for (let index = 0; index < count; index += 1) {
    model.setParameterValueByIndex(
      snapshot.parameterIndices[index]!,
      snapshot.values[index]!,
      1,
    );
  }
}
