import type { CubismModel } from "@cubism/model/cubismmodel";
import { CubismUpdateOrder, ICubismUpdater } from "@cubism/motion/icubismupdater";
import type { CubismBehaviorGroup } from "./cubism-behavior-group";

/** Keep behavior expressions in Cubism's expression phase, after automatic blink. */
export class CubismBehaviorExpressionUpdater extends ICubismUpdater {
  constructor(
    private readonly active: () => CubismBehaviorGroup | null,
    private readonly persistent: () => CubismBehaviorGroup | null,
  ) {
    super(CubismUpdateOrder.CubismUpdateOrder_Expression);
  }

  onLateUpdate(model: CubismModel, deltaSeconds: number): void {
    // Preserve saved layer order, including the speaking face layer applied last.
    this.active()?.updateExpressions(model, deltaSeconds);
    this.persistent()?.updateExpressions(model, deltaSeconds);
  }
}
