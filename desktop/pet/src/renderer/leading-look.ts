import type { CubismLook } from "@cubism/effect/cubismlook";
import type { CubismTargetPoint } from "@cubism/math/cubismtargetpoint";
import type { CubismModel } from "@cubism/model/cubismmodel";
import { CubismUpdateOrder, ICubismUpdater } from "@cubism/motion/icubismupdater";

/** The eyes reach a new target in a few frames, as a saccade does. */
const EYE_SECONDS = 0.04;
/**
 * How much further than their final turn the eyes look while the head is
 * still behind: the eyes and head share the turn, so as the head arrives the
 * eyes ease back to where they rest now.
 */
const LEAD = 0.6;
const clamp = (value: number) => Math.max(-1, Math.min(1, value));

/**
 * Live2D's look-at with the eyes leading, as a person's do: they jump to the
 * target first, the head follows at the drag's own pace, and the eyes ease
 * back as it arrives. Once still, the pose is the same as the SDK's.
 */
export class LeadingLookUpdater extends ICubismUpdater {
  private target = { x: 0, y: 0 };
  private eyes = { x: 0, y: 0 };

  constructor(private readonly head: CubismLook, private readonly eyeLook: CubismLook, private readonly drag: CubismTargetPoint) {
    super(CubismUpdateOrder.CubismUpdateOrder_Drag);
  }

  setTarget(x: number, y: number): void {
    this.target = { x: clamp(x), y: clamp(y) };
    this.drag.set(this.target.x, this.target.y);
  }

  onLateUpdate(model: CubismModel, delta: number): void {
    if (!model) return;
    this.drag.update(delta);
    const headX = this.drag.getX(), headY = this.drag.getY();
    const step = 1 - Math.exp(-Math.max(0, delta) / EYE_SECONDS);
    this.eyes.x += (this.target.x - this.eyes.x) * step;
    this.eyes.y += (this.target.y - this.eyes.y) * step;
    this.head.updateParameters(model, headX, headY);
    this.eyeLook.updateParameters(model, clamp(this.eyes.x + LEAD * (this.eyes.x - headX)), clamp(this.eyes.y + LEAD * (this.eyes.y - headY)));
  }
}
