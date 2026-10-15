import type { CubismModel } from "@cubism/model/cubismmodel";
import { CubismUpdateOrder, ICubismUpdater } from "@cubism/motion/icubismupdater";
import type { NaturalIdleSettings } from "../shared/contracts";
import { NATURAL_IDLE_INPUT_PARAMETERS, NaturalIdleMotion } from "./natural-idle-motion";

/** Write rig inputs before physics: models such as 悠小喵 have separate output IDs. */
export class CubismNaturalIdleUpdater extends ICubismUpdater {
  private readonly motion: NaturalIdleMotion;
  private readonly indices: number[];

  constructor(model: CubismModel, private readonly available: () => boolean, random?: () => number) {
    super((CubismUpdateOrder.CubismUpdateOrder_Breath + CubismUpdateOrder.CubismUpdateOrder_Physics) / 2);
    this.motion = new NaturalIdleMotion(random);
    // Looking up a missing Cubism ID creates a virtual parameter. Only touch real rig inputs.
    const ids = model.getModel().parameters.ids;
    this.indices = NATURAL_IDLE_INPUT_PARAMETERS.map((id) => ids.indexOf(id));
  }

  configure(value: NaturalIdleSettings | undefined): void {
    this.motion.configure(value);
  }

  onLateUpdate(model: CubismModel, deltaSeconds: number): void {
    const pose = this.motion.update(deltaSeconds, this.available());
    this.add(model, this.indices[0]!, pose.x);
    this.add(model, this.indices[1]!, pose.y);
    this.add(model, this.indices[2]!, pose.z);
  }

  private add(model: CubismModel, index: number, offset: number): void {
    if (index < 0 || offset === 0) return;
    // Cubism clamps to each parameter's actual min/max here.
    model.setParameterValueByIndex(index, model.getParameterValueByIndex(index) + offset);
  }
}
