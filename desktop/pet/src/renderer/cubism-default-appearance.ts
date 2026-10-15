import type { CubismModel } from "@cubism/model/cubismmodel";
import type { ACubismMotion } from "@cubism/motion/acubismmotion";
import { CubismExpressionMotion, ExpressionBlendType } from "@cubism/motion/cubismexpressionmotion";
import { CubismUpdateOrder, ICubismUpdater } from "@cubism/motion/icubismupdater";

/** Fixed values compiled from authored switches, never accumulated into saved motion state. */
export class CubismDefaultAppearance extends ICubismUpdater {
  private values: ReadonlyMap<number, number> = new Map();
  private parameterIds: ReadonlySet<string> = new Set();

  constructor(private readonly model: CubismModel) {
    // After automatic head inputs, just before physics consumes them.
    super(CubismUpdateOrder.CubismUpdateOrder_Physics - 1);
  }

  configure(expressions: readonly ACubismMotion[], parameterValues: Readonly<Record<string, number>> = {}): void {
    const ids = this.model.getModel().parameters.ids;
    const values = new Map<number, number>();
    const parameterIds = new Set<string>();
    for (const expression of expressions) {
      const getter = (expression as CubismExpressionMotion).getExpressionParameters;
      if (typeof getter !== "function") throw new Error("默认外观素材不是可读取的 Expression");
      const local = new Map<number, number>();
      for (const parameter of getter.call(expression)) {
        const id = parameter.parameterId.getString();
        const index = ids.indexOf(id);
        if (index < 0 || !Number.isFinite(parameter.value)) continue;
        const base = local.get(index) ?? this.model.getParameterDefaultValue(index);
        const value = parameter.blendType === ExpressionBlendType.Overwrite
          ? parameter.value
          : parameter.blendType === ExpressionBlendType.Multiply
            ? base * parameter.value
            : base + parameter.value;
        local.set(index, Math.max(this.model.getParameterMinimumValue(index), Math.min(this.model.getParameterMaximumValue(index), value)));
        parameterIds.add(id);
      }
      // Each switch is evaluated against the authored baseline. Later switches
      // win on shared channels instead of adding incompatible variants together.
      for (const [index, value] of local) values.set(index, value);
    }
    // Compile only explicit overrides. Never create Cubism virtual parameters
    // for stale IDs, and let manual choices win over overlapping expressions.
    for (const [id, value] of Object.entries(parameterValues)) {
      const index = ids.indexOf(id);
      if (index < 0 || !Number.isFinite(value)) continue;
      values.set(index, Math.max(this.model.getParameterMinimumValue(index), Math.min(this.model.getParameterMaximumValue(index), value)));
      parameterIds.add(id);
    }
    this.values = values;
    this.parameterIds = parameterIds;
  }

  affects(parameters: ReadonlySet<string>): boolean {
    for (const id of this.parameterIds) if (parameters.has(id)) return true;
    return false;
  }

  parameters(): ReadonlySet<string> { return this.parameterIds; }

  onLateUpdate(model: CubismModel): void {
    this.apply(model);
  }

  /** Also enforce after physics/lipsync so authored clips cannot flash a hidden accessory. */
  apply(model: CubismModel): void {
    for (const [index, value] of this.values) model.setParameterValueByIndex(index, value);
  }
}
