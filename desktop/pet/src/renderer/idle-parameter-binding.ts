import type { CubismModel } from "@cubism/model/cubismmodel";
import type { CubismPhysics } from "@cubism/physics/cubismphysics";
import type { ExpressiveIdleSettings, IdleChannel } from "../shared/contracts";
export interface IdleBinding {
  id: string;
  index: number;
  min: number;
  max: number;
  base: number;
}
export interface IdleRig {
  channels: Partial<Record<IdleChannel, IdleBinding>>;
  visibleX?: IdleBinding;
  visibleY?: IdleBinding;
  visibleZ?: IdleBinding;
  directions: Record<"x" | "y" | "z", number>;
  breath?: IdleBinding;
  browL?: IdleBinding;
  browR?: IdleBinding;
}
/** Only resolve real Core IDs: an unknown ID must never create a virtual parameter. */
export function resolveIdleRig(model: CubismModel, physics: CubismPhysics | null, settings: ExpressiveIdleSettings): IdleRig {
  const raw = model.getModel().parameters;
  const binding = (id: string): IdleBinding | undefined => {
    const index = raw.ids.indexOf(id);
    if (index < 0)
      return undefined;
    return { id, index, min: raw.minimumValues[index]!, max: raw.maximumValues[index]!, base: raw.defaultValues[index]! };
  };
  const outputs = physics?._physicsRig?.outputs ?? [];
  const isOutput = (id: string) => outputs.some(output => output.destination.id.getString() === id);
  const channels: IdleRig["channels"] = {};
  const names: Record<IdleChannel, string> = { headX: "ParamAngleX", headY: "ParamAngleY", headZ: "ParamAngleZ",
    bodyX: "ParamBodyAngleX", bodyY: "ParamBodyAngleY", bodyZ: "ParamBodyAngleZ",
    eyeX: "ParamEyeBallX", eyeY: "ParamEyeBallY", eyeLOpen: "ParamEyeLOpen", eyeROpen: "ParamEyeROpen" };
  const assigned = new Set<number>();
  for (const key of Object.keys(names) as IdleChannel[]) {
    const override = settings.bindings?.[key];
    if (override === null)
      continue;
    const name = override ?? (key.startsWith("body") && binding(names[key] + "1") && isOutput(names[key] + "1") ? names[key] + "1" : names[key]);
    const item = binding(name);
    // Conflicting manual assignments cannot result in two drivers fighting over one parameter.
    if (item && !assigned.has(item.index)) {
      channels[key] = item;
      assigned.add(item.index);
    }
  }
  const visible = (axis: "X" | "Y" | "Z") => {
    const head = channels[axis === "X" ? "headX" : axis === "Y" ? "headY" : "headZ"];
    return head && isOutput(head.id + "1") ? binding(head.id + "1") ?? head : head;
  };
  const directions: IdleRig["directions"] = { x: 1, y: 1, z: 1 };
  for (const [axis, key] of [["x", "bodyX"], ["y", "bodyY"], ["z", "bodyZ"]] as const) {
    const body = channels[key];
    // Separate authored body outputs often have the opposite sign from their head input.
    const output = body && outputs.filter(output => output.destination.id.getString() === body.id).at(-1);
    directions[axis] = settings.bodyDirections?.[axis] ?? (output?.reflect ? -1 : 1);
  }
  const extra = (id: string) => { const item = binding(id); return item && !assigned.has(item.index) ? item : undefined; };
  return { channels, visibleX: visible("X"), visibleY: visible("Y"), visibleZ: visible("Z"), directions,
    breath: extra("ParamBreath"), browL: extra("ParamBrowLY"), browR: extra("ParamBrowRY") };
}
export function centeredValue(binding: IdleBinding, normalized: number): number {
  const extent = normalized < 0 ? binding.base - binding.min : binding.max - binding.base;
  return binding.base + Math.max(-1, Math.min(1, normalized)) * extent;
}
/** Weights belong to this in-memory Physics instance and are restored on disable/release. */
export class IdlePhysicsOwnership {
  private readonly outputs: Array<{
    id: string;
    original: number;
    value: {
      weight: number;
    };
  }>;
  constructor(physics: CubismPhysics | null) {
    this.outputs = (physics?._physicsRig?.outputs ?? []).map(output => ({ id: output.destination.id.getString(), original: output.weight, value: output }));
  }
  restore(): void {
    for (const output of this.outputs)
      output.value.weight = output.original;
  }
  reserve(id: string, strength = 1): void {
    for (const output of this.outputs)
      if (output.id === id)
        output.value.weight = Math.min(output.value.weight, output.original * (1 - strength));
  }
}
