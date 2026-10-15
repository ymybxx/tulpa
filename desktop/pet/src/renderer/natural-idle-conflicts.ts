import type { ACubismMotion } from "@cubism/motion/acubismmotion";
import type { CubismExpressionMotion } from "@cubism/motion/cubismexpressionmotion";
import { CubismPhysicsJson } from "@cubism/physics/cubismphysicsjson";
import { NATURAL_IDLE_INPUT_PARAMETERS } from "./natural-idle-motion";

interface PhysicsParameterLinks {
  inputs: readonly string[];
  outputs: readonly string[];
}

/** Follow the real physics connections, including custom intermediate parameters. */
export function naturalIdleAffectedParameters(links: readonly PhysicsParameterLinks[]): ReadonlySet<string> {
  const affected = new Set<string>(NATURAL_IDLE_INPUT_PARAMETERS);
  const outputsByInput = new Map<string, Set<string>>();
  for (const link of links) {
    for (const input of link.inputs) {
      const outputs = outputsByInput.get(input) ?? new Set<string>();
      for (const output of link.outputs) outputs.add(output);
      outputsByInput.set(input, outputs);
    }
  }
  const pending = [...affected];
  for (let index = 0; index < pending.length; index++) {
    for (const output of outputsByInput.get(pending[index]!) ?? []) {
      if (affected.has(output)) continue;
      affected.add(output);
      pending.push(output);
    }
  }
  return affected;
}

/** Computed once while loading the model, never in the frame loop. */
export function readNaturalIdlePhysicsParameters(buffer: ArrayBuffer): ReadonlySet<string> {
  const json = new CubismPhysicsJson(buffer, buffer.byteLength);
  try {
    const links: PhysicsParameterLinks[] = [];
    for (let rig = 0; rig < json.getSubRigCount(); rig++) {
      const inputs: string[] = [];
      const outputs: string[] = [];
      for (let i = 0; i < json.getInputCount(rig); i++) {
        if (json.getInputWeight(rig, i) !== 0) inputs.push(json.getInputSourceId(rig, i).getString());
      }
      for (let i = 0; i < json.getOutputCount(rig); i++) {
        if (json.getOutputWeight(rig, i) !== 0) outputs.push(json.getOutputDestinationId(rig, i).getString());
      }
      links.push({ inputs, outputs });
    }
    return naturalIdleAffectedParameters(links);
  } finally {
    json.release();
  }
}

/** Static appearance switches do not own the pose unless they write affected parameters. */
export function expressionBlocksNaturalIdle(
  expression: ACubismMotion,
  affected: ReadonlySet<string>,
): boolean {
  const getter = (expression as CubismExpressionMotion).getExpressionParameters;
  if (typeof getter !== "function") return true;
  try {
    const parameters = getter.call(expression);
    if (!Array.isArray(parameters)) return true;
    return parameters.some(({ parameterId }) => {
      const id = parameterId?.getString();
      return typeof id !== "string" || affected.has(id);
    });
  } catch {
    return true;
  }
}
