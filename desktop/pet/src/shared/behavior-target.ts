import { PET_BEHAVIOR_STATES, PET_EMOTIONS, type BehaviorCandidate, type PetBehaviorState, type PetEmotion, type PetManifest } from "./contracts";

export type BehaviorTarget = { kind: "state"; key: PetBehaviorState } | { kind: "emotion"; key: PetEmotion };

export function isBehaviorTarget(value: unknown): value is BehaviorTarget {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const target = value as Record<string, unknown>;
  if (Object.keys(target).some((key) => key !== "kind" && key !== "key")) return false;
  return target.kind === "state" ? PET_BEHAVIOR_STATES.includes(target.key as PetBehaviorState)
    : target.kind === "emotion" && PET_EMOTIONS.includes(target.key as PetEmotion);
}

export function targetId(target: BehaviorTarget): string { return `${target.kind}:${target.key}`; }

export function targetCandidates(manifest: PetManifest, target: BehaviorTarget): BehaviorCandidate[] {
  return target.kind === "state" ? manifest.behaviors.states[target.key] : manifest.behaviors.emotions[target.key];
}

export function targetUsesStatePlayback(target: BehaviorTarget): boolean {
  return target.kind === "state" && target.key !== "click";
}
