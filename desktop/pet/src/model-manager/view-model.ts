import {
  PET_BEHAVIOR_STATES,
  PET_EMOTIONS,
  type BehaviorCandidate,
  type ModelEditCapabilityManifest,
  type PetManifest,
  type TextureEditUnit,
} from "../shared/contracts";
import type { BehaviorKind } from "./draft";

export interface BehaviorRow {
  key: string;
  label: string;
  candidates: BehaviorCandidate[];
}

export function initialTextureFilterValue(requestedPartId: string | null): string {
  return requestedPartId ?? "";
}

export function textureUnitsForPart(
  capability: ModelEditCapabilityManifest,
  partId: string,
): TextureEditUnit[] {
  return capability.textureUnits.filter((unit) =>
    !partId || unit.semanticPartIds.includes(partId)
  );
}

export const STATE_LABELS: Record<string, string> = {
  standby: "睡眠（未唤醒）",
  acknowledging: "回应唤醒",
  listening: "聆听（已唤醒）",
  thinking: "思考",
  click: "点击",
};

export const EMOTION_LABELS: Record<string, string> = {
  neutral: "平静",
  angry: "生气",
  happy: "开心",
  shy: "害羞",
  sad: "伤心",
};

export function buildBehaviorRows(
  manifest: PetManifest,
  kind: BehaviorKind,
): BehaviorRow[] {
  const keys = kind === "state" ? PET_BEHAVIOR_STATES : PET_EMOTIONS;
  const labels = kind === "state" ? STATE_LABELS : EMOTION_LABELS;
  const map = kind === "state" ? manifest.behaviors.states : manifest.behaviors.emotions;
  return keys.map((key) => ({
    key,
    label: labels[key] ?? key,
    candidates: structuredClone((map as Record<string, BehaviorCandidate[]>)[key] ?? []),
  }));
}
