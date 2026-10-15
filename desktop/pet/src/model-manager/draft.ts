import {
  PET_BEHAVIOR_STATES,
  PET_EMOTIONS,
  type BehaviorCandidate,
  type ModelPackage,
  type PetManifest,
} from "../shared/contracts";
import { normalizeDefaultAppearance, normalizeOutfits } from "../shared/default-appearance";
import { normalizePartGroups } from "../shared/part-groups";

export type BehaviorKind = "state" | "emotion";
export type AssetKind = "motion" | "expression";

export interface ModelDraft {
  manifest: PetManifest;
  readonly savedManifest: PetManifest;
}

export function createDraft(model: ModelPackage): ModelDraft {
  return {
    manifest: normalizeManifest(structuredClone(model.manifest)),
    savedManifest: normalizeManifest(structuredClone(model.manifest)),
  };
}

export function behaviorCandidates(
  manifest: PetManifest,
  kind: BehaviorKind,
  key: string,
): BehaviorCandidate[] {
  const map = kind === "state" ? manifest.behaviors.states : manifest.behaviors.emotions;
  const candidates = (map as Record<string, BehaviorCandidate[]>)[key];
  if (!candidates) throw new Error(`未知的${kind === "state" ? "状态" : "情绪"}：${key}`);
  return candidates;
}

export function saveBehaviorCandidate(
  draft: ModelDraft,
  kind: BehaviorKind,
  key: string,
  candidate: BehaviorCandidate,
): void {
  validateCandidate(draft.manifest, candidate);
  const values = behaviorCandidates(draft.manifest, kind, key);
  const index = values.findIndex((value) => value.id === candidate.id);
  const saved = structuredClone(candidate);
  if (index < 0) values.push(saved);
  else values[index] = saved;
}

export function deleteBehaviorCandidate(
  draft: ModelDraft,
  kind: BehaviorKind,
  key: string,
  candidateId: string,
): void {
  const values = behaviorCandidates(draft.manifest, kind, key);
  const index = values.findIndex((candidate) => candidate.id === candidateId);
  if (index >= 0) values.splice(index, 1);
}

export function renameAsset(
  draft: ModelDraft,
  kind: AssetKind,
  assetId: string,
  name: string,
): void {
  const assets = kind === "motion"
    ? draft.manifest.motionAssets
    : draft.manifest.expressionAssets;
  const asset = assets[assetId];
  if (!asset) throw new Error(`素材不存在：${assetId}`);
  const normalized = name.trim().replace(/\s+/g, " ").slice(0, 80);
  if (!normalized) throw new Error("素材名称不能为空");
  asset.name = normalized;
}

export function hasUnsavedChanges(draft: ModelDraft): boolean {
  return canonical(draft.manifest) !== canonical(draft.savedManifest);
}

export function isDraftValid(draft: ModelDraft): boolean {
  const { manifest } = draft;
  const lipSyncValid = manifest.lipsync.mode === "viseme"
    ? Object.values(manifest.lipsync.vowelParameters).every(Boolean)
    : manifest.lipsync.mode === "open-form"
      ? Boolean(manifest.lipsync.mouthOpenParameter?.trim())
      : false;
  return Boolean(
    manifest.name.trim() &&
    lipSyncValid &&
    finiteBetween(manifest.display.scale, 0.05, 5) &&
    finiteBetween(manifest.display.x, 0, 1) &&
    finiteBetween(manifest.display.y, 0, 1) &&
    finiteBetween(manifest.lipsync.maxOpen, 0, 1),
  );
}

export function toManifest(draft: ModelDraft): PetManifest {
  return normalizeManifest(structuredClone(draft.manifest));
}

function normalizeManifest(manifest: PetManifest): PetManifest {
  if (manifest.defaultAppearance !== undefined) {
    manifest.defaultAppearance = normalizeDefaultAppearance(manifest.defaultAppearance, manifest.expressionAssets);
  }
  if (manifest.outfits !== undefined) {
    manifest.outfits = normalizeOutfits(manifest.outfits, manifest.expressionAssets);
    if (!manifest.outfits.length) delete manifest.outfits;
  }
  if (manifest.partGroups !== undefined) manifest.partGroups = normalizePartGroups(manifest.partGroups);
  const motionIds = new Set(Object.keys(manifest.motionAssets));
  const expressionIds = new Set(Object.keys(manifest.expressionAssets));
  normalizeCandidateMap(manifest.behaviors.states, PET_BEHAVIOR_STATES, motionIds, expressionIds);
  normalizeCandidateMap(manifest.behaviors.emotions, PET_EMOTIONS, motionIds, expressionIds);
  return manifest;
}

function normalizeCandidateMap(
  map: Record<string, BehaviorCandidate[]>,
  keys: readonly string[],
  motionIds: Set<string>,
  expressionIds: Set<string>,
): void {
  const used = new Set<string>();
  for (const key of keys) {
    map[key] = (map[key] ?? []).flatMap((candidate) => {
      const valid = Boolean(
        candidate.id &&
        !used.has(candidate.id) &&
        candidate.motionAssetIds.length + candidate.expressionAssetIds.length > 0 &&
        candidate.motionAssetIds.every((id) => motionIds.has(id)) &&
        candidate.expressionAssetIds.every((id) => expressionIds.has(id)),
      );
      if (!valid) return [];
      used.add(candidate.id);
      return [{
        id: candidate.id,
        motionAssetIds: [...new Set(candidate.motionAssetIds)],
        expressionAssetIds: [...new Set(candidate.expressionAssetIds)],
      }];
    });
  }
}

function validateCandidate(manifest: PetManifest, candidate: BehaviorCandidate): void {
  if (!candidate.id.trim()) throw new Error("候选 ID 不能为空");
  if (!candidate.motionAssetIds.length && !candidate.expressionAssetIds.length) {
    throw new Error("候选至少需要一个 Motion 或 Expression 素材");
  }
  for (const id of candidate.motionAssetIds) {
    if (!manifest.motionAssets[id]) throw new Error(`动作素材不存在：${id}`);
  }
  for (const id of candidate.expressionAssetIds) {
    if (!manifest.expressionAssets[id]) throw new Error(`表情素材不存在：${id}`);
  }
}

function canonical(value: unknown): string {
  return JSON.stringify(value);
}

function finiteBetween(value: number, minimum: number, maximum: number): boolean {
  return Number.isFinite(value) && value >= minimum && value <= maximum;
}
