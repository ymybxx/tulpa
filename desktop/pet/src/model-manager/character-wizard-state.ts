import {
  resolveTextureEditImageModel,
  type TextureEditImageModel,
  type CharacterGenerationJob,
  type TexturePromptOptimizationState,
  type TextureReferenceDeliveryDecision,
  type TextureReferencePromptId,
  type TextureVisualAttributeCategory,
  type TextureVisualEditSpec,
} from "../shared/character-generation";

export type CharacterWizardStep =
  | "generating-candidates"
  | "choose-candidate"
  | "processing"
  | "budget"
  | "parts"
  | "confirm-base-preview"
  | "confirm-preview"
  | "done"
  | "failed";

export function characterPreviewModelId(job: CharacterGenerationJob): string | null {
  if (job.input.operation === "texture-edit") return job.input.sourceModelId;
  return job.outputModelId ?? job.input.sourceModelId;
}

export function characterWizardStep(job: CharacterGenerationJob): CharacterWizardStep {
  if (job.paused && job.pauseReason === "budget") return "budget";
  if (job.paused && job.pauseReason === "needs-review") return "parts";
  if (job.phase === "base-preview") return "confirm-base-preview";
  if (job.phase === "preview") return "confirm-preview";
  if (job.phase === "saved") return "done";
  if (job.phase === "failed" || job.phase === "cancelled") return "failed";
  if (job.phase === "draft") {
    return job.candidates.length > 0 ? "choose-candidate" : "generating-candidates";
  }
  return "processing";
}

export type TexturePromptSubmissionChoice =
  | "current"
  | "pending"
  | "optimized";

export function texturePromptSubmissionChoice(
  optimizedPrompt: string | null,
  state: TexturePromptOptimizationState,
): TexturePromptSubmissionChoice {
  if (!optimizedPrompt) return "current";
  if (state === "accepted") return "optimized";
  if (state === "rejected") return "current";
  return "pending";
}

export interface TextureReferenceDeliveryState extends TextureReferenceDeliveryDecision {
  send: boolean;
  locked: boolean;
}

export function textureReferenceDeliveryState(
  useOptimizedPrompt: boolean,
  available: ReadonlyArray<{ selectionId: string; promptId: TextureReferencePromptId }>,
  optionalOverrides: readonly string[],
): TextureReferenceDeliveryState[] {
  if (!useOptimizedPrompt) {
    return available.map((reference) => ({
      ...reference,
      required: true,
      send: true,
      locked: true,
      appearanceScope: null,
      allowedAttributes: [],
      regionKeys: [],
    }));
  }
  const selected = new Set(optionalOverrides.length > 0
    ? optionalOverrides
    : available.map((reference) => reference.selectionId));
  return available.map((reference) => ({
    ...reference,
    required: false,
    send: selected.has(reference.selectionId),
    locked: false,
    appearanceScope: null,
    allowedAttributes: [],
    regionKeys: [],
  }));
}

export interface TextureAdjustmentReturnState {
  selectedTextureUnitIds: string[];
  savedTextureUnitIds: string[];
  originalPrompt: string;
  imageModel: TextureEditImageModel;
}

export function textureAdjustmentReturnState(
  job: CharacterGenerationJob,
): TextureAdjustmentReturnState {
  if (job.input.operation !== "texture-edit") {
    throw new Error("只有纹理编辑任务可以返回调整");
  }
  const textureUnitIds = [...new Set(job.input.textureUnitIds ?? [])];
  const savedTextureUnitIds = textureUnitIds.filter((id) =>
    job.artifacts.textureEdits?.[id]?.status === "accepted"
  );
  const saved = new Set(savedTextureUnitIds);
  return {
    selectedTextureUnitIds: savedTextureUnitIds.length === 0
      ? textureUnitIds
      : textureUnitIds.filter((id) => !saved.has(id)),
    savedTextureUnitIds,
    originalPrompt: job.input.originalDescription?.trim() || job.input.description,
    imageModel: resolveTextureEditImageModel(job.input.textureImageModel),
  };
}

const VISUAL_CATEGORY_LABELS: Record<TextureVisualAttributeCategory, string> = {
  palette: "配色",
  rendering: "渲染",
  shading: "光影",
  linework: "线稿",
  material: "材质",
  motif: "图案",
  tone: "整体气质",
};

export function texturePromptReview(spec: TextureVisualEditSpec) {
  return {
    attributes: spec.styleProfiles.flatMap((profile) =>
      (Object.entries(profile.attributes) as Array<[
        TextureVisualAttributeCategory,
        NonNullable<(typeof profile.attributes)[TextureVisualAttributeCategory]>,
      ]>).map(([category, attribute]) => ({
        profileId: profile.id,
        scope: profile.scope,
        category: VISUAL_CATEGORY_LABELS[category],
        description: attribute.description,
      }))
    ),
    regions: spec.regions.map((region) => ({
      key: region.key,
      role: region.role,
      profileIds: region.styleProfileIds,
      clauseIds: region.clauseIds,
    })),
  };
}
