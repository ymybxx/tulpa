import { describe, expect, it } from "vitest";

import type {
  CharacterGenerationJob,
  TextureReferencePromptId,
} from "../shared/character-generation";
import {
  characterPreviewModelId,
  characterWizardStep,
  textureAdjustmentReturnState,
  textureReferenceDeliveryState,
  texturePromptReview,
  texturePromptSubmissionChoice,
} from "./character-wizard-state";

describe("characterPreviewModelId", () => {
  it("locks texture acceptance previews to the task source model", () => {
    expect(characterPreviewModelId({
      ...base(),
      input: {
        operation: "texture-edit",
        description: "edit stockings",
        referenceImage: null,
        sourceModelId: "task-source-model",
        textureUnitIds: ["left-leg", "right-leg"],
      },
      outputModelId: "outside-selected-model",
    })).toBe("task-source-model");
  });

  it("uses the generated model for non-texture preview tasks", () => {
    expect(characterPreviewModelId({
      ...base(),
      input: {
        operation: "part-restyle",
        description: "edit eyes",
        referenceImage: null,
        sourceModelId: "task-source-model",
        editPartId: "eyes",
      },
      outputModelId: "generated-model",
    })).toBe("generated-model");
  });
});

describe("characterWizardStep", () => {
  it("keeps candidate selection and Live2D preview as two explicit confirmations", () => {
    const draft = base();
    expect(characterWizardStep({ ...draft, candidates: [{
      id: "a",
      localPath: "/a",
      seed: null,
      compatible: true,
      rejectionReason: null,
    }] }))
      .toBe("choose-candidate");
    expect(characterWizardStep({ ...draft, phase: "preview", outputModelId: "model" }))
      .toBe("confirm-preview");
    expect(characterWizardStep({ ...draft, phase: "base-preview", outputModelId: "base-model" }))
      .toBe("confirm-base-preview");
  });

  it("surfaces budget and part-review pauses", () => {
    expect(characterWizardStep({ ...base(), paused: true, pauseReason: "budget" })).toBe("budget");
    expect(characterWizardStep({ ...base(), paused: true, pauseReason: "needs-review" })).toBe("parts");
  });
});

describe("texturePromptSubmissionChoice", () => {
  it("allows the current prompt before optimization is requested", () => {
    expect(texturePromptSubmissionChoice(null, "not-requested")).toBe("current");
  });

  it("requires an explicit choice after an optimization suggestion arrives", () => {
    expect(texturePromptSubmissionChoice("optimized", "not-requested")).toBe("pending");
  });

  it("records which prompt version will be submitted", () => {
    expect(texturePromptSubmissionChoice("optimized", "accepted")).toBe("optimized");
    expect(texturePromptSubmissionChoice("optimized", "rejected")).toBe("current");
  });
});

describe("textureReferenceDeliveryState", () => {
  it("sends all selected references by default for an accepted optimization", () => {
    expect(textureReferenceDeliveryState(true, [identity("front", "P0"), identity("upload", "R1")], [])).toEqual([
      expect.objectContaining({ selectionId: "front", promptId: "P0", required: false, send: true, locked: false }),
      expect.objectContaining({ selectionId: "upload", promptId: "R1", required: false, send: true, locked: false }),
    ]);
  });

  it("allows the user to keep only a chosen reference", () => {
    const state = textureReferenceDeliveryState(true, [identity("front", "P0"), identity("upload", "R1")], ["upload"]);
    expect(state).toEqual([
      expect.objectContaining({ selectionId: "front", send: false, locked: false }),
      expect.objectContaining({ selectionId: "upload", send: true, locked: false }),
    ]);
  });

  it("sends every selected reference when the optimized prompt is not used", () => {
    const state = textureReferenceDeliveryState(false, [
      identity("front", "P0"),
      identity("upload", "R1"),
    ], []);
    expect(state).toEqual([
      expect.objectContaining({ selectionId: "front", required: true, send: true, locked: true }),
      expect.objectContaining({ selectionId: "upload", required: true, send: true, locked: true }),
    ]);
  });
});

describe("textureAdjustmentReturnState", () => {
  it("defaults to retrying only textures that were not saved", () => {
    const state = textureAdjustmentReturnState({
      ...base(),
      input: {
        operation: "texture-edit",
        description: "优化后的提示词",
        originalDescription: "用户原始提示词",
        referenceImage: null,
        sourceModelId: "model",
        textureUnitIds: ["body", "hair", "body"],
        textureImageModel: "gpt-image-1.5",
      },
      artifacts: {
        ...base().artifacts,
        textureEdits: {
          body: textureResult("body", "accepted"),
          hair: textureResult("hair", "failed"),
        },
      },
    });

    expect(state).toEqual({
      selectedTextureUnitIds: ["hair"],
      savedTextureUnitIds: ["body"],
      originalPrompt: "用户原始提示词",
      imageModel: "gpt-image-2.5-sunburst",
    });
  });

  it("returns the full original selection when nothing was accepted", () => {
    const state = textureAdjustmentReturnState({
      ...base(),
      input: {
        operation: "texture-edit",
        description: "当前提示词",
        referenceImage: null,
        sourceModelId: "model",
        textureUnitIds: ["body", "hair"],
      },
      artifacts: {
        ...base().artifacts,
        textureEdits: {
          body: textureResult("body", "review"),
          hair: textureResult("hair", "failed"),
        },
      },
    });

    expect(state.selectedTextureUnitIds).toEqual(["body", "hair"]);
    expect(state.savedTextureUnitIds).toEqual([]);
    expect(state.originalPrompt).toBe("当前提示词");
    expect(state.imageModel).toBe("gpt-image-2.5-sunburst");
  });
});

describe("texturePromptReview", () => {
  it("presents visual attributes and region mappings", () => {
    const review = texturePromptReview({
      intent: "style-transfer",
      styleProfiles: [{
        id: "S1",
        scope: "粉色小熊风格",
        attributes: {
          palette: { description: "低饱和浅粉" },
          motif: { description: "小熊与爱心" },
        },
      }],
      regions: [{
        key: "A",
        role: "左上臂袖子",
        styleProfileIds: ["S1"],
        clauseIds: ["C1", "C3"],
        continuityGroup: "sleeves",
      }],
      warnings: [],
    });

    expect(review.attributes).toEqual([
      { profileId: "S1", scope: "粉色小熊风格", category: "配色", description: "低饱和浅粉" },
      { profileId: "S1", scope: "粉色小熊风格", category: "图案", description: "小熊与爱心" },
    ]);
    expect(review.regions).toEqual([{
      key: "A",
      role: "左上臂袖子",
      profileIds: ["S1"],
      clauseIds: ["C1", "C3"],
    }]);
  });
});

function base(): CharacterGenerationJob {
  return {
    version: 2,
    id: "job",
    templateId: "template",
    input: { description: "girl", referenceImage: null, sourceModelId: null },
    phase: "draft",
    paused: false,
    pauseReason: null,
    revision: 0,
    candidates: [],
    selectedCandidateId: null,
    falRequests: [],
    cache: {},
    artifacts: {
      normalizedImage: null,
      subjectImage: null,
      coarseLayers: [],
      completionSheets: {},
      slotTextures: {},
      repairSlotIds: [],
      requestedCompletions: [],
    },
    budget: { limitUsd: 0.8, estimatedUsd: 0, actualUsd: 0, pendingOperation: null },
    issues: [],
    outputModelId: null,
    createdAt: "",
    updatedAt: "",
  };
}

function textureResult(
  textureUnitId: string,
  status: NonNullable<CharacterGenerationJob["artifacts"]["textureEdits"]>[string]["status"],
): NonNullable<CharacterGenerationJob["artifacts"]["textureEdits"]>[string] {
  return {
    textureUnitId,
    status,
    requestId: null,
    endpoint: null,
    estimatedUsd: 0,
    sourcePath: null,
    sourcePreviewPath: null,
    workImagePath: null,
    workMaskPath: null,
    rawResultPath: null,
    editedPatchPath: status === "accepted" ? `/tmp/${textureUnitId}.png` : null,
    editedPreviewPath: null,
    layout: null,
    quality: null,
    error: null,
  };
}

function identity(selectionId: string, promptId: TextureReferencePromptId) {
  return { selectionId, promptId };
}
