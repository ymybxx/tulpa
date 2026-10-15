import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { CharacterGenerationProvider } from "../../src/shared/character-generation";
import type { ModelPackage } from "../../src/shared/contracts";
import { CharacterArtifactStore } from "./artifact-store";
import { CharacterGenerationJobStore } from "./job-store";
import {
  ALL_MATERIALS_REPAIR_ID,
  CharacterGenerationService,
  REALIGN_TO_TEMPLATE_ID,
} from "./service";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("CharacterGenerationService", () => {
  it.each([
    ["recolor", "locked"],
    ["alpha-remove", "shrink-only"],
  ] as const)("applies the accepted %s intent with the same Image 2.5 model", async (intent, alphaPolicy) => {
    const harness = await createHarness();
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body"]);
    if (!plan.fits) throw new Error(plan.reason);
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: intent === "alpha-remove" ? "去掉袜口" : "袜子改成粉色",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body"],
      textureBatchPlanHash: plan.plan.hash,
      promptOptimization: "accepted",
      promptVisualSpec: { intent, styleProfiles: [], regions: [], warnings: [] },
    });

    const preview = await harness.service.generateCandidates(job.id);

    expect(preview.phase).toBe("preview");
    expect(harness.provider.editMaskedTexture).toHaveBeenCalledWith(expect.objectContaining({
      imageModel: "gpt-image-2.5-sunburst",
    }));
    const calls = vi.mocked(harness.artifacts.constrainTextureEditResult).mock.calls;
    expect(calls).toHaveLength(1);
    expect(calls[0]![7]).toBe(alphaPolicy);
  });

  it("uses one GPT request for a jointly packed selection and splits independent results", async () => {
    const harness = await createHarness({
      editMaskedTexture: vi.fn(async () => ({
        requestId: "texture-batch",
        urls: ["https://cdn/texture-batch.png"],
      })),
    });
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body", "texture-hair"]);
    if (!plan.fits) throw new Error(plan.reason);
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: "改成干净的淡粉色赛璐璐纹理",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body", "texture-hair"],
      textureBatchPlanHash: plan.plan.hash,
      textureReferences: [{
        id: "reference-upload",
        source: "upload",
        name: "粉色卫衣",
        description: "参考小熊与爱心排布",
        path: "/uploaded/pink-hoodie.png",
        promptId: "R2",
        deliveryUse: "required",
      }],
    });

    const preview = await harness.service.generateCandidates(job.id);

    expect(preview.phase).toBe("preview");
    expect(harness.provider.editMaskedTexture).toHaveBeenCalledTimes(1);
    expect(harness.provider.inpaintImage).not.toHaveBeenCalled();
    expect(preview.budget.estimatedUsd).toBe(plan.plan.estimatedUsd);
    expect(preview.artifacts.textureEdits).toMatchObject({
      "texture-body": { status: "review", endpoint: "openai/gpt-image-2.5/sunburst/edit" },
      "texture-hair": { status: "review", endpoint: "openai/gpt-image-2.5/sunburst/edit" },
    });
    expect(harness.provider.editMaskedTexture).toHaveBeenCalledWith(expect.objectContaining({
      imageModel: "gpt-image-2.5-sunburst",
      referenceImagePaths: ["/local/reference-01.png"],
    }));
    const submittedPrompt = vi.mocked(harness.provider.editMaskedTexture).mock.calls[0]![0].prompt;
    expect(submittedPrompt).not.toContain("界面名称为");
    expect(submittedPrompt).toContain("具体部位语义只采用最终提示词中的视觉识别结果");
    expect(submittedPrompt).toContain("完整覆盖原可见像素");
    expect(submittedPrompt).toContain("用户说明：参考小熊与爱心排布");
    expect(submittedPrompt).toContain("R2（输入图片第 2 张）");
    expect(submittedPrompt).not.toContain("用途=");
    expect(submittedPrompt.match(/参考 R2/g)).toHaveLength(1);
    expect(harness.artifacts.snapshotImage).toHaveBeenCalledWith(
      "job-1",
      "/uploaded/pink-hoodie.png",
      "reference-01",
    );
    expect(preview.input.textureReferences).toEqual([expect.objectContaining({
      name: "粉色卫衣",
      path: "/local/reference-01.png",
    })]);
    expect(preview.artifacts.textureEdits?.["texture-body"].requestId).toBe("texture-batch");
    expect(preview.artifacts.textureEdits?.["texture-hair"].requestId).toBe("texture-batch");
    expect(preview.outputModelId).toBeNull();
    expect(vi.mocked(harness.artifacts.constrainTextureEditResult).mock.calls
      .every((call) => call[7] === "shrink-only")).toBe(true);

    const accepted = await harness.service.acceptTexture(job.id, "texture-body", true);
    expect(accepted.job.outputModelId).toBeNull();
    expect(accepted.delta).toMatchObject({ modelId: "mao", streamId: "job:job-1" });
    expect(harness.builder.build).not.toHaveBeenCalled();
    const reverted = await harness.service.acceptTexture(job.id, "texture-body", false);
    expect(reverted.job.artifacts.textureEdits?.["texture-body"].status).toBe("review");
    expect(harness.texturePreview.snapshot).toHaveBeenCalledTimes(2);
  });

  it("draws a selection too big for one canvas as several requests, each after the first anchored to it", async () => {
    let call = 0;
    const harness = await createHarness({
      editMaskedTexture: vi.fn(async () => {
        call += 1;
        return { requestId: `canvas-${call}`, urls: [`https://cdn/canvas-${call}.png`] };
      }),
    }, { largeTextures: true });
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body", "texture-hair"]);
    if (!plan.fits) throw new Error(plan.reason);
    expect(plan.canvases).toHaveLength(2);
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: "整体改成淡粉色",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body", "texture-hair"],
      textureBatchPlanHash: plan.hash,
      textureCanvasDescriptions: ["第一张：A 改成淡粉色", "第二张：B 改成淡粉色"],
      budgetLimitUsd: plan.estimatedUsd,
    });

    const preview = await harness.service.generateCandidates(job.id);

    expect(preview.phase).toBe("preview");
    expect(harness.provider.editMaskedTexture).toHaveBeenCalledTimes(2);
    const prompts = vi.mocked(harness.provider.editMaskedTexture).mock.calls.map((args) => args[0].prompt);
    expect(prompts[0]).toContain("第一张：A 改成淡粉色");
    expect(prompts[0]).not.toContain("上一张画布");
    expect(prompts[1]).toContain("第二张：B 改成淡粉色");
    expect(prompts[1]).toContain("上一张画布上已经画好的结果");
    // The second canvas is built with the first canvas's result as its anchor.
    const builds = vi.mocked(harness.artifacts.createTextureEditBatchWorkCanvas).mock.calls;
    expect(builds[0]![3]).toBe("texture-batch-1");
    expect(builds[0]![4]).toBeNull();
    expect(builds[1]![4]).toEqual({ imagePath: "/local/texture-batch-1-raw.png", plan: plan.canvases[0] });
    expect(preview.budget.estimatedUsd).toBeCloseTo(plan.estimatedUsd);
    expect(preview.artifacts.textureBatches?.map((batch) => batch.requestId)).toEqual(["canvas-1", "canvas-2"]);
    expect(Object.values(preview.artifacts.textureEdits ?? {}).map((result) => result.status)).toEqual(["review", "review"]);
    const regions = plan.canvases.flatMap((canvas) => canvas.items.map((item) => item.textureUnitId));
    expect(preview.artifacts.textureEdits?.[regions[0]!]?.requestId).toBe("canvas-1");
    expect(preview.artifacts.textureEdits?.[regions[1]!]?.requestId).toBe("canvas-2");
  });

  it("builds the AI work canvas from the current material atlas instead of the base atlas", async () => {
    const harness = await createHarness();
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body", "texture-hair"]);
    if (!plan.fits) throw new Error(plan.reason);
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: "在当前粉色素材上继续增加爱心图案",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body", "texture-hair"],
      textureBatchPlanHash: plan.plan.hash,
    });

    await harness.service.generateCandidates(job.id);

    expect(harness.resolveCurrentTextureAtlas).toHaveBeenCalledWith("mao", "textures/atlas.png");
    const entries = vi.mocked(harness.artifacts.createTextureEditBatchWorkCanvas).mock.calls[0]?.[1];
    expect(entries).toHaveLength(2);
    expect(entries?.every((entry) => entry.atlasPath === "/active/mao/textures/atlas.png")).toBe(true);
  });

  it("keeps stable prompt IDs and input positions after filtering references", async () => {
    const harness = await createHarness();
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body"]);
    if (!plan.fits) throw new Error(plan.reason);
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: "修改鞋子材质",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body"],
      textureBatchPlanHash: plan.plan.hash,
      textureReferences: [{
        id: "front",
        source: "model-preview",
        name: "当前正面图",
        description: "",
        path: "/uploaded/front.png",
        promptId: "P0",
        deliveryUse: "supplemental",
      }, {
        id: "reference-two",
        source: "upload",
        name: "鞋子参考.png",
        description: "深红色鞋面",
        path: "/uploaded/shoe.png",
        promptId: "R2",
        deliveryUse: "required",
      }],
    });

    await harness.service.generateCandidates(job.id);

    const call = vi.mocked(harness.provider.editMaskedTexture).mock.calls[0]![0];
    expect(call.referenceImagePaths).toEqual(["/local/reference-01.png", "/local/reference-02.png"]);
    expect(call.prompt).toContain("P0（输入图片第 2 张）：用户确认发送的当前模型位置参考");
    expect(call.prompt).toContain("参考 R2（输入图片第 3 张）：用户上传图片“鞋子参考.png”");
  });

  it("submits the complete multi-region optimized prompt without dropping later regions", async () => {
    const editMaskedTexture = vi.fn(async () => ({
      requestId: "texture-long-prompt",
      urls: ["https://cdn/texture-long-prompt.png"],
    }));
    const harness = await createHarness({ editMaskedTexture });
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body", "texture-hair"]);
    if (!plan.fits) throw new Error(plan.reason);
    const finalRegionConstraint = "区域 D 必须保持横向带子的位置、方向、宽度和连接关系";
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: `${"保持原图结构。".repeat(560)}${finalRegionConstraint}`,
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body", "texture-hair"],
      textureBatchPlanHash: plan.plan.hash,
    });

    await harness.service.generateCandidates(job.id);

    expect(editMaskedTexture).toHaveBeenCalledTimes(1);
    expect(editMaskedTexture.mock.calls[0]?.[0].prompt).toContain(finalRegionConstraint);
  });

  it("recognizes the compiled visual prompt sections without appending contradictory legacy rules", async () => {
    const editMaskedTexture = vi.fn(async () => ({
      requestId: "texture-compiled-prompt",
      urls: ["https://cdn/texture-compiled-prompt.png"],
    }));
    const harness = await createHarness({ editMaskedTexture });
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body"]);
    if (!plan.fits) throw new Error(plan.reason);
    const compiledPrompt = [
      "任务：",
      "编辑输入图 1 中标记的 Live2D UV 纹理。用户要求：改成带有自然做旧颗粒的粉色布料。",
      "参考：",
      "- R1：用户上传图片“reference.png”；仅参考布料的材质。",
      "视觉方向 S1（粉色做旧布料）：",
      "- 材质：柔和棉布与用户要求的自然做旧颗粒。",
      "区域：",
      "- A（衣服主体）：应用 S1。",
      "技术约束：",
      "只修改 A；保持原 UV 位置、比例、外轮廓和连接关系。",
      "保持原始 Alpha 和透明轮廓不变。",
    ].join("\n");
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: compiledPrompt,
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body"],
      textureBatchPlanHash: plan.plan.hash,
      textureReferences: [{
        id: "reference-upload",
        source: "upload",
        name: "reference.png",
        description: "参考布料材质",
        path: "/uploaded/reference.png",
      }],
    });

    await harness.service.generateCandidates(job.id);

    const submitted = editMaskedTexture.mock.calls[0]?.[0].prompt ?? "";
    expect(submitted.match(/用户上传图片“reference\.png”/g)).toHaveLength(1);
    expect(submitted.match(/保持原始 Alpha/g)).toHaveLength(1);
    expect(submitted).not.toContain("禁止污点、写实噪声、颗粒");
    expect(submitted).toContain("用户明确要求的做旧、颗粒");
  });

  it("saves accepted textures, discards the rest, and cancels the reviewed task", async () => {
    const harness = await createHarness();
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body", "texture-hair"]);
    if (!plan.fits) throw new Error(plan.reason);
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: "改成淡粉色针织纹理",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body", "texture-hair"],
      textureBatchPlanHash: plan.plan.hash,
    });
    await harness.service.generateCandidates(job.id);
    await harness.service.acceptTexture(job.id, "texture-body", true);

    const [cancelled, duplicate] = await Promise.all([
      harness.service.saveAcceptedTexturesAndCancel(job.id),
      harness.service.saveAcceptedTexturesAndCancel(job.id),
    ]);

    expect(cancelled.phase).toBe("cancelled");
    expect(duplicate).toEqual(cancelled);
    expect(cancelled.outputModelId).toBeNull();
    expect(cancelled.artifacts.textureEdits?.["texture-body"].status).toBe("accepted");
    expect(cancelled.artifacts.textureEdits?.["texture-hair"]).toMatchObject({
      status: "cancelled",
      error: "返回调整，未采用结果已丢弃",
    });
    expect(harness.textureMaterials.save).toHaveBeenCalledWith(expect.objectContaining({
      modelId: "mao",
      name: "AI 素材",
      prompt: "改成淡粉色针织纹理",
      edits: [expect.objectContaining({
        textureUnitId: "texture-body",
        atlasFile: "textures/atlas.png",
      })],
    }));
    expect(harness.textureMaterials.save).toHaveBeenCalledTimes(1);
    expect(harness.provider.editMaskedTexture).toHaveBeenCalledTimes(1);
    expect(harness.deleteGeneratedModel).not.toHaveBeenCalledWith("generated-model");
  });

  it("cancels without changing the model when no texture result was accepted", async () => {
    const harness = await createHarness();
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body", "texture-hair"]);
    if (!plan.fits) throw new Error(plan.reason);
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: "修改纹理",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body", "texture-hair"],
      textureBatchPlanHash: plan.plan.hash,
    });
    await harness.service.generateCandidates(job.id);

    const cancelled = await harness.service.saveAcceptedTexturesAndCancel(job.id);

    expect(cancelled.phase).toBe("cancelled");
    expect(cancelled.artifacts.textureEdits?.["texture-body"].status).toBe("cancelled");
    expect(cancelled.artifacts.textureEdits?.["texture-hair"].status).toBe("cancelled");
    expect(harness.textureMaterials.save).not.toHaveBeenCalled();
    expect(harness.deleteGeneratedModel).not.toHaveBeenCalled();
    expect(harness.provider.editMaskedTexture).toHaveBeenCalledTimes(1);
  });

  it("keeps an advisory quality warning reviewable and lets the user accept it", async () => {
    const harness = await createHarness();
    harness.artifacts.constrainTextureEditResult = vi.fn(async (_jobId, _source, _raw, _mask, _layout, _label, name) => ({
      path: `/local/${name}.png`,
      sourcePreviewPath: `/local/${name}-source-preview.png`,
      previewPath: `/local/${name}-preview.png`,
      quality: {
        accepted: false,
        issues: ["新增高频边缘较多，请人工检查"],
        changedPixelRatio: 0.75,
        alphaChangedPixelRatio: 0.25,
        removedAlphaRatio: 0.18,
        ignoredExpansionPixelRatio: 0,
        sharpnessRatio: 0.91,
      },
    }));
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body"]);
    if (!plan.fits) throw new Error(plan.reason);
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: "粉色高跟鞋",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body"],
      textureBatchPlanHash: plan.plan.hash,
    });

    const preview = await harness.service.generateCandidates(job.id);
    expect(preview.artifacts.textureEdits?.["texture-body"]).toMatchObject({
      status: "review",
      quality: { accepted: false },
    });

    const accepted = await harness.service.acceptTexture(job.id, "texture-body", true);
    expect(accepted.job.artifacts.textureEdits?.["texture-body"].status).toBe("accepted");
    expect(accepted.job.outputModelId).toBeNull();
  });

  it("does not automatically repeat an interrupted paid texture request after resume", async () => {
    let markSubmitted!: () => void;
    const submitted = new Promise<void>((done) => (markSubmitted = done));
    const neverFinishes = new Promise<{ requestId: string; urls: string[] }>(() => undefined);
    const editMaskedTexture = vi.fn(async (input) => {
      await input.onSubmitted?.("openai/gpt-image-2.5/sunburst/edit", "interrupted-texture");
      markSubmitted();
      return neverFinishes;
    });
    const harness = await createHarness({ editMaskedTexture });
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body"]);
    if (!plan.fits) throw new Error(plan.reason);
    const job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: "改成干净的淡粉色纹理",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body"],
      textureBatchPlanHash: plan.plan.hash,
    });
    void harness.service.generateCandidates(job.id);
    await submitted;

    const paused = await harness.service.pause(job.id);
    expect(paused.artifacts.textureEdits?.["texture-body"]).toMatchObject({
      status: "cancelled",
      estimatedUsd: 0.2,
    });

    const resumed = await harness.service.resume(job.id);
    const preview = await harness.service.generateCandidates(resumed.id);

    expect(preview.phase).toBe("preview");
    expect(preview.artifacts.textureEdits?.["texture-body"].status).toBe("cancelled");
    expect(preview.budget.estimatedUsd).toBe(0.2);
    expect(editMaskedTexture).toHaveBeenCalledTimes(1);
  });

  it("marks every unprocessed texture result cancelled when the whole job is cancelled", async () => {
    const harness = await createHarness();
    const plan = await harness.service.planTextureBatch("girl-v1", ["texture-body", "texture-hair"]);
    if (!plan.fits) throw new Error(plan.reason);
    let job = await harness.service.createJob({
      operation: "texture-edit",
      templateId: "girl-v1",
      description: "修改纹理",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["texture-body", "texture-hair"],
      textureBatchPlanHash: plan.plan.hash,
    });
    job = await harness.jobs.save({
      ...job,
      artifacts: {
        ...job.artifacts,
        textureEdits: {
          "texture-body": {
            textureUnitId: "texture-body",
            status: "pending",
            requestId: null,
            endpoint: null,
            estimatedUsd: 0,
            sourcePath: null,
            sourcePreviewPath: null,
            workImagePath: null,
            workMaskPath: null,
            rawResultPath: null,
            editedPatchPath: null,
            editedPreviewPath: null,
            layout: null,
            quality: null,
            error: null,
          },
          "texture-hair": {
            textureUnitId: "texture-hair",
            status: "running",
            requestId: null,
            endpoint: "fal-ai/gpt-image-1.5/edit",
            estimatedUsd: 0.08,
            sourcePath: null,
            sourcePreviewPath: null,
            workImagePath: null,
            workMaskPath: null,
            rawResultPath: null,
            editedPatchPath: null,
            editedPreviewPath: null,
            layout: null,
            quality: null,
            error: null,
          },
        },
      },
    });

    const cancelled = await harness.service.cancel(job.id);

    expect(cancelled.artifacts.textureEdits?.["texture-body"].status).toBe("cancelled");
    expect(cancelled.artifacts.textureEdits?.["texture-hair"].status).toBe("cancelled");
  });

  it("modifies only the two primary iris meshes and preserves the original model atlas", async () => {
    const harness = await createHarness();
    const job = await harness.service.createJob({
      operation: "eye-restyle",
      templateId: "girl-v1",
      description: "淡粉色虹膜和柔和星形高光",
      referenceImage: null,
      sourceModelId: null,
    });

    const preview = await harness.service.generateCandidates(job.id);

    expect(preview.phase).toBe("preview");
    expect(preview.artifacts.eyeSlotIds).toEqual(["iris-left", "iris-right"]);
    expect(preview.artifacts.slotTextures).toEqual({
      "iris-left": "/local/eye-constrained.png",
      "iris-right": "/local/eye-constrained.png",
    });
    expect(harness.provider.generateCandidates).not.toHaveBeenCalled();
    expect(harness.provider.inpaintImage).toHaveBeenCalledWith(expect.objectContaining({
      imagePath: "/local/eye-crop-image.png",
      maskPath: "/local/eye-crop-mask.png",
      imageSize: "landscape_4_3",
    }));
    expect(harness.builder.build).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.stringContaining("眼睛"),
      {
        preserveUnchanged: true,
        deltaReferencePath: path.join("/template", "alignment-guide.png"),
      },
    );
  });

  it("restyles only the preflight-selected ArtMeshes and preserves model geometry", async () => {
    const harness = await createHarness();
    const job = await harness.service.createJob({
      operation: "part-restyle",
      templateId: "girl-v1",
      description: "改成柔和深栗色发丝纹理",
      referenceImage: null,
      sourceModelId: "mao",
      editPartId: "hair-front",
      editMode: "appearance",
    });

    const preview = await harness.service.generateCandidates(job.id);

    expect(preview.phase).toBe("preview");
    expect(preview.artifacts.editSlotIds).toEqual(["hair-front"]);
    expect(preview.artifacts.slotTextures).toEqual({
      "hair-front": "/local/part-constrained.png",
    });
    expect(harness.artifacts.createExactSlotEditMask).toHaveBeenCalledWith(
      "job-1",
      path.join("/template", "alignment-guide.png"),
      "/template",
      expect.anything(),
      expect.anything(),
      [expect.objectContaining({ id: "hair-front" })],
      "part-edit-mask",
      expect.anything(),
    );
    expect(harness.provider.inpaintImage).toHaveBeenCalledWith(expect.objectContaining({
      imagePath: "/local/part-crop-image.png",
      maskPath: "/local/part-crop-mask.png",
      imageSize: { width: 1024, height: 1024 },
    }));
    expect(harness.builder.build).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.stringContaining("刘海"),
      { preserveUnchanged: true, deltaReferencePath: path.join("/template", "alignment-guide.png") },
    );
  });

  it("uses low redraw strength and clean texture constraints for recoloring", async () => {
    const harness = await createHarness();
    const job = await harness.service.createJob({
      operation: "part-restyle",
      templateId: "girl-v1",
      description: "白色衣服改成干净的粉色卡哇伊风格",
      referenceImage: null,
      sourceModelId: "mao",
      editPartId: "body",
      editMode: "appearance",
    });

    await harness.service.generateCandidates(job.id);

    expect(harness.provider.inpaintImage).toHaveBeenCalledWith(expect.objectContaining({
      strength: 0.42,
      guidanceScale: 3.5,
      numInferenceSteps: 30,
      acceleration: "none",
      negativePrompt: expect.stringMatching(/dirt.*stains.*noise/i),
      prompt: expect.stringContaining("只替换指定颜色"),
    }));
    expect(harness.artifacts.constrainMaskedCropEdit).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.anything(),
      expect.anything(),
      expect.anything(),
      expect.anything(),
      expect.anything(),
      "preserve",
    );
  });

  it("uses the lowest redraw strength for removal and cleanup requests", async () => {
    const harness = await createHarness();
    const job = await harness.service.createJob({
      operation: "part-restyle",
      templateId: "girl-v1",
      description: "去除腿上的蓝色涂鸦",
      referenceImage: null,
      sourceModelId: "mao",
      editPartId: "body",
      editMode: "appearance",
    });

    await harness.service.generateCandidates(job.id);

    expect(harness.provider.inpaintImage).toHaveBeenCalledWith(expect.objectContaining({
      strength: 0.34,
      prompt: expect.stringContaining("不得重画整个部位"),
    }));
  });

  it("rejects contour-changing requests before calling or charging fal", async () => {
    const harness = await createHarness();
    const job = await harness.service.createJob({
      operation: "part-restyle",
      templateId: "girl-v1",
      description: "把帽子变成皇冠",
      referenceImage: null,
      sourceModelId: "mao",
      editPartId: "hair-front",
      editMode: "appearance",
    });

    const failed = await harness.service.generateCandidates(job.id);

    expect(failed.phase).toBe("failed");
    expect(failed.budget.estimatedUsd).toBe(0);
    expect(failed.issues[0]?.message).toMatch(/轮廓或结构.*未调用 fal/);
    expect(harness.provider.inpaintImage).not.toHaveBeenCalled();
  });

  it("persists four locally valid candidates without a second vision gate", async () => {
    const harness = await createHarness();
    const job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "黑发少女",
      referenceImage: null,
      sourceModelId: null,
    });

    const generated = await harness.service.generateCandidates(job.id);

    expect(generated.candidates.map((item) => item.localPath)).toEqual([
      "/local/candidate-0.png",
      "/local/candidate-1.png",
      "/local/candidate-2.png",
      "/local/candidate-3.png",
    ]);
    expect(generated.budget.estimatedUsd).toBe(0.08);
    expect(generated.candidates.every((candidate) => candidate.compatible)).toBe(true);
    expect(harness.download).toHaveBeenCalledTimes(4);
    expect(harness.provider.analyzeImages).not.toHaveBeenCalled();
    expect(harness.provider.generateCandidates).toHaveBeenCalledWith(expect.objectContaining({
      templateGuidePath: path.join("/template", "alignment-guide.png"),
      imageSize: { width: 1200, height: 2000 },
    }));
    expect(JSON.stringify(generated)).not.toContain("fal-key");
  });

  it("leaves visual style selection to the user instead of a heuristic vision gate", async () => {
    const harness = await createHarness({
      analyzeImages: vi.fn(async () => ({
        requestId: "vision-style",
        content: JSON.stringify({ candidates: [
          { index: 0, compatible: false, reason: "写实照片风格" },
          { index: 1, compatible: true, reason: "二维动漫立绘" },
          { index: 2, compatible: false, reason: "三维渲染风格" },
          { index: 3, compatible: true, reason: "二维赛璐璐立绘" },
        ] }),
      })),
    });
    const job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });

    const generated = await harness.service.generateCandidates(job.id);

    expect(generated.candidates.every((candidate) => candidate.compatible)).toBe(true);
    expect(harness.provider.analyzeImages).not.toHaveBeenCalled();
    await expect(harness.service.selectCandidate(job.id, "candidate-0"))
      .resolves.toMatchObject({ phase: "decomposing", selectedCandidateId: "candidate-0" });
  });

  it("fails safely when every candidate is incompatible and clears them on retry", async () => {
    const harness = await createHarness();
    harness.artifacts.inspectCandidateBackground = vi.fn(async () => ({
      compatible: false,
      reason: "背景或画布无效",
    }));
    const job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });

    const failed = await harness.service.generateCandidates(job.id);
    expect(failed.phase).toBe("failed");
    expect(failed.issues.at(-1)?.code).toBe("no-template-compatible-candidates");

    const retried = await harness.service.retry(job.id);
    expect(retried.phase).toBe("draft");
    expect(retried.candidates).toEqual([]);
  });

  it("clears an unwanted candidate batch before regenerating with a new revision", async () => {
    const harness = await createHarness();
    const created = await harness.service.createJob({
      templateId: "girl-v1",
      description: "粉色短裙少女",
      referenceImage: null,
      sourceModelId: null,
    });
    const generated = await harness.service.generateCandidates(created.id);

    const reset = await harness.service.regenerateCandidates(generated.id);

    expect(reset.phase).toBe("draft");
    expect(reset.candidates).toEqual([]);
    expect(reset.selectedCandidateId).toBeNull();
    expect(reset.revision).toBe(generated.revision + 1);
    expect(reset.budget.estimatedUsd).toBe(generated.budget.estimatedUsd);
  });

  it("discards a late provider result after cancellation", async () => {
    let resolve!: (value: { requestId: string; urls: string[]; seed: number | null }) => void;
    let markSubmitted!: () => void;
    const submitted = new Promise<void>((done) => (markSubmitted = done));
    const pending = new Promise<{ requestId: string; urls: string[]; seed: number | null }>(
      (done) => (resolve = done),
    );
    const harness = await createHarness({
      generateCandidates: vi.fn(async (input) => {
        await input.onSubmitted?.("fal-ai/qwen-image-2/edit", "active-request");
        markSubmitted();
        return pending;
      }),
    });
    const job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });
    const generation = harness.service.generateCandidates(job.id);

    await submitted;
    expect((await harness.jobs.load(job.id)).falRequests).toEqual(expect.arrayContaining([
      expect.objectContaining({ requestId: "active-request", operation: "in-flight" }),
    ]));
    await harness.service.cancel(job.id);
    resolve({ requestId: "late", urls: ["https://cdn/late.png"], seed: null });
    await generation;

    const cancelled = await harness.jobs.load(job.id);
    expect(cancelled.phase).toBe("cancelled");
    expect(cancelled.candidates).toEqual([]);
    expect(harness.download).not.toHaveBeenCalled();
    expect(harness.provider.cancel).toHaveBeenCalledWith(
      "fal-ai/qwen-image-2/edit",
      "active-request",
    );
  });

  it("clears a deleted preview model from a cancelled job", async () => {
    const harness = await createHarness();
    const created = await harness.service.createJob({
      operation: "part-restyle",
      templateId: "girl-v1",
      description: "修改衣服颜色",
      referenceImage: null,
      sourceModelId: "source-model",
      editPartId: "body",
      editMode: "appearance",
    });
    await harness.jobs.save({
      ...created,
      phase: "preview",
      outputModelId: "temporary-preview",
    });

    const cancelled = await harness.service.cancel(created.id);

    expect(cancelled.phase).toBe("cancelled");
    expect(cancelled.outputModelId).toBeNull();
    expect(harness.deleteGeneratedModel).toHaveBeenCalledWith("temporary-preview");
    expect((await harness.jobs.load(created.id)).outputModelId).toBeNull();
  });

  it("aligns, validates and splits a selected candidate before building the base preview", async () => {
    const harness = await createHarness();
    let job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });
    job = await harness.jobs.save({
      ...job,
      candidates: [{ id: "candidate-0", localPath: "/local/candidate.png", seed: 1, compatible: true, rejectionReason: null }],
    });
    harness.provider.editImage = vi.fn(async () => ({
      requestId: "normalize-1",
      urls: ["https://cdn/normalized.png"],
    }));
    harness.provider.decomposeImage = vi.fn(async () => ({
      requestId: "layer-1",
      urls: ["https://cdn/layer-0.png", "https://cdn/layer-1.png"],
    }));
    harness.provider.removeBackground = vi.fn(async () => ({
      requestId: "background-1",
      url: "https://cdn/subject.png",
      maskUrl: null,
    }));
    harness.provider.segmentImage = vi.fn(async () => ({
      requestId: "segment-1",
      url: "https://cdn/segmented.png",
    }));
    harness.provider.analyzeImages = vi.fn(async () => ({
      requestId: "vision-1",
      content: '{"aligned":true,"reason":"五官与关节坐标一致"}',
    }));

    await harness.service.selectCandidate(job.id, "candidate-0");
    const preview = await harness.service.processSelected(job.id);

    expect(preview.phase).toBe("base-preview");
    expect(preview.outputModelId).toBe("generated-model");
    expect(preview.artifacts.subjectImage).toContain("subject-rgba");
    expect(preview.artifacts.coarseLayers).toHaveLength(0);
    expect(preview.artifacts.slotTextures.body).toContain("subject-rgba");
    expect(preview.artifacts.completionSheets).toEqual({});
    expect(preview.budget.estimatedUsd).toBe(0.04);
    expect(harness.provider.editImage).toHaveBeenCalledOnce();
    expect(harness.provider.removeBackground).toHaveBeenCalledOnce();
    expect(harness.provider.decomposeImage).not.toHaveBeenCalled();
    expect(harness.provider.segmentImage).not.toHaveBeenCalled();
    expect(harness.provider.analyzeImages).not.toHaveBeenCalled();
    expect(harness.builder.build).toHaveBeenCalledOnce();
  });

  it("generates only the completion groups selected after base preview", async () => {
    const harness = await createHarness();
    let job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });
    job = await harness.jobs.save({
      ...job,
      candidates: [{ id: "candidate-0", localPath: "/local/candidate.png", seed: 1, compatible: true, rejectionReason: null }],
    });
    harness.provider.editImage = vi.fn(async () => ({
      requestId: "normalize-1",
      urls: ["https://cdn/normalized.png"],
    }));
    harness.provider.decomposeImage = vi.fn(async () => ({
      requestId: "layer",
      urls: ["https://cdn/layer-0.png", "https://cdn/layer-1.png"],
    }));
    harness.provider.removeBackground = vi.fn(async () => ({
      requestId: "background",
      url: "https://cdn/subject.png",
      maskUrl: null,
    }));
    harness.provider.segmentImage = vi.fn(async () => ({
      requestId: "segment",
      url: "https://cdn/segmented.png",
    }));
    await harness.service.selectCandidate(job.id, "candidate-0");
    const base = await harness.service.processSelected(job.id);
    const completing = await harness.service.beginCompletion(base.id, ["body-hidden"]);
    const preview = await harness.service.processSelected(completing.id);

    expect(preview.phase).toBe("preview");
    expect(preview.budget.estimatedUsd).toBe(0.075);
    expect(preview.artifacts.completionSheets["completion:body-hidden"]).toContain("completion-body-hidden");
    expect(preview.artifacts.slotTextures.body).toContain("completion-body-hidden");
    expect(harness.provider.editImage).toHaveBeenCalledTimes(2);
    expect(harness.provider.segmentImage).not.toHaveBeenCalled();
    expect(harness.provider.analyzeImages).not.toHaveBeenCalled();
    expect(harness.provider.decomposeImage).not.toHaveBeenCalled();
  });

  it("builds a reviewable preview and records a warning when coverage is imperfect", async () => {
    const harness = await createHarness();
    let job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "轮廓不兼容的少女",
      referenceImage: null,
      sourceModelId: null,
    });
    job = await harness.jobs.save({
      ...job,
      candidates: [{ id: "candidate-0", localPath: "/local/candidate.png", seed: 1, compatible: true, rejectionReason: null }],
    });
    vi.mocked(harness.artifacts.inspectTemplateCoverage)
      .mockResolvedValue({ coveredRatio: 0.72, uncoveredPixels: 280, subjectPixels: 1_000 });

    await harness.service.selectCandidate(job.id, "candidate-0");
    const result = await harness.service.processSelected(job.id);

    expect(result.phase).toBe("base-preview");
    expect(result.issues.at(-1)).toMatchObject({
      code: "projection-coverage-warning",
      message: expect.stringContaining("72.0%"),
    });
    expect(harness.builder.build).toHaveBeenCalledOnce();
  });

  it("marks only the selected preview part for Qwen Edit repair", async () => {
    const harness = await createHarness();
    let job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });
    job = await harness.jobs.save({
      ...job,
      phase: "preview",
      outputModelId: "draft-model",
      artifacts: {
        normalizedImage: "/local/normalized.png",
        subjectImage: "/local/subject.png",
        coarseLayers: ["/local/layer.png"],
        completionSheets: { "completion:body-hidden": "/local/body-hidden.png" },
        slotTextures: { body: "/local/body.png" },
        repairSlotIds: [],
        requestedCompletions: ["body-hidden"],
      },
    });

    const restarted = await harness.service.regeneratePart(job.id, "body");

    expect(restarted.artifacts.slotTextures.body).toBeUndefined();
    expect(restarted.phase).toBe("completing");
    expect(restarted.artifacts.completionSheets["completion:body-hidden"]).toBeUndefined();
    expect(restarted.artifacts.requestedCompletions).toEqual(["body-hidden"]);
    expect(restarted.artifacts.repairSlotIds).toEqual(["body"]);
  });

  it("restarts every material pass without discarding the selected character", async () => {
    const harness = await createHarness();
    let job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });
    job = await harness.jobs.save({
      ...job,
      phase: "validating",
      paused: true,
      pauseReason: "needs-review",
      selectedCandidateId: "candidate-0",
      issues: [
        { code: "crack", message: "脸部异常", slotId: "face", recoverable: true },
        { code: "crack", message: "头发异常", slotId: "hair", recoverable: true },
      ],
      artifacts: {
        normalizedImage: "/local/normalized.png",
        subjectImage: "/local/subject.png",
        coarseLayers: ["/local/unused-layer.png"],
        completionSheets: {
          face: "/local/face.png",
          hair: "/local/hair.png",
          skin: "/local/skin.png",
          outfit: "/local/outfit.png",
        },
        slotTextures: { body: "/local/body.png" },
        repairSlotIds: [],
        requestedCompletions: [],
      },
    });

    const restarted = await harness.service.regeneratePart(job.id, ALL_MATERIALS_REPAIR_ID);

    expect(restarted.phase).toBe("decomposing");
    expect(restarted.selectedCandidateId).toBe("candidate-0");
    expect(restarted.artifacts.normalizedImage).toBeNull();
    expect(restarted.artifacts.subjectImage).toBeNull();
    expect(restarted.artifacts.completionSheets).toEqual({});
    expect(restarted.artifacts.slotTextures).toEqual({});
    expect(restarted.artifacts.repairSlotIds).toEqual([]);
    expect(restarted.artifacts.requestedCompletions).toEqual([]);
    expect(restarted.issues).toEqual([]);
  });

  it("discards the bad projection when the user requests template realignment", async () => {
    const harness = await createHarness();
    let job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });
    job = await harness.jobs.save({
      ...job,
      phase: "base-preview",
      selectedCandidateId: "candidate-0",
      outputModelId: "bad-preview",
      artifacts: {
        ...job.artifacts,
        normalizedImage: "/local/bad-alignment.png",
        subjectImage: "/local/bad-subject.png",
        slotTextures: { body: "/local/bad-subject.png" },
      },
    });

    const restarted = await harness.service.regeneratePart(job.id, REALIGN_TO_TEMPLATE_ID);

    expect(restarted.phase).toBe("draft");
    expect(restarted.candidates).toEqual([]);
    expect(restarted.selectedCandidateId).toBeNull();
    expect(restarted.outputModelId).toBeNull();
    expect(restarted.artifacts.normalizedImage).toBeNull();
    expect(restarted.artifacts.subjectImage).toBeNull();
    expect(restarted.artifacts.slotTextures).toEqual({});
  });

  it("cancels the active fal request when the user pauses", async () => {
    let markSubmitted!: () => void;
    const submitted = new Promise<void>((done) => (markSubmitted = done));
    const pending = new Promise<{ requestId: string; urls: string[]; seed: number | null }>(() => undefined);
    const harness = await createHarness({
      generateCandidates: vi.fn(async (input) => {
        await input.onSubmitted?.("fal-ai/qwen-image-2/edit", "pause-request");
        markSubmitted();
        return pending;
      }),
    });
    const job = await harness.service.createJob({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });
    void harness.service.generateCandidates(job.id);
    await submitted;

    const paused = await harness.service.pause(job.id);

    expect(paused.paused).toBe(true);
    expect(paused.pauseReason).toBe("user");
    expect(harness.provider.cancel).toHaveBeenCalledWith(
      "fal-ai/qwen-image-2/edit",
      "pause-request",
    );
  });
});

async function createHarness(
  overrides: Partial<CharacterGenerationProvider> = {},
  options: { largeTextures?: boolean } = {},
) {
  const root = await mkdtemp(path.join(os.tmpdir(), "character-service-"));
  roots.push(root);
  const jobs = new CharacterGenerationJobStore(path.join(root, "jobs"), () => "job-1");
  const download = vi.fn(async (_jobId: string, _url: string, name: string) =>
    `/local/${name}.png`
  );
  const artifacts = new CharacterArtifactStore(path.join(root, "artifacts"));
  artifacts.snapshotImage = vi.fn(async (_jobId, _source, name) => `/local/${name}.png`);
  artifacts.download = download;
  artifacts.applyMask = vi.fn(async (_jobId, _source, _mask, name) => `/local/${name}.png`);
  artifacts.assertSameImageSize = vi.fn(async () => undefined);
  artifacts.imageSize = vi.fn(async () => ({ width: 1200, height: 2000 }));
  artifacts.inspectTemplateCoverage = vi.fn(async () => ({
    coveredRatio: 0.995,
    uncoveredPixels: 5,
    subjectPixels: 1_000,
  }));
  artifacts.normalizeCandidateBackground = vi.fn(async (_jobId, _source, name) => `/local/${name}.png`);
  artifacts.alignCandidateToGuide = vi.fn(async (_jobId, _guide, _candidate, name) => `/local/${name}.png`);
  artifacts.inspectCandidateBackground = vi.fn(async () => ({ compatible: true, reason: null }));
  artifacts.createSlotEditMask = vi.fn(async () => "/local/eye-mask.png");
  artifacts.createExactSlotEditMask = vi.fn(async () => "/local/part-mask.png");
  artifacts.createMaskedEditCrop = vi.fn(async () => ({
    imagePath: "/local/part-crop-image.png",
    maskPath: "/local/part-crop-mask.png",
    source: { x: 100, y: 100, width: 500, height: 500 },
    output: { width: 1024, height: 1024 },
  }));
  artifacts.constrainMaskedCropEdit = vi.fn(async () => "/local/part-constrained.png");
  artifacts.createEyeEditCrop = vi.fn(async () => ({
    imagePath: "/local/eye-crop-image.png",
    maskPath: "/local/eye-crop-mask.png",
    source: { x: 300, y: 300, width: 600, height: 450 },
    output: { width: 1024, height: 768 },
  }));
  artifacts.constrainEyeCropEdit = vi.fn(async () => "/local/eye-constrained.png");
  artifacts.createTextureEditBatchWorkCanvas = vi.fn(async (_jobId, entries, plan, name) => ({
    imagePath: `/local/${name}-work.png`,
    maskPath: `/local/${name}-mask.png`,
    imageSize: plan.imageSize,
    plan,
    items: Object.fromEntries(plan.items.map((item) => [item.textureUnitId, {
      sourcePath: `/local/${name}-${item.textureUnitId}-source.png`,
      layout: {
        canvasWidth: plan.canvasWidth,
        canvasHeight: plan.canvasHeight,
        offsetX: item.x,
        offsetY: item.y,
        scale: item.scale,
        sourceWidth: item.sourceWidth,
        sourceHeight: item.sourceHeight,
      },
    }])),
  }));
  artifacts.constrainTextureEditResult = vi.fn(async (_jobId, _source, _raw, _mask, _layout, _label, name) => ({
    path: `/local/${name}.png`,
    sourcePreviewPath: `/local/${name}-source-preview.png`,
    previewPath: `/local/${name}-preview.png`,
    quality: {
      accepted: true,
      issues: [],
      changedPixelRatio: 0.4,
      alphaChangedPixelRatio: 0,
      removedAlphaRatio: 0,
      ignoredExpansionPixelRatio: 0,
      sharpnessRatio: 1,
    },
  }));
  const provider: CharacterGenerationProvider = {
    generateCandidates: vi.fn(async () => ({
      requestId: "request-1",
      urls: Array.from({ length: 4 }, (_, index) => `https://cdn/${index}.png`),
      seed: 9,
    })),
    editImage: vi.fn(async () => ({
      requestId: "edit-default",
      urls: ["https://cdn/edit.png"],
    })),
    inpaintImage: vi.fn(async () => ({
      requestId: "inpaint-default",
      urls: ["https://cdn/eye.png"],
    })),
    editMaskedTexture: vi.fn(async () => ({
      requestId: "texture-default",
      urls: ["https://cdn/texture.png"],
    })),
    decomposeImage: vi.fn(async () => ({
      requestId: "layer-default",
      urls: ["https://cdn/layer-0.png", "https://cdn/layer-1.png"],
    })),
    removeBackground: vi.fn(async () => ({
      requestId: "background-default",
      url: "https://cdn/subject.png",
      maskUrl: null,
    })),
    segmentImage: vi.fn(async () => ({
      requestId: "segment-default",
      url: "https://cdn/segment.png",
    })),
    analyzeImages: vi.fn(async (input) => input.imagePaths.length === 2
      ? {
          requestId: "alignment-vision",
          content: JSON.stringify({ aligned: true, reason: "五官与关节坐标一致" }),
        }
      : {
          requestId: "candidate-vision",
          content: JSON.stringify({ candidates: Array.from({ length: 4 }, (_, index) => ({
            index,
            compatible: true,
            reason: "二维动漫立绘",
          })) }),
        }),
    cancel: vi.fn(async () => undefined),
    ...overrides,
  };
  const builder = {
    build: vi.fn(async () => ({ id: "generated-model" }) as ModelPackage),
  };
  const deleteGeneratedModel = vi.fn(async () => undefined);
  const textureMaterials = { save: vi.fn(async () => undefined) };
  const texturePreview = {
    snapshot: vi.fn(async (input: { jobId: string; modelId: string; revision: number }) => ({
      modelId: input.modelId,
      streamId: `job:${input.jobId}`,
      revision: input.revision + 1,
      patchSets: [],
      atlases: [{
        atlasIndex: 0,
        atlasFile: "textures/atlas.png",
        url: `pet-generation://${input.jobId}/preview-atlas-0.png`,
      }],
    })),
  };
  const resolveCurrentTextureAtlas = vi.fn(async (modelId: string, atlasFile: string) =>
    `/active/${modelId}/${atlasFile}`
  );
  const service = new CharacterGenerationService({
    jobs,
    artifacts,
    providerFactory: async () => provider,
    resolveCurrentTextureAtlas,
    templates: {
      load: vi.fn(async () => ({
        id: "girl-v1",
        name: "Girl",
        root: "/template",
        manifest: {
          schemaVersion: 2,
          id: "girl-v1",
          name: "Girl",
          modelEntry: "girl.model3.json",
          alignmentGuide: "alignment-guide.png",
          coverage: "full-body",
          canvas: { width: 1200, height: 2000, originX: 600, originY: 1000, pixelsPerUnit: 1200 },
          parameters: { eyeBlink: ["ParamEyeOpen"], lipSync: ["ParamMouthOpenY"] },
          textureAtlases: [options.largeTextures
            ? { file: "textures/atlas.png", width: 8192, height: 8192 }
            : { file: "textures/atlas.png", width: 1024, height: 2048 }],
          slots: [{
            id: "body",
            name: "Body",
            drawableId: "D_BODY",
            partId: "PART_BODY",
            group: "body",
            source: "visible",
            required: true,
            zIndex: 1,
            atlas: options.largeTextures
              ? { texture: 0, x: 0, y: 0, width: 2000, height: 2000 }
              : { texture: 0, x: 0, y: 0, width: 178, height: 709 },
            anchor: { x: 89, y: 355 },
            canvasBounds: { x: -0.5, y: -0.8, width: 1, height: 1.6 },
            mesh: {
              vertices: [-0.5, 0.8, 0.5, 0.8, 0.5, -0.8, -0.5, -0.8],
              uvs: [0, 1, 1, 1, 1, 0, 0, 0],
              indices: [0, 1, 2, 0, 2, 3],
            },
            silhouetteMask: "masks/body.png",
            completionPrompt: "separate and complete the full body",
            editPartId: "body",
            textureUnitId: "texture-body",
          }, {
            id: "hair-front",
            name: "刘海与前发 · D_HAIR_FRONT",
            drawableId: "D_HAIR_FRONT",
            partId: "PartHairFront",
            group: "hair",
            source: "visible",
            required: false,
            zIndex: 2,
            atlas: options.largeTextures
              ? { texture: 0, x: 4000, y: 0, width: 2000, height: 2000 }
              : { texture: 0, x: 200, y: 20, width: 95, height: 169 },
            anchor: { x: 248, y: 105 },
            canvasBounds: { x: -0.3, y: 0.35, width: 0.6, height: 0.5 },
            mesh: {
              vertices: [-0.3, 0.85, 0.3, 0.85, 0.3, 0.35, -0.3, 0.35],
              uvs: [0, 1, 1, 1, 1, 0, 0, 0],
              indices: [0, 1, 2, 0, 2, 3],
            },
            silhouetteMask: "masks/hair-front.png",
            completionPrompt: "restyle front hair",
            editPartId: "hair-front",
            textureUnitId: "texture-hair",
          }, {
            id: "iris-left",
            name: "Left iris",
            drawableId: "D_IRIS_L",
            partId: "PartEyeBall",
            group: "face",
            source: "visible",
            required: false,
            zIndex: 2,
            atlas: { texture: 0, x: 10, y: 10, width: 80, height: 80 },
            anchor: { x: 50, y: 50 },
            canvasBounds: { x: -0.3, y: 0.2, width: 0.2, height: 0.2 },
            mesh: {
              vertices: [-0.3, 0.4, -0.1, 0.4, -0.1, 0.2, -0.3, 0.2],
              uvs: [0, 1, 1, 1, 1, 0, 0, 0],
              indices: [0, 1, 2, 0, 2, 3],
            },
            silhouetteMask: "masks/iris-left.png",
            completionPrompt: "iris",
          }, {
            id: "iris-right",
            name: "Right iris",
            drawableId: "D_IRIS_R",
            partId: "PartEyeBall",
            group: "face",
            source: "visible",
            required: false,
            zIndex: 3,
            atlas: { texture: 0, x: 100, y: 10, width: 80, height: 80 },
            anchor: { x: 140, y: 50 },
            canvasBounds: { x: 0.1, y: 0.2, width: 0.2, height: 0.2 },
            mesh: {
              vertices: [0.1, 0.4, 0.3, 0.4, 0.3, 0.2, 0.1, 0.2],
              uvs: [0, 1, 1, 1, 1, 0, 0, 0],
              indices: [0, 1, 2, 0, 2, 3],
            },
            silhouetteMask: "masks/iris-right.png",
            completionPrompt: "iris",
          }],
        },
      })),
    },
    builder,
    textureMaterials,
    texturePreview,
    deleteGeneratedModel,
  });
  return {
    service,
    jobs,
    download,
    artifacts,
    provider,
    builder,
    textureMaterials,
    texturePreview,
    deleteGeneratedModel,
    resolveCurrentTextureAtlas,
  };
}
