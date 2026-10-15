import { EventEmitter } from "node:events";
import path from "node:path";

import {
  CHARACTER_COMPLETION_KINDS,
  DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
  planTextureEditBatch,
  reserveEstimatedCost,
  resolveTextureEditImageModel,
  textureEditAlphaPolicy,
  textureEditBatchPromptRegions,
  textureReferenceLabels,
  transitionCharacterJob,
  type CharacterCompletionKind,
  type CharacterGenerationJob,
  type CharacterGenerationProgress,
  type CharacterGenerationProvider,
  type CharacterProviderStatus,
  type MasterTemplateSlot,
  type TextureEditResult,
  type TextureEditBatchArtifact,
  type TextureEditBatchPlan,
  type TextureEditBatchPlanResult,
  type TextureEditImageModel,
} from "../../src/shared/character-generation";
import type {
  CharacterTexturePreviewChange,
  ModelPackage,
  TextureAtlasDelta,
  TextureEditUnit,
} from "../../src/shared/contracts";
import type {
  CharacterArtifactStore,
  TextureEditBatchWorkAsset,
} from "./artifact-store";
import type { CharacterGenerationJobStore, CreateCharacterJobInput } from "./job-store";
import { validateRequiredSlots } from "./layer-analysis";
import type { MasterTemplatePackage } from "./master-template-store";

const COSTS_USD = {
  poseCandidate: 0.02,
  referencedCandidate: 0.035,
  templateAlignment: 0.035,
  backgroundRemoval: 0.005,
  completion: 0.035,
  eyeRestylePerMegapixel: 0.03,
  partRestylePerMegapixel: 0.03,
} as const;

const COMPLETION_PASSES: Array<{
  id: CharacterCompletionKind;
  label: string;
  prompt: string;
  matches(slot: MasterTemplateSlot): boolean;
}> = [
  {
    id: "eyes",
    label: "闭眼眼皮与睫毛",
    prompt: "仅补全角色闭眼动画所需的左右上眼皮、下眼皮和睫毛素材，严格沿用现有眼型、眼距、线稿粗细和妆容；保持睁眼五官、脸型、发型与身体完全不变。",
    matches: (slot) => /eye/i.test(slot.partId) && !/eyeball/i.test(slot.partId),
  },
  {
    id: "mouth",
    label: "口腔、牙齿与舌头",
    prompt: "仅补全角色嘴部内部素材：保持脸型、嘴角位置、唇色和周围皮肤完全不变，在原位置生成自然微张嘴所需的口腔内部、牙齿和舌头。不得移动或重画其他五官。",
    matches: (slot) => /mouth/i.test(slot.partId),
  },
  {
    id: "face-base",
    label: "被五官和头发遮挡的脸部底层",
    prompt: "仅补全头发和五官下方连续的脸部底色，严格保持原脸型、肤色、光照、头部尺寸和位置，不改变可见眼睛、鼻子、嘴巴与发型。",
    matches: (slot) => /(?:face|ear)/i.test(slot.partId) && !/(?:eye|brow|nose|mouth)/i.test(slot.partId),
  },
  {
    id: "back-hair",
    label: "被脸和身体遮挡的后发",
    prompt: "仅补全角色后发中被脸、颈部和衣服遮挡的部分，从发根到发梢保持同一发色、线稿、明暗和走向，不改变可见前发、脸或身体位置。",
    matches: (slot) => /hairback/i.test(slot.partId),
  },
  {
    id: "body-hidden",
    label: "身体与四肢遮挡层",
    prompt: "仅补全衣服和头发下方的颈部、躯干、双臂、双手、双腿和脚，保持现有姿势、比例、肤色与关节位置完全一致，不改变服装外观。",
    matches: (slot) => ["body", "arm-left", "arm-right", "leg-left", "leg-right"].includes(slot.group),
  },
  {
    id: "clothing-hidden",
    label: "服装遮挡层",
    prompt: "仅补全被头发和手臂遮挡的服装、裙子或裤子、鞋子与配饰，保持现有款式、花纹、颜色、轮廓和身体姿势完全一致。",
    matches: (slot) => ["clothing", "shoe-left", "shoe-right", "accessory"].includes(slot.group),
  },
];

export const ALL_MATERIALS_REPAIR_ID = "__all-materials__";
export const REALIGN_TO_TEMPLATE_ID = "__realign-to-template__";

class TemplateAlignmentError extends Error {}

interface ActiveOperation {
  controller: AbortController;
  provider: CharacterGenerationProvider | null;
  revision: number;
  requests: Array<{ endpoint: string; requestId: string }>;
}

interface TemplateLoader {
  load(id: string): Promise<MasterTemplatePackage>;
}

interface ModelBuilder {
  build(
    job: CharacterGenerationJob,
    template: MasterTemplatePackage,
    name: string,
    options?: {
      preserveUnchanged?: boolean;
      deltaReferencePath?: string;
      textureEdits?: Array<{
        textureUnitId: string;
        atlasIndex: number;
        bounds: { x: number; y: number; width: number; height: number };
        maskPath: string;
        editedPatchPath: string;
      }>;
    },
  ): Promise<ModelPackage>;
}

interface TextureMaterialSaver {
  save(input: {
    modelId: string;
    name: string;
    prompt: string;
    /** False only adds the versions to 换色 without wearing them. */
    apply?: boolean;
    edits: Array<{
      textureUnitId: string;
      label: string;
      atlasIndex: number;
      atlasFile: string;
      bounds: { x: number; y: number; width: number; height: number };
      maskPath: string;
      editedPatchPath: string;
    }>;
  }): Promise<void>;
}

interface TexturePreviewComposer {
  snapshot(input: {
    jobId: string;
    modelId: string;
    revision: number;
    enabled: boolean;
    affectedAtlases: Array<{ atlasIndex: number; atlasFile: string }>;
    edits: Array<{
      textureUnitId: string;
      atlasIndex: number;
      atlasFile: string;
      bounds: { x: number; y: number; width: number; height: number };
      maskPath: string;
      editedPatchPath: string;
    }>;
  }): Promise<TextureAtlasDelta>;
}

export interface CharacterGenerationServiceDependencies {
  jobs: CharacterGenerationJobStore;
  artifacts: CharacterArtifactStore;
  providerFactory(): Promise<CharacterGenerationProvider>;
  resolveCurrentTextureAtlas?(modelId: string, atlasFile: string): Promise<string>;
  templates?: TemplateLoader;
  builder?: ModelBuilder;
  textureMaterials?: TextureMaterialSaver;
  texturePreview?: TexturePreviewComposer;
  deleteGeneratedModel?(id: string): Promise<void>;
}

export class CharacterGenerationService extends EventEmitter {
  private readonly active = new Map<string, ActiveOperation>();
  private readonly textureReviewFinishes = new Map<string, Promise<CharacterGenerationJob>>();

  constructor(private readonly dependencies: CharacterGenerationServiceDependencies) {
    super();
  }

  async createJob(input: CreateCharacterJobInput): Promise<CharacterGenerationJob> {
    if (input.operation === "texture-edit") {
      const planned = await this.planTextureBatch(
        input.templateId,
        input.textureUnitIds ?? [],
        input.textureImageModel ?? DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
      );
      if (!planned.fits) throw new Error(planned.reason);
      if (!input.textureBatchPlanHash || input.textureBatchPlanHash !== planned.hash) {
        throw new Error("纹理排版已经变化，请返回确认页重新检查后再提交");
      }
    }
    const job = await this.dependencies.jobs.create(input);
    if (input.operation !== "texture-edit" || !(input.textureReferences?.length)) return job;
    job.input.textureReferences = await Promise.all(input.textureReferences.map(async (reference, index) => ({
      ...reference,
      path: await this.dependencies.artifacts.snapshotImage(
        job.id,
        reference.path,
        `reference-${String(index + 1).padStart(2, "0")}`,
      ),
    })));
    await this.dependencies.jobs.save(job);
    return job;
  }

  async planTextureBatch(
    templateId: string,
    textureUnitIds: readonly string[],
    imageModel: TextureEditImageModel = DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
  ): Promise<TextureEditBatchPlanResult> {
    if (!this.dependencies.templates) throw new Error("纹理母版加载器尚未配置");
    const template = await this.dependencies.templates.load(templateId);
    const selectedIds = [...new Set(textureUnitIds)];
    const units = resolveTextureUnits(template, selectedIds);
    return planTextureEditBatch(units.map((unit) => ({
      id: unit.id,
      label: unit.label,
      width: unit.bounds.width,
      height: unit.bounds.height,
    })), imageModel);
  }

  async listJobs(): Promise<CharacterGenerationJob[]> {
    return Promise.all((await this.dependencies.jobs.list()).map((job) =>
      this.ensureTextureEditPreviews(job),
    ));
  }

  listResumable(): Promise<CharacterGenerationJob[]> {
    return this.dependencies.jobs.listResumable();
  }

  async load(jobId: string): Promise<CharacterGenerationJob> {
    return this.ensureTextureEditPreviews(await this.dependencies.jobs.load(jobId));
  }

  private async ensureTextureEditPreviews(
    job: CharacterGenerationJob,
  ): Promise<CharacterGenerationJob> {
    if (
      job.input.operation !== "texture-edit" ||
      !this.dependencies.templates ||
      !job.artifacts.textureEdits
    ) return job;
    const missing = Object.values(job.artifacts.textureEdits).filter((result) =>
      result.sourcePath &&
      result.editedPatchPath &&
      (!result.sourcePreviewPath || !result.editedPreviewPath)
    );
    if (missing.length === 0) return job;

    let template: MasterTemplatePackage;
    try {
      template = await this.dependencies.templates.load(job.templateId);
    } catch {
      // A stale task can outlive an imported template. Keep it visible instead
      // of making the entire task list fail while attempting a preview backfill.
      return job;
    }
    let textureEdits = { ...job.artifacts.textureEdits };
    let changed = false;
    for (const result of missing) {
      try {
        const unit = resolveTextureUnits(template, [result.textureUnitId])[0];
        if (!unit || !result.sourcePath || !result.editedPatchPath) continue;
        const previews = await this.dependencies.artifacts.createTextureEditPreviews(
          job.id,
          result.sourcePath,
          result.editedPatchPath,
          path.join(template.root, unit.maskPath),
          `texture-${result.textureUnitId}`,
        );
        textureEdits = {
          ...textureEdits,
          [result.textureUnitId]: {
            ...result,
            sourcePreviewPath: previews.sourcePreviewPath,
            editedPreviewPath: previews.editedPreviewPath,
          },
        };
        changed = true;
      } catch {
        // A deleted or incomplete historical artifact must not make the whole
        // task list unavailable. New and intact jobs are backfilled normally.
      }
    }
    if (!changed) return job;
    const updated = {
      ...job,
      artifacts: { ...job.artifacts, textureEdits },
    };
    try {
      await this.dependencies.jobs.save(updated);
    } catch {
      // The in-memory result can still render the clean previews. A later load
      // may retry persistence without blocking the model manager meanwhile.
    }
    return updated;
  }

  async generateCandidates(jobId: string): Promise<CharacterGenerationJob> {
    let job = await this.dependencies.jobs.load(jobId);
    if (job.input.operation === "texture-edit") {
      try {
        return await this.generateTexturePreview(job);
      } catch (error) {
        // Failures before any image request (no key, stale layout) must still
        // end the job visibly instead of leaving it in draft forever.
        const current = await this.dependencies.jobs.load(jobId);
        if (current.phase !== "draft") throw error;
        const failed = transitionCharacterJob(current, "failed");
        failed.issues = [...failed.issues, {
          code: "texture-edit-failed",
          message: errorMessage(error),
          slotId: null,
          recoverable: true,
        }];
        await this.dependencies.jobs.save(failed);
        this.progress(failed, 1, failed.issues.at(-1)!.message);
        return failed;
      }
    }
    if (job.input.operation === "eye-restyle") return this.generateEyePreview(job);
    if (job.input.operation === "part-restyle") return this.generatePartPreview(job);
    if (job.phase !== "draft" || job.paused) return job;
    job = reserveEstimatedCost(
      job,
      "generate-four-candidates",
      (job.input.referenceImage ? COSTS_USD.referencedCandidate : COSTS_USD.poseCandidate) * 4,
    );
    await this.dependencies.jobs.save(job);
    if (job.paused) {
      this.progress(job, 0, "预计费用超过任务上限，等待确认");
      return job;
    }
    if (!this.dependencies.templates) throw new Error("角色母版加载器尚未配置");
    const template = await this.dependencies.templates.load(job.templateId);
    const alignmentGuide = path.join(template.root, template.manifest.alignmentGuide);
    const generationSize = await this.dependencies.artifacts.imageSize(alignmentGuide);
    const provider = await this.dependencies.providerFactory();
    const controller = new AbortController();
    const operation: ActiveOperation = { controller, provider, revision: job.revision, requests: [] };
    this.active.set(jobId, operation);
    this.progress(job, 0.05, "正在生成 4 张标准姿势候选…");
    try {
      const result = await provider.generateCandidates({
        prompt: job.input.description,
        referenceImagePath: job.input.referenceImage,
        templateGuidePath: alignmentGuide,
        imageSize: generationSize,
        count: 4,
        signal: controller.signal,
        onSubmitted: (endpoint, requestId) =>
          this.trackSubmittedRequest(jobId, operation, endpoint, requestId),
        onStatus: (status) => this.progress(
          job,
          0.05,
          providerStatusText("生成 4 张角色候选", status),
        ),
      });
      const current = await this.dependencies.jobs.load(jobId);
      if (!this.isCurrent(current, operation)) return current;
      if (result.urls.length !== 4) throw new Error("fal 未返回 4 张候选图");
      const candidates = await Promise.all(result.urls.map(async (url, index) => {
        const downloadedPath = await this.dependencies.artifacts.download(
          jobId,
          url,
          `candidate-${index}-raw`,
        );
        const normalizedBackground = await this.dependencies.artifacts.normalizeCandidateBackground(
          jobId,
          downloadedPath,
          `candidate-${index}-background`,
        );
        const localPath = await this.dependencies.artifacts.alignCandidateToGuide(
          jobId,
          alignmentGuide,
          normalizedBackground,
          `candidate-${index}`,
        );
        let localCompatibility = await this.dependencies.artifacts.inspectCandidateBackground(localPath);
        try {
          await this.dependencies.artifacts.assertSameImageSize(alignmentGuide, localPath);
        } catch (error) {
          localCompatibility = { compatible: false, reason: errorMessage(error) };
        }
        return {
          id: `candidate-${index}`,
          localPath,
          previewUrl: `pet-generation://${jobId}/${encodeURIComponent(path.basename(localPath))}`,
          seed: result.seed,
          compatible: false,
          rejectionReason: "尚未完成母版兼容性检查",
          localCompatibility,
        };
      }));
      const checkedCandidates = candidates.map((candidate) => ({
        id: candidate.id,
        localPath: candidate.localPath,
        previewUrl: candidate.previewUrl,
        seed: candidate.seed,
        compatible: candidate.localCompatibility.compatible,
        rejectionReason: !candidate.localCompatibility.compatible
          ? candidate.localCompatibility.reason
          : null,
      }));
      const latest = await this.dependencies.jobs.load(jobId);
      if (!this.isCurrent(latest, operation)) return latest;
      let completed = this.addRequest(
        latest,
        "generate-candidates",
        job.input.referenceImage
          ? "fal-ai/qwen-image-2/edit"
          : "fal-ai/z-image/turbo/controlnet",
        result.requestId,
      );
      if (!checkedCandidates.some((candidate) => candidate.compatible)) {
        const failed = transitionCharacterJob({
          ...completed,
          candidates: checkedCandidates,
        }, "failed");
        failed.issues = [...failed.issues, {
          code: "no-template-compatible-candidates",
          message: `这批候选均未通过 ${template.name} 的几何、画布或纯白背景检查，请重新生成。`,
          slotId: null,
          recoverable: true,
        }];
        await this.dependencies.jobs.save(failed);
        this.progress(failed, 1, failed.issues.at(-1)!.message);
        return failed;
      }
      const saved = await this.dependencies.jobs.save({
        ...completed,
        candidates: checkedCandidates,
        updatedAt: new Date().toISOString(),
      });
      this.progress(saved, 1, "请选择一张角色定稿");
      return saved;
    } catch (error) {
      const current = await this.dependencies.jobs.load(jobId);
      if (controller.signal.aborted || current.phase === "cancelled") return current;
      const failed = transitionCharacterJob(current, "failed");
      failed.issues = [...failed.issues, {
        code: "candidate-generation-failed",
        message: errorMessage(error),
        slotId: null,
        recoverable: true,
      }];
      await this.dependencies.jobs.save(failed);
      this.progress(failed, 1, failed.issues.at(-1)!.message);
      return failed;
    } finally {
      if (this.active.get(jobId) === operation) this.active.delete(jobId);
    }
  }

  private async generateTexturePreview(job: CharacterGenerationJob): Promise<CharacterGenerationJob> {
    if (job.phase !== "draft" || job.paused) return job;
    if (!this.dependencies.templates || !this.dependencies.builder) {
      throw new Error("纹理高保真编辑处理器尚未配置");
    }
    const template = await this.dependencies.templates.load(job.templateId);
    const selectedIds = [...new Set(job.input.textureUnitIds ?? [])];
    if (selectedIds.length === 0) throw new Error("请至少选择一个纹理素材块");
    const units = resolveTextureUnits(template, selectedIds);
    const imageModel = textureImageModel(job);
    const planned = planTextureEditBatch(units.map((unit) => ({
      id: unit.id,
      label: unit.label,
      width: unit.bounds.width,
      height: unit.bounds.height,
    })), imageModel);
    if (!planned.fits) throw new Error(planned.reason);
    if (job.input.textureBatchPlanHash !== planned.hash) {
      throw new Error("纹理排版已经变化，请重新选择纹理并确认排版");
    }
    const previousResults = job.artifacts.textureEdits ?? {};
    job.artifacts.textureEdits = Object.fromEntries(units.map((unit) => {
      const previous = previousResults[unit.id];
      if (!previous) return [unit.id, emptyTextureResult(unit.id)];
      // A persisted `running` result means the owning process disappeared before
      // it could finish. Retrying it automatically could create a second paid
      // image request, so require an explicit per-texture retry instead.
      return [unit.id, previous.status === "running"
        ? {
          ...previous,
          status: "cancelled" as const,
          error: "上次高保真编辑被中断，未自动重试，避免产生隐藏费用",
        }
        : previous];
    }));
    await this.dependencies.jobs.save(job);
    const pendingUnits = units.filter((unit) =>
      job.artifacts.textureEdits?.[unit.id]?.status === "pending"
    );
    const provider = pendingUnits.length > 0 ? await this.dependencies.providerFactory() : null;
    const controller = new AbortController();
    const operation: ActiveOperation = { controller, provider, revision: job.revision, requests: [] };
    this.active.set(job.id, operation);
    try {
      if (pendingUnits.length > 0) {
        const unitById = new Map(units.map((unit) => [unit.id, unit]));
        const canvases = planned.canvases;
        // What the first canvas painted, carried onto the later ones; a resumed
        // job picks it up from the canvas finished before the interruption.
        let anchor = firstCanvasResult(job, canvases);
        for (const [index, plan] of canvases.entries()) {
          const canvasUnits = plan.items.flatMap((item) => unitById.get(item.textureUnitId) ?? []);
          let current = await this.dependencies.jobs.load(job.id);
          if (!this.isCurrent(current, operation)) return current;
          const waiting = canvasUnits.filter((unit) => current.artifacts.textureEdits?.[unit.id]?.status === "pending");
          if (!waiting.length) continue;
          const label = canvases.length > 1 ? `第 ${index + 1}/${canvases.length} 张画布` : "联合画布";
          const work = await this.prepareTextureBatchWork(
            current,
            template,
            canvasUnits,
            plan,
            index > 0 ? anchor : null,
            canvases.length > 1 ? `texture-batch-${index + 1}` : "texture-batch",
          );
          current = await this.reserve(current, "texture-edit-batch", plan.estimatedUsd);
          if (current.paused) return current;
          const running = { ...current.artifacts.textureEdits };
          for (const unit of waiting) {
            const item = work.items[unit.id];
            if (!item) throw new Error(`${label}缺少纹理 ${unit.label}`);
            running[unit.id] = {
              ...(running[unit.id] ?? emptyTextureResult(unit.id)),
              status: "running",
              estimatedUsd: plan.estimatedUsd,
              sourcePath: item.sourcePath,
              workImagePath: work.imagePath,
              workMaskPath: work.maskPath,
              layout: item.layout,
              error: null,
            };
          }
          current.artifacts.textureEdits = running;
          const artifact: TextureEditBatchArtifact = {
            plan,
            requestId: null,
            endpoint: null,
            workImagePath: work.imagePath,
            workMaskPath: work.maskPath,
            rawResultPath: null,
            error: null,
          };
          current.artifacts.textureBatch = artifact;
          current.artifacts.textureBatches = withCanvasArtifact(current.artifacts.textureBatches, index, artifact);
          await this.dependencies.jobs.save(current);
          this.progress(
            current,
            0.12 + 0.8 * index / canvases.length,
            textureDrawingText(index, canvases.length),
          );
          try {
            const edited = await this.editTextureBatch(current, template, waiting, provider!, operation, work, {
              index,
              count: canvases.length,
              description: job.input.textureCanvasDescriptions?.[index] ?? job.input.description,
              anchored: Boolean(plan.anchor && index > 0 && anchor),
            });
            const raw = edited.artifacts.textureBatches?.[index]?.rawResultPath;
            if (!anchor && raw) anchor = { imagePath: raw, plan };
          } catch (error) {
            current = await this.dependencies.jobs.load(job.id);
            if (controller.signal.aborted) return current;
            const failed = { ...current.artifacts.textureEdits };
            for (const unit of waiting) {
              const result = failed[unit.id] ?? emptyTextureResult(unit.id);
              if (result.status === "running" || result.status === "pending") {
                failed[unit.id] = { ...result, status: "failed", error: errorMessage(error) };
              }
            }
            current.artifacts.textureEdits = failed;
            const failedArtifact = { ...(current.artifacts.textureBatches?.[index] ?? artifact), error: errorMessage(error) };
            current.artifacts.textureBatch = failedArtifact;
            current.artifacts.textureBatches = withCanvasArtifact(current.artifacts.textureBatches, index, failedArtifact);
            current.issues = [...current.issues, {
              code: "texture-batch-edit-failed",
              message: `${canvases.length > 1 ? label : "联合纹理编辑"}失败：${errorMessage(error)}`,
              slotId: null,
              recoverable: true,
            }];
            await this.dependencies.jobs.save(current);
          }
        }
      }
      let current = await this.dependencies.jobs.load(job.id);
      if (controller.signal.aborted || current.phase === "cancelled") return current;
      current = transitionCharacterJob(current, "decomposing");
      operation.revision = current.revision;
      current = transitionCharacterJob(current, "aligning");
      operation.revision = current.revision;
      current = transitionCharacterJob(current, "validating");
      operation.revision = current.revision;
      current = transitionCharacterJob(current, "preview");
      operation.revision = current.revision;
      await this.dependencies.jobs.save(current);
      const reviewCount = Object.values(current.artifacts.textureEdits ?? {}).filter((item) => item.status === "review").length;
      const failedCount = Object.values(current.artifacts.textureEdits ?? {}).filter((item) => item.status === "failed").length;
      const cancelledCount = Object.values(current.artifacts.textureEdits ?? {}).filter((item) => item.status === "cancelled").length;
      this.progress(
        current,
        1,
        failedCount || cancelledCount
          ? `画好了 ${reviewCount} 块，${failedCount + cancelledCount} 块没画成`
          : "画好了",
      );
      return current;
    } finally {
      if (this.active.get(job.id) === operation) this.active.delete(job.id);
    }
  }

  private async editTextureBatch(
    job: CharacterGenerationJob,
    template: MasterTemplatePackage,
    units: readonly TextureEditUnit[],
    provider: CharacterGenerationProvider,
    operation: ActiveOperation,
    work: TextureEditBatchWorkAsset,
    canvas: { index: number; count: number; description: string; anchored: boolean } = {
      index: 0,
      count: 1,
      description: job.input.description,
      anchored: false,
    },
  ): Promise<CharacterGenerationJob> {
    const action = textureDrawingText(canvas.index, canvas.count);
    const response = await provider.editMaskedTexture({
      prompt: textureBatchEditPrompt(
        units,
        work.plan,
        canvas.description,
        job.input.textureReferences ?? [],
        canvas.anchored,
      ),
      imagePath: work.imagePath,
      maskPath: work.maskPath,
      imageSize: work.imageSize,
      imageModel: work.plan.imageModel,
      referenceImagePaths: (job.input.textureReferences ?? []).map((reference) => reference.path),
      signal: operation.controller.signal,
      onSubmitted: (endpoint, requestId) =>
        this.trackSubmittedRequest(job.id, operation, endpoint, requestId),
      onStatus: (status) => this.progress(
        job,
        0.35,
        status.stage === "queued"
          ? `${action}，排队中${status.queuePosition ? `，前面还有 ${status.queuePosition} 个` : ""}`
          : action,
      ),
    });
    let current = await this.currentJob(job.id, operation);
    if (!this.isCurrent(current, operation)) return current;
    const endpoint = response.endpoint ?? "openai/gpt-image-2.5/sunburst/edit";
    current = this.addRequest(
      current,
      "texture-edit-batch",
      endpoint,
      response.requestId,
    );
    const resultUrl = response.urls[0];
    if (!resultUrl) throw new Error("图片服务没有返回联合纹理图片");
    const raw = await this.dependencies.artifacts.download(
      job.id,
      resultUrl,
      canvas.count > 1 ? `texture-batch-${canvas.index + 1}-raw` : "texture-batch-raw",
    );
    const artifact: TextureEditBatchArtifact = {
      ...(current.artifacts.textureBatches?.[canvas.index] ?? {
        plan: work.plan,
        workImagePath: work.imagePath,
        workMaskPath: work.maskPath,
        error: null,
      }),
      requestId: response.requestId,
      endpoint,
      rawResultPath: raw,
    };
    current.artifacts.textureBatch = artifact;
    current.artifacts.textureBatches = withCanvasArtifact(current.artifacts.textureBatches, canvas.index, artifact);
    const results = { ...current.artifacts.textureEdits };
    for (const unit of units) {
      const item = work.items[unit.id];
      const previous = results[unit.id] ?? emptyTextureResult(unit.id);
      if (!item) {
        results[unit.id] = { ...previous, status: "failed", error: "联合画布缺少该纹理" };
        continue;
      }
      try {
        const constrained = await this.dependencies.artifacts.constrainTextureEditResult(
          job.id,
          item.sourcePath,
          raw,
          path.join(template.root, unit.maskPath),
          item.layout,
          unit.label,
          `texture-${unit.id}-edited`,
          job.input.textureAlphaPolicy ?? textureEditAlphaPolicy(job.input.promptVisualSpec),
        );
        results[unit.id] = {
          textureUnitId: unit.id,
          status: "review",
          requestId: response.requestId,
          endpoint,
          estimatedUsd: work.plan.estimatedUsd,
          sourcePath: item.sourcePath,
          sourcePreviewPath: constrained.sourcePreviewPath,
          workImagePath: work.imagePath,
          workMaskPath: work.maskPath,
          rawResultPath: raw,
          editedPatchPath: constrained.path,
          editedPreviewPath: constrained.previewPath,
          layout: item.layout,
          quality: constrained.quality,
          error: null,
        };
      } catch (error) {
        results[unit.id] = {
          ...previous,
          requestId: response.requestId,
          endpoint,
          rawResultPath: raw,
          status: "failed",
          error: errorMessage(error),
        };
      }
    }
    current.artifacts.textureEdits = results;
    await this.dependencies.jobs.save(current);
    return current;
  }

  private async prepareTextureBatchWork(
    job: CharacterGenerationJob,
    template: MasterTemplatePackage,
    units: readonly TextureEditUnit[],
    plan: TextureEditBatchPlan,
    anchor: { imagePath: string; plan: TextureEditBatchPlan } | null = null,
    name = "texture-batch",
  ): Promise<TextureEditBatchWorkAsset> {
    const entries = await Promise.all(units.map(async (unit) => ({
      atlasPath: await this.resolveCurrentTextureAtlas(job, template, unit),
      unitMaskPath: path.join(template.root, unit.maskPath),
      modelMaskPath: path.join(template.root, unit.modelMaskPath),
      unit,
    })));
    return this.dependencies.artifacts.createTextureEditBatchWorkCanvas(
      job.id,
      entries,
      plan,
      name,
      anchor,
    );
  }

  private async resolveCurrentTextureAtlas(
    job: CharacterGenerationJob,
    template: MasterTemplatePackage,
    unit: TextureEditUnit,
  ): Promise<string> {
    if (job.input.sourceModelId && this.dependencies.resolveCurrentTextureAtlas) {
      return this.dependencies.resolveCurrentTextureAtlas(job.input.sourceModelId, unit.atlasFile);
    }
    if (unit.sourceAtlasFile) return path.join(template.root, unit.sourceAtlasFile);
    return path.join(template.root, unit.atlasFile);
  }

  private async generateEyePreview(job: CharacterGenerationJob): Promise<CharacterGenerationJob> {
    if (job.phase !== "draft" || job.paused) return job;
    if (!this.dependencies.templates || !this.dependencies.builder) {
      throw new Error("眼睛修改处理器尚未配置");
    }
    const template = await this.dependencies.templates.load(job.templateId);
    const eyeSlots = primaryEyeSlots(template.manifest.slots);
    const guidePath = path.join(template.root, template.manifest.alignmentGuide);
    // The full-body guide is intentionally not sent to fal. Qwen inpaint may
    // internally resize tall images and redraw unmasked pixels. A fixed 4:3
    // eye crop is both safer and the minimum billed megapixel operation.
    const estimatedCost = COSTS_USD.eyeRestylePerMegapixel;
    job = await this.reserve(job, "eye-restyle", estimatedCost);
    if (job.paused) return job;
    const provider = await this.dependencies.providerFactory();
    const controller = new AbortController();
    const operation: ActiveOperation = {
      controller,
      provider,
      revision: job.revision,
      requests: [],
    };
    this.active.set(job.id, operation);
    try {
      this.progress(job, 0.05, "正在建立眼睛安全编辑区；眼睛之外不会被修改");
      const eyeMask = await this.dependencies.artifacts.createSlotEditMask(
        job.id,
        guidePath,
        template.manifest.canvas,
        eyeSlots,
        "eye-edit-mask",
      );
      job.artifacts.eyeMask = eyeMask;
      job.artifacts.eyeSlotIds = eyeSlots.map((slot) => slot.id);
      await this.dependencies.jobs.save(job);
      const eyeCrop = await this.dependencies.artifacts.createEyeEditCrop(
        job.id,
        guidePath,
        eyeMask,
        "eye-edit-crop",
      );
      const result = await provider.inpaintImage({
        prompt: eyeRestylePrompt(job.input.description),
        negativePrompt: eyeRestyleNegativePrompt(),
        imagePath: eyeCrop.imagePath,
        maskPath: eyeCrop.maskPath,
        imageSize: "landscape_4_3",
        strength: 0.52,
        guidanceScale: 4,
        numInferenceSteps: 32,
        signal: controller.signal,
        onSubmitted: (endpoint, requestId) =>
          this.trackSubmittedRequest(job.id, operation, endpoint, requestId),
        onStatus: (status) => this.progress(
          job,
          0.2,
          providerStatusText("仅重绘左右虹膜、瞳孔与高光", status),
        ),
      });
      let current = await this.currentJob(job.id, operation);
      if (!this.isCurrent(current, operation)) return current;
      current = this.addRequest(
        current,
        "eye-restyle",
        "fal-ai/qwen-image-edit/inpaint",
        result.requestId,
      );
      const raw = await this.dependencies.artifacts.download(
        job.id,
        result.urls[0]!,
        "eye-edit-raw",
      );
      const constrained = await this.dependencies.artifacts.constrainEyeCropEdit(
        job.id,
        guidePath,
        raw,
        eyeMask,
        eyeCrop,
        "eye-edit-constrained",
      );
      current.artifacts.eyeEditedImage = constrained;
      current.artifacts.slotTextures = Object.fromEntries(
        eyeSlots.map((slot) => [slot.id, constrained]),
      );
      current = transitionCharacterJob(current, "decomposing");
      operation.revision = current.revision;
      await this.dependencies.jobs.save(current);
      current = transitionCharacterJob(current, "aligning");
      operation.revision = current.revision;
      await this.dependencies.jobs.save(current);
      this.progress(current, 0.82, "正在将眼睛安全区域写回原始 ArtMesh；其余贴图保持原样");
      const model = await this.dependencies.builder.build(
        current,
        template,
        `眼睛 · ${job.input.description.slice(0, 30)}`,
        {
          preserveUnchanged: true,
          deltaReferencePath: guidePath,
        },
      );
      if (current.outputModelId && current.outputModelId !== model.id) {
        await this.dependencies.deleteGeneratedModel?.(current.outputModelId);
      }
      current = transitionCharacterJob({ ...current, outputModelId: model.id }, "validating");
      operation.revision = current.revision;
      await this.dependencies.jobs.save(current);
      current = transitionCharacterJob(current, "preview");
      operation.revision = current.revision;
      await this.dependencies.jobs.save(current);
      this.progress(current, 1, "眼睛预览已准备好；请检查眨眼、视线移动和表情后再保存");
      return current;
    } catch (error) {
      const current = await this.dependencies.jobs.load(job.id);
      if (controller.signal.aborted || current.phase === "cancelled") return current;
      if (current.paused) return current;
      const failed = transitionCharacterJob(current, "failed");
      failed.issues = [...failed.issues, {
        code: "eye-restyle-failed",
        message: errorMessage(error),
        slotId: "eyes",
        recoverable: true,
      }];
      await this.dependencies.jobs.save(failed);
      this.progress(failed, 1, failed.issues.at(-1)!.message);
      return failed;
    } finally {
      if (this.active.get(job.id) === operation) this.active.delete(job.id);
    }
  }

  private async generatePartPreview(job: CharacterGenerationJob): Promise<CharacterGenerationJob> {
    if (job.phase !== "draft" || job.paused) return job;
    if (!this.dependencies.templates || !this.dependencies.builder) {
      throw new Error("模型部位修改处理器尚未配置");
    }
    if (job.input.editMode !== "appearance") {
      throw new Error("当前只允许修改原网格内的外观；轮廓修改需要先完成全动作扫描");
    }
    const template = await this.dependencies.templates.load(job.templateId);
    const partId = job.input.editPartId?.trim();
    if (!partId) throw new Error("没有选择需要修改的模型部位");
    const slots = template.manifest.slots.filter((slot) =>
      slot.editPartId === partId && slot.source === "visible"
    );
    if (slots.length === 0) throw new Error("所选部位没有可安全修改的可见 ArtMesh");
    const partLabel = partLabelFromSlots(slots, partId);
    try {
      assertAppearanceOnlyRequest(partLabel, job.input.description);
    } catch (error) {
      const failed = transitionCharacterJob(job, "failed");
      failed.issues = [...failed.issues, {
        code: "part-geometry-change-unsupported",
        message: errorMessage(error),
        slotId: partId,
        recoverable: true,
      }];
      await this.dependencies.jobs.save(failed);
      this.progress(failed, 1, failed.issues.at(-1)!.message);
      return failed;
    }
    const editProfile = partEditProfile(job.input.description);
    const guidePath = path.join(template.root, template.manifest.alignmentGuide);
    job = await this.reserve(job, `part-restyle:${partId}`, COSTS_USD.partRestylePerMegapixel);
    if (job.paused) return job;
    const provider = await this.dependencies.providerFactory();
    const controller = new AbortController();
    const operation: ActiveOperation = {
      controller,
      provider,
      revision: job.revision,
      requests: [],
    };
    this.active.set(job.id, operation);
    try {
      this.progress(job, 0.05, `正在从原始纹理和 UV 建立“${partLabel}”精确安全区`);
      const editMask = await this.dependencies.artifacts.createExactSlotEditMask(
        job.id,
        guidePath,
        template.root,
        template.manifest.canvas,
        template.manifest.textureAtlases,
        slots,
        "part-edit-mask",
        template.manifest.slots,
      );
      job.artifacts.editMask = editMask;
      job.artifacts.editSlotIds = slots.map((slot) => slot.id);
      await this.dependencies.jobs.save(job);
      const crop = await this.dependencies.artifacts.createMaskedEditCrop(
        job.id,
        guidePath,
        editMask,
        "part-edit-crop",
      );
      const result = await provider.inpaintImage({
        prompt: partRestylePrompt(partLabel, job.input.description, editProfile.intent),
        negativePrompt: partRestyleNegativePrompt(partLabel),
        imagePath: crop.imagePath,
        maskPath: crop.maskPath,
        imageSize: crop.output,
        strength: editProfile.strength,
        guidanceScale: editProfile.guidanceScale,
        numInferenceSteps: editProfile.numInferenceSteps,
        acceleration: "none",
        signal: controller.signal,
        onSubmitted: (endpoint, requestId) =>
          this.trackSubmittedRequest(job.id, operation, endpoint, requestId),
        onStatus: (status) => this.progress(
          job,
          0.2,
          providerStatusText(`仅重绘${partLabel}的原有可见像素`, status),
        ),
      });
      let current = await this.currentJob(job.id, operation);
      if (!this.isCurrent(current, operation)) return current;
      current = this.addRequest(
        current,
        "part-restyle",
        "fal-ai/qwen-image-edit/inpaint",
        result.requestId,
      );
      const raw = await this.dependencies.artifacts.download(
        job.id,
        result.urls[0]!,
        "part-edit-raw",
      );
      const constrained = await this.dependencies.artifacts.constrainMaskedCropEdit(
        job.id,
        guidePath,
        raw,
        editMask,
        crop,
        partLabel,
        "part-edit-constrained",
        editProfile.qualityMode,
      );
      current.artifacts.editedImage = constrained;
      current.artifacts.slotTextures = Object.fromEntries(
        slots.map((slot) => [slot.id, constrained]),
      );
      current = transitionCharacterJob(current, "decomposing");
      operation.revision = current.revision;
      await this.dependencies.jobs.save(current);
      current = transitionCharacterJob(current, "aligning");
      operation.revision = current.revision;
      await this.dependencies.jobs.save(current);
      this.progress(current, 0.82, `正在按原 UV 写回${partLabel}；透明度、网格和其他部位保持不变`);
      const model = await this.dependencies.builder.build(
        current,
        template,
        `${partLabel} · ${job.input.description.slice(0, 30)}`,
        {
          preserveUnchanged: true,
          deltaReferencePath: guidePath,
        },
      );
      if (current.outputModelId && current.outputModelId !== model.id) {
        await this.dependencies.deleteGeneratedModel?.(current.outputModelId);
      }
      current = transitionCharacterJob({ ...current, outputModelId: model.id }, "validating");
      operation.revision = current.revision;
      await this.dependencies.jobs.save(current);
      current = transitionCharacterJob(current, "preview");
      operation.revision = current.revision;
      await this.dependencies.jobs.save(current);
      this.progress(current, 1, `${partLabel}预览已准备好；请检查待机、动作和表情后再保存`);
      return current;
    } catch (error) {
      const current = await this.dependencies.jobs.load(job.id);
      if (controller.signal.aborted || current.phase === "cancelled") return current;
      if (current.paused) return current;
      const failed = transitionCharacterJob(current, "failed");
      failed.issues = [...failed.issues, {
        code: "part-restyle-failed",
        message: errorMessage(error),
        slotId: partId,
        recoverable: true,
      }];
      await this.dependencies.jobs.save(failed);
      this.progress(failed, 1, failed.issues.at(-1)!.message);
      return failed;
    } finally {
      if (this.active.get(job.id) === operation) this.active.delete(job.id);
    }
  }

  async selectCandidate(jobId: string, candidateId: string): Promise<CharacterGenerationJob> {
    const job = await this.dependencies.jobs.load(jobId);
    if (job.phase !== "draft") throw new Error("只能在候选阶段选择角色定稿");
    const candidate = job.candidates.find((item) => item.id === candidateId);
    if (!candidate) {
      throw new Error("选择的候选图不存在");
    }
    if (!candidate.compatible) {
      throw new Error(candidate.rejectionReason ?? "该候选与当前 Live2D 母版不兼容");
    }
    const selected = transitionCharacterJob({
      ...job,
      selectedCandidateId: candidateId,
      issues: [],
      artifacts: {
        normalizedImage: null,
        subjectImage: null,
        coarseLayers: [],
        completionSheets: {},
        slotTextures: {},
        repairSlotIds: [],
        requestedCompletions: [],
      },
    }, "decomposing");
    await this.dependencies.jobs.save(selected);
    this.progress(selected, 0, "正在使用角色定稿生成基础网格预览…");
    return selected;
  }

  async processSelected(jobId: string): Promise<CharacterGenerationJob> {
    return this.processSelectedV3(jobId);
  }

  private async processSelectedV3(jobId: string): Promise<CharacterGenerationJob> {
    if (!this.dependencies.templates || !this.dependencies.builder) {
      throw new Error("角色生成处理器尚未配置");
    }
    let job = await this.dependencies.jobs.load(jobId);
    if (!["decomposing", "completing"].includes(job.phase) || job.paused) return job;
    const isBaseProjection = job.phase === "decomposing";
    const selected = job.candidates.find((candidate) => candidate.id === job.selectedCandidateId);
    if (!selected) throw new Error("请先选择角色定稿");
    const template = await this.dependencies.templates.load(job.templateId);
    const needsBackgroundRemoval = isBaseProjection && !job.artifacts.subjectImage;
    const provider = needsBackgroundRemoval ||
        isBaseProjection || job.artifacts.requestedCompletions.length > 0
      ? await this.dependencies.providerFactory()
      : null;
    const controller = new AbortController();
    const operation: ActiveOperation = {
      controller,
      provider,
      revision: job.revision,
      requests: [],
    };
    this.active.set(jobId, operation);
    const submitted = (endpoint: string, requestId: string): Promise<void> =>
      this.trackSubmittedRequest(jobId, operation, endpoint, requestId);
    try {
      const alignmentGuide = path.join(template.root, template.manifest.alignmentGuide);
      if (isBaseProjection && !job.artifacts.normalizedImage) {
        if (!provider) throw new Error("人物母版对齐服务尚未配置");
        job = await this.reserve(job, "normalize", COSTS_USD.templateAlignment);
        if (job.paused) return job;
        const guideSize = await this.dependencies.artifacts.imageSize(alignmentGuide);
        const aligned = await provider.editImage({
          prompt: templateAlignmentPrompt(template.name, job.input.description),
          imagePaths: [selected.localPath, alignmentGuide],
          imageSize: guideSize,
          signal: controller.signal,
          onSubmitted: submitted,
          onStatus: (status) => this.progress(
            job,
            0.06,
            providerStatusText(`将角色精确对齐到 ${template.name}`, status),
          ),
        });
        job = await this.currentJob(jobId, operation);
        if (job.phase === "cancelled") return job;
        job = this.addRequest(job, "normalize", "fal-ai/qwen-image-2/edit", aligned.requestId);
        const normalizedImage = await this.dependencies.artifacts.download(
          jobId,
          aligned.urls[0]!,
          "aligned-to-template",
        );
        await this.dependencies.artifacts.assertSameImageSize(alignmentGuide, normalizedImage);
        job.artifacts.normalizedImage = normalizedImage;
        await this.dependencies.jobs.save(job);
      }

      if (isBaseProjection && !job.artifacts.subjectImage) {
        if (!provider) throw new Error("人物背景提取服务尚未配置");
        if (!job.artifacts.normalizedImage) throw new Error("缺少母版对齐后的角色图");
        const normalizedImage = job.artifacts.normalizedImage;
        job = await this.reserve(job, "remove-background", COSTS_USD.backgroundRemoval);
        if (job.paused) return job;
        const removed = await provider.removeBackground({
          imagePath: normalizedImage,
          signal: controller.signal,
          onSubmitted: submitted,
          onStatus: (status) => this.progress(job, 0.18, providerStatusText("提取对齐人物的透明边缘", status)),
        });
        job = await this.currentJob(jobId, operation);
        if (job.phase === "cancelled") return job;
        job = this.addRequest(job, "remove-background", "fal-ai/birefnet/v2", removed.requestId);
        const subjectImage = await this.dependencies.artifacts.download(
          jobId,
          removed.url,
          "subject-rgba",
        );
        await this.dependencies.artifacts.assertSameImageSize(normalizedImage, subjectImage);
        job.artifacts.subjectImage = subjectImage;
        await this.dependencies.jobs.save(job);
      }

      if (!job.artifacts.subjectImage) throw new Error("缺少基础人物透明图");
      const subjectImage = job.artifacts.subjectImage;

      if (isBaseProjection) {
        const coverage = await this.dependencies.artifacts.inspectTemplateCoverage(
          subjectImage,
          template.manifest.canvas,
          template.manifest.slots,
        );
        if (coverage.subjectPixels === 0) {
          throw new TemplateAlignmentError("没有检测到可投影的人物像素");
        }
        if (coverage.coveredRatio < 0.98) {
          const covered = Math.max(0, coverage.coveredRatio * 100).toFixed(1);
          job.issues = [...job.issues.filter((issue) => issue.code !== "projection-coverage-warning"), {
            code: "projection-coverage-warning",
            message: `基础网格覆盖人物 ${covered}%；超出网格的发梢或服装会在预览中被裁切，请以实际动画为准。`,
            slotId: null,
            recoverable: true,
          }];
        }
        job.artifacts.requestedCompletions = [];
        await this.dependencies.jobs.save(job);
      } else {
        if (!provider) throw new Error("局部补全服务尚未配置");
        const requested = job.artifacts.requestedCompletions ?? [];
        for (let index = 0; index < requested.length; index += 1) {
          const kind = requested[index]!;
          const pass = completionPass(kind);
          const artifactId = `completion:${kind}`;
          if (job.artifacts.completionSheets[artifactId]) continue;
          job = await this.reserve(job, `completion-${kind}`, COSTS_USD.completion);
          if (job.paused) return job;
          const completed = await provider.editImage({
            prompt: `图像 1 是已经通过基础网格对齐的标准角色。${pass.prompt}` +
              " 输出必须保持与图像 1 完全相同的画布尺寸、人物位置、身体比例和相机；" +
              "除指定的被遮挡素材外，所有可见像素、颜色、线稿和透明边界都必须保持不变。不得裁切、缩放或移动人物。",
            imagePaths: [subjectImage],
            signal: controller.signal,
            onSubmitted: submitted,
            onStatus: (status) => this.progress(
              job,
              0.18 + index / Math.max(1, requested.length) * 0.55,
              providerStatusText(`补全${pass.label}`, status),
            ),
          });
          job = await this.currentJob(jobId, operation);
          if (job.phase === "cancelled") return job;
          job = this.addRequest(job, `completion-${kind}`, "fal-ai/qwen-image-2/edit", completed.requestId);
          const sheet = await this.dependencies.artifacts.download(
            jobId,
            completed.urls[0]!,
            `completion-${kind}`,
          );
          await this.dependencies.artifacts.assertSameImageSize(subjectImage, sheet);
          job.artifacts.completionSheets[artifactId] = sheet;
          await this.dependencies.jobs.save(job);
        }
      }

      const slotTextures: Record<string, string> = {};
      // A neutral full-character image may only be projected into drawables
      // that are visible in the mother's neutral pose. Projecting it into
      // hidden/alternate meshes duplicates face, shoulder and clothing pixels
      // when Cubism later changes pose or opacity.
      for (const slot of template.manifest.slots.filter((item) => item.source === "visible")) {
        slotTextures[slot.id] = subjectImage;
      }
      if (!isBaseProjection) {
        for (const kind of job.artifacts.requestedCompletions ?? []) {
          const pass = completionPass(kind);
          const sheet = job.artifacts.completionSheets[`completion:${kind}`];
          if (!sheet) throw new Error(`缺少 ${pass.label} 补全素材`);
          for (const slot of template.manifest.slots.filter(pass.matches)) slotTextures[slot.id] = sheet;
        }
      }
      job.artifacts.slotTextures = slotTextures;
      const missing = validateRequiredSlots(template.manifest.slots, slotTextures);
      if (missing.length > 0) throw new Error(`缺少必要 Drawable: ${missing.join(", ")}`);
      job = transitionCharacterJob(job, "aligning");
      operation.revision = job.revision;
      await this.dependencies.jobs.save(job);
      this.progress(job, 0.76, `正在按 ${template.manifest.slots.length} 个 Cubism 网格烘焙纹理…`);

      const model = await this.dependencies.builder.build(
        job,
        template,
        job.input.description.slice(0, 40) || "AI 角色",
      );
      if (job.outputModelId && job.outputModelId !== model.id) {
        await this.dependencies.deleteGeneratedModel?.(job.outputModelId);
      }
      if (!isBaseProjection) {
        job = transitionCharacterJob(job, "validating");
        operation.revision = job.revision;
        await this.dependencies.jobs.save(job);
      }
      job = transitionCharacterJob(
        { ...job, outputModelId: model.id },
        isBaseProjection ? "base-preview" : "preview",
      );
      operation.revision = job.revision;
      await this.dependencies.jobs.save(job);
      this.progress(
        job,
        1,
        isBaseProjection
          ? `基础网格预览已准备好；完整人物图已反投影到 ${template.manifest.slots.length} 个 Cubism 网格，请检查中立外观、转头和四肢跟随`
          : "局部补全后的 Live2D 预览已准备好，请检查眨眼、口型、身体和遮挡动作",
      );
      return job;
    } catch (error) {
      const current = await this.dependencies.jobs.load(jobId);
      if (controller.signal.aborted || current.phase === "cancelled") return current;
      if (current.paused) return current;
      const failed = transitionCharacterJob(current, "failed");
      failed.issues = [...failed.issues, {
        code: error instanceof TemplateAlignmentError
          ? "template-alignment-failed"
          : "pipeline-v2-failed",
        message: errorMessage(error),
        slotId: null,
        recoverable: true,
      }];
      await this.dependencies.jobs.save(failed);
      this.progress(failed, 1, failed.issues.at(-1)!.message);
      return failed;
    } finally {
      if (this.active.get(jobId) === operation) this.active.delete(jobId);
    }
  }

  async beginCompletion(
    jobId: string,
    requestedKinds: CharacterCompletionKind[],
  ): Promise<CharacterGenerationJob> {
    const job = await this.dependencies.jobs.load(jobId);
    if (job.phase !== "base-preview" || !job.outputModelId) {
      throw new Error("请先完成并确认基础网格预览");
    }
    const requested = [...new Set(requestedKinds)];
    if (requested.some((kind) => !CHARACTER_COMPLETION_KINDS.includes(kind))) {
      throw new Error("包含未知的局部补全类型");
    }
    const completionSheets = Object.fromEntries(
      Object.entries(job.artifacts.completionSheets)
        .filter(([key]) => !key.startsWith("completion:")),
    );
    const completing = transitionCharacterJob({
      ...job,
      issues: [],
      artifacts: {
        ...job.artifacts,
        completionSheets,
        slotTextures: {},
        repairSlotIds: [],
        requestedCompletions: requested,
      },
    }, "completing");
    await this.dependencies.jobs.save(completing);
    this.progress(
      completing,
      0,
      requested.length === 0
        ? "不需要局部补全，正在生成最终预览"
        : `准备补全 ${requested.map((kind) => completionPass(kind).label).join("、")}`,
    );
    return completing;
  }

  async confirmSave(jobId: string): Promise<CharacterGenerationJob> {
    const job = await this.dependencies.jobs.load(jobId);
    if (job.phase !== "preview") throw new Error("没有可确认的预览模型");
    if (job.input.operation === "texture-edit") {
      throw new Error("纹理结果请通过“保存已采用结果并返回调整”完成");
    }
    if (!job.outputModelId) throw new Error("没有可确认的预览模型");
    const saved = transitionCharacterJob(job, "saved");
    await this.dependencies.jobs.save(saved);
    this.progress(saved, 1, "角色已保存到模型库");
    await this.dependencies.artifacts.removeJob(jobId);
    return saved;
  }

  async saveAcceptedTexturesAndCancel(jobId: string, apply = true): Promise<CharacterGenerationJob> {
    const existing = this.textureReviewFinishes.get(jobId);
    if (existing) return existing;
    const finishing = this.finishTextureReview(jobId, apply);
    this.textureReviewFinishes.set(jobId, finishing);
    try {
      return await finishing;
    } finally {
      if (this.textureReviewFinishes.get(jobId) === finishing) {
        this.textureReviewFinishes.delete(jobId);
      }
    }
  }

  private async finishTextureReview(jobId: string, apply: boolean): Promise<CharacterGenerationJob> {
    let job = await this.dependencies.jobs.load(jobId);
    if (job.input.operation !== "texture-edit" || job.phase !== "preview") {
      throw new Error("当前任务不是可返回调整的纹理编辑任务");
    }
    const acceptedResultCount = Object.values(job.artifacts.textureEdits ?? {})
      .filter((result) => result.status === "accepted").length;
    const accepted = acceptedResultCount > 0 ? await this.acceptedTextureEdits(job) : [];
    if (accepted.length !== acceptedResultCount) {
      throw new Error("已采用纹理的生成素材不完整，无法安全保存");
    }
    if (accepted.length > 0) {
      if (!this.dependencies.textureMaterials) throw new Error("当前模型素材库尚未配置");
      if (!job.input.sourceModelId) throw new Error("纹理任务缺少源模型");
      await this.dependencies.textureMaterials.save({
        modelId: job.input.sourceModelId,
        name: "AI 素材",
        // The user's words name the version in 换色; the compiled prompt is internal.
        prompt: job.input.originalDescription?.trim() || job.input.description,
        edits: accepted,
        apply,
      });
    }
    if (job.outputModelId) await this.dependencies.deleteGeneratedModel?.(job.outputModelId);
    job = transitionCharacterJob({
      ...job,
      outputModelId: null,
      artifacts: {
        ...job.artifacts,
        textureEdits: discardUnacceptedTextureResults(
          job.artifacts.textureEdits,
          "返回调整，未采用结果已丢弃",
        ),
      },
    }, "cancelled");
    await this.dependencies.jobs.save(job);
    await this.dependencies.artifacts.removeJob(jobId);
    this.progress(
      job,
      1,
      accepted.length > 0
        ? `已保存并应用 ${accepted.length} 个纹理，其他结果已丢弃`
        : "未采用任何纹理，任务已取消并保留原始调整内容",
    );
    return job;
  }

  async acceptTexture(
    jobId: string,
    textureUnitId: string,
    accepted: boolean,
  ): Promise<CharacterTexturePreviewChange> {
    let job = await this.dependencies.jobs.load(jobId);
    if (job.input.operation !== "texture-edit" || job.phase !== "preview") {
      throw new Error("当前任务不是可确认的纹理编辑任务");
    }
    const current = job.artifacts.textureEdits?.[textureUnitId];
    if (!current || !["review", "accepted"].includes(current.status) || !current.editedPatchPath) {
      throw new Error("该纹理没有可接受的合格结果");
    }
    job.artifacts.textureEdits = {
      ...job.artifacts.textureEdits,
      [textureUnitId]: { ...current, status: accepted ? "accepted" : "review" },
    };
    await this.dependencies.jobs.save(job);
    return this.buildAcceptedTexturePreview(job, textureUnitId);
  }

  async texturePreview(jobId: string, enabled: boolean): Promise<TextureAtlasDelta> {
    const job = await this.dependencies.jobs.load(jobId);
    if (job.input.operation !== "texture-edit" || job.phase !== "preview") {
      throw new Error("当前任务没有可用的纹理动画预览");
    }
    if (!this.dependencies.templates) throw new Error("纹理母版加载器尚未配置");
    const template = await this.dependencies.templates.load(job.templateId);
    const units = resolveTextureUnits(template, job.input.textureUnitIds ?? []);
    return this.composeTexturePreview(job, units, enabled);
  }

  private async acceptedTextureEdits(job: CharacterGenerationJob) {
    if (!this.dependencies.templates) throw new Error("当前模型素材库尚未配置");
    const template = await this.dependencies.templates.load(job.templateId);
    const units = resolveTextureUnits(template, job.input.textureUnitIds ?? []);
    return units.flatMap((unit) => {
      const result = job.artifacts.textureEdits?.[unit.id];
      return result?.status === "accepted" && result.editedPatchPath
        ? [{
          textureUnitId: unit.id,
          label: unit.label,
          atlasIndex: unit.atlasIndex,
          atlasFile: unit.atlasFile,
          bounds: unit.bounds,
          maskPath: path.join(template.root, unit.maskPath),
          editedPatchPath: result.editedPatchPath,
        }]
        : [];
    });
  }

  private async buildAcceptedTexturePreview(
    job: CharacterGenerationJob,
    changedTextureUnitId: string,
  ): Promise<CharacterTexturePreviewChange> {
    if (!this.dependencies.templates || !this.dependencies.texturePreview) {
      throw new Error("纹理预览热更新器尚未配置");
    }
    if (!job.input.sourceModelId) throw new Error("纹理任务缺少源模型");
    const template = await this.dependencies.templates.load(job.templateId);
    const units = resolveTextureUnits(template, job.input.textureUnitIds ?? []);
    const changedUnit = units.find((unit) => unit.id === changedTextureUnitId);
    if (!changedUnit) throw new Error("变更的纹理已失效");
    if (job.outputModelId) await this.dependencies.deleteGeneratedModel?.(job.outputModelId);
    const updated = { ...job, outputModelId: null, updatedAt: new Date().toISOString() };
    await this.dependencies.jobs.save(updated);
    const delta = await this.composeTexturePreview(job, units, true);
    const acceptedCount = Object.values(job.artifacts.textureEdits ?? {})
      .filter((item) => item.status === "accepted").length;
    this.progress(updated, 1, `已热更新 ${acceptedCount} 个采用纹理到独立动画预览`);
    return { job: updated, delta };
  }

  private async composeTexturePreview(
    job: CharacterGenerationJob,
    affectedUnits: readonly TextureEditUnit[],
    includeAccepted: boolean,
  ): Promise<TextureAtlasDelta> {
    if (!this.dependencies.templates || !this.dependencies.texturePreview) {
      throw new Error("纹理预览热更新器尚未配置");
    }
    if (!job.input.sourceModelId) throw new Error("纹理任务缺少源模型");
    const template = await this.dependencies.templates.load(job.templateId);
    const allUnits = resolveTextureUnits(template, job.input.textureUnitIds ?? []);
    const edits = includeAccepted ? allUnits.flatMap((unit) => {
      const result = job.artifacts.textureEdits?.[unit.id];
      return result?.status === "accepted" && result.editedPatchPath
        ? [{
          textureUnitId: unit.id,
          atlasIndex: unit.atlasIndex,
          atlasFile: unit.atlasFile,
          bounds: unit.bounds,
          maskPath: path.join(template.root, unit.maskPath),
          editedPatchPath: result.editedPatchPath,
        }]
        : [];
    }) : [];
    const affectedAtlases = [...new Map(affectedUnits.map((unit) => [unit.atlasFile, {
      atlasIndex: unit.atlasIndex,
      atlasFile: unit.atlasFile,
    }])).values()];
    return this.dependencies.texturePreview.snapshot({
      jobId: job.id,
      modelId: job.input.sourceModelId,
      revision: job.revision,
      enabled: includeAccepted,
      affectedAtlases,
      edits,
    });
  }

  async pause(jobId: string): Promise<CharacterGenerationJob> {
    const operation = this.active.get(jobId);
    operation?.controller.abort(new DOMException("Paused", "AbortError"));
    if (operation) {
      await Promise.allSettled(operation.requests.map((request) =>
        operation.provider?.cancel(request.endpoint, request.requestId)
      ));
    }
    const job = await this.dependencies.jobs.load(jobId);
    if (["saved", "cancelled"].includes(job.phase)) return job;
    const textureEdits = job.input.operation === "texture-edit"
      ? mapInterruptedTextureResults(job.artifacts.textureEdits, "任务已暂停，未自动重试以避免隐藏费用")
      : job.artifacts.textureEdits;
    const paused = {
      ...job,
      paused: true,
      pauseReason: "user" as const,
      artifacts: { ...job.artifacts, textureEdits },
      updatedAt: new Date().toISOString(),
    };
    await this.dependencies.jobs.save(paused);
    this.progress(paused, 0, "任务已暂停");
    return paused;
  }

  async resume(jobId: string): Promise<CharacterGenerationJob> {
    const job = await this.dependencies.jobs.load(jobId);
    if (!job.paused) return job;
    const restartPipeline = job.phase === "aligning" || job.phase === "validating";
    const resumed: CharacterGenerationJob = {
      ...job,
      phase: restartPipeline ? "decomposing" : job.phase,
      paused: false,
      pauseReason: null,
      revision: job.revision + 1,
      budget: { ...job.budget, pendingOperation: null },
      updatedAt: new Date().toISOString(),
    };
    await this.dependencies.jobs.save(resumed);
    this.progress(resumed, 0, "任务继续");
    return resumed;
  }

  async approveBudget(jobId: string, limitUsd: number): Promise<CharacterGenerationJob> {
    const job = await this.dependencies.jobs.load(jobId);
    if (!Number.isFinite(limitUsd) || limitUsd <= job.budget.limitUsd || limitUsd > 10) {
      throw new Error("新的费用上限必须高于当前上限且不超过 10 USD");
    }
    const approved = {
      ...job,
      phase: job.phase === "aligning" || job.phase === "validating"
        ? "decomposing" as const
        : job.phase,
      budget: { ...job.budget, limitUsd, pendingOperation: null },
      paused: false,
      pauseReason: null,
      revision: job.revision + 1,
      updatedAt: new Date().toISOString(),
    };
    await this.dependencies.jobs.save(approved);
    return approved;
  }

  async retry(jobId: string): Promise<CharacterGenerationJob> {
    const job = await this.dependencies.jobs.load(jobId);
    if (job.phase !== "failed") throw new Error("只有失败任务可以重试");
    if (job.input.operation === "eye-restyle" || job.input.operation === "part-restyle") {
      const retried = transitionCharacterJob({
        ...job,
        outputModelId: null,
        issues: [],
        artifacts: resetAlignedArtifacts(job),
      }, "draft");
      await this.dependencies.jobs.save(retried);
      return retried;
    }
    const regenerateCandidates = job.issues.some((issue) =>
      ["template-alignment-failed", "no-template-compatible-candidates"].includes(issue.code)
    );
    const phase = regenerateCandidates ? "draft" : job.selectedCandidateId ? "decomposing" : "draft";
    const retried = transitionCharacterJob({
      ...job,
      candidates: regenerateCandidates || !job.selectedCandidateId ? [] : job.candidates,
      selectedCandidateId: regenerateCandidates ? null : job.selectedCandidateId,
      outputModelId: regenerateCandidates ? null : job.outputModelId,
      issues: [],
      artifacts: regenerateCandidates ? resetAlignedArtifacts(job) : job.artifacts,
    }, phase);
    await this.dependencies.jobs.save(retried);
    return retried;
  }

  async regenerateCandidates(jobId: string): Promise<CharacterGenerationJob> {
    const job = await this.dependencies.jobs.load(jobId);
    if (job.phase !== "draft" || job.paused) {
      throw new Error("只能在候选选择阶段重新生成角色");
    }
    const regenerated: CharacterGenerationJob = {
      ...job,
      candidates: [],
      selectedCandidateId: null,
      outputModelId: null,
      issues: [],
      artifacts: resetAlignedArtifacts(job),
      revision: job.revision + 1,
      updatedAt: new Date().toISOString(),
    };
    await this.dependencies.jobs.save(regenerated);
    return regenerated;
  }

  async regeneratePart(jobId: string, slotId: string): Promise<CharacterGenerationJob> {
    const job = await this.dependencies.jobs.load(jobId);
    if (job.input.operation === "eye-restyle" || job.input.operation === "part-restyle") {
      const expected = job.input.operation === "eye-restyle" ? "eyes" : job.input.editPartId;
      if (slotId !== expected || job.phase !== "preview") {
        throw new Error("只能从当前部位预览重新生成同一部位");
      }
      if (job.outputModelId) await this.dependencies.deleteGeneratedModel?.(job.outputModelId);
      const restarted: CharacterGenerationJob = {
        ...job,
        phase: "draft",
        outputModelId: null,
        paused: false,
        pauseReason: null,
        revision: job.revision + 1,
        issues: [],
        artifacts: resetAlignedArtifacts(job),
        updatedAt: new Date().toISOString(),
      };
      await this.dependencies.jobs.save(restarted);
      return restarted;
    }
    if (!["base-preview", "preview"].includes(job.phase) && job.pauseReason !== "needs-review") {
      throw new Error("只能从预览或部件检查阶段返工");
    }
    if (!this.dependencies.templates) throw new Error("角色母版加载器尚未配置");
    const template = await this.dependencies.templates.load(job.templateId);
    const realignToTemplate = slotId === REALIGN_TO_TEMPLATE_ID;
    const repairAllMaterials = slotId === ALL_MATERIALS_REPAIR_ID;
    const slot = template.manifest.slots.find((item) => item.id === slotId);
    const requestedKind = CHARACTER_COMPLETION_KINDS.includes(slotId as CharacterCompletionKind)
      ? slotId as CharacterCompletionKind
      : slot
        ? COMPLETION_PASSES.find((pass) => pass.matches(slot))?.id ?? null
        : null;
    if (!repairAllMaterials && !realignToTemplate && !requestedKind) {
      throw new Error("需要返工的部件不存在");
    }
    const affectedSlots = new Set(template.manifest.slots
      .filter((item) => repairAllMaterials || realignToTemplate ||
        (requestedKind !== null && completionPass(requestedKind).matches(item)))
      .map((item) => item.id));
    const slotTextures = Object.fromEntries(Object.entries(job.artifacts.slotTextures)
      .filter(([id]) => !affectedSlots.has(id)));
    const completionSheets = { ...job.artifacts.completionSheets };
    if (repairAllMaterials || realignToTemplate) {
      for (const key of Object.keys(completionSheets)) delete completionSheets[key];
    } else if (requestedKind) {
      delete completionSheets[`completion:${requestedKind}`];
    }
    const restarted: CharacterGenerationJob = {
      ...job,
      phase: realignToTemplate ? "draft" : repairAllMaterials ? "decomposing" : "completing",
      candidates: realignToTemplate ? [] : job.candidates,
      selectedCandidateId: realignToTemplate ? null : job.selectedCandidateId,
      outputModelId: realignToTemplate || repairAllMaterials ? null : job.outputModelId,
      paused: false,
      pauseReason: null,
      revision: job.revision + 1,
      cache: {},
      artifacts: realignToTemplate || repairAllMaterials ? resetAlignedArtifacts(job) : {
        ...job.artifacts,
        completionSheets,
        slotTextures,
        repairSlotIds: [...new Set([...job.artifacts.repairSlotIds, slotId])],
        requestedCompletions: !requestedKind ? [] : [requestedKind],
      },
      issues: repairAllMaterials || realignToTemplate
        ? []
        : job.issues.filter((issue) => issue.slotId !== slotId),
      updatedAt: new Date().toISOString(),
    };
    await this.dependencies.jobs.save(restarted);
    if ((realignToTemplate || repairAllMaterials) && job.outputModelId) {
      await this.dependencies.deleteGeneratedModel?.(job.outputModelId);
    }
    return restarted;
  }

  async cancel(jobId: string): Promise<CharacterGenerationJob> {
    const operation = this.active.get(jobId);
    operation?.controller.abort(new DOMException("Cancelled", "AbortError"));
    let job = await this.dependencies.jobs.load(jobId);
    if (job.phase === "cancelled" || job.phase === "saved") return job;
    const requests = [
      ...job.falRequests.filter((request) => request.revision === job.revision)
        .map(({ endpoint, requestId }) => ({ endpoint, requestId })),
      ...(operation?.requests ?? []),
    ].filter((request, index, values) =>
      values.findIndex((candidate) =>
        candidate.endpoint === request.endpoint && candidate.requestId === request.requestId
      ) === index
    );
    if (requests.length > 0) {
      const cancelProvider = operation?.provider ?? await this.dependencies.providerFactory();
      await Promise.allSettled(requests.map((request) =>
        cancelProvider.cancel(request.endpoint, request.requestId)
      ));
    }
    const outputModelId = job.outputModelId;
    const textureEdits = job.input.operation === "texture-edit"
      ? mapCancelledTextureResults(job.artifacts.textureEdits, "整个纹理编辑任务已取消")
      : job.artifacts.textureEdits;
    job = transitionCharacterJob({
      ...job,
      outputModelId: null,
      artifacts: { ...job.artifacts, textureEdits },
    }, "cancelled");
    await this.dependencies.jobs.save(job);
    if (outputModelId) await this.dependencies.deleteGeneratedModel?.(outputModelId);
    await this.dependencies.artifacts.removeJob(jobId);
    this.progress(job, 1, "任务已取消");
    return job;
  }

  private isCurrent(job: CharacterGenerationJob, operation: ActiveOperation): boolean {
    return job.phase !== "cancelled" && job.revision === operation.revision && !operation.controller.signal.aborted;
  }

  private async currentJob(jobId: string, operation: ActiveOperation): Promise<CharacterGenerationJob> {
    const job = await this.dependencies.jobs.load(jobId);
    if (!this.isCurrent(job, operation)) return job;
    return job;
  }

  private async reserve(
    job: CharacterGenerationJob,
    name: string,
    amount: number,
  ): Promise<CharacterGenerationJob> {
    const reserved = reserveEstimatedCost(job, name, amount);
    await this.dependencies.jobs.save(reserved);
    if (reserved.paused) this.progress(reserved, 0, "预计费用超过任务上限，等待确认");
    return reserved;
  }

  private addRequest(
    job: CharacterGenerationJob,
    operation: string,
    endpoint: string,
    requestId: string,
  ): CharacterGenerationJob {
    const existing = job.falRequests.findIndex((request) =>
      request.endpoint === endpoint && request.requestId === requestId
    );
    if (existing >= 0) {
      return {
        ...job,
        falRequests: job.falRequests.map((request, index) => index === existing
          ? { ...request, operation }
          : request),
      };
    }
    return {
      ...job,
      falRequests: [...job.falRequests, {
        operation,
        endpoint,
        requestId,
        revision: job.revision,
      }],
    };
  }

  private async trackSubmittedRequest(
    jobId: string,
    active: ActiveOperation,
    endpoint: string,
    requestId: string,
  ): Promise<void> {
    active.requests.push({ endpoint, requestId });
    const current = await this.dependencies.jobs.load(jobId);
    if (!this.isCurrent(current, active)) return;
    await this.dependencies.jobs.save(this.addRequest(
      current,
      "in-flight",
      endpoint,
      requestId,
    ));
  }

  private progress(job: CharacterGenerationJob, progress: number, detail: string): void {
    this.emit("progress", {
      jobId: job.id,
      phase: job.phase,
      progress,
      detail,
      job,
    } satisfies CharacterGenerationProgress);
  }
}

function flattenSlots(slots: MasterTemplatePackage["manifest"]["slots"]): MasterTemplatePackage["manifest"]["slots"] {
  return slots;
}

function emptyTextureResult(textureUnitId: string): TextureEditResult {
  return {
    textureUnitId,
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
  };
}

function mapInterruptedTextureResults(
  results: Record<string, TextureEditResult> | undefined,
  error: string,
): Record<string, TextureEditResult> {
  return Object.fromEntries(Object.entries(results ?? {}).map(([id, result]) => [
    id,
    result.status === "running"
      ? { ...result, status: "cancelled" as const, error }
      : result,
  ]));
}

function mapCancelledTextureResults(
  results: Record<string, TextureEditResult> | undefined,
  error: string,
): Record<string, TextureEditResult> {
  return Object.fromEntries(Object.entries(results ?? {}).map(([id, result]) => [
    id,
    ["pending", "running"].includes(result.status)
      ? { ...result, status: "cancelled" as const, error }
      : result,
  ]));
}

function discardUnacceptedTextureResults(
  results: Record<string, TextureEditResult> | undefined,
  error: string,
): Record<string, TextureEditResult> {
  return Object.fromEntries(Object.entries(results ?? {}).map(([id, result]) => [
    id,
    result.status === "accepted"
      ? result
      : { ...result, status: "cancelled" as const, error },
  ]));
}

function resolveTextureUnits(
  template: MasterTemplatePackage,
  selectedIds: string[],
): TextureEditUnit[] {
  const requested = [...new Set(selectedIds)];
  const byId = new Map<string, MasterTemplateSlot[]>();
  for (const slot of template.manifest.slots) {
    if (!slot.textureUnitId) continue;
    const slots = byId.get(slot.textureUnitId) ?? [];
    slots.push(slot);
    byId.set(slot.textureUnitId, slots);
  }
  return requested.map((id) => {
    const slots = byId.get(id);
    if (!slots?.length) throw new Error(`选择的纹理素材不存在或能力清单已过期: ${id}`);
    const first = slots[0]!;
    const atlas = template.manifest.textureAtlases[first.atlas.texture];
    if (!atlas) throw new Error(`纹理 ${id} 指向不存在的图集`);
    const minX = Math.min(...slots.map((slot) => slot.canvasBounds.x));
    const minY = Math.min(...slots.map((slot) => slot.canvasBounds.y));
    const maxX = Math.max(...slots.map((slot) => slot.canvasBounds.x + slot.canvasBounds.width));
    const maxY = Math.max(...slots.map((slot) => slot.canvasBounds.y + slot.canvasBounds.height));
    return {
      id,
      atlasIndex: first.atlas.texture,
      atlasFile: atlas.file,
      maskHash: id.split("-").at(-1) ?? id,
      bounds: { x: first.atlas.x, y: first.atlas.y, width: first.atlas.width, height: first.atlas.height },
      maskPath: first.silhouetteMask,
      previewPath: first.silhouetteMask,
      // Service reconstruction does not consume the neutral-pose locator; the
      // optimizer reads its exact path from the capability manifest instead.
      modelMaskPath: first.silhouetteMask,
      modelMaskBounds: {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
        canvasWidth: 1,
        canvasHeight: 1,
      },
      drawableIds: slots.map((slot) => slot.drawableId),
      partIds: [...new Set(slots.map((slot) => slot.partId))],
      semanticPartIds: [...new Set(slots.flatMap((slot) => slot.editPartId ? [slot.editPartId] : []))],
      // Template slots do not carry motion visibility metadata. The UI-facing
      // capability manifest is the authoritative source for these annotations.
      revealedByMotionIds: [],
      label: first.name.split(" · ")[0]?.trim() || id,
      visiblePixels: 0,
      modelBounds: { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
    };
  });
}

function textureBatchEditPrompt(
  units: readonly TextureEditUnit[],
  plan: TextureEditBatchPlan,
  description: string,
  references: ReadonlyArray<NonNullable<CharacterGenerationJob["input"]["textureReferences"]>[number]> = [],
  anchored = false,
): string {
  const anchor = anchored && plan.anchor
    ? `画布 x=${plan.anchor.x}, y=${plan.anchor.y}, 宽=${plan.anchor.width}, 高=${plan.anchor.height} 处的缩略图是同一次修改在上一张画布上已经画好的结果，只读、不可修改、不要复制进其他区域；本张各区域的配色、花纹尺度、线稿粗细和材质表现必须与它一致`
    : null;
  const regions = textureEditBatchPromptRegions(plan).map((region) => {
    if (!units.some((unit) => unit.id === region.textureUnitId)) {
      throw new Error("联合画布包含已失效的纹理区域");
    }
    return `区域 ${region.key} 位于画布 x=${region.x}, y=${region.y}, 宽=${region.workWidth}, 高=${region.workHeight}；具体部位语义只采用最终提示词中的视觉识别结果`;
  }).join("；");
  const hasReferenceSection = /(^|\n)参考(?:图)?：/.test(description);
  const hasGlobalConstraint = /(^|\n)(?:统一要求|技术约束)：/.test(description);
  const hasAlphaConstraint = /(^|\n)(?:透明要求：|.*(?:Alpha|透明轮廓))/.test(description);
  return [
    "第 1 张图片是 Live2D UV 纹理联合编辑画布，多个编辑区域彼此独立；后续图片全部是只读参考图",
    regions,
    anchor,
    textureReferenceDeliveryPrompt(references, !hasReferenceSection),
    description.trim().slice(0, 6_000),
    "只修改第 1 张画布中合并蒙版允许的像素，间距、未选区域以及各区域之间的空白必须逐像素保持不变；不得把任何参考图内容整体复制进画布",
    "画布中每个纹理外侧的彩色框和 A/B/C/D 标签位于不可编辑的保护留白中，不得修改或把标签复制到纹理中",
    hasGlobalConstraint ? null : "保持每个纹理的原位置、比例、线稿结构和连接关系；除用户明确要求删除的部分外，完整覆盖原可见像素并保持外轮廓，不得缩小、偏移、串色或露出画布背景",
    hasAlphaConstraint ? null : "仅在用户明确要求删除内容时使用真实 PNG 透明度，其余像素保持原始透明度；不得用白底、黑底、棋盘格或背景色伪装透明，透明轮廓不能向外扩大",
    "输出边缘必须清晰且可直接用于纹理：禁止用户未要求的异常污点、背景侵入、白边、双重边缘、额外物体、文字或标志；用户明确要求的做旧、颗粒、水彩或磨损等视觉效果不属于异常，不得被本规则抵消",
    "除用户明确要求的变化外，不得擅自改变任何纹理的颜色、材质、阴影或外观",
  ].filter((item): item is string => Boolean(item)).join("；");
}

/** The first canvas that came back with a result, as the anchor for the rest. */
function firstCanvasResult(
  job: CharacterGenerationJob,
  canvases: readonly TextureEditBatchPlan[],
): { imagePath: string; plan: TextureEditBatchPlan } | null {
  for (const [index, plan] of canvases.entries()) {
    const artifact = job.artifacts.textureBatches?.[index];
    if (artifact?.rawResultPath && artifact.plan.hash === plan.hash) return { imagePath: artifact.rawResultPath, plan };
  }
  return null;
}

function withCanvasArtifact(
  list: readonly TextureEditBatchArtifact[] | undefined,
  index: number,
  artifact: TextureEditBatchArtifact,
): TextureEditBatchArtifact[] {
  const next = [...(list ?? [])];
  next[index] = artifact;
  return next;
}

function textureReferenceDeliveryPrompt(
  references: ReadonlyArray<NonNullable<CharacterGenerationJob["input"]["textureReferences"]>[number]>,
  includeDetails: boolean,
): string {
  if (references.length === 0) return "本次没有额外参考图，只依据用户要求和原纹理进行修改";
  const labels = textureReferenceLabels(references);
  return references.map((reference, index) => {
    const promptId = reference.promptId ?? labels[index];
    const inputNumber = index + 2;
    if (reference.deliveryUse === "supplemental" && reference.source === "model-preview") {
      return `P0（输入图片第 ${inputNumber} 张）：用户确认发送的当前模型位置参考，不参与外观覆盖`;
    }
    if (reference.deliveryUse === "supplemental") {
      return `${promptId}（输入图片第 ${inputNumber} 张）：用户确认发送的补充只读参考，不得覆盖已确认视觉方向`;
    }
    if (reference.source === "model-preview") {
      return `P0（输入图片第 ${inputNumber} 张）：优化提示词实际使用的当前模型正面外观参考`;
    }
    if (!includeDetails) return `${promptId}（输入图片第 ${inputNumber} 张）：优化提示词实际使用的外观参考`;
    if (reference.source === "texture-unit") {
      return `参考 ${promptId}（输入图片第 ${inputNumber} 张）：模型素材“${reference.name}”`;
    }
    const description = reference.description.trim()
      ? `；用户说明：${reference.description.trim().slice(0, 300)}`
      : "";
    return `参考 ${promptId}（输入图片第 ${inputNumber} 张）：用户上传图片“${reference.name}”${description}`;
  }).join("；");
}

function textureImageModel(job: CharacterGenerationJob): TextureEditImageModel {
  return resolveTextureEditImageModel(job.input.textureImageModel);
}

function templateAlignmentPrompt(templateName: string, description: string): string {
  return `图1是已选角色，图2是 ${templateName} 的中立几何参考。` +
    "只使用图2的画布、正面站姿、头身比例、人物外轮廓、头顶、下巴、双眼、嘴、肩、肘、腕、髋、膝、踝和鞋底坐标；" +
    "完整保留图1的二维日系动漫画风、脸部身份、发型、服装类型和配色，但所有可见轮廓必须收敛到图2现有网格范围内。" +
    "若图1的发型或服装轮廓超出图2，只能在不改变主要外观特征的前提下缩短或收拢，禁止复制图2的人物身份、颜色、道具或特效。" +
    "双眼和嘴必须落在图2对应位置，四肢必须完整分离；输出与图2完全相同尺寸的纯白背景单人正面全身图，不得增加文字、阴影或额外人物。" +
    `用户要求：${description.trim().slice(0, 180)}`;
}

/** What the person sees while a texture edit draws: which canvas, nothing about services or requests. */
function textureDrawingText(index: number, count: number): string {
  return count > 1 ? `正在画第 ${index + 1} 张，共 ${count} 张` : "正在画";
}

function providerStatusText(action: string, status: CharacterProviderStatus): string {
  const elapsed = formatElapsed(status.elapsedMs);
  if (status.stage === "queued") {
    const position = status.queuePosition === null ? "" : `，前方约 ${status.queuePosition} 个任务`;
    return `${action} · fal 排队中${position} · 已等待 ${elapsed}`;
  }
  if (status.stage === "completed") return `${action} · 图片服务已完成 · 用时 ${elapsed}`;
  const message = status.message ? ` · ${status.message}` : "";
  return `${action} · fal 正在推理 · 已用 ${elapsed}${message}`;
}

function formatElapsed(elapsedMs: number): string {
  const seconds = Math.max(0, Math.floor(elapsedMs / 1_000));
  if (seconds < 60) return `${seconds} 秒`;
  return `${Math.floor(seconds / 60)} 分 ${String(seconds % 60).padStart(2, "0")} 秒`;
}

function resetAlignedArtifacts(job: CharacterGenerationJob): CharacterGenerationJob["artifacts"] {
  return {
    ...job.artifacts,
    normalizedImage: null,
    subjectImage: null,
    coarseLayers: [],
    completionSheets: {},
    slotTextures: {},
    repairSlotIds: [],
    requestedCompletions: [],
    eyeMask: null,
    eyeEditedImage: null,
    eyeSlotIds: [],
    editMask: null,
    editedImage: null,
    editSlotIds: [],
  };
}

function primaryEyeSlots(slots: MasterTemplateSlot[]): MasterTemplateSlot[] {
  const candidates = slots.filter((slot) =>
    /eye.?ball/i.test(slot.partId.replaceAll("_", "")) && slot.source === "visible"
  );
  const pick = (side: "left" | "right"): MasterTemplateSlot | null => {
    const sideSlots = candidates.filter((slot) => {
      const centerX = slot.canvasBounds.x + slot.canvasBounds.width / 2;
      return side === "left" ? centerX < 0 : centerX >= 0;
    });
    return sideSlots.sort((left, right) =>
      right.canvasBounds.width * right.canvasBounds.height -
      left.canvasBounds.width * left.canvasBounds.height
    )[0] ?? null;
  };
  const selected = [pick("left"), pick("right")].filter(
    (slot): slot is MasterTemplateSlot => slot !== null,
  );
  if (selected.length !== 2) {
    throw new Error("当前母版没有可安全识别的左右虹膜主网格，不能自动修改眼睛");
  }
  return selected;
}

function eyeRestylePrompt(description: string): string {
  return [
    "只修改白色蒙版内的左右虹膜、瞳孔和眼睛高光",
    `眼睛要求：${description.trim().slice(0, 220)}`,
    "左右眼必须属于同一角色，颜色、瞳孔大小、虹膜纹理和高光方向严格对称一致",
    "严格保持原眼睛位置、眼距、眼型、大小、视线方向、眼白、上下眼皮、睫毛、眉毛和脸部完全不变",
    "保持原始二次元线稿、上色方式、光照和清晰度，不增加文字、符号或额外瞳孔",
  ].join("；");
}

function eyeRestyleNegativePrompt(): string {
  return [
    "photo, photorealistic, realistic skin, pores, photographic lighting, 3d render",
    "dirt, stain, speckle, noise, grain, watercolor, muddy color, uneven shading, blur",
    "change face shape, move eyes, resize eyes, asymmetric eyes, crossed eyes",
    "extra pupil, missing pupil, malformed iris, text, watermark",
  ].join(", ");
}

function partLabelFromSlots(slots: MasterTemplateSlot[], fallback: string): string {
  const value = slots[0]?.name.split(" · ")[0]?.trim();
  return value || fallback;
}

type PartEditIntent = "cleanup" | "recolor" | "texture";

interface PartEditProfile {
  intent: PartEditIntent;
  strength: number;
  guidanceScale: number;
  numInferenceSteps: number;
  qualityMode: "preserve" | "restyle";
}

function partEditProfile(description: string): PartEditProfile {
  if (/(?:去除|去掉|删除|清除|擦除|移除|消除|不要|remove|erase|delete|clean\s*up)/i.test(description)) {
    return {
      intent: "cleanup",
      strength: 0.34,
      guidanceScale: 3.5,
      numInferenceSteps: 30,
      qualityMode: "preserve",
    };
  }
  if (/(?:红色?|橙色?|黄色?|绿色?|青色?|蓝色?|紫色?|粉色?|黑色?|白色?|灰色?|棕色?|褐色?|栗色?|金色?|银色?|颜色|colour|color)/i.test(description)) {
    return {
      intent: "recolor",
      strength: 0.42,
      guidanceScale: 3.5,
      numInferenceSteps: 30,
      qualityMode: "preserve",
    };
  }
  return {
    intent: "texture",
    strength: 0.55,
    guidanceScale: 4,
    numInferenceSteps: 34,
    qualityMode: "restyle",
  };
}

function partRestylePrompt(
  label: string,
  description: string,
  intent: PartEditIntent,
): string {
  const intentRule = intent === "cleanup"
    ? "只清除指定污渍、涂鸦或图案，用周围相邻的干净平涂色和原线稿连续补齐；不得重画整个部位"
    : intent === "recolor"
      ? "只替换指定颜色，逐像素保留原有线稿、阴影边界、高光、褶皱和材质细节；不得重新设计或重画部位"
      : "在保留原有线稿、阴影边界、褶皱、清晰度和赛璐璐上色方式的前提下修改纹理";
  return [
    `只修改白色蒙版内的${label}可见纹理`,
    `修改要求：${description.trim().slice(0, 220)}`,
    intentRule,
    `严格保持${label}的原始位置、外轮廓、透明边缘、面积、朝向和已有结构不变`,
    "输出必须是干净锐利的二次元 Live2D 贴图：平滑纯净色块、稳定线稿、无颗粒、无污点、无斑驳、无写实皮肤纹理、无水彩笔触",
    "严格保持蒙版外所有像素、其他部位、人物身份、姿势、线稿风格、光照和清晰度完全不变",
    "不得扩大轮廓、补画到透明区域、移动部位、改变人物身份或增加文字和额外物体",
  ].join("；");
}

function partRestyleNegativePrompt(label: string): string {
  return [
    "photo, photorealistic, realistic skin, pores, photographic lighting, 3d render, CGI",
    "dirt, dirty texture, stains, blemishes, freckles, speckles, noise, grain, scratches",
    "watercolor, oil paint, painterly brush, muddy color, uneven shading, color banding",
    "blur, soft focus, low resolution, compression artifacts, broken line art, doubled edges",
    `redraw whole ${label}, change silhouette, change anatomy, extra object, text, watermark`,
  ].join(", ").slice(0, 500);
}

function assertAppearanceOnlyRequest(label: string, description: string): void {
  const value = description.trim();
  const structuralNoun = "皇冠|王冠|帽子|礼帽|发型|长发|短发|马尾|辫子|刘海|裙子|裤子|鞋子|靴子|外套|上衣|眼镜|翅膀|尾巴|耳朵|蝴蝶结|配饰|crown|hat|hairstyle|ponytail|braid|skirt|pants|shoes|boots|glasses|wings|tail";
  const replacesStructure = new RegExp(
    `(?:变成|变为|换成|替换为|改成|增加|添加|加上|新增|replace(?:\\s+with)?|turn\\s+into|add)\\s*.{0,12}(?:${structuralNoun})`,
    "i",
  ).test(value);
  const changesDimensions = /(?:加长|变长|更长|缩短|变短|更短|变大|放大|变小|缩小|扩大|收窄|变宽|更宽|改变轮廓|改变形状|longer|shorter|larger|smaller|wider|narrower|silhouette|reshape)/i.test(value);
  if (!replacesStructure && !changesDimensions) return;
  throw new Error(
    `“${value.slice(0, 40)}”会改变${label}的轮廓或结构；当前只能安全修改原网格内的颜色和纹理。` +
    "本次未调用 fal，也不会产生费用。",
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function completionPass(kind: CharacterCompletionKind): (typeof COMPLETION_PASSES)[number] {
  const pass = COMPLETION_PASSES.find((candidate) => candidate.id === kind);
  if (!pass) throw new Error(`未知局部补全类型: ${kind}`);
  return pass;
}
