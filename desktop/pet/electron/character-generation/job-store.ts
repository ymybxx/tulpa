import { randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  CHARACTER_GENERATION_PHASES,
  DEFAULT_CHARACTER_BUDGET_USD,
  reconcileCharacterJobEstimate,
  resolveTextureEditImageModel,
  textureEditAlphaPolicy,
  type CharacterGenerationJob,
  type TextureEditImageModel,
  type TextureReferenceSnapshot,
  type TextureReferencePromptId,
  type TextureVisualEditSpec,
} from "../../src/shared/character-generation";

export interface CreateCharacterJobInput {
  operation?: "eye-restyle" | "part-restyle" | "texture-edit" | "character";
  templateId: string;
  description: string;
  referenceImage: string | null;
  sourceModelId: string | null;
  editPartId?: string | null;
  editMode?: "appearance" | "contour" | null;
  textureUnitIds?: string[];
  originalDescription?: string;
  optimizedDescription?: string | null;
  promptOptimization?: "not-requested" | "accepted" | "rejected";
  promptVisualSpec?: TextureVisualEditSpec | null;
  textureBatchPlanHash?: string;
  textureCanvasDescriptions?: string[];
  textureImageModel?: TextureEditImageModel;
  textureReferences?: TextureReferenceSnapshot[];
  budgetLimitUsd?: number;
}

export class CharacterGenerationJobStore {
  private readonly saveQueues = new Map<string, Promise<void>>();

  constructor(
    readonly root: string,
    private readonly createId: () => string = randomUUID,
    private readonly now: () => Date = () => new Date(),
  ) {}

  async create(input: CreateCharacterJobInput): Promise<CharacterGenerationJob> {
    const description = input.description.trim();
    if (!description) throw new Error("请输入角色描述");
    const timestamp = this.now().toISOString();
    const job: CharacterGenerationJob = {
      version: 2,
      id: this.createId(),
      templateId: input.templateId,
      input: {
        operation: input.operation ?? "character",
        description,
        referenceImage: input.referenceImage,
        sourceModelId: input.sourceModelId,
        editPartId: input.editPartId ?? null,
        editMode: input.editMode ?? null,
        textureUnitIds: [...new Set(input.textureUnitIds ?? [])],
        textureAlphaPolicy: input.operation === "texture-edit"
          ? textureEditAlphaPolicy(input.promptOptimization === "accepted" ? input.promptVisualSpec : null)
          : undefined,
        textureBatchMode: input.operation === "texture-edit" ? "joint-v1" : undefined,
        textureBatchPlanHash: input.textureBatchPlanHash,
        ...input.operation === "texture-edit" && input.textureCanvasDescriptions?.length
          ? { textureCanvasDescriptions: input.textureCanvasDescriptions.map((text) => text.trim().slice(0, 8_000)) }
          : {},
        textureImageModel: input.operation === "texture-edit"
          ? resolveTextureEditImageModel(input.textureImageModel)
          : undefined,
        originalDescription: input.operation === "texture-edit"
          ? (input.originalDescription ?? description).trim()
          : undefined,
        optimizedDescription: input.operation === "texture-edit"
          ? input.optimizedDescription?.trim() || null
          : undefined,
        promptOptimization: input.operation === "texture-edit"
          ? (input.promptOptimization ?? "not-requested")
          : undefined,
        promptVisualSpec: input.operation === "texture-edit" && input.promptVisualSpec
          ? structuredClone(input.promptVisualSpec)
          : null,
        textureReferences: input.operation === "texture-edit"
          ? (input.textureReferences ?? []).map((reference) => ({ ...reference }))
          : undefined,
      },
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
        eyeMask: null,
        eyeEditedImage: null,
        eyeSlotIds: [],
        editMask: null,
        editedImage: null,
        editSlotIds: [],
        textureEdits: {},
        textureBatch: null,
        textureBatches: [],
      },
      budget: {
        limitUsd: input.budgetLimitUsd ?? DEFAULT_CHARACTER_BUDGET_USD,
        estimatedUsd: 0,
        actualUsd: 0,
        pendingOperation: null,
      },
      issues: [],
      outputModelId: null,
      createdAt: timestamp,
      updatedAt: timestamp,
    };
    await this.save(job);
    return job;
  }

  async save(job: CharacterGenerationJob): Promise<CharacterGenerationJob> {
    validateJob(job);
    const previous = this.saveQueues.get(job.id) ?? Promise.resolve();
    const current = previous.catch(() => undefined).then(() => this.writeJob(job));
    this.saveQueues.set(job.id, current);
    try {
      await current;
    } finally {
      if (this.saveQueues.get(job.id) === current) this.saveQueues.delete(job.id);
    }
    return job;
  }

  private async writeJob(job: CharacterGenerationJob): Promise<void> {
    const directory = path.join(this.root, job.id);
    await mkdir(directory, { recursive: true });
    const target = path.join(directory, "job.json");
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, `${JSON.stringify(job, null, 2)}\n`, { mode: 0o600 });
    await rename(temporary, target);
  }

  async load(id: string): Promise<CharacterGenerationJob> {
    if (!/^[a-zA-Z0-9_-]+$/.test(id)) throw new Error("任务 ID 无效");
    const value = JSON.parse(await readFile(path.join(this.root, id, "job.json"), "utf8")) as unknown;
    validateJob(value);
    const job = value as CharacterGenerationJob;
    job.artifacts.requestedCompletions ??= [];
    job.artifacts.eyeMask ??= null;
    job.artifacts.eyeEditedImage ??= null;
    job.artifacts.eyeSlotIds ??= [];
    job.artifacts.editMask ??= null;
    job.artifacts.editedImage ??= null;
    job.artifacts.editSlotIds ??= [];
    job.artifacts.textureEdits ??= {};
    job.artifacts.textureBatch ??= null;
    job.artifacts.textureBatches ??= job.artifacts.textureBatch ? [job.artifacts.textureBatch] : [];
    let referenceIndex = 0;
    for (const [index, reference] of (job.input.textureReferences ?? []).entries()) {
      const legacy = reference as typeof reference & {
        purpose?: string;
        instruction?: string;
        source?: "model-preview" | "upload" | "texture-unit";
        description?: string;
      };
      legacy.source ??= /当前正面图/.test(legacy.name)
        ? "model-preview"
        : legacy.name.includes(" · ")
          ? "texture-unit"
          : "upload";
      legacy.description = legacy.source === "upload"
        ? (legacy.description ?? legacy.instruction ?? "").trim().slice(0, 300)
        : "";
      const fallbackPromptId = legacy.source === "model-preview"
        ? "P0"
        : `R${++referenceIndex}`;
      legacy.promptId ??= fallbackPromptId as TextureReferencePromptId;
      legacy.deliveryUse ??= "required";
      delete legacy.purpose;
      delete legacy.instruction;
    }
    if (
      job.input.operation === "texture-edit" &&
      (
        !["locked", "shrink-only"].includes(job.input.textureAlphaPolicy ?? "") ||
        job.input.textureBatchMode !== "joint-v1"
      ) &&
      !["saved", "cancelled"].includes(job.phase)
    ) {
      job.phase = "cancelled";
      job.paused = false;
      job.pauseReason = null;
      if (!job.issues.some((issue) => issue.code === "texture-batch-policy-outdated")) {
        job.issues.push({
          code: "texture-batch-policy-outdated",
          message: "此纹理任务创建于联合画布升级之前，请重新选择纹理并创建任务",
          slotId: null,
          recoverable: false,
        });
      }
    }
    return reconcileCharacterJobEstimate(job);
  }

  async list(): Promise<CharacterGenerationJob[]> {
    await mkdir(this.root, { recursive: true });
    const entries = await readdir(this.root, { withFileTypes: true });
    const values = await Promise.all(entries.filter((entry) => entry.isDirectory()).map(async (entry) => {
      try {
        return await this.load(entry.name);
      } catch {
        return null;
      }
    }));
    return values.filter((value): value is CharacterGenerationJob => value !== null)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async listResumable(): Promise<CharacterGenerationJob[]> {
    return (await this.list()).filter((job) =>
      !["saved", "cancelled"].includes(job.phase) &&
      job.input.operation !== "eye-restyle" && job.input.operation !== "part-restyle"
    );
  }
}

function validateJob(value: unknown): asserts value is CharacterGenerationJob {
  if (!isRecord(value) || value.version !== 2 || typeof value.id !== "string") {
    throw new Error("角色生成任务格式无效");
  }
  if (!CHARACTER_GENERATION_PHASES.includes(value.phase as CharacterGenerationJob["phase"])) {
    throw new Error("角色生成任务阶段无效");
  }
  if (!isRecord(value.input) || typeof value.input.description !== "string") {
    throw new Error("角色生成任务输入无效");
  }
  if (!isRecord(value.budget) || typeof value.budget.limitUsd !== "number") {
    throw new Error("角色生成任务预算无效");
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
