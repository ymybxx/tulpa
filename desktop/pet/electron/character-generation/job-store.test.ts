import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { CharacterGenerationJobStore } from "./job-store";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("CharacterGenerationJobStore", () => {
  it.each(["gpt-image-1.5", "gpt-image-2"] as const)("preserves saved %s job contents when loading after migration", async (legacyModel) => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-jobs-"));
    roots.push(root);
    const store = new CharacterGenerationJobStore(root, () => "legacy-job", () => new Date(0));
    const job = await store.create({
      operation: "texture-edit",
      templateId: "model-edit-mao",
      description: "旧任务原始提示词",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["hat"],
    });
    job.phase = "saved";
    job.input.textureImageModel = legacyModel;
    job.input.textureAlphaPolicy = "locked";
    job.budget.estimatedUsd = 0.16;
    await store.save(job);
    const file = path.join(root, job.id, "job.json");
    const before = await readFile(file, "utf8");

    const loaded = await store.load(job.id);

    expect(loaded.input).toEqual(job.input);
    expect(loaded.budget).toEqual(job.budget);
    expect(loaded.artifacts).toEqual(job.artifacts);
    expect(await readFile(file, "utf8")).toBe(before);
  });

  it("persists and resumes unfinished jobs without secrets", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-jobs-"));
    roots.push(root);
    const store = new CharacterGenerationJobStore(root, () => "job-1", () => new Date(0));
    const created = await store.create({
      templateId: "girl-v1",
      description: "少女",
      referenceImage: null,
      sourceModelId: null,
    });

    expect(created.phase).toBe("draft");
    expect((await store.listResumable()).map((job) => job.id)).toEqual(["job-1"]);
    const raw = await readFile(path.join(root, "job-1", "job.json"), "utf8");
    expect(raw).not.toMatch(/FAL_KEY|fal-secret|token/i);

    await store.save({ ...created, phase: "saved", outputModelId: "model-1" });
    expect(await store.listResumable()).toEqual([]);
  });

  it("does not resume texture edits created before the joint batch policy", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-jobs-"));
    roots.push(root);
    const store = new CharacterGenerationJobStore(root, () => "texture-job", () => new Date(0));
    const created = await store.create({
      operation: "texture-edit",
      templateId: "model-edit-mao",
      description: "去掉袜子",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["leg"],
    });
    expect(created.input.textureImageModel).toBe("gpt-image-2.5-sunburst");
    expect(created.input.textureAlphaPolicy).toBe("shrink-only");

    delete created.input.textureAlphaPolicy;
    await store.save(created);
    const loaded = await store.load(created.id);

    expect(loaded.phase).toBe("cancelled");
    expect(loaded.issues).toContainEqual(expect.objectContaining({
      code: "texture-batch-policy-outdated",
      recoverable: false,
    }));
    expect(await store.listResumable()).toEqual([]);
  });

  it.each(["gpt-image-1.5", "gpt-image-2"] as const)("migrates new jobs requested with legacy %s without a transparency switch", async (legacyModel) => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-jobs-"));
    roots.push(root);
    const store = new CharacterGenerationJobStore(root, () => "texture-job-15", () => new Date(0));
    const created = await store.create({
      operation: "texture-edit",
      templateId: "model-edit-mao",
      description: "去掉袜子",
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["leg"],
      textureImageModel: legacyModel,
    });

    expect(created.input.textureImageModel).toBe("gpt-image-2.5-sunburst");
    expect(created.input.textureAlphaPolicy).toBe("shrink-only");
  });

  it("persists the accepted visual specification with the texture job", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-jobs-"));
    roots.push(root);
    const store = new CharacterGenerationJobStore(root, () => "visual-spec-job", () => new Date(0));
    const visualSpec = {
      intent: "recolor" as const,
      styleProfiles: [{
        id: "S1",
        scope: "白色头发",
        attributes: {
          palette: { description: "纯白主色与冷灰阴影" },
        },
      }],
      regions: [{
        key: "A",
        role: "刘海",
        styleProfileIds: ["S1"],
        clauseIds: ["C1"],
      }],
      warnings: [],
    };
    const created = await store.create({
      operation: "texture-edit",
      templateId: "model-edit-mao",
      description: "compiled",
      originalDescription: "纯白色头发",
      optimizedDescription: "compiled",
      promptOptimization: "accepted",
      promptVisualSpec: visualSpec,
      referenceImage: null,
      sourceModelId: "mao",
      textureUnitIds: ["bangs"],
    });

    expect(created.input.textureAlphaPolicy).toBe("locked");
    expect(created.input.promptVisualSpec).toEqual(visualSpec);
    expect((await store.load(created.id)).input.promptVisualSpec).toEqual(visualSpec);
  });

  it("normalizes legacy reference-purpose records without cancelling the task", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-jobs-"));
    roots.push(root);
    const store = new CharacterGenerationJobStore(root, () => "legacy-position", () => new Date(0));
    const created = await store.create({
      operation: "texture-edit",
      templateId: "model-edit-hiyori",
      description: "浅粉蕾丝风格",
      referenceImage: null,
      sourceModelId: "hiyori",
      textureUnitIds: ["shoe"],
      textureBatchPlanHash: "plan",
      textureReferences: [{
        id: "front",
        source: "model-preview",
        name: "hiyori 当前正面图",
        description: "",
        path: "/snapshot/front.png",
      }],
    });
    const legacy = created.input.textureReferences?.[0] as unknown as Record<string, unknown>;
    delete legacy.source;
    delete legacy.description;
    legacy.purpose = "overall-style";
    legacy.instruction = "错误地参考外观";
    await store.save(created);

    const loaded = await store.load(created.id);

    expect(loaded.phase).toBe("draft");
    expect(loaded.input.textureReferences?.[0]).toMatchObject({
      source: "model-preview",
      description: "",
      promptId: "P0",
      deliveryUse: "required",
    });
    expect(loaded.input.textureReferences?.[0]).not.toHaveProperty("purpose");
    expect(loaded.input.textureReferences?.[0]).not.toHaveProperty("instruction");
  });
});
