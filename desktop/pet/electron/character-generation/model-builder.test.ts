import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { PNG } from "pngjs";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ModelPackage } from "../../src/shared/contracts";
import type { CharacterGenerationJob } from "../../src/shared/character-generation";
import type { MasterTemplatePackage } from "./master-template-store";
import { GeneratedModelBuilder } from "./model-builder";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("GeneratedModelBuilder", () => {
  it("replaces fixed atlas slots and restores the previously selected model", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "model-builder-"));
    roots.push(root);
    const template = await createTemplate(root);
    const slotPath = path.join(root, "body.png");
    await writePng(slotPath, 2, 2, Array(4).fill([20, 40, 60, 255]).flat());
    const previous = { id: "previous" } as ModelPackage;
    const generated = { id: "generated" } as ModelPackage;
    const importFolder = vi.fn(async (folder: string) => {
      const atlas = PNG.sync.read(await readFile(path.join(folder, "textures", "atlas.png")));
      const effects = PNG.sync.read(await readFile(path.join(folder, "textures", "effects.png")));
      expect([...atlas.data.slice(0, 4)]).toEqual([20, 40, 60, 255]);
      expect([...effects.data]).toEqual(Array(16).fill(0));
      await expect(readFile(path.join(folder, ".edit-capabilities", "manifest.json")))
        .rejects.toMatchObject({ code: "ENOENT" });
      return { model: generated, report: { status: "valid", checks: [], warnings: [], errors: [] } } as const;
    });
    const select = vi.fn(async () => previous);
    const builder = new GeneratedModelBuilder(path.join(root, "outputs"), {
      selected: vi.fn(async () => previous),
      importFolder,
      select,
    });

    const model = await builder.build(job(slotPath), template, "AI 少女");

    expect(model).toBe(generated);
    expect(importFolder).toHaveBeenCalledOnce();
    expect(select).toHaveBeenCalledWith("previous");
  });

  it("preserves every atlas that is not part of an eye-only restyle", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "model-builder-"));
    roots.push(root);
    const template = await createTemplate(root);
    const generated = { id: "generated" } as ModelPackage;
    const importFolder = vi.fn(async (folder: string) => {
      const atlas = PNG.sync.read(await readFile(path.join(folder, "textures", "atlas.png")));
      const effects = PNG.sync.read(await readFile(path.join(folder, "textures", "effects.png")));
      expect([...atlas.data]).toEqual(Array(16).fill(0));
      expect([...effects.data]).toEqual(Array(4).fill([80, 90, 100, 255]).flat());
      return { model: generated, report: { status: "valid", checks: [], warnings: [], errors: [] } } as const;
    });
    const builder = new GeneratedModelBuilder(path.join(root, "outputs"), {
      selected: vi.fn(async () => null),
      importFolder,
      select: vi.fn(async () => generated),
    });
    const eyeJob = job("");
    eyeJob.artifacts.slotTextures = {};

    await builder.build(eyeJob, template, "Eyes", { preserveUnchanged: true });

    expect(importFolder).toHaveBeenCalledOnce();
  });

  it("writes masked RGBA, permits alpha shrink, blocks expansion and preserves unselected bytes", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "model-builder-"));
    roots.push(root);
    const template = await createTemplate(root);
    const atlasPath = path.join(template.root, "textures", "atlas.png");
    const original = [
      10, 20, 30, 40, 50, 60, 70, 80,
      90, 100, 110, 120, 130, 140, 150, 160,
    ];
    await writePng(atlasPath, 2, 2, original);
    const patchPath = path.join(root, "patch.png");
    const maskPath = path.join(root, "unit-mask.png");
    await writePng(patchPath, 2, 1, [
      201, 202, 203, 10,
      211, 212, 213, 200,
    ]);
    await writePng(maskPath, 2, 1, [
      255, 255, 255, 64,
      255, 255, 255, 255,
    ]);
    const generated = { id: "generated" } as ModelPackage;
    const importFolder = vi.fn(async (folder: string) => {
      const atlas = PNG.sync.read(await readFile(path.join(folder, "textures", "atlas.png")));
      expect([...atlas.data]).toEqual([
        201, 202, 203, 10,
        211, 212, 213, 80,
        90, 100, 110, 120,
        130, 140, 150, 160,
      ]);
      return { model: generated, report: { status: "valid", checks: [], warnings: [], errors: [] } } as const;
    });
    const builder = new GeneratedModelBuilder(path.join(root, "outputs"), {
      selected: vi.fn(async () => null),
      importFolder,
      select: vi.fn(async () => generated),
    });
    const textureJob = job("");
    textureJob.artifacts.slotTextures = {};

    await builder.build(textureJob, template, "Texture", {
      preserveUnchanged: true,
      textureEdits: [{
        textureUnitId: "unit-1",
        atlasIndex: 0,
        bounds: { x: 0, y: 0, width: 2, height: 1 },
        maskPath,
        editedPatchPath: patchPath,
      }],
    });

    expect(importFolder).toHaveBeenCalledOnce();
  });

  it("bakes an explicitly resolved compiled appearance and removes copied appearance state", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "model-builder-"));
    roots.push(root);
    const template = await createTemplate(root);
    const appearanceRoot = path.join(template.root, ".appearance");
    await mkdir(appearanceRoot, { recursive: true });
    await writeFile(path.join(appearanceRoot, "library.json"), JSON.stringify({
      schemaVersion: 2,
      modelId: "source-model",
      variants: [],
      activeAssignments: {},
    }));
    await writePng(
      path.join(root, "compiled-atlas.png"),
      2,
      2,
      [
        1, 2, 3, 255, 4, 5, 6, 255,
        7, 8, 9, 255, 10, 11, 12, 255,
      ],
    );
    const patchPath = path.join(root, "appearance-patch.png");
    const maskPath = path.join(root, "appearance-mask.png");
    await writePng(patchPath, 1, 1, [201, 202, 203, 255]);
    await writePng(maskPath, 1, 1, [255, 255, 255, 255]);
    const generated = { id: "generated" } as ModelPackage;
    const importFolder = vi.fn(async (folder: string) => {
      const atlas = PNG.sync.read(await readFile(path.join(folder, "textures", "atlas.png")));
      expect([...atlas.data]).toEqual([
        201, 202, 203, 255, 4, 5, 6, 255,
        7, 8, 9, 255, 10, 11, 12, 255,
      ]);
      await expect(readFile(path.join(folder, ".appearance", "runtime", "textures", "atlas.png")))
        .rejects.toMatchObject({ code: "ENOENT" });
      await expect(readFile(path.join(folder, ".appearance", "library.json")))
        .rejects.toMatchObject({ code: "ENOENT" });
      return { model: generated, report: { status: "valid", checks: [], warnings: [], errors: [] } } as const;
    });
    const resolveCompiledAtlas = vi.fn(async (_modelId: string, atlasFile: string) =>
      atlasFile === "textures/atlas.png" ? path.join(root, "compiled-atlas.png") : null
    );
    const builder = new GeneratedModelBuilder(path.join(root, "outputs"), {
      selected: vi.fn(async () => null),
      importFolder,
      select: vi.fn(async () => generated),
    }, resolveCompiledAtlas);
    const textureJob = job("");
    textureJob.input.sourceModelId = "source-model";
    textureJob.artifacts.slotTextures = {};

    await builder.build(textureJob, template, "Texture preview", {
      preserveUnchanged: true,
      textureEdits: [{
        textureUnitId: "unit-1",
        atlasIndex: 0,
        bounds: { x: 0, y: 0, width: 1, height: 1 },
        maskPath,
        editedPatchPath: patchPath,
      }],
    });

    expect(importFolder).toHaveBeenCalledOnce();
    expect(resolveCompiledAtlas).toHaveBeenCalledWith("source-model", "textures/atlas.png");
  });
});

async function createTemplate(root: string): Promise<MasterTemplatePackage> {
  const templateRoot = path.join(root, "template");
  await mkdir(path.join(templateRoot, "textures"), { recursive: true });
  await mkdir(path.join(templateRoot, "masks"), { recursive: true });
  await mkdir(path.join(templateRoot, ".edit-capabilities"), { recursive: true });
  await writePng(path.join(templateRoot, "textures", "atlas.png"), 2, 2, Array(16).fill(0));
  await writePng(path.join(templateRoot, "textures", "effects.png"), 2, 2, Array(4).fill([80, 90, 100, 255]).flat());
  await writePng(path.join(templateRoot, "masks", "body.png"), 2, 2, [
    0, 0, 0, 255, 0, 0, 0, 255,
    0, 0, 0, 255, 0, 0, 0, 255,
  ]);
  await writeFile(path.join(templateRoot, "girl.model3.json"), "{}");
  await writeFile(path.join(templateRoot, "pet.manifest.json"), JSON.stringify({ name: "template" }));
  await writeFile(path.join(templateRoot, "master.template.json"), "{}");
  await writeFile(path.join(templateRoot, ".edit-capabilities", "manifest.json"), "{}");
  return {
    id: "girl-v1",
    name: "Girl",
    root: templateRoot,
    manifest: {
      schemaVersion: 2,
      id: "girl-v1",
      name: "Girl",
      modelEntry: "girl.model3.json",
      alignmentGuide: "alignment-guide.png",
      coverage: "full-body",
      canvas: { width: 2, height: 2, originX: 1, originY: 1, pixelsPerUnit: 1 },
      parameters: { eyeBlink: [], lipSync: [] },
      textureAtlases: [
        { file: "textures/atlas.png", width: 2, height: 2 },
        { file: "textures/effects.png", width: 2, height: 2 },
      ],
      slots: [{
        id: "body",
        name: "Body",
        drawableId: "D_BODY",
        partId: "PART_BODY",
        group: "body",
        source: "completion",
        required: true,
        zIndex: 1,
        atlas: { texture: 0, x: 0, y: 0, width: 2, height: 2 },
        anchor: { x: 1, y: 1 },
        canvasBounds: { x: -1, y: -1, width: 2, height: 2 },
        mesh: {
          vertices: [-1, 1, 1, 1, 1, -1, -1, -1],
          uvs: [0, 1, 1, 1, 1, 0, 0, 0],
          indices: [0, 1, 2, 0, 2, 3],
        },
        silhouetteMask: "masks/body.png",
        completionPrompt: "complete",
      }, {
        id: "effect",
        name: "Template effect",
        drawableId: "D_EFFECT",
        partId: "PART_EFFECT",
        group: "template",
        source: "template",
        required: false,
        zIndex: 2,
        atlas: { texture: 1, x: 0, y: 0, width: 2, height: 2 },
        anchor: { x: 1, y: 1 },
        canvasBounds: { x: -1, y: -1, width: 2, height: 2 },
        mesh: {
          vertices: [-1, 1, 1, 1, 1, -1, -1, -1],
          uvs: [0, 1, 1, 1, 1, 0, 0, 0],
          indices: [0, 1, 2, 0, 2, 3],
        },
        silhouetteMask: "masks/body.png",
        completionPrompt: "discard",
      }],
    },
  };
}

function job(slotPath: string): CharacterGenerationJob {
  return {
    version: 2,
    id: "job-1",
    templateId: "girl-v1",
    input: { description: "少女", referenceImage: null, sourceModelId: null },
    phase: "validating",
    paused: false,
    pauseReason: null,
    revision: 3,
    candidates: [],
    selectedCandidateId: null,
    falRequests: [],
    cache: {},
    artifacts: {
      normalizedImage: null,
      subjectImage: null,
      coarseLayers: [],
      completionSheets: {},
      slotTextures: { body: slotPath },
      repairSlotIds: [],
    },
    budget: { limitUsd: 0.8, estimatedUsd: 0.3, actualUsd: 0, pendingOperation: null },
    issues: [],
    outputModelId: null,
    createdAt: new Date(0).toISOString(),
    updatedAt: new Date(0).toISOString(),
  };
}

async function writePng(file: string, width: number, height: number, rgba: number[]): Promise<void> {
  const png = new PNG({ width, height });
  png.data.set(rgba);
  await writeFile(file, PNG.sync.write(png));
}
