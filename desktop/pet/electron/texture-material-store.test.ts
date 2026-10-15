import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { PNG } from "pngjs";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  ModelEditCapabilityManifest,
  ModelPackage,
  TextureEditUnit,
} from "../src/shared/contracts";
import type { ModelStore } from "./model-store";
import { AtlasCompactionCoordinator } from "./atlas-compaction-coordinator";
import { TextureMaterialStore } from "./texture-material-store";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("TextureMaterialStore", () => {
  it("treats initial and generated images as peer materials with one selection path", async () => {
    const harness = await createHarness([{ id: "leg", atlasIndex: 0, atlasFile: "textures/atlas.png" }]);
    const initialized = await harness.store.list("mao", harness.capability);
    expect(initialized.schemaVersion).toBe(4);
    expect(initialized.materials).toHaveLength(1);
    const initial = initialized.materials[0]!;
    expect(initialized.activeAssignments.leg).toBe(initial.id);
    expect(initial.imageUrl).toContain("/.edit-capabilities/texture-units/");

    const patch = path.join(harness.root, "pink.png");
    await writePng(patch, solidPng(2, 2, [220, 120, 180, 255]));
    const saved = await harness.store.saveAndApply({
      modelId: "mao",
      name: "粉色素材",
      prompt: "改成粉色",
      sourceHash: "topology-v1",
      edits: [{
        textureUnitId: "leg",
        label: "腿",
        atlasIndex: 0,
        atlasFile: "textures/atlas.png",
        bounds: { x: 0, y: 0, width: 2, height: 2 },
        maskPath: harness.maskPaths.leg!,
        editedPatchPath: patch,
      }],
    });
    const generated = saved.library.materials.find((item) => item.id !== initial.id)!;
    expect(generated.imageUrl).toContain("/.appearance/materials/");
    expect(saved.library.activeAssignments.leg).toBe(generated.id);
    const persisted = JSON.parse(await readFile(
      path.join(harness.modelRoot, ".appearance/library.json"),
      "utf8",
    )) as { materials: Array<{ id: string; maskFile: string }> };
    expect(persisted.materials.find((item) => item.id === generated.id)?.maskFile)
      .toMatch(/^\.appearance\/journal-assets\/masks\/[a-f0-9]{64}\.png$/);
    await harness.store.list("mao", harness.capability);
    const rescanned = JSON.parse(await readFile(
      path.join(harness.modelRoot, ".appearance/library.json"),
      "utf8",
    )) as { materials: Array<{ id: string; maskFile: string }> };
    expect(rescanned.materials.find((item) => item.id === generated.id)?.maskFile)
      .toBe(persisted.materials.find((item) => item.id === generated.id)?.maskFile);
    const generatedSnapshot = await harness.store.appearanceSnapshot("mao", ["textures/atlas.png"]);
    expect(generatedSnapshot.delta.patchSets[0]?.patches).toEqual([
      expect.objectContaining({
        textureUnitId: "leg",
        maskUrl: expect.stringContaining("/.appearance/journal-assets/masks/"),
      }),
    ]);
    await expect(harness.store.resolveMaterialImage("mao", "leg", initial.id))
      .resolves.toContain("leg-original.png");
    await expect(harness.store.resolveMaterialImage("mao", "other-unit", initial.id))
      .rejects.toThrow("所选参考素材已经不存在");
    await expect(harness.store.delete("mao", generated.id)).rejects.toThrow("当前使用中的素材不能删除");

    const restored = await harness.store.select("mao", "leg", initial.id);
    expect(restored.library.activeAssignments.leg).toBe(initial.id);
    expect((await harness.store.appearanceSnapshot("mao", ["textures/atlas.png"]))
      .delta.patchSets[0]?.patches).toEqual([]);
    const deleted = await harness.store.delete("mao", generated.id);
    expect(deleted.library.materials.map((item) => item.id)).toEqual([initial.id]);
    expect(deleted.delta.atlases).toEqual([]);
  });

  it("can add a version to 换色 without wearing it", async () => {
    const harness = await createHarness([{ id: "leg", atlasIndex: 0, atlasFile: "textures/atlas.png" }]);
    const initialized = await harness.store.list("mao", harness.capability);
    const initial = initialized.materials[0]!;
    const patch = path.join(harness.root, "pink.png");
    await writePng(patch, solidPng(2, 2, [220, 120, 180, 255]));
    const saved = await harness.store.saveAndApply({
      modelId: "mao",
      name: "粉色素材",
      prompt: "改成粉色",
      sourceHash: "topology-v1",
      apply: false,
      edits: [{
        textureUnitId: "leg",
        label: "腿",
        atlasIndex: 0,
        atlasFile: "textures/atlas.png",
        bounds: { x: 0, y: 0, width: 2, height: 2 },
        maskPath: harness.maskPaths.leg!,
        editedPatchPath: patch,
      }],
    });
    expect(saved.library.materials).toHaveLength(2);
    expect(saved.library.activeAssignments.leg).toBe(initial.id);
    expect((await harness.store.appearanceSnapshot("mao", ["textures/atlas.png"]))
      .delta.patchSets[0]?.patches).toEqual([]);
  });

  it("snapshots only the atlas affected by a switch", async () => {
    const harness = await createHarness([
      { id: "unit-0", atlasIndex: 0, atlasFile: "textures/atlas-0.png" },
      { id: "unit-1", atlasIndex: 1, atlasFile: "textures/atlas-1.png" },
    ]);
    const initial = await harness.store.list("mao", harness.capability);
    const patch = path.join(harness.root, "changed.png");
    await writePng(patch, solidPng(2, 2, [100, 110, 120, 255]));
    const saved = await harness.store.saveAndApply({
      modelId: "mao",
      name: "图集 0 新素材",
      prompt: "修改图集 0",
      sourceHash: "topology-v1",
      edits: [{
        textureUnitId: "unit-0",
        label: "素材 0",
        atlasIndex: 0,
        atlasFile: "textures/atlas-0.png",
        bounds: { x: 0, y: 0, width: 2, height: 2 },
        maskPath: harness.maskPaths["unit-0"]!,
        editedPatchPath: patch,
      }],
    });
    const runtime1 = await harness.store.resolveCompiledAtlas(
      "mao",
      path.join(harness.modelRoot, "textures/atlas-1.png"),
    );
    expect(runtime1).toBeNull();
    const initial0 = initial.materials.find((item) => item.textureUnitId === "unit-0")!;
    const switched = await harness.store.select("mao", "unit-0", initial0.id);
    expect(switched.affectedAtlasFiles).toEqual(["textures/atlas-0.png"]);
    expect(switched.delta).toMatchObject({
      streamId: "model:mao",
      patchSets: [{
        atlasIndex: 0,
        atlasFile: "textures/atlas-0.png",
        patches: [],
      }],
      atlases: [],
    });
    expect(await harness.store.resolveCompiledAtlas(
      "mao",
      path.join(harness.modelRoot, "textures/atlas-1.png"),
    )).toBeNull();
    expect(saved.library.activeAssignments["unit-1"]).toBe(initial.activeAssignments["unit-1"]);
  });

  it("puts a whole outfit on in one change: one revision, the touched atlases rebuilt once", async () => {
    const harness = await createHarness([
      { id: "unit-0", atlasIndex: 0, atlasFile: "textures/atlas-0.png" },
      { id: "unit-1", atlasIndex: 1, atlasFile: "textures/atlas-1.png" },
    ]);
    const initial = await harness.store.list("mao", harness.capability);
    const patch = path.join(harness.root, "outfit.png");
    await writePng(patch, solidPng(2, 2, [200, 40, 60, 255]));
    const saved = await harness.store.saveAndApply({
      modelId: "mao", name: "AI 素材", prompt: "和风巫女", sourceHash: "topology-v1",
      edits: (["unit-0", "unit-1"] as const).map((unit, index) => ({
        textureUnitId: unit, label: unit, atlasIndex: index, atlasFile: `textures/atlas-${index}.png`,
        bounds: { x: 0, y: 0, width: 2, height: 2 }, maskPath: harness.maskPaths[unit]!, editedPatchPath: patch,
      })),
    });
    const back = initial.materials.map((item) => ({ textureUnitId: item.textureUnitId, materialId: item.id }));
    const restored = await harness.store.selectMany("mao", back);
    expect(restored.library.revision).toBe(saved.library.revision + 1);
    expect(restored.library.activeAssignments).toEqual(initial.activeAssignments);
    expect([...restored.affectedAtlasFiles].sort()).toEqual(["textures/atlas-0.png", "textures/atlas-1.png"]);
    // Asking for what is already worn changes nothing.
    const again = await harness.store.selectMany("mao", back);
    expect(again.library.revision).toBe(restored.library.revision);
    expect(again.affectedAtlasFiles).toEqual([]);
    await expect(harness.store.selectMany("mao", [{ textureUnitId: "unit-1", materialId: back[0]!.materialId }])).rejects.toThrow("素材与当前纹理不匹配");
  });

  it("uses empty patch sets to make a base-only render bundle authoritative", async () => {
    const harness = await createHarness([
      { id: "unit-0", atlasIndex: 0, atlasFile: "textures/atlas-0.png" },
      { id: "unit-1", atlasIndex: 1, atlasFile: "textures/atlas-1.png" },
    ]);
    const initial = await harness.store.list("mao", harness.capability);
    const patch = path.join(harness.root, "custom.png");
    await writePng(patch, solidPng(2, 2, [100, 110, 120, 255]));
    const saved = await harness.store.saveAndApply({
      modelId: "mao",
      name: "自定义素材",
      prompt: "修改图集 0",
      sourceHash: "topology-v1",
      edits: [{
        textureUnitId: "unit-0",
        label: "素材 0",
        atlasIndex: 0,
        atlasFile: "textures/atlas-0.png",
        bounds: { x: 0, y: 0, width: 2, height: 2 },
        maskPath: harness.maskPaths["unit-0"]!,
        editedPatchPath: patch,
      }],
    });
    expect((await harness.store.appearanceSnapshot("mao")).delta.patchSets[0]?.patches)
      .toHaveLength(1);

    await harness.store.select(
      "mao",
      "unit-0",
      initial.activeAssignments["unit-0"]!,
    );
    const restored = await harness.store.appearanceSnapshot("mao");
    expect(restored.delta.revision).toBeGreaterThan(saved.library.revision);
    expect(restored.delta.patchSets).toEqual([
      expect.objectContaining({ atlasIndex: 0, patches: [] }),
      expect.objectContaining({ atlasIndex: 1, patches: [] }),
    ]);
  });

  it("returns a selected patch snapshot before background compaction finishes", async () => {
    let blockCompaction = false;
    let releaseCompaction!: () => void;
    const blocked = new Promise<void>((resolve) => { releaseCompaction = resolve; });
    const compact = vi.fn(async () => {
      if (blockCompaction) await blocked;
    });
    const coordinator = new AtlasCompactionCoordinator({ compact });
    const harness = await createHarness(
      [{ id: "leg", atlasIndex: 0, atlasFile: "textures/atlas.png" }],
      true,
      coordinator,
    );
    const initial = await harness.store.list("mao", harness.capability);
    const patch = path.join(harness.root, "generated.png");
    await writePng(patch, solidPng(2, 2, [220, 120, 180, 255]));
    const saved = await harness.store.saveAndApply({
      modelId: "mao",
      name: "生成素材",
      prompt: "改成粉色",
      sourceHash: "topology-v1",
      edits: [{
        textureUnitId: "leg",
        label: "腿",
        atlasIndex: 0,
        atlasFile: "textures/atlas.png",
        bounds: { x: 0, y: 0, width: 2, height: 2 },
        maskPath: harness.maskPaths.leg!,
        editedPatchPath: patch,
      }],
    });
    expect(await coordinator.flush(1_000)).toBe(true);
    const generatedId = saved.library.activeAssignments.leg!;
    await harness.store.select("mao", "leg", initial.activeAssignments.leg!);

    blockCompaction = true;
    const selected = await harness.store.select("mao", "leg", generatedId);

    expect(selected.library.activeAssignments.leg).toBe(generatedId);
    expect(selected.delta.patchSets[0]?.patches).toEqual([
      expect.objectContaining({ textureUnitId: "leg" }),
    ]);
    expect(compact).toHaveBeenCalledTimes(2);
    expect(await coordinator.flush(5)).toBe(false);
    releaseCompaction();
    expect(await coordinator.flush(1_000)).toBe(true);
  });

  it("migrates schema v3 variants into the same material directory as seeded images", async () => {
    const harness = await createHarness([{ id: "sleeve", atlasIndex: 0, atlasFile: "textures/atlas.png" }], false);
    const legacyRoot = path.join(harness.modelRoot, ".appearance", "variants", "variant-pink");
    await mkdir(legacyRoot, { recursive: true });
    await writePng(path.join(legacyRoot, "patch.png"), solidPng(2, 2, [240, 120, 180, 255]));
    await writePng(path.join(legacyRoot, "mask.png"), solidPng(2, 2, [255, 255, 255, 255]));
    await writeFile(path.join(harness.modelRoot, ".appearance", "library.json"), `${JSON.stringify({
      schemaVersion: 3,
      modelId: "mao",
      revision: 7,
      variants: [{
        id: "variant-pink",
        textureUnitId: "sleeve",
        label: "袖子",
        atlasIndex: 0,
        atlasFile: "textures/atlas.png",
        bounds: { x: 0, y: 0, width: 2, height: 2 },
        name: "粉色袖子",
        prompt: "改成粉色",
        createdAt: "2026-08-19T00:00:00.000Z",
        patchFile: ".appearance/variants/variant-pink/patch.png",
        maskFile: ".appearance/variants/variant-pink/mask.png",
      }],
      activeAssignments: { sleeve: "variant-pink" },
      updatedAt: "2026-08-19T00:00:00.000Z",
    }, null, 2)}\n`);
    const library = await harness.store.list("mao", harness.capability);
    expect(library.schemaVersion).toBe(4);
    expect(library.materials).toHaveLength(2);
    expect(library.activeAssignments.sleeve).toBe("material-pink");
    expect(library.materials.some((item) => item.imageUrl.includes("/.appearance/materials/"))).toBe(true);
    expect(library.materials.some((item) => item.imageUrl.includes("/.edit-capabilities/texture-units/"))).toBe(true);
  });

  it("migrates an active schema v3 appearance before the first render bundle", async () => {
    const harness = await createHarness([
      { id: "sleeve", atlasIndex: 0, atlasFile: "textures/atlas.png" },
    ], false);
    const legacyRoot = path.join(harness.modelRoot, ".appearance", "variants", "variant-pink");
    await mkdir(legacyRoot, { recursive: true });
    await writePng(path.join(legacyRoot, "patch.png"), solidPng(2, 2, [240, 120, 180, 255]));
    await writePng(path.join(legacyRoot, "mask.png"), solidPng(2, 2, [255, 255, 255, 255]));
    await writeFile(path.join(harness.modelRoot, ".appearance", "library.json"), `${JSON.stringify({
      schemaVersion: 3,
      modelId: "mao",
      revision: 7,
      variants: [{
        id: "variant-pink",
        textureUnitId: "sleeve",
        label: "袖子",
        atlasIndex: 0,
        atlasFile: "textures/atlas.png",
        bounds: { x: 0, y: 0, width: 2, height: 2 },
        name: "粉色袖子",
        prompt: "改成粉色",
        createdAt: "2026-08-19T00:00:00.000Z",
        patchFile: ".appearance/variants/variant-pink/patch.png",
        maskFile: ".appearance/variants/variant-pink/mask.png",
      }],
      activeAssignments: { sleeve: "variant-pink" },
      updatedAt: "2026-08-19T00:00:00.000Z",
    }, null, 2)}\n`);

    const snapshot = await harness.store.appearanceSnapshot("mao");

    expect(snapshot.delta).toMatchObject({
      modelId: "mao",
      revision: 8,
      patchSets: [{
        atlasFile: "textures/atlas.png",
        patches: [{ textureUnitId: "sleeve" }],
      }],
    });
    const persisted = JSON.parse(await readFile(
      path.join(harness.modelRoot, ".appearance", "library.json"),
      "utf8",
    )) as { schemaVersion: number; revision: number; materials: Array<{ maskFile: string }> };
    expect(persisted.schemaVersion).toBe(4);
    expect(persisted.revision).toBe(8);
    expect(persisted.materials[0]?.maskFile)
      .toMatch(/^\.appearance\/journal-assets\/masks\/[a-f0-9]{64}\.png$/);
  });

  it("atomically persists an immutable mask before returning a schema v4 snapshot", async () => {
    const harness = await createHarness([
      { id: "leg", atlasIndex: 0, atlasFile: "textures/atlas.png" },
    ]);
    const patch = path.join(harness.root, "legacy-material.png");
    await writePng(patch, solidPng(2, 2, [90, 100, 110, 255]));
    const saved = await harness.store.saveAndApply({
      modelId: "mao",
      name: "旧素材",
      prompt: "旧素材",
      sourceHash: "topology-v1",
      edits: [{
        textureUnitId: "leg",
        label: "腿",
        atlasIndex: 0,
        atlasFile: "textures/atlas.png",
        bounds: { x: 0, y: 0, width: 2, height: 2 },
        maskPath: harness.maskPaths.leg!,
        editedPatchPath: patch,
      }],
    });
    const libraryPath = path.join(harness.modelRoot, ".appearance", "library.json");
    const legacy = JSON.parse(await readFile(libraryPath, "utf8")) as {
      revision: number;
      materials: Array<{ id: string; maskFile: string }>;
    };
    const activeId = saved.library.activeAssignments.leg!;
    legacy.materials.find((item) => item.id === activeId)!.maskFile =
      ".edit-capabilities/masks/leg.png";
    await writeFile(libraryPath, `${JSON.stringify(legacy, null, 2)}\n`);

    const snapshot = await harness.store.appearanceSnapshot("mao");
    const persisted = JSON.parse(await readFile(libraryPath, "utf8")) as {
      revision: number;
      materials: Array<{ id: string; maskFile: string }>;
    };

    expect(snapshot.delta.revision).toBe(legacy.revision + 1);
    expect(persisted.revision).toBe(legacy.revision + 1);
    expect(persisted.materials.find((item) => item.id === activeId)?.maskFile)
      .toMatch(/^\.appearance\/journal-assets\/masks\/[a-f0-9]{64}\.png$/);
    await rm(harness.maskPaths.leg!, { force: true });
    await expect(harness.store.appearanceSnapshot("mao")).resolves.toMatchObject({
      delta: { revision: persisted.revision },
    });
  });

  it("self-heals a copied schema v4 library model id before rendering", async () => {
    const harness = await createHarness([
      { id: "leg", atlasIndex: 0, atlasFile: "textures/atlas.png" },
    ]);
    const libraryPath = path.join(harness.modelRoot, ".appearance", "library.json");
    const copied = JSON.parse(await readFile(libraryPath, "utf8")) as {
      modelId: string;
      revision: number;
    };
    copied.modelId = "old-import-id";
    await writeFile(libraryPath, `${JSON.stringify(copied, null, 2)}\n`);

    const snapshot = await harness.store.appearanceSnapshot("mao");
    const persisted = JSON.parse(await readFile(libraryPath, "utf8")) as {
      modelId: string;
      revision: number;
    };

    expect(snapshot.delta.modelId).toBe("mao");
    expect(snapshot.delta.revision).toBe(copied.revision + 1);
    expect(persisted).toMatchObject({ modelId: "mao", revision: copied.revision + 1 });
  });

  it("omits historical inactive atlases from an authoritative snapshot", async () => {
    const harness = await createHarness([
      { id: "leg", atlasIndex: 0, atlasFile: "textures/atlas.png" },
    ]);
    const libraryPath = path.join(harness.modelRoot, ".appearance", "library.json");
    const library = JSON.parse(await readFile(libraryPath, "utf8")) as {
      materials: Array<Record<string, unknown>>;
    };
    library.materials.push({
      id: "material-stale",
      textureUnitId: "removed-unit",
      label: "已移除",
      atlasIndex: 9,
      atlasFile: "textures/stale-atlas.png",
      bounds: { x: 0, y: 0, width: 2, height: 2 },
      name: "历史素材",
      prompt: "",
      createdAt: "2026-08-19T00:00:00.000Z",
      imageFile: ".appearance/materials/material-stale/image.png",
      maskFile: ".appearance/journal-assets/masks/missing.png",
      contentHash: "stale",
    });
    await writeFile(libraryPath, `${JSON.stringify(library, null, 2)}\n`);

    const snapshot = await harness.store.appearanceSnapshot("mao");

    expect(snapshot.delta.patchSets.map((item) => item.atlasFile))
      .toEqual(["textures/atlas.png"]);
  });

  it("serializes rapid selections and keeps the last submitted material active", async () => {
    const harness = await createHarness([{ id: "hat", atlasIndex: 0, atlasFile: "textures/atlas.png" }]);
    const initial = await harness.store.list("mao", harness.capability);
    const make = async (name: string, rgba: [number, number, number, number]) => {
      const patch = path.join(harness.root, `${name}.png`);
      await writePng(patch, solidPng(2, 2, rgba));
      return harness.store.saveAndApply({
        modelId: "mao",
        name,
        prompt: name,
        sourceHash: "topology-v1",
        edits: [{
          textureUnitId: "hat",
          label: "帽子",
          atlasIndex: 0,
          atlasFile: "textures/atlas.png",
          bounds: { x: 0, y: 0, width: 2, height: 2 },
          maskPath: harness.maskPaths.hat!,
          editedPatchPath: patch,
        }],
      });
    };
    const pink = await make("pink", [240, 120, 180, 255]);
    const blue = await make("blue", [80, 120, 240, 255]);
    const pinkId = pink.library.materials.find((item) => item.name === "pink")!.id;
    const blueId = blue.library.materials.find((item) => item.name === "blue")!.id;
    await Promise.all([
      harness.store.select("mao", "hat", pinkId),
      harness.store.select("mao", "hat", blueId),
    ]);
    expect((await harness.store.list("mao", harness.capability)).activeAssignments.hat).toBe(blueId);
    expect(initial.materials).toHaveLength(1);
  });

  it("removes historical prompts from legacy AI material display names", async () => {
    const harness = await createHarness([{ id: "leg", atlasIndex: 0, atlasFile: "textures/atlas.png" }]);
    const initialized = await harness.store.list("mao", harness.capability);
    const patch = path.join(harness.root, "legacy.png");
    await writePng(patch, solidPng(2, 2, [220, 180, 180, 255]));
    await harness.store.saveAndApply({
      modelId: "mao",
      name: "AI 素材 · 用户要求：去掉黑袜",
      prompt: "用户要求：去掉黑袜",
      sourceHash: "topology-v1",
      edits: [{
        textureUnitId: "leg",
        label: "腿",
        atlasIndex: 0,
        atlasFile: "textures/atlas.png",
        bounds: { x: 0, y: 0, width: 2, height: 2 },
        maskPath: harness.maskPaths.leg!,
        editedPatchPath: patch,
      }],
    });

    const reloaded = await harness.store.list("mao", harness.capability);
    expect(reloaded.materials.find((item) => item.id !== initialized.materials[0]!.id)?.name)
      .toBe("AI 素材");
  });
});

async function createHarness(
  specs: Array<{ id: string; atlasIndex: number; atlasFile: string }>,
  initialize = true,
  compaction?: AtlasCompactionCoordinator,
) {
  const root = await mkdtemp(path.join(os.tmpdir(), "texture-materials-v4-"));
  roots.push(root);
  const modelRoot = path.join(root, "models", "mao");
  const maskPaths: Record<string, string> = {};
  const units: TextureEditUnit[] = [];
  for (const spec of specs) {
    const atlasPath = path.join(modelRoot, spec.atlasFile);
    await mkdir(path.dirname(atlasPath), { recursive: true });
    await writePng(atlasPath, solidPng(2, 2, [10 + spec.atlasIndex, 20, 30, 255]));
    const maskFile = `.edit-capabilities/masks/${spec.id}.png`;
    const previewFile = `.edit-capabilities/texture-units/${spec.id}-original.png`;
    maskPaths[spec.id] = path.join(modelRoot, maskFile);
    await mkdir(path.dirname(maskPaths[spec.id]!), { recursive: true });
    await mkdir(path.dirname(path.join(modelRoot, previewFile)), { recursive: true });
    await writePng(maskPaths[spec.id]!, solidPng(2, 2, [255, 255, 255, 255]));
    await writePng(path.join(modelRoot, previewFile), solidPng(2, 2, [10 + spec.atlasIndex, 20, 30, 255]));
    units.push({
      id: spec.id,
      label: spec.id,
      atlasIndex: spec.atlasIndex,
      atlasFile: spec.atlasFile,
      bounds: { x: 0, y: 0, width: 2, height: 2 },
      maskPath: maskFile,
      originalPreviewPath: previewFile,
    } as TextureEditUnit);
  }
  const model = { id: "mao" } as ModelPackage;
  const models = {
    load: async () => model,
    resolveModelRoot: () => modelRoot,
    resolveModelFile: (_id: string, relative: string) => path.join(modelRoot, relative),
  } as unknown as ModelStore;
  const capability = {
    modelId: "mao",
    sourceHash: "topology-v1",
    generatedAt: "2026-08-20T00:00:00.000Z",
    textureUnits: units,
  } as ModelEditCapabilityManifest;
  const store = new TextureMaterialStore(
    models,
    compaction ?? new AtlasCompactionCoordinator({ compact: async () => undefined }),
  );
  if (initialize) await store.list("mao", capability);
  return { root, modelRoot, maskPaths, capability, store };
}

function solidPng(width: number, height: number, rgba: [number, number, number, number]): PNG {
  const png = new PNG({ width, height });
  for (let offset = 0; offset < png.data.length; offset += 4) png.data.set(rgba, offset);
  return png;
}

async function writePng(file: string, png: PNG): Promise<void> {
  await writeFile(file, PNG.sync.write(png));
}
