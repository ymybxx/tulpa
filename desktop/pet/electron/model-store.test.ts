import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import JSZip from "jszip";
import { afterEach, describe, expect, it } from "vitest";

import { ModelStore } from "./model-store";
import { idlePresetSettings } from "../src/shared/natural-idle-settings";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { recursive: true, force: true }),
    ),
  );
});

describe("ModelStore", () => {
  it("saves per-model layer visibility and groups without altering source assets", async () => {
    const root = await temporaryDirectory(), source = path.join(root, "layer-model");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const model = (await store.importFolder(source)).model!;
    const entry = store.resolveModelFile(model.id, model.manifest.entry), before = await readFile(entry);
    model.manifest.defaultAppearance = { expressionAssetIds: [], hiddenDrawableIds: ["ArtMesh2", "ArtMesh1", "ArtMesh2"],
      layerGroups: [{ id: "hat", name: "帽子", drawableIds: ["ArtMesh2", "ArtMesh1"] }] };
    await store.saveManifest(model.id, model.manifest);
    const loaded = await new ModelStore(path.join(root, "user-data")).load(model.id);
    expect(loaded.manifest.defaultAppearance).toEqual({ expressionAssetIds: [], hiddenDrawableIds: ["ArtMesh1", "ArtMesh2"],
      layerGroups: [{ id: "hat", name: "帽子", drawableIds: ["ArtMesh1", "ArtMesh2"] }] });
    expect(await readFile(entry)).toEqual(before);
    loaded.manifest.defaultAppearance = { ...loaded.manifest.defaultAppearance!, hiddenDrawableIds: [] };
    const restored = await store.saveManifest(model.id, loaded.manifest);
    expect(restored.manifest.defaultAppearance).not.toHaveProperty("hiddenDrawableIds");
    expect(restored.manifest.defaultAppearance!.layerGroups).toHaveLength(1);
  });
  it("saves native default values without creating expression assets or changing model files", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "native-defaults");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const model = (await store.importFolder(source)).model!;
    const entry = path.join(store.resolveModelRoot(model.id), model.manifest.entry);
    const before = await readFile(entry, "utf8");
    const assets = structuredClone(model.manifest.expressionAssets);
    model.manifest.defaultAppearance = { expressionAssetIds: [], parameterValues: { ParamAngleX: -12.5, ParamMouthOpenY: 0 } };
    await store.saveManifest(model.id, model.manifest);
    const loaded = await store.load(model.id);
    expect(loaded.manifest.defaultAppearance).toEqual(model.manifest.defaultAppearance);
    expect(loaded.manifest.expressionAssets).toEqual(assets);
    expect(await readFile(entry, "utf8")).toBe(before);
    loaded.manifest.defaultAppearance = { expressionAssetIds: [], parameterValues: {} };
    expect((await store.saveManifest(model.id, loaded.manifest)).manifest.defaultAppearance).toEqual({ expressionAssetIds: [] });
  });

  it("persists ordered default appearance selections and removes stale references", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "default-appearance");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const imported = (await store.importFolder(source)).model!;
    expect(imported.manifest).not.toHaveProperty("defaultAppearance");
    const id = Object.keys(imported.manifest.expressionAssets)[0]!;
    expect(id).toBeTruthy();
    imported.manifest.defaultAppearance = { expressionAssetIds: [id, "missing", id] };
    const saved = await store.saveManifest(imported.id, imported.manifest);
    expect(saved.manifest.defaultAppearance).toEqual({ expressionAssetIds: [id] });
    expect((await store.load(imported.id)).manifest.defaultAppearance).toEqual(saved.manifest.defaultAppearance);
    const removed = await store.deleteAsset(imported.id, "expression", id);
    expect(removed.manifest.defaultAppearance).toEqual({ expressionAssetIds: [] });
  });

  it("writes chosen parameter values as a new named toggle", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "create-toggle");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const imported = (await store.importFolder(source)).model!;
    const { model, assetId } = await store.createToggle(imported.id, " 戴/耳麦 ", { ParamMic: 1, ParamAngleX: -3 });
    const asset = model.manifest.expressionAssets[assetId]!;
    expect(asset.name).toBe("戴耳麦");
    expect(asset.file).toBe("expressions/戴耳麦.exp3.json");
    expect(JSON.parse(await readFile(store.resolveModelFile(model.id, asset.file), "utf8"))).toEqual({
      Type: "Live2D Expression",
      Parameters: [
        { Id: "ParamMic", Value: 1, Blend: "Overwrite" },
        { Id: "ParamAngleX", Value: -3, Blend: "Overwrite" },
      ],
    });
    const again = await store.createToggle(imported.id, "戴耳麦", { ParamMic: 0.5 });
    expect(again.model.manifest.expressionAssets[again.assetId]!.file).toBe("expressions/戴耳麦-2.exp3.json");
    await expect(store.createToggle(imported.id, "空", {})).rejects.toThrow("至少一个参数");
    await expect(store.createToggle(imported.id, "  ", { ParamMic: 1 })).rejects.toThrow("名字");
  });

  it("keeps named outfits and drops toggles the model no longer has", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "outfits");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const imported = (await store.importFolder(source)).model!;
    const id = Object.keys(imported.manifest.expressionAssets)[0]!;
    imported.manifest.outfits = [
      { id: "a", name: " 上班 ", appearance: { expressionAssetIds: [id, "missing"], parameterValues: { ParamAngleX: 2 } } },
      { id: "a", name: "重复", appearance: { expressionAssetIds: [] } },
      { id: "b", name: "", appearance: { expressionAssetIds: [] } },
    ];
    const saved = await store.saveManifest(imported.id, imported.manifest);
    expect(saved.manifest.outfits).toEqual([
      { id: "a", name: "上班", appearance: { expressionAssetIds: [id], parameterValues: { ParamAngleX: 2 } } },
    ]);
    const removed = await store.deleteAsset(imported.id, "expression", id);
    expect(removed.manifest.outfits![0]!.appearance.expressionAssetIds).toEqual([]);
  });

  it("persists optional natural idle settings without opting in existing models", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "idle-model");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const imported = (await store.importFolder(source)).model!;
    expect(imported.manifest).not.toHaveProperty("naturalIdle");
    imported.manifest.naturalIdle = { enabled: true, intensity: 1.3, speed: 0.9 };
    await store.saveManifest(imported.id, imported.manifest);
    expect((await store.load(imported.id)).manifest.naturalIdle).toEqual(imported.manifest.naturalIdle);
    imported.manifest.naturalIdle = { enabled: false, intensity: 100, speed: -5 };
    const normalized = await store.saveManifest(imported.id, imported.manifest);
    expect(normalized.manifest.naturalIdle).toEqual({ enabled: false, intensity: 2, speed: 0.5 });
  });

  it("saves expressive profiles and manual rig mappings per model", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "expressive-model");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const imported = (await store.importFolder(source)).model!;
    const settings = idlePresetSettings("active");
    settings.expressive!.bodyIntensity = 1.2;
    settings.expressive!.reverseInterval = 4;
    settings.expressive!.bindings = { bodyX: "CustomBody", eyeY: null };
    imported.manifest.naturalIdle = settings;
    await store.saveManifest(imported.id, imported.manifest);
    expect((await new ModelStore(path.join(root, "user-data")).load(imported.id)).manifest.naturalIdle).toEqual(settings);
  });

  it("imports a valid model folder with empty behavior configuration", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "girl");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));

    const result = await store.importFolder(source);
    expect(result.report.errors).toEqual([]);
    const imported = result.model!;

    expect(imported.name).toBe("girl");
    expect(imported.capabilities.motionGroups).toEqual(["Idle", "Wave"]);
    expect(imported.manifest.schemaVersion).toBe(3);
    expect(imported.manifest.display.x).toBe(0.5);
    expect(imported.manifest.display.y).toBe(0.9);
    expect(Object.values(imported.manifest.behaviors.states).flat()).toEqual([]);
    expect(Object.values(imported.manifest.behaviors.emotions).flat()).toEqual([]);
    expect(Object.keys(imported.manifest.behaviors.emotions)).toEqual([
      "neutral", "angry", "happy", "shy", "sad",
    ]);
    expect(Object.values(imported.manifest.motionAssets)).toEqual(
      expect.arrayContaining([expect.objectContaining({ group: "Idle" }), expect.objectContaining({ group: "Wave" })]),
    );
    expect(imported.url).toMatch(/^pet-model:\/\//);
  });

  it("uses the model-declared LipSync parameter and repairs stale manifests", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "custom-mouth");
    await writeModel(source, ["ParamA"]);
    const store = new ModelStore(path.join(root, "user-data"));
    let imported = (await store.importFolder(source)).model!;

    expect(imported.capabilities.lipSyncParameters).toEqual(["ParamA"]);
    expect(imported.manifest.lipsync.mouthOpenParameter).toBe("ParamA");
    expect(imported.manifest.lipsync).toMatchObject({
      mode: "open-form",
      source: "automatic",
      needsConfirmation: true,
      maxOpen: 1,
    });

    const manifestPath = path.join(
      store.resolveModelRoot(imported.id),
      "pet.manifest.json",
    );
    imported.manifest.lipsync.mouthOpenParameter = "ParamMouthOpenY";
    imported.manifest.lipsync.maxOpen = 0.7;
    await writeFile(manifestPath, JSON.stringify(imported.manifest));

    imported = await store.load(imported.id);
    expect(imported.manifest.lipsync.mouthOpenParameter).toBe("ParamA");
    expect(imported.manifest.lipsync.maxOpen).toBe(1);
    expect(JSON.parse(await readFile(manifestPath, "utf8")).lipsync.mouthOpenParameter)
      .toBe("ParamA");

    imported.manifest.lipsync.source = "manual";
    imported.manifest.lipsync.needsConfirmation = false;
    imported = await store.saveManifest(imported.id, imported.manifest);
    expect(imported.manifest.lipsync).toMatchObject({
      source: "manual",
      needsConfirmation: false,
      mouthOpenParameter: "ParamA",
    });
  });

  it("keeps a VBridger rig's jaw, pucker/widen and funnel, automatic or chosen by hand", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "vbridger-mouth");
    await writeModel(source, ["ParamMouthOpenY"], [
      "ParamMouthOpenY", "ParamMouthForm", "ParamJawOpen", "ParamMouthPuckerWiden", "ParamMouthFunnel",
    ]);
    const store = new ModelStore(path.join(root, "user-data"));

    let imported = (await store.importFolder(source)).model!;
    expect(imported.manifest.lipsync).toMatchObject({
      mode: "open-form",
      jawParameter: "ParamJawOpen",
      puckerWidenParameter: "ParamMouthPuckerWiden",
      funnelParameter: "ParamMouthFunnel",
    });

    imported.manifest.lipsync.source = "manual";
    imported.manifest.lipsync.funnelParameter = null;
    imported.manifest.lipsync.jawParameter = "NotInThisModel";
    imported = await store.saveManifest(imported.id, imported.manifest);
    expect(imported.manifest.lipsync).toMatchObject({
      source: "manual",
      jawParameter: null,
      puckerWidenParameter: "ParamMouthPuckerWiden",
      funnelParameter: null,
    });
  });

  it("uses DisplayInfo to detect a complete AEIOU model", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "viseme-mouth");
    await writeModel(
      source,
      ["ParamA"],
      ["ParamA", "ParamI", "ParamU", "ParamE", "ParamO"],
    );
    const store = new ModelStore(path.join(root, "user-data"));

    const imported = (await store.importFolder(source)).model!;

    expect(imported.capabilities.parameters?.map((item) => item.id)).toEqual([
      "ParamA",
      "ParamI",
      "ParamU",
      "ParamE",
      "ParamO",
    ]);
    expect(imported.manifest.lipsync).toMatchObject({
      mode: "viseme",
      source: "automatic",
      needsConfirmation: false,
      vowelParameters: {
        a: "ParamA",
        e: "ParamE",
        i: "ParamI",
        o: "ParamO",
        u: "ParamU",
      },
    });
  });

  it("changes the model resource revision when a referenced texture changes", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "girl");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const imported = (await store.importFolder(source)).model!;
    const installedRoot = store.resolveModelRoot(imported.id);

    await writeFile(path.join(installedRoot, "texture.png"), "updated-texture");
    const updated = await store.load(imported.id);

    expect(updated.url).not.toBe(imported.url);
    expect(updated.url).toMatch(/\/__revision-[a-f0-9]{16}\//);
    expect(store.resolveModelFile(imported.id, new URL(updated.url).pathname.slice(1)))
      .toBe(path.join(installedRoot, "girl.model3.json"));
  });

  it("resolves referenced files relative to a nested model3 entry", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "nested-model");
    await writeModel(path.join(source, "runtime"), ["ParamMouthOpenY"], [
      "ParamMouthOpenY",
      "ParamMouthForm",
    ]);
    const store = new ModelStore(path.join(root, "user-data"));

    const result = await store.importFolder(source);

    expect(result.report.errors).toEqual([]);
    expect(result.model?.manifest.entry).toBe("runtime/girl.model3.json");
    expect(result.model?.manifest.lipsync).toMatchObject({
      mode: "open-form",
      mouthOpenParameter: "ParamMouthOpenY",
      mouthFormParameter: "ParamMouthForm",
    });
  });

  it("rejects ZIP traversal paths", async () => {
    const root = await temporaryDirectory();
    const archive = new JSZip();
    archive.file("../outside.model3.json", "{}");
    const archivePath = path.join(root, "unsafe.zip");
    await writeFile(archivePath, await archive.generateAsync({ type: "nodebuffer" }));
    const store = new ModelStore(path.join(root, "user-data"));

    const result = await store.importZip(archivePath);
    expect(result.model).toBeNull();
    expect(result.report.errors[0]?.message).toMatch(/不安全路径/);
  });

  it("rejects packages with missing referenced resources", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "broken");
    await mkdir(source, { recursive: true });
    await writeFile(
      path.join(source, "broken.model3.json"),
      JSON.stringify({ FileReferences: { Moc: "broken.moc3", Textures: ["missing.png"] } }),
    );
    await writeFile(path.join(source, "broken.moc3"), "moc");
    const store = new ModelStore(path.join(root, "user-data"));

    const result = await store.importFolder(source);
    expect(result.model).toBeNull();
    expect(result.report.status).toBe("invalid");
  });

  it("rebuilds a supplied v1 manifest without migrating its mappings", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "custom");
    await writeModel(source);
    await writeFile(
      path.join(source, "pet.manifest.json"),
      JSON.stringify({
        schemaVersion: 1,
        id: "portable-id",
        name: "Custom Girl",
        renderer: "live2d-cubism",
        entry: "girl.model3.json",
        display: { scale: 1.25, x: 0.4, y: 0.9 },
        stateMotions: { standby: { group: "Idle" } },
        emotions: { happy: "happy" },
        gestures: { wave: { group: "Wave" } },
        hitAreas: { head: ["Head"], body: ["Body"] },
        lipsync: {
          mouthOpenParameter: "ParamMouthOpenY",
          mouthFormParameter: "ParamMouthForm",
          maxOpen: 0.85,
        },
      }),
    );
    const store = new ModelStore(path.join(root, "user-data"));

    const imported = (await store.importFolder(source)).model!;

    expect(imported.name).toBe("Custom Girl");
    expect(imported.manifest.id).not.toBe("portable-id");
    expect(imported.manifest.schemaVersion).toBe(3);
    expect(imported.manifest.display.scale).toBe(0.95);
    expect(Object.values(imported.manifest.behaviors.states).flat()).toEqual([]);
    expect(Object.values(imported.manifest.behaviors.emotions).flat()).toEqual([]);
  });

  it("backs up v2, preserves assets and display settings, and clears semantic mappings", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "legacy");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const imported = (await store.importFolder(source)).model!;
    const modelRoot = store.resolveModelRoot(imported.id);
    const motionId = Object.keys(imported.manifest.motionAssets)[0]!;
    const expressionId = Object.keys(imported.manifest.expressionAssets)[0]!;
    imported.manifest.motionAssets[motionId]!.name = "旧动作名称";
    imported.manifest.expressionAssets[expressionId]!.name = "旧表情名称";
    const legacy = {
      ...imported.manifest,
      schemaVersion: 2,
      display: { scale: 1.3, x: 0.2, y: 0.8 },
      statePools: { unbound: [motionId], standby: [motionId], acknowledging: [], listening: [], thinking: [], speaking: [] },
      emotionPools: { neutral: [], happy: [expressionId], affectionate: [], shy: [], amused: [], sad: [], concerned: [], angry: [], surprised: [], sleepy: [] },
      emotionMotionPools: { neutral: [], happy: [motionId], affectionate: [], shy: [], amused: [], sad: [], concerned: [], angry: [], surprised: [], sleepy: [] },
      gesturePools: { none: [], nod: [], shake_head: [], wave: [motionId], lean_in: [], shrug: [], cheer: [] },
      hitAreas: { head: ["Head"], body: ["Body"] },
    } as Record<string, unknown>;
    delete legacy.behaviors;
    await writeFile(path.join(modelRoot, "pet.manifest.json"), JSON.stringify(legacy));

    const migrated = await store.load(imported.id);
    const backup = JSON.parse(await readFile(path.join(modelRoot, "pet.manifest.v2.backup.json"), "utf8"));

    expect(backup.schemaVersion).toBe(2);
    expect(migrated.manifest.schemaVersion).toBe(3);
    expect(migrated.manifest.display.scale).toBe(1.3);
    expect(migrated.manifest.motionAssets[motionId]?.name).toBe("旧动作名称");
    expect(migrated.manifest.expressionAssets[expressionId]?.name).toBe("旧表情名称");
    expect(Object.values(migrated.manifest.behaviors.states).flat()).toEqual([]);
    expect(Object.values(migrated.manifest.behaviors.emotions).flat()).toEqual([]);
  });

  it("registers valid unreferenced assets from a complete model", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "girl");
    await writeModel(source);
    await writeFile(
      path.join(source, "bonus.motion3.json"),
      JSON.stringify({ Meta: { Duration: 1 }, Curves: [] }),
    );
    await writeFile(
      path.join(source, "sparkle.exp3.json"),
      JSON.stringify({ Type: "Live2D Expression", Parameters: [] }),
    );
    const store = new ModelStore(path.join(root, "user-data"));

    const imported = (await store.importFolder(source)).model!;

    expect(Object.values(imported.manifest.motionAssets)).toEqual(
      expect.arrayContaining([expect.objectContaining({ file: "bonus.motion3.json" })]),
    );
    expect(Object.values(imported.manifest.expressionAssets)).toEqual(
      expect.arrayContaining([expect.objectContaining({ file: "sparkle.exp3.json" })]),
    );
  });

  it("deletes the complete candidate when one referenced asset is removed", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "girl");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    let imported = (await store.importFolder(source)).model!;
    const motionId = Object.values(imported.manifest.motionAssets)
      .find((asset) => asset.group === "Wave")!.id;
    const expressionId = Object.values(imported.manifest.expressionAssets)[0]!.id;
    imported.manifest.behaviors.emotions.happy = [{
      id: "combined-happy",
      motionAssetIds: [motionId],
      expressionAssetIds: [expressionId],
    }];
    imported = await store.saveManifest(imported.id, imported.manifest);

    const updated = await store.deleteAsset(imported.id, "motion", motionId);

    expect(updated.manifest.motionAssets[motionId]).toBeUndefined();
    expect(updated.manifest.behaviors.emotions.happy).toEqual([]);
    expect(updated.manifest.expressionAssets[expressionId]).toBeDefined();
  });

  it("repairs a stale v3 manifest by dropping rather than truncating a candidate", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "girl");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const imported = (await store.importFolder(source)).model!;
    const motionId = Object.keys(imported.manifest.motionAssets)[0]!;
    imported.manifest.behaviors.states.standby = [{
      id: "stale-combination",
      motionAssetIds: [motionId, "missing-motion"],
      expressionAssetIds: [],
    }];
    await writeFile(
      path.join(store.resolveModelRoot(imported.id), "pet.manifest.json"),
      JSON.stringify(imported.manifest),
    );

    const repaired = await store.load(imported.id);

    expect(repaired.manifest.behaviors.states.standby).toEqual([]);
  });

  it("does not retain the selected archive in the installed model", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "girl");
    await writeModel(source);
    const archive = new JSZip();
    for (const file of await readdir(source)) {
      archive.file(file, await readFile(path.join(source, file)));
    }
    const archivePath = path.join(root, "girl.zip");
    await writeFile(archivePath, await archive.generateAsync({ type: "nodebuffer" }));
    const store = new ModelStore(path.join(root, "user-data"));

    const imported = (await store.importZip(archivePath)).model!;
    const installedFiles = await readdir(path.join(store.modelsRoot, imported.id));

    expect(installedFiles.some((file) => file.endsWith(".zip"))).toBe(false);
  });

  it("saves a validated preview and refuses to delete the final model", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "girl");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const model = (await store.importFolder(source)).model!;
    const png = Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      Buffer.from("preview"),
    ]);

    const updated = await store.savePreview(model.id, png);

    expect(updated.previewUrl).toMatch(/\.preview\.png$/);
    await expect(store.savePreview(model.id, Buffer.from("not png"))).rejects.toThrow(/PNG/);
    await expect(store.delete(model.id)).rejects.toThrow(/最后一个模型/);
  });

  it("stamps a look by how it looks at rest, and records which stamp its picture was made under", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "stamped");
    await writeModel(source);
    const store = new ModelStore(path.join(root, "user-data"));
    const model = (await store.importFolder(source)).model!;
    const png = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.from("preview")]);
    const before = await store.lookStamp(model.id);

    model.manifest.name = "改了名字";
    await store.saveManifest(model.id, model.manifest);
    expect(await store.lookStamp(model.id)).toBe(before);

    model.manifest.defaultAppearance = { expressionAssetIds: [], parameterValues: { ParamAngleX: 10 } };
    await store.saveManifest(model.id, model.manifest);
    expect(await store.lookStamp(model.id)).not.toBe(before);

    await store.savePreview(model.id, png, "look-1");
    expect(await store.previewKey(model.id)).toBe("look-1");
    await store.savePreview(model.id, png);
    expect(await store.previewKey(model.id)).toBeNull();
  });

  it("deletes the selected model and selects a remaining model", async () => {
    const root = await temporaryDirectory();
    const firstSource = path.join(root, "first");
    const secondSource = path.join(root, "second");
    await writeModel(firstSource);
    await writeModel(secondSource);
    const store = new ModelStore(path.join(root, "user-data"));
    const first = (await store.importFolder(firstSource)).model!;
    const second = (await store.importFolder(secondSource)).model!;

    const deleted = await store.delete(second.id);

    expect(deleted).toMatchObject({ deletedId: second.id });
    expect(deleted.selectedModel.id).toBe(first.id);
    expect((await store.selected())?.id).toBe(first.id);
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(os.tmpdir(), "fish-memory-pet-test-"));
  temporaryDirectories.push(directory);
  return directory;
}

async function writeModel(
  root: string,
  lipSyncParameters: string[] = [],
  displayParameters: string[] = [],
): Promise<void> {
  await mkdir(root, { recursive: true });
  await writeFile(
    path.join(root, "girl.model3.json"),
    JSON.stringify({
      FileReferences: {
        Moc: "girl.moc3",
        Textures: ["texture.png"],
        ...(displayParameters.length ? { DisplayInfo: "girl.cdi3.json" } : {}),
        Motions: {
          Idle: [{ File: "idle.motion3.json" }],
          Wave: [{ File: "wave.motion3.json" }],
        },
        Expressions: [{ Name: "happy", File: "happy.exp3.json" }],
      },
      HitAreas: [{ Name: "Head" }, { Name: "Body" }],
      Groups: lipSyncParameters.length
        ? [{ Target: "Parameter", Name: "LipSync", Ids: lipSyncParameters }]
        : [],
    }),
  );
  if (displayParameters.length) {
    await writeFile(
      path.join(root, "girl.cdi3.json"),
      JSON.stringify({
        Parameters: displayParameters.map((id) => ({
          Id: id,
          GroupId: "ParamGroupMouth",
          Name: id,
        })),
      }),
    );
  }
  for (const file of [
    "girl.moc3",
    "texture.png",
    "idle.motion3.json",
    "wave.motion3.json",
    "happy.exp3.json",
  ]) {
    await writeFile(path.join(root, file), file);
  }
}
