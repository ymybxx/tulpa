import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it, onTestFinished, vi } from "vitest";
import { PNG } from "pngjs";

import type { ModelEditCapabilityManifest } from "../src/shared/contracts";
import * as coreInspector from "./cubism-core-inspector";
import {
  ModelEditCapabilityStore,
  textureMaskIdentity,
  visiblePartIdsFromMotion,
} from "./model-edit-capabilities";

describe("Live2D texture unit deduplication", () => {
  it("deduplicates twenty eye ArtMesh references arranged as seven exact UV masks", () => {
    const patterns = Array.from({ length: 7 }, (_, index) => ({
      atlasIndex: index % 2,
      bounds: { x: 20 + index * 11, y: 40 + index * 7, width: 4, height: 3 },
      alpha: Uint8Array.from({ length: 12 }, (_value, pixel) =>
        pixel === index % 12 ? 0 : 255
      ),
    }));
    const eyeMeshes = Array.from({ length: 20 }, (_, index) => patterns[index % patterns.length]!);

    const unique = new Set(eyeMeshes.map((entry) =>
      textureMaskIdentity(entry.atlasIndex, entry.bounds, entry.alpha)
    ));

    expect(eyeMeshes).toHaveLength(20);
    expect(unique).toHaveLength(7);
  });

  it("does not merge equal silhouettes from different atlases or coordinates", () => {
    const alpha = Uint8Array.from([255, 255, 255, 255]);
    const base = textureMaskIdentity(0, { x: 0, y: 0, width: 2, height: 2 }, alpha);

    expect(textureMaskIdentity(1, { x: 0, y: 0, width: 2, height: 2 }, alpha)).not.toBe(base);
    expect(textureMaskIdentity(0, { x: 1, y: 0, width: 2, height: 2 }, alpha)).not.toBe(base);
  });

  it("shares one in-flight structural scan between concurrent callers", async () => {
    const store = new ModelEditCapabilityStore({} as never, "/unused");
    let releaseFirst!: () => void;
    const firstMayFinish = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    let calls = 0;
    (store as unknown as {
      scanFresh(id: string): Promise<ModelEditCapabilityManifest>;
    }).scanFresh = async () => {
      calls += 1;
      const call = calls;
      if (call === 1) await firstMayFinish;
      return { sourceHash: `appearance-${call}` } as ModelEditCapabilityManifest;
    };

    const beforeMutation = store.scan("mao");
    await Promise.resolve();
    const afterMutation = store.scan("mao");
    await Promise.resolve();
    expect(calls).toBe(1);

    releaseFirst();
    expect((await beforeMutation).sourceHash).toBe("appearance-1");
    expect((await afterMutation).sourceHash).toBe("appearance-1");
    expect(calls).toBe(1);
  });
});

describe("motion-only texture discovery", () => {
  it("finds Parts revealed by linear and Bezier PartOpacity curves", () => {
    expect(visiblePartIdsFromMotion({
      Curves: [
        { Target: "PartOpacity", Id: "PartArmLB", Segments: [0, 0, 0, 0.5, 1] },
        { Target: "PartOpacity", Id: "PartArmRB", Segments: [0, 0, 1, 0.2, 0.7, 0.8, 0.7, 1, 0] },
        { Target: "PartOpacity", Id: "PartArmLA", Segments: [0, 0, 0, 1, 0] },
        { Target: "Parameter", Id: "ParamArm", Segments: [0, 0, 0, 1, 1] },
      ],
    })).toEqual(["PartArmLB", "PartArmRB"]);
  });
});

describe("authoritative appearance snapshots", () => {
  it("loads valid imported textures with trailing data even when part names are unclassified", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "capability-imported-appearance-"));
    onTestFinished(() => rm(root, { recursive: true, force: true }));
    const atlas = new PNG({ width: 16, height: 16 });
    atlas.data.fill(255);
    const bytes = Buffer.concat([PNG.sync.write(atlas), Buffer.from([0x8f, 0x57, 0x86, 0x98])]);
    await writeFile(path.join(root, "atlas.png"), bytes);
    await writeFile(path.join(root, "model.moc3"), "moc inspected below");
    await writeFile(path.join(root, "model.cdi3.json"), JSON.stringify({
      Parts: [{ Id: "Part32", Name: "左小腿" }],
    }));
    await writeFile(path.join(root, "model.model3.json"), JSON.stringify({
      FileReferences: { Moc: "model.moc3", Textures: ["atlas.png"], DisplayInfo: "model.cdi3.json" },
    }));
    const inspection = vi.spyOn(coreInspector, "inspectCubismModel").mockResolvedValue({
      mocVersion: 5, usesAdvancedBlendMode: false, usesOffscreenDrawing: false,
      canvas: { width: 100, height: 100, originX: 50, originY: 50, pixelsPerUnit: 100 },
      parameters: [], parts: [{ id: "Part32", parentIndex: -1 }],
      drawables: [{
        id: "ArtMesh149", texture: 0, drawOrder: 0, renderOrder: 0, opacity: 1, partIndex: 0,
        vertices: [-0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1],
        uvs: [0, 0, 1, 0, 1, 1, 0, 1], indices: [0, 1, 2, 0, 2, 3], masks: [],
      }],
    });
    onTestFinished(() => inspection.mockRestore());
    const models = {
      load: async () => ({ id: "imported", name: "Imported", manifest: { entry: "model.model3.json", motionAssets: {} } }),
      resolveModelRoot: () => root,
      resolveModelFile: (_id: string, relative: string) => path.join(root, relative),
    };
    const store = new ModelEditCapabilityStore(models as never, "/unused");
    const capability = await store.current("imported");
    expect(capability.textureUnits).toHaveLength(1);
    expect(capability.parts).toEqual([expect.objectContaining({
      // A Chinese Part name is recognized, and editability no longer depends on it.
      group: "leg", label: "腿",
      operations: expect.objectContaining({ appearance: { supported: true, alphaPolicy: "shrink-only" } }),
    })]);
    await store.ensureTexturePreview("imported", capability.textureUnits[0]!.previewPath);
    expect(PNG.sync.read(await readFile(path.join(root, capability.textureUnits[0]!.previewPath))).width).toBeGreaterThan(0);
    expect(await readFile(path.join(root, "atlas.png"))).toEqual(bytes);
  });

  const cached = {
    schemaVersion: 12,
    modelId: "mao",
    sourceHash: "topology-v1",
    textureUnits: [{ id: "unit-a" }, { id: "unit-b" }],
  } as ModelEditCapabilityManifest;
  const source = {
    modelRoot: "/models/mao",
    sourceHash: "topology-v1",
    baseTexturePaths: ["/models/mao/textures/atlas-0.png"],
    activeTexturePaths: ["/models/mao/.appearance/runtime/atlas-0.png"],
  };

  it("keeps the structural manifest unchanged when only runtime atlas pixels changed", async () => {
    const store = new ModelEditCapabilityStore({} as never, "/unused");
    store.get = async () => cached;
    (store as unknown as {
      loadSource(id: string): Promise<typeof source>;
      appearanceFingerprint(files: readonly string[]): Promise<string>;
      readAppearanceState(id: string): Promise<unknown>;
    }).loadSource = async () => source;
    (store as unknown as {
      appearanceFingerprint(files: readonly string[]): Promise<string>;
    }).appearanceFingerprint = async () => "appearance-v2";
    (store as unknown as {
      readAppearanceState(id: string): Promise<unknown>;
    }).readAppearanceState = async () => ({
      schemaVersion: 1,
      rendererVersion: 1,
      fingerprint: "appearance-v1",
    });
    (store as unknown as {
      appearanceArtifactsComplete(value: unknown): Promise<boolean>;
    }).appearanceArtifactsComplete = async () => true;
    let refresh: { atlases: readonly string[]; units: readonly string[] } | null = null;
    store.refreshAppearance = async (_id, atlases, units) => {
      refresh = { atlases, units };
      return cached;
    };

    await expect(store.current("mao")).resolves.toBe(cached);
    expect(refresh).toBeNull();
  });

  it("reuses the structural snapshot without hashing active appearance pixels", async () => {
    const store = new ModelEditCapabilityStore({} as never, "/unused");
    store.get = async () => cached;
    (store as unknown as {
      loadSource(id: string): Promise<typeof source>;
    }).loadSource = async () => source;
    (store as unknown as {
      appearanceFingerprint(files: readonly string[]): Promise<string>;
    }).appearanceFingerprint = async () => "appearance-v2";
    (store as unknown as {
      readAppearanceState(id: string): Promise<unknown>;
    }).readAppearanceState = async () => ({
      schemaVersion: 2,
      rendererVersion: 1,
      appearanceKey: "appearance-v2",
      generatedAt: "2026-08-20T00:00:00.000Z",
    });
    (store as unknown as {
      appearanceArtifactsComplete(value: unknown): Promise<boolean>;
    }).appearanceArtifactsComplete = async () => true;
    let refreshes = 0;
    store.refreshAppearance = async () => {
      refreshes += 1;
      return cached;
    };

    await expect(store.current("mao")).resolves.toBe(cached);
    expect(refreshes).toBe(0);
  });

  it("validates a cached structural manifest without resolving compiled atlases", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "capability-structural-current-"));
    onTestFinished(() => rm(root, { recursive: true, force: true }));
    const texture = path.join(root, "textures", "atlas.png");
    await mkdir(path.dirname(texture), { recursive: true });
    await writeFile(path.join(root, "model.model3.json"), JSON.stringify({
      FileReferences: {
        Moc: "model.moc3",
        Textures: ["textures/atlas.png"],
      },
    }));
    await writeFile(path.join(root, "model.moc3"), "moc");
    await writeFile(texture, "atlas");
    const models = {
      load: async () => ({
        manifest: { entry: "model.model3.json", motionAssets: {} },
      }),
      resolveModelRoot: () => root,
      resolveModelFile: (_id: string, relative: string) => path.join(root, relative),
    };
    const resolveCompiledAtlas = vi.fn(async () => {
      throw new Error("cached current must not wait for compaction");
    });
    const store = new ModelEditCapabilityStore(
      models as never,
      "/unused",
      resolveCompiledAtlas,
    );
    const structural = await (store as unknown as {
      loadSource(id: string, resolveAppearance: boolean): Promise<{ sourceHash: string }>;
    }).loadSource("mao", false);
    store.get = async () => ({ ...cached, sourceHash: structural.sourceHash });

    await expect(store.current("mao")).resolves.toMatchObject({
      sourceHash: structural.sourceHash,
    });
    expect(resolveCompiledAtlas).not.toHaveBeenCalled();
  });

  it("self-heals a matching state when its committed guide artifacts are missing", async () => {
    const store = new ModelEditCapabilityStore({} as never, "/unused");
    store.get = async () => cached;
    (store as unknown as {
      loadSource(id: string): Promise<typeof source>;
      appearanceFingerprint(files: readonly string[]): Promise<string>;
      readAppearanceState(id: string): Promise<unknown>;
      appearanceArtifactsComplete(value: unknown): Promise<boolean>;
    }).loadSource = async () => source;
    (store as unknown as {
      appearanceFingerprint(files: readonly string[]): Promise<string>;
    }).appearanceFingerprint = async () => "appearance-v2";
    (store as unknown as {
      readAppearanceState(id: string): Promise<unknown>;
    }).readAppearanceState = async () => ({
      schemaVersion: 2,
      rendererVersion: 1,
      appearanceKey: "appearance-v2",
      generatedAt: "2026-08-20T00:00:00.000Z",
    });
    (store as unknown as {
      appearanceArtifactsComplete(value: unknown): Promise<boolean>;
    }).appearanceArtifactsComplete = async () => false;
    let refreshes = 0;
    store.refreshAppearance = async () => {
      refreshes += 1;
      return cached;
    };

    await expect(store.ensureAlignmentGuide("mao")).resolves.toMatchObject({
      appearanceKey: "appearance-v2",
    });
    expect(refreshes).toBe(1);
  });
});
