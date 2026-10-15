import { describe, expect, it } from "vitest";

import {
  buildTexturePatchSnapshot,
  type TexturePatchLibraryView,
  type TexturePatchSnapshotResolver,
} from "./texture-patch-snapshot";

describe("buildTexturePatchSnapshot", () => {
  it("projects every active non-base material on the affected atlas", async () => {
    const result = await buildTexturePatchSnapshot(library(), ["textures/atlas-0.png"], resolver());

    expect(result.delta).toMatchObject({
      modelId: "mao",
      streamId: "model:mao",
      revision: 7,
      atlases: [],
      patchSets: [{
        atlasIndex: 0,
        sourceWidth: 4096,
        sourceHeight: 8192,
      }],
    });
    expect(result.delta.patchSets[0]!.patches.map((item) => item.textureUnitId))
      .toEqual(["left-leg", "right-leg"]);
    expect(result.delta.patchSets[0]!.patches[0]).toMatchObject({
      imageUrl: expect.stringContaining("content=left-custom"),
      maskUrl: expect.stringContaining("content=left-mask"),
    });
    expect(result.contentKeys.get("textures/atlas-0.png")).toMatch(/^[a-f0-9]{64}$/);
  });

  it("emits an empty patch set when the affected atlas returns to base", async () => {
    const value = library();
    value.activeAssignments = {
      "left-leg": "left-base",
      "right-leg": "right-base",
      hair: "hair-base",
    };

    const result = await buildTexturePatchSnapshot(value, ["textures/atlas-0.png"], resolver());

    expect(result.delta.patchSets).toHaveLength(1);
    expect(result.delta.patchSets[0]!.patches).toEqual([]);
  });

  it("changes the content key when active material content changes", async () => {
    const first = await buildTexturePatchSnapshot(library(), ["textures/atlas-0.png"], resolver());
    const changed = library();
    const material = changed.materials.find((item) => item.id === "left-custom")!;
    material.contentHash = "left-custom-v2";
    material.imageFile = "left-custom-v2.png";
    const second = await buildTexturePatchSnapshot(changed, ["textures/atlas-0.png"], resolver());

    expect(second.contentKeys.get("textures/atlas-0.png"))
      .not.toBe(first.contentKeys.get("textures/atlas-0.png"));
  });

  it("rejects a generated material whose immutable bytes no longer match its journal", async () => {
    const changed = library();
    changed.materials.find((item) => item.id === "left-custom")!.contentHash = "tampered";

    await expect(buildTexturePatchSnapshot(
      changed,
      ["textures/atlas-0.png"],
      resolver(),
    )).rejects.toThrow("素材内容已被意外修改");
  });
});

function library(): TexturePatchLibraryView {
  return {
    modelId: "mao",
    revision: 7,
    materials: [
      material("left-base", "left-leg", 0, "textures/atlas-0.png", "base:left", "left-base.png", "left-mask.png"),
      material("left-custom", "left-leg", 0, "textures/atlas-0.png", "left-custom", "left-custom.png", "left-mask.png"),
      material("right-base", "right-leg", 0, "textures/atlas-0.png", "base:right", "right-base.png", "right-mask.png"),
      material("right-custom", "right-leg", 0, "textures/atlas-0.png", "right-custom", "right-custom.png", "right-mask.png"),
      material("right-unused", "right-leg", 0, "textures/atlas-0.png", "right-unused", "right-unused.png", "right-mask.png"),
      material("hair-base", "hair", 1, "textures/atlas-1.png", "base:hair", "hair-base.png", "hair-mask.png"),
    ],
    activeAssignments: {
      "left-leg": "left-custom",
      "right-leg": "right-custom",
      hair: "hair-base",
    },
    baseContentHashes: {
      "left-leg": "base:left",
      "right-leg": "base:right",
      hair: "base:hair",
    },
  };
}

function material(
  id: string,
  textureUnitId: string,
  atlasIndex: number,
  atlasFile: string,
  contentHash: string,
  imageFile: string,
  maskFile: string,
) {
  return {
    id,
    textureUnitId,
    atlasIndex,
    atlasFile,
    bounds: textureUnitId === "right-leg"
      ? { x: 20, y: 0, width: 10, height: 20 }
      : { x: 0, y: 0, width: 10, height: 20 },
    imageFile,
    maskFile,
    contentHash,
  };
}

function resolver(): TexturePatchSnapshotResolver {
  return {
    atlasIdentity: async (atlasFile) => ({
      width: atlasFile.endsWith("0.png") ? 4096 : 2048,
      height: atlasFile.endsWith("0.png") ? 8192 : 2048,
      sha256: atlasFile.endsWith("0.png") ? "atlas-zero" : "atlas-one",
    }),
    maskIdentity: async (maskFile) => ({
      width: 10,
      height: 20,
      sha256: maskFile.replace(".png", "-mask"),
    }),
    imageIdentity: async (imageFile) => ({
      width: 10,
      height: 20,
      sha256: imageFile.replace(".png", ""),
    }),
    modelUrl: (relative, contentHash) =>
      `pet-model://mao/${relative}?content=${encodeURIComponent(contentHash)}`,
  };
}
