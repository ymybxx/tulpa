import { describe, expect, it, vi } from "vitest";

import type { TextureAtlasDelta, TextureAtlasPatchSet } from "../../src/shared/contracts";
import {
  buildTexturePreviewSnapshot,
  type TexturePreviewJobEdit,
} from "./texture-preview-snapshot";

describe("buildTexturePreviewSnapshot", () => {
  it("returns the complete official patch sets for affected atlases when disabled", () => {
    const official = officialSnapshot();
    const original = structuredClone(official);

    const result = buildTexturePreviewSnapshot({
      official,
      jobId: "job-7",
      revision: 11,
      enabled: false,
      affectedAtlases: [
        { atlasIndex: 1, atlasFile: "textures/atlas-1.png" },
        { atlasIndex: 0, atlasFile: "textures/atlas-0.png" },
      ],
      // Disabled restoration must not depend on job artifacts still existing.
      acceptedEdits: [jobEdit("body", 0, {})],
    });

    expect(result).toMatchObject({
      modelId: "mao",
      streamId: "model:mao",
      revision: 6,
      atlases: [],
    });
    expect(result.patchSets.map((set) => set.atlasIndex)).toEqual([0, 1]);
    expect(result.patchSets[0]!.patches).toEqual(official.patchSets[0]!.patches);
    expect(result.patchSets[1]!.patches).toEqual(official.patchSets[1]!.patches);
    expect(result.patchSets[0]).not.toBe(official.patchSets[0]);
    expect(official).toEqual(original);
  });

  it("deterministically replaces and appends accepted job patches per atlas", () => {
    const resolveImageUrl = vi.fn((edit: TexturePreviewJobEdit) =>
      `pet-generation://job-7/${edit.imagePath}`
    );
    const resolveMaskUrl = vi.fn((edit: TexturePreviewJobEdit) =>
      `pet-generation://job-7/${edit.maskPath}`
    );

    const result = buildTexturePreviewSnapshot({
      official: officialSnapshot(),
      jobId: "job-7",
      revision: 12,
      enabled: true,
      affectedAtlases: [
        { atlasIndex: 1, atlasFile: "textures/atlas-1.png" },
        { atlasIndex: 0, atlasFile: "textures/atlas-0.png" },
      ],
      acceptedEdits: [
        jobEdit("new-accessory", 1, {
          imagePath: "accessory.png",
          maskPath: "accessory-mask.png",
        }),
        jobEdit("body", 0, {
          imageUrl: "pet-generation://job-7/body.png",
          maskUrl: "pet-generation://job-7/body-mask.png",
        }),
      ],
      resolver: { imageUrl: resolveImageUrl, maskUrl: resolveMaskUrl },
    });

    expect(result.patchSets.map((set) => set.atlasIndex)).toEqual([0, 1]);
    expect(result.patchSets[0]!.patches.map((patch) => patch.textureUnitId))
      .toEqual(["body", "shoe"]);
    expect(result.patchSets[0]!.patches).toEqual([
      expect.objectContaining({
        textureUnitId: "body",
        imageUrl: "pet-generation://job-7/body.png",
        maskUrl: "pet-generation://job-7/body-mask.png",
      }),
      expect.objectContaining({
        textureUnitId: "shoe",
        imageUrl: "pet-model://mao/materials/shoe.png?content=shoe",
      }),
    ]);
    expect(result.patchSets[1]!.patches.map((patch) => patch.textureUnitId))
      .toEqual(["hair", "new-accessory"]);
    expect(result.patchSets[1]!.patches[1]).toMatchObject({
      imageUrl: "pet-generation://job-7/accessory.png",
      maskUrl: "pet-generation://job-7/accessory-mask.png",
    });
    expect(resolveImageUrl).toHaveBeenCalledTimes(1);
    expect(resolveMaskUrl).toHaveBeenCalledTimes(1);
  });

  it("rejects ambiguous job edits and affected official full replacements", () => {
    const duplicate = jobEdit("body", 0, {
      imageUrl: "pet-generation://job-7/body.png",
      maskUrl: "pet-generation://job-7/body-mask.png",
    });
    expect(() => buildTexturePreviewSnapshot({
      official: officialSnapshot(),
      jobId: "job-7",
      revision: 13,
      enabled: true,
      affectedAtlases: [{ atlasIndex: 0, atlasFile: "textures/atlas-0.png" }],
      acceptedEdits: [duplicate, { ...duplicate }],
    })).toThrow("重复");

    const full = officialSnapshot();
    full.patchSets = full.patchSets.slice(1);
    full.atlases = [{
      atlasIndex: 0,
      atlasFile: "textures/atlas-0.png",
      url: "pet-model://mao/compiled/atlas-0.png",
    }];
    expect(() => buildTexturePreviewSnapshot({
      official: full,
      jobId: "job-7",
      revision: 14,
      enabled: false,
      affectedAtlases: [{ atlasIndex: 0, atlasFile: "textures/atlas-0.png" }],
      acceptedEdits: [],
    })).toThrow("不能混合 patch 与完整图集替换");
  });

  it("validates merged job patch bounds against the official base atlas", () => {
    expect(() => buildTexturePreviewSnapshot({
      official: officialSnapshot(),
      jobId: "job-7",
      revision: 15,
      enabled: true,
      affectedAtlases: [{ atlasIndex: 0, atlasFile: "textures/atlas-0.png" }],
      acceptedEdits: [{
        ...jobEdit("body", 0, {
          imageUrl: "pet-generation://job-7/body.png",
          maskUrl: "pet-generation://job-7/body-mask.png",
        }),
        bounds: { x: 95, y: 0, width: 10, height: 10 },
      }],
    })).toThrow("超出基础图集范围");
  });
});

function officialSnapshot(): TextureAtlasDelta {
  return {
    modelId: "mao",
    streamId: "model:mao",
    revision: 6,
    patchSets: [
      patchSet(0, "textures/atlas-0.png", [
        patch("body", 0, "body"),
        patch("shoe", 20, "shoe"),
      ]),
      patchSet(1, "textures/atlas-1.png", [
        patch("hair", 0, "hair"),
      ]),
    ],
    atlases: [],
  };
}

function patchSet(
  atlasIndex: number,
  atlasFile: string,
  patches: TextureAtlasPatchSet["patches"],
): TextureAtlasPatchSet {
  return {
    atlasIndex,
    atlasFile,
    sourceWidth: 100,
    sourceHeight: 100,
    baseUrl: `pet-model://mao/${atlasFile}?content=base-${atlasIndex}&textureBase=1`,
    patches,
  };
}

function patch(textureUnitId: string, x: number, content: string) {
  return {
    textureUnitId,
    bounds: { x, y: 0, width: 10, height: 10 },
    imageUrl: `pet-model://mao/materials/${content}.png?content=${content}`,
    maskUrl: `pet-model://mao/masks/${content}.png?content=${content}-mask`,
  };
}

function jobEdit(
  textureUnitId: string,
  atlasIndex: number,
  resource: Pick<TexturePreviewJobEdit, "imageUrl" | "maskUrl" | "imagePath" | "maskPath">,
): TexturePreviewJobEdit {
  return {
    textureUnitId,
    atlasIndex,
    atlasFile: `textures/atlas-${atlasIndex}.png`,
    bounds: {
      x: textureUnitId === "body" ? 0 : 40,
      y: 0,
      width: 10,
      height: 10,
    },
    ...resource,
  };
}
