import { describe, expect, it } from "vitest";

import type { TextureAtlasDelta } from "./contracts";
import { validateTextureAtlasDelta } from "./texture-patch-contract";

describe("validateTextureAtlasDelta", () => {
  it("accepts one deterministic self-contained patch set", () => {
    expect(validateTextureAtlasDelta(delta())).toEqual(delta());
  });

  it("rejects a snapshot that mixes patch and full-atlas replacement", () => {
    const value = delta();
    value.atlases.push({
      atlasIndex: 1,
      atlasFile: "textures/atlas-1.png",
      url: "pet-model://mao/runtime-1.png",
    });

    expect(() => validateTextureAtlasDelta(value))
      .toThrow("不能混合 patch 与完整图集替换");
  });

  it("rejects out-of-range bounds and unstable patch order", () => {
    const outOfRange = delta();
    outOfRange.patchSets[0]!.patches[0]!.bounds.x = 4;
    expect(() => validateTextureAtlasDelta(outOfRange)).toThrow("超出基础图集范围");

    const unordered = delta();
    unordered.patchSets[0]!.patches = [
      { ...unordered.patchSets[0]!.patches[0]!, textureUnitId: "z" },
      { ...unordered.patchSets[0]!.patches[0]!, textureUnitId: "a" },
    ];
    expect(() => validateTextureAtlasDelta(unordered)).toThrow("顺序不稳定");
  });
});

function delta(): TextureAtlasDelta {
  return {
    modelId: "mao",
    streamId: "model:mao",
    revision: 2,
    patchSets: [{
      atlasIndex: 0,
      atlasFile: "textures/atlas.png",
      sourceWidth: 4,
      sourceHeight: 4,
      baseUrl: "pet-model://mao/textures/atlas.png?content=base",
      patches: [{
        textureUnitId: "leg",
        bounds: { x: 1, y: 1, width: 2, height: 2 },
        imageUrl: "pet-model://mao/material.png?content=material",
        maskUrl: "pet-model://mao/mask.png?content=mask",
      }],
    }],
    atlases: [],
  };
}
