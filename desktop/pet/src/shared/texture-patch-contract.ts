import type { TextureAtlasDelta } from "./contracts";

export function validateTextureAtlasDelta(delta: TextureAtlasDelta): TextureAtlasDelta {
  if (!delta.modelId.trim()) throw new Error("纹理更新缺少模型 ID");
  if (!delta.streamId.trim()) throw new Error("纹理更新缺少数据流 ID");
  if (!Number.isSafeInteger(delta.revision) || delta.revision < 0) {
    throw new Error("纹理更新 revision 无效");
  }
  if (delta.atlases.length > 0 && delta.patchSets.length > 0) {
    throw new Error("同一纹理 snapshot 不能混合 patch 与完整图集替换");
  }

  const patchAtlases = new Set<number>();
  let previousAtlasIndex = -1;
  for (const set of delta.patchSets) {
    if (!Number.isSafeInteger(set.atlasIndex) || set.atlasIndex < 0) {
      throw new Error("patch 图集索引无效");
    }
    if (set.atlasIndex <= previousAtlasIndex) throw new Error("patch 图集顺序不稳定");
    previousAtlasIndex = set.atlasIndex;
    if (patchAtlases.has(set.atlasIndex)) throw new Error(`patch 图集 ${set.atlasIndex} 重复`);
    patchAtlases.add(set.atlasIndex);
    if (!set.atlasFile.trim() || !set.baseUrl.trim()) throw new Error("patch 图集资源不能为空");
    if (
      !Number.isSafeInteger(set.sourceWidth) || set.sourceWidth <= 0 ||
      !Number.isSafeInteger(set.sourceHeight) || set.sourceHeight <= 0
    ) throw new Error(`patch 图集 ${set.atlasIndex} 尺寸无效`);

    let previousTextureUnitId = "";
    const units = new Set<string>();
    for (const patch of set.patches) {
      if (!patch.textureUnitId.trim()) throw new Error("patch 缺少纹理单元 ID");
      if (previousTextureUnitId && patch.textureUnitId.localeCompare(previousTextureUnitId) <= 0) {
        throw new Error(`patch 图集 ${set.atlasIndex} 的纹理单元顺序不稳定`);
      }
      previousTextureUnitId = patch.textureUnitId;
      if (units.has(patch.textureUnitId)) throw new Error(`纹理单元 ${patch.textureUnitId} 重复`);
      units.add(patch.textureUnitId);
      const { x, y, width, height } = patch.bounds;
      if (
        ![x, y, width, height].every(Number.isSafeInteger) ||
        x < 0 || y < 0 || width <= 0 || height <= 0 ||
        x + width > set.sourceWidth || y + height > set.sourceHeight
      ) throw new Error(`纹理单元 ${patch.textureUnitId} 超出基础图集范围`);
      if (!patch.imageUrl.trim() || !patch.maskUrl.trim()) {
        throw new Error(`纹理单元 ${patch.textureUnitId} 的 patch 资源不能为空`);
      }
    }
  }

  const fullAtlases = new Set<number>();
  for (const atlas of delta.atlases) {
    if (!Number.isSafeInteger(atlas.atlasIndex) || atlas.atlasIndex < 0) {
      throw new Error("完整图集索引无效");
    }
    if (fullAtlases.has(atlas.atlasIndex)) throw new Error(`完整图集 ${atlas.atlasIndex} 重复`);
    if (patchAtlases.has(atlas.atlasIndex)) {
      throw new Error(`同一图集不能同时使用 patch 和完整替换：${atlas.atlasIndex}`);
    }
    fullAtlases.add(atlas.atlasIndex);
  }
  return delta;
}
