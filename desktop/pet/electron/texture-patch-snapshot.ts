import { createHash } from "node:crypto";

import type {
  TextureAtlasDelta,
  TextureAtlasPatchSet,
  TexturePatchUpdate,
} from "../src/shared/contracts";
import { validateTextureAtlasDelta } from "../src/shared/texture-patch-contract";

export interface TexturePatchMaterialView {
  id: string;
  textureUnitId: string;
  atlasIndex: number;
  atlasFile: string;
  bounds: { x: number; y: number; width: number; height: number };
  imageFile: string;
  maskFile: string;
  contentHash: string;
}

export interface TexturePatchLibraryView {
  modelId: string;
  revision: number;
  materials: TexturePatchMaterialView[];
  activeAssignments: Record<string, string>;
  baseContentHashes: Record<string, string>;
}

export interface TexturePatchFileIdentity {
  width: number;
  height: number;
  sha256: string;
}

export interface TexturePatchSnapshotResolver {
  atlasIdentity(atlasFile: string): Promise<TexturePatchFileIdentity>;
  imageIdentity(imageFile: string): Promise<TexturePatchFileIdentity>;
  maskIdentity(maskFile: string): Promise<TexturePatchFileIdentity>;
  modelUrl(relative: string, contentHash: string): string;
}

export async function buildTexturePatchSnapshot(
  library: TexturePatchLibraryView,
  affectedAtlasFiles: readonly string[],
  resolver: TexturePatchSnapshotResolver,
  streamId = `model:${library.modelId}`,
): Promise<{ delta: TextureAtlasDelta; contentKeys: Map<string, string> }> {
  const materialById = new Map(library.materials.map((material) => [material.id, material]));
  const active = Object.entries(library.activeAssignments).flatMap(([textureUnitId, materialId]) => {
    const material = materialById.get(materialId);
    return material?.textureUnitId === textureUnitId ? [material] : [];
  });
  const atlasFiles = [...new Set(affectedAtlasFiles)];
  const patchSetsWithKeys = await Promise.all(atlasFiles.map(async (atlasFile) => {
    const atlasMaterials = library.materials.filter((material) => material.atlasFile === atlasFile);
    const first = atlasMaterials[0];
    if (!first) throw new Error(`无法确定 patch 图集：${atlasFile}`);
    const atlasIdentity = await resolver.atlasIdentity(atlasFile);
    const activePatches = active
      .filter((material) => material.atlasFile === atlasFile)
      .filter((material) => material.contentHash !== library.baseContentHashes[material.textureUnitId])
      .sort((left, right) => left.textureUnitId.localeCompare(right.textureUnitId));
    const resolved = await Promise.all(activePatches.map(async (material) => {
      const [imageIdentity, maskIdentity] = await Promise.all([
        resolver.imageIdentity(material.imageFile),
        resolver.maskIdentity(material.maskFile),
      ]);
      if (
        imageIdentity.width !== material.bounds.width ||
        imageIdentity.height !== material.bounds.height
      ) throw new Error(`纹理单元 ${material.textureUnitId} 的素材尺寸不一致`);
      if (imageIdentity.sha256 !== material.contentHash) {
        throw new Error(`纹理单元 ${material.textureUnitId} 的素材内容已被意外修改`);
      }
      if (
        maskIdentity.width !== material.bounds.width ||
        maskIdentity.height !== material.bounds.height
      ) throw new Error(`纹理单元 ${material.textureUnitId} 的蒙版尺寸不一致`);
      const patch: TexturePatchUpdate = {
        textureUnitId: material.textureUnitId,
        bounds: { ...material.bounds },
        imageUrl: resolver.modelUrl(material.imageFile, material.contentHash),
        maskUrl: resolver.modelUrl(material.maskFile, maskIdentity.sha256),
      };
      return { material, imageIdentity, maskIdentity, patch };
    }));
    const patchSet: TextureAtlasPatchSet = {
      atlasIndex: first.atlasIndex,
      atlasFile,
      sourceWidth: atlasIdentity.width,
      sourceHeight: atlasIdentity.height,
      baseUrl: immutableBaseUrl(resolver.modelUrl(atlasFile, atlasIdentity.sha256)),
      patches: resolved.map((item) => item.patch),
    };
    const contentKey = stableHash({
      base: atlasIdentity.sha256,
      atlasFile,
      patches: resolved.map(({ material, imageIdentity, maskIdentity }) => ({
        id: material.id,
        textureUnitId: material.textureUnitId,
        contentHash: imageIdentity.sha256,
        maskHash: maskIdentity.sha256,
        bounds: material.bounds,
      })),
    });
    return { patchSet, contentKey };
  }));
  patchSetsWithKeys.sort((left, right) => left.patchSet.atlasIndex - right.patchSet.atlasIndex);
  const delta: TextureAtlasDelta = {
    modelId: library.modelId,
    streamId,
    revision: library.revision,
    patchSets: patchSetsWithKeys.map((item) => item.patchSet),
    atlases: [],
  };
  validateTextureAtlasDelta(delta);
  return {
    delta,
    contentKeys: new Map(patchSetsWithKeys.map((item) => [item.patchSet.atlasFile, item.contentKey])),
  };
}

function stableHash(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

function immutableBaseUrl(value: string): string {
  const url = new URL(value);
  url.searchParams.set("textureBase", "1");
  return url.toString();
}
