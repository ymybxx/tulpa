import type {
  TextureAtlasDelta,
  TextureAtlasPatchSet,
  TexturePatchUpdate,
} from "../../src/shared/contracts";
import { validateTextureAtlasDelta } from "../../src/shared/texture-patch-contract";

export interface TexturePreviewAffectedAtlas {
  atlasIndex: number;
  atlasFile: string;
}

export interface TexturePreviewJobEdit {
  textureUnitId: string;
  atlasIndex: number;
  atlasFile: string;
  bounds: { x: number; y: number; width: number; height: number };
  /** A ready-to-render job-scoped URL. */
  imageUrl?: string;
  /** A ready-to-render job-scoped URL. */
  maskUrl?: string;
  /** Opaque source value used by an injected URL resolver. */
  imagePath?: string;
  /** Opaque source value used by an injected URL resolver. */
  maskPath?: string;
}

export interface TexturePreviewJobUrlResolver {
  imageUrl(edit: TexturePreviewJobEdit): string;
  maskUrl(edit: TexturePreviewJobEdit): string;
}

export interface TexturePreviewSnapshotInput {
  /** Complete official material snapshot for every affected atlas. */
  official: TextureAtlasDelta;
  jobId: string;
  revision: number;
  enabled: boolean;
  affectedAtlases: readonly TexturePreviewAffectedAtlas[];
  acceptedEdits: readonly TexturePreviewJobEdit[];
  resolver?: TexturePreviewJobUrlResolver;
}

/**
 * Projects a complete job-scoped preview snapshot over the official appearance.
 *
 * This function deliberately performs no I/O. Callers may either supply ready
 * job URLs or inject a resolver that maps their job artifact paths to URLs.
 */
export function buildTexturePreviewSnapshot(
  input: TexturePreviewSnapshotInput,
): TextureAtlasDelta {
  validateTextureAtlasDelta(input.official);
  const jobId = input.jobId.trim();
  if (!jobId) throw new Error("纹理预览缺少任务 ID");

  const affected = normalizeAffectedAtlases(input.affectedAtlases);
  const officialByIndex = new Map(
    input.official.patchSets.map((patchSet) => [patchSet.atlasIndex, patchSet]),
  );
  const fullAtlasIndices = new Set(input.official.atlases.map((atlas) => atlas.atlasIndex));
  const patchSets = affected.map((atlas) => {
    if (fullAtlasIndices.has(atlas.atlasIndex)) {
      throw new Error(`受影响图集 ${atlas.atlasIndex} 仍使用完整替换，无法建立 patch 预览`);
    }
    const official = officialByIndex.get(atlas.atlasIndex);
    if (!official) throw new Error(`官方快照缺少受影响图集 ${atlas.atlasIndex} 的完整 patch set`);
    if (official.atlasFile !== atlas.atlasFile) {
      throw new Error(`受影响图集 ${atlas.atlasIndex} 的文件与官方快照不一致`);
    }
    return clonePatchSet(official);
  });

  if (input.enabled) mergeAcceptedEdits(patchSets, affected, input.acceptedEdits, input.resolver);

  const delta: TextureAtlasDelta = {
    modelId: input.official.modelId,
    streamId: input.enabled ? `job:${jobId}` : input.official.streamId,
    revision: input.enabled ? input.revision : input.official.revision,
    patchSets,
    atlases: [],
  };
  return validateTextureAtlasDelta(delta);
}

function normalizeAffectedAtlases(
  values: readonly TexturePreviewAffectedAtlas[],
): TexturePreviewAffectedAtlas[] {
  const indices = new Set<number>();
  const files = new Set<string>();
  const result = values.map((value) => {
    if (!Number.isSafeInteger(value.atlasIndex) || value.atlasIndex < 0) {
      throw new Error("受影响图集索引无效");
    }
    const atlasFile = value.atlasFile.trim();
    if (!atlasFile) throw new Error("受影响图集文件不能为空");
    if (indices.has(value.atlasIndex) || files.has(atlasFile)) {
      throw new Error(`受影响图集重复：${value.atlasIndex}`);
    }
    indices.add(value.atlasIndex);
    files.add(atlasFile);
    return { atlasIndex: value.atlasIndex, atlasFile };
  });
  result.sort((left, right) => left.atlasIndex - right.atlasIndex);
  return result;
}

function mergeAcceptedEdits(
  patchSets: TextureAtlasPatchSet[],
  affected: readonly TexturePreviewAffectedAtlas[],
  edits: readonly TexturePreviewJobEdit[],
  resolver: TexturePreviewJobUrlResolver | undefined,
): void {
  const affectedByIndex = new Map(affected.map((atlas) => [atlas.atlasIndex, atlas]));
  const affectedByFile = new Map(affected.map((atlas) => [atlas.atlasFile, atlas]));
  const patchSetByIndex = new Map(patchSets.map((patchSet) => [patchSet.atlasIndex, patchSet]));
  const seen = new Set<string>();

  for (const edit of edits) {
    const atlasByIndex = affectedByIndex.get(edit.atlasIndex);
    const atlasByFile = affectedByFile.get(edit.atlasFile);
    if (!atlasByIndex && !atlasByFile) continue;
    if (!atlasByIndex || !atlasByFile || atlasByIndex !== atlasByFile) {
      throw new Error(`纹理单元 ${edit.textureUnitId} 的受影响图集不一致`);
    }
    const textureUnitId = edit.textureUnitId.trim();
    if (!textureUnitId) throw new Error("job patch 缺少纹理单元 ID");
    const key = `${edit.atlasIndex}\0${textureUnitId}`;
    if (seen.has(key)) throw new Error(`job patch 纹理单元 ${textureUnitId} 重复`);
    seen.add(key);

    const patchSet = patchSetByIndex.get(edit.atlasIndex);
    if (!patchSet) throw new Error(`受影响图集 ${edit.atlasIndex} 缺少 patch set`);
    const jobPatch: TexturePatchUpdate = {
      textureUnitId,
      bounds: { ...edit.bounds },
      imageUrl: resolveJobUrl("image", edit, resolver),
      maskUrl: resolveJobUrl("mask", edit, resolver),
    };
    const index = patchSet.patches.findIndex((patch) => patch.textureUnitId === textureUnitId);
    if (index >= 0) patchSet.patches[index] = jobPatch;
    else patchSet.patches.push(jobPatch);
  }

  for (const patchSet of patchSets) {
    patchSet.patches.sort((left, right) => left.textureUnitId.localeCompare(right.textureUnitId));
  }
}

function resolveJobUrl(
  kind: "image" | "mask",
  edit: TexturePreviewJobEdit,
  resolver: TexturePreviewJobUrlResolver | undefined,
): string {
  const direct = kind === "image" ? edit.imageUrl : edit.maskUrl;
  const resolved = direct?.trim() || resolver?.[`${kind}Url`](edit).trim();
  if (!resolved) throw new Error(`纹理单元 ${edit.textureUnitId} 缺少 job ${kind} URL`);
  return resolved;
}

function clonePatchSet(value: TextureAtlasPatchSet): TextureAtlasPatchSet {
  return {
    atlasIndex: value.atlasIndex,
    atlasFile: value.atlasFile,
    sourceWidth: value.sourceWidth,
    sourceHeight: value.sourceHeight,
    baseUrl: value.baseUrl,
    patches: value.patches.map((patch) => ({
      textureUnitId: patch.textureUnitId,
      bounds: { ...patch.bounds },
      imageUrl: patch.imageUrl,
      maskUrl: patch.maskUrl,
    })),
  };
}
