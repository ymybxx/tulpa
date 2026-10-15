import { access, lstat, realpath } from "node:fs/promises";
import path from "node:path";

import type { TextureAtlasDelta } from "../src/shared/contracts";
import type { AtlasCompactionRequest } from "./atlas-compaction-coordinator";
import {
  composeTextureAtlasDelta,
  type TextureAtlasOutput,
} from "./texture-atlas-composer";

const MAX_ALLOWED_ROOTS = 16;
const MAX_ATLASES = 32;
const MAX_EDITS = 1_024;
const MAX_SERIALIZED_BYTES = 2 * 1024 * 1024;
const CONTENT_KEY_PATTERN = /^[a-f0-9]{64}$/;

export type AtlasCompactionCompose = (input: {
  modelId: string;
  streamId: string;
  revision: number;
  outputs: readonly TextureAtlasOutput[];
  edits: AtlasCompactionRequest["atlases"][number]["edits"];
}) => Promise<TextureAtlasDelta>;

export interface AtlasCompactionJobDependencies {
  compose?: AtlasCompactionCompose;
}

export interface AtlasCompactionJobResult {
  modelId: string;
  revision: number;
  contentKeys: string[];
  built: number;
  reused: number;
}

export async function executeAtlasCompactionRequest(
  request: AtlasCompactionRequest,
  dependencies: AtlasCompactionJobDependencies = {},
): Promise<AtlasCompactionJobResult> {
  assertAtlasCompactionRequest(request);
  await validateAtlasCompactionRequestPaths(request);
  const missing: AtlasCompactionRequest["atlases"] = [];
  let reused = 0;
  for (const atlas of request.atlases) {
    if (await exists(atlas.targetPath)) reused += 1;
    else missing.push(atlas);
  }
  if (missing.length > 0) {
    const compose = dependencies.compose ?? composeTextureAtlasDelta;
    await compose({
      modelId: request.modelId,
      streamId: `compaction:${request.modelId}`,
      revision: request.revision,
      outputs: missing.map((atlas) => ({
        atlasIndex: atlas.edits[0]?.atlasIndex ?? 0,
        atlasFile: atlas.atlasFile,
        sourcePath: atlas.sourcePath,
        targetPath: atlas.targetPath,
        url: `compiled://${atlas.contentKey}`,
      })),
      edits: missing.flatMap((atlas) => atlas.edits),
    });
    await Promise.all(missing.map(async (atlas) => {
      const info = await lstat(atlas.targetPath);
      if (!info.isFile()) throw new Error(`压实结果不是文件：${atlas.targetPath}`);
    }));
  }
  return {
    modelId: request.modelId,
    revision: request.revision,
    contentKeys: request.atlases.map((atlas) => atlas.contentKey).sort(),
    built: missing.length,
    reused,
  };
}

export function assertAtlasCompactionRequest(
  value: unknown,
): asserts value is AtlasCompactionRequest {
  if (!isRecord(value)) throw new Error("图集压实请求必须是对象");
  if (typeof value.modelId !== "string" || !value.modelId.trim() || value.modelId.length > 256) {
    throw new Error("图集压实请求的模型 ID 无效");
  }
  if (!Number.isSafeInteger(value.revision) || Number(value.revision) < 0) {
    throw new Error("图集压实请求的版本号无效");
  }
  if (
    !Array.isArray(value.allowedRoots) ||
    value.allowedRoots.length === 0 ||
    value.allowedRoots.length > MAX_ALLOWED_ROOTS ||
    !value.allowedRoots.every(validAbsolutePath)
  ) throw new Error("图集压实请求的允许目录无效");
  if (!Array.isArray(value.atlases) || value.atlases.length > MAX_ATLASES) {
    throw new Error("图集压实请求包含过多图集");
  }
  if (Buffer.byteLength(JSON.stringify(value), "utf8") > MAX_SERIALIZED_BYTES) {
    throw new Error("图集压实请求超过大小限制");
  }
  const seenTargets = new Set<string>();
  const seenAtlasFiles = new Set<string>();
  let editCount = 0;
  for (const atlas of value.atlases) {
    if (!isRecord(atlas)) throw new Error("图集压实项必须是对象");
    if (typeof atlas.contentKey !== "string" || !CONTENT_KEY_PATTERN.test(atlas.contentKey)) {
      throw new Error("图集压实内容键无效");
    }
    if (typeof atlas.atlasFile !== "string" || !validRelativePath(atlas.atlasFile)) {
      throw new Error("图集压实相对路径无效");
    }
    if (!validAbsolutePath(atlas.sourcePath) || !validAbsolutePath(atlas.targetPath)) {
      throw new Error("图集压实文件路径无效");
    }
    if (!Array.isArray(atlas.edits)) throw new Error("图集压实编辑列表无效");
    editCount += atlas.edits.length;
    if (editCount > MAX_EDITS) throw new Error("图集压实请求包含过多编辑");
    const atlasIndices = new Set<number>();
    for (const edit of atlas.edits) {
      if (!isRecord(edit) || typeof edit.textureUnitId !== "string" || !edit.textureUnitId) {
        throw new Error("图集压实编辑无效");
      }
      if (!Number.isSafeInteger(edit.atlasIndex) || Number(edit.atlasIndex) < 0) {
        throw new Error("图集压实图集索引无效");
      }
      atlasIndices.add(Number(edit.atlasIndex));
      if (!validBounds(edit.bounds)) throw new Error("图集压实编辑范围无效");
      if (!validAbsolutePath(edit.maskPath) || !validAbsolutePath(edit.imagePath)) {
        throw new Error("图集压实编辑文件路径无效");
      }
    }
    if (atlasIndices.size > 1) throw new Error("一个压实项不能包含多个图集索引");
    if (seenTargets.has(atlas.targetPath)) throw new Error("图集压实目标路径重复");
    if (seenAtlasFiles.has(atlas.atlasFile)) throw new Error("图集压实相对路径重复");
    seenTargets.add(atlas.targetPath);
    seenAtlasFiles.add(atlas.atlasFile);
  }
}

/** Re-validates every worker path after structured cloning and before any I/O. */
export async function validateAtlasCompactionRequestPaths(
  request: AtlasCompactionRequest,
): Promise<void> {
  assertAtlasCompactionRequest(request);
  const roots = await Promise.all(request.allowedRoots.map(async (root) => ({
    lexical: path.resolve(root),
    real: await realpath(root),
  })));
  for (const atlas of request.atlases) {
    await validateExistingFile(atlas.sourcePath, roots.map((root) => root.real));
    for (const edit of atlas.edits) {
      await validateExistingFile(edit.maskPath, roots.map((root) => root.real));
      await validateExistingFile(edit.imagePath, roots.map((root) => root.real));
    }
    await validateTarget(atlas.targetPath, roots);
    const relativeTarget = roots
      .filter((root) => isWithin(root.lexical, path.resolve(atlas.targetPath)))
      .map((root) => path.relative(root.lexical, atlas.targetPath))
      .find((relative) => relative.split(path.sep).includes(atlas.contentKey));
    if (!relativeTarget) throw new Error("图集压实目标不在内容键目录中");
    const normalizedAtlasFile = atlas.atlasFile.split("/").join(path.sep);
    if (!path.resolve(atlas.targetPath).endsWith(`${path.sep}${normalizedAtlasFile}`)) {
      throw new Error("图集压实目标与图集相对路径不一致");
    }
  }
}

async function validateExistingFile(file: string, roots: readonly string[]): Promise<void> {
  const resolved = await realpath(file);
  if (!roots.some((root) => isWithin(root, resolved))) {
    throw new Error(`图集压实文件位于允许目录之外：${file}`);
  }
  const info = await lstat(resolved);
  if (!info.isFile()) throw new Error(`图集压实输入不是文件：${file}`);
}

async function validateTarget(
  file: string,
  roots: ReadonlyArray<{ lexical: string; real: string }>,
): Promise<void> {
  const resolved = path.resolve(file);
  if (!roots.some((root) => isWithin(root.lexical, resolved))) {
    throw new Error(`图集压实目标位于允许目录之外：${file}`);
  }
  let ancestor = resolved;
  for (;;) {
    try {
      const actual = await realpath(ancestor);
      if (!roots.some((root) => isWithin(root.real, actual))) {
        throw new Error(`图集压实目标位于允许目录之外：${file}`);
      }
      return;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      const parent = path.dirname(ancestor);
      if (parent === ancestor) throw new Error(`图集压实目标没有有效父目录：${file}`);
      ancestor = parent;
    }
  }
}

function validAbsolutePath(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= 4_096 && path.isAbsolute(value);
}

function validRelativePath(value: string): boolean {
  if (value.length === 0 || value.length > 2_048 || path.posix.isAbsolute(value)) return false;
  const normalized = path.posix.normalize(value.replaceAll("\\", "/"));
  return normalized !== ".." && !normalized.startsWith("../") && normalized === value.replaceAll("\\", "/");
}

function validBounds(value: unknown): boolean {
  if (!isRecord(value)) return false;
  return [value.x, value.y, value.width, value.height].every(Number.isSafeInteger) &&
    Number(value.x) >= 0 && Number(value.y) >= 0 &&
    Number(value.width) > 0 && Number(value.height) > 0;
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function exists(file: string): Promise<boolean> {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}
