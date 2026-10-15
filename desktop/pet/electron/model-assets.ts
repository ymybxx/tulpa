import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import type {
  ExpressionAsset,
  MotionAsset,
} from "../src/shared/contracts";

interface CubismModelJson {
  FileReferences?: {
    Motions?: unknown;
    Expressions?: unknown;
  };
}

export interface ModelAssetCatalog {
  motionAssets: Record<string, MotionAsset>;
  expressionAssets: Record<string, ExpressionAsset>;
}

export async function scanModelAssets(
  root: string,
  entry: string,
  model: CubismModelJson,
): Promise<ModelAssetCatalog> {
  const motionAssets: Record<string, MotionAsset> = {};
  const expressionAssets: Record<string, ExpressionAsset> = {};
  const references = isRecord(model.FileReferences) ? model.FileReferences : {};
  const modelDirectory = path.posix.dirname(entry);

  if (isRecord(references.Motions)) {
    for (const [group, motions] of Object.entries(references.Motions)) {
      if (!Array.isArray(motions)) continue;
      for (const [index, value] of motions.entries()) {
        if (!isRecord(value) || typeof value.File !== "string") continue;
        const file = joinModelPath(modelDirectory, value.File);
        const id = stableId("motion", group, String(index), file);
        motionAssets[id] = {
          id,
          name: assetName(file, ".motion3.json"),
          file,
          group,
          index,
          contentHash: await contentHash(root, file, value),
        };
      }
    }
  }

  if (Array.isArray(references.Expressions)) {
    for (const value of references.Expressions) {
      if (
        !isRecord(value) ||
        typeof value.File !== "string" ||
        typeof value.Name !== "string"
      ) {
        continue;
      }
      const file = joinModelPath(modelDirectory, value.File);
      const id = stableId("expression", value.Name, file);
      expressionAssets[id] = {
        id,
        name: assetName(file, ".exp3.json"),
        file,
        expression: value.Name,
        contentHash: await contentHash(root, file, value),
      };
    }
  }

  return { motionAssets, expressionAssets };
}

function joinModelPath(directory: string, relative: string): string {
  const normalized = relative.replaceAll("\\", "/");
  if (
    !normalized ||
    normalized.startsWith("/") ||
    /^[a-z]+:/i.test(normalized) ||
    normalized.split("/").some((segment) => segment === ".." || segment === "")
  ) {
    throw new Error(`模型包含不安全路径：${relative}`);
  }
  const joined = path.posix.normalize(path.posix.join(directory, normalized));
  if (joined === ".." || joined.startsWith("../")) {
    throw new Error(`模型包含不安全路径：${relative}`);
  }
  return joined;
}

async function contentHash(
  root: string,
  file: string,
  descriptor: Record<string, unknown>,
): Promise<string> {
  return createHash("sha256")
    .update(await readFile(path.join(root, ...file.split("/"))))
    .update("\0")
    .update(JSON.stringify(descriptor))
    .digest("hex");
}

function stableId(type: string, ...parts: string[]): string {
  const suffix = createHash("sha256")
    .update([type, ...parts].join("\0"))
    .digest("hex")
    .slice(0, 16);
  return `${type}-${suffix}`;
}

function assetName(file: string, suffix: string): string {
  const basename = path.posix.basename(file);
  return basename.toLowerCase().endsWith(suffix)
    ? basename.slice(0, -suffix.length)
    : basename;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
