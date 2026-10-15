import { access, cp, lstat, mkdir, readFile, readdir, realpath, rename, rm } from "node:fs/promises";
import path from "node:path";

import { PNG } from "pngjs";

import {
  validateMasterTemplateManifest,
  type MasterTemplateManifest,
  type MasterTemplateSlot,
} from "../../src/shared/character-generation";

const MANIFEST_FILE = "master.template.json";

export interface MasterTemplatePackage {
  id: string;
  name: string;
  root: string;
  manifest: MasterTemplateManifest;
}

export class MasterTemplateStore {
  readonly root: string;

  constructor(userDataPath: string) {
    this.root = path.join(userDataPath, "master-templates");
  }

  async list(): Promise<MasterTemplatePackage[]> {
    await mkdir(this.root, { recursive: true });
    const entries = await readdir(this.root, { withFileTypes: true });
    const values = await Promise.all(entries
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
      .map(async (entry) => {
        try {
          return await this.load(entry.name);
        } catch {
          return null;
        }
      }));
    return values.filter((value): value is MasterTemplatePackage => value !== null)
      .sort((left, right) => left.name.localeCompare(right.name));
  }

  async load(id: string): Promise<MasterTemplatePackage> {
    if (!/^[a-zA-Z0-9_-]+$/.test(id)) throw new Error("母版 ID 无效");
    const root = path.join(this.root, id);
    return this.readAndValidate(root);
  }

  async installFolder(source: string): Promise<MasterTemplatePackage> {
    const sourcePackage = await this.readAndValidate(source);
    await mkdir(this.root, { recursive: true });
    const target = path.join(this.root, sourcePackage.id);
    try {
      await access(target);
      throw new Error(`母版 ${sourcePackage.id} 已存在`);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    const staging = path.join(this.root, `.import-${sourcePackage.id}-${Date.now()}`);
    try {
      await cp(source, staging, { recursive: true, force: false, errorOnExist: true });
      await this.readAndValidate(staging);
      await rename(staging, target);
    } catch (error) {
      await rm(staging, { recursive: true, force: true });
      throw error;
    }
    return this.load(sourcePackage.id);
  }

  private async readAndValidate(root: string): Promise<MasterTemplatePackage> {
    const manifest = validateMasterTemplateManifest(
      JSON.parse(await readFile(path.join(root, MANIFEST_FILE), "utf8")) as unknown,
    );
    await requiredFile(root, manifest.modelEntry);
    await requiredFile(root, manifest.alignmentGuide);
    await validateAlignmentGuide(root, manifest);
    const modelJson = JSON.parse(await readFile(path.join(root, manifest.modelEntry), "utf8")) as unknown;
    await Promise.all(requiredRuntimeReferences(modelJson, manifest.modelEntry).map((file) =>
      requiredFile(root, file)
    ));
    await requiredFile(root, "pet.manifest.json");
    await Promise.all(manifest.textureAtlases.map(async (atlas) => {
      await requiredFile(root, atlas.file);
      await validatePngDimensions(root, atlas.file, atlas.width, atlas.height, "图集");
    }));
    await Promise.all(manifest.slots.map(async (slot) => {
      await requiredFile(root, slot.silhouetteMask);
      await validatePngDimensions(
        root,
        slot.silhouetteMask,
        slot.atlas.width,
        slot.atlas.height,
        `槽位 ${slot.id} 的蒙版`,
      );
    }));
    return { id: manifest.id, name: manifest.name, root, manifest };
  }
}

async function validateAlignmentGuide(root: string, manifest: MasterTemplateManifest): Promise<void> {
  const relative = manifest.alignmentGuide;
  let image: PNG;
  try {
    image = PNG.sync.read(await readFile(path.join(root, relative)));
  } catch {
    throw new Error(`母版对齐参考图不是有效 PNG: ${relative}`);
  }
  if (
    image.width < 512 || image.height < 512 || image.width > 2048 || image.height > 2048 ||
    image.width % 16 !== 0 || image.height % 16 !== 0
  ) {
    throw new Error("母版对齐参考图宽高必须为 512～2048 且是 16 的倍数");
  }
  const canvasRatio = manifest.canvas.width / manifest.canvas.height;
  const imageRatio = image.width / image.height;
  if (Math.abs(canvasRatio - imageRatio) / canvasRatio > 0.01) {
    throw new Error("母版对齐参考图宽高比必须与 Cubism 画布一致");
  }
}

async function validatePngDimensions(
  root: string,
  relative: string,
  width: number,
  height: number,
  label: string,
): Promise<void> {
  let image: PNG;
  try {
    image = PNG.sync.read(await readFile(path.join(root, relative)));
  } catch {
    throw new Error(`${label}不是有效 PNG: ${relative}`);
  }
  if (image.width !== width || image.height !== height) {
    throw new Error(`${label}尺寸应为 ${width}×${height}，实际为 ${image.width}×${image.height}`);
  }
}

function requiredRuntimeReferences(value: unknown, modelEntry: string): string[] {
  const source = isRecord(value) && isRecord(value.FileReferences)
    ? value.FileReferences
    : {};
  const missing = [
    typeof source.Moc === "string" && source.Moc ? null : "Moc",
    typeof source.Physics === "string" && source.Physics ? null : "Physics",
    Array.isArray(source.Textures) && source.Textures.length > 0 ? null : "Textures",
    isRecord(source.Motions) && Object.keys(source.Motions).length > 0 ? null : "Motions",
    Array.isArray(source.Expressions) && source.Expressions.length > 0 ? null : "Expressions",
  ].filter((item): item is string => item !== null);
  if (missing.length > 0) throw new Error(`正式母版缺少固定运行资源: ${missing.join(", ")}`);
  const files: string[] = [source.Moc as string, source.Physics as string];
  files.push(...(source.Textures as unknown[]).filter((item): item is string => typeof item === "string"));
  for (const group of Object.values(source.Motions as Record<string, unknown>)) {
    if (!Array.isArray(group)) continue;
    for (const motion of group) {
      if (isRecord(motion) && typeof motion.File === "string") files.push(motion.File);
    }
  }
  for (const expression of source.Expressions as unknown[]) {
    if (isRecord(expression) && typeof expression.File === "string") files.push(expression.File);
  }
  const directory = path.posix.dirname(modelEntry.replaceAll("\\", "/"));
  return files.map((file) => path.posix.join(directory, file.replaceAll("\\", "/")));
}

async function requiredFile(root: string, relative: string): Promise<void> {
  const resolved = path.resolve(root, relative);
  if (!resolved.startsWith(`${path.resolve(root)}${path.sep}`)) {
    throw new Error(`母版资源路径不安全: ${relative}`);
  }
  const info = await lstat(resolved).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") throw new Error(`母版缺少必要资源: ${relative}`);
    throw error;
  });
  if (info.isSymbolicLink()) throw new Error(`母版资源不能是符号链接: ${relative}`);
  if (!info.isFile()) throw new Error(`母版资源不是文件: ${relative}`);
  const [realRoot, realFile] = await Promise.all([realpath(root), realpath(resolved)]);
  if (!realFile.startsWith(`${realRoot}${path.sep}`)) {
    throw new Error(`母版资源越过安装目录: ${relative}`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
