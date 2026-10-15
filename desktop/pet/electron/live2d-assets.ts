import { createHash } from "node:crypto";
import { access, copyFile, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import JSZip from "jszip";

import { net } from "electron";

import {
  BUNDLED_CORE, BUNDLED_MODEL_DIR, CORE_ARCHIVE_ENTRY, CUBISM_SDK_ARCHIVE_SHA256, CUBISM_SDK_ARCHIVE_URL,
  CUBISM_SDK_VERSION, MODEL_FILES, SAMPLE_ROOT, isCompatibleR5CoreSource,
} from "./live2d-sources";

export { BUNDLED_LIVE2D_DIR, isCompatibleR5CoreSource } from "./live2d-sources";

export class Live2dAssetInstaller {
  readonly root: string;
  readonly corePath: string;
  readonly modelRoot: string;

  /** `bundledRoot`: the app's own copy (Core + Mao); downloads are only the fallback. */
  constructor(userDataPath: string, private readonly bundledRoot: string | null = null) {
    this.root = path.join(userDataPath, "live2d");
    this.corePath = path.join(this.root, "runtime", "live2dcubismcore.min.js");
    this.modelRoot = path.join(this.root, "default-model", "Mao");
  }

  async ready(): Promise<boolean> {
    try {
      await this.validateCore();
      await this.validateModel();
      return true;
    } catch {
      return false;
    }
  }

  async install(report: (detail: string) => void): Promise<void> {
    await mkdir(this.root, { recursive: true });
    await mkdir(path.dirname(this.corePath), { recursive: true });
    if (!(await this.validCore()) && !(await this.copyBundledCore())) {
      report("正在下载形象运行库…");
      await this.installR5Core();
      await this.validateCore();
    }
    let completed = 0;
    await mapConcurrent(MODEL_FILES, 4, async (relative) => {
      const target = path.join(this.modelRoot, ...relative.split("/"));
      if (!(await fileExists(target)) && !(await this.copyBundled(path.join(BUNDLED_MODEL_DIR, relative), target))) {
        await downloadWithRetry(`${SAMPLE_ROOT}/${relative}`, target);
      }
      completed += 1;
      report(`正在准备默认形象 (${completed}/${MODEL_FILES.length})…`);
    });
    await this.validateModel();
    await rm(path.join(this.modelRoot, "pet.manifest.json"), { force: true });
  }

  resolve(relative: string): string {
    const root = path.resolve(this.root);
    const target = path.resolve(root, relative);
    if (target !== root && !target.startsWith(`${root}${path.sep}`)) {
      throw new Error("Invalid Live2D asset path");
    }
    return target;
  }

  /** Copies one file out of the app bundle; false when the bundle does not have it. */
  private async copyBundled(relative: string, target: string): Promise<boolean> {
    if (!this.bundledRoot) return false;
    const source = path.join(this.bundledRoot, ...relative.split(/[\\/]/u));
    if (!(await fileExists(source))) return false;
    await mkdir(path.dirname(target), { recursive: true });
    const temporary = `${target}.bundled`;
    await copyFile(source, temporary);
    await rename(temporary, target);
    return true;
  }

  private async copyBundledCore(): Promise<boolean> {
    if (!(await this.copyBundled(BUNDLED_CORE, this.corePath))) return false;
    if (await this.validCore()) return true;
    await rm(this.corePath, { force: true });
    return false;
  }

  private async validCore(): Promise<boolean> {
    try {
      await this.validateCore();
      return true;
    } catch {
      return false;
    }
  }

  private async validateCore(): Promise<void> {
    const content = await readFile(this.corePath, "utf8");
    if (!isCompatibleR5CoreSource(content)) {
      throw new Error("Live2D Cubism Core 版本过旧或校验失败，需要 R5 / moc3 version 6");
    }
  }

  private async installR5Core(): Promise<void> {
    const archive = path.join(this.root, "downloads", `CubismSdkForWeb-${CUBISM_SDK_VERSION}.zip`);
    await downloadWithRetry(CUBISM_SDK_ARCHIVE_URL, archive);
    try {
      const bytes = await readFile(archive);
      const checksum = createHash("sha256").update(bytes).digest("hex");
      if (checksum !== CUBISM_SDK_ARCHIVE_SHA256) {
        throw new Error("Live2D Cubism SDK 下载校验失败");
      }
      const zip = await JSZip.loadAsync(bytes);
      const entry = zip.file(CORE_ARCHIVE_ENTRY);
      if (!entry) throw new Error("Live2D Cubism SDK 中缺少 Web Core");
      const core = await entry.async("nodebuffer");
      const temporary = `${this.corePath}.tmp`;
      await mkdir(path.dirname(this.corePath), { recursive: true });
      await writeFile(temporary, core);
      await rename(temporary, this.corePath);
    } finally {
      await rm(archive, { force: true });
    }
  }

  private async validateModel(): Promise<void> {
    const model = JSON.parse(
      await readFile(path.join(this.modelRoot, "Mao.model3.json"), "utf8"),
    ) as { FileReferences?: { Moc?: unknown; Textures?: unknown } };
    if (model.FileReferences?.Moc !== "Mao.moc3" || !Array.isArray(model.FileReferences.Textures)) {
      throw new Error("默认 Live2D 模型校验失败");
    }
    await Promise.all(MODEL_FILES.map((relative) => access(path.join(this.modelRoot, relative))));
  }
}


async function downloadWithRetry(url: string, target: string): Promise<void> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      await download(url, target);
      return;
    } catch (error) {
      lastError = error;
      if (attempt < 3) await new Promise((resolve) => setTimeout(resolve, 400 * attempt));
    }
  }
  throw lastError;
}

async function download(url: string, target: string): Promise<void> {
  await mkdir(path.dirname(target), { recursive: true });
  const temporary = `${target}.part`;
  const downloaded = await fileSize(temporary);
  const response = await net.fetch(url, {
    bypassCustomProtocolHandlers: true,
    headers: downloaded > 0 ? { Range: `bytes=${downloaded}-` } : undefined,
  });
  if (!response.ok) throw new Error(`下载失败 (${response.status}): ${url}`);
  const content = Buffer.from(await response.arrayBuffer());
  if (!content.length && response.status !== 206) throw new Error(`下载内容为空: ${url}`);
  if (response.status === 206 && downloaded > 0) {
    await writeFile(temporary, content, { flag: "a" });
  } else {
    await writeFile(temporary, content);
  }
  await rename(temporary, target);
}

async function fileSize(file: string): Promise<number> {
  try {
    return (await stat(file)).size;
  } catch {
    return 0;
  }
}

async function mapConcurrent<T>(
  values: T[],
  concurrency: number,
  operation: (value: T) => Promise<void>,
): Promise<void> {
  const queue = [...values];
  await Promise.all(
    Array.from({ length: concurrency }, async () => {
      while (queue.length) {
        const value = queue.shift();
        if (value !== undefined) await operation(value);
      }
    }),
  );
}

async function fileExists(file: string): Promise<boolean> {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}
