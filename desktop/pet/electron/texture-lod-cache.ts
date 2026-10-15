import { createHash, randomUUID } from "node:crypto";
import { access, mkdir, open, rename, stat, unlink, writeFile } from "node:fs/promises";
import path from "node:path";

import { nativeImage } from "electron";
import { PET_TEXTURE_DETAIL_LEVELS } from "../src/shared/texture-quality";
import { pruneBoundedFileCache } from "./bounded-file-cache";

const SUPPORTED_DIMENSIONS = new Set<number>(PET_TEXTURE_DETAIL_LEVELS);

/** Persistent, bounded-concurrency cache for desktop-pet atlas detail levels. */
export class TextureLodCache {
  private readonly pending = new Map<string, Promise<string>>();
  private readonly waiters: Array<() => void> = [];
  private active = 0;

  constructor(
    private readonly root: string,
    private readonly concurrency = 2,
    private readonly maximumBytes = 512 * 1024 * 1024,
  ) {}

  async resolve(source: string, maximumDimension: number): Promise<string> {
    if (!SUPPORTED_DIMENSIONS.has(maximumDimension)) return source;
    const dimensions = await pngDimensions(source);
    if (dimensions && Math.max(dimensions.width, dimensions.height) <= maximumDimension) return source;
    const info = await stat(source);
    const key = createHash("sha256")
      .update(source)
      .update("\0")
      .update(String(info.size))
      .update("\0")
      .update(String(info.mtimeMs))
      .update("\0")
      .update(String(maximumDimension))
      .digest("hex")
      .slice(0, 24);
    const target = path.join(this.root, String(maximumDimension), `${key}.png`);
    if (await exists(target)) return target;
    const existing = this.pending.get(key);
    if (existing) return existing;
    const task = this.generate(source, target, maximumDimension)
      .then((result) => {
        void this.prune(key).catch((error) => {
          console.warn("清理 Live2D 纹理 LOD 缓存失败", error);
        });
        return result;
      })
      .finally(() => this.pending.delete(key));
    this.pending.set(key, task);
    return task;
  }

  private async generate(
    source: string,
    target: string,
    maximumDimension: number,
  ): Promise<string> {
    await this.acquire();
    const temporary = `${target}.${randomUUID()}.tmp`;
    try {
      if (await exists(target)) return target;
      const image = nativeImage.createFromPath(source);
      if (image.isEmpty()) throw new Error(`无法读取 Live2D 纹理：${source}`);
      const size = image.getSize();
      if (Math.max(size.width, size.height) <= maximumDimension) return source;
      const scale = maximumDimension / Math.max(size.width, size.height);
      const resized = image.resize({
        width: Math.max(1, Math.round(size.width * scale)),
        height: Math.max(1, Math.round(size.height * scale)),
        quality: "best",
      });
      await mkdir(path.dirname(target), { recursive: true });
      await writeFile(temporary, resized.toPNG());
      await rename(temporary, target);
      return target;
    } finally {
      await unlink(temporary).catch(() => undefined);
      this.release();
    }
  }

  private async acquire(): Promise<void> {
    if (this.active < this.concurrency) {
      this.active += 1;
      return;
    }
    await new Promise<void>((resolve) => this.waiters.push(resolve));
    this.active += 1;
  }

  private release(): void {
    this.active = Math.max(0, this.active - 1);
    this.waiters.shift()?.();
  }

  private async prune(currentKey: string): Promise<void> {
    await pruneBoundedFileCache({
      root: this.root,
      maximumBytes: this.maximumBytes,
      protectedKeys: new Set([currentKey, ...this.pending.keys()]),
      preserveNewest: 2,
      keyForFile: (relative) => {
        const key = path.basename(relative, ".png");
        return /^[a-f0-9]{24}$/.test(key) ? key : null;
      },
    });
  }
}

export function requestedTextureDimension(value: string | null): number | null {
  if (!value) return null;
  const parsed = Number(value);
  return SUPPORTED_DIMENSIONS.has(parsed) ? parsed : null;
}

async function exists(file: string): Promise<boolean> {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}

async function pngDimensions(file: string): Promise<{ width: number; height: number } | null> {
  const handle = await open(file, "r");
  try {
    const header = Buffer.allocUnsafe(24);
    const { bytesRead } = await handle.read(header, 0, header.byteLength, 0);
    if (
      bytesRead < header.byteLength ||
      header.readUInt32BE(0) !== 0x89504e47 ||
      header.readUInt32BE(4) !== 0x0d0a1a0a ||
      header.toString("ascii", 12, 16) !== "IHDR"
    ) return null;
    return { width: header.readUInt32BE(16), height: header.readUInt32BE(20) };
  } finally {
    await handle.close();
  }
}
