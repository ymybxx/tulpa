import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, stat } from "node:fs/promises";

export interface TextureFileIdentity {
  width: number;
  height: number;
  sha256: string;
}

interface CacheEntry {
  size: number;
  mtimeMs: number;
  value: Promise<TextureFileIdentity>;
}

export class TextureFileIdentityCache {
  private readonly entries = new Map<string, CacheEntry>();

  async identify(file: string): Promise<TextureFileIdentity> {
    const info = await stat(file);
    const cached = this.entries.get(file);
    if (cached && cached.size === info.size && cached.mtimeMs === info.mtimeMs) {
      return cached.value;
    }
    const value = identifyPng(file).catch((error) => {
      if (this.entries.get(file)?.value === value) this.entries.delete(file);
      throw error;
    });
    this.entries.set(file, { size: info.size, mtimeMs: info.mtimeMs, value });
    return value;
  }
}

async function identifyPng(file: string): Promise<TextureFileIdentity> {
  const handle = await open(file, "r");
  let width = 0;
  let height = 0;
  try {
    const header = Buffer.allocUnsafe(24);
    const { bytesRead } = await handle.read(header, 0, header.length, 0);
    if (
      bytesRead !== header.length ||
      header.readUInt32BE(0) !== 0x89504e47 ||
      header.readUInt32BE(4) !== 0x0d0a1a0a ||
      header.toString("ascii", 12, 16) !== "IHDR"
    ) throw new Error(`纹理不是有效 PNG：${file}`);
    width = header.readUInt32BE(16);
    height = header.readUInt32BE(20);
    if (width <= 0 || height <= 0) throw new Error(`纹理 PNG 尺寸无效：${file}`);
  } finally {
    await handle.close();
  }

  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk as Buffer);
  return { width, height, sha256: hash.digest("hex") };
}
