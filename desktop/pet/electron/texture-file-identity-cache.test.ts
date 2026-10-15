import { mkdtemp, rm, utimes, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { PNG } from "pngjs";
import { afterEach, describe, expect, it } from "vitest";

import { TextureFileIdentityCache } from "./texture-file-identity-cache";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("TextureFileIdentityCache", () => {
  it("returns PNG dimensions and a byte-content SHA-256", async () => {
    const file = await pngFile(7, 11, [10, 20, 30, 255]);
    const identity = await new TextureFileIdentityCache().identify(file);

    expect(identity).toMatchObject({ width: 7, height: 11 });
    expect(identity.sha256).toMatch(/^[a-f0-9]{64}$/);
  });

  it("invalidates its stat cache after the file changes", async () => {
    const file = await pngFile(2, 3, [10, 20, 30, 255]);
    const cache = new TextureFileIdentityCache();
    const first = await cache.identify(file);
    await writeFile(file, PNG.sync.write(solidPng(2, 3, [30, 20, 10, 255])));
    const now = new Date(Date.now() + 2_000);
    await utimes(file, now, now);
    const second = await cache.identify(file);

    expect(second.sha256).not.toBe(first.sha256);
    expect(second).toMatchObject({ width: 2, height: 3 });
  });
});

async function pngFile(
  width: number,
  height: number,
  rgba: [number, number, number, number],
): Promise<string> {
  const root = await mkdtemp(path.join(os.tmpdir(), "texture-identity-"));
  roots.push(root);
  const file = path.join(root, "image.png");
  await writeFile(file, PNG.sync.write(solidPng(width, height, rgba)));
  return file;
}

function solidPng(
  width: number,
  height: number,
  rgba: [number, number, number, number],
): PNG {
  const png = new PNG({ width, height });
  for (let offset = 0; offset < png.data.length; offset += 4) png.data.set(rgba, offset);
  return png;
}
