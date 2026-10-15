import { mkdir, mkdtemp, rm, stat, utimes, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it, onTestFinished } from "vitest";

import { pruneBoundedFileCache } from "./bounded-file-cache";

describe("pruneBoundedFileCache", () => {
  it("deletes only the least-recent inactive entries until the cache is bounded", async () => {
    const root = await createTemporaryDirectory("bounded-cache-");
    await cacheEntry(root, "old", 6, 1_000);
    await cacheEntry(root, "middle", 6, 2_000);
    await cacheEntry(root, "new", 6, 3_000);

    const result = await pruneBoundedFileCache({
      root,
      maximumBytes: 12,
      protectedKeys: new Set(),
    });

    expect(result.removedKeys).toEqual(["old"]);
    expect(result.beforeBytes).toBe(18);
    expect(result.afterBytes).toBe(12);
    await expect(stat(path.join(root, "old"))).rejects.toMatchObject({ code: "ENOENT" });
    await expect(stat(path.join(root, "middle", "atlas.png"))).resolves.toBeDefined();
    await expect(stat(path.join(root, "new", "atlas.png"))).resolves.toBeDefined();
  });

  it("protects current and in-flight keys even when they exceed the byte limit", async () => {
    const root = await createTemporaryDirectory("bounded-cache-protected-");
    await cacheEntry(root, "current", 9, 1_000);
    await cacheEntry(root, "in-flight", 9, 2_000);
    await cacheEntry(root, "inactive", 9, 3_000);

    const result = await pruneBoundedFileCache({
      root,
      maximumBytes: 5,
      protectedKeys: new Set(["current", "in-flight"]),
    });

    expect(result.removedKeys).toEqual(["inactive"]);
    expect(result.afterBytes).toBe(18);
    expect(result.limitSatisfied).toBe(false);
    await expect(stat(path.join(root, "current", "atlas.png"))).resolves.toBeDefined();
    await expect(stat(path.join(root, "in-flight", "atlas.png"))).resolves.toBeDefined();
  });

  it("can retain two recent entries without touching their mtimes", async () => {
    const root = await createTemporaryDirectory("bounded-cache-recent-");
    await cacheEntry(root, "old", 5, 1_000);
    await cacheEntry(root, "recent-1", 5, 2_000);
    await cacheEntry(root, "recent-2", 5, 3_000);
    const recentFile = path.join(root, "recent-2", "atlas.png");
    const before = await stat(recentFile);

    const result = await pruneBoundedFileCache({
      root,
      maximumBytes: 5,
      protectedKeys: new Set(),
      preserveNewest: 2,
    });

    expect(result.removedKeys).toEqual(["old"]);
    expect(result.afterBytes).toBe(10);
    expect(result.limitSatisfied).toBe(false);
    expect((await stat(recentFile)).mtimeMs).toBe(before.mtimeMs);
  });

  it("supports file caches whose content key is derived from a nested file name", async () => {
    const root = await createTemporaryDirectory("bounded-file-cache-files-");
    await mkdir(path.join(root, "2048"), { recursive: true });
    await writeFile(path.join(root, "2048", "old.png"), Buffer.alloc(7));
    await writeFile(path.join(root, "2048", "keep.png"), Buffer.alloc(7));
    await setFileTime(path.join(root, "2048", "old.png"), 1_000);
    await setFileTime(path.join(root, "2048", "keep.png"), 2_000);

    const result = await pruneBoundedFileCache({
      root,
      maximumBytes: 7,
      protectedKeys: new Set(["keep"]),
      keyForFile: (relative) => path.basename(relative, path.extname(relative)),
    });

    expect(result.removedKeys).toEqual(["old"]);
    await expect(stat(path.join(root, "2048", "old.png"))).rejects.toMatchObject({ code: "ENOENT" });
    await expect(stat(path.join(root, "2048", "keep.png"))).resolves.toBeDefined();
  });
});

async function cacheEntry(root: string, key: string, bytes: number, timeMs: number): Promise<void> {
  const file = path.join(root, key, "atlas.png");
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, Buffer.alloc(bytes));
  await setFileTime(file, timeMs);
  await setFileTime(path.dirname(file), timeMs);
}

async function setFileTime(file: string, timeMs: number): Promise<void> {
  const date = new Date(timeMs);
  await utimes(file, date, date);
}

async function createTemporaryDirectory(prefix: string): Promise<string> {
  const directory = await mkdtemp(path.join(os.tmpdir(), prefix));
  onTestFinished(() => rm(directory, { recursive: true, force: true }));
  return directory;
}
