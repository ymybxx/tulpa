import { lstat, readdir, rm, rmdir } from "node:fs/promises";
import path from "node:path";

export interface BoundedFileCacheOptions {
  root: string;
  maximumBytes: number;
  protectedKeys: ReadonlySet<string>;
  preserveNewest?: number;
  keyForFile?: (relativePath: string) => string | null;
}

export interface BoundedFileCacheResult {
  beforeBytes: number;
  afterBytes: number;
  removedKeys: string[];
  limitSatisfied: boolean;
}

interface CacheEntry {
  key: string;
  files: string[];
  bytes: number;
  lastUsedMs: number;
}

/**
 * Removes immutable cache entries in least-recently-used order without touching
 * retained files. Protected entries are never removed, even if they exceed the
 * configured limit.
 */
export async function pruneBoundedFileCache(
  options: BoundedFileCacheOptions,
): Promise<BoundedFileCacheResult> {
  if (!Number.isSafeInteger(options.maximumBytes) || options.maximumBytes < 0) {
    throw new Error("缓存容量必须是非负整数");
  }
  const preserveNewest = options.preserveNewest ?? 0;
  if (!Number.isSafeInteger(preserveNewest) || preserveNewest < 0) {
    throw new Error("保留条目数量必须是非负整数");
  }
  const keyForFile = options.keyForFile ?? defaultKeyForFile;
  const entries = await scanEntries(options.root, keyForFile);
  const beforeBytes = entries.reduce((sum, entry) => sum + entry.bytes, 0);
  let afterBytes = beforeBytes;
  const protectedKeys = new Set(options.protectedKeys);
  const newest = [...entries]
    .sort(compareNewestFirst)
    .slice(0, preserveNewest);
  for (const entry of newest) protectedKeys.add(entry.key);

  const removedKeys: string[] = [];
  const candidates = entries
    .filter((entry) => !protectedKeys.has(entry.key))
    .sort(compareOldestFirst);
  for (const entry of candidates) {
    if (afterBytes <= options.maximumBytes) break;
    await Promise.all(entry.files.map((file) => rm(file, { force: true })));
    await Promise.all(entry.files.map((file) => removeEmptyParents(path.dirname(file), options.root)));
    afterBytes = Math.max(0, afterBytes - entry.bytes);
    removedKeys.push(entry.key);
  }
  return {
    beforeBytes,
    afterBytes,
    removedKeys,
    limitSatisfied: afterBytes <= options.maximumBytes,
  };
}

async function scanEntries(
  root: string,
  keyForFile: (relativePath: string) => string | null,
): Promise<CacheEntry[]> {
  const byKey = new Map<string, CacheEntry>();
  for (const file of await walkFiles(root)) {
    const relative = path.relative(root, file.path);
    if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) continue;
    const key = keyForFile(relative.split(path.sep).join("/"));
    if (!key) continue;
    const existing = byKey.get(key) ?? {
      key,
      files: [],
      bytes: 0,
      lastUsedMs: 0,
    };
    existing.files.push(file.path);
    existing.bytes += file.bytes;
    existing.lastUsedMs = Math.max(existing.lastUsedMs, file.lastUsedMs);
    byKey.set(key, existing);
  }
  return [...byKey.values()];
}

async function walkFiles(root: string): Promise<Array<{
  path: string;
  bytes: number;
  lastUsedMs: number;
}>> {
  let children;
  try {
    children = await readdir(root, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
  const files = await Promise.all(children.map(async (child) => {
    const target = path.join(root, child.name);
    if (child.isDirectory()) return walkFiles(target);
    const info = await lstat(target).catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return null;
      throw error;
    });
    if (!info) return [];
    return [{
      path: target,
      bytes: info.size,
      lastUsedMs: Math.max(info.atimeMs, info.mtimeMs),
    }];
  }));
  return files.flat();
}

async function removeEmptyParents(directory: string, root: string): Promise<void> {
  const resolvedRoot = path.resolve(root);
  let current = path.resolve(directory);
  while (current !== resolvedRoot && isWithin(resolvedRoot, current)) {
    try {
      await rmdir(current);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code === "ENOENT") {
        // Continue to a potentially empty parent.
      } else if (code === "ENOTEMPTY" || code === "EEXIST") {
        return;
      } else {
        throw error;
      }
    }
    current = path.dirname(current);
  }
}

function defaultKeyForFile(relativePath: string): string | null {
  return relativePath.split("/")[0] || null;
}

function compareOldestFirst(left: CacheEntry, right: CacheEntry): number {
  return left.lastUsedMs - right.lastUsedMs || left.key.localeCompare(right.key);
}

function compareNewestFirst(left: CacheEntry, right: CacheEntry): number {
  return right.lastUsedMs - left.lastUsedMs || left.key.localeCompare(right.key);
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}
