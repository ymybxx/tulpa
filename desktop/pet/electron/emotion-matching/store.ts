import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";
import type { EmotionAssessment, EmotionMatchingJob } from "../../src/shared/emotion-matching";
import { isBehaviorTarget } from "../../src/shared/behavior-target";

/** Compact results only. Sample images and provider credentials never reach this store. */
export class EmotionMatchingStore {
  private tail = Promise.resolve();
  constructor(private readonly root: string) {}

  async job(modelId: string): Promise<EmotionMatchingJob | null> {
    try {
      const value = JSON.parse(await readFile(this.file(modelId), "utf8"));
      if (![1, 2].includes(value.version) || value.job?.modelId !== modelId || !Array.isArray(value.job.suggestions)) return null;
      // Keep existing emotion suggestions reviewable without re-analysis or writes.
      value.job.suggestions = value.job.suggestions.flatMap((suggestion: Record<string, unknown>) => {
        if (!suggestion || typeof suggestion !== "object") return [];
        if (isBehaviorTarget(suggestion.target)) return [suggestion];
        const target = { kind: "emotion", key: suggestion.emotion };
        if (value.version !== 1 || !isBehaviorTarget(target)) return [];
        const { emotion: _legacyEmotion, ...rest } = suggestion;
        return [{ ...rest, target }];
      });
      const job = value.job as EmotionMatchingJob;
      if (["running", "queued"].includes(job.status)) {
        job.status = "interrupted";
        job.message = "上次分析已中断，继续分析可复用已有结果";
      }
      return job;
    } catch { return null; }
  }

  save(job: EmotionMatchingJob): Promise<void> {
    const bytes = JSON.stringify({ version: 2, job });
    const pending = this.tail.catch(() => undefined).then(async () => {
      await mkdir(this.root, { recursive: true });
      await atomicWrite(this.file(job.modelId), bytes);
    });
    this.tail = pending;
    return pending;
  }

  async cached(key: string): Promise<EmotionAssessment | null> {
    try {
      const value = JSON.parse(await readFile(this.cacheFile(key), "utf8"));
      return value.version === 2 ? value.result as EmotionAssessment : null;
    } catch { return null; }
  }

  async cache(key: string, result: EmotionAssessment): Promise<void> {
    const directory = path.join(this.root, "cache");
    await mkdir(directory, { recursive: true });
    await atomicWrite(this.cacheFile(key), JSON.stringify({ version: 2, result }));
    // Small JSON files: cap both bytes and count, never keep full image evidence.
    const entries = await Promise.all((await readdir(directory)).filter((name) => /^[a-f0-9]{64}\.json$/.test(name)).map(async (name) => {
      const file = path.join(directory, name);
      const info = await stat(file);
      return { file, bytes: info.size, modified: info.mtimeMs };
    }));
    let size = entries.reduce((total, e) => total + e.bytes, 0);
    let count = entries.length;
    for (const entry of entries.sort((a, b) => a.modified - b.modified)) {
      if (size <= 32 * 1024 * 1024 && count <= 4000) break;
      await rm(entry.file, { force: true });
      size -= entry.bytes;
      count--;
    }
  }

  private file(id: string): string {
    if (!/^[a-z0-9][a-z0-9-]{0,79}$/.test(id)) throw new Error("模型 ID 无效");
    return path.join(this.root, `${id}.json`);
  }
  private cacheFile(key: string): string {
    if (!/^[a-f0-9]{64}$/.test(key)) throw new Error("分析缓存 ID 无效");
    return path.join(this.root, "cache", `${key}.json`);
  }
}

async function atomicWrite(file: string, bytes: string): Promise<void> {
  const temporary = `${file}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, bytes, { mode: 0o600 });
    await rename(temporary, file);
  } finally { await rm(temporary, { force: true }); }
}
