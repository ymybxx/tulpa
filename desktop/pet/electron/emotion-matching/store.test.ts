import { afterEach, describe, expect, it } from "vitest";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { EmotionMatchingStore } from "./store";
import type { EmotionMatchingJob } from "../../src/shared/emotion-matching";

const directories: string[] = [];
afterEach(async () => { await Promise.all(directories.splice(0).map((p) => rm(p, { recursive: true, force: true }))); });

describe("behavior matching persistence compatibility", () => {
  it("reads old emotion suggestions without rewriting the user's saved analysis", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "behavior-store-")); directories.push(directory);
    const file = path.join(directory, "girl.json");
    const legacy = { version: 1, job: { modelId: "girl", status: "completed", suggestions: [
      { id: "old", emotion: "happy", candidate: { id: "ai-old", motionAssetIds: [], expressionAssetIds: ["smile"] }, adopted: true },
    ] } };
    const bytes = JSON.stringify(legacy); await writeFile(file, bytes);
    const job = await new EmotionMatchingStore(directory).job("girl");
    expect(job?.suggestions[0]).toMatchObject({ id: "old", target: { kind: "emotion", key: "happy" }, adopted: true });
    expect(job?.suggestions[0]?.candidate.id).toBe("ai-old");
    expect(await readFile(file, "utf8")).toBe(bytes);
  });

  it("round-trips state suggestions and does not mistake emotion-only caches for state analysis", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "behavior-store-")); directories.push(directory);
    const store = new EmotionMatchingStore(directory);
    const job = { modelId: "girl", status: "completed", suggestions: [{ id: "new", target: { kind: "state", key: "standby" } }] } as EmotionMatchingJob;
    await store.save(job);
    expect(await store.job("girl")).toEqual(job);
    const cacheKey = "a".repeat(64);
    await store.cache(cacheKey, { assetId: "asset", classification: "uncertain", matches: [], evidence: [], reason: "", needsMoreFrames: false });
    const cacheFile = path.join(directory, "cache", `${cacheKey}.json`);
    const cached = JSON.parse(await readFile(cacheFile, "utf8"));
    expect(cached.version).toBe(2);
    cached.version = 1; await writeFile(cacheFile, JSON.stringify(cached));
    expect(await store.cached(cacheKey)).toBeNull();
  });
});
