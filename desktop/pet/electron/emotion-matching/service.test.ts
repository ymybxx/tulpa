import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { packageFixture } from "../../src/model-manager/test-fixture";
import type { EmotionAssessment, EmotionAsset, EmotionMatchingJob, EmotionSample } from "../../src/shared/emotion-matching";
import { EmotionMatchingService } from "./service";
import { EmotionMatchingStore } from "./store";
import { EmotionAnalysisError } from "./client";
import { PET_BEHAVIOR_STATES } from "../../src/shared/contracts";
import { targetCandidates, targetId } from "../../src/shared/behavior-target";

const directories: string[] = [];
const services: EmotionMatchingService[] = [];
afterEach(async () => {
  await Promise.all(services.splice(0).map((service) => service.shutdown()));
  await Promise.all(directories.splice(0).map((p) => rm(p, { recursive: true, force: true })));
  vi.restoreAllMocks();
});
async function fixture() {
  const directory = await mkdtemp(path.join(os.tmpdir(), "emotion-service-test-")); directories.push(directory);
  const model = packageFixture();
  const asset: EmotionAsset = { id: "smile", name: "smile", kind: "expression", contentHash: "b", durationMs: 1200, timestampsMs: [1200], parameters: [], candidate: { id: "smile", motionAssetIds: [], expressionAssetIds: ["smile"] } };
  const source = { bundle: { model, appearance: { modelId: model.id, streamId: "test", revision: 0, patchSets: [], atlases: [] } }, fingerprint: "source", assets: [asset] };
  const updates: EmotionMatchingJob[] = [];
  const sample = vi.fn(async (input: EmotionAsset, mode: EmotionSample["mode"]): Promise<EmotionSample> => ({ asset: input, mode, frames: [{ id: `${input.id}:1`, timeMs: 1200, image: "data:image/jpeg;base64,AA==" }], faceExpressionIds: ["smile"], elapsedMs: 1 }));
  const sampler = { load: vi.fn(async () => undefined), sample, dispose: vi.fn(async () => undefined) };
  const analyze = vi.fn(async (samples: EmotionSample[]): Promise<EmotionAssessment[]> => samples.map((s) => ({ assetId: s.asset.id, classification: "behavior", matches: [{ target: { kind: "emotion", key: "happy" }, strength: "strong" }], evidence: [{ frameId: s.frames[0]!.id, observation: "嘴角上扬" }], reason: "微笑", needsMoreFrames: false })));
  const route = { id: "vision", name: "Vision", model: "test-vision", baseUrl: "https://example.com/v1", apiKey: "test-not-a-key" };
  const service = new EmotionMatchingService(new EmotionMatchingStore(directory), {
    source: async () => source, route: async () => route, sampler: () => sampler,
    client: () => ({ analyze }), updated: (job) => updates.push(job),
  });
  services.push(service);
  const done = () => vi.waitFor(async () => expect((await service.get(model.id))?.phase).toBe("finished"), { timeout: 5000 });
  return { service, source, model, updates, sampler, analyze, done, route };
}

describe("background emotion matching", () => {
  it("shares one classification across all states and emotions, validating each playback lifecycle once", async () => {
    const f = await fixture();
    const original = f.analyze.getMockImplementation()!;
    f.analyze.mockImplementation(async (samples) => (await original(samples)).map((result) => ({ ...result,
      matches: [...result.matches, ...PET_BEHAVIOR_STATES.map((key) => ({ target: { kind: "state" as const, key }, strength: "strong" as const }))],
    })));
    const before = structuredClone(f.model.manifest);
    await f.service.start("girl"); await f.done();
    const job = (await f.service.get("girl"))!;
    expect(job.status).toBe("completed");
    expect(job.suggestions.map((s) => targetId(s.target)).sort()).toEqual(["emotion:happy", ...PET_BEHAVIOR_STATES.map((key) => `state:${key}`)].sort());
    expect(f.sampler.sample.mock.calls.map((call) => call[1])).toEqual(["asset", "runtime", "speaking", "state"]);
    expect(f.analyze).toHaveBeenCalledTimes(4);
    expect(f.model.manifest).toEqual(before);
    const manifest = structuredClone(before);
    for (const suggestion of job.suggestions) targetCandidates(manifest, suggestion.target).push(suggestion.candidate);
    const refs = job.suggestions.map((s) => ({ jobId: job.id, suggestionId: s.id, candidateId: s.candidate.id }));
    await expect(f.service.validate("girl", manifest, refs)).resolves.toBeUndefined();
    const moved = structuredClone(manifest);
    const state = moved.behaviors.states.listening.pop()!;
    moved.behaviors.states.thinking.push(state);
    await expect(f.service.validate("girl", moved, refs)).rejects.toThrow("目标已修改");
    moved.behaviors.states.thinking.pop();
    await expect(f.service.validate("girl", moved, refs)).resolves.toBeUndefined();
    await f.service.adopted("girl", refs, manifest);
    expect((await f.service.get("girl"))?.suggestions.every((s) => s.adopted)).toBe(true);
    f.analyze.mockClear(); f.sampler.sample.mockClear();
    await f.service.start("girl"); await f.done();
    expect(f.analyze).not.toHaveBeenCalled(); expect(f.sampler.sample).not.toHaveBeenCalled();
  });

  it("does not run transient or speaking validation for state-only candidates", async () => {
    const f = await fixture(); const original = f.analyze.getMockImplementation()!;
    f.analyze.mockImplementation(async (samples) => (await original(samples)).map((result) => ({ ...result,
      matches: [{ target: { kind: "state", key: "thinking" }, strength: "strong" }],
    })));
    await f.service.start("girl"); await f.done();
    expect(f.sampler.sample.mock.calls.map((call) => call[1])).toEqual(["asset", "state"]);
    expect(f.sampler.sample.mock.calls[1]![0].timestampsMs.at(-1)).toBeGreaterThan(2000);
    expect((await f.service.get("girl"))?.suggestions[0]?.target).toEqual({ kind: "state", key: "thinking" });
  });

  it("omits state suggestions rejected by actual state playback validation", async () => {
    const f = await fixture(); const original = f.analyze.getMockImplementation()!;
    f.analyze.mockImplementation(async (samples) => (await original(samples)).map((result) => samples[0]!.mode === "state"
      ? { ...result, classification: "utility", matches: [] }
      : { ...result, matches: [{ target: { kind: "state", key: "standby" }, strength: "strong" }] }));
    await f.service.start("girl"); await f.done();
    expect((await f.service.get("girl"))?.suggestions).toEqual([]);
  });
  it("does not strand a queued job after a failed initial write", async () => {
    const f = await fixture();
    vi.spyOn(EmotionMatchingStore.prototype, "save").mockRejectedValueOnce(new Error("disk full"));
    await expect(f.service.start("girl")).rejects.toThrow("disk full");
    expect(f.sampler.load).not.toHaveBeenCalled();
    await f.service.start("girl"); await f.done();
    expect((await f.service.get("girl"))?.status).toBe("completed");
  });
  it("deduplicates starts, produces reviewed suggestions without changing the manifest, and reuses results without GPU or AI work", async () => {
    const f = await fixture(); const original = structuredClone(f.model.manifest);
    const [a, b] = await Promise.all([f.service.start("girl"), f.service.start("girl")]);
    expect(a.id).toBe(b.id);
    await f.done();
    const job = await f.service.get("girl");
    expect(job?.status).toBe("completed"); expect(job?.suggestions).toHaveLength(1);
    expect(f.model.manifest).toEqual(original); expect(f.sampler.dispose).toHaveBeenCalledOnce();
    f.analyze.mockClear(); f.sampler.load.mockClear(); f.sampler.sample.mockClear();
    await f.service.start("girl"); await f.done();
    expect(f.analyze).not.toHaveBeenCalled(); expect(f.sampler.load).not.toHaveBeenCalled();
    expect((await f.service.get("girl"))?.cachedAssets).toBe(1);
    await f.service.shutdown();
  });

  it("rejects stale appearance, changed ordered candidates and draft lipsync at adoption", async () => {
    const f = await fixture(); await f.service.start("girl"); await f.done();
    const job = (await f.service.get("girl"))!; const suggestion = job.suggestions[0]!;
    const manifest = structuredClone(f.model.manifest); manifest.behaviors.emotions.happy.push(suggestion.candidate);
    const refs = [{ jobId: job.id, suggestionId: suggestion.id, candidateId: suggestion.candidate.id }];
    await expect(f.service.validate("girl", manifest, refs)).resolves.toBeUndefined();
    manifest.lipsync.maxOpen = 0.9;
    await expect(f.service.validate("girl", manifest, refs)).rejects.toThrow("口型");
    manifest.lipsync = suggestion.lipsync;
    manifest.defaultAppearance = { expressionAssetIds: ["smile"] };
    await expect(f.service.validate("girl", manifest, refs)).rejects.toThrow("默认外观");
    manifest.defaultAppearance = { expressionAssetIds: [], parameterValues: { ParamAngleX: 12 } };
    await expect(f.service.validate("girl", manifest, refs)).rejects.toThrow("默认外观");
    delete manifest.defaultAppearance;
    f.source.fingerprint = "changed";
    await expect(f.service.validate("girl", manifest, refs)).rejects.toThrow("外观");
  });

  it("ignores a cancelled provider response and keeps a newer task authoritative", async () => {
    const f = await fixture();
    let resolve!: (value: EmotionAssessment[]) => void;
    f.analyze.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    const old = await f.service.start("girl"); await vi.waitFor(() => expect(resolve).toBeTypeOf("function"));
    await f.service.cancel(old.id);
    const current = await f.service.start("girl");
    // The replacement must complete even if the old provider ignores abort.
    await f.done();
    resolve([]);
    await Promise.resolve();
    expect((await f.service.get("girl"))?.id).toBe(current.id);
    expect((await f.service.get("girl"))?.suggestions).toHaveLength(1);
    const newIndex = f.updates.findIndex((job) => job.id === current.id);
    expect(f.updates.slice(newIndex).every((job) => job.id === current.id)).toBe(true);
  });

  it("keeps a new run authoritative when an older disk read finishes late", async () => {
    const f = await fixture();
    let finishRead!: (job: EmotionMatchingJob | null) => void;
    vi.spyOn(EmotionMatchingStore.prototype, "job").mockImplementationOnce(() => new Promise((resolve) => { finishRead = resolve; }));
    const staleRead = f.service.get("girl");
    const started = await f.service.start("girl");
    finishRead({ ...started, id: "old-job", status: "interrupted", revision: 99 });
    expect((await staleRead)?.id).toBe(started.id);
    await f.done();
    expect((await f.service.get("girl"))?.status).toBe("completed");
  });

  it("publishes the actual validation stage while the provider is still pending", async () => {
    const f = await fixture();
    const original = f.analyze.getMockImplementation()!;
    let finish!: (results: EmotionAssessment[]) => void;
    f.analyze.mockImplementation(async (samples) => samples[0]!.mode === "runtime"
      ? new Promise((resolve) => { finish = resolve; }) : original(samples));
    const job = await f.service.start("girl");
    await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
    const current = await f.service.get("girl");
    expect(current).toMatchObject({ phase: "validating", totalAssets: 1 });
    expect(current?.message).toContain("正在验证实际播放：smile");
    expect(f.updates.some((job) => job.totalAssets === 1 && job.phase === "preparing")).toBe(true);
    await f.service.cancel(job.id);
    await f.done();
    finish([]);
  });

  it("does not mark successful batch neighbours failed when one refinement fails", async () => {
    const f = await fixture();
    for (const id of ["good", "refine", "last"]) f.source.assets.push({ ...f.source.assets[0]!, id, contentHash: id, name: id });
    f.analyze.mockImplementation(async (samples) => {
      if (samples.length === 1 && samples[0]!.asset.id === "refine") throw new EmotionAnalysisError("补充请求失败 (503)");
      return samples.map((sample) => ({ assetId: sample.asset.id, classification: "uncertain", matches: [], evidence: [], reason: "不确定", needsMoreFrames: sample.asset.id === "refine" }));
    });
    await f.service.start("girl"); await f.done();
    const job = await f.service.get("girl");
    expect(job?.processedAssets).toBe(4);
    expect(job?.failures).toEqual([{ assetId: "refine", message: "补充请求失败 (503)" }]);
  });

  it("keeps other assets usable after a corrupt asset and stops on provider configuration errors", async () => {
    const f = await fixture();
    f.source.assets.push({ ...f.source.assets[0]!, id: "broken", error: "Invalid motion JSON" });
    await f.service.start("girl"); await f.done();
    expect((await f.service.get("girl"))?.status).toBe("partial");
    expect((await f.service.get("girl"))?.suggestions).toHaveLength(1);
    f.route.model = "unsupported";
    f.analyze.mockRejectedValue(new EmotionAnalysisError("不支持图片", true));
    await f.service.start("girl"); await f.done();
    expect((await f.service.get("girl"))?.status).toBe("failed");
    expect((await f.service.get("girl"))?.message).toContain("不支持图片");
  });

  it("stops on a broken core model instead of repeatedly loading it for every asset", async () => {
    const f = await fixture();
    f.sampler.load.mockRejectedValue(new Error("moc invalid"));
    await f.service.start("girl"); await f.done();
    expect(f.sampler.load).toHaveBeenCalledOnce(); expect(f.analyze).not.toHaveBeenCalled();
    expect((await f.service.get("girl"))?.status).toBe("failed");
    expect(f.sampler.dispose).toHaveBeenCalledOnce();
  });

  it("cancels the request immediately when the target appearance changes", async () => {
    const f = await fixture();
    let finish!: (results: EmotionAssessment[]) => void;
    f.analyze.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    await f.service.start("girl"); await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
    const signal = (f.analyze.mock.calls[0] as unknown as [unknown, AbortSignal])[1];
    f.service.sourceChanged("girl");
    expect(signal.aborted).toBe(true); finish([]);
    await f.done();
    expect((await f.service.get("girl"))?.status).toBe("stale");
    expect((await f.service.get("girl"))?.suggestions).toEqual([]);
  });
});
