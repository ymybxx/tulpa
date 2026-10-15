import { afterEach, describe, expect, it, vi } from "vitest";
import { EmotionAnalysisClient, validateAssessments } from "./client";
import { describeAsset } from "./asset-analysis";
import type { EmotionSample } from "../../src/shared/emotion-matching";

const sample = { asset: { id: "a" }, frames: [{ id: "a:0" }] } as EmotionSample;
const valid = { assetId: "a", classification: "behavior", matches: [{ target: { kind: "emotion", key: "happy" }, strength: "strong" }], evidence: [{ frameId: "a:0", observation: "smile" }], reason: "smile", needsMoreFrames: false };
const route = { id: "vision", name: "Vision", model: "test", baseUrl: "https://example.com/v1", apiKey: "test-key" };
afterEach(() => vi.useRealTimers());

describe("emotion provider failures", () => {
  it("preserves the validation reason after the bounded retry", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn(async () => new Response(JSON.stringify({ choices: [{ message: { content: JSON.stringify({ batchId: "wrong", assets: [] }) } }] })));
    const client = new EmotionAnalysisClient(route, fetcher);
    const result = expect(client.analyze([sample], new AbortController().signal)).rejects.toThrow("批次身份或结构无效");
    await vi.advanceTimersByTimeAsync(1000);
    await result;
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("times out a fetch that never settles and does not wait forever after retry", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn(() => new Promise<Response>(() => undefined));
    const result = expect(new EmotionAnalysisClient(route, fetcher).analyze([sample], new AbortController().signal)).rejects.toThrow("图片分析超时");
    await vi.advanceTimersByTimeAsync(181_000);
    await result;
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("cancels immediately while the response body is stalled", async () => {
    const cancelled = vi.fn();
    const stream = new ReadableStream({ cancel: cancelled });
    const fetcher = vi.fn(async () => new Response(stream));
    const controller = new AbortController();
    const result = expect(new EmotionAnalysisClient(route, fetcher).analyze([sample], controller.signal)).rejects.toThrow();
    await vi.waitFor(() => expect(stream.locked).toBe(true));
    controller.abort();
    await result;
    expect(cancelled).toHaveBeenCalledOnce();
    expect(fetcher).toHaveBeenCalledOnce();
  });
});
describe("emotion evidence and sampling plans", () => {
  it("accepts state and emotion evidence together but rejects unknown or repeated targets", () => {
    const state = { target: { kind: "state", key: "listening" }, strength: "strong" };
    const result = { ...valid, matches: [...valid.matches, state] };
    expect(validateAssessments({ batchId: "batch", assets: [result] }, [sample], "batch")).toEqual([result]);
    for (const target of [{ kind: "state", key: "speaking" }, { kind: "state", key: "unbound" }, { kind: "state", key: "happy" }, { kind: "emotion", key: "thinking" }]) {
      expect(validateAssessments({ batchId: "batch", assets: [{ ...result, matches: [{ ...state, target }] }] }, [sample], "batch")).toEqual([]);
    }
    expect(validateAssessments({ batchId: "batch", assets: [{ ...result, matches: [state, state] }] }, [sample], "batch")).toEqual([]);
  });
  it("rejects cross-batch, unknown and repeated identities", () => {
    expect(() => validateAssessments({ batchId: "old", assets: [valid] }, [sample], "batch")).toThrow();
    expect(() => validateAssessments({ batchId: "batch", assets: [{ ...valid, assetId: "invented" }] }, [sample], "batch")).toThrow();
    expect(() => validateAssessments({ batchId: "batch", assets: [valid, valid] }, [sample], "batch")).toThrow();
  });
  it("does not turn missing, invalid or unsupported evidence into a neutral match", () => {
    expect(validateAssessments({ batchId: "batch", assets: [] }, [sample], "batch")).toEqual([]);
    expect(validateAssessments({ batchId: "batch", assets: [{ ...valid, evidence: [{ frameId: "other:0", observation: "happy" }] }] }, [sample], "batch")).toEqual([]);
    expect(validateAssessments({ batchId: "batch", assets: [{ ...valid, classification: "utility" }] }, [sample], "batch")).toEqual([]);
    expect(validateAssessments({ batchId: "batch", assets: [valid] }, [sample], "batch")).toEqual([valid]);
  });
  it("covers an entire long motion and expression fade instead of taking only the start", () => {
    const motion = describeAsset("motion", { Meta: { Duration: 20 }, Curves: [{ Target: "Parameter", Id: "Eye", Segments: [0, 0, 0, 17, 1, 0, 20, 0] }] });
    expect(motion.timestampsMs.at(-1)).toBeGreaterThan(19_900);
    expect(motion.timestampsMs).toContain(17_000);
    expect(motion.timestampsMs.length).toBeLessThanOrEqual(7);
    expect(describeAsset("expression", { FadeInTime: 4 }).timestampsMs[0]).toBeGreaterThan(4000);
    expect(() => describeAsset("motion", { Meta: { Duration: Infinity } })).toThrow();
  });
});
