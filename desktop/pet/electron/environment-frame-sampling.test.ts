import { describe, expect, it, vi } from "vitest";

import {
  AdaptiveEnvironmentFrameSampling,
  AUDIOVISUAL_FRAME_INTERVAL_MS,
  SCREEN_FRAME_INTERVAL_MS,
} from "./environment-frame-sampling";

const SPEECH = {
  active: true,
  reason: "speech_start",
  runId: "91b1f58b-c368-4216-b2be-b9967a0cdecd",
  speechGeneration: 7,
} as const;

describe("adaptive environment frame sampling", () => {
  it("keeps audiovisual sampling at 1s and forces a zero-second speech frame", () => {
    vi.useFakeTimers();
    const applied: Array<[number, boolean]> = [];
    const sampling = new AdaptiveEnvironmentFrameSampling({
      apply: (interval, force) => applied.push([interval, force]),
      speechTimeoutMs: 30_000,
    });

    sampling.setPolicy({
      operationEnabled: false,
      screenEnabled: true,
      audioEnabled: true,
    });
    sampling.handleSpeechActivity(SPEECH);
    sampling.handleSpeechActivity({ ...SPEECH, active: false, reason: "speech_end" });

    expect(AUDIOVISUAL_FRAME_INTERVAL_MS).toBe(1_000);
    expect(applied).toEqual([
      [SCREEN_FRAME_INTERVAL_MS, true],
      [AUDIOVISUAL_FRAME_INTERVAL_MS, false],
    ]);
    vi.useRealTimers();
  });

  it("forces an immediate frame at speech start even when screen-only stays 1s", () => {
    vi.useFakeTimers();
    const applied: Array<[number, boolean]> = [];
    const sampling = new AdaptiveEnvironmentFrameSampling({
      apply: (interval, force) => applied.push([interval, force]),
    });

    sampling.setPolicy({
      operationEnabled: false,
      screenEnabled: true,
      audioEnabled: false,
    });
    sampling.handleSpeechActivity(SPEECH);

    expect(applied).toEqual([[SCREEN_FRAME_INTERVAL_MS, true]]);
    sampling.dispose();
    vi.useRealTimers();
  });

  it("refreshes the burst deadline and restores baseline after timeout", () => {
    vi.useFakeTimers();
    const applied: Array<[number, boolean]> = [];
    const sampling = new AdaptiveEnvironmentFrameSampling({
      apply: (interval, force) => applied.push([interval, force]),
      speechTimeoutMs: 100,
    });
    sampling.setPolicy({
      operationEnabled: false,
      screenEnabled: true,
      audioEnabled: true,
    });
    sampling.handleSpeechActivity(SPEECH);
    vi.advanceTimersByTime(90);
    sampling.handleSpeechActivity({ ...SPEECH, speechGeneration: 8 });
    vi.advanceTimersByTime(99);
    expect(applied.at(-1)).toEqual([SCREEN_FRAME_INTERVAL_MS, true]);
    vi.advanceTimersByTime(1);
    expect(applied.at(-1)).toEqual([AUDIOVISUAL_FRAME_INTERVAL_MS, false]);
    vi.useRealTimers();
  });

  it("does not request screen samples for audio-only perception", () => {
    const apply = vi.fn();
    const sampling = new AdaptiveEnvironmentFrameSampling({ apply });
    sampling.setPolicy({ operationEnabled: false, screenEnabled: false, audioEnabled: true });
    sampling.handleSpeechActivity(SPEECH);
    expect(apply).not.toHaveBeenCalled();
    sampling.dispose();
  });

  it("restores the current baseline when Companion stops", () => {
    vi.useFakeTimers();
    const applied: Array<[number, boolean]> = [];
    const sampling = new AdaptiveEnvironmentFrameSampling({
      apply: (interval, force) => applied.push([interval, force]),
    });
    sampling.setPolicy({
      operationEnabled: false,
      screenEnabled: true,
      audioEnabled: true,
    });
    sampling.handleSpeechActivity(SPEECH);
    sampling.handleSpeechActivity({
      ...SPEECH,
      active: false,
      reason: "companion_stopped",
    });

    expect(applied.at(-1)).toEqual([AUDIOVISUAL_FRAME_INTERVAL_MS, false]);
    vi.useRealTimers();
  });
});
