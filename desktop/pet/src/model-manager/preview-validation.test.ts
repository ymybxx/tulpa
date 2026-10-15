import { afterEach, describe, expect, it, vi } from "vitest";

import type { LipShape } from "../renderer/pet-renderer";
import { PreviewValidationSequence } from "./preview-validation";

afterEach(() => vi.useRealTimers());

describe("PreviewValidationSequence", () => {
  it("demonstrates gaze, all vowel mouth shapes, speaking state, and a motion", () => {
    vi.useFakeTimers();
    const calls: string[] = [];
    const sequence = new PreviewValidationSequence({
      setState: (value) => calls.push(`state:${value}`),
      setGaze: (x, y) => calls.push(`gaze:${x}:${y}`),
      setLipShape: (value: LipShape) => calls.push(vowel(value)),
      playFirstMotion: () => calls.push("motion"),
      reset: () => calls.push("reset"),
    });

    sequence.start();
    vi.runAllTimers();

    expect(calls).toContain("state:speaking");
    expect(calls).toContain("gaze:-0.35:0.08");
    expect(calls).toContain("gaze:0.35:-0.04");
    expect(calls).toEqual(expect.arrayContaining(["mouth:a", "mouth:e", "mouth:i", "mouth:o", "mouth:u"]));
    expect(calls).toContain("motion");
    expect(calls.at(-1)).toBe("reset");
  });

  it("cancels stale timers before starting a new preview", () => {
    vi.useFakeTimers();
    const reset = vi.fn();
    const sequence = new PreviewValidationSequence({
      setState: vi.fn(),
      setGaze: vi.fn(),
      setLipShape: vi.fn(),
      playFirstMotion: vi.fn(),
      reset,
    });

    sequence.start();
    sequence.cancel();
    vi.runAllTimers();

    expect(reset).not.toHaveBeenCalled();
  });
});

function vowel(value: LipShape): string {
  const values = { a: value.a, e: value.e, i: value.i, o: value.o, u: value.u };
  const selected = Object.entries(values).sort((left, right) => right[1] - left[1])[0];
  return selected && selected[1] > 0 ? `mouth:${selected[0]}` : "mouth:closed";
}
