import { describe, expect, it } from "vitest";

import { decodeAudioFrame } from "../src/shared/audio-protocol";

describe("decodeAudioFrame", () => {
  it("decodes generation, sample position, and PCM payload", () => {
    const value = new Uint8Array(29);
    value[0] = 1;
    value.set([
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1,
    ], 1);
    new DataView(value.buffer).setBigUint64(17, 640n, true);
    value.set([1, 0, 2, 0], 25);

    expect(decodeAudioFrame(value)).toEqual({
      generation: "00000000-0000-0000-0000-000000000001",
      samplePosition: 640,
      pcm: new Uint8Array([1, 0, 2, 0]).buffer,
    });
  });

  it("rejects incomplete frames", () => {
    expect(decodeAudioFrame(new Uint8Array(24))).toBeNull();
  });
});
