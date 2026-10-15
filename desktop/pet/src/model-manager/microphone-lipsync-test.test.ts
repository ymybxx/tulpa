import { afterEach, describe, expect, it, vi } from "vitest";

import { floatSamplesToPcm16, MicrophoneLipSyncTest } from "./microphone-lipsync-test";
afterEach(() => vi.unstubAllGlobals());

it("releases a late microphone grant after switching to a different preview test", async () => {
  let grant!: (value: MediaStream) => void;
  const track = { stop: vi.fn() };
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: () => new Promise<MediaStream>(resolve => { grant = resolve; }) } });
  const context = vi.fn(); vi.stubGlobal("AudioContext", context);
  const microphone = new MicrophoneLipSyncTest(vi.fn());
  const starting = microphone.start();
  await microphone.stop();
  grant({ getTracks: () => [track] } as unknown as MediaStream);
  await starting;
  expect(track.stop).toHaveBeenCalledOnce();
  expect(context).not.toHaveBeenCalled();
});

describe("floatSamplesToPcm16", () => {
  it("converts and clamps microphone samples for the shared lip-sync worklet", () => {
    expect([...new Int16Array(floatSamplesToPcm16(
      Float32Array.from([-2, -1, -0.5, 0, 0.5, 1, 2]),
    ))]).toEqual([-32768, -32768, -16384, 0, 16384, 32767, 32767]);
  });
});
