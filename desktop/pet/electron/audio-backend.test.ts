import { describe, expect, it } from "vitest";

import { platformAudioBackend } from "./audio-backend";

describe("platformAudioBackend", () => {
  it("uses LiveKit on Apple silicon and the WebRTC fallback elsewhere", () => {
    expect(platformAudioBackend("darwin", "arm64")).toBe("livekit");
    expect(platformAudioBackend("darwin", "x64")).toBe("sounddevice");
    expect(platformAudioBackend("win32", "x64")).toBe("sounddevice");
  });
});
