import type { DesktopAudioBackend } from "../src/shared/contracts";

/**
 * Voice always uses LiveKit (Apple Voice Processing I/O) where its helper
 * exists, macOS on Apple silicon. Elsewhere the WebRTC/sounddevice path stays
 * as the automatic fallback until a native processor exists for that platform.
 * There is no user-facing choice.
 */
export function platformAudioBackend(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): DesktopAudioBackend {
  return platform === "darwin" && arch === "arm64" ? "livekit" : "sounddevice";
}
