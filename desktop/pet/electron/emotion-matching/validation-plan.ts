import type { EmotionAsset } from "../../src/shared/emotion-matching";

/** Observe a persistent expression or both sides of a motion's first loop boundary. */
export function stateValidationAsset(asset: EmotionAsset): EmotionAsset {
  const times = asset.kind === "motion"
    ? [...asset.timestampsMs, asset.durationMs + 17, asset.durationMs + 200]
    : [...asset.timestampsMs, Math.max(asset.durationMs + 200, 2400)];
  return { ...asset, timestampsMs: [...new Set(times.map((ms) => Math.min(119_000, ms)))].sort((a, b) => a - b) };
}
