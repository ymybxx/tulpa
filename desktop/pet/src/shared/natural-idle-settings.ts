import type { ExpressiveIdleSettings, IdleChannel, IdlePreset, NaturalIdleSettings } from "./contracts";
export const IDLE_CHANNELS: readonly IdleChannel[] = ["headX", "headY", "headZ", "bodyX", "bodyY", "bodyZ", "eyeX", "eyeY", "eyeLOpen", "eyeROpen"];
export const IDLE_PRESET_LABELS: Record<IdlePreset, string> = { gentle: "轻柔", natural: "自然", active: "活跃" };
export function expressiveIdleDefaults(preset: IdlePreset): ExpressiveIdleSettings {
  const active = preset === "active", gentle = preset === "gentle";
  return {
    preset, bodyIntensity: 1, eyeActivity: 1,
    pauseProbability: active ? 0.1 : gentle ? 0.3 : 0.18,
    largeMotionProbability: active ? 0.45 : gentle ? 0.1 : 0.25,
    reverseInterval: active ? 2 : 3,
    followDelay: active ? 0.2 : 0.28,
    yawRange: active ? 22 : gentle ? 13 : 18,
    pitchRange: active ? 13 : gentle ? 7 : 10,
    rollRange: active ? 17 : gentle ? 10 : 14,
    blink: true,
  };
}
export function idlePresetSettings(preset: IdlePreset): NaturalIdleSettings {
  return { enabled: true, intensity: 1, speed: 1, expressive: expressiveIdleDefaults(preset) };
}
/** Keep older manifests disabled and tolerate incomplete imported settings. */
export function normalizeNaturalIdleSettings(value: unknown): NaturalIdleSettings {
  const settings = value && typeof value === "object"
    ? value as Record<string, unknown>
    : {};
  return {
    enabled: settings.enabled === true,
    intensity: bounded(settings.intensity, 0, 2, 1),
    speed: bounded(settings.speed, 0.5, 1.5, 1),
    ...(isRecord(settings.expressive) ? { expressive: normalizeExpressive(settings.expressive) } : {}),
  };
}
function normalizeExpressive(value: Record<string, unknown>): ExpressiveIdleSettings {
  const preset: IdlePreset = value.preset === "gentle" || value.preset === "active" ? value.preset : "natural";
  const defaults = expressiveIdleDefaults(preset);
  const result: ExpressiveIdleSettings = {
    preset,
    bodyIntensity: bounded(value.bodyIntensity, 0, 2, defaults.bodyIntensity),
    eyeActivity: bounded(value.eyeActivity, 0, 2, defaults.eyeActivity),
    pauseProbability: bounded(value.pauseProbability, 0, 0.8, defaults.pauseProbability),
    largeMotionProbability: bounded(value.largeMotionProbability, 0, 1, defaults.largeMotionProbability),
    reverseInterval: bounded(value.reverseInterval, 0, 10, defaults.reverseInterval),
    followDelay: bounded(value.followDelay, 0.08, 0.8, defaults.followDelay),
    yawRange: bounded(value.yawRange, 0, 28, defaults.yawRange),
    pitchRange: bounded(value.pitchRange, 0, 25, defaults.pitchRange),
    rollRange: bounded(value.rollRange, 0, 25, defaults.rollRange),
    blink: value.blink !== false,
  };
  if (isRecord(value.bindings)) {
    const bindings: NonNullable<ExpressiveIdleSettings["bindings"]> = {};
    for (const key of IDLE_CHANNELS) {
      const id = value.bindings[key];
      if (id === null || (typeof id === "string" && id.length > 0 && id.length <= 256))
        bindings[key] = id;
    }
    if (Object.keys(bindings).length)
      result.bindings = bindings;
  }
  if (isRecord(value.bodyDirections)) {
    const directions: NonNullable<ExpressiveIdleSettings["bodyDirections"]> = {};
    for (const axis of ["x", "y", "z"] as const) {
      const direction = value.bodyDirections[axis];
      if (direction === -1 || direction === 1)
        directions[axis] = direction;
    }
    if (Object.keys(directions).length)
      result.bodyDirections = directions;
  }
  return result;
}
function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
function bounded(value: unknown, min: number, max: number, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(max, Math.max(min, value))
    : fallback;
}
