export const MIN_PET_MODEL_SCALE = 0.2;
/** Keep the transparent macOS panel bounded; additional magnification is in-canvas. */
export const MAX_PET_WINDOW_SCALE = 3;
export const MAX_PET_MODEL_SCALE = 64;

export function normalizePetModelScale(value: number, fallback = 1): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(MIN_PET_MODEL_SCALE, Math.min(MAX_PET_MODEL_SCALE, value));
}

export function petModelScaleFromPinch(current: number, deltaY: number): number {
  if (!Number.isFinite(deltaY)) return normalizePetModelScale(current);
  const next = normalizePetModelScale(current) * Math.exp(-deltaY * 0.01);
  if (!Number.isFinite(next)) return deltaY < 0 ? MAX_PET_MODEL_SCALE : MIN_PET_MODEL_SCALE;
  return normalizePetModelScale(next);
}

export function petWindowScale(modelScale: number): number {
  return Math.min(normalizePetModelScale(modelScale), MAX_PET_WINDOW_SCALE);
}

export function petContentZoom(modelScale: number): number {
  return Math.max(1, normalizePetModelScale(modelScale) / MAX_PET_WINDOW_SCALE);
}
