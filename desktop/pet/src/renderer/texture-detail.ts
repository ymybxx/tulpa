import { PET_TEXTURE_DETAIL_LEVELS } from "../shared/texture-quality";

export { PET_TEXTURE_DETAIL_LEVELS } from "../shared/texture-quality";

export function supportedTextureDimensions(
  ceiling: number,
  gpuMaximum: number,
  levels: readonly number[] = PET_TEXTURE_DETAIL_LEVELS,
): number[] {
  const limit = Math.min(ceiling, gpuMaximum);
  const supported = levels.filter((level) => level <= limit);
  return supported.length > 0 ? supported : [Math.max(1, Math.floor(limit))];
}

/**
 * Chooses texture detail from the model's actual on-screen pixel size.
 * Hysteresis prevents a pinch gesture from repeatedly switching both atlas sets.
 */
export function adaptiveTextureDimension(
  renderedLongEdgePixels: number,
  current: number,
  levels: readonly number[] = PET_TEXTURE_DETAIL_LEVELS,
): number {
  const sorted = [...new Set(levels)]
    .filter((value) => Number.isFinite(value) && value > 0)
    .map((value) => Math.floor(value))
    .sort((left, right) => left - right);
  if (sorted.length === 0) return current;
  const high = sorted[sorted.length - 1]!;
  if (sorted.length === 1) return high;
  const pixels = Math.max(0, renderedLongEdgePixels);
  let index = sorted.findIndex((level) => level >= current);
  if (index < 0) index = sorted.length - 1;
  while (index < sorted.length - 1 && pixels > sorted[index]! * 0.9) index += 1;
  while (index > 0 && pixels < sorted[index - 1]! * 0.72) index -= 1;
  return sorted[index]!;
}

export function textureDetailRequestUrl(url: string, maximumDimension: number): string {
  const value = Math.max(1, Math.floor(maximumDimension));
  const parsed = new URL(url);
  if (parsed.protocol === "pet-model:") {
    parsed.searchParams.set("textureMaxDimension", String(value));
  }
  return parsed.toString();
}
