export interface CharacterFrame {
  /** Crop rectangle in source pixels; may extend past the source (fill with transparency). */
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Finds a crop that puts the character's body centerline in the middle.
 *
 * The centerline is the median of each row's opaque midpoint, so props that
 * stick out on one side for a few rows (a staff, a hat brim, a tail) do not
 * pull the character off-center the way a plain bounding box would.
 */
export function characterFrame(
  rgba: ArrayLike<number>,
  width: number,
  height: number,
  padding = 12,
  alphaThreshold = 8,
): CharacterFrame | null {
  const midpoints: number[] = [];
  let left = width;
  let right = -1;
  let top = -1;
  let bottom = -1;
  for (let y = 0; y < height; y++) {
    let rowLeft = -1;
    let rowRight = -1;
    for (let x = 0; x < width; x++) {
      if (rgba[(y * width + x) * 4 + 3]! <= alphaThreshold) continue;
      if (rowLeft < 0) rowLeft = x;
      rowRight = x;
    }
    if (rowLeft < 0) continue;
    if (top < 0) top = y;
    bottom = y;
    left = Math.min(left, rowLeft);
    right = Math.max(right, rowRight);
    midpoints.push((rowLeft + rowRight) / 2);
  }
  if (right < 0) return null;
  midpoints.sort((a, b) => a - b);
  const center = midpoints[Math.floor(midpoints.length / 2)]!;
  const halfWidth = Math.ceil(Math.max(center - left, right - center)) + padding;
  return {
    left: Math.round(center) - halfWidth,
    top: Math.max(0, top - padding),
    width: halfWidth * 2 + 1,
    height: Math.min(height - 1, bottom + padding) - Math.max(0, top - padding) + 1,
  };
}
