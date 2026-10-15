/** Match Cubism's packed-mask limits: 36 in one texture, 32 each with multiple textures. */
export function cubismMaskBufferCount(
  drawableMasks: readonly ArrayLike<number>[],
  offscreenMasks: readonly ArrayLike<number>[] = [],
): number {
  // The two clipping managers allocate their own textures. Include hidden meshes
  // because an expression or motion can reveal them after initialization.
  const groups = Math.max(uniqueMaskCount(drawableMasks), uniqueMaskCount(offscreenMasks));
  return groups <= 36 ? 1 : Math.ceil(groups / 32);
}

function uniqueMaskCount(masks: readonly ArrayLike<number>[]): number {
  const groups = new Set<string>();
  for (const mask of masks) {
    if (mask.length === 0) continue;
    // Cubism shares a clipping context when the mask IDs match in any order.
    groups.add(Array.from(mask).sort((a, b) => a - b).join(","));
  }
  return groups.size;
}
