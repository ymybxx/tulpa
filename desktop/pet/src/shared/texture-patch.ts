export interface TexturePatchBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TextureSize {
  width: number;
  height: number;
}

export interface TexturePatchDestination {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export function texturePatchDestination(
  bounds: TexturePatchBounds,
  source: TextureSize,
  target: TextureSize,
): TexturePatchDestination {
  assertSize(source, "基础图集");
  assertSize(target, "目标图集");
  if (
    ![bounds.x, bounds.y, bounds.width, bounds.height].every(Number.isFinite) ||
    bounds.x < 0 || bounds.y < 0 || bounds.width <= 0 || bounds.height <= 0 ||
    bounds.x + bounds.width > source.width || bounds.y + bounds.height > source.height
  ) throw new Error("纹理 patch 超出基础图集范围");
  return {
    left: Math.floor(bounds.x * target.width / source.width),
    top: Math.floor(bounds.y * target.height / source.height),
    right: Math.ceil((bounds.x + bounds.width) * target.width / source.width),
    bottom: Math.ceil((bounds.y + bounds.height) * target.height / source.height),
  };
}

export function texturePatchFramebufferRect(
  destination: TexturePatchDestination,
  framebufferHeight: number,
): { x: number; y: number; width: number; height: number } {
  if (!Number.isFinite(framebufferHeight) || framebufferHeight <= 0) {
    throw new Error("framebuffer 高度无效");
  }
  if (
    destination.left < 0 || destination.top < 0 ||
    destination.right <= destination.left || destination.bottom <= destination.top ||
    destination.bottom > framebufferHeight
  ) throw new Error("纹理 patch 的 framebuffer 区域无效");
  return {
    x: destination.left,
    // Live2D atlases and patches are uploaded from ImageBitmap without a
    // vertical flip. The PNG's first (top) row therefore occupies texture
    // storage row / framebuffer y = 0. Mirroring through framebufferHeight
    // here would move every patch to the opposite side of the atlas.
    y: destination.top,
    width: destination.right - destination.left,
    height: destination.bottom - destination.top,
  };
}

export function composeTexturePatchPixel(
  current: readonly number[],
  patch: readonly number[],
  maskAlpha: number,
): [number, number, number, number] {
  if (current.length < 4 || patch.length < 4) throw new Error("RGBA 像素不完整");
  if (maskAlpha <= 0) return [current[0]!, current[1]!, current[2]!, current[3]!];
  return [
    patch[0]!,
    patch[1]!,
    patch[2]!,
    Math.min(current[3]!, patch[3]!),
  ];
}

export function applyTexturePatchPixel(
  target: Uint8Array,
  targetOffset: number,
  patch: Uint8Array,
  patchOffset: number,
  maskAlpha: number,
): void {
  if (maskAlpha <= 0) return;
  target[targetOffset] = patch[patchOffset]!;
  target[targetOffset + 1] = patch[patchOffset + 1]!;
  target[targetOffset + 2] = patch[patchOffset + 2]!;
  target[targetOffset + 3] = Math.min(
    target[targetOffset + 3]!,
    patch[patchOffset + 3]!,
  );
}

function assertSize(value: TextureSize, label: string): void {
  if (
    !Number.isFinite(value.width) || value.width <= 0 ||
    !Number.isFinite(value.height) || value.height <= 0
  ) throw new Error(`${label}尺寸无效`);
}
