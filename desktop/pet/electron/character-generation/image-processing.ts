export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface SlotBounds {
  x: number;
  y: number;
  width: number;
  height: number;
  anchor: { x: number; y: number };
}

export function preserveVisiblePixels(original: RgbaImage, completed: RgbaImage): RgbaImage {
  assertSameSize(original, completed);
  const data = completed.data.slice();
  for (let offset = 0; offset < data.length; offset += 4) {
    if ((original.data[offset + 3] ?? 0) === 0) continue;
    data[offset] = original.data[offset]!;
    data[offset + 1] = original.data[offset + 1]!;
    data[offset + 2] = original.data[offset + 2]!;
    data[offset + 3] = original.data[offset + 3]!;
  }
  return { width: original.width, height: original.height, data };
}

export function cropToVisibleBounds(image: RgbaImage): RgbaImage {
  if (image.data.length !== image.width * image.height * 4) throw new Error("RGBA 图片格式无效");
  let left = image.width;
  let top = image.height;
  let right = -1;
  let bottom = -1;
  for (let y = 0; y < image.height; y += 1) {
    for (let x = 0; x < image.width; x += 1) {
      if (image.data[(y * image.width + x) * 4 + 3] === 0) continue;
      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x);
      bottom = Math.max(bottom, y);
    }
  }
  if (right < left || bottom < top) throw new Error("部件图片没有可见像素");
  const width = right - left + 1;
  const height = bottom - top + 1;
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const sourceStart = ((top + y) * image.width + left) * 4;
    data.set(image.data.subarray(sourceStart, sourceStart + width * 4), y * width * 4);
  }
  return { width, height, data };
}

export function alignToSlot(
  source: SlotBounds,
  target: SlotBounds,
  constraints: { minScale: number; maxScale: number; aspectTolerance: number },
): { scale: number; translateX: number; translateY: number } {
  if (source.width <= 0 || source.height <= 0 || target.width <= 0 || target.height <= 0) {
    throw new Error("部件轮廓尺寸无效");
  }
  const sourceAspect = source.width / source.height;
  const targetAspect = target.width / target.height;
  const aspectDifference = Math.abs(sourceAspect / targetAspect - 1);
  if (aspectDifference > constraints.aspectTolerance) {
    throw new Error("角色部件轮廓超出母版容差，请重新生成该部件");
  }
  const scale = Math.min(target.width / source.width, target.height / source.height);
  if (scale < constraints.minScale || scale > constraints.maxScale) {
    throw new Error("角色部件缩放超出母版容差，请重新生成该部件");
  }
  return {
    scale,
    translateX: target.anchor.x - source.anchor.x * scale,
    translateY: target.anchor.y - source.anchor.y * scale,
  };
}

function assertSameSize(left: RgbaImage, right: RgbaImage): void {
  const leftLength = left.width * left.height * 4;
  const rightLength = right.width * right.height * 4;
  if (
    left.width !== right.width || left.height !== right.height ||
    left.data.length !== leftLength || right.data.length !== rightLength
  ) throw new Error("RGBA 图片尺寸不一致");
}
