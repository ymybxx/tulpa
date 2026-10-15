import type {
  MasterTemplateCanvas,
  MasterTemplateSlot,
} from "../../src/shared/character-generation";
import type { RgbaImage } from "./image-processing";

export interface MeshSlotTexture {
  slot: MasterTemplateSlot;
  /** Full neutral-pose canvas. Transparent pixels are written as transparent. */
  image: RgbaImage;
  /**
   * When present, only the RGB difference between image and reference is
   * applied to the existing atlas. This is used for eye restyling so the
   * original iris alpha, line art and separate Live2D highlight layers are
   * never baked into the base texture a second time.
   */
  reference?: RgbaImage;
  mask?: RgbaImage;
}

/**
 * Inverse-bakes neutral-pose character pixels into Cubism UV triangles.
 * This preserves the original ArtMesh topology instead of stretching a crop
 * into a hand-written rectangle.
 */
export function composeMeshAtlas(
  base: RgbaImage,
  canvas: MasterTemplateCanvas,
  textures: MeshSlotTexture[],
): RgbaImage {
  validateImage(base);
  const output: RgbaImage = { ...base, data: base.data.slice() };
  for (const texture of [...textures].sort((left, right) => left.slot.zIndex - right.slot.zIndex)) {
    bakeDrawable(output, canvas, texture);
  }
  return output;
}

function bakeDrawable(
  atlas: RgbaImage,
  canvas: MasterTemplateCanvas,
  texture: MeshSlotTexture,
): void {
  validateImage(texture.image);
  if (texture.reference) {
    validateImage(texture.reference);
    if (
      texture.reference.width !== texture.image.width ||
      texture.reference.height !== texture.image.height
    ) {
      throw new Error(`槽位 ${texture.slot.id} 的差异参考图尺寸不一致`);
    }
  }
  const { slot } = texture;
  if (texture.mask) {
    validateImage(texture.mask);
    if (texture.mask.width !== slot.atlas.width || texture.mask.height !== slot.atlas.height) {
      throw new Error(`槽位 ${slot.id} 的轮廓蒙版尺寸不一致`);
    }
  }
  const destination = pairs(slot.mesh.uvs).map(([u, v]) => ({
    x: u * atlas.width,
    y: (1 - v) * atlas.height,
  }));
  const source = pairs(slot.mesh.vertices).map(([x, y]) => ({
    x: ((canvas.originX + x * canvas.pixelsPerUnit) / canvas.width) * texture.image.width,
    y: ((canvas.originY - y * canvas.pixelsPerUnit) / canvas.height) * texture.image.height,
  }));
  const visited = new Uint8Array(atlas.width * atlas.height);
  for (let offset = 0; offset < slot.mesh.indices.length; offset += 3) {
    const i0 = slot.mesh.indices[offset]!;
    const i1 = slot.mesh.indices[offset + 1]!;
    const i2 = slot.mesh.indices[offset + 2]!;
    rasterTriangle(
      atlas,
      texture,
      [destination[i0]!, destination[i1]!, destination[i2]!],
      [source[i0]!, source[i1]!, source[i2]!],
      visited,
    );
  }
}

function rasterTriangle(
  atlas: RgbaImage,
  texture: MeshSlotTexture,
  destination: [Point, Point, Point],
  source: [Point, Point, Point],
  visited: Uint8Array,
): void {
  const [a, b, c] = destination;
  const area = edge(a, b, c);
  if (Math.abs(area) < 1e-9) return;
  const minX = clamp(Math.floor(Math.min(a.x, b.x, c.x)), 0, atlas.width - 1);
  const maxX = clamp(Math.ceil(Math.max(a.x, b.x, c.x)), 0, atlas.width - 1);
  const minY = clamp(Math.floor(Math.min(a.y, b.y, c.y)), 0, atlas.height - 1);
  const maxY = clamp(Math.ceil(Math.max(a.y, b.y, c.y)), 0, atlas.height - 1);
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const point = { x: x + 0.5, y: y + 0.5 };
      const weights = [
        edge(b, c, point) / area,
        edge(c, a, point) / area,
        edge(a, b, point) / area,
      ] as const;
      if (weights.some((weight) => weight < -1e-5)) continue;
      const pixelIndex = y * atlas.width + x;
      if (visited[pixelIndex]) continue;
      visited[pixelIndex] = 1;
      const sourceX = source[0].x * weights[0] + source[1].x * weights[1] + source[2].x * weights[2];
      const sourceY = source[0].y * weights[0] + source[1].y * weights[1] + source[2].y * weights[2];
      const rgba = bilinear(texture.image, sourceX, sourceY);
      const maskX = x - texture.slot.atlas.x;
      const maskY = y - texture.slot.atlas.y;
      const maskAlpha = texture.mask && maskX >= 0 && maskY >= 0 &&
          maskX < texture.mask.width && maskY < texture.mask.height
        ? texture.mask.data[(maskY * texture.mask.width + maskX) * 4 + 3]! / 255
        : 1;
      const targetOffset = (y * atlas.width + x) * 4;
      if (texture.reference) {
        const reference = bilinear(texture.reference, sourceX, sourceY);
        for (let channel = 0; channel < 3; channel += 1) {
          const delta = rgba[channel]! - reference[channel]!;
          atlas.data[targetOffset + channel] = clamp(
            Math.round(atlas.data[targetOffset + channel]! + delta * maskAlpha),
            0,
            255,
          );
        }
        // Eye-only edits must never change the original atlas silhouette.
        // Blink and gaze meshes rely on this alpha matching the original rig.
      } else {
        atlas.data[targetOffset] = rgba[0];
        atlas.data[targetOffset + 1] = rgba[1];
        atlas.data[targetOffset + 2] = rgba[2];
        atlas.data[targetOffset + 3] = Math.round(rgba[3] * maskAlpha);
      }
    }
  }
}

function bilinear(image: RgbaImage, x: number, y: number): [number, number, number, number] {
  const px = Math.max(0, Math.min(image.width - 1, x));
  const py = Math.max(0, Math.min(image.height - 1, y));
  const x0 = Math.floor(px);
  const y0 = Math.floor(py);
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const fx = px - x0;
  const fy = py - y0;
  const result = [0, 0, 0, 0] as [number, number, number, number];
  for (let channel = 0; channel < 4; channel += 1) {
    const top = value(image, x0, y0, channel) * (1 - fx) + value(image, x1, y0, channel) * fx;
    const bottom = value(image, x0, y1, channel) * (1 - fx) + value(image, x1, y1, channel) * fx;
    result[channel] = Math.round(top * (1 - fy) + bottom * fy);
  }
  return result;
}

function value(image: RgbaImage, x: number, y: number, channel: number): number {
  return image.data[(y * image.width + x) * 4 + channel]!;
}

function pairs(values: number[]): Array<[number, number]> {
  const output: Array<[number, number]> = [];
  for (let index = 0; index < values.length; index += 2) {
    output.push([values[index]!, values[index + 1]!]);
  }
  return output;
}

function edge(a: Point, b: Point, point: Point): number {
  return (point.x - a.x) * (b.y - a.y) - (point.y - a.y) * (b.x - a.x);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function validateImage(image: RgbaImage): void {
  if (image.width <= 0 || image.height <= 0 || image.data.length !== image.width * image.height * 4) {
    throw new Error("RGBA 图片格式无效");
  }
}

interface Point {
  x: number;
  y: number;
}
