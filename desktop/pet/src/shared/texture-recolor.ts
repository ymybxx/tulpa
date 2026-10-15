/**
 * Recolors a texture piece to one target color without any AI: every pixel
 * keeps its lightness relative to the piece — line art stays darkest,
 * highlights stay brightest — and takes the target's hue. The piece's typical
 * lightness lands on the target's, so white stockings can become skin and blue
 * hair can become pink with the same shading.
 */

export interface RecolorTarget {
  /** sRGB, 0–255. */
  r: number;
  g: number;
  b: number;
}

const BINS = 1024;

/** Rewrites `pixels` (RGBA) in place wherever `mask` (alpha of an RGBA mask, same size) is set. */
export function recolorPixels(pixels: Uint8Array | Uint8ClampedArray, mask: Uint8Array | Uint8ClampedArray | null, target: RecolorTarget): void {
  const lightness = new Float32Array(pixels.length / 4);
  const histogram = new Float64Array(BINS);
  let weight = 0;
  for (let index = 0, pixel = 0; index < pixels.length; index += 4, pixel += 1) {
    const alpha = pixels[index + 3]!;
    if (!alpha || (mask && !mask[index + 3])) {
      lightness[pixel] = -1;
      continue;
    }
    const value = fastLightness(pixels[index]!, pixels[index + 1]!, pixels[index + 2]!);
    lightness[pixel] = value;
    histogram[Math.min(BINS - 1, Math.floor(value * BINS))]! += alpha;
    weight += alpha;
  }
  if (!weight) return;
  // The piece's typical lightness: the alpha-weighted median.
  let median = 0.5;
  for (let bin = 0, seen = 0; bin < BINS; bin += 1) {
    seen += histogram[bin]!;
    if (seen >= weight / 2) {
      median = (bin + 0.5) / BINS;
      break;
    }
  }
  median = Math.min(0.98, Math.max(0.02, median));
  const [okTargetLightness, targetChroma, hue] = okLch(target);
  // Measured the same way as the pixels, so the median lands exactly on it.
  const targetLightness = fastLightness(target.r, target.g, target.b);
  // Back to OKLab lightness for output, exact at the target and at black and white.
  const toOk = (value: number) => value <= targetLightness
    ? value * okTargetLightness / Math.max(1e-6, targetLightness)
    : okTargetLightness + (value - targetLightness) * (1 - okTargetLightness) / Math.max(1e-6, 1 - targetLightness);
  // Each output color depends only on the pixel's lightness, so a table does the work.
  const table = new Uint8Array(BINS * 3);
  for (let bin = 0; bin < BINS; bin += 1) {
    const source = (bin + 0.5) / BINS;
    // The piece's median lands on the target's lightness. Below it lightness
    // scales, so line art stays dark; above it the piece's own highlights keep
    // their full step, eased in just before white.
    const mapped = source <= median
      ? targetLightness * source / median
      : softCeiling(targetLightness + source - median);
    // At the target's own lightness the color is exactly the target; darker
    // and lighter shades carry chroma in proportion, fading toward black and white.
    const room = (value: number) => Math.sin(Math.PI * Math.min(1, Math.max(0, value)));
    const chroma = targetChroma * Math.min(1.3, room(mapped) / Math.max(0.05, room(targetLightness)));
    const [r, g, b] = okLchToSrgb(toOk(mapped), chroma, hue);
    table.set([r, g, b], bin * 3);
  }
  for (let index = 0, pixel = 0; index < pixels.length; index += 4, pixel += 1) {
    const value = lightness[pixel]!;
    if (value < 0) continue;
    const bin = Math.min(BINS - 1, Math.floor(value * BINS)) * 3;
    pixels[index] = table[bin]!;
    pixels[index + 1] = table[bin + 1]!;
    pixels[index + 2] = table[bin + 2]!;
  }
}

export function parseHexColor(value: string): RecolorTarget | null {
  const match = /^#?([0-9a-f]{6})$/iu.exec(value.trim());
  if (!match) return null;
  const number = Number.parseInt(match[1]!, 16);
  return { r: number >> 16 & 255, g: number >> 8 & 255, b: number & 255 };
}

/** Lightness beyond 0.9 approaches white without reaching or clipping at it. */
function softCeiling(value: number): number {
  const knee = 0.9;
  return value <= knee ? value : knee + (0.995 - knee) * Math.tanh((value - knee) / (0.995 - knee));
}

function linear(channel: number): number {
  const value = channel / 255;
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

function encode(value: number): number {
  const clamped = Math.min(1, Math.max(0, value));
  const srgb = clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * clamped ** (1 / 2.4) - 0.055;
  return Math.round(srgb * 255);
}

function okLab(r: number, g: number, b: number): [number, number, number] {
  const lr = linear(r), lg = linear(g), lb = linear(b);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [
    0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
  ];
}

const LINEAR = Float32Array.from({ length: 256 }, (_, channel) => linear(channel));
const CUBE_ROOT_STEPS = 65536;
const CUBE_ROOT = Float32Array.from({ length: CUBE_ROOT_STEPS + 1 }, (_, step) => Math.cbrt(step / CUBE_ROOT_STEPS));

/**
 * Perceptual lightness from two table lookups: the cube root of relative
 * luminance, which equals OKLab lightness for greys and stays within a few
 * percent for colors — only order and step size matter here.
 */
function fastLightness(r: number, g: number, b: number): number {
  const luminance = 0.2126 * LINEAR[r]! + 0.7152 * LINEAR[g]! + 0.0722 * LINEAR[b]!;
  return CUBE_ROOT[Math.round(luminance * CUBE_ROOT_STEPS)]!;
}

function okLch(color: RecolorTarget): [number, number, number] {
  const [l, a, b] = okLab(color.r, color.g, color.b);
  return [l, Math.hypot(a, b), Math.atan2(b, a)];
}

/** OKLCH to sRGB, lowering chroma until the color fits in sRGB. */
function okLchToSrgb(lightness: number, chroma: number, hue: number): [number, number, number] {
  for (let c = chroma; ; c *= 0.9) {
    const a = c * Math.cos(hue), b = c * Math.sin(hue);
    const l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3;
    const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    const bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
    const inside = [r, g, bl].every((value) => value >= -0.001 && value <= 1.001);
    if (inside || c < 0.001) return [encode(r), encode(g), encode(bl)];
  }
}
