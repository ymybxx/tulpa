import { PNG } from "pngjs";

import { TEXTURE_EDIT_REGION_COLORS } from "../../src/shared/character-generation";

type RegionPlacement = {
  x: number;
  y: number;
  width: number;
  height: number;
  canvasWidth: number;
  canvasHeight: number;
};

export interface AnnotatedModelRegion {
  key: string;
  maskBytes: Uint8Array;
  regionIndex: number;
  placement: RegionPlacement;
}

/** Overlay an exact model-space ArtMesh mask on the full neutral reference image. */
export function composeModelLocatorPng(
  referenceBytes: Uint8Array,
  maskBytes: Uint8Array,
  regionIndex: number,
  placement: RegionPlacement,
  regionKey = alphabeticalRegionKey(regionIndex),
): Buffer {
  const reference = PNG.sync.read(Buffer.from(referenceBytes));
  const mask = PNG.sync.read(Buffer.from(maskBytes));
  validatePlacement(reference, mask, placement);
  const output = PNG.sync.read(PNG.sync.write(reference));
  overlayRegion(output, mask, regionIndex, placement);
  paintModelRegionBadge(output, mask, regionIndex, regionKey, placement);
  return PNG.sync.write(output);
}

/** Overlay every selected texture on one full-model reference using the same A/B/C/D keys as the edit canvas. */
export function composeAnnotatedModelReferencePng(
  referenceBytes: Uint8Array,
  regions: readonly AnnotatedModelRegion[],
): Buffer {
  const reference = PNG.sync.read(Buffer.from(referenceBytes));
  const output = PNG.sync.read(PNG.sync.write(reference));
  const decoded = regions.map((region) => {
    const mask = PNG.sync.read(Buffer.from(region.maskBytes));
    validatePlacement(reference, mask, region.placement);
    overlayRegion(output, mask, region.regionIndex, region.placement);
    return { ...region, mask };
  });
  for (const region of decoded) {
    paintModelRegionBadge(output, region.mask, region.regionIndex, region.key, region.placement);
  }
  return PNG.sync.write(output);
}

/**
 * Many pieces go on several sheets, each small enough to be sent at full size:
 * one shrunk sheet leaves its letters a few pixels tall, and the model can no
 * longer tell which letter marks which piece.
 */
/** Pieces paired with where they sit on the model, a few to a sheet so every letter and crop stays readable. */
export function composePieceSheetPngs(
  regions: ReadonlyArray<{ key: string; regionIndex: number; image: PNG }>,
  perSheet = 24,
): Buffer[] {
  const sheets: Buffer[] = [];
  const count = Math.max(1, Math.ceil(regions.length / perSheet));
  const size = Math.ceil(regions.length / count);
  for (let start = 0; start < regions.length; start += size) {
    sheets.push(composeRegionSheetPng(regions.slice(start, start + size), 200, 40, 4, 412));
  }
  return sheets;
}

export function composeRegionSheetPngs(
  regions: ReadonlyArray<{ key: string; regionIndex: number; image: PNG }>,
  perSheet = 36,
): Buffer[] {
  if (regions.length <= 16) return [composeRegionSheetPng(regions)];
  const sheets: Buffer[] = [];
  const count = Math.ceil(regions.length / perSheet);
  const size = Math.ceil(regions.length / count);
  for (let start = 0; start < regions.length; start += size) {
    sheets.push(composeRegionSheetPng(regions.slice(start, start + size), 196, 44, 6));
  }
  return sheets;
}

/**
 * Every selected texture on one grey sheet (so white and pale pieces show),
 * each shrunk into a cell and framed with its region letter — one image for
 * the reasoning model instead of one per region.
 */
export function composeRegionSheetPng(
  regions: ReadonlyArray<{ key: string; regionIndex: number; image: PNG }>,
  cell = 240,
  gutter = 36,
  maxColumns = 6,
  cellWidth = cell,
): Buffer {
  const columns = Math.max(1, Math.min(maxColumns, Math.ceil(Math.sqrt(regions.length))));
  const rows = Math.max(1, Math.ceil(regions.length / columns));
  const output = new PNG({ width: columns * (cellWidth + gutter) + gutter, height: rows * (cell + gutter) + gutter });
  for (let offset = 0; offset < output.data.length; offset += 4) output.data.set([150, 150, 158, 255], offset);
  regions.forEach((region, index) => {
    const { image } = region;
    const fit = Math.min(1, cellWidth / image.width, cell / image.height);
    const width = Math.max(1, Math.round(image.width * fit));
    const height = Math.max(1, Math.round(image.height * fit));
    const left = gutter + (index % columns) * (cellWidth + gutter) + Math.floor((cellWidth - width) / 2);
    const top = gutter + Math.floor(index / columns) * (cell + gutter) + Math.floor((cell - height) / 2);
    const step = 1 / fit;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        // Average the source block each sheet pixel covers.
        let red = 0, green = 0, blue = 0, alpha = 0, count = 0;
        const sy0 = Math.floor(y * step), sy1 = Math.min(image.height, Math.max(sy0 + 1, Math.floor((y + 1) * step)));
        const sx0 = Math.floor(x * step), sx1 = Math.min(image.width, Math.max(sx0 + 1, Math.floor((x + 1) * step)));
        for (let sy = sy0; sy < sy1; sy += 1) {
          for (let sx = sx0; sx < sx1; sx += 1) {
            const source = (sy * image.width + sx) * 4;
            const a = image.data[source + 3]! / 255;
            red += image.data[source]! * a;
            green += image.data[source + 1]! * a;
            blue += image.data[source + 2]! * a;
            alpha += a;
            count += 1;
          }
        }
        const coverage = alpha / count;
        const target = ((top + y) * output.width + left + x) * 4;
        output.data[target] = Math.round(red / count + output.data[target]! * (1 - coverage));
        output.data[target + 1] = Math.round(green / count + output.data[target + 1]! * (1 - coverage));
        output.data[target + 2] = Math.round(blue / count + output.data[target + 2]! * (1 - coverage));
      }
    }
    paintTextureRegionMarker(output, region.regionIndex, region.key, { x: left, y: top, width, height }, gutter);
  });
  return PNG.sync.write(output);
}

/** Paint an A/B/C/D badge into protected gutter pixels beside one texture. */
export function paintTextureRegionMarker(
  image: PNG,
  regionIndex: number,
  regionKey: string,
  bounds: { x: number; y: number; width: number; height: number },
  gutter: number,
): void {
  const color = TEXTURE_EDIT_REGION_COLORS[regionIndex % TEXTURE_EDIT_REGION_COLORS.length]!;
  const frame = Math.max(2, Math.min(5, Math.floor(gutter / 12)));
  paintOutlineOutside(image, bounds, color, frame);
  // A badge is 11 glyph pixels tall including its padding. Keep it wholly in
  // the protected gutter so neither the label nor its background can enter an
  // editable texture region.
  const scale = Math.max(1, Math.min(5, Math.floor((gutter - 4) / 11)));
  const badgeWidth = textPixelWidth(regionKey, scale) + scale * 4;
  const badgeHeight = 7 * scale + scale * 4;
  const badgeX = clamp(bounds.x, 0, image.width - badgeWidth);
  const belowY = bounds.y + bounds.height + Math.max(2, Math.floor((gutter - badgeHeight) / 2));
  const badgeY = belowY + badgeHeight <= image.height
    ? belowY
    : clamp(bounds.y - badgeHeight - 2, 0, image.height - badgeHeight);
  paintBadge(image, regionKey, badgeX, badgeY, scale, color);
}

function validatePlacement(reference: PNG, mask: PNG, placement: RegionPlacement): void {
  if (
    reference.width !== placement.canvasWidth || reference.height !== placement.canvasHeight ||
    mask.width !== placement.width || mask.height !== placement.height ||
    placement.x < 0 || placement.y < 0 ||
    placement.x + placement.width > reference.width || placement.y + placement.height > reference.height
  ) {
    throw new Error("纹理人物定位蒙版尺寸与完整人物参考图不一致，请重新预检");
  }
}

function overlayRegion(output: PNG, mask: PNG, regionIndex: number, placement: RegionPlacement): void {
  const color = TEXTURE_EDIT_REGION_COLORS[regionIndex % TEXTURE_EDIT_REGION_COLORS.length]!;
  for (let y = 0; y < mask.height; y += 1) {
    for (let x = 0; x < mask.width; x += 1) {
      const maskOffset = (y * mask.width + x) * 4;
      const maskAlpha = mask.data[maskOffset + 3]! / 255;
      if (maskAlpha <= 0.03) continue;
      const outputOffset = ((placement.y + y) * output.width + placement.x + x) * 4;
      const strength = 0.35 + maskAlpha * 0.45;
      for (let channel = 0; channel < 3; channel += 1) {
        output.data[outputOffset + channel] = Math.round(
          output.data[outputOffset + channel]! * (1 - strength) + color[channel]! * strength,
        );
      }
      output.data[outputOffset + 3] = 255;
    }
  }
}

function paintModelRegionBadge(
  output: PNG,
  mask: PNG,
  regionIndex: number,
  regionKey: string,
  placement: RegionPlacement,
): void {
  // Tiny fixtures (and genuinely tiny preview images) do not have enough
  // protected context for a readable badge. The precise colour overlay still
  // identifies the region without overwriting unrelated pixels.
  if (output.width < 32 || output.height < 32) return;
  const local = opaqueBounds(mask) ?? { x: 0, y: 0, width: mask.width, height: mask.height };
  const color = TEXTURE_EDIT_REGION_COLORS[regionIndex % TEXTURE_EDIT_REGION_COLORS.length]!;
  const scale = clamp(Math.floor(Math.min(output.width, output.height) / 360), 2, 8);
  const badgeWidth = textPixelWidth(regionKey, scale) + scale * 4;
  const badgeHeight = 7 * scale + scale * 4;
  const anchorX = placement.x + local.x + Math.floor(local.width / 2);
  const top = placement.y + local.y;
  const bottom = top + local.height;
  const badgeX = clamp(anchorX - Math.floor(badgeWidth / 2), 0, output.width - badgeWidth);
  const badgeY = top >= badgeHeight + scale * 2
    ? top - badgeHeight - scale * 2
    : clamp(bottom + scale * 2, 0, output.height - badgeHeight);
  paintBadge(output, regionKey, badgeX, badgeY, scale, color);
}

function opaqueBounds(image: PNG): { x: number; y: number; width: number; height: number } | null {
  let minimumX = image.width;
  let minimumY = image.height;
  let maximumX = -1;
  let maximumY = -1;
  for (let y = 0; y < image.height; y += 1) {
    for (let x = 0; x < image.width; x += 1) {
      if (image.data[(y * image.width + x) * 4 + 3]! <= 8) continue;
      minimumX = Math.min(minimumX, x);
      minimumY = Math.min(minimumY, y);
      maximumX = Math.max(maximumX, x);
      maximumY = Math.max(maximumY, y);
    }
  }
  return maximumX < minimumX ? null : {
    x: minimumX,
    y: minimumY,
    width: maximumX - minimumX + 1,
    height: maximumY - minimumY + 1,
  };
}

function paintOutlineOutside(
  image: PNG,
  bounds: { x: number; y: number; width: number; height: number },
  color: readonly [number, number, number],
  thickness: number,
): void {
  const left = bounds.x - thickness;
  const top = bounds.y - thickness;
  const right = bounds.x + bounds.width;
  const bottom = bounds.y + bounds.height;
  const rgba: readonly [number, number, number, number] = [color[0], color[1], color[2], 255];
  fillRect(image, left, top, bounds.width + thickness * 2, thickness, rgba);
  fillRect(image, left, bottom, bounds.width + thickness * 2, thickness, rgba);
  fillRect(image, left, top, thickness, bounds.height + thickness * 2, rgba);
  fillRect(image, right, top, thickness, bounds.height + thickness * 2, rgba);
}

function paintBadge(
  image: PNG,
  text: string,
  x: number,
  y: number,
  scale: number,
  color: readonly [number, number, number],
): void {
  const width = textPixelWidth(text, scale) + scale * 4;
  const height = 7 * scale + scale * 4;
  fillRect(image, x, y, width, height, [20, 24, 27, 255]);
  fillRect(
    image,
    x + scale,
    y + scale,
    width - scale * 2,
    height - scale * 2,
    [color[0], color[1], color[2], 255],
  );
  let cursor = x + scale * 2;
  for (const character of text.toUpperCase()) {
    const glyph = LETTER_GLYPHS[character] ?? LETTER_GLYPHS["?"]!;
    for (let row = 0; row < glyph.length; row += 1) {
      for (let column = 0; column < glyph[row]!.length; column += 1) {
        if (glyph[row]![column] !== "1") continue;
        fillRect(image, cursor + column * scale, y + scale * 2 + row * scale, scale, scale, [255, 255, 255, 255]);
      }
    }
    cursor += 6 * scale;
  }
}

function fillRect(
  image: PNG,
  x: number,
  y: number,
  width: number,
  height: number,
  color: readonly [number, number, number, number],
): void {
  const x0 = clamp(Math.floor(x), 0, image.width);
  const y0 = clamp(Math.floor(y), 0, image.height);
  const x1 = clamp(Math.ceil(x + width), 0, image.width);
  const y1 = clamp(Math.ceil(y + height), 0, image.height);
  for (let py = y0; py < y1; py += 1) {
    for (let px = x0; px < x1; px += 1) image.data.set(color, (py * image.width + px) * 4);
  }
}

function textPixelWidth(value: string, scale: number): number {
  return Math.max(5 * scale, value.length * 6 * scale - scale);
}

function alphabeticalRegionKey(index: number): string {
  let value = index + 1;
  let output = "";
  while (value > 0) {
    value -= 1;
    output = String.fromCharCode(65 + value % 26) + output;
    value = Math.floor(value / 26);
  }
  return output;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

const LETTER_GLYPHS: Record<string, readonly string[]> = {
  A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
  B: ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
  C: ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
  D: ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
  E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
  F: ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
  G: ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
  H: ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
  I: ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
  J: ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
  K: ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
  L: ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
  M: ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
  N: ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
  O: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
  P: ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
  Q: ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
  R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
  S: ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
  T: ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
  U: ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
  V: ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
  W: ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
  X: ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
  Y: ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
  Z: ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
  "?": ["01110", "10001", "00010", "00100", "00100", "00000", "00100"],
};

/**
 * Where one piece sits on the model, the way hovering it in the editor shows
 * it: the model cropped around the piece, the piece in its own colours and
 * everything else dimmed. Unlike a colour-coded overlay, nothing here can be
 * mistaken for the piece's own colour.
 */
export function composePieceLocatorPng(reference: PNG, maskBytes: Uint8Array, placement: RegionPlacement): PNG {
  const mask = PNG.sync.read(Buffer.from(maskBytes));
  validatePlacement(reference, mask, placement);
  const side = Math.min(Math.max(reference.width, reference.height), Math.max(260, Math.round(Math.max(placement.width, placement.height) * 1.8)));
  const left = clamp(Math.round(placement.x + placement.width / 2 - side / 2), 0, Math.max(0, reference.width - side));
  const top = clamp(Math.round(placement.y + placement.height / 2 - side / 2), 0, Math.max(0, reference.height - side));
  const width = Math.min(side, reference.width - left);
  const height = Math.min(side, reference.height - top);
  const output = new PNG({ width, height });
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sx = left + x, sy = top + y;
      const source = (sy * reference.width + sx) * 4;
      const mx = sx - placement.x, my = sy - placement.y;
      const inside = mx >= 0 && my >= 0 && mx < mask.width && my < mask.height
        ? mask.data[(my * mask.width + mx) * 4 + 3]! / 255 : 0;
      const alpha = reference.data[source + 3]! / 255;
      // Dim the rest towards the sheet's grey; keep the piece as drawn.
      const keep = 0.28 + 0.72 * inside;
      const target = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const colour = reference.data[source + channel]! * alpha + 150 * (1 - alpha);
        output.data[target + channel] = Math.round(colour * keep + 40 * (1 - keep));
      }
      output.data[target + 3] = 255;
    }
  }
  return output;
}

/** One sheet cell: the texture on the left, where it sits on the model on the right. */
export function pairPieceTile(texture: PNG, locator: PNG, half = 200, gap = 12): PNG {
  const output = new PNG({ width: half * 2 + gap, height: half });
  const place = (image: PNG, offsetX: number, opaque: boolean) => {
    const fit = Math.min(1, half / image.width, half / image.height);
    const width = Math.max(1, Math.round(image.width * fit));
    const height = Math.max(1, Math.round(image.height * fit));
    const left = offsetX + Math.floor((half - width) / 2);
    const top = Math.floor((half - height) / 2);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const sx = Math.min(image.width - 1, Math.floor(x / fit));
        const sy = Math.min(image.height - 1, Math.floor(y / fit));
        const source = (sy * image.width + sx) * 4;
        const target = ((top + y) * output.width + left + x) * 4;
        for (let channel = 0; channel < 4; channel += 1) output.data[target + channel] = image.data[source + channel]!;
        if (opaque) output.data[target + 3] = 255;
      }
    }
  };
  place(texture, 0, false);
  place(locator, half + gap, true);
  return output;
}
