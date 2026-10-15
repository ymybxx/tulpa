import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { PNG } from "pngjs";

import type { TextureEditUnit } from "../src/shared/contracts";
import { parseHexColor, recolorPixels } from "../src/shared/texture-recolor";
import { decodePng } from "./png-reader";

export interface RecoloredPatch {
  textureUnitId: string;
  label: string;
  atlasIndex: number;
  atlasFile: string;
  bounds: TextureEditUnit["bounds"];
  /** Absolute paths inside `directory`. */
  maskPath: string;
  editedPatchPath: string;
}

/**
 * Writes each piece recolored to `hex` into `directory`: its current pixels
 * (whatever version it wears) re-toned by lightness, and its UV mask beside it.
 * `name` keeps files of different previews apart so a renderer still loading
 * one never reads another's bytes.
 */
export async function writeRecoloredPatches(input: {
  units: readonly TextureEditUnit[];
  hex: string;
  directory: string;
  name: string;
  currentImage(unit: TextureEditUnit): Promise<string>;
  maskFile(unit: TextureEditUnit): string;
  /** True once a newer request replaced this one; the rest is skipped. */
  superseded?(): boolean;
}): Promise<RecoloredPatch[] | null> {
  const parsed = parseHexColor(input.hex);
  if (!parsed) throw new Error("颜色无效");
  const target = parsed;
  await mkdir(input.directory, { recursive: true });
  const patches: RecoloredPatch[] = [];
  // One piece at a time, handing the thread back in between: a big part
  // would otherwise freeze the app for seconds.
  for (const unit of input.units) {
    if (input.superseded?.()) return null;
    await new Promise((resolve) => setImmediate(resolve));
    patches.push(await recolorOne(unit));
  }
  return patches;

  async function recolorOne(unit: TextureEditUnit): Promise<RecoloredPatch> {
    const [image, maskBytes] = await Promise.all([
      input.currentImage(unit).then((file) => readFile(file)).then(decodePng),
      readFile(input.maskFile(unit)),
    ]);
    const mask = PNG.sync.read(maskBytes);
    if (image.width !== unit.bounds.width || image.height !== unit.bounds.height) {
      throw new Error(`${unit.label}的当前纹理尺寸不一致`);
    }
    recolorPixels(image.data, mask.width === image.width && mask.height === image.height ? mask.data : null, target);
    const key = createHash("sha256").update(unit.id).digest("hex").slice(0, 16);
    const editedPatchPath = path.join(input.directory, `recolor-${key}-${input.name}.png`);
    const maskPath = path.join(input.directory, `recolor-${key}-mask.png`);
    await Promise.all([
      // Fast compression: these files are read once or twice, then removed.
      writeFile(editedPatchPath, PNG.sync.write(image, { deflateLevel: 1, filterType: 0 } as Parameters<typeof PNG.sync.write>[1])),
      writeFile(maskPath, maskBytes),
    ]);
    return {
      textureUnitId: unit.id,
      label: unit.label,
      atlasIndex: unit.atlasIndex,
      atlasFile: unit.atlasFile,
      bounds: unit.bounds,
      maskPath,
      editedPatchPath,
    };
  }
}
