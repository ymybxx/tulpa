import { readFile, writeFile } from "node:fs/promises";

import { PNG } from "pngjs";

/**
 * Build a display-only texture preview. The patch remains a rectangular
 * write-back artifact, while the preview exposes only pixels covered by the
 * texture unit's UV mask.
 */
export function createMaskedTexturePreview(patch: PNG, mask: PNG): PNG {
  if (patch.width !== mask.width || patch.height !== mask.height) {
    throw new Error("纹理版本与 UV 蒙版尺寸不一致");
  }
  const preview = new PNG({ width: patch.width, height: patch.height });
  preview.data.set(patch.data);
  for (let offset = 0; offset < preview.data.length; offset += 4) {
    preview.data[offset + 3] = Math.min(
      patch.data[offset + 3]!,
      mask.data[offset + 3]!,
    );
  }
  return preview;
}

export async function writeMaskedTexturePreview(
  patchPath: string,
  maskPath: string,
  target: string,
): Promise<void> {
  const [patch, mask] = await Promise.all([
    readFile(patchPath).then((value) => PNG.sync.read(value)),
    readFile(maskPath).then((value) => PNG.sync.read(value)),
  ]);
  await writeFile(target, PNG.sync.write(createMaskedTexturePreview(patch, mask)));
}
