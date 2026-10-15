import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { PNG } from "pngjs";

import type { TextureAtlasDelta } from "../src/shared/contracts";
import { applyTextureEdit } from "./character-generation/model-builder";

export interface TextureAtlasEdit {
  textureUnitId: string;
  atlasIndex: number;
  bounds: { x: number; y: number; width: number; height: number };
  maskPath: string;
  imagePath: string;
}

export interface TextureAtlasOutput {
  atlasIndex: number;
  atlasFile: string;
  sourcePath: string;
  targetPath: string;
  url: string;
}

export async function composeTextureAtlasDelta(input: {
  modelId: string;
  streamId: string;
  revision: number;
  outputs: readonly TextureAtlasOutput[];
  edits: readonly TextureAtlasEdit[];
}): Promise<TextureAtlasDelta> {
  const temporaryFiles: Array<{ temporary: string; target: string }> = [];
  try {
    for (const output of input.outputs) {
      const source = await readPng(output.sourcePath);
      const image = {
        width: source.width,
        height: source.height,
        data: new Uint8Array(source.data),
      };
      for (const edit of input.edits.filter((item) => item.atlasIndex === output.atlasIndex)) {
        await applyTextureEdit(image, {
          textureUnitId: edit.textureUnitId,
          atlasIndex: edit.atlasIndex,
          bounds: edit.bounds,
          maskPath: edit.maskPath,
          editedPatchPath: edit.imagePath,
        });
      }
      const png = new PNG({ width: image.width, height: image.height, filterType: 4 });
      png.data.set(image.data);
      await mkdir(path.dirname(output.targetPath), { recursive: true });
      const temporary = `${output.targetPath}.${randomUUID()}.tmp`;
      await writePng(temporary, png);
      temporaryFiles.push({ temporary, target: output.targetPath });
    }
    for (const file of temporaryFiles) await rename(file.temporary, file.target);
  } catch (error) {
    await Promise.all(temporaryFiles.map(({ temporary }) => rm(temporary, { force: true })));
    throw error;
  }
  return {
    modelId: input.modelId,
    streamId: input.streamId,
    revision: input.revision,
    patchSets: [],
    atlases: input.outputs.map((output) => ({
      atlasIndex: output.atlasIndex,
      atlasFile: output.atlasFile,
      url: output.url,
    })),
  };
}

function readPng(file: string): Promise<PNG> {
  return readFile(file).then((data) => new Promise<PNG>((resolve, reject) => {
    new PNG().parse(data, (error, png) => error ? reject(error) : resolve(png));
  }));
}

function writePng(file: string, png: PNG): Promise<void> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const stream = png.pack();
    stream.on("data", (chunk: Buffer) => chunks.push(chunk));
    stream.once("error", reject);
    stream.once("end", () => {
      writeFile(file, Buffer.concat(chunks)).then(resolve, reject);
    });
  });
}
