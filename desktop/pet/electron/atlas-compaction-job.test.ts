import { mkdir, mkdtemp, readFile, rm, stat, utimes, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it, onTestFinished, vi } from "vitest";
import { PNG } from "pngjs";

import {
  executeAtlasCompactionRequest,
  validateAtlasCompactionRequestPaths,
} from "./atlas-compaction-job";
import type { AtlasCompactionRequest } from "./atlas-compaction-coordinator";

describe("executeAtlasCompactionRequest", () => {
  it("reuses a content-addressed output without rewriting it", async () => {
    const fixture = await requestFixture();
    await mkdir(path.dirname(fixture.request.atlases[0]!.targetPath), { recursive: true });
    await writeFile(fixture.request.atlases[0]!.targetPath, "compiled");
    const originalTime = new Date(10_000);
    await utimes(fixture.request.atlases[0]!.targetPath, originalTime, originalTime);
    const compose = vi.fn();

    const result = await executeAtlasCompactionRequest(fixture.request, { compose });

    expect(compose).not.toHaveBeenCalled();
    expect(result).toMatchObject({ built: 0, reused: 1 });
    expect((await stat(fixture.request.atlases[0]!.targetPath)).mtimeMs).toBe(originalTime.getTime());
    expect(await readFile(fixture.request.atlases[0]!.targetPath, "utf8")).toBe("compiled");
  });

  it("builds only missing outputs and reports stable content keys", async () => {
    const fixture = await requestFixture();
    const compose = vi.fn(async (input: Parameters<NonNullable<Parameters<typeof executeAtlasCompactionRequest>[1]>["compose"]>[0]) => {
      await Promise.all(input.outputs.map(async (output) => {
        await mkdir(path.dirname(output.targetPath), { recursive: true });
        await writeFile(output.targetPath, "new");
      }));
      return {
        modelId: input.modelId,
        streamId: input.streamId,
        revision: input.revision,
        patchSets: [],
        atlases: [],
      };
    });

    const result = await executeAtlasCompactionRequest(fixture.request, { compose });

    expect(compose).toHaveBeenCalledOnce();
    expect(result).toEqual({
      modelId: "mao",
      revision: 7,
      contentKeys: [fixture.contentKey],
      built: 1,
      reused: 0,
    });
  });

  it("composes a real PNG once and then treats the immutable path as a cache hit", async () => {
    const fixture = await pngRequestFixture();

    const first = await executeAtlasCompactionRequest(fixture.request);
    const compiled = await readFile(fixture.request.atlases[0]!.targetPath);
    const firstTime = (await stat(fixture.request.atlases[0]!.targetPath)).mtimeMs;
    const second = await executeAtlasCompactionRequest(fixture.request);

    expect(first).toMatchObject({ built: 1, reused: 0 });
    expect(second).toMatchObject({ built: 0, reused: 1 });
    expect(await readFile(fixture.request.atlases[0]!.targetPath)).toEqual(compiled);
    expect((await stat(fixture.request.atlases[0]!.targetPath)).mtimeMs).toBe(firstTime);
    const output = PNG.sync.read(compiled);
    expect([...output.data.subarray(0, 4)]).toEqual([255, 0, 0, 128]);
  });

  it("rejects sources outside the allowed roots before composition", async () => {
    const fixture = await requestFixture();
    const outside = await createTemporaryDirectory("atlas-outside-");
    fixture.request.atlases[0]!.edits[0]!.imagePath = path.join(outside, "patch.png");
    await writeFile(fixture.request.atlases[0]!.edits[0]!.imagePath, "patch");
    const compose = vi.fn();

    await expect(executeAtlasCompactionRequest(fixture.request, { compose }))
      .rejects.toThrow("允许目录之外");
    expect(compose).not.toHaveBeenCalled();
  });

  it("rejects a target that is not namespaced by its immutable content key", async () => {
    const fixture = await requestFixture();
    fixture.request.atlases[0]!.targetPath = path.join(fixture.root, "compiled", "mutable", "textures", "atlas.png");

    await expect(validateAtlasCompactionRequestPaths(fixture.request))
      .rejects.toThrow("内容键目录");
  });
});

async function requestFixture(): Promise<{
  root: string;
  contentKey: string;
  request: AtlasCompactionRequest;
}> {
  const root = await createTemporaryDirectory("atlas-job-");
  const contentKey = "a".repeat(64);
  const sourcePath = path.join(root, "textures", "atlas.png");
  const maskPath = path.join(root, "masks", "mask.png");
  const imagePath = path.join(root, "materials", "patch.png");
  await Promise.all([sourcePath, maskPath, imagePath].map(async (file) => {
    await mkdir(path.dirname(file), { recursive: true });
    await writeFile(file, "fixture");
  }));
  return {
    root,
    contentKey,
    request: {
      modelId: "mao",
      revision: 7,
      allowedRoots: [root],
      atlases: [{
        contentKey,
        atlasFile: "textures/atlas.png",
        sourcePath,
        targetPath: path.join(root, ".appearance", "compiled", contentKey, "textures", "atlas.png"),
        edits: [{
          textureUnitId: "leg",
          atlasIndex: 0,
          bounds: { x: 1, y: 2, width: 3, height: 4 },
          maskPath,
          imagePath,
        }],
      }],
    },
  };
}

async function pngRequestFixture(): Promise<{
  request: AtlasCompactionRequest;
}> {
  const fixture = await requestFixture();
  const atlas = fixture.request.atlases[0]!;
  atlas.edits[0]!.bounds = { x: 0, y: 0, width: 1, height: 1 };
  await writePng(atlas.sourcePath, 2, 2, [20, 30, 40, 255]);
  await writePng(atlas.edits[0]!.maskPath, 1, 1, [255, 255, 255, 255]);
  await writePng(atlas.edits[0]!.imagePath, 1, 1, [255, 0, 0, 128]);
  return { request: fixture.request };
}

async function writePng(
  file: string,
  width: number,
  height: number,
  pixel: readonly [number, number, number, number],
): Promise<void> {
  const png = new PNG({ width, height });
  for (let offset = 0; offset < png.data.length; offset += 4) {
    png.data.set(pixel, offset);
  }
  await writeFile(file, PNG.sync.write(png, { filterType: 4 }));
}

async function createTemporaryDirectory(prefix: string): Promise<string> {
  const directory = await mkdtemp(path.join(os.tmpdir(), prefix));
  onTestFinished(() => rm(directory, { recursive: true, force: true }));
  return directory;
}
