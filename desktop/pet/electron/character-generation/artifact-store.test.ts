import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { PNG } from "pngjs";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { TextureEditUnit } from "../../src/shared/contracts";
import {
  planTextureEditBatch,
  type MasterTemplateSlot,
} from "../../src/shared/character-generation";
import { CharacterArtifactStore, generationCacheKey } from "./artifact-store";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("CharacterArtifactStore", () => {
  it("builds one exact batch canvas while keeping each texture source independent", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const atlas = new PNG({ width: 100, height: 200 });
    for (let y = 0; y < atlas.height; y += 1) {
      for (let x = 0; x < atlas.width; x += 1) {
        atlas.data.set(x < 50 ? [200, 20, 30, 255] : [20, 80, 220, 255], (y * atlas.width + x) * 4);
      }
    }
    const mask = new PNG({ width: 50, height: 200 });
    mask.data.fill(255);
    const guide = new PNG({ width: 20, height: 30 });
    for (let offset = 0; offset < guide.data.length; offset += 4) {
      guide.data.set([17, 91, 143, 255], offset);
    }
    const modelMask = new PNG({ width: 2, height: 2 });
    modelMask.data.fill(255);
    const atlasPath = path.join(root, "atlas.png");
    const maskPath = path.join(root, "mask.png");
    const guidePath = path.join(root, "guide.png");
    const modelMaskPath = path.join(root, "model-mask.png");
    await Promise.all([
      writeFile(atlasPath, PNG.sync.write(atlas)),
      writeFile(maskPath, PNG.sync.write(mask)),
      writeFile(guidePath, PNG.sync.write(guide)),
      writeFile(modelMaskPath, PNG.sync.write(modelMask)),
    ]);
    const unit = (id: string, x: number, label: string): TextureEditUnit => ({
      id,
      atlasIndex: 0,
      atlasFile: "atlas.png",
      maskHash: id,
      bounds: { x, y: 0, width: 50, height: 200 },
      maskPath: "mask.png",
      previewPath: "preview.png",
      modelMaskPath: "model-mask.png",
      modelMaskBounds: { x: 2, y: 4, width: 2, height: 2, canvasWidth: 20, canvasHeight: 30 },
      drawableIds: [`D_${id}`],
      partIds: [id],
      semanticPartIds: [],
      revealedByMotionIds: [],
      label,
      visiblePixels: 10_000,
      modelBounds: { x: 0, y: 0, width: 0.2, height: 0.5 },
    });
    const left = unit("left", 0, "Left");
    const right = unit("right", 50, "Right");
    const planned = planTextureEditBatch([
      { id: left.id, label: left.label, width: 50, height: 200 },
      { id: right.id, label: right.label, width: 50, height: 200 },
    ]);
    if (!planned.fits) throw new Error(planned.reason);
    const store = new CharacterArtifactStore(root);

    const work = await store.createTextureEditBatchWorkCanvas(
      "job",
      [left, right].map((entry) => ({
        atlasPath,
        unitMaskPath: maskPath,
        modelMaskPath,
        unit: entry,
      })),
      planned.plan,
      "batch",
    );
    const combinedMask = PNG.sync.read(await readFile(work.maskPath));
    const leftLayout = planned.plan.items.find((item) => item.textureUnitId === "left")!;
    const rightLayout = planned.plan.items.find((item) => item.textureUnitId === "right")!;
    const leftOffset = (leftLayout.y * combinedMask.width + leftLayout.x) * 4;
    const rightOffset = (rightLayout.y * combinedMask.width + rightLayout.x) * 4;
    const betweenX = leftLayout.x + leftLayout.workWidth;
    const betweenOffset = (leftLayout.y * combinedMask.width + betweenX) * 4;
    const leftSource = PNG.sync.read(await readFile(work.items.left!.sourcePath));
    const rightSource = PNG.sync.read(await readFile(work.items.right!.sourcePath));
    const workImage = PNG.sync.read(await readFile(work.imagePath));

    expect([...combinedMask.data.subarray(leftOffset, leftOffset + 4)]).toEqual([255, 255, 255, 0]);
    expect([...combinedMask.data.subarray(rightOffset, rightOffset + 4)]).toEqual([255, 255, 255, 0]);
    expect([...combinedMask.data.subarray(betweenOffset, betweenOffset + 4)]).toEqual([255, 255, 255, 255]);
    expect([...leftSource.data.subarray(0, 4)]).toEqual([200, 20, 30, 255]);
    expect([...rightSource.data.subarray(0, 4)]).toEqual([20, 80, 220, 255]);
    expect(work.items.left!.layout.scale).toBe(leftLayout.scale);
    expect(work.items.right!.layout.scale).toBe(rightLayout.scale);
    // Coloured frames and A/B badges live only in protected pixels.
    const markerPixels = [...workImage.data].filter((value, index) =>
      index % 4 !== 3 && (value < 30 || value > 245)
    ).length;
    expect(markerPixels).toBeGreaterThan(0);
    // Reference images are no longer painted into the edit canvas; an unused
    // corner remains the neutral protected background.
    const lastPixel = [...workImage.data.subarray(workImage.data.length - 4)];
    expect(lastPixel).toEqual([246, 247, 248, 255]);

    // A later canvas shows this canvas's result, shrunk, in its protected anchor area.
    const used = Math.max(...planned.plan.items.map((item) => item.x + item.workWidth));
    const anchor = { x: used + planned.plan.gutter, y: planned.plan.gutter, width: 150, height: 150 };
    expect(anchor.x + anchor.width).toBeLessThan(planned.plan.canvasWidth);
    const later = await store.createTextureEditBatchWorkCanvas(
      "job",
      [left, right].map((entry) => ({ atlasPath, unitMaskPath: maskPath, modelMaskPath, unit: entry })),
      { ...planned.plan, anchor },
      "batch-2",
      { imagePath: work.imagePath, plan: planned.plan },
    );
    const laterImage = PNG.sync.read(await readFile(later.imagePath));
    const laterMask = PNG.sync.read(await readFile(later.maskPath));
    const anchorPixels: number[][] = [];
    for (let y = anchor.y; y < anchor.y + anchor.height; y += 4) {
      for (let x = anchor.x; x < anchor.x + anchor.width; x += 4) {
        const offset = (y * laterImage.width + x) * 4;
        anchorPixels.push([...laterImage.data.subarray(offset, offset + 4)]);
        expect(laterMask.data[offset + 3]).toBe(255);
      }
    }
    expect(anchorPixels.some(([r, g, b]) => r! > 150 && g! < 80 && b! < 80)).toBe(true);
    expect(anchorPixels.some(([r, g, b]) => b! > 150 && r! < 80)).toBe(true);
  });

  it("shrinks a piece onto the canvas and maps the result back to its full size", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const atlas = new PNG({ width: 40, height: 40 });
    for (let y = 0; y < 40; y += 1) {
      for (let x = 0; x < 40; x += 1) {
        // Left half red, right half fully transparent.
        atlas.data.set(x < 20 ? [200, 20, 30, 255] : [0, 0, 0, 0], (y * 40 + x) * 4);
      }
    }
    const mask = new PNG({ width: 40, height: 40 });
    for (let offset = 0; offset < mask.data.length; offset += 4) {
      const x = (offset / 4) % 40;
      mask.data.set([255, 255, 255, x < 20 ? 255 : 0], offset);
    }
    const atlasPath = path.join(root, "atlas.png");
    const maskPath = path.join(root, "mask.png");
    await Promise.all([writeFile(atlasPath, PNG.sync.write(atlas)), writeFile(maskPath, PNG.sync.write(mask))]);
    const unit = {
      id: "big", atlasIndex: 0, atlasFile: "atlas.png", maskHash: "big",
      bounds: { x: 0, y: 0, width: 40, height: 40 }, maskPath: "mask.png", previewPath: "p.png",
      modelMaskPath: "m.png", modelMaskBounds: { x: 0, y: 0, width: 1, height: 1, canvasWidth: 1, canvasHeight: 1 },
      drawableIds: ["D"], partIds: ["P"], semanticPartIds: [], revealedByMotionIds: [], label: "大块",
      visiblePixels: 800, modelBounds: { x: 0, y: 0, width: 1, height: 1 },
    } satisfies TextureEditUnit;
    const planned = planTextureEditBatch([{ id: "big", label: "大块", width: 40, height: 40 }]);
    if (!planned.fits) throw new Error(planned.reason);
    const plan = {
      ...planned.plan,
      items: [{ ...planned.plan.items[0]!, scale: 0.5, workWidth: 20, workHeight: 20 }],
    };
    const store = new CharacterArtifactStore(root);
    const work = await store.createTextureEditBatchWorkCanvas(
      "job", [{ atlasPath, unitMaskPath: maskPath, modelMaskPath: maskPath, unit }], plan, "shrunk");
    const image = PNG.sync.read(await readFile(work.imagePath));
    const combined = PNG.sync.read(await readFile(work.maskPath));
    const at = (png: PNG, x: number, y: number) => [...png.data.subarray((y * png.width + x) * 4, (y * png.width + x) * 4 + 4)];
    const item = plan.items[0]!;
    expect(at(image, item.x + 2, item.y + 2)).toEqual([200, 20, 30, 255]);
    expect(at(image, item.x + 15, item.y + 2)).toEqual([246, 246, 246, 255]);
    expect(at(combined, item.x + 2, item.y + 2)[3]).toBe(0);
    expect(at(combined, item.x + 15, item.y + 2)[3]).toBe(255);
    // Just past the shrunk piece is protected canvas.
    expect(at(combined, item.x + 21, item.y + 2)[3]).toBe(255);

    const constrained = await store.constrainTextureEditResult(
      "job", work.items.big!.sourcePath, work.imagePath, maskPath, work.items.big!.layout, "大块", "big-edited");
    const output = PNG.sync.read(await readFile(constrained.path));
    expect(output.width).toBe(40);
    expect(at(output, 5, 5)).toEqual([200, 20, 30, 255]);
  });

  it("downloads HTTPS results immediately to one job directory", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const fetcher = vi.fn(async () => new Response(new Uint8Array([1, 2, 3]), {
      status: 200,
      headers: { "content-type": "image/png" },
    }));
    const store = new CharacterArtifactStore(root, fetcher);

    const local = await store.download("job-1", "https://fal.media/result.png", "candidate-0");

    expect(local).toBe(path.join(root, "job-1", "candidate-0.png"));
    expect([...await readFile(local)]).toEqual([1, 2, 3]);
    expect(fetcher).toHaveBeenCalledWith(
      "https://fal.media/result.png",
      expect.objectContaining({ redirect: "follow" }),
    );
  });

  it("rejects non-HTTPS and oversized resources", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const store = new CharacterArtifactStore(
      root,
      async () => new Response(new Uint8Array(11), { status: 200 }),
      10,
    );

    await expect(store.download("job", "http://example.com/a.png", "a")).rejects.toThrow(/HTTPS/);
    await expect(store.download("job", "https://example.com/a.png", "a")).rejects.toThrow(/过大/);
  });

  it("applies a binary mask locally without changing source colors", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const source = new PNG({ width: 2, height: 1 });
    source.data.set([200, 100, 50, 255, 20, 40, 60, 128]);
    const mask = new PNG({ width: 2, height: 1 });
    mask.data.set([255, 255, 255, 255, 0, 0, 0, 255]);
    const fetcher = vi.fn(async (url: string) => new Response(
      url.endsWith("source.png") ? PNG.sync.write(source) : PNG.sync.write(mask),
      { status: 200, headers: { "content-type": "image/png" } },
    ));
    const store = new CharacterArtifactStore(root, fetcher);
    const sourcePath = await store.download("job", "https://cdn/source.png", "source");
    const maskPath = await store.download("job", "https://cdn/mask.png", "mask");

    const resultPath = await store.applyMask("job", sourcePath, maskPath, "result");
    const result = PNG.sync.read(await readFile(resultPath));

    expect([...result.data]).toEqual([
      200, 100, 50, 255,
      20, 40, 60, 0,
    ]);
  });

  it("accepts only targeted completion images with the original canvas size", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const reference = new PNG({ width: 4, height: 6 });
    const matching = new PNG({ width: 4, height: 6 });
    const cropped = new PNG({ width: 4, height: 5 });
    const fetcher = vi.fn(async (url: string) => {
      const image = url.includes("matching") ? matching : url.includes("cropped") ? cropped : reference;
      return new Response(PNG.sync.write(image), {
        status: 200,
        headers: { "content-type": "image/png" },
      });
    });
    const store = new CharacterArtifactStore(root, fetcher);
    const referencePath = await store.download("job", "https://cdn/reference.png", "reference");
    const matchingPath = await store.download("job", "https://cdn/matching.png", "matching");
    const croppedPath = await store.download("job", "https://cdn/cropped.png", "cropped");

    await expect(store.assertSameImageSize(referencePath, matchingPath)).resolves.toBeUndefined();
    await expect(store.assertSameImageSize(referencePath, croppedPath)).rejects.toThrow(/4×6.*4×5/);
  });

  it("rejects fake transparency checkerboards but accepts a plain white candidate border", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const plain = new PNG({ width: 100, height: 100 });
    const checker = new PNG({ width: 100, height: 100 });
    for (let offset = 0; offset < plain.data.length; offset += 4) {
      plain.data.set([255, 255, 255, 255], offset);
    }
    for (let y = 0; y < checker.height; y += 1) {
      for (let x = 0; x < checker.width; x += 1) {
        const value = (Math.floor(x / 8) + Math.floor(y / 8)) % 2 === 0 ? 255 : 205;
        checker.data.set([value, value, value, 255], (y * checker.width + x) * 4);
      }
    }
    const store = new CharacterArtifactStore(root, async (url) => new Response(
      PNG.sync.write(url.includes("checker") ? checker : plain),
      { status: 200, headers: { "content-type": "image/png" } },
    ));
    const plainPath = await store.download("job", "https://cdn/plain.png", "plain");
    const checkerPath = await store.download("job", "https://cdn/checker.png", "checker");

    await expect(store.inspectCandidateBackground(plainPath)).resolves.toEqual({
      compatible: true,
      reason: null,
    });
    await expect(store.inspectCandidateBackground(checkerPath)).resolves.toEqual({
      compatible: false,
      reason: expect.stringContaining("棋盘格"),
    });
  });

  it("removes a border-connected checker floor without erasing an enclosed white garment", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const image = new PNG({ width: 20, height: 20 });
    for (let y = 0; y < image.height; y += 1) {
      for (let x = 0; x < image.width; x += 1) {
        const background = y < 10 ? 255 : (Math.floor(x / 2) + Math.floor(y / 2)) % 2 ? 205 : 245;
        image.data.set([background, background, background, 255], (y * image.width + x) * 4);
      }
    }
    for (let y = 4; y <= 15; y += 1) {
      for (let x = 6; x <= 13; x += 1) image.data.set([20, 20, 20, 255], (y * 20 + x) * 4);
    }
    for (let y = 6; y <= 13; y += 1) {
      for (let x = 8; x <= 11; x += 1) image.data.set([250, 250, 250, 255], (y * 20 + x) * 4);
    }
    const source = path.join(root, "source.png");
    await import("node:fs/promises").then(({ writeFile }) => writeFile(source, PNG.sync.write(image)));
    const store = new CharacterArtifactStore(root);
    const normalizedPath = await store.normalizeCandidateBackground("job", source, "normalized");
    const normalized = PNG.sync.read(await readFile(normalizedPath));

    expect([...normalized.data.subarray((15 * 20 + 1) * 4, (15 * 20 + 1) * 4 + 4)])
      .toEqual([255, 255, 255, 255]);
    expect([...normalized.data.subarray((8 * 20 + 9) * 4, (8 * 20 + 9) * 4 + 4)])
      .toEqual([250, 250, 250, 255]);
  });

  it("deterministically reframes a generated person to the guide foreground bounds", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const guide = new PNG({ width: 100, height: 200 });
    const candidate = new PNG({ width: 100, height: 200 });
    for (const image of [guide, candidate]) {
      for (let offset = 0; offset < image.data.length; offset += 4) {
        image.data.set([255, 255, 255, 255], offset);
      }
    }
    for (let y = 30; y <= 169; y += 1) {
      for (let x = 30; x <= 69; x += 1) guide.data.set([30, 60, 90, 255], (y * 100 + x) * 4);
    }
    for (let y = 5; y <= 194; y += 1) {
      for (let x = 20; x <= 79; x += 1) candidate.data.set([120, 30, 60, 255], (y * 100 + x) * 4);
    }
    const guidePath = path.join(root, "guide.png");
    const candidatePath = path.join(root, "candidate.png");
    const { writeFile } = await import("node:fs/promises");
    await writeFile(guidePath, PNG.sync.write(guide));
    await writeFile(candidatePath, PNG.sync.write(candidate));
    const store = new CharacterArtifactStore(root);
    const alignedPath = await store.alignCandidateToGuide("job", guidePath, candidatePath, "aligned");
    const aligned = PNG.sync.read(await readFile(alignedPath));

    expect(aligned.width).toBe(100);
    expect(aligned.height).toBe(200);
    const occupiedRows = Array.from({ length: aligned.height }, (_, y) => y).filter((y) =>
      Array.from({ length: aligned.width }, (_, x) => x).some((x) => {
        const offset = (y * aligned.width + x) * 4;
        return aligned.data[offset]! < 245 || aligned.data[offset + 1]! < 245 || aligned.data[offset + 2]! < 245;
      })
    );
    expect([occupiedRows[0], occupiedRows.at(-1)]).toEqual([30, 169]);
    expect([...aligned.data.subarray((40 * 100 + 30) * 4, (40 * 100 + 30) * 4 + 3)])
      .toEqual([120, 30, 60]);
    expect([...aligned.data.subarray((10 * 100 + 10) * 4, (10 * 100 + 10) * 4 + 3)])
      .toEqual([255, 255, 255]);
  });

  it("measures character pixels outside the replaceable Live2D mesh union", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const subject = new PNG({ width: 4, height: 4 });
    subject.data.set([20, 40, 60, 255], 0);
    subject.data.set([20, 40, 60, 255], (3 * 4 + 3) * 4);
    const subjectPath = path.join(root, "subject.png");
    const { writeFile } = await import("node:fs/promises");
    await writeFile(subjectPath, PNG.sync.write(subject));
    const slot: MasterTemplateSlot = {
      id: "body",
      name: "Body",
      drawableId: "D_BODY",
      partId: "PartBody",
      group: "body",
      source: "visible",
      required: true,
      zIndex: 0,
      atlas: { texture: 0, x: 0, y: 0, width: 2, height: 2 },
      anchor: { x: 1, y: 1 },
      canvasBounds: { x: 0, y: -2, width: 2, height: 2 },
      mesh: {
        vertices: [0, 0, 2, 0, 2, -2, 0, -2],
        uvs: [0, 1, 1, 1, 1, 0, 0, 0],
        indices: [0, 1, 2, 0, 2, 3],
      },
      silhouetteMask: "masks/body.png",
      completionPrompt: "body",
    };
    const store = new CharacterArtifactStore(root);

    const result = await store.inspectTemplateCoverage(
      subjectPath,
      { width: 4, height: 4, originX: 0, originY: 0, pixelsPerUnit: 1 },
      [slot],
    );

    expect(result.subjectPixels).toBe(2);
    expect(result.uncoveredPixels).toBe(1);
    expect(result.coveredRatio).toBe(0.5);
  });

  it("locks every pixel outside the iris meshes and preserves the original alpha and seam", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const reference = new PNG({ width: 24, height: 12 });
    const generated = new PNG({ width: 24, height: 12 });
    for (let offset = 0; offset < reference.data.length; offset += 4) {
      reference.data.set([40, 60, 80, 173], offset);
      generated.data.set([220, 30, 120, 255], offset);
    }
    const referencePath = path.join(root, "reference.png");
    const generatedPath = path.join(root, "generated.png");
    await writeFile(referencePath, PNG.sync.write(reference));
    await writeFile(generatedPath, PNG.sync.write(generated));
    const slot: MasterTemplateSlot = {
      id: "iris-left",
      name: "Iris",
      drawableId: "D_IRIS",
      partId: "PartEyeBall",
      group: "face",
      source: "visible",
      required: false,
      zIndex: 1,
      atlas: { texture: 0, x: 0, y: 0, width: 10, height: 10 },
      anchor: { x: 5, y: 5 },
      canvasBounds: { x: 4, y: -8, width: 12, height: 8 },
      mesh: {
        vertices: [4, 0, 16, 0, 16, -8, 4, -8],
        uvs: [0, 1, 1, 1, 1, 0, 0, 0],
        indices: [0, 1, 2, 0, 2, 3],
      },
      silhouetteMask: "masks/iris.png",
      completionPrompt: "iris",
    };
    const store = new CharacterArtifactStore(root);
    const maskPath = await store.createSlotEditMask(
      "job",
      referencePath,
      { width: 24, height: 12, originX: 0, originY: 0, pixelsPerUnit: 1 },
      [slot],
      "mask",
    );
    const resultPath = await store.constrainMaskedEdit(
      "job",
      referencePath,
      generatedPath,
      maskPath,
      "safe",
    );
    const result = PNG.sync.read(await readFile(resultPath));

    expect([...result.data.subarray(0, 4)]).toEqual([40, 60, 80, 173]);
    const boundary = (4 * 24 + 4) * 4;
    expect([...result.data.subarray(boundary, boundary + 4)]).toEqual([40, 60, 80, 173]);
    const interior = (4 * 24 + 10) * 4;
    expect([...result.data.subarray(interior, interior + 3)]).not.toEqual([40, 60, 80]);
    expect(result.data[interior + 3]).toBe(173);
  });

  it("accepts a clean recolor that preserves the original line art", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const reference = new PNG({ width: 64, height: 64 });
    const candidate = new PNG({ width: 64, height: 64 });
    const mask = new PNG({ width: 64, height: 64 });
    for (let y = 0; y < 64; y += 1) {
      for (let x = 0; x < 64; x += 1) {
        const offset = (y * 64 + x) * 4;
        const line = x >= 30 && x <= 32;
        reference.data.set(line ? [25, 30, 40, 255] : [80, 110, 180, 255], offset);
        candidate.data.set(line ? [25, 30, 40, 255] : [210, 120, 170, 255], offset);
        mask.data.set([255, 255, 255, 255], offset);
      }
    }
    const referencePath = path.join(root, "reference.png");
    const candidatePath = path.join(root, "candidate.png");
    const maskPath = path.join(root, "mask.png");
    await Promise.all([
      writeFile(referencePath, PNG.sync.write(reference)),
      writeFile(candidatePath, PNG.sync.write(candidate)),
      writeFile(maskPath, PNG.sync.write(mask)),
    ]);
    const store = new CharacterArtifactStore(root);

    await expect(store.constrainMaskedCropEdit(
      "job",
      referencePath,
      candidatePath,
      maskPath,
      {
        imagePath: referencePath,
        maskPath,
        source: { x: 0, y: 0, width: 64, height: 64 },
        output: { width: 64, height: 64 },
      },
      "衣服",
      "clean",
      "preserve",
    )).resolves.toMatch(/clean\.png$/);
  });

  it("rejects generated dirt and high-frequency speckles before texture writeback", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const reference = new PNG({ width: 64, height: 64 });
    const candidate = new PNG({ width: 64, height: 64 });
    const mask = new PNG({ width: 64, height: 64 });
    for (let y = 0; y < 64; y += 1) {
      for (let x = 0; x < 64; x += 1) {
        const offset = (y * 64 + x) * 4;
        reference.data.set([130, 150, 190, 255], offset);
        const noise = (x + y) % 2 === 0 ? 55 : 220;
        candidate.data.set([noise, Math.max(0, noise - 25), Math.min(255, noise + 20), 255], offset);
        mask.data.set([255, 255, 255, 255], offset);
      }
    }
    const referencePath = path.join(root, "reference.png");
    const candidatePath = path.join(root, "candidate.png");
    const maskPath = path.join(root, "mask.png");
    await Promise.all([
      writeFile(referencePath, PNG.sync.write(reference)),
      writeFile(candidatePath, PNG.sync.write(candidate)),
      writeFile(maskPath, PNG.sync.write(mask)),
    ]);
    const store = new CharacterArtifactStore(root);

    await expect(store.constrainMaskedCropEdit(
      "job",
      referencePath,
      candidatePath,
      maskPath,
      {
        imagePath: referencePath,
        maskPath,
        source: { x: 0, y: 0, width: 64, height: 64 },
        output: { width: 64, height: 64 },
      },
      "衣服",
      "dirty",
      "preserve",
    )).rejects.toThrow(/污点或噪声/);
  });

  it("reports dirty GPT texture pixels without taking the final decision from the user", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const source = new PNG({ width: 32, height: 32 });
    const candidate = new PNG({ width: 64, height: 64 });
    const mask = new PNG({ width: 32, height: 32 });
    for (let offset = 0; offset < source.data.length; offset += 4) {
      source.data.set([120, 145, 190, 173], offset);
      mask.data.set([255, 255, 255, 255], offset);
    }
    for (let y = 0; y < candidate.height; y += 1) {
      for (let x = 0; x < candidate.width; x += 1) {
        const value = (Math.floor(x / 2) + Math.floor(y / 2)) % 2 === 0 ? 35 : 230;
        candidate.data.set([value, Math.max(0, value - 20), Math.min(255, value + 15), 255], (y * candidate.width + x) * 4);
      }
    }
    const sourcePath = path.join(root, "texture-source.png");
    const candidatePath = path.join(root, "texture-candidate.png");
    const maskPath = path.join(root, "texture-mask.png");
    await Promise.all([
      writeFile(sourcePath, PNG.sync.write(source)),
      writeFile(candidatePath, PNG.sync.write(candidate)),
      writeFile(maskPath, PNG.sync.write(mask)),
    ]);
    const store = new CharacterArtifactStore(root);

    const result = await store.constrainTextureEditResult(
      "job",
      sourcePath,
      candidatePath,
      maskPath,
      {
        canvasWidth: 64,
        canvasHeight: 64,
        offsetX: 0,
        offsetY: 0,
        scale: 2,
        sourceWidth: 32,
        sourceHeight: 32,
      },
      "帽子纹理",
      "dirty-texture",
    );

    expect(result.quality.accepted).toBe(false);
    expect(result.quality.issues.join(" ")).toMatch(/颗粒或斑点/);
    expect(PNG.sync.read(await readFile(result.path))).toBeTruthy();
  });

  it("cleans generated white fringes and pads transparent RGB while preserving alpha", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const source = new PNG({ width: 12, height: 12 });
    const candidate = new PNG({ width: 24, height: 24 });
    const mask = new PNG({ width: 12, height: 12 });
    for (let y = 0; y < source.height; y += 1) {
      for (let x = 0; x < source.width; x += 1) {
        const offset = (y * source.width + x) * 4;
        const visible = x >= 3 && x <= 8 && y >= 3 && y <= 8;
        source.data.set([30, 60, 120, visible ? 255 : 0], offset);
        mask.data.set([255, 255, 255, 255], offset);
        for (let oy = 0; oy < 2; oy += 1) {
          for (let ox = 0; ox < 2; ox += 1) {
            const candidateOffset = ((y * 2 + oy) * candidate.width + x * 2 + ox) * 4;
            const boundary = visible && (x === 3 || x === 8 || y === 3 || y === 8);
            candidate.data.set(boundary ? [255, 255, 255, 255] : [230, 50, 145, 255], candidateOffset);
          }
        }
      }
    }
    const sourcePath = path.join(root, "edge-source.png");
    const candidatePath = path.join(root, "edge-candidate.png");
    const maskPath = path.join(root, "edge-mask.png");
    await Promise.all([
      writeFile(sourcePath, PNG.sync.write(source)),
      writeFile(candidatePath, PNG.sync.write(candidate)),
      writeFile(maskPath, PNG.sync.write(mask)),
    ]);
    const store = new CharacterArtifactStore(root);

    const result = await store.constrainTextureEditResult(
      "job",
      sourcePath,
      candidatePath,
      maskPath,
      {
        canvasWidth: 24,
        canvasHeight: 24,
        offsetX: 0,
        offsetY: 0,
        scale: 2,
        sourceWidth: 12,
        sourceHeight: 12,
      },
      "鞋子纹理",
      "clean-edge",
    );
    const output = PNG.sync.read(await readFile(result.path));
    const boundary = (3 * output.width + 3) * 4;
    const transparentPadding = (3 * output.width + 2) * 4;

    expect([...output.data.subarray(boundary, boundary + 3)]).toEqual([230, 50, 145]);
    expect(output.data[boundary + 3]).toBe(255);
    expect([...output.data.subarray(transparentPadding, transparentPadding + 3)]).toEqual(
      [...output.data.subarray(boundary, boundary + 3)],
    );
    expect(output.data[transparentPadding + 3]).toBe(0);
  });

  it("removes dark AI fringe fragments from light fine texture edges without erasing interior line art", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const source = new PNG({ width: 20, height: 20 });
    const candidate = new PNG({ width: 40, height: 40 });
    const mask = new PNG({ width: 20, height: 20 });
    for (let y = 0; y < source.height; y += 1) {
      for (let x = 0; x < source.width; x += 1) {
        const offset = (y * source.width + x) * 4;
        const visible = x >= 2 && x <= 17 && y >= 2 && y <= 17;
        source.data.set([225, 160, 100, visible ? 255 : 0], offset);
        mask.data.set([255, 255, 255, 255], offset);
        const edgeDepth = visible
          ? Math.min(x - 1, y - 1, 18 - x, 18 - y)
          : 0;
        const intentionalInteriorLine = x === 10 && y >= 8 && y <= 12;
        const generatedColor = intentionalInteriorLine || (visible && edgeDepth <= 3)
          ? [15, 18, 22, 255]
          : [246, 244, 248, 255];
        for (let oy = 0; oy < 2; oy += 1) {
          for (let ox = 0; ox < 2; ox += 1) {
            candidate.data.set(
              generatedColor,
              ((y * 2 + oy) * candidate.width + x * 2 + ox) * 4,
            );
          }
        }
      }
    }
    const sourcePath = path.join(root, "light-hair-source.png");
    const candidatePath = path.join(root, "light-hair-candidate.png");
    const maskPath = path.join(root, "light-hair-mask.png");
    await Promise.all([
      writeFile(sourcePath, PNG.sync.write(source)),
      writeFile(candidatePath, PNG.sync.write(candidate)),
      writeFile(maskPath, PNG.sync.write(mask)),
    ]);
    const store = new CharacterArtifactStore(root);

    const result = await store.constrainTextureEditResult(
      "job",
      sourcePath,
      candidatePath,
      maskPath,
      {
        canvasWidth: 40,
        canvasHeight: 40,
        offsetX: 0,
        offsetY: 0,
        scale: 2,
        sourceWidth: 20,
        sourceHeight: 20,
      },
      "白色头发纹理",
      "clean-dark-fragments",
      "locked",
    );
    const output = PNG.sync.read(await readFile(result.path));
    const firstEdge = (2 * output.width + 2) * 4;
    const thirdEdge = (4 * output.width + 4) * 4;
    const interiorLine = (10 * output.width + 10) * 4;

    expect([...output.data.subarray(firstEdge, firstEdge + 3)]).toEqual([246, 244, 248]);
    expect([...output.data.subarray(thirdEdge, thirdEdge + 3)]).toEqual([246, 244, 248]);
    expect([...output.data.subarray(interiorLine, interiorLine + 3)]).toEqual([15, 18, 22]);
  });

  it("does not brighten a legitimately dark generated material at its silhouette", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const source = new PNG({ width: 14, height: 14 });
    const candidate = new PNG({ width: 28, height: 28 });
    const mask = new PNG({ width: 14, height: 14 });
    for (let y = 0; y < source.height; y += 1) {
      for (let x = 0; x < source.width; x += 1) {
        const offset = (y * source.width + x) * 4;
        const visible = x >= 2 && x <= 11 && y >= 2 && y <= 11;
        source.data.set([240, 190, 170, visible ? 255 : 0], offset);
        mask.data.set([255, 255, 255, 255], offset);
        const boundary = visible && (x === 2 || x === 11 || y === 2 || y === 11);
        const generatedColor = boundary ? [15, 18, 24, 255] : [25, 30, 40, 255];
        for (let oy = 0; oy < 2; oy += 1) {
          for (let ox = 0; ox < 2; ox += 1) {
            candidate.data.set(
              generatedColor,
              ((y * 2 + oy) * candidate.width + x * 2 + ox) * 4,
            );
          }
        }
      }
    }
    const sourcePath = path.join(root, "dark-source.png");
    const candidatePath = path.join(root, "dark-candidate.png");
    const maskPath = path.join(root, "dark-mask.png");
    await Promise.all([
      writeFile(sourcePath, PNG.sync.write(source)),
      writeFile(candidatePath, PNG.sync.write(candidate)),
      writeFile(maskPath, PNG.sync.write(mask)),
    ]);
    const store = new CharacterArtifactStore(root);
    const result = await store.constrainTextureEditResult(
      "job",
      sourcePath,
      candidatePath,
      maskPath,
      {
        canvasWidth: 28,
        canvasHeight: 28,
        offsetX: 0,
        offsetY: 0,
        scale: 2,
        sourceWidth: 14,
        sourceHeight: 14,
      },
      "深色纹理",
      "preserve-dark-edge",
      "locked",
    );
    const output = PNG.sync.read(await readFile(result.path));
    const edge = (2 * output.width + 2) * 4;

    expect([...output.data.subarray(edge, edge + 3)]).toEqual([15, 18, 24]);
  });

  it("uses gray UV masks only as edit positions without blending the old RGB", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const source = new PNG({ width: 4, height: 1 });
    const candidate = new PNG({ width: 8, height: 2 });
    const mask = new PNG({ width: 4, height: 1 });
    const originalAlphas = [77, 96, 128, 200];
    const maskAlphas = [0, 64, 128, 255];
    for (let x = 0; x < source.width; x += 1) {
      const offset = x * 4;
      source.data.set([30, 60, 120, originalAlphas[x]!], offset);
      mask.data.set([255, 255, 255, maskAlphas[x]!], offset);
      for (let y = 0; y < candidate.height; y += 1) {
        for (let ox = 0; ox < 2; ox += 1) {
          candidate.data.set([230, 50, 145, 255], (y * candidate.width + x * 2 + ox) * 4);
        }
      }
    }
    const sourcePath = path.join(root, "gray-mask-source.png");
    const candidatePath = path.join(root, "gray-mask-candidate.png");
    const maskPath = path.join(root, "gray-mask.png");
    await Promise.all([
      writeFile(sourcePath, PNG.sync.write(source)),
      writeFile(candidatePath, PNG.sync.write(candidate)),
      writeFile(maskPath, PNG.sync.write(mask)),
    ]);
    const store = new CharacterArtifactStore(root);

    const result = await store.constrainTextureEditResult(
      "job",
      sourcePath,
      candidatePath,
      maskPath,
      {
        canvasWidth: 8,
        canvasHeight: 2,
        offsetX: 0,
        offsetY: 0,
        scale: 2,
        sourceWidth: 4,
        sourceHeight: 1,
      },
      "袖子纹理",
      "gray-mask-rgb",
      "locked",
    );
    const output = PNG.sync.read(await readFile(result.path));

    expect([...output.data.subarray(0, 4)]).toEqual([30, 60, 120, 77]);
    for (let x = 1; x < output.width; x += 1) {
      const offset = x * 4;
      expect([...output.data.subarray(offset, offset + 3)]).toEqual([230, 50, 145]);
      expect(output.data[offset + 3]).toBe(originalAlphas[x]);
    }
  });

  it("writes generated transparency, preserves pixels outside the mask and ignores alpha expansion", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const source = new PNG({ width: 8, height: 8 });
    const candidate = new PNG({ width: 16, height: 16 });
    const mask = new PNG({ width: 8, height: 8 });
    for (let y = 0; y < source.height; y += 1) {
      for (let x = 0; x < source.width; x += 1) {
        const offset = (y * source.width + x) * 4;
        const sourceAlpha = x === 7 ? 96 : 255;
        source.data.set([40, 80, 160, sourceAlpha], offset);
        mask.data.set([255, 255, 255, x === 0 && y === 7 ? 0 : 255], offset);
        for (let oy = 0; oy < 2; oy += 1) {
          for (let ox = 0; ox < 2; ox += 1) {
            const candidateOffset = ((y * 2 + oy) * candidate.width + x * 2 + ox) * 4;
            candidate.data.set([210, 90, 130, x < 4 ? 0 : 255], candidateOffset);
          }
        }
      }
    }
    const sourcePath = path.join(root, "alpha-source.png");
    const candidatePath = path.join(root, "alpha-candidate.png");
    const maskPath = path.join(root, "alpha-mask.png");
    await Promise.all([
      writeFile(sourcePath, PNG.sync.write(source)),
      writeFile(candidatePath, PNG.sync.write(candidate)),
      writeFile(maskPath, PNG.sync.write(mask)),
    ]);
    const store = new CharacterArtifactStore(root);

    const result = await store.constrainTextureEditResult(
      "job",
      sourcePath,
      candidatePath,
      maskPath,
      {
        canvasWidth: 16,
        canvasHeight: 16,
        offsetX: 0,
        offsetY: 0,
        scale: 2,
        sourceWidth: 8,
        sourceHeight: 8,
      },
      "腿部纹理",
      "alpha-shrink",
    );
    const output = PNG.sync.read(await readFile(result.path));
    const removed = (2 * output.width + 2) * 4;
    const expansionAttempt = (2 * output.width + 7) * 4;
    const protectedPixel = (7 * output.width) * 4;

    expect(output.data[removed + 3]).toBe(0);
    expect(output.data[expansionAttempt + 3]).toBe(96);
    expect([...output.data.subarray(protectedPixel, protectedPixel + 4)]).toEqual([40, 80, 160, 255]);
    expect(result.quality.alphaChangedPixelRatio).toBeGreaterThan(0.4);
    expect(result.quality.removedAlphaRatio).toBeGreaterThan(0.4);
    expect(result.quality.ignoredExpansionPixelRatio).toBeGreaterThan(0);
  });

  it("inpaints a fixed 4:3 eye crop and maps a differently sized fal result back exactly", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const reference = new PNG({ width: 120, height: 180 });
    const mask = new PNG({ width: 120, height: 180 });
    for (let offset = 0; offset < reference.data.length; offset += 4) {
      reference.data.set([40, 60, 80, 171], offset);
      mask.data.set([0, 0, 0, 255], offset);
    }
    for (let y = 42; y < 48; y += 1) {
      for (const startX of [45, 69]) {
        for (let x = startX; x < startX + 6; x += 1) {
          mask.data.set([255, 255, 255, 255], (y * mask.width + x) * 4);
        }
      }
    }
    const referencePath = path.join(root, "reference.png");
    const maskPath = path.join(root, "mask.png");
    await writeFile(referencePath, PNG.sync.write(reference));
    await writeFile(maskPath, PNG.sync.write(mask));
    const store = new CharacterArtifactStore(root);

    const crop = await store.createEyeEditCrop("job", referencePath, maskPath, "crop");
    const cropImage = PNG.sync.read(await readFile(crop.imagePath));
    const cropMask = PNG.sync.read(await readFile(crop.maskPath));
    expect([cropImage.width, cropImage.height]).toEqual([1024, 768]);
    expect([cropMask.width, cropMask.height]).toEqual([1024, 768]);
    expect(crop.source.width / crop.source.height).toBeCloseTo(4 / 3, 2);

    // fal is allowed to return another resolution as long as it keeps 4:3.
    const falResult = new PNG({ width: 800, height: 600 });
    for (let offset = 0; offset < falResult.data.length; offset += 4) {
      falResult.data.set([210, 20, 120, 255], offset);
    }
    const falPath = path.join(root, "fal.png");
    await writeFile(falPath, PNG.sync.write(falResult));
    const outputPath = await store.constrainEyeCropEdit(
      "job",
      referencePath,
      falPath,
      maskPath,
      crop,
      "safe-eye",
    );
    const output = PNG.sync.read(await readFile(outputPath));

    expect([...output.data.subarray(0, 4)]).toEqual([40, 60, 80, 171]);
    const eyeOffset = (44 * output.width + 47) * 4;
    expect([...output.data.subarray(eyeOffset, eyeOffset + 3)]).not.toEqual([40, 60, 80]);
    expect(output.data[eyeOffset + 3]).toBe(171);
  });

  it("builds a part mask from both ArtMesh triangles and the original atlas alpha", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const reference = new PNG({ width: 100, height: 100 });
    const silhouette = new PNG({ width: 60, height: 60 });
    for (let offset = 0; offset < reference.data.length; offset += 4) {
      reference.data.set([80, 90, 100, 255], offset);
    }
    for (let y = 10; y < 50; y += 1) {
      for (let x = 10; x < 50; x += 1) {
        silhouette.data.set([255, 255, 255, 255], (y * silhouette.width + x) * 4);
      }
    }
    const referencePath = path.join(root, "reference.png");
    const maskDirectory = path.join(root, "template", "masks");
    await mkdir(maskDirectory, { recursive: true });
    await writeFile(referencePath, PNG.sync.write(reference));
    await writeFile(path.join(maskDirectory, "part.png"), PNG.sync.write(silhouette));
    const store = new CharacterArtifactStore(root);
    const slot: MasterTemplateSlot = {
      id: "part",
      name: "Part",
      drawableId: "D_PART",
      partId: "Part",
      group: "face",
      source: "visible",
      required: false,
      zIndex: 1,
      atlas: { texture: 0, x: 20, y: 20, width: 60, height: 60 },
      anchor: { x: 50, y: 50 },
      canvasBounds: { x: -0.5, y: -0.5, width: 1, height: 1 },
      mesh: {
        vertices: [-0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5],
        uvs: [0.2, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
        indices: [0, 1, 2, 0, 2, 3],
      },
      silhouetteMask: "masks/part.png",
      completionPrompt: "part",
    };

    const maskPath = await store.createExactSlotEditMask(
      "job",
      referencePath,
      path.join(root, "template"),
      { width: 100, height: 100, originX: 50, originY: 50, pixelsPerUnit: 50 },
      [{ width: 100, height: 100 }],
      [slot],
      "exact",
    );
    const mask = PNG.sync.read(await readFile(maskPath));

    expect(mask.data[(27 * 100 + 27) * 4]).toBe(0);
    expect(mask.data[(50 * 100 + 50) * 4]).toBeGreaterThan(0);
    expect(mask.data[(76 * 100 + 76) * 4]).toBe(0);
  });

  it("removes pixels hidden by a higher unselected ArtMesh from the safe edit mask", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "character-artifacts-"));
    roots.push(root);
    const reference = new PNG({ width: 100, height: 100 });
    const selectedSilhouette = new PNG({ width: 60, height: 60 });
    const occluderSilhouette = new PNG({ width: 60, height: 60 });
    for (let offset = 0; offset < reference.data.length; offset += 4) {
      reference.data.set([80, 90, 100, 255], offset);
    }
    for (let y = 8; y < 52; y += 1) {
      for (let x = 8; x < 52; x += 1) {
        selectedSilhouette.data.set(
          [255, 255, 255, 255],
          (y * selectedSilhouette.width + x) * 4,
        );
      }
    }
    for (let y = 24; y < 36; y += 1) {
      for (let x = 24; x < 36; x += 1) {
        occluderSilhouette.data.set(
          [255, 255, 255, 255],
          (y * occluderSilhouette.width + x) * 4,
        );
      }
    }
    const referencePath = path.join(root, "reference.png");
    const maskDirectory = path.join(root, "template", "masks");
    await mkdir(maskDirectory, { recursive: true });
    await writeFile(referencePath, PNG.sync.write(reference));
    await writeFile(path.join(maskDirectory, "selected.png"), PNG.sync.write(selectedSilhouette));
    await writeFile(path.join(maskDirectory, "occluder.png"), PNG.sync.write(occluderSilhouette));
    const baseSlot: MasterTemplateSlot = {
      id: "selected",
      name: "Selected",
      drawableId: "D_SELECTED",
      partId: "PartSelected",
      group: "face",
      source: "visible",
      required: false,
      zIndex: 1,
      atlas: { texture: 0, x: 20, y: 20, width: 60, height: 60 },
      anchor: { x: 50, y: 50 },
      canvasBounds: { x: -0.5, y: -0.5, width: 1, height: 1 },
      mesh: {
        vertices: [-0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5],
        uvs: [0.2, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
        indices: [0, 1, 2, 0, 2, 3],
      },
      silhouetteMask: "masks/selected.png",
      completionPrompt: "selected",
    };
    const occluder: MasterTemplateSlot = {
      ...baseSlot,
      id: "occluder",
      name: "Occluder",
      drawableId: "D_OCCLUDER",
      partId: "PartOccluder",
      zIndex: 2,
      silhouetteMask: "masks/occluder.png",
    };
    const store = new CharacterArtifactStore(root);

    const maskPath = await store.createExactSlotEditMask(
      "job",
      referencePath,
      path.join(root, "template"),
      { width: 100, height: 100, originX: 50, originY: 50, pixelsPerUnit: 50 },
      [{ width: 100, height: 100 }],
      [baseSlot],
      "occlusion-safe",
      [baseSlot, occluder],
    );
    const mask = PNG.sync.read(await readFile(maskPath));

    expect(mask.data[(50 * 100 + 50) * 4]).toBe(0);
    expect(mask.data[(42 * 100 + 42) * 4]).toBeGreaterThan(0);
  });
});

describe("generationCacheKey", () => {
  it("is stable for equal nested inputs regardless of object key order", () => {
    expect(generationCacheKey("layer", { b: 2, a: { y: 1, x: 0 } })).toBe(
      generationCacheKey("layer", { a: { x: 0, y: 1 }, b: 2 }),
    );
    expect(generationCacheKey("layer", { a: 1 })).not.toBe(
      generationCacheKey("edit", { a: 1 }),
    );
  });
});
