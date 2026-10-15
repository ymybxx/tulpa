import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { PNG } from "pngjs";

import {
  textureEditBatchPromptRegions,
  type MasterTemplateCanvas,
  type MasterTemplateSlot,
  type TextureEditBatchPlan,
  type TextureEditImageSize,
  type TextureEditQualityReport,
  type TextureEditWorkLayout,
} from "../../src/shared/character-generation";
import type { TextureEditUnit } from "../../src/shared/contracts";
import { createMaskedTexturePreview } from "../masked-texture-preview";
import { decodePng } from "../png-reader";
import {
  paintTextureRegionMarker,
} from "./model-locator";

type Fetcher = (input: string, init?: RequestInit) => Promise<Response>;

export interface MaskedEditCrop {
  imagePath: string;
  maskPath: string;
  source: { x: number; y: number; width: number; height: number };
  output: { width: number; height: number };
}

export type EyeEditCrop = MaskedEditCrop;

export interface TextureEditBatchWorkAsset {
  imagePath: string;
  maskPath: string;
  imageSize: TextureEditImageSize;
  plan: TextureEditBatchPlan;
  items: Record<string, {
    sourcePath: string;
    layout: TextureEditWorkLayout;
  }>;
}

export class CharacterArtifactStore {
  constructor(
    readonly root: string,
    private readonly fetcher: Fetcher = fetch,
    private readonly maxBytes = 25 * 1024 * 1024,
  ) {}

  async download(jobId: string, url: string, name: string): Promise<string> {
    const parsed = new URL(url);
    const safeDataUrl = parsed.protocol === "data:" &&
      /^data:image\/(?:png|jpeg|webp);base64,/i.test(url);
    if (parsed.protocol !== "https:" && !safeDataUrl) {
      throw new Error("只允许读取图片服务返回的 HTTPS 或 base64 图片资源");
    }
    const response = await this.fetcher(url, { redirect: "follow" });
    if (!response.ok) throw new Error(`下载生成素材失败 (${response.status})`);
    const declared = Number(response.headers.get("content-length") ?? "0");
    if (declared > this.maxBytes) throw new Error("生成素材过大");
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength > this.maxBytes) throw new Error("生成素材过大");
    const directory = this.jobDirectory(jobId);
    await mkdir(directory, { recursive: true });
    const extension = imageExtension(response.headers.get("content-type"), parsed.pathname);
    const safeName = name.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "asset";
    const target = path.join(directory, `${safeName}${extension}`);
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, bytes, { mode: 0o600 });
    await rename(temporary, target);
    return target;
  }

  async snapshotImage(jobId: string, sourcePath: string, name: string): Promise<string> {
    const bytes = await readFile(sourcePath);
    if (bytes.byteLength === 0 || bytes.byteLength > this.maxBytes) {
      throw new Error("参考图片为空或文件过大");
    }
    const extension = path.extname(sourcePath).toLowerCase();
    if (![".png", ".jpg", ".jpeg", ".webp"].includes(extension)) {
      throw new Error("参考图片只支持 PNG、JPEG 或 WebP");
    }
    const directory = this.jobDirectory(jobId);
    await mkdir(directory, { recursive: true });
    const safeName = name.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "reference";
    const target = path.join(directory, `${safeName}${extension}`);
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, bytes, { mode: 0o600 });
    await rename(temporary, target);
    return target;
  }

  async applyMask(jobId: string, sourcePath: string, maskPath: string, name: string): Promise<string> {
    const source = PNG.sync.read(await readFile(sourcePath));
    const mask = PNG.sync.read(await readFile(maskPath));
    if (source.width <= 0 || source.height <= 0 || mask.width <= 0 || mask.height <= 0) {
      throw new Error("分割蒙版尺寸无效");
    }
    const output = new PNG({ width: source.width, height: source.height });
    const useMaskAlpha = shouldUseMaskAlpha(mask);
    for (let y = 0; y < source.height; y += 1) {
      const maskY = Math.min(mask.height - 1, Math.floor((y + 0.5) * mask.height / source.height));
      for (let x = 0; x < source.width; x += 1) {
        const maskX = Math.min(mask.width - 1, Math.floor((x + 0.5) * mask.width / source.width));
        const sourceOffset = (y * source.width + x) * 4;
        const maskOffset = (maskY * mask.width + maskX) * 4;
        const strength = useMaskAlpha
          ? mask.data[maskOffset + 3]!
          : Math.round(
            mask.data[maskOffset]! * 0.2126 +
            mask.data[maskOffset + 1]! * 0.7152 +
            mask.data[maskOffset + 2]! * 0.0722,
          );
        output.data[sourceOffset] = source.data[sourceOffset]!;
        output.data[sourceOffset + 1] = source.data[sourceOffset + 1]!;
        output.data[sourceOffset + 2] = source.data[sourceOffset + 2]!;
        output.data[sourceOffset + 3] = Math.round(source.data[sourceOffset + 3]! * strength / 255);
      }
    }
    const directory = this.jobDirectory(jobId);
    await mkdir(directory, { recursive: true });
    const safeName = name.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "asset";
    const target = path.join(directory, `${safeName}.png`);
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, PNG.sync.write(output), { mode: 0o600 });
    await rename(temporary, target);
    return target;
  }

  async assertSameImageSize(referencePath: string, candidatePath: string): Promise<void> {
    const [reference, candidate] = await Promise.all([
      readFile(referencePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(candidatePath).then((bytes) => PNG.sync.read(bytes)),
    ]);
    if (reference.width !== candidate.width || reference.height !== candidate.height) {
      throw new Error(
        `局部补全画布尺寸发生变化：应为 ${reference.width}×${reference.height}，` +
        `实际为 ${candidate.width}×${candidate.height}`,
      );
    }
  }

  async imageSize(imagePath: string): Promise<{ width: number; height: number }> {
    const image = PNG.sync.read(await readFile(imagePath));
    if (image.width <= 0 || image.height <= 0) throw new Error("图片尺寸无效");
    return { width: image.width, height: image.height };
  }

  async createSlotEditMask(
    jobId: string,
    referencePath: string,
    canvas: MasterTemplateCanvas,
    slots: MasterTemplateSlot[],
    name: string,
  ): Promise<string> {
    if (slots.length === 0) throw new Error("没有可修改的眼睛槽位");
    const reference = PNG.sync.read(await readFile(referencePath));
    const coverage = new Uint8Array(reference.width * reference.height);
    for (const slot of slots) {
      const vertices: Point[] = [];
      for (let index = 0; index < slot.mesh.vertices.length; index += 2) {
        vertices.push({
          x: (canvas.originX + slot.mesh.vertices[index]! * canvas.pixelsPerUnit) /
            canvas.width * reference.width,
          y: (canvas.originY - slot.mesh.vertices[index + 1]! * canvas.pixelsPerUnit) /
            canvas.height * reference.height,
        });
      }
      for (let index = 0; index < slot.mesh.indices.length; index += 3) {
        fillCoverageTriangle(
          coverage,
          reference.width,
          reference.height,
          vertices[slot.mesh.indices[index]!]!,
          vertices[slot.mesh.indices[index + 1]!]!,
          vertices[slot.mesh.indices[index + 2]!]!,
        );
      }
    }
    const distance = interiorDistance(coverage, reference.width, reference.height, 4);
    const mask = new PNG({ width: reference.width, height: reference.height });
    let editablePixels = 0;
    for (let index = 0; index < coverage.length; index += 1) {
      const value = distance[index] === 0 ? 0 : Math.min(255, (distance[index]! - 1) * 85);
      const offset = index * 4;
      mask.data[offset] = value;
      mask.data[offset + 1] = value;
      mask.data[offset + 2] = value;
      mask.data[offset + 3] = 255;
      if (value > 0) editablePixels += 1;
    }
    if (editablePixels === 0) throw new Error("眼睛网格过小，无法建立安全编辑区");
    return this.writePng(jobId, mask, name);
  }

  /**
   * Projects the original atlas alpha of selected Drawables into the neutral
   * render. Unlike a rectangular/semantic mask, this is bounded by both the
   * Cubism triangles and the installed texture's visible silhouette.
   */
  async createExactSlotEditMask(
    jobId: string,
    referencePath: string,
    templateRoot: string,
    canvas: MasterTemplateCanvas,
    atlases: Array<{ width: number; height: number }>,
    slots: MasterTemplateSlot[],
    name: string,
    sceneSlots: MasterTemplateSlot[] = slots,
  ): Promise<string> {
    if (slots.length === 0) throw new Error("所选部位没有可修改的可见网格");
    const reference = PNG.sync.read(await readFile(referencePath));
    const strengths = new Uint8Array(reference.width * reference.height);
    const selected = new Set(slots.map((slot) => slot.id));
    for (const slot of [...sceneSlots].sort((left, right) => left.zIndex - right.zIndex)) {
      if (slot.source !== "visible") continue;
      const atlas = atlases[slot.atlas.texture];
      if (!atlas) throw new Error(`槽位 ${slot.id} 引用了不存在的纹理图集`);
      const silhouette = PNG.sync.read(
        await readFile(path.resolve(templateRoot, slot.silhouetteMask)),
      );
      if (silhouette.width !== slot.atlas.width || silhouette.height !== slot.atlas.height) {
        throw new Error(`槽位 ${slot.id} 的原始轮廓蒙版尺寸不一致`);
      }
      const layer = new Uint8Array(reference.width * reference.height);
      const destination = pairs(slot.mesh.vertices).map(([x, y]) => ({
        x: (canvas.originX + x * canvas.pixelsPerUnit) / canvas.width * reference.width,
        y: (canvas.originY - y * canvas.pixelsPerUnit) / canvas.height * reference.height,
      }));
      const source = pairs(slot.mesh.uvs).map(([u, v]) => ({
        x: u * atlas.width - slot.atlas.x,
        y: (1 - v) * atlas.height - slot.atlas.y,
      }));
      for (let index = 0; index < slot.mesh.indices.length; index += 3) {
        const i0 = slot.mesh.indices[index]!;
        const i1 = slot.mesh.indices[index + 1]!;
        const i2 = slot.mesh.indices[index + 2]!;
        rasterMaskTriangle(
          layer,
          reference.width,
          reference.height,
          silhouette,
          [destination[i0]!, destination[i1]!, destination[i2]!],
          [source[i0]!, source[i1]!, source[i2]!],
        );
      }
      for (let index = 0; index < strengths.length; index += 1) {
        const alpha = layer[index]! / 255;
        if (alpha <= 0) continue;
        strengths[index] = selected.has(slot.id)
          ? Math.round(255 * alpha + strengths[index]! * (1 - alpha))
          : Math.round(strengths[index]! * (1 - alpha));
      }
    }
    const coverage = strengths.map((value) => value > 24 ? 1 : 0);
    const distance = interiorDistance(coverage, reference.width, reference.height, 3);
    const mask = new PNG({ width: reference.width, height: reference.height });
    let editablePixels = 0;
    for (let index = 0; index < strengths.length; index += 1) {
      // Keep the semi-transparent atlas edge untouched. It is the most common
      // source of visible seams after AI edits.
      const feather = distance[index] === 0 ? 0 : Math.min(255, (distance[index]! - 1) * 128);
      const value = Math.min(strengths[index]!, feather);
      const offset = index * 4;
      mask.data[offset] = value;
      mask.data[offset + 1] = value;
      mask.data[offset + 2] = value;
      mask.data[offset + 3] = 255;
      if (value > 0) editablePixels += 1;
    }
    if (editablePixels < 16) throw new Error("所选部位的安全编辑区域过小");
    return this.writePng(jobId, mask, name);
  }

  async constrainMaskedEdit(
    jobId: string,
    referencePath: string,
    candidatePath: string,
    maskPath: string,
    name: string,
  ): Promise<string> {
    const [reference, candidate, mask] = await Promise.all([
      readFile(referencePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(candidatePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(maskPath).then((bytes) => PNG.sync.read(bytes)),
    ]);
    for (const image of [candidate, mask]) {
      if (image.width !== reference.width || image.height !== reference.height) {
        throw new Error("眼睛生成结果的画布尺寸发生变化");
      }
    }
    const output = new PNG({ width: reference.width, height: reference.height });
    let editablePixels = 0;
    let changedPixels = 0;
    let difference = 0;
    for (let index = 0; index < reference.width * reference.height; index += 1) {
      const offset = index * 4;
      const strength = mask.data[offset]! / 255;
      let pixelDifference = 0;
      for (let channel = 0; channel < 3; channel += 1) {
        const original = reference.data[offset + channel]!;
        const generated = candidate.data[offset + channel]!;
        pixelDifference += Math.abs(generated - original);
        output.data[offset + channel] = Math.round(original * (1 - strength) + generated * strength);
      }
      // The generated image is never allowed to change geometry or transparency.
      output.data[offset + 3] = reference.data[offset + 3]!;
      if (strength > 0) {
        editablePixels += 1;
        difference += pixelDifference / 3;
        if (pixelDifference >= 18) changedPixels += 1;
      }
    }
    assertEffectiveEyeEdit(editablePixels, changedPixels, difference);
    return this.writePng(jobId, output, name);
  }

  async createEyeEditCrop(
    jobId: string,
    referencePath: string,
    maskPath: string,
    name: string,
  ): Promise<EyeEditCrop> {
    const [reference, mask] = await Promise.all([
      readFile(referencePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(maskPath).then((bytes) => PNG.sync.read(bytes)),
    ]);
    if (reference.width !== mask.width || reference.height !== mask.height) {
      throw new Error("眼睛编辑蒙版与母版画布尺寸不一致");
    }
    const bounds = maskBounds(mask);
    if (!bounds) throw new Error("眼睛编辑蒙版为空");
    const source = eyeContextCrop(reference.width, reference.height, bounds);
    const output = { width: 1024, height: 768 } as const;
    const croppedReference = resizeRegion(reference, source, output);
    const croppedMask = resizeRegion(mask, source, output);
    const imagePath = await this.writePng(jobId, croppedReference, `${name}-image`);
    const croppedMaskPath = await this.writePng(jobId, croppedMask, `${name}-mask`);
    return { imagePath, maskPath: croppedMaskPath, source, output };
  }

  async createMaskedEditCrop(
    jobId: string,
    referencePath: string,
    maskPath: string,
    name: string,
  ): Promise<MaskedEditCrop> {
    const [reference, mask] = await Promise.all([
      readFile(referencePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(maskPath).then((bytes) => PNG.sync.read(bytes)),
    ]);
    if (reference.width !== mask.width || reference.height !== mask.height) {
      throw new Error("部位编辑蒙版与模型画布尺寸不一致");
    }
    const bounds = maskBounds(mask);
    if (!bounds) throw new Error("部位编辑蒙版为空");
    const output = editOutputSize(bounds);
    const source = contextCrop(reference.width, reference.height, bounds, output.width / output.height);
    const imagePath = await this.writePng(
      jobId,
      resizeRegion(reference, source, output),
      `${name}-image`,
    );
    const croppedMaskPath = await this.writePng(
      jobId,
      resizeRegion(mask, source, output),
      `${name}-mask`,
    );
    return { imagePath, maskPath: croppedMaskPath, source, output };
  }

  async createTextureEditBatchWorkCanvas(
    jobId: string,
    entries: ReadonlyArray<{
      atlasPath: string;
      unitMaskPath: string;
      modelMaskPath: string;
      unit: TextureEditUnit;
    }>,
    plan: TextureEditBatchPlan,
    name: string,
    /** The first canvas's result and its plan, shown read-only in `plan.anchor`. */
    anchor?: { imagePath: string; plan: TextureEditBatchPlan } | null,
  ): Promise<TextureEditBatchWorkAsset> {
    const canvas = new PNG({ width: plan.canvasWidth, height: plan.canvasHeight });
    const mask = new PNG({ width: plan.canvasWidth, height: plan.canvasHeight });
    fillPng(canvas, [246, 247, 248, 255]);
    fillPng(mask, [255, 255, 255, 255]);
    const entryById = new Map(entries.map((entry) => [entry.unit.id, entry]));
    const atlasCache = new Map<string, PNG>();
    const items: TextureEditBatchWorkAsset["items"] = {};
    for (const layout of plan.items) {
      const entry = entryById.get(layout.textureUnitId);
      if (!entry) throw new Error(`排版引用了不存在的纹理 ${layout.textureUnitId}`);
      let atlas = atlasCache.get(entry.atlasPath);
      if (!atlas) {
        atlas = decodePng(await readFile(entry.atlasPath));
        atlasCache.set(entry.atlasPath, atlas);
      }
      const unitMask = PNG.sync.read(await readFile(entry.unitMaskPath));
      const bounds = entry.unit.bounds;
      if (
        bounds.x < 0 || bounds.y < 0 || bounds.width !== layout.sourceWidth ||
        bounds.height !== layout.sourceHeight ||
        bounds.x + bounds.width > atlas.width || bounds.y + bounds.height > atlas.height ||
        unitMask.width !== bounds.width || unitMask.height !== bounds.height
      ) throw new Error(`纹理 ${entry.unit.label} 与批量排版尺寸不一致`);
      const source = new PNG({ width: bounds.width, height: bounds.height });
      for (let y = 0; y < bounds.height; y += 1) {
        for (let x = 0; x < bounds.width; x += 1) {
          const atlasOffset = ((bounds.y + y) * atlas.width + bounds.x + x) * 4;
          const localOffset = (y * bounds.width + x) * 4;
          source.data.set(atlas.data.subarray(atlasOffset, atlasOffset + 4), localOffset);
          if (layout.scale < 1) continue;
          const alpha = atlas.data[atlasOffset + 3]! / 255;
          const editable = unitMask.data[localOffset + 3]!;
          for (let oy = 0; oy < layout.scale; oy += 1) {
            for (let ox = 0; ox < layout.scale; ox += 1) {
              const targetX = layout.x + x * layout.scale + ox;
              const targetY = layout.y + y * layout.scale + oy;
              const targetOffset = (targetY * plan.canvasWidth + targetX) * 4;
              for (let channel = 0; channel < 3; channel += 1) {
                canvas.data[targetOffset + channel] = Math.round(
                  atlas.data[atlasOffset + channel]! * alpha + 246 * (1 - alpha),
                );
                mask.data[targetOffset + channel] = 255;
              }
              canvas.data[targetOffset + 3] = 255;
              mask.data[targetOffset + 3] = 255 - editable;
            }
          }
        }
      }
      if (layout.scale < 1) paintShrunkTexture(canvas, mask, source, unitMask, layout, plan.canvasWidth);
      items[layout.textureUnitId] = {
        sourcePath: await this.writePng(jobId, source, `${name}-${layout.textureUnitId}-source`),
        layout: {
          canvasWidth: plan.canvasWidth,
          canvasHeight: plan.canvasHeight,
          offsetX: layout.x,
          offsetY: layout.y,
          scale: layout.scale,
          sourceWidth: layout.sourceWidth,
          sourceHeight: layout.sourceHeight,
        },
      };
    }
    if (plan.anchor && anchor) {
      // The mask stays opaque here, so the image model sees it but cannot change it.
      paintTextureAnchor(canvas, decodePng(await readFile(anchor.imagePath)), anchor.plan, plan.anchor);
    }
    const promptRegions = textureEditBatchPromptRegions(plan);
    for (let index = 0; index < promptRegions.length; index += 1) {
      const region = promptRegions[index]!;
      paintTextureRegionMarker(canvas, index, region.key, {
        x: region.x,
        y: region.y,
        width: region.workWidth,
        height: region.workHeight,
      }, plan.gutter);
    }
    return {
      imagePath: await this.writePng(jobId, canvas, `${name}-work`),
      maskPath: await this.writePng(jobId, mask, `${name}-mask`),
      imageSize: plan.imageSize,
      plan,
      items,
    };
  }

  async constrainTextureEditResult(
    jobId: string,
    sourcePath: string,
    candidatePath: string,
    unitMaskPath: string,
    layout: TextureEditWorkLayout,
    label: string,
    name: string,
    alphaPolicy: "locked" | "shrink-only" = "shrink-only",
  ): Promise<{
    path: string;
    sourcePreviewPath: string;
    previewPath: string;
    quality: TextureEditQualityReport;
  }> {
    const [source, candidate, unitMask] = await Promise.all([
      readFile(sourcePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(candidatePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(unitMaskPath).then((bytes) => PNG.sync.read(bytes)),
    ]);
    if (
      source.width !== layout.sourceWidth || source.height !== layout.sourceHeight ||
      unitMask.width !== source.width || unitMask.height !== source.height
    ) throw new Error(`${label}原始纹理与 UV 蒙版尺寸不一致`);
    const expectedRatio = layout.canvasWidth / layout.canvasHeight;
    const actualRatio = candidate.width / candidate.height;
    if (Math.abs(actualRatio / expectedRatio - 1) > 0.03) {
      throw new Error(`${label}生成结果比例异常，未写回模型`);
    }
    const output = new PNG({ width: source.width, height: source.height });
    output.data.set(source.data);
    const normalizedMask = new PNG({ width: source.width, height: source.height });
    let editablePixels = 0;
    let changedPixels = 0;
    let alphaChangedPixels = 0;
    let ignoredExpansionPixels = 0;
    let originalAlphaMass = 0;
    let removedAlphaMass = 0;
    let totalDifference = 0;
    for (let y = 0; y < source.height; y += 1) {
      for (let x = 0; x < source.width; x += 1) {
        const offset = (y * source.width + x) * 4;
        const maskValue = unitMask.data[offset + 3]!;
        const editable = maskValue > 0;
        const binaryMaskValue = editable ? 255 : 0;
        normalizedMask.data.set([binaryMaskValue, binaryMaskValue, binaryMaskValue, 255], offset);
        if (!editable) continue;
        const workX = layout.offsetX + (x + 0.5) * layout.scale;
        const workY = layout.offsetY + (y + 0.5) * layout.scale;
        const candidateX = workX / layout.canvasWidth * candidate.width - 0.5;
        const candidateY = workY / layout.canvasHeight * candidate.height - 0.5;
        const generated = bilinearPixel(candidate, candidateX, candidateY);
        const originalAlpha = source.data[offset + 3]!;
        const generatedAlpha = generated[3]!;
        const constrainedAlpha = alphaPolicy === "locked"
          ? originalAlpha
          : Math.min(originalAlpha, generatedAlpha);
        const alphaDifference = originalAlpha - constrainedAlpha;
        let difference = 0;
        for (let channel = 0; channel < 3; channel += 1) {
          const original = source.data[offset + channel]!;
          difference += Math.abs(generated[channel]! - original);
          // The UV mask only determines whether this pixel is editable. Its
          // antialiasing/gray value must never blend the old texture colour
          // back into the generated material.
          output.data[offset + channel] = generated[channel]!;
        }
        output.data[offset + 3] = constrainedAlpha;
        editablePixels += 1;
        originalAlphaMass += originalAlpha;
        removedAlphaMass += alphaDifference;
        totalDifference += difference / 3;
        if (alphaDifference >= 4) alphaChangedPixels += 1;
        if (alphaPolicy === "shrink-only" && generatedAlpha > originalAlpha + 4) {
          ignoredExpansionPixels += 1;
        }
        if (difference >= 18 || alphaDifference >= 4) changedPixels += 1;
      }
    }
    if (editablePixels === 0) throw new Error(`${label}的 UV 蒙版为空`);
    cleanTextureAlphaEdges(output, normalizedMask);
    const changedPixelRatio = changedPixels / editablePixels;
    const alphaChangedPixelRatio = alphaChangedPixels / editablePixels;
    const removedAlphaRatio = originalAlphaMass > 0 ? removedAlphaMass / originalAlphaMass : 0;
    const ignoredExpansionPixelRatio = ignoredExpansionPixels / editablePixels;
    const meanDifference = totalDifference / editablePixels;
    const issues: string[] = [];
    if (changedPixelRatio < 0.02 || (meanDifference < 1.5 && alphaChangedPixelRatio < 0.02)) {
      issues.push(`修改幅度很小（修改像素 ${(changedPixelRatio * 100).toFixed(1)}%）`);
    }
    if (removedAlphaRatio > 0.95) {
      issues.push(`结果几乎清空了整个纹理（透明度减少 ${(removedAlphaRatio * 100).toFixed(1)}%）`);
    }
    if (ignoredExpansionPixelRatio > 0.01) {
      issues.push(`AI 尝试扩大 ${(ignoredExpansionPixelRatio * 100).toFixed(1)}% 的像素，已按缩减模式忽略`);
    }
    issues.push(...inspectCleanPartTexture(source, output, normalizedMask, label, "restyle"));
    const sharpnessRatio = textureSharpnessRatio(source, output, normalizedMask);
    if (sharpnessRatio < 0.58) {
      issues.push(`结果可能比原图模糊（相对清晰度 ${(sharpnessRatio * 100).toFixed(0)}%）`);
    }
    const quality: TextureEditQualityReport = {
      accepted: issues.length === 0,
      issues,
      changedPixelRatio,
      alphaChangedPixelRatio,
      removedAlphaRatio,
      ignoredExpansionPixelRatio,
      sharpnessRatio,
    };
    const [patchPath, sourcePreviewPath, previewPath] = await Promise.all([
      this.writePng(jobId, output, name),
      this.writePng(
        jobId,
        createMaskedTexturePreview(source, unitMask),
        `${name}-source-preview`,
      ),
      this.writePng(
        jobId,
        createMaskedTexturePreview(output, unitMask),
        `${name}-preview`,
      ),
    ]);
    return { path: patchPath, sourcePreviewPath, previewPath, quality };
  }

  async createTextureEditPreviews(
    jobId: string,
    sourcePath: string,
    editedPatchPath: string,
    unitMaskPath: string,
    name: string,
  ): Promise<{ sourcePreviewPath: string; editedPreviewPath: string }> {
    const [source, edited, unitMask] = await Promise.all([
      readFile(sourcePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(editedPatchPath).then((bytes) => PNG.sync.read(bytes)),
      readFile(unitMaskPath).then((bytes) => PNG.sync.read(bytes)),
    ]);
    const [sourcePreviewPath, editedPreviewPath] = await Promise.all([
      this.writePng(
        jobId,
        createMaskedTexturePreview(source, unitMask),
        `${name}-source-preview`,
      ),
      this.writePng(
        jobId,
        createMaskedTexturePreview(edited, unitMask),
        `${name}-preview`,
      ),
    ]);
    return { sourcePreviewPath, editedPreviewPath };
  }

  async constrainEyeCropEdit(
    jobId: string,
    referencePath: string,
    candidatePath: string,
    fullMaskPath: string,
    crop: EyeEditCrop,
    name: string,
  ): Promise<string> {
    const [reference, candidate, mask] = await Promise.all([
      readFile(referencePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(candidatePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(fullMaskPath).then((bytes) => PNG.sync.read(bytes)),
    ]);
    if (reference.width !== mask.width || reference.height !== mask.height) {
      throw new Error("眼睛编辑蒙版与母版画布尺寸不一致");
    }
    const expectedRatio = crop.output.width / crop.output.height;
    const actualRatio = candidate.width / candidate.height;
    if (!Number.isFinite(actualRatio) || Math.abs(actualRatio / expectedRatio - 1) > 0.03) {
      throw new Error(
        `fal 眼区结果比例异常：应为 4:3，实际为 ${candidate.width}×${candidate.height}`,
      );
    }
    const output = new PNG({ width: reference.width, height: reference.height });
    output.data.set(reference.data);
    let editablePixels = 0;
    let changedPixels = 0;
    let difference = 0;
    const maximumX = Math.min(reference.width, crop.source.x + crop.source.width);
    const maximumY = Math.min(reference.height, crop.source.y + crop.source.height);
    for (let y = crop.source.y; y < maximumY; y += 1) {
      for (let x = crop.source.x; x < maximumX; x += 1) {
        const targetOffset = (y * reference.width + x) * 4;
        const strength = mask.data[targetOffset]! / 255;
        if (strength <= 0) continue;
        const sourceX = ((x + 0.5 - crop.source.x) / crop.source.width) * candidate.width - 0.5;
        const sourceY = ((y + 0.5 - crop.source.y) / crop.source.height) * candidate.height - 0.5;
        const generated = bilinearPixel(candidate, sourceX, sourceY);
        let pixelDifference = 0;
        for (let channel = 0; channel < 3; channel += 1) {
          const original = reference.data[targetOffset + channel]!;
          pixelDifference += Math.abs(generated[channel]! - original);
          output.data[targetOffset + channel] = Math.round(
            original * (1 - strength) + generated[channel]! * strength,
          );
        }
        // Geometry and transparency always come from the installed model.
        output.data[targetOffset + 3] = reference.data[targetOffset + 3]!;
        editablePixels += 1;
        difference += pixelDifference / 3;
        if (pixelDifference >= 18) changedPixels += 1;
      }
    }
    assertEffectiveEyeEdit(editablePixels, changedPixels, difference);
    return this.writePng(jobId, output, name);
  }

  async constrainMaskedCropEdit(
    jobId: string,
    referencePath: string,
    candidatePath: string,
    fullMaskPath: string,
    crop: MaskedEditCrop,
    partLabel: string,
    name: string,
    qualityMode: "preserve" | "restyle" = "restyle",
  ): Promise<string> {
    const [reference, candidate, mask] = await Promise.all([
      readFile(referencePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(candidatePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(fullMaskPath).then((bytes) => PNG.sync.read(bytes)),
    ]);
    if (reference.width !== mask.width || reference.height !== mask.height) {
      throw new Error("部位编辑蒙版与模型画布尺寸不一致");
    }
    const expectedRatio = crop.output.width / crop.output.height;
    const actualRatio = candidate.width / candidate.height;
    if (!Number.isFinite(actualRatio) || Math.abs(actualRatio / expectedRatio - 1) > 0.03) {
      throw new Error(
        `fal 返回的${partLabel}图片比例异常：应为 ${crop.output.width}:${crop.output.height}，` +
        `实际为 ${candidate.width}×${candidate.height}`,
      );
    }
    const output = new PNG({ width: reference.width, height: reference.height });
    output.data.set(reference.data);
    let editablePixels = 0;
    let changedPixels = 0;
    let difference = 0;
    const maximumX = Math.min(reference.width, crop.source.x + crop.source.width);
    const maximumY = Math.min(reference.height, crop.source.y + crop.source.height);
    for (let y = crop.source.y; y < maximumY; y += 1) {
      for (let x = crop.source.x; x < maximumX; x += 1) {
        const targetOffset = (y * reference.width + x) * 4;
        const strength = mask.data[targetOffset]! / 255;
        if (strength <= 0) continue;
        const sourceX = ((x + 0.5 - crop.source.x) / crop.source.width) * candidate.width - 0.5;
        const sourceY = ((y + 0.5 - crop.source.y) / crop.source.height) * candidate.height - 0.5;
        const generated = bilinearPixel(candidate, sourceX, sourceY);
        let pixelDifference = 0;
        for (let channel = 0; channel < 3; channel += 1) {
          const original = reference.data[targetOffset + channel]!;
          pixelDifference += Math.abs(generated[channel]! - original);
          output.data[targetOffset + channel] = Math.round(
            original * (1 - strength) + generated[channel]! * strength,
          );
        }
        // A part-restyle is RGB-only. Alpha, triangles, UVs and draw order are
        // always inherited from the installed model.
        output.data[targetOffset + 3] = reference.data[targetOffset + 3]!;
        editablePixels += 1;
        difference += pixelDifference / 3;
        if (pixelDifference >= 18) changedPixels += 1;
      }
    }
    assertEffectivePartEdit(partLabel, editablePixels, changedPixels, difference);
    assertCleanPartTexture(reference, output, mask, partLabel, qualityMode);
    return this.writePng(jobId, output, name);
  }

  async inspectTemplateCoverage(
    subjectPath: string,
    canvas: MasterTemplateCanvas,
    slots: MasterTemplateSlot[],
  ): Promise<{ coveredRatio: number; uncoveredPixels: number; subjectPixels: number }> {
    const subject = PNG.sync.read(await readFile(subjectPath));
    const coverage = new Uint8Array(subject.width * subject.height);
    for (const slot of slots) {
      if (slot.source === "template") continue;
      const vertices: Array<{ x: number; y: number }> = [];
      for (let index = 0; index < slot.mesh.vertices.length; index += 2) {
        const modelX = slot.mesh.vertices[index]!;
        const modelY = slot.mesh.vertices[index + 1]!;
        vertices.push({
          x: (canvas.originX + modelX * canvas.pixelsPerUnit) / canvas.width * subject.width,
          y: (canvas.originY - modelY * canvas.pixelsPerUnit) / canvas.height * subject.height,
        });
      }
      for (let index = 0; index < slot.mesh.indices.length; index += 3) {
        const a = vertices[slot.mesh.indices[index]!]!;
        const b = vertices[slot.mesh.indices[index + 1]!]!;
        const c = vertices[slot.mesh.indices[index + 2]!]!;
        fillCoverageTriangle(coverage, subject.width, subject.height, a, b, c);
      }
    }
    let subjectPixels = 0;
    let uncoveredPixels = 0;
    for (let index = 0; index < coverage.length; index += 1) {
      if (subject.data[index * 4 + 3]! < 64) continue;
      subjectPixels += 1;
      if (!coverage[index]) uncoveredPixels += 1;
    }
    return {
      coveredRatio: subjectPixels === 0 ? 0 : 1 - uncoveredPixels / subjectPixels,
      uncoveredPixels,
      subjectPixels,
    };
  }

  async normalizeCandidateBackground(jobId: string, sourcePath: string, name: string): Promise<string> {
    const image = PNG.sync.read(await readFile(sourcePath));
    const visited = new Uint8Array(image.width * image.height);
    const queue: number[] = [];
    const enqueue = (x: number, y: number): void => {
      const index = y * image.width + x;
      if (visited[index] || !isRemovableBackground(image, index)) return;
      visited[index] = 1;
      queue.push(index);
    };
    for (let x = 0; x < image.width; x += 1) {
      enqueue(x, 0);
      enqueue(x, image.height - 1);
    }
    for (let y = 0; y < image.height; y += 1) {
      enqueue(0, y);
      enqueue(image.width - 1, y);
    }
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const index = queue[cursor]!;
      const x = index % image.width;
      const y = Math.floor(index / image.width);
      if (x > 0) enqueue(x - 1, y);
      if (x + 1 < image.width) enqueue(x + 1, y);
      if (y > 0) enqueue(x, y - 1);
      if (y + 1 < image.height) enqueue(x, y + 1);
    }
    for (const index of queue) {
      const offset = index * 4;
      image.data[offset] = 255;
      image.data[offset + 1] = 255;
      image.data[offset + 2] = 255;
      image.data[offset + 3] = 255;
    }
    const directory = this.jobDirectory(jobId);
    await mkdir(directory, { recursive: true });
    const target = path.join(directory, `${name.replace(/[^a-zA-Z0-9_-]+/g, "-")}.png`);
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, PNG.sync.write(image), { mode: 0o600 });
    await rename(temporary, target);
    return target;
  }

  async alignCandidateToGuide(
    jobId: string,
    guidePath: string,
    candidatePath: string,
    name: string,
  ): Promise<string> {
    const [guide, candidate] = await Promise.all([
      readFile(guidePath).then((bytes) => PNG.sync.read(bytes)),
      readFile(candidatePath).then((bytes) => PNG.sync.read(bytes)),
    ]);
    const guideBounds = foregroundBounds(guide);
    const candidateBounds = foregroundBounds(candidate);
    if (!guideBounds || !candidateBounds) throw new Error("无法识别母版或候选人物的有效轮廓");
    // Live2D needs a stable head-to-foot scale. Hair and skirts legitimately change
    // the silhouette width, so fitting by the wider axis made otherwise identical
    // poses appear at visibly different heights.
    const scale = guideBounds.height / candidateBounds.height;
    if (!Number.isFinite(scale) || scale <= 0) throw new Error("候选人物轮廓缩放比例无效");
    const targetWidth = candidateBounds.width * scale;
    const targetHeight = candidateBounds.height * scale;
    const targetX = guideBounds.x + (guideBounds.width - targetWidth) / 2;
    const targetY = guideBounds.y;
    if (targetWidth > guide.width * 0.9) {
      throw new Error("候选人物横向轮廓过宽，无法在不压缩身体比例的情况下对齐母版");
    }
    const output = new PNG({ width: guide.width, height: guide.height });
    for (let offset = 0; offset < output.data.length; offset += 4) {
      output.data.set([255, 255, 255, 255], offset);
    }
    const minX = Math.max(0, Math.floor(targetX));
    const maxX = Math.min(output.width - 1, Math.ceil(targetX + targetWidth));
    const minY = Math.max(0, Math.floor(targetY));
    const maxY = Math.min(output.height - 1, Math.ceil(targetY + targetHeight));
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const sourceX = candidateBounds.x + (x + 0.5 - targetX) / scale - 0.5;
        const sourceY = candidateBounds.y + (y + 0.5 - targetY) / scale - 0.5;
        if (
          sourceX < candidateBounds.x || sourceX > candidateBounds.x + candidateBounds.width - 1 ||
          sourceY < candidateBounds.y || sourceY > candidateBounds.y + candidateBounds.height - 1
        ) continue;
        const rgba = bilinearPixel(candidate, sourceX, sourceY);
        output.data.set(rgba, (y * output.width + x) * 4);
      }
    }
    const directory = this.jobDirectory(jobId);
    await mkdir(directory, { recursive: true });
    const target = path.join(directory, `${name.replace(/[^a-zA-Z0-9_-]+/g, "-")}.png`);
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, PNG.sync.write(output), { mode: 0o600 });
    await rename(temporary, target);
    return target;
  }

  async inspectCandidateBackground(candidatePath: string): Promise<{
    compatible: boolean;
    reason: string | null;
  }> {
    const image = PNG.sync.read(await readFile(candidatePath));
    if (image.width <= 0 || image.height <= 0) {
      return { compatible: false, reason: "候选图尺寸无效" };
    }
    const borderX = Math.max(1, Math.floor(image.width * 0.08));
    const borderY = Math.max(1, Math.floor(image.height * 0.08));
    let borderPixels = 0;
    let darkBorderPixels = 0;
    let transparentBorderPixels = 0;
    for (let y = 0; y < image.height; y += 1) {
      for (let x = 0; x < image.width; x += 1) {
        if (x >= borderX && x < image.width - borderX && y >= borderY && y < image.height - borderY) {
          continue;
        }
        const offset = (y * image.width + x) * 4;
        const alpha = image.data[offset + 3]!;
        const luminance = image.data[offset]! * 0.2126 +
          image.data[offset + 1]! * 0.7152 + image.data[offset + 2]! * 0.0722;
        borderPixels += 1;
        if (alpha < 250) transparentBorderPixels += 1;
        if (luminance < 235) darkBorderPixels += 1;
      }
    }
    const transparentRatio = transparentBorderPixels / borderPixels;
    if (transparentRatio > 0.005) {
      return { compatible: false, reason: "背景包含透明像素，必须使用纯白背景" };
    }
    const darkRatio = darkBorderPixels / borderPixels;
    if (darkRatio > 0.12) {
      return {
        compatible: false,
        reason: "背景不是均匀纯白色，疑似包含棋盘格、网格或大面积阴影",
      };
    }
    return { compatible: true, reason: null };
  }

  jobDirectory(jobId: string): string {
    if (!/^[a-zA-Z0-9_-]+$/.test(jobId)) throw new Error("任务 ID 无效");
    return path.join(this.root, jobId);
  }

  resolve(jobId: string, relative: string): string {
    const name = decodeURIComponent(relative).replace(/^\/+/, "");
    if (!name || path.basename(name) !== name || name.includes("..")) {
      throw new Error("生成素材路径无效");
    }
    return path.join(this.jobDirectory(jobId), name);
  }

  async removeJob(jobId: string): Promise<void> {
    await rm(this.jobDirectory(jobId), { recursive: true, force: true });
  }

  private async writePng(jobId: string, image: PNG, name: string): Promise<string> {
    const directory = this.jobDirectory(jobId);
    await mkdir(directory, { recursive: true });
    const safeName = name.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "asset";
    const target = path.join(directory, `${safeName}.png`);
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, PNG.sync.write(image), { mode: 0o600 });
    await rename(temporary, target);
    return target;
  }
}

/**
 * A piece drawn smaller than it is: each work pixel averages its block of
 * source pixels (alpha-weighted, over the canvas backdrop) and is editable
 * when any pixel of the block is.
 */
function paintShrunkTexture(
  canvas: PNG,
  mask: PNG,
  source: PNG,
  unitMask: PNG,
  layout: { x: number; y: number; scale: number; workWidth: number; workHeight: number },
  canvasWidth: number,
): void {
  const block = Math.round(1 / layout.scale);
  for (let ty = 0; ty < layout.workHeight; ty += 1) {
    for (let tx = 0; tx < layout.workWidth; tx += 1) {
      let red = 0, green = 0, blue = 0, alpha = 0, count = 0, editable = 0;
      for (let y = ty * block; y < Math.min(source.height, (ty + 1) * block); y += 1) {
        for (let x = tx * block; x < Math.min(source.width, (tx + 1) * block); x += 1) {
          const offset = (y * source.width + x) * 4;
          const a = source.data[offset + 3]! / 255;
          red += source.data[offset]! * a;
          green += source.data[offset + 1]! * a;
          blue += source.data[offset + 2]! * a;
          alpha += a;
          count += 1;
          editable = Math.max(editable, unitMask.data[offset + 3]!);
        }
      }
      if (!count) continue;
      const coverage = alpha / count;
      const targetOffset = ((layout.y + ty) * canvasWidth + layout.x + tx) * 4;
      canvas.data[targetOffset] = Math.round(red / count + 246 * (1 - coverage));
      canvas.data[targetOffset + 1] = Math.round(green / count + 246 * (1 - coverage));
      canvas.data[targetOffset + 2] = Math.round(blue / count + 246 * (1 - coverage));
      canvas.data[targetOffset + 3] = 255;
      mask.data.set([255, 255, 255, 255 - editable], targetOffset);
    }
  }
}

/**
 * Shrinks the part of an earlier canvas's result that holds its textures into
 * `target`, centered, keeping its proportions.
 */
function paintTextureAnchor(
  canvas: PNG,
  result: PNG,
  resultPlan: TextureEditBatchPlan,
  target: { x: number; y: number; width: number; height: number },
): void {
  const left = Math.min(...resultPlan.items.map((item) => item.x));
  const top = Math.min(...resultPlan.items.map((item) => item.y));
  const right = Math.max(...resultPlan.items.map((item) => item.x + item.workWidth));
  const bottom = Math.max(...resultPlan.items.map((item) => item.y + item.workHeight));
  // The result may come back at another resolution than the canvas it was drawn on.
  const kx = result.width / resultPlan.canvasWidth;
  const ky = result.height / resultPlan.canvasHeight;
  const fit = Math.min(target.width / (right - left), target.height / (bottom - top));
  const width = Math.max(1, Math.round((right - left) * fit));
  const height = Math.max(1, Math.round((bottom - top) * fit));
  const offsetX = target.x + Math.floor((target.width - width) / 2);
  const offsetY = target.y + Math.floor((target.height - height) / 2);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sourceX = (left + (x + 0.5) / fit) * kx - 0.5;
      const sourceY = (top + (y + 0.5) / fit) * ky - 0.5;
      const pixel = bilinearPixel(result, sourceX, sourceY);
      const alpha = pixel[3] / 255;
      const offset = ((offsetY + y) * canvas.width + offsetX + x) * 4;
      // Onto the same light backdrop as the rest of the canvas.
      canvas.data.set([
        Math.round(pixel[0] * alpha + 246 * (1 - alpha)),
        Math.round(pixel[1] * alpha + 247 * (1 - alpha)),
        Math.round(pixel[2] * alpha + 248 * (1 - alpha)),
        255,
      ], offset);
    }
  }
}

function fillPng(image: PNG, rgba: readonly [number, number, number, number]): void {
  for (let offset = 0; offset < image.data.length; offset += 4) image.data.set(rgba, offset);
}

function textureSharpnessRatio(source: PNG, output: PNG, mask: PNG): number {
  let original = 0;
  let edited = 0;
  let count = 0;
  for (let y = 2; y < source.height - 2; y += 1) {
    for (let x = 2; x < source.width - 2; x += 1) {
      if (!isMaskInterior(mask, x, y)) continue;
      original += rgbGradient(source, x, y);
      edited += rgbGradient(output, x, y);
      count += 1;
    }
  }
  if (count === 0 || original / count < 2) return 1;
  return edited / original;
}

function maskBounds(image: PNG): { x: number; y: number; width: number; height: number } | null {
  let minimumX = image.width;
  let minimumY = image.height;
  let maximumX = -1;
  let maximumY = -1;
  for (let y = 0; y < image.height; y += 1) {
    for (let x = 0; x < image.width; x += 1) {
      if (image.data[(y * image.width + x) * 4]! === 0) continue;
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

function eyeContextCrop(
  imageWidth: number,
  imageHeight: number,
  eyes: { x: number; y: number; width: number; height: number },
): { x: number; y: number; width: number; height: number } {
  const aspect = 4 / 3;
  let width = Math.max(eyes.width * 4, eyes.height * 8, imageWidth * 0.32);
  let height = width / aspect;
  if (height < eyes.height * 6) {
    height = eyes.height * 6;
    width = height * aspect;
  }
  if (width > imageWidth) {
    width = imageWidth;
    height = width / aspect;
  }
  if (height > imageHeight) {
    height = imageHeight;
    width = height * aspect;
  }
  width = Math.max(1, Math.round(width));
  height = Math.max(1, Math.round(height));
  const centerX = eyes.x + eyes.width / 2;
  const centerY = eyes.y + eyes.height / 2;
  const x = Math.max(0, Math.min(imageWidth - width, Math.round(centerX - width / 2)));
  const y = Math.max(0, Math.min(imageHeight - height, Math.round(centerY - height / 2)));
  return { x, y, width, height };
}

function editOutputSize(bounds: { width: number; height: number }): { width: number; height: number } {
  const ratio = bounds.width / Math.max(1, bounds.height);
  if (ratio >= 1.35) return { width: 1024, height: 768 };
  if (ratio <= 0.74) return { width: 768, height: 1024 };
  return { width: 1024, height: 1024 };
}

function contextCrop(
  imageWidth: number,
  imageHeight: number,
  bounds: { x: number; y: number; width: number; height: number },
  aspect: number,
): { x: number; y: number; width: number; height: number } {
  let width = Math.max(bounds.width * 2.4, 96);
  let height = Math.max(bounds.height * 2.4, 96);
  if (width / height < aspect) width = height * aspect;
  else height = width / aspect;
  if (width > imageWidth) {
    width = imageWidth;
    height = width / aspect;
  }
  if (height > imageHeight) {
    height = imageHeight;
    width = height * aspect;
  }
  width = Math.max(1, Math.round(width));
  height = Math.max(1, Math.round(height));
  const centerX = bounds.x + bounds.width / 2;
  const centerY = bounds.y + bounds.height / 2;
  return {
    x: Math.max(0, Math.min(imageWidth - width, Math.round(centerX - width / 2))),
    y: Math.max(0, Math.min(imageHeight - height, Math.round(centerY - height / 2))),
    width,
    height,
  };
}

function resizeRegion(
  image: PNG,
  region: { x: number; y: number; width: number; height: number },
  outputSize: { width: number; height: number },
): PNG {
  const output = new PNG({ width: outputSize.width, height: outputSize.height });
  for (let y = 0; y < output.height; y += 1) {
    for (let x = 0; x < output.width; x += 1) {
      const sourceX = region.x + ((x + 0.5) / output.width) * region.width - 0.5;
      const sourceY = region.y + ((y + 0.5) / output.height) * region.height - 0.5;
      output.data.set(bilinearPixel(image, sourceX, sourceY), (y * output.width + x) * 4);
    }
  }
  return output;
}

function assertEffectiveEyeEdit(
  editablePixels: number,
  changedPixels: number,
  difference: number,
): void {
  if (editablePixels === 0) throw new Error("眼睛编辑蒙版为空");
  const changedRatio = changedPixels / editablePixels;
  const meanDifference = difference / editablePixels;
  if (changedRatio < 0.03 || meanDifference < 2) {
    throw new Error("AI 没有有效修改眼睛，请换一个更明确的描述后重试");
  }
}

function assertEffectivePartEdit(
  label: string,
  editablePixels: number,
  changedPixels: number,
  difference: number,
): void {
  if (editablePixels === 0) throw new Error(`${label}编辑蒙版为空`);
  const changedRatio = changedPixels / editablePixels;
  const meanDifference = difference / editablePixels;
  if (changedRatio < 0.02 || meanDifference < 1.5) {
    throw new Error(`AI 没有有效修改${label}，请换一个更明确的描述后重试`);
  }
}

function assertCleanPartTexture(
  reference: PNG,
  output: PNG,
  mask: PNG,
  label: string,
  mode: "preserve" | "restyle",
): void {
  const issues = inspectCleanPartTexture(reference, output, mask, label, mode);
  if (issues.length > 0) {
    throw new Error(
      `fal 返回的${label}贴图出现模糊、污点或噪声，已阻止写回模型；` +
      "请重试，原模型不会被修改。",
    );
  }
}

function inspectCleanPartTexture(
  reference: PNG,
  output: PNG,
  mask: PNG,
  label: string,
  mode: "preserve" | "restyle",
): string[] {
  let originalEdges = 0;
  let retainedEdges = 0;
  let flatPixels = 0;
  let newHardEdges = 0;
  let newSpeckles = 0;
  for (let y = 2; y < reference.height - 2; y += 1) {
    for (let x = 2; x < reference.width - 2; x += 1) {
      if (!isMaskInterior(mask, x, y)) continue;
      const originalGradient = rgbGradient(reference, x, y);
      const generatedGradient = rgbGradient(output, x, y);
      if (originalGradient >= 42) {
        originalEdges += 1;
        if (maximumRgbGradient(output, x, y) >= Math.max(18, originalGradient * 0.38)) {
          retainedEdges += 1;
        }
      } else if (originalGradient <= 14) {
        flatPixels += 1;
        if (generatedGradient >= 48) newHardEdges += 1;
        const originalLaplacian = rgbLaplacian(reference, x, y);
        const generatedLaplacian = rgbLaplacian(output, x, y);
        if (originalLaplacian <= 10 && generatedLaplacian >= 34) newSpeckles += 1;
      }
    }
  }
  const retainedRatio = originalEdges === 0 ? 1 : retainedEdges / originalEdges;
  const newEdgeRatio = flatPixels === 0 ? 0 : newHardEdges / flatPixels;
  const speckleRatio = flatPixels === 0 ? 0 : newSpeckles / flatPixels;
  const minimumRetained = mode === "preserve" ? 0.62 : 0.45;
  const maximumNewEdges = mode === "preserve" ? 0.035 : 0.08;
  const maximumSpeckles = mode === "preserve" ? 0.025 : 0.06;
  const issues: string[] = [];
  if (originalEdges >= 24 && retainedRatio < minimumRetained) {
    issues.push(`${label}内部线稿变化较大（原始主要边缘保留 ${(retainedRatio * 100).toFixed(0)}%）`);
  }
  if (flatPixels >= 100 && newEdgeRatio > maximumNewEdges) {
    issues.push(`${label}新增高频边缘较多（${(newEdgeRatio * 100).toFixed(1)}%），请检查锯齿或脏边`);
  }
  if (flatPixels >= 100 && speckleRatio > maximumSpeckles) {
    issues.push(`${label}可能包含颗粒或斑点（${(speckleRatio * 100).toFixed(1)}%）`);
  }
  return issues;
}

function cleanTextureAlphaEdges(output: PNG, mask: PNG): void {
  const generated = new PNG({ width: output.width, height: output.height });
  generated.data.set(output.data);
  const stableInterior = buildStableInteriorMap(generated, mask);
  for (let y = 0; y < output.height; y += 1) {
    for (let x = 0; x < output.width; x += 1) {
      const offset = (y * output.width + x) * 4;
      if (mask.data[offset]! < 192) continue;
      const alpha = output.data[offset + 3]!;
      if (alpha === 0) continue;
      const edgeDepth = transparentEdgeDepth(generated, x, y, 4);
      if (edgeDepth === null) continue;
      const interiorColor = robustInteriorColor(generated, stableInterior, x, y, 24);
      if (!interiorColor) continue;
      const currentColor: [number, number, number] = [
        generated.data[offset]!,
        generated.data[offset + 1]!,
        generated.data[offset + 2]!,
      ];
      if (!shouldReplaceEdgeColor(currentColor, interiorColor, alpha, edgeDepth)) continue;
      for (let channel = 0; channel < 3; channel += 1) {
        // Use a robust sample of the generated material, never a single
        // nearest pixel. A nearest pixel can itself be dark line art, which
        // produced black speckles around light hair and other thin shapes.
        output.data[offset + channel] = interiorColor[channel]!;
      }
    }
  }
  // Transparent RGB padding is sampled by WebGL around an ArtMesh edge even
  // though its alpha remains zero. Extend the already-cleaned visible colour
  // into a small transparent ring to prevent white/black atlas fringes.
  const cleaned = new PNG({ width: output.width, height: output.height });
  cleaned.data.set(output.data);
  for (let y = 0; y < output.height; y += 1) {
    for (let x = 0; x < output.width; x += 1) {
      const offset = (y * output.width + x) * 4;
      if (mask.data[offset]! < 192 || output.data[offset + 3]! !== 0) continue;
      const visible = nearestAlphaPixel(output, x, y, 3, (value) => value > 0);
      if (!visible) continue;
      const visibleOffset = (visible.y * output.width + visible.x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        output.data[offset + channel] = cleaned.data[visibleOffset + channel]!;
      }
    }
  }
}

function buildStableInteriorMap(image: PNG, mask: PNG): Uint8Array {
  const stable = new Uint8Array(image.width * image.height);
  for (let y = 0; y < image.height; y += 1) {
    for (let x = 0; x < image.width; x += 1) {
      const pixel = y * image.width + x;
      const offset = pixel * 4;
      if (
        mask.data[offset]! >= 192 &&
        image.data[offset + 3]! >= 245 &&
        !touchesTransparentPixel(image, x, y, 1)
      ) stable[pixel] = 1;
    }
  }
  return stable;
}

function robustInteriorColor(
  image: PNG,
  stableInterior: Uint8Array,
  x: number,
  y: number,
  maximumRadius: number,
): [number, number, number] | null {
  const samples: Array<[number, number, number]> = [];
  for (let radius = 1; radius <= maximumRadius; radius += 1) {
    for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
      for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
        if (Math.max(Math.abs(offsetX), Math.abs(offsetY)) !== radius) continue;
        const px = x + offsetX;
        const py = y + offsetY;
        if (px < 0 || py < 0 || px >= image.width || py >= image.height) continue;
        if (stableInterior[py * image.width + px] !== 1) continue;
        const offset = (py * image.width + px) * 4;
        samples.push([
          image.data[offset]!,
          image.data[offset + 1]!,
          image.data[offset + 2]!,
        ]);
      }
    }
    // Use a neighbourhood instead of one nearest pixel, but keep it local so
    // nearby decorations do not colour a different material island.
    if (radius >= 8 && samples.length >= 32) break;
  }
  if (samples.length === 0) return null;
  return [
    percentile(samples.map((sample) => sample[0]), 0.6),
    percentile(samples.map((sample) => sample[1]), 0.6),
    percentile(samples.map((sample) => sample[2]), 0.6),
  ];
}

function percentile(values: number[], ratio: number): number {
  values.sort((left, right) => left - right);
  return values[Math.min(values.length - 1, Math.floor((values.length - 1) * ratio))]!;
}

function transparentEdgeDepth(image: PNG, x: number, y: number, maximumDepth: number): number | null {
  for (let radius = 1; radius <= maximumDepth; radius += 1) {
    if (touchesTransparentPixel(image, x, y, radius)) return radius;
  }
  return null;
}

function shouldReplaceEdgeColor(
  current: [number, number, number],
  interior: [number, number, number],
  alpha: number,
  edgeDepth: number,
): boolean {
  if (alpha < 245) return true;
  const redDifference = current[0] - interior[0];
  const greenDifference = current[1] - interior[1];
  const blueDifference = current[2] - interior[2];
  const distance = Math.sqrt(
    redDifference * redDifference +
    greenDifference * greenDifference +
    blueDifference * blueDifference,
  );
  const luminanceDifference = Math.abs(rgbLuminance(current) - rgbLuminance(interior));
  const distanceThreshold = edgeDepth <= 2 ? 48 : 82;
  const luminanceThreshold = edgeDepth <= 2 ? 22 : 42;
  return distance >= distanceThreshold && luminanceDifference >= luminanceThreshold;
}

function rgbLuminance(color: [number, number, number]): number {
  return color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722;
}

function touchesTransparentPixel(image: PNG, x: number, y: number, radius: number): boolean {
  for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
    for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
      const px = x + offsetX;
      const py = y + offsetY;
      if (px < 0 || py < 0 || px >= image.width || py >= image.height) return true;
      if (image.data[(py * image.width + px) * 4 + 3]! <= 8) return true;
    }
  }
  return false;
}

function nearestAlphaPixel(
  image: PNG,
  x: number,
  y: number,
  maximumRadius: number,
  matches: (alpha: number, x: number, y: number) => boolean,
): { x: number; y: number } | null {
  for (let radius = 1; radius <= maximumRadius; radius += 1) {
    let best: { x: number; y: number; distance: number } | null = null;
    for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {
      for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {
        if (Math.max(Math.abs(offsetX), Math.abs(offsetY)) !== radius) continue;
        const px = x + offsetX;
        const py = y + offsetY;
        if (px < 0 || py < 0 || px >= image.width || py >= image.height) continue;
        const alpha = image.data[(py * image.width + px) * 4 + 3]!;
        if (!matches(alpha, px, py)) continue;
        const distance = offsetX * offsetX + offsetY * offsetY;
        if (!best || distance < best.distance) best = { x: px, y: py, distance };
      }
    }
    if (best) return { x: best.x, y: best.y };
  }
  return null;
}

function isMaskInterior(mask: PNG, x: number, y: number): boolean {
  for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
    for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
      if (mask.data[((y + offsetY) * mask.width + x + offsetX) * 4]! < 192) return false;
    }
  }
  return true;
}

function maximumRgbGradient(image: PNG, x: number, y: number): number {
  let maximum = 0;
  for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
    for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
      maximum = Math.max(maximum, rgbGradient(image, x + offsetX, y + offsetY));
    }
  }
  return maximum;
}

function rgbGradient(image: PNG, x: number, y: number): number {
  const left = (y * image.width + x - 1) * 4;
  const right = (y * image.width + x + 1) * 4;
  const top = ((y - 1) * image.width + x) * 4;
  const bottom = ((y + 1) * image.width + x) * 4;
  let total = 0;
  for (let channel = 0; channel < 3; channel += 1) {
    total += Math.abs(image.data[right + channel]! - image.data[left + channel]!);
    total += Math.abs(image.data[bottom + channel]! - image.data[top + channel]!);
  }
  return total / 6;
}

function rgbLaplacian(image: PNG, x: number, y: number): number {
  const center = (y * image.width + x) * 4;
  const left = center - 4;
  const right = center + 4;
  const top = center - image.width * 4;
  const bottom = center + image.width * 4;
  let total = 0;
  for (let channel = 0; channel < 3; channel += 1) {
    total += Math.abs(
      image.data[center + channel]! * 4 -
      image.data[left + channel]! -
      image.data[right + channel]! -
      image.data[top + channel]! -
      image.data[bottom + channel]!,
    );
  }
  return total / 3;
}

function interiorDistance(
  coverage: Uint8Array,
  width: number,
  height: number,
  maximum: number,
): Uint8Array {
  let current = coverage.slice();
  const distance = new Uint8Array(coverage.length);
  for (let level = 1; level <= maximum; level += 1) {
    const eroded = new Uint8Array(coverage.length);
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const index = y * width + x;
        if (!current[index]) continue;
        let inside = true;
        for (let offsetY = -1; offsetY <= 1 && inside; offsetY += 1) {
          for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
            if (!current[(y + offsetY) * width + x + offsetX]) {
              inside = false;
              break;
            }
          }
        }
        if (inside) eroded[index] = 1;
      }
    }
    for (let index = 0; index < eroded.length; index += 1) {
      if (eroded[index]) distance[index] = level;
    }
    current = eroded;
  }
  return distance;
}

function isRemovableBackground(image: PNG, index: number): boolean {
  const offset = index * 4;
  const red = image.data[offset]!;
  const green = image.data[offset + 1]!;
  const blue = image.data[offset + 2]!;
  const alpha = image.data[offset + 3]!;
  const spread = Math.max(red, green, blue) - Math.min(red, green, blue);
  const luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
  return alpha < 250 || (spread <= 24 && luminance >= 175);
}

function foregroundBounds(image: PNG): { x: number; y: number; width: number; height: number } | null {
  let minimumX = image.width;
  let minimumY = image.height;
  let maximumX = -1;
  let maximumY = -1;
  for (let y = 0; y < image.height; y += 1) {
    for (let x = 0; x < image.width; x += 1) {
      if (isRemovableBackground(image, y * image.width + x)) continue;
      minimumX = Math.min(minimumX, x);
      minimumY = Math.min(minimumY, y);
      maximumX = Math.max(maximumX, x);
      maximumY = Math.max(maximumY, y);
    }
  }
  return maximumX < minimumX || maximumY < minimumY ? null : {
    x: minimumX,
    y: minimumY,
    width: maximumX - minimumX + 1,
    height: maximumY - minimumY + 1,
  };
}

function fillCoverageTriangle(
  coverage: Uint8Array,
  width: number,
  height: number,
  a: Point,
  b: Point,
  c: Point,
): void {
  const area = triangleEdge(a, b, c);
  if (Math.abs(area) < 1e-8) return;
  const minX = Math.max(0, Math.floor(Math.min(a.x, b.x, c.x)));
  const maxX = Math.min(width - 1, Math.ceil(Math.max(a.x, b.x, c.x)));
  const minY = Math.max(0, Math.floor(Math.min(a.y, b.y, c.y)));
  const maxY = Math.min(height - 1, Math.ceil(Math.max(a.y, b.y, c.y)));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const point = { x: x + 0.5, y: y + 0.5 };
      const w0 = triangleEdge(b, c, point) / area;
      const w1 = triangleEdge(c, a, point) / area;
      const w2 = triangleEdge(a, b, point) / area;
      if (w0 >= -1e-5 && w1 >= -1e-5 && w2 >= -1e-5) coverage[y * width + x] = 1;
    }
  }
}

function rasterMaskTriangle(
  strengths: Uint8Array,
  width: number,
  height: number,
  sourceImage: PNG,
  destination: [Point, Point, Point],
  source: [Point, Point, Point],
): void {
  const [a, b, c] = destination;
  const area = edgePoint(a, b, c);
  if (Math.abs(area) < 1e-9) return;
  const minX = Math.max(0, Math.floor(Math.min(a.x, b.x, c.x)));
  const maxX = Math.min(width - 1, Math.ceil(Math.max(a.x, b.x, c.x)));
  const minY = Math.max(0, Math.floor(Math.min(a.y, b.y, c.y)));
  const maxY = Math.min(height - 1, Math.ceil(Math.max(a.y, b.y, c.y)));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const point = { x: x + 0.5, y: y + 0.5 };
      const weights = [
        edgePoint(b, c, point) / area,
        edgePoint(c, a, point) / area,
        edgePoint(a, b, point) / area,
      ] as const;
      if (weights.some((weight) => weight < -1e-5)) continue;
      const sourceX = source[0].x * weights[0] + source[1].x * weights[1] + source[2].x * weights[2];
      const sourceY = source[0].y * weights[0] + source[1].y * weights[1] + source[2].y * weights[2];
      const alpha = bilinearPixel(sourceImage, sourceX, sourceY)[3]!;
      const index = y * width + x;
      strengths[index] = Math.max(strengths[index]!, alpha);
    }
  }
}

function edgePoint(a: Point, b: Point, point: Point): number {
  return (point.x - a.x) * (b.y - a.y) - (point.y - a.y) * (b.x - a.x);
}

function pairs(values: number[]): Array<[number, number]> {
  const result: Array<[number, number]> = [];
  for (let index = 0; index < values.length; index += 2) {
    result.push([values[index]!, values[index + 1]!]);
  }
  return result;
}

function triangleEdge(a: Point, b: Point, point: Point): number {
  return (point.x - a.x) * (b.y - a.y) - (point.y - a.y) * (b.x - a.x);
}

interface Point {
  x: number;
  y: number;
}

function bilinearPixel(image: PNG, x: number, y: number): [number, number, number, number] {
  const x0 = Math.max(0, Math.min(image.width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(image.height - 1, Math.floor(y)));
  const x1 = Math.min(image.width - 1, x0 + 1);
  const y1 = Math.min(image.height - 1, y0 + 1);
  const fx = x - Math.floor(x);
  const fy = y - Math.floor(y);
  const rgba = [0, 0, 0, 0] as [number, number, number, number];
  for (let channel = 0; channel < 4; channel += 1) {
    const top = image.data[(y0 * image.width + x0) * 4 + channel]! * (1 - fx) +
      image.data[(y0 * image.width + x1) * 4 + channel]! * fx;
    const bottom = image.data[(y1 * image.width + x0) * 4 + channel]! * (1 - fx) +
      image.data[(y1 * image.width + x1) * 4 + channel]! * fx;
    rgba[channel] = Math.round(top * (1 - fy) + bottom * fy);
  }
  return rgba;
}

function shouldUseMaskAlpha(mask: PNG): boolean {
  let minimumRgb = 255;
  let maximumRgb = 0;
  let minimumAlpha = 255;
  let maximumAlpha = 0;
  for (let offset = 0; offset < mask.data.length; offset += 4) {
    minimumRgb = Math.min(minimumRgb, mask.data[offset]!, mask.data[offset + 1]!, mask.data[offset + 2]!);
    maximumRgb = Math.max(maximumRgb, mask.data[offset]!, mask.data[offset + 1]!, mask.data[offset + 2]!);
    minimumAlpha = Math.min(minimumAlpha, mask.data[offset + 3]!);
    maximumAlpha = Math.max(maximumAlpha, mask.data[offset + 3]!);
  }
  return maximumRgb - minimumRgb < 16 && maximumAlpha - minimumAlpha >= 16;
}

export function generationCacheKey(operation: string, input: unknown): string {
  return createHash("sha256")
    .update(operation)
    .update("\0")
    .update(stableJson(input))
    .digest("hex");
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right));
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function imageExtension(contentType: string | null, pathname: string): string {
  if (contentType?.includes("jpeg")) return ".jpg";
  if (contentType?.includes("webp")) return ".webp";
  const extension = path.extname(pathname).toLowerCase();
  return [".png", ".jpg", ".jpeg", ".webp"].includes(extension) ? extension : ".png";
}
