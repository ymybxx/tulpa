import type { CubismRenderer_WebGL } from "@cubism/rendering/cubismrenderer_webgl";

import type { TextureAtlasPatchSet, TextureAtlasUpdate } from "../shared/contracts";
import {
  CubismTexturePatchCompositor,
  type TexturePatchCandidateResource,
} from "./cubism-texture-patch-compositor";
import { textureDetailRequestUrl } from "./texture-detail";
import { allOrRelease } from "./all-or-release";

export type TextureRecipe =
  | { kind: "patch-set"; value: TextureAtlasPatchSet }
  | { kind: "full-atlas"; value: TextureAtlasUpdate };

interface TextureResource extends TexturePatchCandidateResource {
  url: string;
  recipe: TextureRecipe;
}

export interface CubismTextureStoreOptions {
  maxTextureDimension?: number;
  onCreated?: (resource: { width: number; height: number }) => void;
  onDeleted?: () => void;
}

/** Owns model atlas GPU resources and performs atomic hot replacement. */
export class CubismTextureStore {
  private resources: Array<TextureResource | null> = [];
  private maximumDimension: number;
  private readonly patchCompositor: CubismTexturePatchCompositor;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    private readonly renderer: CubismRenderer_WebGL,
    private readonly options: CubismTextureStoreOptions = {},
  ) {
    this.maximumDimension = normalizeMaximumDimension(options.maxTextureDimension);
    this.patchCompositor = new CubismTexturePatchCompositor(gl);
  }

  async loadAll(urls: string[]): Promise<void> {
    const loaded = await allOrRelease(
      urls.map((url, atlasIndex) => this.load(
        url,
        this.maximumDimension,
        { kind: "full-atlas", value: { atlasIndex, atlasFile: "", url } },
      )),
      (resource) => this.delete(resource),
    );
    try {
      for (let index = 0; index < loaded.length; index += 1) {
        this.renderer.bindTexture(index, loaded[index]!.texture);
      }
      this.resources = loaded;
      this.renderer.setIsPremultipliedAlpha(true);
    } catch (error) {
      for (const resource of loaded) this.delete(resource);
      throw error;
    }
  }

  async replace(
    updates: TextureAtlasUpdate[],
    validate: () => void,
  ): Promise<void> {
    const candidates = await allOrRelease(
      updates.map(async (update) => ({
        atlasIndex: update.atlasIndex,
        resource: await this.load(update.url, this.maximumDimension, {
          kind: "full-atlas",
          value: update,
        }),
      })),
      (candidate) => this.delete(candidate.resource),
    );
    const previous: Array<{ atlasIndex: number; resource: TextureResource | null }> = [];
    try {
      for (const candidate of candidates) {
        previous.push({
          atlasIndex: candidate.atlasIndex,
          resource: this.resources[candidate.atlasIndex] ?? null,
        });
        this.renderer.bindTexture(candidate.atlasIndex, candidate.resource.texture);
        this.resources[candidate.atlasIndex] = candidate.resource;
      }
      validate();
    } catch (error) {
      for (const item of previous) {
        if (item.resource) this.renderer.bindTexture(item.atlasIndex, item.resource.texture);
        this.resources[item.atlasIndex] = item.resource;
      }
      for (const candidate of candidates) this.delete(candidate.resource);
      throw error;
    }
    for (const item of previous) {
      if (item.resource) this.delete(item.resource);
    }
  }

  async replacePatchSets(
    patchSets: TextureAtlasPatchSet[],
    validate: () => void,
    isCurrent: () => boolean = () => true,
  ): Promise<void> {
    const candidates: Array<{
      atlasIndex: number;
      previous: TextureResource;
      candidate: TextureResource;
    }> = [];
    try {
      for (const patchSet of patchSets) {
        const previous = this.resources[patchSet.atlasIndex];
        if (!previous) throw new Error(`纹理图集索引无效：${patchSet.atlasIndex}`);
        const candidate = await this.load(patchSet.baseUrl, this.maximumDimension, {
          kind: "patch-set",
          value: patchSet,
        });
        try {
          await this.patchCompositor.apply(candidate, patchSet, isCurrent);
        } catch (error) {
          this.delete(candidate);
          throw error;
        }
        candidates.push({ atlasIndex: patchSet.atlasIndex, previous, candidate });
      }
    } catch (error) {
      for (const item of candidates) this.delete(item.candidate);
      throw error;
    }
    const committed = commitTextureCandidates<
      TextureResource,
      (typeof candidates)[number]
    >({
      candidates,
      isCurrent,
      bind: (item, value) => this.renderer.bindTexture(item.atlasIndex, value.texture),
      validate,
      release: (resource) => this.delete(resource),
    });
    if (!committed) return;
    for (const item of candidates) this.resources[item.atlasIndex] = item.candidate;
  }

  async setMaximumDimension(maximumDimension: number, validate: () => void): Promise<void> {
    const next = normalizeMaximumDimension(maximumDimension);
    if (next === this.maximumDimension) return;
    // Replace one atlas at a time. Keeping the previous texture bound until the
    // next bitmap is ready avoids both a blank frame and a second full atlas set.
    for (let atlasIndex = 0; atlasIndex < this.resources.length; atlasIndex += 1) {
      const previous = this.resources[atlasIndex];
      if (!previous) continue;
      const candidate = await this.load(
        previous.recipe.kind === "patch-set" ? previous.recipe.value.baseUrl : previous.url,
        next,
        previous.recipe,
      );
      try {
        if (previous.recipe.kind === "patch-set") {
          await this.patchCompositor.apply(candidate, previous.recipe.value, () => true);
        }
      } catch (error) {
        this.delete(candidate);
        throw error;
      }
      this.resources[atlasIndex] = commitTextureCandidate({
        previous,
        candidate,
        isCurrent: () => true,
        bind: (resource) => this.renderer.bindTexture(atlasIndex, resource.texture),
        validate,
        release: (resource) => this.delete(resource),
      });
    }
    this.maximumDimension = next;
  }

  maxDimension(): number {
    return this.maximumDimension;
  }

  urls(): string[] {
    return this.resources.map((resource) => resource?.url ?? "");
  }

  recipes(): TextureRecipe[] {
    return this.resources.flatMap((resource) => resource ? [resource.recipe] : []);
  }

  count(): number {
    return this.resources.length;
  }

  dimensions(): Array<{ width: number; height: number }> {
    return this.resources.flatMap((resource) => resource
      ? [{ width: resource.width, height: resource.height }]
      : []);
  }

  release(): void {
    for (const resource of this.resources) {
      if (resource) this.delete(resource);
    }
    this.resources.length = 0;
    this.patchCompositor.release();
  }

  private async load(
    url: string,
    maximumDimension = this.maximumDimension,
    recipe: TextureRecipe = {
      kind: "full-atlas",
      value: { atlasIndex: 0, atlasFile: "", url },
    },
  ): Promise<TextureResource> {
    const requestUrl = textureDetailRequestUrl(url, maximumDimension);
    const response = await fetch(requestUrl);
    if (!response.ok) throw new Error(`读取 Live2D 纹理失败 (${response.status}): ${url}`);
    const source = await createImageBitmap(await response.blob(), {
      premultiplyAlpha: "premultiply",
      colorSpaceConversion: "default",
    });
    // The pet-model protocol normally serves a persistent pre-scaled atlas.
    // Keep this renderer-side resize as a fallback for external URLs.
    const maximum = maximumDimension;
    const scale = Math.min(1, maximum / Math.max(source.width, source.height));
    const bitmap = scale < 1
      ? await createImageBitmap(source, 0, 0, source.width, source.height, {
        resizeWidth: Math.max(1, Math.round(source.width * scale)),
        resizeHeight: Math.max(1, Math.round(source.height * scale)),
        resizeQuality: "high",
        premultiplyAlpha: "premultiply",
        colorSpaceConversion: "default",
      })
      : source;
    if (bitmap !== source) source.close();
    const texture = this.gl.createTexture();
    if (!texture) {
      bitmap.close();
      throw new Error("创建 Live2D GPU 纹理失败");
    }
    try {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.bindTexture(gl.TEXTURE_2D, null);
      const resource = { url, texture, width: bitmap.width, height: bitmap.height, recipe };
      this.options.onCreated?.(resource);
      return resource;
    } catch (error) {
      this.gl.deleteTexture(texture);
      throw error;
    } finally {
      bitmap.close();
    }
  }

  private delete(resource: TextureResource): void {
    this.gl.deleteTexture(resource.texture);
    this.options.onDeleted?.();
  }
}

function normalizeMaximumDimension(value: number | undefined): number {
  if (!Number.isFinite(value)) return Number.MAX_SAFE_INTEGER;
  return Math.max(1, Math.floor(value!));
}

export function commitTextureCandidate<T>(input: {
  previous: T;
  candidate: T;
  isCurrent(): boolean;
  bind(value: T): void;
  validate(): void;
  release(value: T): void;
}): T {
  if (!input.isCurrent()) {
    input.release(input.candidate);
    return input.previous;
  }
  input.bind(input.candidate);
  try {
    input.validate();
  } catch (error) {
    input.bind(input.previous);
    input.release(input.candidate);
    throw error;
  }
  input.release(input.previous);
  return input.candidate;
}

export function commitTextureCandidates<T, C extends { previous: T; candidate: T }>(input: {
  candidates: readonly C[];
  isCurrent(): boolean;
  bind(item: C, value: T): void;
  validate(): void;
  release(value: T): void;
}): boolean {
  if (!input.isCurrent()) {
    for (const item of input.candidates) input.release(item.candidate);
    return false;
  }
  try {
    for (const item of input.candidates) input.bind(item, item.candidate);
    if (!input.isCurrent()) {
      for (const item of input.candidates) input.bind(item, item.previous);
      for (const item of input.candidates) input.release(item.candidate);
      return false;
    }
    input.validate();
  } catch (error) {
    for (const item of input.candidates) input.bind(item, item.previous);
    for (const item of input.candidates) input.release(item.candidate);
    throw error;
  }
  for (const item of input.candidates) input.release(item.previous);
  return true;
}
