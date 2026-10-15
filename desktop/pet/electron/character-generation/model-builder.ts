import { randomUUID } from "node:crypto";
import { copyFile, cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { PNG } from "pngjs";

import type { ModelImportResult, ModelPackage } from "../../src/shared/contracts";
import { applyTexturePatchPixel } from "../../src/shared/texture-patch";
import type {
  CharacterGenerationJob,
  MasterTemplateSlot,
} from "../../src/shared/character-generation";
import type { RgbaImage } from "./image-processing";
import { composeMeshAtlas, type MeshSlotTexture } from "./mesh-atlas-composer";
import type { MasterTemplatePackage } from "./master-template-store";

export interface GeneratedModelStore {
  selected(): Promise<ModelPackage | null>;
  importFolder(source: string): Promise<ModelImportResult>;
  select(id: string): Promise<ModelPackage>;
}

export interface GeneratedModelBuildOptions {
  /** Keep every atlas pixel that is not explicitly supplied by this job. */
  preserveUnchanged?: boolean;
  /** Apply supplied canvases as RGB deltas from this neutral render. */
  deltaReferencePath?: string;
  textureEdits?: Array<{
    textureUnitId: string;
    atlasIndex: number;
    bounds: { x: number; y: number; width: number; height: number };
    maskPath: string;
    editedPatchPath: string;
  }>;
}

export type CompiledAtlasResolver = (
  modelId: string,
  atlasFile: string,
) => Promise<string | null>;

export class GeneratedModelBuilder {
  constructor(
    private readonly outputsRoot: string,
    private readonly models: GeneratedModelStore,
    private readonly resolveCompiledAtlas?: CompiledAtlasResolver,
  ) {}

  async build(
    job: CharacterGenerationJob,
    template: MasterTemplatePackage,
    name: string,
    options: GeneratedModelBuildOptions = {},
  ): Promise<ModelPackage> {
    const previous = await this.models.selected();
    const staging = path.join(this.outputsRoot, `${job.id}-${randomUUID()}`);
    await mkdir(this.outputsRoot, { recursive: true });
    try {
      await cp(template.root, staging, { recursive: true, force: false });
      const isTexturePreview = Boolean(options.preserveUnchanged && options.textureEdits?.length);
      if (isTexturePreview) {
        // A generated file consumer needs a real atlas. Resolve the immutable
        // compiled snapshot explicitly; renderer protocol substitution is not
        // part of the appearance source of truth.
        await materializeActiveAppearance(
          staging,
          template,
          job.input.sourceModelId,
          this.resolveCompiledAtlas,
        );
      }
      await this.writeAtlases(staging, job, template, options);
      if (isTexturePreview) {
        // A generated preview is a self-contained snapshot, not another
        // editable appearance library. Keeping the copied runtime directory
        // would reintroduce stale atlas overrides at pet-model:// load time.
        await rm(path.join(staging, ".appearance"), { recursive: true, force: true });
      }
      await this.renameManifest(staging, name);
      await rm(path.join(staging, "master.template.json"), { force: true });
      await rm(path.join(staging, "masks"), { recursive: true, force: true });
      // Capability data describes the source texture hash and must never be
      // copied into a derived model. The generated model is scanned afresh.
      await rm(path.join(staging, ".edit-capabilities"), { recursive: true, force: true });
      const result = await this.models.importFolder(staging);
      if (!result.model) {
        const message = result.report.errors[0]?.message ?? "生成模型导入失败";
        throw new Error(message);
      }
      if (previous) await this.models.select(previous.id);
      return result.model;
    } finally {
      await rm(staging, { recursive: true, force: true });
    }
  }

  private async writeAtlases(
    staging: string,
    job: CharacterGenerationJob,
    template: MasterTemplatePackage,
    options: GeneratedModelBuildOptions,
  ): Promise<void> {
    const slots = template.manifest.slots;
    const deltaReference = options.deltaReferencePath
      ? await readPng(options.deltaReferencePath)
      : undefined;
    for (const slot of options.preserveUnchanged ? [] : slots) {
      if (slot.required && !job.artifacts.slotTextures[slot.id]) {
        throw new Error(`缺少必要部件: ${slot.name}`);
      }
    }
    for (let textureIndex = 0; textureIndex < template.manifest.textureAtlases.length; textureIndex += 1) {
      const atlas = template.manifest.textureAtlases[textureIndex]!;
      const target = path.join(staging, atlas.file);
      const base = await readPng(target);
      if (base.width !== atlas.width || base.height !== atlas.height) {
        throw new Error(`母版图集尺寸与清单不一致: ${atlas.file}`);
      }
      const textureEdits = (options.textureEdits ?? []).filter((edit) => edit.atlasIndex === textureIndex);
      for (const edit of textureEdits) {
        await applyTextureEdit(base, edit);
      }
      const textures: MeshSlotTexture[] = [];
      const atlasSlots = slots.filter((item) =>
        item.atlas.texture === textureIndex &&
        (!options.preserveUnchanged || Boolean(job.artifacts.slotTextures[item.id]))
      );
      for (const slot of atlasSlots) {
        const file = job.artifacts.slotTextures[slot.id];
        textures.push({
          slot,
          // Missing completion meshes and template-only effects must both be
          // transparent. Preserving their old pixels leaks the mother character
          // into turns, blinks and pose transitions. A one-pixel source clears
          // the exact ArtMesh triangles without a canvas-sized allocation.
          image: file ? await readPng(file) : transparentPixel(),
          reference: file ? deltaReference : undefined,
          mask: await readPng(path.join(staging, slot.silhouetteMask)),
        });
      }
      await writePng(target, composeMeshAtlas(base, template.manifest.canvas, textures));
    }
  }

  private async renameManifest(staging: string, name: string): Promise<void> {
    const file = path.join(staging, "pet.manifest.json");
    const value = JSON.parse(await readFile(file, "utf8")) as Record<string, unknown>;
    value.name = name.trim() || "AI 角色";
    await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  }
}

async function materializeActiveAppearance(
  staging: string,
  template: MasterTemplatePackage,
  modelId: string | null,
  resolveCompiledAtlas: CompiledAtlasResolver | undefined,
): Promise<void> {
  for (const atlas of template.manifest.textureAtlases) {
    const targetAtlas = path.join(staging, atlas.file);
    if (modelId && resolveCompiledAtlas) {
      const compiled = await resolveCompiledAtlas(modelId, atlas.file);
      if (compiled) await copyFile(compiled, targetAtlas);
      continue;
    }
    // One-way compatibility for old tests/imports. Production injects the
    // compiled-atlas resolver and never treats this directory as authority.
    const runtimeAtlas = path.join(staging, ".appearance", "runtime", atlas.file);
    try {
      await copyFile(runtimeAtlas, targetAtlas);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
}

export async function applyTextureEdit(
  atlas: RgbaImage,
  edit: NonNullable<GeneratedModelBuildOptions["textureEdits"]>[number],
): Promise<void> {
  const [patch, mask] = await Promise.all([readPng(edit.editedPatchPath), readPng(edit.maskPath)]);
  const bounds = edit.bounds;
  if (
    patch.width !== bounds.width || patch.height !== bounds.height ||
    mask.width !== bounds.width || mask.height !== bounds.height
  ) throw new Error(`纹理 ${edit.textureUnitId} 的结果或蒙版尺寸不一致`);
  if (
    bounds.x < 0 || bounds.y < 0 ||
    bounds.x + bounds.width > atlas.width || bounds.y + bounds.height > atlas.height
  ) throw new Error(`纹理 ${edit.textureUnitId} 超出图集范围`);
  for (let y = 0; y < bounds.height; y += 1) {
    for (let x = 0; x < bounds.width; x += 1) {
      const localOffset = (y * bounds.width + x) * 4;
      if (mask.data[localOffset + 3]! === 0) continue;
      const atlasOffset = ((bounds.y + y) * atlas.width + bounds.x + x) * 4;
      applyTexturePatchPixel(
        atlas.data,
        atlasOffset,
        patch.data,
        localOffset,
        mask.data[localOffset + 3]!,
      );
    }
  }
}

async function readPng(file: string): Promise<RgbaImage> {
  const value = PNG.sync.read(await readFile(file));
  return { width: value.width, height: value.height, data: new Uint8Array(value.data) };
}

async function writePng(file: string, image: RgbaImage): Promise<void> {
  const value = new PNG({ width: image.width, height: image.height });
  value.data.set(image.data);
  await writeFile(file, PNG.sync.write(value));
}

function transparentPixel(): RgbaImage {
  return { width: 1, height: 1, data: new Uint8Array(4) };
}
