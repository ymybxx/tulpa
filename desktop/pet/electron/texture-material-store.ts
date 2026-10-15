import { createHash, randomUUID } from "node:crypto";
import { access, copyFile, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import type {
  ModelEditCapabilityManifest,
  TextureAtlasDelta,
  TextureMaterial,
  TextureMaterialChangeResult,
  TextureMaterialLibrary,
} from "../src/shared/contracts";
import { writeMaskedTexturePreview } from "./masked-texture-preview";
import type { ModelStore } from "./model-store";
import {
  AtlasCompactionCoordinator,
  type AtlasCompactionRequest,
} from "./atlas-compaction-coordinator";
import type { TextureAtlasEdit } from "./texture-atlas-composer";
import { executeAtlasCompactionRequest } from "./atlas-compaction-job";
import { TextureFileIdentityCache } from "./texture-file-identity-cache";
import {
  buildTexturePatchSnapshot,
  type TexturePatchLibraryView,
} from "./texture-patch-snapshot";

const APPEARANCE_DIRECTORY = ".appearance";
const MATERIAL_DIRECTORY = `${APPEARANCE_DIRECTORY}/materials`;
const LIBRARY_FILE = "library.json";
const JOURNAL_MASK_DIRECTORY = `${APPEARANCE_DIRECTORY}/journal-assets/masks`;

export interface SaveTextureMaterialEdit {
  textureUnitId: string;
  label: string;
  atlasIndex: number;
  atlasFile: string;
  bounds: { x: number; y: number; width: number; height: number };
  maskPath: string;
  editedPatchPath: string;
}

export interface SaveTextureMaterialInput {
  modelId: string;
  name: string;
  prompt: string;
  sourceHash: string;
  edits: SaveTextureMaterialEdit[];
  /** False keeps the current look and only adds the versions to 换色. */
  apply?: boolean;
  origin?: "recolor";
}

export interface TextureMaterialStoreChangeResult extends TextureMaterialChangeResult {
  affectedAtlasFiles: string[];
  affectedTextureUnitIds: string[];
}

interface StoredTextureMaterial {
  id: string;
  textureUnitId: string;
  label: string;
  atlasIndex: number;
  atlasFile: string;
  bounds: { x: number; y: number; width: number; height: number };
  name: string;
  prompt: string;
  createdAt: string;
  imageFile: string;
  maskFile: string;
  contentHash: string;
  /** Made by 换色 (no AI); absent for AI results and the original. */
  origin?: "recolor";
}

interface StoredTextureMaterialLibrary {
  schemaVersion: 4;
  modelId: string;
  revision: number;
  materials: StoredTextureMaterial[];
  activeAssignments: Record<string, string>;
  baseContentHashes: Record<string, string>;
  updatedAt: string;
}

interface StoredTextureMaterialLibraryV3 {
  schemaVersion: 3;
  modelId: string;
  revision: number;
  variants: Array<{
    id: string;
    textureUnitId: string;
    label: string;
    atlasIndex: number;
    atlasFile: string;
    bounds: { x: number; y: number; width: number; height: number };
    name: string;
    prompt: string;
    createdAt: string;
    patchFile: string;
    maskFile: string;
    previewFile?: string;
  }>;
  activeAssignments: Record<string, string>;
  updatedAt: string;
}

export class TextureMaterialStore {
  private readonly mutations = new Map<string, Promise<void>>();
  private readonly identities = new TextureFileIdentityCache();

  constructor(
    private readonly models: ModelStore,
    private readonly compaction = new AtlasCompactionCoordinator({
      compact: (request) => executeAtlasCompactionRequest(request).then(() => undefined),
    }),
  ) {}

  async list(
    modelId: string,
    capability?: ModelEditCapabilityManifest,
  ): Promise<TextureMaterialLibrary> {
    // Seeding/migration can write the journal, so it belongs to the same
    // per-model mutation lane as selection and render-snapshot migration.
    return this.mutate(modelId, async () => {
      await this.models.load(modelId);
      return this.toPublic(await this.ensureLibrary(modelId, capability));
    });
  }

  async appearanceSnapshot(
    modelId: string,
    affectedAtlasFiles?: readonly string[],
    streamId = officialStreamId(modelId),
  ) {
    // Snapshot preparation may migrate old journal state, so serialize it with
    // selection writes. This also prevents a bundle from observing half of a
    // concurrent material mutation.
    return this.mutate(modelId, async () => {
      const raw = await this.readRaw(modelId);
      if (!raw) {
        return {
          delta: {
            modelId,
            streamId,
            revision: 0,
            patchSets: [],
            atlases: [],
          } satisfies TextureAtlasDelta,
          contentKeys: new Map<string, string>(),
        };
      }

      let migrated = raw.schemaVersion === 3;
      let library = raw.schemaVersion === 3
        ? await this.migrateV3(modelId, raw)
        : raw;
      if (library.modelId !== modelId) {
        library = { ...library, modelId };
        migrated = true;
      }
      const prepared = await this.snapshotLibrary(modelId, library);
      const maskPathsChanged = library.materials.some((material, index) =>
        prepared.library.materials[index]?.maskFile !== material.maskFile
      );
      let snapshotLibrary = prepared.library;
      if (migrated || maskPathsChanged) {
        snapshotLibrary = {
          ...snapshotLibrary,
          modelId,
          revision: library.revision + 1,
          updatedAt: new Date().toISOString(),
        };
        // Immutable mask assets are copied before this atomic journal write.
        // Once a render bundle is returned, a capability rescan may safely
        // replace its mutable .edit-capabilities directory.
        await this.write(modelId, snapshotLibrary);
      }

      // A render bundle is authoritative for every currently assigned atlas,
      // including an atlas whose selected materials are all base materials.
      // Historical inactive materials must not pull obsolete atlas topology
      // back into a current snapshot.
      const atlasFiles = affectedAtlasFiles ?? unique(
        activeMaterials(snapshotLibrary).map((material) => material.atlasFile),
      );
      return this.buildSnapshot(modelId, snapshotLibrary, atlasFiles, streamId);
    });
  }

  async flushCompaction(timeoutMs: number): Promise<boolean> {
    return this.compaction.flush(timeoutMs);
  }

  private async buildSnapshot(
    modelId: string,
    library: TexturePatchLibraryView,
    atlasFiles: readonly string[],
    streamId: string,
  ) {
    return buildTexturePatchSnapshot(library, atlasFiles, {
      atlasIdentity: (atlasFile) => this.identities.identify(
        this.models.resolveModelFile(modelId, atlasFile),
      ),
      imageIdentity: (imageFile) => this.identities.identify(
        this.models.resolveModelFile(modelId, imageFile),
      ),
      maskIdentity: (maskFile) => this.identities.identify(
        this.models.resolveModelFile(modelId, maskFile),
      ),
      modelUrl: (relative, contentHash) => {
        const encoded = relative.split("/").map(encodeURIComponent).join("/");
        return `pet-model://${modelId}/${encoded}?content=${encodeURIComponent(contentHash)}`;
      },
    }, streamId);
  }

  async saveAndApply(input: SaveTextureMaterialInput): Promise<TextureMaterialStoreChangeResult> {
    return this.mutate(input.modelId, async () => {
      if (!input.edits.length) throw new Error("没有已接受的纹理可保存");
      const current = await this.readV4(input.modelId);
      const createdRoots: string[] = [];
      try {
        const createdAt = new Date().toISOString();
        const materials: StoredTextureMaterial[] = [];
        for (const edit of input.edits) {
          const id = `material-${randomUUID()}`;
          const root = this.materialRoot(input.modelId, id);
          createdRoots.push(root);
          await mkdir(root, { recursive: true });
          const imageFile = `${MATERIAL_DIRECTORY}/${id}/image.png`;
          const target = this.models.resolveModelFile(input.modelId, imageFile);
          await writeMaskedTexturePreview(edit.editedPatchPath, edit.maskPath, target);
          const maskFile = await this.ensureJournalMask(
            input.modelId,
            modelRelativePath(input.modelId, edit.maskPath, this.models),
          );
          materials.push({
            id,
            textureUnitId: edit.textureUnitId,
            label: edit.label,
            atlasIndex: edit.atlasIndex,
            atlasFile: edit.atlasFile,
            bounds: edit.bounds,
            name: cleanName(input.name, edit.label),
            prompt: input.prompt.trim(),
            createdAt,
            imageFile,
            maskFile,
            contentHash: await hashFile(target),
            ...input.origin ? { origin: input.origin } : {},
          });
        }
        const affectedAtlasFiles = unique(materials.map((item) => item.atlasFile));
        const next: StoredTextureMaterialLibrary = {
          ...current,
          revision: current.revision + 1,
          materials: [...current.materials, ...materials],
          activeAssignments: input.apply === false ? current.activeAssignments : {
            ...current.activeAssignments,
            ...Object.fromEntries(materials.map((item) => [item.textureUnitId, item.id])),
          },
          updatedAt: new Date().toISOString(),
        };
        const delta = await this.commitJournal(input.modelId, next, affectedAtlasFiles);
        return this.changeResult(next, delta, affectedAtlasFiles, materials.map((item) => item.textureUnitId));
      } catch (error) {
        await Promise.all(createdRoots.map((root) => rm(root, { recursive: true, force: true })));
        throw error;
      }
    });
  }

  async select(
    modelId: string,
    textureUnitId: string,
    materialId: string,
  ): Promise<TextureMaterialStoreChangeResult> {
    return this.mutate(modelId, async () => {
      const current = await this.readV4(modelId);
      const material = current.materials.find((item) => item.id === materialId);
      if (!material) throw new Error("素材不存在");
      if (material.textureUnitId !== textureUnitId) throw new Error("素材与当前纹理不匹配");
      if (current.activeAssignments[textureUnitId] === materialId) {
        return this.changeResult(current, emptyDelta(modelId, current.revision), [], []);
      }
      const next: StoredTextureMaterialLibrary = {
        ...current,
        revision: current.revision + 1,
        activeAssignments: { ...current.activeAssignments, [textureUnitId]: materialId },
        updatedAt: new Date().toISOString(),
      };
      const affectedAtlasFiles = [material.atlasFile];
      const delta = await this.commitJournal(modelId, next, affectedAtlasFiles);
      return this.changeResult(next, delta, affectedAtlasFiles, [textureUnitId]);
    });
  }

  /** Puts on several materials as one change: one revision and one atlas rebuild for a whole outfit. */
  async selectMany(
    modelId: string,
    assignments: ReadonlyArray<{ textureUnitId: string; materialId: string }>,
  ): Promise<TextureMaterialStoreChangeResult> {
    return this.mutate(modelId, async () => {
      const current = await this.readV4(modelId);
      const byId = new Map(current.materials.map((item) => [item.id, item]));
      const activeAssignments = { ...current.activeAssignments };
      const atlasFiles = new Set<string>();
      const units: string[] = [];
      for (const { textureUnitId, materialId } of assignments) {
        const material = byId.get(materialId);
        if (!material) throw new Error("素材不存在");
        if (material.textureUnitId !== textureUnitId) throw new Error("素材与当前纹理不匹配");
        if (activeAssignments[textureUnitId] === materialId) continue;
        activeAssignments[textureUnitId] = materialId;
        atlasFiles.add(material.atlasFile);
        units.push(textureUnitId);
      }
      if (!units.length) return this.changeResult(current, emptyDelta(modelId, current.revision), [], []);
      const next: StoredTextureMaterialLibrary = {
        ...current,
        revision: current.revision + 1,
        activeAssignments,
        updatedAt: new Date().toISOString(),
      };
      const affectedAtlasFiles = [...atlasFiles];
      const delta = await this.commitJournal(modelId, next, affectedAtlasFiles);
      return this.changeResult(next, delta, affectedAtlasFiles, units);
    });
  }

  async delete(modelId: string, materialId: string): Promise<TextureMaterialStoreChangeResult> {
    return this.mutate(modelId, async () => {
      const current = await this.readV4(modelId);
      const material = current.materials.find((item) => item.id === materialId);
      if (!material) throw new Error("素材不存在");
      if (material.id === initialMaterialId(modelId, material.textureUnitId)) {
        throw new Error("模型原始素材必须保留，不能删除");
      }
      if (current.activeAssignments[material.textureUnitId] === materialId) {
        throw new Error("当前使用中的素材不能删除，请先切换到其他素材");
      }
      const next: StoredTextureMaterialLibrary = {
        ...current,
        revision: current.revision + 1,
        materials: current.materials.filter((item) => item.id !== materialId),
        updatedAt: new Date().toISOString(),
      };
      await this.write(modelId, next);
      await rm(this.materialRoot(modelId, materialId), { recursive: true, force: true });
      return this.changeResult(next, emptyDelta(modelId, next.revision), [], []);
    });
  }

  async activeAppearanceKey(modelId: string, sourceHash: string): Promise<string> {
    await this.waitForMutation(modelId);
    const library = await this.readV4(modelId);
    const assignments = Object.entries(library.activeAssignments)
      .sort(([left], [right]) => left.localeCompare(right));
    return createHash("sha256")
      .update(JSON.stringify({ sourceHash, assignments, rendererVersion: 1 }))
      .digest("hex");
  }

  async resolveActiveMaterialImage(modelId: string, textureUnitId: string): Promise<string> {
    await this.waitForMutation(modelId);
    const library = await this.readV4(modelId);
    const materialId = library.activeAssignments[textureUnitId];
    const material = library.materials.find((item) => item.id === materialId);
    if (!material) throw new Error(`纹理 ${textureUnitId} 没有当前素材`);
    return this.models.resolveModelFile(modelId, material.imageFile);
  }

  async resolveMaterialImage(
    modelId: string,
    textureUnitId: string,
    materialId: string,
  ): Promise<string> {
    await this.waitForMutation(modelId);
    const library = await this.readV4(modelId);
    const material = library.materials.find((item) =>
      item.id === materialId && item.textureUnitId === textureUnitId
    );
    if (!material) throw new Error("所选参考素材已经不存在，请重新选择");
    return this.models.resolveModelFile(modelId, material.imageFile);
  }

  async resolveCompiledAtlas(modelId: string, baseFile: string): Promise<string | null> {
    await this.waitForMutation(modelId);
    const modelRoot = this.models.resolveModelRoot(modelId);
    const relative = path.relative(modelRoot, baseFile).replaceAll(path.sep, "/");
    if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) return null;
    const raw = await this.readRaw(modelId);
    if (raw?.schemaVersion !== 4 || !raw.materials.some((item) => item.atlasFile === relative)) {
      return null;
    }
    const { library } = await this.snapshotLibrary(modelId, raw);
    const snapshot = await this.buildSnapshot(
      modelId,
      library,
      [relative],
      officialStreamId(modelId),
    );
    const request = this.compactionRequest(modelId, library, snapshot.contentKeys, snapshot.delta);
    const atlas = request.atlases[0];
    if (!atlas) return null;
    try {
      await access(atlas.targetPath);
      return atlas.targetPath;
    } catch {
      await this.compaction.ensure(request);
      await access(atlas.targetPath);
      return atlas.targetPath;
    }
  }

  private async ensureLibrary(
    modelId: string,
    capability?: ModelEditCapabilityManifest,
  ): Promise<StoredTextureMaterialLibrary> {
    const raw = await this.readRaw(modelId);
    if (raw?.schemaVersion === 4) {
      return capability ? this.ensureInitialMaterials(modelId, raw, capability) : raw;
    }
    if (!capability) throw new Error("初始化素材库需要模型纹理能力清单");
    const migrated = raw?.schemaVersion === 3
      ? await this.migrateV3(modelId, raw)
      : emptyLibrary(modelId);
    return this.ensureInitialMaterials(modelId, migrated, capability);
  }

  private async ensureInitialMaterials(
    modelId: string,
    library: StoredTextureMaterialLibrary,
    capability: ModelEditCapabilityManifest,
  ): Promise<StoredTextureMaterialLibrary> {
    const materials = [...library.materials];
    const activeAssignments = { ...library.activeAssignments };
    const baseContentHashes = { ...library.baseContentHashes };
    const duplicateInitialRoots = new Set<string>();
    let changed = library.modelId !== modelId;
    for (const unit of capability.textureUnits) {
      const initialId = initialMaterialId(modelId, unit.id);
      const originalImageFile = unit.originalPreviewPath ?? unit.previewPath;
      const baseContentHash = `base:${unit.maskHash}`;
      for (let index = 0; index < materials.length; index += 1) {
        const item = materials[index]!;
        if (item.textureUnitId !== unit.id) continue;
        const initial = item.id === initialId;
        const maskFile = initial
          ? unit.maskPath
          : await this.ensureStoredMaterialMask(modelId, item.maskFile, unit.maskPath);
        const sanitizedName = !initial && /^AI 素材\s*[·・:：-]/.test(item.name)
          ? "AI 素材"
          : item.name;
        if (initial && item.imageFile !== originalImageFile) {
          duplicateInitialRoots.add(this.materialRoot(modelId, initialId));
        }
        if (
          item.maskFile !== maskFile ||
          item.atlasFile !== unit.atlasFile ||
          item.atlasIndex !== unit.atlasIndex ||
          item.name !== sanitizedName ||
          (initial && (
            item.imageFile !== originalImageFile ||
            item.contentHash !== baseContentHash
          ))
        ) {
          materials[index] = {
            ...item,
            label: unit.label,
            atlasIndex: unit.atlasIndex,
            atlasFile: unit.atlasFile,
            bounds: unit.bounds,
            maskFile,
            name: sanitizedName,
            ...(initial ? { imageFile: originalImageFile, contentHash: baseContentHash } : {}),
          };
          changed = true;
        }
      }
      if (!materials.some((item) => item.id === initialId)) {
        materials.push({
          id: initialId,
          textureUnitId: unit.id,
          label: unit.label,
          atlasIndex: unit.atlasIndex,
          atlasFile: unit.atlasFile,
          bounds: unit.bounds,
          name: `${unit.label} 素材`,
          prompt: "",
          createdAt: capability.generatedAt,
          imageFile: originalImageFile,
          maskFile: unit.maskPath,
          contentHash: baseContentHash,
        });
        activeAssignments[unit.id] ??= initialId;
        changed = true;
      }
      if (baseContentHashes[unit.id] !== baseContentHash) {
        baseContentHashes[unit.id] = baseContentHash;
        changed = true;
      }
      const available = materials.filter((item) => item.textureUnitId === unit.id);
      if (!available.some((item) => item.id === activeAssignments[unit.id])) {
        activeAssignments[unit.id] = available[0]!.id;
        changed = true;
      }
    }
    if (!changed) return library;
    const next: StoredTextureMaterialLibrary = {
      ...library,
      modelId,
      revision: library.revision + 1,
      materials,
      activeAssignments,
      baseContentHashes,
      updatedAt: new Date().toISOString(),
    };
    await this.write(modelId, next);
    void this.cleanupDuplicateRoots([...duplicateInitialRoots]);
    const materialById = new Map(next.materials.map((item) => [item.id, item]));
    const active = Object.entries(next.activeAssignments).flatMap(([unitId, materialId]) => {
      const material = materialById.get(materialId);
      return material?.textureUnitId === unitId ? [material] : [];
    });
    const redundantRuntimeAtlases = unique(capability.textureUnits.map((unit) => unit.atlasFile))
      .filter((atlasFile) => !active.some((material) =>
        material.atlasFile === atlasFile &&
        material.contentHash !== next.baseContentHashes[material.textureUnitId]
      ));
    await this.cleanupRuntimeAtlases(modelId, redundantRuntimeAtlases);
    return next;
  }

  private async cleanupDuplicateRoots(roots: readonly string[]): Promise<void> {
    for (const root of roots) {
      try {
        await rm(root, { recursive: true, force: true });
      } catch {
        // Derived original previews are recoverable; retry on the next library load.
      }
    }
  }

  private async cleanupRuntimeAtlases(modelId: string, atlasFiles: readonly string[]): Promise<void> {
    for (const atlasFile of atlasFiles) {
      try {
        await rm(path.join(this.appearanceRoot(modelId), "runtime", atlasFile), { force: true });
      } catch {
        // Runtime atlases are derived and will be retried on the next load.
      }
    }
  }

  private async migrateV3(
    modelId: string,
    legacy: StoredTextureMaterialLibraryV3,
  ): Promise<StoredTextureMaterialLibrary> {
    const materials: StoredTextureMaterial[] = [];
    for (const variant of legacy.variants) {
      const id = variant.id.replace(/^variant-/, "material-");
      await mkdir(this.materialRoot(modelId, id), { recursive: true });
      const imageFile = `${MATERIAL_DIRECTORY}/${id}/image.png`;
      const target = this.models.resolveModelFile(modelId, imageFile);
      await writeMaskedTexturePreview(
        this.models.resolveModelFile(modelId, variant.patchFile),
        this.models.resolveModelFile(modelId, variant.maskFile),
        target,
      );
      materials.push({
        id,
        textureUnitId: variant.textureUnitId,
        label: variant.label,
        atlasIndex: variant.atlasIndex,
        atlasFile: variant.atlasFile,
        bounds: variant.bounds,
        name: variant.name,
        prompt: variant.prompt,
        createdAt: variant.createdAt,
        imageFile,
        maskFile: variant.maskFile,
        contentHash: await hashFile(target),
      });
    }
    return {
      schemaVersion: 4,
      modelId,
      revision: legacy.revision,
      materials,
      activeAssignments: Object.fromEntries(Object.entries(legacy.activeAssignments).map(
        ([unitId, id]) => [unitId, id.replace(/^variant-/, "material-")],
      )),
      baseContentHashes: {},
      updatedAt: legacy.updatedAt,
    };
  }

  private async commitJournal(
    modelId: string,
    library: StoredTextureMaterialLibrary,
    affectedAtlasFiles: readonly string[],
  ): Promise<TextureAtlasDelta> {
    const { library: snapshotLibrary, activeCustomMaterials } =
      await this.snapshotLibrary(modelId, library);
    const snapshot = await this.buildSnapshot(
      modelId,
      snapshotLibrary,
      unique(affectedAtlasFiles),
      officialStreamId(modelId),
    );
    await this.write(modelId, snapshotLibrary);
    try {
      const compactionAtlasFiles = unique(
        activeCustomMaterials.map((material) => material.atlasFile),
      ).sort();
      const compactionSnapshot = sameStrings(
        compactionAtlasFiles,
        unique(affectedAtlasFiles).sort(),
      )
        ? snapshot
        : await this.buildSnapshot(
            modelId,
            snapshotLibrary,
            compactionAtlasFiles,
            officialStreamId(modelId),
          );
      const request = this.compactionRequest(
        modelId,
        snapshotLibrary,
        compactionSnapshot.contentKeys,
        compactionSnapshot.delta,
      );
      if (request.atlases.length > 0) this.compaction.schedule(request);
    } catch (error) {
      // The journal and renderer snapshot are authoritative. A derived full
      // atlas cache failure must never make the already-persisted selection
      // appear to have failed or roll it back.
      console.warn("安排 Live2D 图集后台压实失败", error);
    }
    return snapshot.delta;
  }

  private async snapshotLibrary(
    modelId: string,
    library: StoredTextureMaterialLibrary,
  ): Promise<{
    library: StoredTextureMaterialLibrary;
    activeCustomMaterials: StoredTextureMaterial[];
  }> {
    const materialById = new Map(library.materials.map((item) => [item.id, item]));
    const active = Object.entries(library.activeAssignments).flatMap(([unitId, materialId]) => {
      const material = materialById.get(materialId);
      return material?.textureUnitId === unitId ? [material] : [];
    });
    const activeCustomMaterials = active.filter((item) =>
      item.contentHash !== library.baseContentHashes[item.textureUnitId]
    );
    const immutableMasks = new Map<string, string>();
    await Promise.all(activeCustomMaterials.map(async (material) => {
      immutableMasks.set(material.id, await this.ensureJournalMask(modelId, material.maskFile));
    }));
    return {
      library: {
        ...library,
        materials: library.materials.map((material) => ({
          ...material,
          maskFile: immutableMasks.get(material.id) ?? material.maskFile,
        })),
      },
      activeCustomMaterials,
    };
  }

  private compactionRequest(
    modelId: string,
    library: TexturePatchLibraryView,
    contentKeys: ReadonlyMap<string, string>,
    delta: TextureAtlasDelta,
  ): AtlasCompactionRequest {
    const materialById = new Map(library.materials.map((item) => [item.id, item]));
    const active = Object.entries(library.activeAssignments).flatMap(([unitId, materialId]) => {
      const material = materialById.get(materialId);
      return material?.textureUnitId === unitId ? [material] : [];
    });
    const modelRoot = this.models.resolveModelRoot(modelId);
    return {
      modelId,
      revision: library.revision,
      allowedRoots: [modelRoot],
      atlases: delta.patchSets.flatMap((patchSet) => {
        const contentKey = contentKeys.get(patchSet.atlasFile);
        if (!contentKey || patchSet.patches.length === 0) return [];
        return [{
          contentKey,
          atlasFile: patchSet.atlasFile,
          sourcePath: this.models.resolveModelFile(modelId, patchSet.atlasFile),
          targetPath: path.join(
            this.appearanceRoot(modelId),
            "compiled",
            contentKey,
            patchSet.atlasFile,
          ),
          edits: active
            .filter((item) => item.atlasFile === patchSet.atlasFile)
            .filter((item) => item.contentHash !== library.baseContentHashes[item.textureUnitId])
            .sort((left, right) => left.textureUnitId.localeCompare(right.textureUnitId))
            .map((item): TextureAtlasEdit => ({
              textureUnitId: item.textureUnitId,
              atlasIndex: item.atlasIndex,
              bounds: item.bounds,
              maskPath: this.models.resolveModelFile(modelId, item.maskFile),
              imagePath: this.models.resolveModelFile(modelId, item.imageFile),
            })),
        }];
      }),
    };
  }

  private async readV4(modelId: string): Promise<StoredTextureMaterialLibrary> {
    const value = await this.readRaw(modelId);
    if (value?.schemaVersion !== 4) throw new Error("模型素材库尚未完成初始化");
    return value;
  }

  private async readRaw(
    modelId: string,
  ): Promise<StoredTextureMaterialLibrary | StoredTextureMaterialLibraryV3 | null> {
    try {
      const value = JSON.parse(await readFile(
        path.join(this.appearanceRoot(modelId), LIBRARY_FILE),
        "utf8",
      )) as StoredTextureMaterialLibrary | StoredTextureMaterialLibraryV3;
      if (value.schemaVersion === 4 && Array.isArray(value.materials)) return value;
      if (value.schemaVersion === 3 && Array.isArray(value.variants)) return value;
      throw new Error("模型素材库格式无效");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw error;
    }
  }

  private async write(modelId: string, library: StoredTextureMaterialLibrary): Promise<void> {
    const root = this.appearanceRoot(modelId);
    await mkdir(root, { recursive: true });
    const target = path.join(root, LIBRARY_FILE);
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, `${JSON.stringify(library, null, 2)}\n`, "utf8");
    await rename(temporary, target);
  }

  private async ensureJournalMask(modelId: string, relativePath: string): Promise<string> {
    const source = this.models.resolveModelFile(modelId, relativePath);
    const identity = await this.identities.identify(source);
    const relative = `${JOURNAL_MASK_DIRECTORY}/${identity.sha256}.png`;
    const target = this.models.resolveModelFile(modelId, relative);
    try {
      await access(target);
      return relative;
    } catch {
      // Copy the immutable journal asset below.
    }
    await mkdir(path.dirname(target), { recursive: true });
    const temporary = `${target}.${randomUUID()}.tmp`;
    try {
      await copyFile(source, temporary);
      await rename(temporary, target);
    } finally {
      await rm(temporary, { force: true });
    }
    return relative;
  }

  private async ensureStoredMaterialMask(
    modelId: string,
    currentPath: string,
    capabilityPath: string,
  ): Promise<string> {
    try {
      return await this.ensureJournalMask(modelId, currentPath);
    } catch (error) {
      if (currentPath === capabilityPath) throw error;
      return this.ensureJournalMask(modelId, capabilityPath);
    }
  }

  private toPublic(library: StoredTextureMaterialLibrary): TextureMaterialLibrary {
    return {
      schemaVersion: 4,
      modelId: library.modelId,
      revision: library.revision,
      activeAssignments: { ...library.activeAssignments },
      updatedAt: library.updatedAt,
      materials: library.materials.map((item): TextureMaterial => ({
        id: item.id,
        textureUnitId: item.textureUnitId,
        label: item.label,
        atlasIndex: item.atlasIndex,
        atlasFile: item.atlasFile,
        bounds: item.bounds,
        name: item.name,
        prompt: item.prompt,
        createdAt: item.createdAt,
        active: library.activeAssignments[item.textureUnitId] === item.id,
        original: item.id === initialMaterialId(library.modelId, item.textureUnitId),
        ...item.origin ? { origin: item.origin } : {},
        imageUrl: `pet-model://${library.modelId}/${item.imageFile
          .split("/").map(encodeURIComponent).join("/")}`,
      })),
    };
  }

  private changeResult(
    library: StoredTextureMaterialLibrary,
    delta: TextureAtlasDelta,
    affectedAtlasFiles: readonly string[],
    affectedTextureUnitIds: readonly string[],
  ): TextureMaterialStoreChangeResult {
    return {
      library: this.toPublic(library),
      delta,
      affectedAtlasFiles: [...affectedAtlasFiles],
      affectedTextureUnitIds: [...new Set(affectedTextureUnitIds)],
    };
  }

  private async waitForMutation(modelId: string): Promise<void> {
    await this.mutations.get(modelId);
  }

  private mutate<T>(modelId: string, operation: () => Promise<T>): Promise<T> {
    const previous = this.mutations.get(modelId) ?? Promise.resolve();
    const result = previous.catch(() => undefined).then(operation);
    const tail = result.then(() => undefined, () => undefined);
    this.mutations.set(modelId, tail);
    void tail.finally(() => {
      if (this.mutations.get(modelId) === tail) this.mutations.delete(modelId);
    });
    return result;
  }

  private appearanceRoot(modelId: string): string {
    return path.join(this.models.resolveModelRoot(modelId), APPEARANCE_DIRECTORY);
  }

  private materialRoot(modelId: string, materialId: string): string {
    if (!/^material-[a-zA-Z0-9-]+$/.test(materialId)) throw new Error("素材 ID 无效");
    return path.join(this.appearanceRoot(modelId), "materials", materialId);
  }
}

function emptyLibrary(modelId: string): StoredTextureMaterialLibrary {
  return {
    schemaVersion: 4,
    modelId,
    revision: 0,
    materials: [],
    activeAssignments: {},
    baseContentHashes: {},
    updatedAt: new Date().toISOString(),
  };
}

function emptyDelta(modelId: string, revision: number): TextureAtlasDelta {
  return { modelId, streamId: officialStreamId(modelId), revision, patchSets: [], atlases: [] };
}

function officialStreamId(modelId: string): string {
  return `model:${modelId}`;
}

function initialMaterialId(modelId: string, textureUnitId: string): string {
  return `material-${createHash("sha256").update(`${modelId}:${textureUnitId}`).digest("hex").slice(0, 24)}`;
}

function modelRelativePath(modelId: string, file: string, models: ModelStore): string {
  if (!path.isAbsolute(file)) return file.replaceAll("\\", "/");
  const relative = path.relative(models.resolveModelRoot(modelId), file);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("纹理蒙版不属于当前模型");
  }
  return relative.replaceAll("\\", "/");
}

function cleanName(value: string, fallback: string): string {
  return value.trim().replace(/\s+/g, " ").slice(0, 80) || fallback;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function activeMaterials(library: TexturePatchLibraryView): TexturePatchLibraryView["materials"] {
  const materialById = new Map(library.materials.map((material) => [material.id, material]));
  return Object.entries(library.activeAssignments).flatMap(([textureUnitId, materialId]) => {
    const material = materialById.get(materialId);
    return material?.textureUnitId === textureUnitId ? [material] : [];
  });
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

async function hashFile(file: string): Promise<string> {
  return createHash("sha256").update(await readFile(file)).digest("hex");
}
