import { createHash } from "node:crypto";
import { access, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import { PNG } from "pngjs";

import {
  type ModelEditCapabilityManifest,
  type ModelEditPartCapability,
  type ModelEditPartGroup,
  type ModelPackage,
  type MotionAsset,
  type TextureEditUnit,
} from "../src/shared/contracts";
import {
  type MasterTemplateManifest,
  type MasterTemplatePartGroup,
  type MasterTemplateSlot,
} from "../src/shared/character-generation";
import { inspectCubismModel, type InspectedCubismModel } from "./cubism-core-inspector";
import { decodePng } from "./png-reader";
import { classifyPart, type PartNode, type SemanticPart } from "./part-classifier";
import type { MasterTemplatePackage } from "./character-generation/master-template-store";
import type { ModelStore } from "./model-store";

const CAPABILITY_DIRECTORY = ".edit-capabilities";
const CAPABILITY_FILE = "manifest.json";
const TEMPLATE_FILE = "template.json";
const GUIDE_FILE = `${CAPABILITY_DIRECTORY}/alignment-guide.png`;
const CURRENT_ATLAS_DIRECTORY = `${CAPABILITY_DIRECTORY}/current-atlases`;
const GUIDE_RENDER_SNAPSHOT_FILE = `${CAPABILITY_DIRECTORY}/guide-render-snapshot.json`;
const APPEARANCE_STATE_FILE = `${CAPABILITY_DIRECTORY}/appearance-state.json`;
const GUIDE_RENDERER_VERSION = 1;

interface CubismModelJson {
  FileReferences?: {
    Moc?: unknown;
    Textures?: unknown;
    Pose?: unknown;
    DisplayInfo?: unknown;
    UserData?: unknown;
  };
  Groups?: unknown;
}

interface AtlasImage {
  index: number;
  relative: string;
  image: PNG;
}

interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

interface SlotAnalysis {
  slot: MasterTemplateSlot;
  semantic: SemanticPart;
  mask: PNG;
  visibleBounds: Bounds;
  meshBounds: Bounds;
  visiblePixels: number;
  revealedByMotionIds: string[];
  /** Drawable ID plus every Part ID and display name above it, for name search. */
  searchText: string;
  /** Never drawn itself; it only gives other pieces their clipping shape. */
  maskOnly: boolean;
}


type ActiveModelFileResolver = (modelId: string, baseFile: string) => Promise<string | null>;

interface ModelEditSource {
  modelPackage: ModelPackage;
  modelRoot: string;
  entryPath: string;
  entryDirectory: string;
  modelJson: CubismModelJson;
  references: NonNullable<CubismModelJson["FileReferences"]>;
  mocPath: string;
  baseTexturePaths: string[];
  activeTexturePaths: string[];
  motionPaths: string[];
  sourceHash: string;
}

interface GuideRenderSnapshot {
  schemaVersion: 1;
  rendererVersion: number;
  model: InspectedCubismModel;
  hiddenPoseParts: string[];
  size: { width: number; height: number };
}

interface AppearanceState {
  schemaVersion: 2;
  rendererVersion: number;
  appearanceKey: string;
  generatedAt: string;
}

type AppearanceKeyResolver = (modelId: string, sourceHash: string) => Promise<string>;

export class ModelEditCapabilityStore {
  private readonly scans = new Map<string, Promise<ModelEditCapabilityManifest>>();
  private readonly appearanceRefreshes = new Map<string, Promise<ModelEditCapabilityManifest>>();
  private readonly previewBuilds = new Map<string, Promise<void>>();
  private readonly previewAtlases = new Map<string, Promise<PNG>>();
  private previewBuildsActive = 0;
  private readonly previewBuildWaiters: Array<() => void> = [];

  constructor(
    private readonly models: ModelStore,
    private readonly corePath: string,
    private readonly resolveCompiledAtlas?: ActiveModelFileResolver,
    private readonly resolveAppearanceKey?: AppearanceKeyResolver,
  ) {}

  async get(modelId: string): Promise<ModelEditCapabilityManifest | null> {
    try {
      const value = JSON.parse(await readFile(this.capabilityPath(modelId), "utf8")) as unknown;
      return parseCapabilityManifest(value, modelId);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
      if (error instanceof Error && error.message.includes("请重新预检")) return null;
      throw error;
    }
  }

  /**
   * Returns the one authoritative appearance snapshot for a model.
   * Missing and obsolete manifests are rebuilt here so callers never need to
   * implement their own get/scan fallback (and accidentally keep stale data).
   */
  async current(modelId: string): Promise<ModelEditCapabilityManifest> {
    const cached = await this.get(modelId);
    if (!cached) return await this.scan(modelId);
    // A cached capability is structural: validate it from model/Moc/base-atlas
    // metadata only. Resolving the current compiled appearance here can wait on
    // multi-second background PNG compaction even though appearance pixels
    // cannot change topology.
    const source = await this.loadSource(modelId, false);
    if (cached.sourceHash !== source.sourceHash) return await this.scan(modelId);
    return cached;
  }

  async ensureAlignmentGuide(modelId: string): Promise<{ path: string; appearanceKey: string }> {
    const capability = await this.current(modelId);
    const source = await this.loadSource(modelId);
    const appearanceKey = await this.appearanceKey(modelId, capability, source);
    const state = await this.readAppearanceState(modelId);
    const guidePath = path.join(source.modelRoot, GUIDE_FILE);
    if (appearanceStateMatches(state, appearanceKey)) {
      try {
        await access(guidePath);
        return { path: guidePath, appearanceKey };
      } catch {
        // Regenerate the derived cache below.
      }
    }
    await this.refreshAppearance(
      modelId,
      source.baseTexturePaths.map((file) => modelRelative(source.modelRoot, file)),
      [],
      appearanceKey,
    );
    return { path: guidePath, appearanceKey };
  }

  async ensureTexturePreview(modelId: string, relativePath: string): Promise<void> {
    const normalized = normalizeModelRelative(relativePath);
    const capability = await this.current(modelId);
    const unit = capability.textureUnits.find((candidate) =>
      candidate.originalPreviewPath === normalized || candidate.previewPath === normalized
    );
    if (!unit) return;
    const target = this.models.resolveModelFile(modelId, normalized);
    try {
      await access(target);
      return;
    } catch {
      // Build the deterministic derived preview below.
    }
    const key = `${modelId}:${normalized}`;
    const existing = this.previewBuilds.get(key);
    if (existing) return await existing;
    const build = this.withPreviewBuildSlot(() =>
      this.buildTexturePreview(modelId, unit, normalized, target)
    );
    this.previewBuilds.set(key, build);
    await build.finally(() => {
      if (this.previewBuilds.get(key) === build) this.previewBuilds.delete(key);
    });
  }

  private async buildTexturePreview(
    modelId: string,
    unit: TextureEditUnit,
    relativePath: string,
    target: string,
  ): Promise<void> {
    const baseAtlasPath = this.models.resolveModelFile(modelId, unit.atlasFile);
    const atlasPath = relativePath === unit.originalPreviewPath
      ? baseAtlasPath
      : await this.resolveCompiledAtlas?.(modelId, baseAtlasPath) ?? baseAtlasPath;
    const [atlas, maskBytes] = await Promise.all([
      this.loadPreviewAtlas(atlasPath),
      readFile(this.models.resolveModelFile(modelId, unit.maskPath)),
    ]);
    const mask = PNG.sync.read(maskBytes);
    const preview = textureUnitPreview(atlas, unit.bounds, mask);
    await writeFileAtomically(target, PNG.sync.write(preview));
  }

  private async withPreviewBuildSlot<T>(operation: () => Promise<T>): Promise<T> {
    if (this.previewBuildsActive >= 2) {
      await new Promise<void>((resolve) => this.previewBuildWaiters.push(resolve));
    }
    this.previewBuildsActive += 1;
    try {
      return await operation();
    } finally {
      this.previewBuildsActive -= 1;
      this.previewBuildWaiters.shift()?.();
    }
  }

  private loadPreviewAtlas(file: string): Promise<PNG> {
    const existing = this.previewAtlases.get(file);
    if (existing) return existing;
    const loading = readFile(file).then(decodePng);
    this.previewAtlases.set(file, loading);
    void loading.finally(() => {
      setTimeout(() => {
        if (this.previewAtlases.get(file) === loading) this.previewAtlases.delete(file);
      }, 750);
    });
    return loading;
  }

  async scan(modelId: string): Promise<ModelEditCapabilityManifest> {
    // One structural scan per model is sufficient. Appearance mutations use
    // refreshAppearance and never need a second concurrent Moc/UV traversal.
    const previous = this.scans.get(modelId);
    if (previous) return await previous;
    const scan = this.scanFresh(modelId);
    this.scans.set(modelId, scan);
    return await scan.finally(() => {
      if (this.scans.get(modelId) === scan) this.scans.delete(modelId);
    });
  }

  private async loadSource(
    modelId: string,
    resolveAppearance = true,
  ): Promise<ModelEditSource> {
    const modelPackage = await this.models.load(modelId);
    const modelRoot = this.models.resolveModelRoot(modelId);
    const entryPath = this.models.resolveModelFile(modelId, modelPackage.manifest.entry);
    const entryDirectory = path.dirname(entryPath);
    const modelJson = JSON.parse(await readFile(entryPath, "utf8")) as CubismModelJson;
    const references = modelJson.FileReferences ?? {};
    if (typeof references.Moc !== "string" || !Array.isArray(references.Textures)) {
      throw new Error("模型缺少 Moc 或纹理图集，无法进行可编辑能力预检");
    }
    const textureNames = references.Textures.filter((item): item is string => typeof item === "string");
    if (textureNames.length === 0) throw new Error("模型没有纹理图集");
    const mocPath = path.resolve(entryDirectory, references.Moc);
    const baseTexturePaths = textureNames.map((file) => path.resolve(entryDirectory, file));
    const activeTexturePaths = resolveAppearance
      ? await Promise.all(baseTexturePaths.map(async (file) =>
          await this.resolveCompiledAtlas?.(modelId, file) ?? file
        ))
      : baseTexturePaths;
    const motionPaths = Object.values(modelPackage.manifest.motionAssets)
      .map((asset) => path.resolve(modelRoot, asset.file));
    // This hash describes topology and editable-unit identity, not the current
    // outfit pixels. Appearance changes are refreshed incrementally below and
    // must never trigger a full Moc/UV capability scan.
    const sourceHash = await hashFileStats([entryPath, mocPath, ...baseTexturePaths, ...motionPaths]);
    return {
      modelPackage,
      modelRoot,
      entryPath,
      entryDirectory,
      modelJson,
      references,
      mocPath,
      baseTexturePaths,
      activeTexturePaths,
      motionPaths,
      sourceHash,
    };
  }

  private async appearanceFingerprint(activeTexturePaths: readonly string[]): Promise<string> {
    return await hashFiles(activeTexturePaths);
  }

  private async appearanceKey(
    modelId: string,
    capability: ModelEditCapabilityManifest,
    source: ModelEditSource,
  ): Promise<string> {
    try {
      const resolved = await this.resolveAppearanceKey?.(modelId, capability.sourceHash);
      if (resolved) return resolved;
    } catch {
      // The first structural scan runs before the material library is seeded.
    }
    return this.appearanceFingerprint(source.activeTexturePaths);
  }

  private async scanFresh(modelId: string): Promise<ModelEditCapabilityManifest> {
    const {
      modelPackage,
      modelRoot,
      entryDirectory,
      modelJson,
      references,
      mocPath,
      baseTexturePaths,
      activeTexturePaths,
      sourceHash,
    } = await this.loadSource(modelId);
    const inspected = await inspectCubismModel(this.corePath, mocPath);
    // UV identity and masks must remain tied to the model topology so generated
    // material variants keep stable IDs. Display pixels, however, must always
    // come from the currently active appearance.
    const topologyAtlases: AtlasImage[] = [];
    for (let index = 0; index < baseTexturePaths.length; index += 1) {
      const file = baseTexturePaths[index]!;
      topologyAtlases.push({
        relative: modelRelative(modelRoot, file),
        image: decodePng(await readFile(file)),
        index,
      });
    }
    const appearanceMatchesBase = activeTexturePaths.every((file, index) =>
      path.resolve(file) === path.resolve(baseTexturePaths[index]!)
    );
    const displayAtlases: AtlasImage[] = appearanceMatchesBase ? topologyAtlases : [];
    if (!appearanceMatchesBase) {
      for (let index = 0; index < activeTexturePaths.length; index += 1) {
        displayAtlases.push({
          relative: modelRelative(modelRoot, baseTexturePaths[index]!),
          image: decodePng(await readFile(activeTexturePaths[index]!)),
          index,
        });
      }
    }
    const partNames = await readPartNames(entryDirectory, references.DisplayInfo);
    const artMeshTags = await readArtMeshTags(entryDirectory, references.UserData);
    const partChain = (partIndex: number): PartNode[] => {
      const chain: PartNode[] = [];
      for (let index = partIndex, guard = 0; index >= 0 && guard < 64; guard += 1) {
        const part = inspected.parts[index];
        if (!part) break;
        chain.push({ id: part.id, name: partNames.get(part.id) ?? "" });
        index = part.parentIndex;
      }
      return chain;
    };
    const hiddenPoseParts = await readHiddenPoseParts(entryDirectory, references.Pose);
    const motionVisibility = await readMotionVisibility(
      modelRoot,
      modelPackage.manifest.motionAssets,
    );
    const capabilityRoot = path.join(modelRoot, CAPABILITY_DIRECTORY);
    await rm(capabilityRoot, { recursive: true, force: true });
    await Promise.all([
      mkdir(path.join(capabilityRoot, "masks"), { recursive: true }),
      mkdir(path.join(capabilityRoot, "texture-units"), { recursive: true }),
    ]);

    const guideSize = scaledGuideSize(inspected.canvas.width, inspected.canvas.height);
    const guide = renderNeutralGuide(inspected, displayAtlases, guideSize, hiddenPoseParts);
    await writeFile(path.join(modelRoot, GUIDE_FILE), PNG.sync.write(guide));
    const guideRenderSnapshot: GuideRenderSnapshot = {
      schemaVersion: 1,
      rendererVersion: GUIDE_RENDERER_VERSION,
      model: inspected,
      hiddenPoseParts: [...hiddenPoseParts],
      size: guideSize,
    };

    const clippingMasks = new Set(inspected.drawables.flatMap((drawable) => drawable.masks));
    const analyses: SlotAnalysis[] = [];
    for (const [drawableIndex, drawable] of inspected.drawables.entries()) {
      const atlas = topologyAtlases[drawable.texture];
      if (!atlas || drawable.indices.length < 3) continue;
      const partId = inspected.parts[drawable.partIndex]?.id ?? "UNBOUND";
      const meshBounds = boundsOfPairs(drawable.vertices);
      if (width(meshBounds) <= 1e-8 || height(meshBounds) <= 1e-8) continue;
      const semantic = classifyPart({
        drawableId: drawable.id,
        chain: partChain(drawable.partIndex),
        tag: artMeshTags.get(drawable.id),
        side: (meshBounds.minX + meshBounds.maxX) / 2 < 0 ? "left" : "right",
      });
      const region = uvRegion(drawable.uvs, atlas.image.width, atlas.image.height);
      const analyzed = rasterizeSlot(drawable, atlas.image, region);
      if (analyzed.visiblePixels === 0) continue;
      const safeId = drawable.id.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      const silhouetteMask = `${CAPABILITY_DIRECTORY}/masks/${safeId}.png`;
      await writeFile(path.join(modelRoot, silhouetteMask), PNG.sync.write(analyzed.mask));
      const chain = partChain(drawable.partIndex);
      const revealedByMotionIds = drawable.opacity <= 0.001 || hiddenPoseParts.has(partId)
        ? motionVisibility.get(partId) ?? []
        : [];
      analyses.push({
        searchText: [drawable.id, ...chain.flatMap((node) => [node.id, node.name])].filter(Boolean).join(" "),
        semantic,
        mask: analyzed.mask,
        meshBounds,
        visibleBounds: analyzed.visibleBounds,
        visiblePixels: analyzed.visiblePixels,
        revealedByMotionIds,
        // A mask is transparent on its own; when its Parts are shown, no motion
        // that fades those Parts can reveal it.
        maskOnly: clippingMasks.has(drawableIndex) && drawable.opacity <= 0.001 &&
          (partChainOpacity(inspected.parts, drawable.partIndex) > 0.001 || !revealedByMotionIds.length),
        slot: {
          id: safeId,
          name: `${semantic.label} · ${drawable.id}`,
          drawableId: drawable.id,
          partId,
          group: semantic.templateGroup,
          // The pixels already exist in the source atlas even when the Part is
          // hidden in the neutral pose. `source` describes the texture origin,
          // not its current PartOpacity.
          source: "visible",
          required: false,
          zIndex: drawable.renderOrder,
          atlas: { texture: drawable.texture, ...region },
          anchor: { x: region.x + region.width / 2, y: region.y + region.height / 2 },
          canvasBounds: {
            x: meshBounds.minX,
            y: meshBounds.minY,
            width: width(meshBounds),
            height: height(meshBounds),
          },
          mesh: {
            vertices: roundedArray(drawable.vertices),
            uvs: roundedArray(drawable.uvs),
            indices: drawable.indices,
          },
          silhouetteMask,
          completionPrompt: `restyle only the existing visible ${semantic.label} texture`,
          editPartId: semantic.id,
        },
      });
    }

    const parts = buildPartCapabilities(analyses, inspected.canvas.pixelsPerUnit);
    const textureUnits = await buildTextureUnits(
      modelRoot,
      topologyAtlases,
      displayAtlases,
      analyses,
      inspected.canvas,
      guideSize,
    );
    const textureUnitByDrawable = new Map(textureUnits.flatMap((unit) =>
      unit.drawableIds.map((drawableId) => [drawableId, unit.id] as const)
    ));
    for (const analysis of analyses) {
      analysis.slot.textureUnitId = textureUnitByDrawable.get(analysis.slot.drawableId);
    }
    // Semantic labels determine which part-level AI operations are available,
    // not whether the model has a valid appearance or usable texture materials.
    if (textureUnits.length === 0) {
      throw new Error("没有找到包含可见像素的 ArtMesh");
    }
    const templateId = `model-edit-${modelId}`;
    const capability: ModelEditCapabilityManifest = {
      schemaVersion: 12,
      modelId,
      modelName: modelPackage.name,
      templateId,
      sourceHash,
      generatedAt: new Date().toISOString(),
      validation: {
        geometry: "neutral-static",
        motionSweep: "not-run",
        alphaPolicy: "shrink-only",
      },
      parts,
      textureUnits,
    };
    const template: MasterTemplateManifest = {
      schemaVersion: 2,
      id: templateId,
      name: `${modelPackage.name} 可编辑母版`,
      modelEntry: modelPackage.manifest.entry,
      alignmentGuide: GUIDE_FILE,
      coverage: "full-body",
      canvas: inspected.canvas,
      parameters: {
        eyeBlink: parameterGroupIds(modelJson, "EyeBlink"),
        lipSync: parameterGroupIds(modelJson, "LipSync"),
      },
      textureAtlases: topologyAtlases.map((atlas) => ({
        file: atlas.relative,
        width: atlas.image.width,
        height: atlas.image.height,
      })),
      slots: analyses.map((analysis) => analysis.slot),
    };
    let initialAppearanceKey: string;
    try {
      initialAppearanceKey = await this.resolveAppearanceKey?.(modelId, sourceHash)
        ?? await this.appearanceFingerprint(activeTexturePaths);
    } catch {
      initialAppearanceKey = await this.appearanceFingerprint(activeTexturePaths);
    }
    const appearanceState: AppearanceState = {
      schemaVersion: 2,
      rendererVersion: GUIDE_RENDERER_VERSION,
      appearanceKey: initialAppearanceKey,
      generatedAt: new Date().toISOString(),
    };
    await Promise.all([
      writeJson(path.join(capabilityRoot, CAPABILITY_FILE), capability),
      writeJson(path.join(capabilityRoot, TEMPLATE_FILE), template),
      writeJson(path.join(modelRoot, GUIDE_RENDER_SNAPSHOT_FILE), guideRenderSnapshot),
      writeJson(path.join(modelRoot, APPEARANCE_STATE_FILE), appearanceState),
    ]);
    return capability;
  }

  /**
   * Refreshes the display pixels derived from changed runtime atlases without
   * re-inspecting Moc, UVs, motions, or rebuilding masks. Texture-unit identity
   * remains tied to the immutable base model topology.
   */
  async refreshAppearance(
    modelId: string,
    affectedAtlasFiles: readonly string[],
    affectedTextureUnitIds: readonly string[],
    appearanceKey?: string,
  ): Promise<ModelEditCapabilityManifest> {
    const previous = this.appearanceRefreshes.get(modelId);
    const refresh = (previous ? previous.catch(() => undefined) : Promise.resolve())
      .then(() => this.refreshAppearanceFresh(
        modelId,
        affectedAtlasFiles,
        affectedTextureUnitIds,
        appearanceKey,
      ));
    this.appearanceRefreshes.set(modelId, refresh);
    return await refresh.finally(() => {
      if (this.appearanceRefreshes.get(modelId) === refresh) {
        this.appearanceRefreshes.delete(modelId);
      }
    });
  }

  private async refreshAppearanceFresh(
    modelId: string,
    affectedAtlasFiles: readonly string[],
    _affectedTextureUnitIds: readonly string[],
    requestedAppearanceKey?: string,
  ): Promise<ModelEditCapabilityManifest> {
    // An appearance operation cannot change Moc, UVs or motion structure. Use
    // the cached structural manifest directly instead of re-hashing every base
    // atlas before each material switch. A missing manifest is the only reason
    // to perform the full scan here.
    const capability = await this.get(modelId) ?? await this.scan(modelId);
    const affected = new Set(affectedAtlasFiles.map(normalizeModelRelative));
    if (affected.size === 0) return capability;

    const source = await this.loadSource(modelId);
    if (requestedAppearanceKey && appearanceStateMatches(
      await this.readAppearanceState(modelId),
      requestedAppearanceKey,
    )) {
      try {
        await access(path.join(source.modelRoot, GUIDE_FILE));
        return capability;
      } catch {
        // Rebuild the missing file below.
      }
    }
    const renderSnapshot = await this.readGuideRenderSnapshot(modelId);
    if (!renderSnapshot) return await this.scan(modelId);
    const displayAtlases: AtlasImage[] = await Promise.all(source.activeTexturePaths.map(async (file, index) => ({
      relative: modelRelative(source.modelRoot, source.baseTexturePaths[index]!),
      image: decodePng(await readFile(file)),
      index,
    })));

    const guide = renderNeutralGuide(
      renderSnapshot.model,
      displayAtlases,
      renderSnapshot.size,
      new Set(renderSnapshot.hiddenPoseParts),
    );
    await writeFileAtomically(
      path.join(source.modelRoot, GUIDE_FILE),
      PNG.sync.write(guide),
    );

    const appearanceState: AppearanceState = {
      schemaVersion: 2,
      rendererVersion: GUIDE_RENDERER_VERSION,
      appearanceKey: requestedAppearanceKey ?? await this.appearanceKey(modelId, capability, source),
      generatedAt: new Date().toISOString(),
    };
    // The structural manifest is intentionally not rewritten: material
    // changes cannot alter Moc, UV, motions or editable-unit identity.
    await writeJsonAtomically(
      path.join(source.modelRoot, APPEARANCE_STATE_FILE),
      appearanceState,
    );
    return capability;
  }

  async load(templateId: string): Promise<MasterTemplatePackage> {
    const modelId = templateId.startsWith("model-edit-")
      ? templateId.slice("model-edit-".length)
      : "";
    if (!modelId) throw new Error("可编辑模型模板 ID 无效");
    const capability = await this.current(modelId);
    if (capability.templateId !== templateId) throw new Error("可编辑模型模板已失效，请重新预检");
    const root = this.models.resolveModelRoot(modelId);
    const manifest = JSON.parse(
      await readFile(path.join(root, CAPABILITY_DIRECTORY, TEMPLATE_FILE), "utf8"),
    ) as MasterTemplateManifest;
    return { id: templateId, name: `${capability.modelName} 可编辑母版`, root, manifest };
  }

  private capabilityPath(modelId: string): string {
    return path.join(this.models.resolveModelRoot(modelId), CAPABILITY_DIRECTORY, CAPABILITY_FILE);
  }

  private async readAppearanceState(modelId: string): Promise<AppearanceState | null> {
    try {
      const value = JSON.parse(await readFile(
        path.join(this.models.resolveModelRoot(modelId), APPEARANCE_STATE_FILE),
        "utf8",
      )) as unknown;
      return parseAppearanceState(value);
    } catch {
      return null;
    }
  }

  /**
   * The appearance state is a commit marker, but users may still have a
   * partially-written snapshot from an interrupted older build. Only verify
   * the small set of authoritative derived files here; individual unit
   * previews are written before the state marker and therefore belong to the
   * same committed snapshot.
   */
  private async appearanceArtifactsComplete(source: ModelEditSource): Promise<boolean> {
    const required = [
      path.join(source.modelRoot, GUIDE_FILE),
      path.join(source.modelRoot, GUIDE_RENDER_SNAPSHOT_FILE),
    ];
    try {
      await Promise.all(required.map((file) => access(file)));
      return true;
    } catch {
      return false;
    }
  }

  private async readGuideRenderSnapshot(modelId: string): Promise<GuideRenderSnapshot | null> {
    try {
      const value = JSON.parse(await readFile(
        path.join(this.models.resolveModelRoot(modelId), GUIDE_RENDER_SNAPSHOT_FILE),
        "utf8",
      )) as unknown;
      return parseGuideRenderSnapshot(value);
    } catch {
      return null;
    }
  }
}

/** The neutral-pose opacity a Part and all its parents multiply to. */
function partChainOpacity(parts: InspectedCubismModel["parts"], partIndex: number): number {
  let opacity = 1;
  for (let index = partIndex, guard = 0; index >= 0 && guard < 64; guard += 1) {
    const part = parts[index];
    if (!part) break;
    opacity *= part.opacity ?? 1;
    index = part.parentIndex;
  }
  return opacity;
}

function buildPartCapabilities(
  analyses: SlotAnalysis[],
  pixelsPerUnit: number,
): ModelEditPartCapability[] {
  const groups = new Map<string, SlotAnalysis[]>();
  for (const analysis of analyses) {
    const values = groups.get(analysis.semantic.id) ?? [];
    values.push(analysis);
    groups.set(analysis.semantic.id, values);
  }
  return [...groups.values()].map((values) => {
    const semantic = values[0]!.semantic;
    const visible = unionBounds(values.map((value) => value.visibleBounds));
    const mesh = unionBounds(values.map((value) => value.meshBounds));
    const horizontal = Math.max(width(visible), 1e-8);
    const vertical = Math.max(height(visible), 1e-8);
    const slack = {
      top: directional(Math.max(0, mesh.maxY - visible.maxY), vertical, pixelsPerUnit),
      right: directional(Math.max(0, mesh.maxX - visible.maxX), horizontal, pixelsPerUnit),
      bottom: directional(Math.max(0, visible.minY - mesh.minY), vertical, pixelsPerUnit),
      left: directional(Math.max(0, visible.minX - mesh.minX), horizontal, pixelsPerUnit),
    };
    const visiblePixels = values.reduce((total, value) => total + value.visiblePixels, 0);
    // Whether a part can be repainted depends on its pixels and UVs, never on
    // whether its name was recognized.
    const appearanceSupported = visiblePixels >= 64;
    return {
      id: semantic.id,
      label: semantic.label,
      group: semantic.group,
      drawableIds: values.map((value) => value.slot.drawableId),
      partIds: [...new Set(values.map((value) => value.slot.partId))],
      visiblePixels,
      operations: {
        appearance: { supported: appearanceSupported, alphaPolicy: "shrink-only" },
        contour: {
          supported: false,
          reason: "尚未通过全动作网格扫描；预检只展示中立姿势下的静态透明余量",
          staticSlack: slack,
        },
      },
    } satisfies ModelEditPartCapability;
  }).sort((left, right) => partOrder(left.group) - partOrder(right.group) || left.label.localeCompare(right.label));
}

function rasterizeSlot(
  drawable: InspectedCubismModel["drawables"][number],
  atlas: PNG,
  region: { x: number; y: number; width: number; height: number },
): { mask: PNG; visibleBounds: Bounds; visiblePixels: number } {
  const output = new PNG({ width: region.width, height: region.height });
  const source = pairs(drawable.uvs).map(([u, v]) => [u * atlas.width, (1 - v) * atlas.height] as const);
  const destination = pairs(drawable.vertices);
  const visibleBounds = emptyBounds();
  const visited = new Uint8Array(region.width * region.height);
  let visiblePixels = 0;
  for (let offset = 0; offset < drawable.indices.length; offset += 3) {
    const indices = drawable.indices.slice(offset, offset + 3);
    const triangle = indices.map((index) => source[index]!);
    const mapped = indices.map((index) => destination[index]!);
    const area = edge(triangle[0]!, triangle[1]!, triangle[2]!);
    if (Math.abs(area) < 1e-8) continue;
    const triangleBounds = boundsOfPoints(triangle);
    const minX = Math.max(region.x, Math.floor(triangleBounds.minX));
    const maxX = Math.min(region.x + region.width - 1, Math.ceil(triangleBounds.maxX));
    const minY = Math.max(region.y, Math.floor(triangleBounds.minY));
    const maxY = Math.min(region.y + region.height - 1, Math.ceil(triangleBounds.maxY));
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const point = [x + 0.5, y + 0.5] as const;
        const weights = [
          edge(triangle[1]!, triangle[2]!, point) / area,
          edge(triangle[2]!, triangle[0]!, point) / area,
          edge(triangle[0]!, triangle[1]!, point) / area,
        ];
        if (weights.some((weight) => weight < -1e-5)) continue;
        const localX = x - region.x;
        const localY = y - region.y;
        const localIndex = localY * region.width + localX;
        if (visited[localIndex]) continue;
        visited[localIndex] = 1;
        const atlasOffset = (y * atlas.width + x) * 4;
        const alpha = atlas.data[atlasOffset + 3]!;
        const outputOffset = localIndex * 4;
        output.data[outputOffset] = 255;
        output.data[outputOffset + 1] = 255;
        output.data[outputOffset + 2] = 255;
        output.data[outputOffset + 3] = alpha;
        if (alpha <= 8) continue;
        visiblePixels += 1;
        include(visibleBounds, [
          mapped[0]![0] * weights[0]! + mapped[1]![0] * weights[1]! + mapped[2]![0] * weights[2]!,
          mapped[0]![1] * weights[0]! + mapped[1]![1] * weights[1]! + mapped[2]![1] * weights[2]!,
        ]);
      }
    }
  }
  return { mask: output, visibleBounds, visiblePixels };
}

function renderNeutralGuide(
  model: InspectedCubismModel,
  atlases: AtlasImage[],
  size: { width: number; height: number },
  hiddenPoseParts: Set<string>,
): PNG {
  const output = new PNG(size);
  for (let offset = 0; offset < output.data.length; offset += 4) {
    output.data[offset] = 255;
    output.data[offset + 1] = 255;
    output.data[offset + 2] = 255;
    output.data[offset + 3] = 255;
  }
  const drawables = [...model.drawables].filter((drawable) => {
    const partId = model.parts[drawable.partIndex]?.id ?? "";
    return drawable.opacity > 0.001 && !hiddenPoseParts.has(partId);
  }).sort((left, right) => left.renderOrder - right.renderOrder);
  for (const drawable of drawables) {
    const atlas = atlases[drawable.texture]?.image;
    if (!atlas) continue;
    const destination = pairs(drawable.vertices).map(([x, y]) => [
      (model.canvas.originX + x * model.canvas.pixelsPerUnit) / model.canvas.width * size.width,
      (model.canvas.originY - y * model.canvas.pixelsPerUnit) / model.canvas.height * size.height,
    ] as const);
    const source = pairs(drawable.uvs).map(([u, v]) => [u * atlas.width, (1 - v) * atlas.height] as const);
    for (let offset = 0; offset < drawable.indices.length; offset += 3) {
      const indices = drawable.indices.slice(offset, offset + 3);
      paintTriangle(
        output,
        atlas,
        indices.map((index) => destination[index]!),
        indices.map((index) => source[index]!),
        drawable.opacity,
      );
    }
  }
  return output;
}

function paintTriangle(
  output: PNG,
  atlas: PNG,
  destination: ReadonlyArray<readonly [number, number]>,
  source: ReadonlyArray<readonly [number, number]>,
  opacity: number,
): void {
  const area = edge(destination[0]!, destination[1]!, destination[2]!);
  if (Math.abs(area) < 1e-8) return;
  const bounds = boundsOfPoints(destination);
  const minX = Math.max(0, Math.floor(bounds.minX));
  const maxX = Math.min(output.width - 1, Math.ceil(bounds.maxX));
  const minY = Math.max(0, Math.floor(bounds.minY));
  const maxY = Math.min(output.height - 1, Math.ceil(bounds.maxY));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const point = [x + 0.5, y + 0.5] as const;
      const weights = [
        edge(destination[1]!, destination[2]!, point) / area,
        edge(destination[2]!, destination[0]!, point) / area,
        edge(destination[0]!, destination[1]!, point) / area,
      ];
      if (weights.some((weight) => weight < -1e-5)) continue;
      const sourceX = source[0]![0] * weights[0]! + source[1]![0] * weights[1]! + source[2]![0] * weights[2]!;
      const sourceY = source[0]![1] * weights[0]! + source[1]![1] * weights[1]! + source[2]![1] * weights[2]!;
      const sx = Math.max(0, Math.min(atlas.width - 1, Math.round(sourceX)));
      const sy = Math.max(0, Math.min(atlas.height - 1, Math.round(sourceY)));
      const sourceOffset = (sy * atlas.width + sx) * 4;
      const targetOffset = (y * output.width + x) * 4;
      const alpha = atlas.data[sourceOffset + 3]! / 255 * opacity;
      if (alpha <= 0) continue;
      for (let channel = 0; channel < 3; channel += 1) {
        output.data[targetOffset + channel] = Math.round(
          atlas.data[sourceOffset + channel]! * alpha + output.data[targetOffset + channel]! * (1 - alpha),
        );
      }
      output.data[targetOffset + 3] = 255;
    }
  }
}

async function readPartNames(directory: string, value: unknown): Promise<Map<string, string>> {
  if (typeof value !== "string") return new Map();
  try {
    const parsed = JSON.parse(await readFile(path.resolve(directory, value), "utf8")) as {
      Parts?: Array<{ Id?: unknown; Name?: unknown }>;
    };
    return new Map((parsed.Parts ?? []).flatMap((part) =>
      typeof part.Id === "string" && typeof part.Name === "string" ? [[part.Id, part.Name]] : []
    ));
  } catch {
    return new Map();
  }
}

/** Rigger-written ArtMesh UserData (`.userdata3.json`), the most reliable part name there is. */
async function readArtMeshTags(directory: string, value: unknown): Promise<Map<string, string>> {
  if (typeof value !== "string") return new Map();
  try {
    const parsed = JSON.parse(await readFile(path.resolve(directory, value), "utf8")) as {
      UserData?: Array<{ Target?: unknown; Id?: unknown; Value?: unknown }>;
    };
    return new Map((parsed.UserData ?? []).flatMap((item) =>
      item.Target === "ArtMesh" && typeof item.Id === "string" && typeof item.Value === "string" && item.Value.trim()
        ? [[item.Id, item.Value.trim().split(/\s+/)[0]!]]
        : []
    ));
  } catch {
    return new Map();
  }
}

async function readHiddenPoseParts(directory: string, value: unknown): Promise<Set<string>> {
  if (typeof value !== "string") return new Set();
  try {
    const parsed = JSON.parse(await readFile(path.resolve(directory, value), "utf8")) as {
      Groups?: Array<Array<{ Id?: unknown }>>;
    };
    return new Set((parsed.Groups ?? []).flatMap((group) =>
      group.slice(1).flatMap((part) => typeof part.Id === "string" ? [part.Id] : [])
    ));
  } catch {
    return new Set();
  }
}

async function readMotionVisibility(
  modelRoot: string,
  assets: Record<string, MotionAsset>,
): Promise<Map<string, string[]>> {
  const result = new Map<string, string[]>();
  // Preserve manifest order so annotations remain deterministic across scans.
  for (const asset of Object.values(assets)) {
    try {
      const parsed = JSON.parse(
        await readFile(path.resolve(modelRoot, asset.file), "utf8"),
      ) as unknown;
      for (const partId of visiblePartIdsFromMotion(parsed)) {
        const ids = result.get(partId) ?? [];
        ids.push(asset.id);
        result.set(partId, ids);
      }
    } catch {
      // A broken optional motion must not make every editable model disappear.
      // Model import validation reports missing resources separately.
    }
  }
  return result;
}

/** Extract Parts that a Cubism motion can make visible via PartOpacity curves. */
export function visiblePartIdsFromMotion(value: unknown): string[] {
  if (!isRecord(value) || !Array.isArray(value.Curves)) return [];
  const ids = new Set<string>();
  for (const curve of value.Curves) {
    if (
      !isRecord(curve) || curve.Target !== "PartOpacity" ||
      typeof curve.Id !== "string" || !Array.isArray(curve.Segments)
    ) continue;
    if (motionCurveValues(curve.Segments).some((item) => item > 0.001)) ids.add(curve.Id);
  }
  return [...ids];
}

function motionCurveValues(segments: unknown[]): number[] {
  const values = segments.filter((item): item is number =>
    typeof item === "number" && Number.isFinite(item)
  );
  if (values.length < 2) return [];
  const output = [values[1]!];
  let cursor = 2;
  while (cursor < values.length) {
    const kind = values[cursor++]!;
    if (kind === 0 || kind === 2 || kind === 3) {
      if (cursor + 1 >= values.length) break;
      output.push(values[cursor + 1]!);
      cursor += 2;
      continue;
    }
    if (kind === 1) {
      if (cursor + 5 >= values.length) break;
      // Bezier control values can reveal the Part between its endpoints.
      output.push(values[cursor + 1]!, values[cursor + 3]!, values[cursor + 5]!);
      cursor += 6;
      continue;
    }
    break;
  }
  return output;
}

function parameterGroupIds(model: CubismModelJson, name: string): string[] {
  if (!Array.isArray(model.Groups)) return [];
  const group = model.Groups.find((item) => isRecord(item) && item.Target === "Parameter" && item.Name === name);
  return isRecord(group) && Array.isArray(group.Ids)
    ? group.Ids.filter((id): id is string => typeof id === "string")
    : [];
}

async function hashFiles(files: readonly string[]): Promise<string> {
  const hash = createHash("sha256");
  for (const file of files) hash.update(await readFile(file));
  return hash.digest("hex");
}

async function hashFileStats(files: readonly string[]): Promise<string> {
  const hash = createHash("sha256");
  for (const file of files) {
    const value = await stat(file);
    hash.update(file);
    hash.update(String(value.size));
    hash.update(String(value.mtimeMs));
  }
  return hash.digest("hex");
}

function parseCapabilityManifest(value: unknown, modelId: string): ModelEditCapabilityManifest {
  if (
    !isRecord(value) || value.schemaVersion !== 12 || value.modelId !== modelId ||
    !Array.isArray(value.parts) || !Array.isArray(value.textureUnits) ||
    !value.textureUnits.every((unit) =>
      isRecord(unit) &&
      typeof unit.sourceAtlasFile === "string" &&
      typeof unit.originalPreviewPath === "string" &&
      Array.isArray(unit.revealedByMotionIds) &&
      unit.revealedByMotionIds.every((id) => typeof id === "string")
    )
  ) {
    throw new Error("模型可编辑能力清单无效，请重新预检");
  }
  return value as unknown as ModelEditCapabilityManifest;
}

function appearanceStateMatches(
  state: AppearanceState | null,
  appearanceKey: string,
): boolean {
  return state?.schemaVersion === 2 &&
    state.rendererVersion === GUIDE_RENDERER_VERSION &&
    state.appearanceKey === appearanceKey;
}

function parseAppearanceState(value: unknown): AppearanceState | null {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 2 ||
    value.rendererVersion !== GUIDE_RENDERER_VERSION ||
    typeof value.appearanceKey !== "string" ||
    typeof value.generatedAt !== "string"
  ) {
    return null;
  }
  return value as unknown as AppearanceState;
}

function parseGuideRenderSnapshot(value: unknown): GuideRenderSnapshot | null {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    value.rendererVersion !== GUIDE_RENDERER_VERSION ||
    !isRecord(value.model) ||
    !isRecord(value.model.canvas) ||
    !Array.isArray(value.model.parameters) ||
    !Array.isArray(value.model.parts) ||
    !Array.isArray(value.model.drawables) ||
    !Array.isArray(value.hiddenPoseParts) ||
    !value.hiddenPoseParts.every((item) => typeof item === "string") ||
    !isRecord(value.size) ||
    typeof value.size.width !== "number" ||
    typeof value.size.height !== "number"
  ) {
    return null;
  }
  return value as unknown as GuideRenderSnapshot;
}

async function buildTextureUnits(
  modelRoot: string,
  topologyAtlases: AtlasImage[],
  displayAtlases: AtlasImage[],
  analyses: SlotAnalysis[],
  canvas: InspectedCubismModel["canvas"],
  guideSize: { width: number; height: number },
): Promise<TextureEditUnit[]> {
  const grouped = new Map<string, SlotAnalysis[]>();
  for (const analysis of analyses) {
    const region = analysis.slot.atlas;
    const hash = textureMaskHash(region.texture, region, analysis.mask);
    const key = `${region.texture}:${hash}`;
    const values = grouped.get(key) ?? [];
    values.push(analysis);
    grouped.set(key, values);
  }
  const counters = new Map<string, number>();
  const units: TextureEditUnit[] = [];
  for (const [key, values] of grouped) {
    const first = values[0]!;
    const topologyAtlas = topologyAtlases[first.slot.atlas.texture]!;
    const hash = key.slice(key.indexOf(":") + 1);
    const semanticKey = first.semantic.id;
    const ordinal = (counters.get(semanticKey) ?? 0) + 1;
    counters.set(semanticKey, ordinal);
    const id = `texture-${first.slot.atlas.texture}-${hash.slice(0, 16)}`;
    const maskPath = `${CAPABILITY_DIRECTORY}/texture-units/${id}-mask.png`;
    const originalPreviewPath = `${CAPABILITY_DIRECTORY}/texture-units/${id}-original-preview.png`;
    const previewPath = `${CAPABILITY_DIRECTORY}/texture-units/${id}-preview.png`;
    const modelMaskPath = `${CAPABILITY_DIRECTORY}/texture-units/${id}-model-mask.png`;
    const modelProjection = renderTextureUnitModelMask(values, displayAtlases, canvas, guideSize);
    await Promise.all([
      writeFile(path.join(modelRoot, maskPath), PNG.sync.write(first.mask)),
      writeFile(path.join(modelRoot, modelMaskPath), PNG.sync.write(modelProjection.image)),
    ]);
    const modelBounds = unionBounds(values.map((value) => value.meshBounds));
    const left = (canvas.originX + modelBounds.minX * canvas.pixelsPerUnit) / canvas.width;
    const top = (canvas.originY - modelBounds.maxY * canvas.pixelsPerUnit) / canvas.height;
    const right = (canvas.originX + modelBounds.maxX * canvas.pixelsPerUnit) / canvas.width;
    const bottom = (canvas.originY - modelBounds.minY * canvas.pixelsPerUnit) / canvas.height;
    units.push({
      id,
      atlasIndex: first.slot.atlas.texture,
      sourceAtlasFile: topologyAtlas.relative,
      atlasFile: topologyAtlas.relative,
      maskHash: hash,
      bounds: {
        x: first.slot.atlas.x,
        y: first.slot.atlas.y,
        width: first.slot.atlas.width,
        height: first.slot.atlas.height,
      },
      maskPath,
      originalPreviewPath,
      previewPath,
      modelMaskPath,
      modelMaskBounds: modelProjection.bounds,
      drawableIds: values.map((value) => value.slot.drawableId),
      partIds: [...new Set(values.map((value) => value.slot.partId))],
      semanticPartIds: [...new Set(values.map((value) => value.semantic.id))],
      searchText: [...new Set(values.flatMap((value) => value.searchText.split(" ")))].join(" "),
      revealedByMotionIds: [...new Set(values.flatMap((value) => value.revealedByMotionIds))],
      label: `${first.semantic.label} · 纹理 ${ordinal}`,
      ...values.every((value) => value.maskOnly) ? { maskOnly: true } : {},
      visiblePixels: first.visiblePixels,
      modelBounds: {
        x: round(left),
        y: round(top),
        width: round(right - left),
        height: round(bottom - top),
      },
    });
  }
  return units.sort((left, right) =>
    left.atlasIndex - right.atlasIndex || left.bounds.y - right.bounds.y || left.bounds.x - right.bounds.x
  );
}

function currentAtlasFile(index: number): string {
  return `${CURRENT_ATLAS_DIRECTORY}/atlas-${index}.png`;
}

function renderTextureUnitModelMask(
  analyses: SlotAnalysis[],
  atlases: AtlasImage[],
  canvas: InspectedCubismModel["canvas"],
  size: { width: number; height: number },
): {
  image: PNG;
  bounds: { x: number; y: number; width: number; height: number; canvasWidth: number; canvasHeight: number };
} {
  const projected = analyses.flatMap((analysis) => pairs(analysis.slot.mesh.vertices).map(([x, y]) => [
    (canvas.originX + x * canvas.pixelsPerUnit) / canvas.width * size.width,
    (canvas.originY - y * canvas.pixelsPerUnit) / canvas.height * size.height,
  ] as const));
  const projectedBounds = boundsOfPoints(projected);
  const left = Math.max(0, Math.floor(projectedBounds.minX));
  const top = Math.max(0, Math.floor(projectedBounds.minY));
  const right = Math.min(size.width, Math.ceil(projectedBounds.maxX) + 1);
  const bottom = Math.min(size.height, Math.ceil(projectedBounds.maxY) + 1);
  const bounds = {
    x: left,
    y: top,
    width: Math.max(1, right - left),
    height: Math.max(1, bottom - top),
    canvasWidth: size.width,
    canvasHeight: size.height,
  };
  const output = new PNG({ width: bounds.width, height: bounds.height });
  for (const analysis of analyses) {
    const atlas = atlases[analysis.slot.atlas.texture]?.image;
    if (!atlas) continue;
    const destination = pairs(analysis.slot.mesh.vertices).map(([x, y]) => [
      (canvas.originX + x * canvas.pixelsPerUnit) / canvas.width * size.width - bounds.x,
      (canvas.originY - y * canvas.pixelsPerUnit) / canvas.height * size.height - bounds.y,
    ] as const);
    const source = pairs(analysis.slot.mesh.uvs).map(([u, v]) => [
      u * atlas.width,
      (1 - v) * atlas.height,
    ] as const);
    for (let offset = 0; offset < analysis.slot.mesh.indices.length; offset += 3) {
      const indices = analysis.slot.mesh.indices.slice(offset, offset + 3);
      paintMaskTriangle(
        output,
        atlas,
        indices.map((index) => destination[index]!),
        indices.map((index) => source[index]!),
      );
    }
  }
  return { image: output, bounds };
}

function paintMaskTriangle(
  output: PNG,
  atlas: PNG,
  destination: ReadonlyArray<readonly [number, number]>,
  source: ReadonlyArray<readonly [number, number]>,
): void {
  const area = edge(destination[0]!, destination[1]!, destination[2]!);
  if (Math.abs(area) < 1e-8) return;
  const bounds = boundsOfPoints(destination);
  const minX = Math.max(0, Math.floor(bounds.minX));
  const maxX = Math.min(output.width - 1, Math.ceil(bounds.maxX));
  const minY = Math.max(0, Math.floor(bounds.minY));
  const maxY = Math.min(output.height - 1, Math.ceil(bounds.maxY));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const point = [x + 0.5, y + 0.5] as const;
      const weights = [
        edge(destination[1]!, destination[2]!, point) / area,
        edge(destination[2]!, destination[0]!, point) / area,
        edge(destination[0]!, destination[1]!, point) / area,
      ];
      if (weights.some((weight) => weight < -1e-5)) continue;
      const sourceX = source[0]![0] * weights[0]! + source[1]![0] * weights[1]! + source[2]![0] * weights[2]!;
      const sourceY = source[0]![1] * weights[0]! + source[1]![1] * weights[1]! + source[2]![1] * weights[2]!;
      const sx = Math.max(0, Math.min(atlas.width - 1, Math.round(sourceX)));
      const sy = Math.max(0, Math.min(atlas.height - 1, Math.round(sourceY)));
      const alpha = atlas.data[(sy * atlas.width + sx) * 4 + 3]!;
      if (alpha <= 8) continue;
      const target = (y * output.width + x) * 4;
      output.data[target] = 255;
      output.data[target + 1] = 255;
      output.data[target + 2] = 255;
      output.data[target + 3] = Math.max(output.data[target + 3]!, alpha);
    }
  }
}

function textureMaskHash(
  atlasIndex: number,
  region: { x: number; y: number; width: number; height: number },
  mask: PNG,
): string {
  const alpha = new Uint8Array(mask.width * mask.height);
  for (let source = 3, target = 0; source < mask.data.length; source += 4, target += 1) {
    alpha[target] = mask.data[source]!;
  }
  return textureMaskIdentity(atlasIndex, region, alpha);
}

export function textureMaskIdentity(
  atlasIndex: number,
  region: { x: number; y: number; width: number; height: number },
  alpha: Uint8Array,
): string {
  if (alpha.length !== region.width * region.height) {
    throw new Error("UV 蒙版像素数量与纹理边界不一致");
  }
  const hash = createHash("sha256");
  hash.update(`${atlasIndex}:${region.x}:${region.y}:${region.width}:${region.height}:`);
  hash.update(alpha);
  return hash.digest("hex");
}

function textureUnitPreview(
  atlas: PNG,
  region: { x: number; y: number; width: number; height: number },
  mask: PNG,
): PNG {
  const output = new PNG({ width: region.width, height: region.height });
  for (let y = 0; y < region.height; y += 1) {
    for (let x = 0; x < region.width; x += 1) {
      const sourceOffset = ((region.y + y) * atlas.width + region.x + x) * 4;
      const targetOffset = (y * region.width + x) * 4;
      output.data[targetOffset] = atlas.data[sourceOffset]!;
      output.data[targetOffset + 1] = atlas.data[sourceOffset + 1]!;
      output.data[targetOffset + 2] = atlas.data[sourceOffset + 2]!;
      output.data[targetOffset + 3] = Math.min(
        atlas.data[sourceOffset + 3]!,
        mask.data[targetOffset + 3]!,
      );
    }
  }
  return output;
}

function modelRelative(root: string, file: string): string {
  const relative = path.relative(root, file).split(path.sep).join("/");
  if (!relative || relative.startsWith("../")) throw new Error("模型资源越过模型目录");
  return relative;
}

function uvRegion(uvs: number[], width: number, height: number) {
  const points = pairs(uvs).map(([u, v]) => [u * width, (1 - v) * height] as const);
  const bounds = boundsOfPoints(points);
  const minX = Math.max(0, Math.floor(bounds.minX));
  const minY = Math.max(0, Math.floor(bounds.minY));
  const maxX = Math.min(width, Math.ceil(bounds.maxX));
  const maxY = Math.min(height, Math.ceil(bounds.maxY));
  return { x: minX, y: minY, width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY) };
}

function scaledGuideSize(width: number, height: number): { width: number; height: number } {
  const scale = Math.min(1, 2000 / width, 2000 / height);
  return {
    width: Math.max(512, Math.round(width * scale / 16) * 16),
    height: Math.max(512, Math.round(height * scale / 16) * 16),
  };
}

function directional(modelUnits: number, reference: number, pixelsPerUnit: number) {
  return { pixels: round(modelUnits * pixelsPerUnit), ratio: round(modelUnits / reference) };
}

function partOrder(group: ModelEditPartGroup): number {
  return ["iris", "eye", "brow", "face", "mouth", "hair-front", "hair-side", "hair-back", "clothing", "body", "arm", "leg", "shoe", "accessory", "effect", "unknown"].indexOf(group);
}


function pairs(values: number[]): Array<readonly [number, number]> {
  const output: Array<readonly [number, number]> = [];
  for (let index = 0; index < values.length; index += 2) output.push([values[index]!, values[index + 1]!]);
  return output;
}

function boundsOfPairs(values: number[]): Bounds {
  return boundsOfPoints(pairs(values));
}

function boundsOfPoints(points: ReadonlyArray<readonly [number, number]>): Bounds {
  const result = emptyBounds();
  for (const point of points) include(result, point);
  return result;
}

function unionBounds(values: Bounds[]): Bounds {
  const result = emptyBounds();
  for (const value of values) {
    include(result, [value.minX, value.minY]);
    include(result, [value.maxX, value.maxY]);
  }
  return result;
}

function emptyBounds(): Bounds {
  return { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
}

function include(target: Bounds, point: readonly [number, number]): void {
  target.minX = Math.min(target.minX, point[0]);
  target.minY = Math.min(target.minY, point[1]);
  target.maxX = Math.max(target.maxX, point[0]);
  target.maxY = Math.max(target.maxY, point[1]);
}

function width(value: Bounds): number { return value.maxX - value.minX; }
function height(value: Bounds): number { return value.maxY - value.minY; }

function edge(a: readonly [number, number], b: readonly [number, number], point: readonly [number, number]): number {
  return (point[0] - a[0]) * (b[1] - a[1]) - (point[1] - a[1]) * (b[0] - a[0]);
}

function roundedArray(values: number[]): number[] {
  return values.map((value) => Math.round(value * 1_000_000) / 1_000_000);
}

function round(value: number): number { return Math.round(value * 10_000) / 10_000; }

async function writeJson(file: string, value: unknown): Promise<void> {
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function writeJsonAtomically(file: string, value: unknown): Promise<void> {
  await writeFileAtomically(file, Buffer.from(`${JSON.stringify(value, null, 2)}\n`, "utf8"));
}

async function writeFileAtomically(file: string, value: Uint8Array): Promise<void> {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}.tmp`;
  try {
    await writeFile(temporary, value);
    await rename(temporary, file);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

function normalizeModelRelative(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
