import { createHash, randomUUID } from "node:crypto";
import {
  access,
  copyFile,
  cp,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import JSZip from "jszip";

import {
  PET_BEHAVIOR_STATES,
  PET_EMOTIONS,
  type BehaviorCandidate,
  type ExpressionAsset,
  type ModelCapabilities,
  type ModelDeleteResult,
  type ModelImportIssue,
  type ModelImportResult,
  type ModelPackage,
  type ModelToggleCreation,
  type MotionAsset,
  type PetBehaviorState,
  type PetManifest,
} from "../src/shared/contracts";
import {
  resolveLipSyncAdapter,
  type LipVowel,
  type ModelParameterDescriptor,
  type VowelParameterMap,
} from "../src/shared/lipsync-adapter";
import { scanModelAssets, type ModelAssetCatalog } from "./model-assets";
import { invalidImportReport, validImportReport } from "./model-import-report";
import { inspectCubismModel } from "./cubism-core-inspector";
import { normalizeNaturalIdleSettings } from "../src/shared/natural-idle-settings";
import { normalizeDefaultAppearance, normalizeOutfits } from "../src/shared/default-appearance";
import { normalizePartGroups } from "../src/shared/part-groups";

const MAX_FILES = 4096;
const MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024;
const SETTINGS_FILE = "desktop-pet.json";
const LEGACY_MANIFEST_BACKUP_FILE = "pet.manifest.v2.backup.json";

interface CubismModelJson {
  [key: string]: unknown;
  FileReferences?: {
    Moc?: unknown;
    Textures?: unknown;
    Physics?: unknown;
    Pose?: unknown;
    UserData?: unknown;
    DisplayInfo?: unknown;
    Motions?: unknown;
    Expressions?: unknown;
  };
  HitAreas?: unknown;
  Groups?: unknown;
}

interface StoredSettings {
  selectedModelId?: string;
}

export class ModelStore {
  readonly modelsRoot: string;
  private readonly settingsPath: string;

  constructor(userDataPath: string, private readonly cubismCorePath?: string) {
    this.modelsRoot = path.join(userDataPath, "models");
    this.settingsPath = path.join(userDataPath, SETTINGS_FILE);
  }

  async initialize(): Promise<void> {
    await mkdir(this.modelsRoot, { recursive: true });
  }

  async list(): Promise<ModelPackage[]> {
    await this.initialize();
    const entries = await readdir(this.modelsRoot, { withFileTypes: true });
    const packages = await Promise.all(
      entries
        .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
        .map(async (entry) => {
          try {
            return await this.load(entry.name);
          } catch {
            return null;
          }
        }),
    );
    return packages
      .filter((item): item is ModelPackage => item !== null)
      .sort((left, right) => left.name.localeCompare(right.name));
  }

  async selected(): Promise<ModelPackage | null> {
    const settings = await this.readSettings();
    if (settings.selectedModelId) {
      try {
        return await this.load(settings.selectedModelId);
      } catch {
        // Fall through to the first still-valid package.
      }
    }
    return (await this.list())[0] ?? null;
  }

  async select(id: string): Promise<ModelPackage> {
    const model = await this.load(id);
    await this.writeSettings({ selectedModelId: id });
    return model;
  }

  /** `name` is shown instead of the folder name, e.g. for the built-in look. */
  async importFolder(source: string, name?: string): Promise<ModelImportResult> {
    try {
      const sourceInfo = await stat(source);
      if (!sourceInfo.isDirectory()) {
        throw new Error("选择的路径不是文件夹");
      }
      return await this.importIntoStaging(async (staging) => {
        await cp(source, staging, { recursive: true, force: false });
      }, name ?? path.basename(source));
    } catch (error) {
      return { model: null, report: invalidImportReport(error) };
    }
  }

  async importZip(source: string): Promise<ModelImportResult> {
    try {
      if (path.extname(source).toLowerCase() !== ".zip") {
        throw new Error("请选择 Live2D ZIP 文件");
      }
      const archive = await JSZip.loadAsync(await readFile(source), {
        checkCRC32: true,
      });
      const files = Object.values(archive.files).filter(
        (entry) => !entry.dir && !entry.name.startsWith("__MACOSX/"),
      );
      if (files.length === 0 || files.length > MAX_FILES) {
        throw new Error("ZIP 文件数量不符合限制");
      }
      return await this.importIntoStaging(async (staging) => {
        let totalBytes = 0;
        for (const entry of files) {
          const zipEntry = entry as typeof entry & {
            unsafeOriginalName?: string;
            unixPermissions?: number | string;
          };
          const originalName = zipEntry.unsafeOriginalName ?? entry.name;
          const relative = safeRelativePath(originalName);
          const mode =
            typeof zipEntry.unixPermissions === "string"
              ? Number.parseInt(zipEntry.unixPermissions, 8)
              : zipEntry.unixPermissions;
          if (typeof mode === "number" && (mode & 0o170000) === 0o120000) {
            throw new Error("ZIP 不能包含符号链接");
          }
          const content = await entry.async("uint8array");
          totalBytes += content.byteLength;
          if (totalBytes > MAX_UNCOMPRESSED_BYTES) {
            throw new Error("ZIP 解压后超过 512 MB 限制");
          }
          const target = safeJoin(staging, relative);
          await mkdir(path.dirname(target), { recursive: true });
          await writeFile(target, content);
        }
      }, path.basename(source, path.extname(source)));
    } catch (error) {
      return { model: null, report: invalidImportReport(error) };
    }
  }

  async deleteAsset(
    id: string,
    kind: "motion" | "expression",
    assetId: string,
  ): Promise<ModelPackage> {
    const current = await this.load(id);
    const root = this.modelRoot(id);
    const asset = kind === "motion"
      ? current.manifest.motionAssets[assetId]
      : current.manifest.expressionAssets[assetId];
    if (!asset) throw new Error("要删除的素材不存在");
    const modelJson = await readModelJson(safeJoin(root, current.manifest.entry));
    removeAssetDescriptor(modelJson, kind, asset);
    await writeJsonAtomically(safeJoin(root, current.manifest.entry), modelJson);
    const catalog = await scanModelAssets(root, current.manifest.entry, modelJson);
    const capabilities = await capabilitiesFrom(
      modelJson,
      root,
      current.manifest.entry,
      this.cubismCorePath,
    );
    const manifest = reconcileManifest(current.manifest, catalog, capabilities, id);
    await writeJsonAtomically(path.join(root, "pet.manifest.json"), manifest);
    const stillReferenced = [
      ...Object.values(catalog.motionAssets),
      ...Object.values(catalog.expressionAssets),
    ].some((candidate) => candidate.file === asset.file);
    if (!stillReferenced) await rm(safeJoin(root, asset.file), { force: true });
    return this.load(id);
  }

  async createToggle(
    id: string,
    name: string,
    parameterValues: Record<string, number>,
  ): Promise<ModelToggleCreation> {
    const current = await this.load(id);
    const label = toggleName(name);
    const entries = Object.entries(parameterValues ?? {});
    if (entries.length === 0) throw new Error("先在「参数」里固定至少一个参数");
    if (entries.length > 128) throw new Error("一个开关最多包含 128 个参数");
    // Values come from the preview's own parameter ranges, which DisplayInfo may not list.
    const parameters = entries.map(([parameterId, value]) => {
      if (!/^[\w.-]{1,128}$/.test(parameterId)) throw new Error(`参数 ID 无效：${parameterId}`);
      if (typeof value !== "number" || !Number.isFinite(value) || Math.abs(value) > 10_000) {
        throw new Error(`参数 ${parameterId} 的数值无效`);
      }
      return { Id: parameterId, Value: value, Blend: "Overwrite" };
    });
    const root = this.modelRoot(id);
    const entry = current.manifest.entry;
    const modelJson = await readModelJson(safeJoin(root, entry));
    const references = ensureRecord(modelJson, "FileReferences");
    const expressions = Array.isArray(references.Expressions) ? references.Expressions : [];
    const usedNames = new Set(expressions.flatMap((item) =>
      isRecord(item) && typeof item.Name === "string" ? [item.Name] : []));
    const modelDirectory = path.posix.dirname(entry);
    let fileName = "";
    for (let attempt = 1; ; attempt += 1) {
      fileName = `expressions/${attempt === 1 ? label : `${label}-${attempt}`}.exp3.json`;
      if (!await fileExists(safeJoin(root, path.posix.join(modelDirectory, fileName)))) break;
    }
    const file = safeJoin(root, path.posix.join(modelDirectory, fileName));
    await mkdir(path.dirname(file), { recursive: true });
    await writeJsonAtomically(file, { Type: "Live2D Expression", Parameters: parameters });
    expressions.push({ Name: availableName(label, usedNames), File: fileName });
    references.Expressions = expressions;
    await writeJsonAtomically(safeJoin(root, entry), modelJson);
    const model = await this.load(id);
    const assetId = Object.keys(model.manifest.expressionAssets)
      .find((assetId) => !Object.hasOwn(current.manifest.expressionAssets, assetId));
    if (!assetId) throw new Error("开关没有建成，请重试");
    return { model, assetId };
  }

  /**
   * What the look shows at rest: the model's files and its saved default
   * appearance, not its behaviors. A card picture made under another stamp
   * is out of date.
   */
  async lookStamp(id: string): Promise<string> {
    const model = await this.load(id);
    const root = this.modelRoot(id);
    const files = await modelRevision(root, model.manifest.entry, await readModelJson(safeJoin(root, model.manifest.entry)), false);
    return createHash("sha256")
      .update(files).update("\n").update(JSON.stringify(model.manifest.defaultAppearance ?? null))
      .digest("hex").slice(0, 16);
  }

  /** The look key the card picture was made under, if it recorded one. */
  async previewKey(id: string): Promise<string | null> {
    try {
      const value = JSON.parse(await readFile(path.join(this.modelRoot(id), ".preview.json"), "utf8")) as { key?: unknown };
      return typeof value.key === "string" ? value.key : null;
    } catch {
      return null;
    }
  }

  async savePreview(
    id: string,
    png: ArrayBuffer | Uint8Array,
    key: string | null = null,
  ): Promise<ModelPackage> {
    const root = this.modelRoot(id);
    await this.load(id);
    const bytes = Buffer.from(png instanceof ArrayBuffer ? new Uint8Array(png) : png);
    if (bytes.byteLength > 4 * 1024 * 1024) throw new Error("预览图不能超过 4 MB");
    if (!bytes.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) {
      throw new Error("预览图必须是 PNG");
    }
    const target = path.join(root, ".preview.png");
    const temporary = `${target}.${randomUUID()}.tmp`;
    await writeFile(temporary, bytes);
    await rename(temporary, target);
    if (key) await writeJsonAtomically(path.join(root, ".preview.json"), { key });
    else await rm(path.join(root, ".preview.json"), { force: true });
    return this.load(id);
  }

  async delete(id: string): Promise<ModelDeleteResult> {
    const models = await this.list();
    if (!models.some((model) => model.id === id)) throw new Error("模型不存在");
    if (models.length <= 1) throw new Error("不能删除最后一个模型");
    await rm(this.modelRoot(id), { recursive: true, force: true });
    const settings = await this.readSettings();
    const selectedModel = settings.selectedModelId === id
      ? models.find((model) => model.id !== id)!
      : await this.selected() ?? models.find((model) => model.id !== id)!;
    await this.writeSettings({ selectedModelId: selectedModel.id });
    return { deletedId: id, selectedModel };
  }

  async saveManifest(id: string, manifest: PetManifest): Promise<ModelPackage> {
    assertAssetDisplayNames(manifest);
    const root = this.modelRoot(id);
    const current = await this.load(id);
    const modelJson = await readModelJson(safeJoin(root, current.manifest.entry));
    const catalog = await scanModelAssets(
      root,
      current.manifest.entry,
      modelJson,
    );
    const validated = validateManifest(
      manifest,
      current.capabilities,
      catalog,
      id,
    );
    validated.entry = current.manifest.entry;
    await writeJsonAtomically(path.join(root, "pet.manifest.json"), validated);
    return this.load(id);
  }

  async load(id: string): Promise<ModelPackage> {
    const root = this.modelRoot(id);
    const manifestValue = JSON.parse(
      await readFile(path.join(root, "pet.manifest.json"), "utf8"),
    ) as unknown;
    const entry = manifestEntry(manifestValue);
    const modelJson = await readModelJson(safeJoin(root, entry));
    const capabilities = await capabilitiesFrom(modelJson, root, entry, this.cubismCorePath);
    const catalog = await scanModelAssets(root, entry, modelJson);
    const versionThree = isVersionThreeManifest(manifestValue);
    const versionTwo = isVersionTwoManifest(manifestValue);
    const manifest = versionThree
      ? validateManifest(manifestValue, capabilities, catalog, id)
      : versionTwo
        ? migrateVersionTwoManifest(manifestValue, capabilities, catalog, id, entry)
        : createManifest(
            id,
            manifestName(manifestValue, id),
            entry,
            capabilities,
            catalog,
          );
    await validateReferences(root, entry, modelJson);
    if (versionTwo) {
      const backup = path.join(root, LEGACY_MANIFEST_BACKUP_FILE);
      if (!await fileExists(backup)) await writeJsonAtomically(backup, manifestValue);
    }
    if (
      !versionThree ||
      manifestNeedsRepair(manifestValue, manifest)
    ) {
      await writeJsonAtomically(path.join(root, "pet.manifest.json"), manifest);
    }
    const revision = await modelRevision(root, entry, modelJson);
    return {
      id,
      name: manifest.name,
      url: modelUrl(id, entry, revision),
      previewUrl: await fileExists(path.join(root, ".preview.png"))
        ? modelUrl(id, ".preview.png", revision)
        : null,
      manifest,
      capabilities,
    };
  }

  resolveModelFile(id: string, relative: string): string {
    return safeJoin(
      this.modelRoot(id),
      safeRelativePath(stripModelRevision(relative)),
    );
  }

  resolveModelRoot(id: string): string {
    return this.modelRoot(id);
  }

  private modelRoot(id: string): string {
    if (!/^[a-z0-9][a-z0-9-]{0,79}$/.test(id)) {
      throw new Error("无效的模型 ID");
    }
    return safeJoin(this.modelsRoot, id);
  }

  private async importIntoStaging(
    populate: (staging: string) => Promise<void>,
    suggestedName: string,
  ): Promise<ModelImportResult> {
    await this.initialize();
    const staging = path.join(this.modelsRoot, `.import-${randomUUID()}`);
    await mkdir(staging, { recursive: false });
    try {
      await populate(staging);
      const entries = await findFiles(staging, (value) =>
        value.toLowerCase().endsWith(".model3.json"),
      );
      if (entries.length !== 1) {
        throw new Error("模型包必须且只能包含一个 .model3.json");
      }
      const entry = path.relative(staging, entries[0]!).split(path.sep).join("/");
      const modelJson = await readModelJson(entries[0]!);
      await validateReferences(staging, entry, modelJson);
      const registered = await registerUnreferencedAssets(staging, entry, modelJson);
      const capabilities = await capabilitiesFrom(
        modelJson,
        staging,
        entry,
        this.cubismCorePath,
      );
      const catalog = await scanModelAssets(staging, entry, modelJson);
      const suppliedManifest = await readSuppliedManifest(staging, entry);
      if (suppliedManifest) {
        const suppliedEntry = manifestEntry(suppliedManifest);
        const entryBesideModel = path.posix.join(
          path.posix.dirname(entry),
          suppliedEntry,
        );
        if (suppliedEntry !== entry && entryBesideModel !== entry) {
          throw new Error("pet.manifest.json 的模型入口与模型包不一致");
        }
      }
      const suppliedName = suppliedManifest && typeof suppliedManifest.name === "string"
        ? suppliedManifest.name
        : "";
      const name = cleanName(
        suppliedName || suggestedName || path.basename(entry, ".model3.json"),
      );
      const id = await availableId(
        this.modelsRoot,
        `${slug(name)}-${createHash("sha1").update(randomUUID()).digest("hex").slice(0, 8)}`,
      );
      let manifest: PetManifest;
      if (suppliedManifest && isVersionThreeManifest(suppliedManifest)) {
        manifest = validateManifest(suppliedManifest, capabilities, catalog, id);
      } else if (suppliedManifest && isVersionTwoManifest(suppliedManifest)) {
        await writeJsonAtomically(
          path.join(staging, LEGACY_MANIFEST_BACKUP_FILE),
          suppliedManifest,
        );
        manifest = migrateVersionTwoManifest(
          suppliedManifest,
          capabilities,
          catalog,
          id,
          entry,
        );
      } else {
        manifest = createManifest(id, name, entry, capabilities, catalog);
      }
      manifest.entry = entry;
      await writeFile(
        path.join(staging, "pet.manifest.json"),
        `${JSON.stringify(manifest, null, 2)}\n`,
        "utf8",
      );
      await rename(staging, this.modelRoot(id));
      await this.writeSettings({ selectedModelId: id });
      const checks: ModelImportIssue[] = [
        { code: "entry", message: "找到唯一模型入口", file: entry },
        {
          code: "assets",
          message: `已登记 ${Object.keys(catalog.motionAssets).length} 个动作和 ${Object.keys(catalog.expressionAssets).length} 个表情`,
          file: entry,
        },
      ];
      if (registered > 0) {
        checks.push({
          code: "unregistered-assets",
          message: `自动登记了 ${registered} 个未注册素材`,
          file: entry,
        });
      }
      return {
        model: await this.load(id),
        report: validImportReport(checks, await modelWarnings(this.modelRoot(id), entry, modelJson)),
      };
    } catch (error) {
      await rm(staging, { recursive: true, force: true });
      throw error;
    }
  }

  private async readSettings(): Promise<StoredSettings> {
    try {
      const parsed = JSON.parse(await readFile(this.settingsPath, "utf8"));
      return typeof parsed === "object" && parsed !== null
        ? (parsed as StoredSettings)
        : {};
    } catch {
      return {};
    }
  }

  private async writeSettings(settings: StoredSettings): Promise<void> {
    const temporary = `${this.settingsPath}.tmp`;
    await writeFile(temporary, `${JSON.stringify(settings, null, 2)}\n`, "utf8");
    await rename(temporary, this.settingsPath);
  }
}

function createManifest(
  id: string,
  name: string,
  entry: string,
  capabilities: ModelCapabilities,
  catalog: ModelAssetCatalog,
): PetManifest {
  const lipSyncAdapter = resolveLipSyncAdapter(capabilities);
  return {
    schemaVersion: 3,
    id,
    name,
    renderer: "live2d-cubism",
    entry,
    display: { scale: 0.95, x: 0.5, y: 0.9 },
    motionAssets: catalog.motionAssets,
    expressionAssets: catalog.expressionAssets,
    behaviors: emptyBehaviors(),
    lipsync: {
      ...lipSyncAdapter,
      maxOpen: 1,
    },
  };
}

function validateManifest(
  value: unknown,
  capabilities: ModelCapabilities,
  catalog: ModelAssetCatalog,
  id: string,
): PetManifest {
  if (!isVersionThreeManifest(value)) {
    throw new Error("pet.manifest.json 必须是对象");
  }
  const entry = manifestEntry(value);
  const name = typeof value.name === "string" ? cleanName(value.name) : id;
  const displayValue = isRecord(value.display) ? value.display : {};
  const display = {
    scale: finiteBetween(displayValue.scale, 0.05, 5, 0.95),
    // Window placement is global. Keep the hidden model anchor centered
    // instead of exposing a second positioning system per model.
    x: 0.5,
    y: 0.9,
  };
  const motionAssets = preserveAssetNames(
    catalog.motionAssets,
    value.motionAssets,
  );
  const expressionAssets = preserveAssetNames(
    catalog.expressionAssets,
    value.expressionAssets,
  );
  const behaviors = validateBehaviors(
    value.behaviors,
    new Set(Object.keys(motionAssets)),
    new Set(Object.keys(expressionAssets)),
  );
  const lipsync = isRecord(value.lipsync) ? value.lipsync : {};
  return {
    schemaVersion: 3,
    id,
    name,
    renderer: "live2d-cubism",
    entry,
    display,
    motionAssets,
    expressionAssets,
    behaviors,
    lipsync: validateLipSync(lipsync, capabilities),
    ...(value.naturalIdle === undefined ? {} : {
      naturalIdle: normalizeNaturalIdleSettings(value.naturalIdle),
    }),
    ...(value.defaultAppearance === undefined ? {} : {
      defaultAppearance: normalizeDefaultAppearance(value.defaultAppearance, expressionAssets),
    }),
    ...(() => {
      const outfits = normalizeOutfits(value.outfits, expressionAssets);
      return outfits.length ? { outfits } : {};
    })(),
    // An empty list is a deliberate choice (everything ungrouped), unlike a missing one.
    ...(value.partGroups === undefined ? {} : { partGroups: normalizePartGroups(value.partGroups) }),
  };
}

function migrateVersionTwoManifest(
  value: Record<string, unknown> & { schemaVersion: 2 },
  capabilities: ModelCapabilities,
  catalog: ModelAssetCatalog,
  id: string,
  entry: string,
): PetManifest {
  const migrated = createManifest(
    id,
    typeof value.name === "string" ? cleanName(value.name) : id,
    entry,
    capabilities,
    catalog,
  );
  const display = isRecord(value.display) ? value.display : {};
  migrated.display = {
    scale: finiteBetween(display.scale, 0.05, 5, 0.95),
    x: 0.5,
    y: 0.9,
  };
  migrated.motionAssets = preserveAssetNames(catalog.motionAssets, value.motionAssets);
  migrated.expressionAssets = preserveAssetNames(catalog.expressionAssets, value.expressionAssets);
  migrated.lipsync = validateLipSync(isRecord(value.lipsync) ? value.lipsync : {}, capabilities);
  return migrated;
}

function emptyBehaviors(): PetManifest["behaviors"] {
  return {
    states: emptyCandidateMap(PET_BEHAVIOR_STATES),
    emotions: emptyCandidateMap(PET_EMOTIONS),
  };
}

function emptyCandidateMap<T extends string>(keys: readonly T[]): Record<T, BehaviorCandidate[]> {
  const result = {} as Record<T, BehaviorCandidate[]>;
  for (const key of keys) result[key] = [];
  return result;
}

function validateBehaviors(
  value: unknown,
  motionIds: Set<string>,
  expressionIds: Set<string>,
): PetManifest["behaviors"] {
  const source = isRecord(value) ? value : {};
  const states = isRecord(source.states) ? source.states : {};
  const emotions = isRecord(source.emotions) ? source.emotions : {};
  return {
    states: validateCandidateMap(states, PET_BEHAVIOR_STATES, motionIds, expressionIds) as Record<
      PetBehaviorState,
      BehaviorCandidate[]
    >,
    emotions: validateCandidateMap(emotions, PET_EMOTIONS, motionIds, expressionIds) as PetManifest[
      "behaviors"
    ]["emotions"],
  };
}

function validateCandidateMap(
  source: Record<string, unknown>,
  keys: readonly string[],
  motionIds: Set<string>,
  expressionIds: Set<string>,
): Record<string, BehaviorCandidate[]> {
  const used = new Set<string>();
  return Object.fromEntries(keys.map((key) => [
    key,
    (Array.isArray(source[key]) ? source[key] : []).flatMap((item): BehaviorCandidate[] => {
      if (!isRecord(item) || typeof item.id !== "string") return [];
      const id = item.id.trim().slice(0, 120);
      if (!id || used.has(id)) return [];
      const rawMotionIds = stringArray(item.motionAssetIds);
      const rawExpressionIds = stringArray(item.expressionAssetIds);
      if (
        !Array.isArray(item.motionAssetIds) ||
        !Array.isArray(item.expressionAssetIds) ||
        rawMotionIds.length !== item.motionAssetIds.length ||
        rawExpressionIds.length !== item.expressionAssetIds.length ||
        rawMotionIds.some((assetId) => !motionIds.has(assetId)) ||
        rawExpressionIds.some((assetId) => !expressionIds.has(assetId))
      ) return [];
      const motionAssetIds = [...new Set(rawMotionIds)];
      const expressionAssetIds = [...new Set(rawExpressionIds)];
      if (!motionAssetIds.length && !expressionAssetIds.length) return [];
      used.add(id);
      return [{ id, motionAssetIds, expressionAssetIds }];
    }),
  ]));
}

function preserveAssetNames<T extends MotionAsset | ExpressionAsset>(
  catalog: Record<string, T>,
  candidate: unknown,
): Record<string, T> {
  const source = isRecord(candidate) ? candidate : {};
  return Object.fromEntries(
    Object.entries(catalog).map(([id, asset]) => {
      const saved = source[id];
      const name = isRecord(saved) && typeof saved.name === "string"
        ? cleanName(saved.name)
        : asset.name;
      return [id, { ...asset, name }];
    }),
  );
}

function assertAssetDisplayNames(manifest: PetManifest): void {
  for (const asset of [
    ...Object.values(manifest.motionAssets),
    ...Object.values(manifest.expressionAssets),
  ]) {
    if (!asset.name.trim()) throw new Error("素材名称不能为空");
  }
}

function removeAssetDescriptor(
  model: CubismModelJson,
  kind: "motion" | "expression",
  asset: MotionAsset | ExpressionAsset,
): void {
  const references = isRecord(model.FileReferences) ? model.FileReferences : {};
  if (kind === "motion") {
    const motion = asset as MotionAsset;
    if (!isRecord(references.Motions)) return;
    const group = references.Motions[motion.group];
    if (!Array.isArray(group)) return;
    group.splice(motion.index, 1);
    if (!group.length) delete references.Motions[motion.group];
    return;
  }
  const expression = asset as ExpressionAsset;
  if (!Array.isArray(references.Expressions)) return;
  references.Expressions = references.Expressions.filter(
    (value) => !(
      isRecord(value) &&
      value.Name === expression.expression
    ),
  );
}

function reconcileManifest(
  previous: PetManifest,
  catalog: ModelAssetCatalog,
  capabilities: ModelCapabilities,
  id: string,
): PetManifest {
  const motionMapping = mapAssetIds(
    previous.motionAssets,
    catalog.motionAssets,
    (asset) => `${asset.group}\0${asset.file}`,
  );
  const expressionMapping = mapAssetIds(
    previous.expressionAssets,
    catalog.expressionAssets,
    (asset) => `${asset.expression}\0${asset.file}`,
  );
  const motionAssets = copyMappedNames(
    previous.motionAssets,
    catalog.motionAssets,
    motionMapping,
  );
  const expressionAssets = copyMappedNames(
    previous.expressionAssets,
    catalog.expressionAssets,
    expressionMapping,
  );
  const candidate: PetManifest = {
    ...previous,
    motionAssets,
    expressionAssets,
    behaviors: {
      states: remapCandidateMap(previous.behaviors.states, motionMapping, expressionMapping),
      emotions: remapCandidateMap(previous.behaviors.emotions, motionMapping, expressionMapping),
    },
  };
  return validateManifest(candidate, capabilities, catalog, id);
}

function mapAssetIds<T>(
  previous: Record<string, T>,
  current: Record<string, T>,
  signature: (asset: T) => string,
): Map<string, string> {
  const result = new Map<string, string>();
  const available = new Map<string, string[]>();
  for (const [id, asset] of Object.entries(current)) {
    const key = signature(asset);
    available.set(key, [...(available.get(key) ?? []), id]);
  }
  for (const [id, asset] of Object.entries(previous)) {
    if (current[id]) {
      result.set(id, id);
      continue;
    }
    const matches = available.get(signature(asset));
    const replacement = matches?.shift();
    if (replacement) result.set(id, replacement);
  }
  return result;
}

function copyMappedNames<T extends MotionAsset | ExpressionAsset>(
  previous: Record<string, T>,
  current: Record<string, T>,
  mapping: Map<string, string>,
): Record<string, T> {
  const result = structuredClone(current);
  for (const [oldId, newId] of mapping) {
    const oldAsset = previous[oldId];
    const newAsset = result[newId];
    if (oldAsset && newAsset) newAsset.name = oldAsset.name;
  }
  return result;
}

function remapCandidateMap<T extends string>(
  candidates: Record<T, BehaviorCandidate[]>,
  motionMapping: Map<string, string>,
  expressionMapping: Map<string, string>,
): Record<T, BehaviorCandidate[]> {
  return Object.fromEntries(
    Object.entries(candidates).map(([key, values]) => [
      key,
      (values as BehaviorCandidate[]).flatMap((candidate) => {
        const motionAssetIds = candidate.motionAssetIds.map((id) => motionMapping.get(id));
        const expressionAssetIds = candidate.expressionAssetIds.map((id) => expressionMapping.get(id));
        // A partially reconstructed candidate was never jointly previewed. Drop the
        // entire combination if even one referenced asset disappeared.
        if (motionAssetIds.some((id) => !id) || expressionAssetIds.some((id) => !id)) {
          return [];
        }
        return [{
          id: candidate.id,
          motionAssetIds: motionAssetIds as string[],
          expressionAssetIds: expressionAssetIds as string[],
        }];
      }),
    ]),
  ) as Record<T, BehaviorCandidate[]>;
}

async function validateReferences(
  packageRoot: string,
  entry: string,
  model: CubismModelJson,
): Promise<void> {
  const modelDirectory = path.posix.dirname(entry);
  const references = referencedFiles(model);
  if (!references.some((value) => value.toLowerCase().endsWith(".moc3"))) {
    throw new Error("模型缺少 .moc3 引用");
  }
  if (!references.some((value) => /\.(png|webp)$/i.test(value))) {
    throw new Error("模型缺少纹理引用");
  }
  for (const reference of references) {
    const relative = path.posix.join(modelDirectory, safeRelativePath(reference));
    await access(safeJoin(packageRoot, relative));
  }
}

async function registerUnreferencedAssets(
  packageRoot: string,
  entry: string,
  model: CubismModelJson,
): Promise<number> {
  const catalog = await scanModelAssets(packageRoot, entry, model);
  const registered = new Set([
    ...Object.values(catalog.motionAssets).map((asset) => asset.file),
    ...Object.values(catalog.expressionAssets).map((asset) => asset.file),
  ]);
  const candidates = await findFiles(packageRoot, (value) =>
    /\.(motion3|exp3)\.json$/i.test(value.replaceAll("\\", "/")),
  );
  const references = ensureRecord(model, "FileReferences");
  const motions = ensureRecord(references, "Motions");
  const expressions = Array.isArray(references.Expressions)
    ? references.Expressions
    : [];
  const importedMotions = Array.isArray(motions.Imported)
    ? motions.Imported
    : [];
  let added = 0;

  for (const absolute of candidates) {
    const relative = path.relative(packageRoot, absolute).split(path.sep).join("/");
    if (registered.has(relative)) continue;
    const descriptorPath = path.posix.relative(path.posix.dirname(entry), relative);
    if (descriptorPath.startsWith("../") || descriptorPath === "..") continue;
    const value = await readAssetJson(absolute);
    if (relative.toLowerCase().endsWith(".motion3.json")) {
      validateMotionJson(value, relative);
      importedMotions.push({ File: descriptorPath });
      motions.Imported = importedMotions;
    } else {
      validateExpressionJson(value, relative);
      const base = assetBaseName(relative, ".exp3.json");
      const used = new Set(
        expressions.flatMap((item) =>
          isRecord(item) && typeof item.Name === "string" ? [item.Name] : [],
        ),
      );
      expressions.push({ Name: availableName(base, used), File: descriptorPath });
      references.Expressions = expressions;
    }
    added += 1;
  }

  if (added > 0) {
    await writeJsonAtomically(safeJoin(packageRoot, entry), model);
  }
  return added;
}

async function modelWarnings(
  packageRoot: string,
  entry: string,
  model: CubismModelJson,
): Promise<ModelImportIssue[]> {
  const warnings: ModelImportIssue[] = [];
  const refs = isRecord(model.FileReferences) ? model.FileReferences : {};
  const modelDirectory = path.posix.dirname(entry);
  if (typeof refs.Moc === "string") {
    const moc = path.posix.join(modelDirectory, safeRelativePath(refs.Moc));
    const mocFile = safeJoin(packageRoot, moc);
    if ((await stat(mocFile)).size < 1_024) {
      warnings.push({ code: "moc-size", message: "MOC 文件异常小，请确认模型完整", file: moc });
    }
    const header = (await readFile(mocFile)).subarray(0, 4).toString("ascii");
    if (header !== "MOC3") {
      warnings.push({ code: "moc-header", message: "MOC 文件头不是标准 MOC3", file: moc });
    }
  }
  for (const reference of referencedFiles(model)) {
    const relative = path.posix.join(modelDirectory, reference);
    if (!(await hasExactPathCase(packageRoot, relative))) {
      warnings.push({
        code: "path-case",
        message: "资源路径大小写与磁盘文件不完全一致",
        file: relative,
      });
    }
  }
  return warnings;
}

async function hasExactPathCase(root: string, relative: string): Promise<boolean> {
  let current = root;
  for (const segment of relative.split("/")) {
    const names = await readdir(current);
    if (!names.includes(segment)) return false;
    current = path.join(current, segment);
  }
  return true;
}

async function readAssetJson(file: string): Promise<Record<string, unknown>> {
  const value: unknown = JSON.parse(await readFile(file, "utf8"));
  if (!isRecord(value)) throw new Error(`Live2D 素材必须是 JSON 对象：${file}`);
  return value;
}

function validateMotionJson(value: Record<string, unknown>, file: string): void {
  if (!isRecord(value.Meta) || !Array.isArray(value.Curves)) {
    throw new Error(`动作文件格式无效：${file}`);
  }
}

function validateExpressionJson(value: Record<string, unknown>, file: string): void {
  if (!Array.isArray(value.Parameters)) {
    throw new Error(`表情文件格式无效：${file}`);
  }
}

function ensureRecord(
  target: Record<string, unknown>,
  key: string,
): Record<string, unknown> {
  if (!isRecord(target[key])) target[key] = {};
  return target[key] as Record<string, unknown>;
}

function assetBaseName(file: string, suffix: string): string {
  const basename = path.posix.basename(file);
  return basename.toLowerCase().endsWith(suffix)
    ? basename.slice(0, -suffix.length)
    : basename;
}

/** A toggle's name doubles as its file name, so path characters are dropped. */
function toggleName(value: unknown): string {
  const name = typeof value === "string"
    ? value.replace(/[\\/:*?"<>|\u0000-\u001f]/g, "").replace(/^\.+/, "").trim()
    : "";
  if (!name) throw new Error("给开关起个名字");
  if ([...name].length > 20) throw new Error("开关名字最多 20 个字");
  return name;
}

function availableName(candidate: string, used: Set<string>): string {
  let result = candidate || "asset";
  let suffix = 2;
  while (used.has(result)) result = `${candidate}-${suffix++}`;
  return result;
}

function referencedFiles(model: CubismModelJson): string[] {
  const refs: Record<string, unknown> = isRecord(model.FileReferences)
    ? model.FileReferences
    : {};
  const values: string[] = [];
  for (const key of ["Moc", "Physics", "Pose", "UserData", "DisplayInfo"]) {
    if (typeof refs[key] === "string") values.push(refs[key]);
  }
  if (Array.isArray(refs.Textures)) {
    values.push(...refs.Textures.filter((item): item is string => typeof item === "string"));
  }
  if (isRecord(refs.Motions)) {
    for (const group of Object.values(refs.Motions)) {
      if (!Array.isArray(group)) continue;
      for (const motion of group) {
        if (isRecord(motion) && typeof motion.File === "string") values.push(motion.File);
      }
    }
  }
  if (Array.isArray(refs.Expressions)) {
    for (const expression of refs.Expressions) {
      if (isRecord(expression) && typeof expression.File === "string") {
        values.push(expression.File);
      }
    }
  }
  return values.map(safeRelativePath);
}

async function capabilitiesFrom(
  model: CubismModelJson,
  root: string,
  entry: string,
  cubismCorePath?: string,
): Promise<ModelCapabilities> {
  const refs: Record<string, unknown> = isRecord(model.FileReferences)
    ? model.FileReferences
    : {};
  const expressions = Array.isArray(refs.Expressions)
    ? refs.Expressions.flatMap((item) =>
        isRecord(item) && typeof item.Name === "string" ? [item.Name] : [],
      )
    : [];
  const hitAreas = Array.isArray(model.HitAreas)
    ? model.HitAreas.flatMap((item) =>
        isRecord(item) && typeof item.Name === "string" ? [item.Name] : [],
      )
    : [];
  const lipSyncParameters = Array.isArray(model.Groups)
    ? model.Groups.flatMap((group) => {
        if (
          !isRecord(group) ||
          group.Target !== "Parameter" ||
          typeof group.Name !== "string" ||
          group.Name.toLowerCase() !== "lipsync"
        ) {
          return [];
        }
        return stringArray(group.Ids);
      })
    : [];
  const diagnostics = await cubismDiagnostics(model, root, entry, cubismCorePath);
  return {
    motionGroups: isRecord(refs.Motions) ? Object.keys(refs.Motions) : [],
    expressions,
    hitAreas,
    lipSyncParameters: [...new Set(lipSyncParameters)],
    ...await displayModelInfo(model, root, entry),
    ...diagnostics,
  };
}

async function cubismDiagnostics(
  model: CubismModelJson,
  root: string,
  entry: string,
  corePath?: string,
): Promise<Pick<
  ModelCapabilities,
  "mocVersion" | "usesMotionSync" | "usesAdvancedBlendMode" | "usesOffscreenDrawing" | "warnings"
>> {
  const refs = isRecord(model.FileReferences) ? model.FileReferences : {};
  const usesMotionSync = Object.entries(refs).some(([key, value]) =>
    /motionsync/i.test(key) || (typeof value === "string" && /\.motionsync3\.json$/i.test(value)),
  );
  const warnings = usesMotionSync
    ? ["模型包含 MotionSync 数据；当前继续使用 tulpa 实时口型，官方 MotionSync 尚未启用。"]
    : [];
  if (!corePath || typeof refs.Moc !== "string") {
    return {
      mocVersion: undefined,
      usesMotionSync,
      usesAdvancedBlendMode: false,
      usesOffscreenDrawing: false,
      warnings,
    };
  }
  const mocRelative = path.posix.join(
    path.posix.dirname(entry),
    safeRelativePath(refs.Moc),
  );
  const inspected = await inspectCubismModel(corePath, safeJoin(root, mocRelative));
  return {
    mocVersion: inspected.mocVersion,
    usesMotionSync,
    usesAdvancedBlendMode: inspected.usesAdvancedBlendMode,
    usesOffscreenDrawing: inspected.usesOffscreenDrawing,
    warnings,
  };
}

async function displayModelInfo(
  model: CubismModelJson,
  root: string,
  entry: string,
): Promise<Pick<ModelCapabilities, "parameters" | "parts">> {
  const refs = isRecord(model.FileReferences) ? model.FileReferences : {};
  if (typeof refs.DisplayInfo !== "string") return { parameters: [] };
  try {
    const relative = path.posix.join(
      path.posix.dirname(entry),
      safeRelativePath(refs.DisplayInfo),
    );
    const parsed = JSON.parse(await readFile(safeJoin(root, relative), "utf8"));
    if (!isRecord(parsed)) return { parameters: [] };
    const parameters = (Array.isArray(parsed.Parameters) ? parsed.Parameters : []).flatMap((item): ModelParameterDescriptor[] => {
      if (!isRecord(item) || typeof item.Id !== "string") return [];
      return [{
        id: item.Id,
        name: typeof item.Name === "string" ? item.Name : item.Id,
        groupId: typeof item.GroupId === "string" ? item.GroupId : null,
      }];
    });
    const parts = (Array.isArray(parsed.Parts) ? parsed.Parts : []).flatMap(item =>
      isRecord(item) && typeof item.Id === "string" ? [{ id: item.Id, name: typeof item.Name === "string" ? item.Name : item.Id }] : []);
    return { parameters, ...(parts.length ? { parts } : {}) };
  } catch {
    return { parameters: [] };
  }
}

function validateLipSync(
  source: Record<string, unknown>,
  capabilities: ModelCapabilities,
): PetManifest["lipsync"] {
  const automatic = resolveLipSyncAdapter(capabilities);
  const storedMaxOpen = finiteBetween(source.maxOpen, 0, 1, 1);
  // 0.7 was the old generated default. Upgrade only that automatic value so
  // new and already-imported standard models both start at full mouth range;
  // explicit manual mappings and any other chosen strength remain untouched.
  const maxOpen = source.source !== "manual" && Math.abs(storedMaxOpen - 0.7) < 1e-9
    ? 1
    : storedMaxOpen;
  if (source.source !== "manual") return { ...automatic, maxOpen };

  const known = new Set([
    ...(capabilities.parameters ?? []).map((item) => item.id),
    ...(capabilities.lipSyncParameters ?? []),
  ]);
  const validId = (value: unknown): string | null => {
    if (typeof value !== "string" || !value.trim()) return null;
    const id = value.trim();
    return known.size === 0 || known.has(id) ? id : null;
  };
  const mode = source.mode === "viseme" ? "viseme" : "open-form";
  const rawVowels = isRecord(source.vowelParameters) ? source.vowelParameters : {};
  const vowelParameters = Object.fromEntries(
    (["a", "e", "i", "o", "u"] as LipVowel[]).map((vowel) => [
      vowel,
      validId(rawVowels[vowel]),
    ]),
  ) as VowelParameterMap;
  const mouthOpenParameter = validId(source.mouthOpenParameter);
  const mouthFormParameter = validId(source.mouthFormParameter);
  const openForm = (value: unknown) => mode === "open-form" ? validId(value) : null;
  const needsConfirmation = mode === "viseme"
    ? Object.values(vowelParameters).some((value) => !value)
    : !mouthOpenParameter;
  return {
    mode,
    source: "manual",
    needsConfirmation,
    mouthOpenParameter: mode === "open-form" ? mouthOpenParameter : null,
    mouthFormParameter: mode === "open-form" ? mouthFormParameter : null,
    jawParameter: openForm(source.jawParameter),
    puckerWidenParameter: openForm(source.puckerWidenParameter),
    funnelParameter: openForm(source.funnelParameter),
    vowelParameters: mode === "viseme"
      ? vowelParameters
      : { a: null, e: null, i: null, o: null, u: null },
    maxOpen,
  };
}

function manifestNeedsRepair(
  source: Record<string, unknown>,
  validated: PetManifest,
): boolean {
  return JSON.stringify(source) !== JSON.stringify(validated);
}

async function readModelJson(file: string): Promise<CubismModelJson> {
  const parsed = JSON.parse(await readFile(file, "utf8"));
  if (!isRecord(parsed)) throw new Error(".model3.json 必须是对象");
  return parsed;
}

async function readSuppliedManifest(
  packageRoot: string,
  entry: string,
): Promise<Record<string, unknown> | null> {
  const candidates = [
    path.join(packageRoot, "pet.manifest.json"),
    path.join(packageRoot, path.posix.dirname(entry), "pet.manifest.json"),
  ];
  for (const candidate of [...new Set(candidates)]) {
    try {
      const parsed: unknown = JSON.parse(await readFile(candidate, "utf8"));
      if (!isRecord(parsed)) {
        throw new Error("pet.manifest.json 必须是对象");
      }
      return parsed;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
      throw error;
    }
  }
  return null;
}

function manifestEntry(value: unknown): string {
  if (!isRecord(value) || typeof value.entry !== "string") {
    throw new Error("manifest 缺少模型入口");
  }
  return safeRelativePath(value.entry);
}

function manifestName(value: unknown, fallback: string): string {
  return isRecord(value) && typeof value.name === "string"
    ? cleanName(value.name)
    : fallback;
}

function isVersionTwoManifest(
  value: unknown,
): value is Record<string, unknown> & { schemaVersion: 2 } {
  return isRecord(value) && value.schemaVersion === 2;
}

function isVersionThreeManifest(
  value: unknown,
): value is Record<string, unknown> & { schemaVersion: 3 } {
  return isRecord(value) && value.schemaVersion === 3;
}

async function findFiles(
  root: string,
  predicate: (relative: string) => boolean,
): Promise<string[]> {
  const result: string[] = [];
  let fileCount = 0;
  let totalBytes = 0;
  const visit = async (directory: string): Promise<void> => {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isSymbolicLink()) throw new Error("模型包不能包含符号链接");
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) await visit(target);
      else if (entry.isFile()) {
        fileCount += 1;
        if (fileCount > MAX_FILES) throw new Error("模型包文件数量超过限制");
        totalBytes += (await stat(target)).size;
        if (totalBytes > MAX_UNCOMPRESSED_BYTES) {
          throw new Error("模型包超过 512 MB 限制");
        }
        if (predicate(path.relative(root, target))) result.push(target);
      }
    }
  };
  await visit(root);
  return result;
}

function safeRelativePath(value: string): string {
  const normalized = value.replaceAll("\\", "/");
  if (
    !normalized ||
    normalized.startsWith("/") ||
    /^[a-z]+:/i.test(normalized) ||
    normalized.split("/").some((segment) => segment === ".." || segment === "")
  ) {
    throw new Error(`模型包含不安全路径：${value}`);
  }
  return normalized;
}

function safeJoin(root: string, relative: string): string {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(resolvedRoot, relative);
  if (resolved !== resolvedRoot && !resolved.startsWith(`${resolvedRoot}${path.sep}`)) {
    throw new Error("模型路径越界");
  }
  return resolved;
}

async function modelRevision(
  root: string,
  entry: string,
  model: CubismModelJson,
  includeManifest = true,
): Promise<string> {
  const modelDirectory = path.posix.dirname(entry);
  const files = [
    ...(includeManifest ? ["pet.manifest.json"] : []),
    entry,
    ...referencedFiles(model).map((relative) =>
      safeRelativePath(path.posix.join(modelDirectory, relative))
    ),
  ];
  const hash = createHash("sha256");
  for (const relative of [...new Set(files)].sort()) {
    const info = await stat(safeJoin(root, relative));
    hash.update(relative);
    hash.update(":");
    hash.update(String(info.size));
    hash.update(":");
    hash.update(String(info.mtimeMs));
    hash.update("\n");
  }
  return hash.digest("hex").slice(0, 16);
}

function stripModelRevision(relative: string): string {
  const [first, ...rest] = relative.replaceAll("\\", "/").split("/");
  if (first && /^__revision-[a-f0-9]{16}$/.test(first)) {
    return rest.join("/");
  }
  return relative;
}

function modelUrl(id: string, entry: string, revision: string): string {
  const encodedEntry = entry.split("/").map(encodeURIComponent).join("/");
  return `pet-model://${id}/__revision-${revision}/${encodedEntry}`;
}

function cleanName(value: string): string {
  const result = value.trim().replace(/\s+/g, " ").slice(0, 80);
  return result || "Live2D Model";
}

function slug(value: string): string {
  const result = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48);
  return result || "live2d";
}

async function availableId(root: string, candidate: string): Promise<string> {
  let result = candidate;
  let suffix = 2;
  while (true) {
    try {
      await access(path.join(root, result));
      result = `${candidate}-${suffix++}`;
    } catch {
      return result;
    }
  }
}

async function fileExists(file: string): Promise<boolean> {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}

async function writeJsonAtomically(file: string, value: unknown): Promise<void> {
  const temporary = `${file}.${randomUUID()}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(temporary, file);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function finiteBetween(
  value: unknown,
  minimum: number,
  maximum: number,
  fallback: number,
): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(maximum, Math.max(minimum, value))
    : fallback;
}
