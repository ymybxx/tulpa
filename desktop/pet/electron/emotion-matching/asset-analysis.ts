import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import type { ModelRenderBundle } from "../../src/shared/contracts";
import type { EmotionAsset } from "../../src/shared/emotion-matching";
import { normalizeDefaultAppearance } from "../../src/shared/default-appearance";

export interface EmotionSource {
  bundle: ModelRenderBundle;
  fingerprint: string;
  assets: EmotionAsset[];
}

export const digest = (value: unknown): string => createHash("sha256").update(JSON.stringify(value)).digest("hex");

/** Scoped to this on-demand service; unchanged large textures are never re-hashed per batch. */
export class EmotionAssetReader {
  private identities = new Map<string, { stamp: string; hash: string }>();
  constructor(private readonly resolve: (modelId: string, file: string) => string) {}

  async source(bundle: ModelRenderBundle): Promise<EmotionSource> {
    const { model } = bundle;
    const entry = this.resolve(model.id, model.manifest.entry);
    const descriptor = await readJson(entry);
    const refs = record(descriptor.FileReferences) ? descriptor.FileReferences : {};
    const dependencies: string[] = [];
    const defaultAppearance = normalizeDefaultAppearance(model.manifest.defaultAppearance, model.manifest.expressionAssets);
    for (const id of defaultAppearance.expressionAssetIds) {
      const asset = model.manifest.expressionAssets[id];
      if (asset) dependencies.push(asset.file);
    }
    for (const key of ["Moc", "Physics", "Pose", "DisplayInfo", "UserData"]) {
      if (typeof refs[key] === "string") dependencies.push(path.posix.join(path.posix.dirname(model.manifest.entry), refs[key] as string));
    }
    if (Array.isArray(refs.Textures)) for (const file of refs.Textures) {
      if (typeof file === "string") dependencies.push(path.posix.join(path.posix.dirname(model.manifest.entry), file));
    }
    const appearanceFiles: string[] = [];
    const addUrl = (url: string) => {
      const parsed = new URL(url);
      if (parsed.protocol !== "pet-model:" || parsed.hostname !== model.id) throw new Error("分析外观来源不匹配");
      appearanceFiles.push(decodeURIComponent(parsed.pathname).replace(/^\//, ""));
    };
    for (const atlas of bundle.appearance.atlases) addUrl(atlas.url);
    for (const set of bundle.appearance.patchSets) {
      addUrl(set.baseUrl);
      for (const patch of set.patches) { addUrl(patch.imageUrl); addUrl(patch.maskUrl); }
    }
    const hashes: Array<[string, string]> = [];
    for (const file of [...new Set([...dependencies, ...appearanceFiles])].sort()) {
      hashes.push([file, await this.fileHash(this.resolve(model.id, file))]);
    }
    const { Motions: _motions, Expressions: _expressions, ...renderRefs } = refs;
    const normalized = { ...descriptor, FileReferences: renderRefs };
    // Appearance revisions may change without changing pixels. Hash content and ordered recipes.
    const recipes = bundle.appearance.patchSets.map((set) => ({
      index: set.atlasIndex, file: set.atlasFile,
      patches: set.patches.map((patch) => ({ id: patch.textureUnitId, bounds: patch.bounds, image: new URL(patch.imageUrl).pathname, mask: new URL(patch.maskUrl).pathname })),
    }));
    const fingerprint = digest({ normalized, hashes, recipes, defaultAppearance, sampler: "cubism-sample-v3" });
    const assets: EmotionAsset[] = [];
    for (const kind of ["expression", "motion"] as const) {
      const catalog = kind === "motion" ? model.manifest.motionAssets : model.manifest.expressionAssets;
      for (const asset of Object.values(catalog)) {
        const candidate = { id: asset.id, motionAssetIds: kind === "motion" ? [asset.id] : [], expressionAssetIds: kind === "expression" ? [asset.id] : [] };
        try {
          const json = await readJson(this.resolve(model.id, asset.file));
          assets.push({ id: asset.id, name: asset.name, kind, contentHash: asset.contentHash, candidate, ...describeAsset(kind, json) });
        } catch (error) {
          assets.push({ id: asset.id, name: asset.name, kind, contentHash: asset.contentHash, candidate,
            durationMs: 0, timestampsMs: [], parameters: [], error: error instanceof Error ? error.message : "素材无法读取" });
        }
      }
    }
    assets.sort((a, b) => (a.kind === b.kind ? a.durationMs - b.durationMs : a.kind === "expression" ? -1 : 1));
    return { bundle, fingerprint, assets };
  }

  private async fileHash(file: string): Promise<string> {
    const before = await stat(file);
    const stamp = `${before.ino}:${before.size}:${before.mtimeMs}:${before.ctimeMs}`;
    const cached = this.identities.get(file);
    if (cached?.stamp === stamp) return cached.hash;
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(file)) hash.update(chunk);
    const after = await stat(file);
    if (after.size !== before.size || after.mtimeMs !== before.mtimeMs || after.ctimeMs !== before.ctimeMs) throw new Error("模型资源在读取期间改变，请重试");
    const result = hash.digest("hex");
    if (this.identities.size > 10_000) this.identities.clear();
    this.identities.set(file, { stamp, hash: result });
    return result;
  }
}

export function describeAsset(kind: "motion" | "expression", value: Record<string, unknown>): Pick<EmotionAsset, "durationMs" | "timestampsMs" | "parameters"> {
  if (kind === "expression") {
    const fade = typeof value.FadeInTime === "number" && value.FadeInTime >= 0 ? value.FadeInTime : 1;
    if (fade > 118) throw new Error("表情淡入超过两分钟，请手动配置");
    const durationMs = Math.ceil((fade + 0.2) * 1000);
    const parameters = Array.isArray(value.Parameters) ? value.Parameters.filter(record).slice(0, 80)
      .map((p) => `${String(p.Id)}=${String(p.Value)} (${String(p.Blend ?? "Add")})`) : [];
    return { durationMs, timestampsMs: [durationMs], parameters };
  }
  const meta = record(value.Meta) ? value.Meta : {};
  const duration = Number(meta.Duration);
  if (!Number.isFinite(duration) || duration <= 0 || duration > 118) throw new Error("动作时长无效或超过两分钟，请手动配置");
  const changes: Array<{ time: number; amount: number }> = [];
  const parameters: string[] = [];
  if (Array.isArray(value.Curves)) for (const curve of value.Curves.filter(record)) {
    const points = curvePoints(curve.Segments);
    if (!points.length) continue;
    const values = points.map((p) => p.value);
    const range = Math.max(...values) - Math.min(...values);
    parameters.push(`${String(curve.Target)}:${String(curve.Id)} ${Math.min(...values).toFixed(2)}…${Math.max(...values).toFixed(2)}`);
    for (let i = 1; i < points.length; i++) {
      const amount = Math.abs(points[i]!.value - points[i - 1]!.value) / Math.max(range, 0.001);
      changes.push({ time: points[i]!.time, amount });
    }
  }
  const last = Math.max(1 / 60, duration - 1 / 60);
  const times = [Math.min(0.2, last), duration * 0.25, duration * 0.5, duration * 0.75, last];
  for (const change of changes.sort((a, b) => b.amount - a.amount)) {
    if (times.length >= 7) break;
    if (change.time > 0 && change.time <= last && times.every((t) => Math.abs(t - change.time) > duration / 12)) times.push(change.time);
  }
  return { durationMs: Math.ceil(duration * 1000), timestampsMs: [...new Set(times.map((t) => Math.round(t * 1000)))].sort((a, b) => a - b), parameters: parameters.slice(0, 80) };
}

function curvePoints(segments: unknown): Array<{ time: number; value: number }> {
  if (!Array.isArray(segments) || segments.length < 2) return [];
  const points = [{ time: Number(segments[0]), value: Number(segments[1]) }];
  for (let i = 2; i < segments.length;) {
    const type = segments[i++];
    const width = type === 1 ? 6 : type === 0 || type === 2 || type === 3 ? 2 : 0;
    if (!width || i + width > segments.length) break;
    // Cubic control times are valid extra sampling anchors, without misreading segment tags.
    for (let j = 0; j < width; j += 2) points.push({ time: Number(segments[i + j]), value: Number(segments[i + j + 1]) });
    i += width;
  }
  return points.filter((p) => Number.isFinite(p.time) && Number.isFinite(p.value));
}

async function readJson(file: string): Promise<Record<string, unknown>> {
  if ((await stat(file)).size > 8 * 1024 * 1024) throw new Error("素材 JSON 过大");
  const value: unknown = JSON.parse(await readFile(file, "utf8"));
  if (!record(value)) throw new Error("素材 JSON 无效");
  return value;
}
export function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
