import type { DefaultAppearanceSettings, ExpressionAsset, ModelLayerGroup, ModelOutfit } from "./contracts";

export function normalizeLayerIds(value: unknown): string[] {
  return Array.isArray(value) ? [...new Set(value.filter((id): id is string =>
    typeof id === "string" && id.length > 0 && id.length <= 256))].sort() : [];
}

function normalizeLayerGroups(value: unknown): ModelLayerGroup[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  return value.flatMap(item => {
    if (!item || typeof item !== "object" || typeof item.id !== "string" || !item.id || item.id.length > 256 || seen.has(item.id)) return [];
    const name = typeof item.name === "string" ? item.name.trim().slice(0, 80) : "";
    const drawableIds = normalizeLayerIds(item.drawableIds);
    if (!name || !drawableIds.length) return [];
    seen.add(item.id);
    return [{ id: item.id, name, drawableIds }];
  });
}

export function normalizeDefaultAppearance(
  value: unknown,
  assets: Record<string, ExpressionAsset>,
): DefaultAppearanceSettings {
  const settings = value && typeof value === "object"
    ? value as { expressionAssetIds?: unknown; parameterValues?: unknown; hiddenDrawableIds?: unknown; layerGroups?: unknown }
    : {};
  const ids = settings.expressionAssetIds;
  const raw = settings.parameterValues;
  const entries = raw && typeof raw === "object" && !Array.isArray(raw)
    ? Object.entries(raw).filter((entry): entry is [string, number] =>
      entry[0].length > 0 && typeof entry[1] === "number" && Number.isFinite(entry[1]))
      .sort(([a], [b]) => a.localeCompare(b))
    : [];
  const hiddenDrawableIds = normalizeLayerIds(settings.hiddenDrawableIds);
  const layerGroups = normalizeLayerGroups(settings.layerGroups);
  return {
    expressionAssetIds: Array.isArray(ids)
      ? [...new Set(ids.filter((id): id is string => typeof id === "string" && Object.hasOwn(assets, id)))]
      : [],
    ...(entries.length ? { parameterValues: Object.fromEntries(entries) } : {}),
    ...(hiddenDrawableIds.length ? { hiddenDrawableIds } : {}),
    ...(layerGroups.length ? { layerGroups } : {}),
  };
}

export const MAX_OUTFITS = 24;
export const MAX_OUTFIT_NAME_LENGTH = 20;

/** Keeps saved 套装 with a name and a unique ID; their toggles follow the model's assets. */
export function normalizeOutfits(value: unknown, assets: Record<string, ExpressionAsset>): ModelOutfit[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const { id, name, appearance } = item as Record<string, unknown>;
    if (typeof id !== "string" || !id || id.length > 64 || seen.has(id)) return [];
    const trimmed = typeof name === "string" ? name.trim().slice(0, MAX_OUTFIT_NAME_LENGTH) : "";
    if (!trimmed) return [];
    seen.add(id);
    return [{ id, name: trimmed, appearance: normalizeDefaultAppearance(appearance, assets) }];
  }).slice(0, MAX_OUTFITS);
}

/** Order-insensitive comparison used to show which 套装 is currently on. */
export function sameAppearance(a: DefaultAppearanceSettings | undefined, b: DefaultAppearanceSettings | undefined): boolean {
  const key = (value: DefaultAppearanceSettings | undefined) => JSON.stringify({
    toggles: [...(value?.expressionAssetIds ?? [])].sort(),
    parameters: Object.entries(value?.parameterValues ?? {}).sort(([x], [y]) => x.localeCompare(y)),
    hidden: [...(value?.hiddenDrawableIds ?? [])].sort(),
  });
  return key(a) === key(b);
}
