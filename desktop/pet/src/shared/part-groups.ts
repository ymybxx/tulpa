import type { ModelEditCapabilityManifest, ModelPartGroup, PetManifest, TextureEditUnit } from "./contracts";

export const MAX_PART_GROUPS = 64;
export const MAX_PART_GROUP_NAME_LENGTH = 20;

/** Colors shared by the group dot, the preview outline and every list that shows the group. */
export const PART_GROUP_COLORS = [
  "#E3A36E", "#5FA0C8", "#8C6BB8", "#D98AA2", "#4F8C76", "#C9A560",
  "#6286BA", "#B07A45", "#9A6FB0", "#5E9E9A", "#C0708A", "#7C8A3E",
  "#E07A5F", "#3D9BE9", "#B58ED8", "#E8B04B", "#6FB8A0", "#A8745A",
  "#D4668F", "#7A86C9",
] as const;

export function partGroupColor(index: number): string {
  return PART_GROUP_COLORS[index % PART_GROUP_COLORS.length]!;
}

/** Names, IDs and drawables are cleaned; a drawable belongs to at most one group. */
export function normalizePartGroups(value: unknown): ModelPartGroup[] {
  if (!Array.isArray(value)) return [];
  const ids = new Set<string>();
  const drawables = new Set<string>();
  const groups: ModelPartGroup[] = [];
  for (const item of value) {
    if (groups.length >= MAX_PART_GROUPS) break;
    if (!item || typeof item !== "object") continue;
    const { id, name, drawableIds } = item as Record<string, unknown>;
    if (typeof id !== "string" || !id || id.length > 80 || ids.has(id)) continue;
    const label = typeof name === "string" ? [...name.trim()].slice(0, MAX_PART_GROUP_NAME_LENGTH).join("") : "";
    if (!label) continue;
    const members = Array.isArray(drawableIds)
      ? drawableIds.filter((drawable): drawable is string =>
        typeof drawable === "string" && drawable.length > 0 && drawable.length <= 256 && !drawables.has(drawable))
      : [];
    for (const drawable of members) drawables.add(drawable);
    ids.add(id);
    groups.push({ id, name: label, drawableIds: [...new Set(members)] });
  }
  return groups;
}

/** The first grouping of a model: every recognized part, leaving unrecognized pieces ungrouped. */
export function automaticPartGroups(capability: ModelEditCapabilityManifest): ModelPartGroup[] {
  return normalizePartGroups(capability.parts
    .filter((part) => part.group !== "unknown")
    .map((part) => ({ id: part.id, name: part.label, drawableIds: part.drawableIds })));
}

/** Saved groups once the user has touched them, otherwise the automatic ones. */
export function effectivePartGroups(manifest: Pick<PetManifest, "partGroups">, capability: ModelEditCapabilityManifest): ModelPartGroup[] {
  return manifest.partGroups ?? automaticPartGroups(capability);
}

export function unitsOfGroup(group: ModelPartGroup, capability: ModelEditCapabilityManifest): TextureEditUnit[] {
  const members = new Set(group.drawableIds);
  return paintableUnits(capability).filter((unit) => unit.drawableIds.some((id) => members.has(id)));
}

/**
 * Every unit a person can see and repaint. Clipping masks are never drawn, so
 * they are left out of every list; they stay in their group untouched.
 */
export function paintableUnits(capability: ModelEditCapabilityManifest): TextureEditUnit[] {
  return capability.textureUnits.filter((unit) => !unit.maskOnly);
}

/** Texture units none of whose drawables belong to any group. */
export function ungroupedUnits(groups: readonly ModelPartGroup[], capability: ModelEditCapabilityManifest): TextureEditUnit[] {
  const grouped = new Set(groups.flatMap((group) => group.drawableIds));
  return paintableUnits(capability).filter((unit) => !unit.drawableIds.some((id) => grouped.has(id)));
}

/** The group owning a drawable, if any. */
export function groupOfDrawable(groups: readonly ModelPartGroup[], drawableId: string): ModelPartGroup | null {
  return groups.find((group) => group.drawableIds.includes(drawableId)) ?? null;
}

/** Moves every drawable of the given units into `targetId`, taking them out of other groups. */
export function moveUnitsToGroup(
  groups: readonly ModelPartGroup[],
  targetId: string,
  units: readonly TextureEditUnit[],
): ModelPartGroup[] {
  const moving = new Set(units.flatMap((unit) => unit.drawableIds));
  return groups.map((group) => {
    const kept = group.drawableIds.filter((id) => !moving.has(id));
    return group.id === targetId ? { ...group, drawableIds: [...kept, ...moving] } : { ...group, drawableIds: kept };
  });
}

/** Moves drawables into `targetId`, taking them out of other groups. */
export function moveDrawablesToGroup(groups: readonly ModelPartGroup[], targetId: string, drawableIds: readonly string[]): ModelPartGroup[] {
  const moving = new Set(drawableIds);
  return groups.map((group) => {
    const kept = group.drawableIds.filter((id) => !moving.has(id));
    return group.id === targetId ? { ...group, drawableIds: [...kept, ...moving] } : { ...group, drawableIds: kept };
  });
}

/** A unit's drawables that no group holds yet; shared textures keep their other owners. */
export function looseDrawables(groups: readonly ModelPartGroup[], unit: TextureEditUnit): string[] {
  const grouped = new Set(groups.flatMap((group) => group.drawableIds));
  return unit.drawableIds.filter((id) => !grouped.has(id));
}

/** Takes the units' drawables out of whichever group holds them. */
export function removeUnitsFromGroups(groups: readonly ModelPartGroup[], units: readonly TextureEditUnit[]): ModelPartGroup[] {
  const removing = new Set(units.flatMap((unit) => unit.drawableIds));
  return groups.map((group) => ({ ...group, drawableIds: group.drawableIds.filter((id) => !removing.has(id)) }));
}

/** Units whose drawable, Part IDs or Part names contain the query (case-insensitive). */
export function unitsMatchingName(capability: ModelEditCapabilityManifest, query: string): TextureEditUnit[] {
  const needle = query.trim().toLocaleLowerCase();
  if (!needle) return [];
  return paintableUnits(capability).filter((unit) =>
    `${unit.searchText ?? ""} ${unit.drawableIds.join(" ")} ${unit.partIds.join(" ")}`.toLocaleLowerCase().includes(needle));
}
