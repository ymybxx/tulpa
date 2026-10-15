import type { TextureEditUnit } from "../shared/contracts";

/** Map exact mesh hits to editable UV units; shared atlas regions stay atomic. */
export class TextureLayerSelection {
  private readonly byLayer = new Map<string, TextureEditUnit[]>();
  private readonly byUnit: Map<string, TextureEditUnit>;

  constructor(units: readonly TextureEditUnit[]) {
    this.byUnit = new Map(units.map(unit => [unit.id, unit]));
    for (const unit of units) for (const layer of new Set(unit.drawableIds)) {
      const values = this.byLayer.get(layer) ?? [];
      values.push(unit); this.byLayer.set(layer, values);
    }
  }

  pick(layerIds: readonly string[]): { matched: TextureEditUnit[]; front: TextureEditUnit[] } {
    const matched = new Map<string, TextureEditUnit>();
    let front: TextureEditUnit[] = [];
    for (const id of layerIds) {
      const units = this.byLayer.get(id) ?? [];
      if (!front.length && units.length) front = units;
      for (const unit of units) matched.set(unit.id, unit);
    }
    return { matched: [...matched.values()], front };
  }

  layers(unitIds: Iterable<string>): string[] {
    const ids = new Set<string>();
    for (const id of unitIds) for (const layer of this.byUnit.get(id)?.drawableIds ?? []) ids.add(layer);
    return [...ids];
  }
}
