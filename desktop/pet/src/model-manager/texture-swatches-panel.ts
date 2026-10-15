import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import { t } from "../i18n";
import type {
  ModelEditCapabilityManifest,
  PetManifest,
  TextureMaterial,
  TextureMaterialLibrary,
} from "../shared/contracts";
import { effectivePartGroups, groupOfDrawable, partGroupColor, unitsOfGroup } from "../shared/part-groups";
import { RECOLOR_PALETTE, recolorName } from "../shared/recolor-palette";
import { pieceSections } from "./piece-names";

export interface TextureSwatchesOptions {
  root: HTMLElement;
  manifest: PetManifest;
  capability: ModelEditCapabilityManifest;
  library: TextureMaterialLibrary;
  highlight: (drawableIds: readonly string[]) => void;
  setPicking: (handler: ((drawableIds: string[]) => void) | null) => void;
  /** Switches pieces to materials; every change applies at once. */
  select: (assignments: Array<{ unitId: string; materialId: string }>) => Promise<void>;
  remove: (materialIds: string[]) => Promise<void>;
  /** Shows these pieces recolored to `hex` in the preview; null shows what the model wears. */
  previewColor: (unitIds: string[], hex: string | null) => Promise<void>;
  /** Saves the recolor as a version; `wear` also puts it on. */
  saveColor: (unitIds: string[], hex: string, wear: boolean) => Promise<void>;
  /** Drawables on screen right now; null when the preview can't tell. */
  shown: () => Set<string> | null;
  /** Live thumbnails of each group of drawables; [] when the preview can't. */
  capture: (groups: readonly (readonly string[])[], size: number) => HTMLCanvasElement[];
  /** Lights these drawables up and zooms onto them; [] zooms out. */
  focus: (drawableIds: readonly string[]) => void;
  openAi: () => void;
}

export interface Swatch {
  key: string;
  label: string;
  ai: boolean;
  original: boolean;
  /** unit → material of this version, for the units of the part that have it. */
  materials: Map<string, TextureMaterial>;
}

export interface PartSwatches {
  id: string;
  name: string;
  color: string;
  drawableIds: string[];
  unitIds: string[];
  swatches: Swatch[];
}

const ALL = "__all__";

/**
 * 装扮 › 换色: every part's versions — the original, saved AI 换装 results and
 * saved colors — and, for the chosen part, a row of colors that recolor it on
 * the spot, no prompt or AI involved. Clicking the model picks the part under
 * the cursor.
 */
export class TextureSwatchesPanel {
  private part = ALL;
  private busy = false;
  private notice = "";
  /** The color shown in the preview but not saved yet. */
  private trying: string | null = null;
  /** The pieces a new color goes on; null is the whole part. */
  private picked: Set<string> | null = null;
  private pickingPieces = false;
  /** The version waiting for its delete to be confirmed. */
  private deleting: string | null = null;

  constructor(private options: TextureSwatchesOptions) {}

  update(options: Partial<TextureSwatchesOptions>): void {
    this.options = { ...this.options, ...options };
    this.render();
  }

  dispose(): void {
    this.options.setPicking(null);
    this.options.highlight([]);
    this.stopTrying();
  }

  private parts(): PartSwatches[] {
    return partSwatches(this.options.manifest, this.options.capability, this.options.library, true);
  }

  private isActive(swatch: Swatch): boolean {
    const active = this.options.library.activeAssignments;
    return [...swatch.materials].every(([unitId, material]) => active[unitId] === material.id);
  }

  render(): void {
    const { root } = this.options;
    this.options.highlight([]);
    root.replaceChildren();
    const parts = this.parts();
    if (this.part !== ALL && !parts.some((part) => part.id === this.part)) this.choose(ALL);
    const chips = el("div", "le-swatch-parts");
    const chip = (id: string, label: string, drawableIds: readonly string[] = []) => {
      const item = button(label, id === this.part ? "on" : "");
      item.setAttribute("aria-pressed", String(id === this.part));
      item.addEventListener("click", () => { this.choose(id); this.render(); });
      if (drawableIds.length) {
        item.addEventListener("mouseenter", () => this.options.highlight(drawableIds));
        item.addEventListener("mouseleave", () => this.options.highlight([]));
      }
      chips.append(item);
    };
    chip(ALL, "改过的");
    for (const part of parts) chip(part.id, part.name, part.drawableIds);
    root.append(chips);
    if (this.notice) root.append(el("p", "le-ai__muted", this.notice));
    const chosen = parts.find((part) => part.id === this.part);
    if (chosen) root.append(this.partSection(chosen, true));
    else {
      const withVersions = parts.filter((part) => part.swatches.length > 1);
      if (withVersions.length) root.append(bindLocalizedText(el("p", "le-ai__muted", ""), "这里是换过颜色或做过 AI 换装的部位。要换新的颜色，点左边身上的部位或上面的部位名"));
      else {
        const empty = el("div", "le-swatch-empty");
        empty.append(bindLocalizedText(el("p", "", ""), "点左边 TA 身上的部位，或上面选一个部位，就能直接换颜色。想换花纹和材质，用「AI 换装」。"));
        const go = bindLocalizedText(button("", "secondary"), "去 AI 换装");
        go.addEventListener("click", () => this.options.openAi());
        empty.append(go);
        root.append(empty);
      }
      const outfits = this.outfits(parts);
      if (outfits) root.append(outfits);
      for (const part of withVersions) root.append(this.partSection(part, false));
    }
    this.syncPicking();
  }

  /**
   * Whole AI 换装 results: one button per result puts on every part it painted
   * at once; that part's other pieces go back to their original.
   */
  private outfits(parts: PartSwatches[]): HTMLElement | null {
    const results = wholeOutfits(parts);
    if (!results.length) return null;
    const section = el("section", "le-swatch-part");
    const head = el("div", "le-ai__row");
    head.append(bindLocalizedText(el("b", "", ""), "整套造型"));
    section.append(head, bindLocalizedText(el("span", "le-swatch__hint", ""), "每次 AI 换装的结果 · 点一下整套换上"));
    const row = el("div", "le-swatch-parts");
    for (const outfit of results) {
      const worn = outfit.assignments.every(({ unitId, materialId }) => this.options.library.activeAssignments[unitId] === materialId);
      const item = button(outfit.label, worn ? "on" : "");
      item.setAttribute("aria-pressed", String(worn));
      item.disabled = this.busy;
      item.addEventListener("click", () => void this.wear(outfit.assignments));
      row.append(item);
    }
    section.append(row);
    return section;
  }

  private async wear(assignments: Array<{ unitId: string; materialId: string }>): Promise<void> {
    if (this.busy) return;
    const active = this.options.library.activeAssignments;
    const changes = assignments.filter(({ unitId, materialId }) => active[unitId] !== materialId);
    if (!changes.length) return;
    this.busy = true;
    this.render();
    try {
      await this.options.select(changes);
    } finally {
      this.busy = false;
      this.render();
    }
  }

  /** Leaving a part takes an unsaved color off the preview. */
  private choose(id: string): void {
    if (id === this.part) return;
    this.stopTrying();
    this.part = id;
    this.notice = "";
    this.picked = null;
    this.pickingPieces = false;
  }

  /** The chosen part's pieces a new color goes on. */
  private targets(part: PartSwatches): string[] {
    return this.picked ? part.unitIds.filter((id) => this.picked!.has(id)) : part.unitIds;
  }

  private partSection(part: PartSwatches, withColors: boolean): HTMLElement {
    const section = el("section", "le-swatch-part");
    const head = el("div", "le-ai__row");
    const dot = el("span", "le-group__dot");
    dot.style.background = part.color;
    head.append(dot, el("b", "", part.name));
    section.append(head);
    section.addEventListener("mouseenter", () => this.options.highlight(part.drawableIds));
    section.addEventListener("mouseleave", () => this.options.highlight([]));
    if (withColors) {
      section.append(bindLocalizedText(el("span", "le-swatch__hint", ""), (part.swatches.length > 1 ? "已有的样子 · 点一下就换上" : "已有的样子 · 现在只有原版，下面换个颜色就会多一个")));
    }
    const grid = el("div", "le-swatches");
    for (const swatch of part.swatches) grid.append(this.swatch(part, swatch));
    section.append(grid);
    if (withColors) section.append(this.colors(part));
    return section;
  }

  /** The recolor row: pick a color, see it on the model, keep it or not. */
  private colors(part: PartSwatches): HTMLElement {
    const box = el("div", "le-recolor");
    box.append(bindLocalizedText(el("b", "", ""), "换个颜色"), bindLocalizedText(el("span", "le-ai__muted", ""), "按原来的明暗重新上色，线稿和阴影都保留"));
    // Which pieces: the whole part unless the person narrows it.
    const targets = this.targets(part);
    const scope = el("div", "le-ai__row");
    scope.append(bindLocalizedText(el("span", "le-ai__grow", ""), "{{copy}}", () => ({ copy: (part.unitIds.length === 1 ? t("换这 1 块") : (this.picked ? t("只换其中 {{v0}} / {{v1}} 块", { v0: targets.length, v1: part.unitIds.length }) : t("整个{{v0}}一起换（{{v1}} 块）", { v0: part.name, v1: part.unitIds.length }))) })));
    if (part.unitIds.length > 1) {
      const choose = bindLocalizedText(button("", "le-link"), (this.pickingPieces ? "好了" : "只换其中几块"));
      choose.disabled = this.busy;
      choose.addEventListener("click", () => { this.pickingPieces = !this.pickingPieces; this.render(); });
      scope.append(choose);
    }
    box.append(scope);
    if (this.pickingPieces && part.unitIds.length > 1) box.append(this.piecePicker(part));
    const row = el("div", "le-recolor__colors");
    for (const color of RECOLOR_PALETTE) {
      const dot = button("", "le-recolor__dot");
      dot.style.background = color.hex;
      dot.title = color.name;
      dot.setAttribute("aria-label", color.name);
      dot.setAttribute("aria-pressed", String(this.trying === color.hex));
      dot.disabled = this.busy || !targets.length;
      dot.addEventListener("click", () => void this.tryColor(part, color.hex));
      row.append(dot);
    }
    const custom = el("label", "le-recolor__custom");
    const input = document.createElement("input");
    input.type = "color";
    input.value = this.trying && !RECOLOR_PALETTE.some((color) => color.hex === this.trying) ? this.trying : "#F4A7C0";
    input.disabled = this.busy;
    bindLocalizedAttribute(input, "aria-label", "自己选颜色");
    // The picker reports every drag step; only the color it settles on is drawn.
    input.addEventListener("change", () => void this.tryColor(part, input.value.toUpperCase()));
    custom.append(input, bindLocalizedText(el("span", "", ""), "自己选"));
    row.append(custom);
    box.append(row);
    if (this.trying) {
      const trying = this.trying;
      const busy = this.busy;
      const actions = el("div", "le-ai__row");
      actions.append(bindLocalizedText(el("span", "le-ai__muted le-ai__grow", ""), "{{copy}}", () => ({ copy: (busy ? t("正在换…") : t("左边是「{{v0}}」的样子", { v0: recolorName(trying) })) })));
      const cancel = bindLocalizedText(button("", "le-link le-link--muted"), "不换了");
      cancel.disabled = this.busy;
      cancel.addEventListener("click", () => { this.stopTrying(); this.render(); });
      const keep = bindLocalizedText(button("", "secondary"), "只存下来");
      bindLocalizedAttribute(keep, "title", "存成一个版本，先不换上");
      keep.disabled = this.busy;
      keep.addEventListener("click", () => void this.saveColor(part, false));
      const wear = bindLocalizedText(button("", "accent"), "用这个颜色");
      wear.disabled = this.busy;
      wear.addEventListener("click", () => void this.saveColor(part, true));
      actions.append(cancel, keep, wear);
      box.append(actions);
    }
    return box;
  }

  private async tryColor(part: PartSwatches, hex: string): Promise<void> {
    this.trying = hex;
    await this.run(() => this.options.previewColor(this.onScreen(part, this.targets(part)), hex));
  }

  /** The part's pieces as thumbnails, each switched in or out of the recolor. */
  private piecePicker(part: PartSwatches): HTMLElement {
    const units = this.options.capability.textureUnits.filter((unit) => part.unitIds.includes(unit.id));
    const shown = this.options.shown();
    const sections = pieceSections(units, part.name, (unit) => !shown || unit.drawableIds.some((id) => shown.has(id)));
    const pieces = sections.flatMap((section) => section.pieces);
    const live = this.options.capture(pieces.map((piece) => piece.unit.drawableIds), 200);
    const shots = new Map(pieces.map((piece, index) => [piece.unit.id, live[index]]));
    const chosen = new Set(this.targets(part));
    const list = el("div", "le-ai__piece-list");
    for (const section of sections) {
      if (sections.length > 1) {
        const title = el("div", "le-ai__piece-section");
        title.append(el("b", "", section.name));
        list.append(title);
      }
      const grid = el("div", "le-ai__piece-grid");
      for (const { unit, name } of section.pieces) {
        const item = el("label", "le-ai__piece");
        const check = document.createElement("input");
        check.type = "checkbox";
        check.checked = chosen.has(unit.id);
        check.setAttribute("aria-label", `${section.name} · ${name}`);
        check.addEventListener("change", () => void this.setPicked(part, unit.id, check.checked));
        item.classList.toggle("on", check.checked);
        const shot = shots.get(unit.id);
        if (shot) {
          shot.className = "le-locator";
          shot.style.width = shot.style.height = "100px";
        }
        item.append(shot ?? el("span", "le-locator"), check, el("span", "", name));
        item.addEventListener("mouseenter", () => this.options.focus(unit.drawableIds));
        item.addEventListener("mouseleave", () => this.options.focus([]));
        grid.append(item);
      }
      list.append(grid);
    }
    return list;
  }

  private async setPicked(part: PartSwatches, unitId: string, on: boolean): Promise<void> {
    const next = new Set(this.targets(part));
    if (on) next.add(unitId); else next.delete(unitId);
    const before = this.trying ? this.onScreen(part, this.targets(part)) : [];
    this.picked = next.size === part.unitIds.length ? null : next;
    if (!this.trying) {
      this.render();
      return;
    }
    // Show the color on the new set of pieces; a piece taken out goes back first.
    const hex = this.trying;
    await this.run(async () => {
      if (before.length) await this.options.previewColor(before, null);
      const targets = this.onScreen(part, this.targets(part));
      if (targets.length) await this.options.previewColor(targets, hex);
    });
  }

  /**
   * The part's pieces on screen now: the preview only needs those (another
   * hairstyle's strands would just slow it down); saving recolors them all.
   */
  private onScreen(part: PartSwatches, unitIds: readonly string[] = part.unitIds): string[] {
    const shown = this.options.shown();
    if (!shown) return [...unitIds];
    const units = new Map(this.options.capability.textureUnits.map((unit) => [unit.id, unit]));
    const visible = unitIds.filter((id) => units.get(id)?.drawableIds.some((drawable) => shown.has(drawable)));
    return visible.length ? visible : [...unitIds];
  }

  private async saveColor(part: PartSwatches, wear: boolean): Promise<void> {
    const hex = this.trying;
    if (!hex) return;
    await this.run(async () => {
      await this.options.saveColor(this.targets(part), hex, wear);
      this.trying = null;
      this.notice = wear ? `换上了「${recolorName(hex)}」，随时可以点「原版」换回` : `「${recolorName(hex)}」存成了一个版本，想用时点它`;
    });
  }

  private stopTrying(): void {
    if (!this.trying) return;
    this.trying = null;
    const part = this.parts().find((item) => item.id === this.part);
    if (part) void this.options.previewColor(this.onScreen(part, this.targets(part)), null).catch(() => undefined);
  }

  private swatch(part: PartSwatches, swatch: Swatch): HTMLElement {
    const active = this.isActive(swatch) && !this.trying;
    const item = el("div", "le-swatch");
    item.classList.toggle("on", active);
    const pick = button("", "le-swatch__pick");
    pick.setAttribute("aria-pressed", String(active));
    bindLocalizedAttribute(pick, "aria-label", "{{v0}} · {{v1}}{{v2}}{{v3}}", () => ({ v0: part.name, v1: swatch.label, v2: swatch.ai ? t("（AI 做的）") : "", v3: active ? t("，正在用") : "" }));
    pick.title = swatch.label;
    pick.disabled = this.busy;
    const sample = [...swatch.materials.values()].sort((left, right) =>
      right.bounds.width * right.bounds.height - left.bounds.width * left.bounds.height)[0];
    if (sample) {
      const image = document.createElement("img");
      image.src = sample.imageUrl;
      image.alt = "";
      image.loading = "lazy";
      pick.append(image);
    }
    if (swatch.ai) pick.append(el("span", "le-swatch__ai", "AI"));
    if (active) pick.append(el("span", "le-swatch__check", "✓"));
    pick.addEventListener("click", () => {
      if (active) return;
      void this.run(async () => {
        // An unsaved color is only in the preview; take it off before switching.
        if (this.trying) {
          const part = this.parts().find((item) => item.id === this.part);
          this.trying = null;
          if (part) await this.options.previewColor(this.onScreen(part, this.targets(part)), null);
        }
        if (this.isActive(swatch)) return;
        await this.options.select([...swatch.materials].map(([unitId, material]) => ({ unitId, materialId: material.id })));
      });
    });
    item.append(pick, el("span", "le-swatch__label", swatch.label));
    if (!swatch.original) {
      const key = `${part.id}:${swatch.key}`;
      if (this.deleting === key) {
        const confirm = el("span", "le-swatch__confirm");
        const yes = bindLocalizedText(button("", "le-link"), "删掉");
        yes.disabled = this.busy;
        yes.addEventListener("click", () => void this.removeVersion(part, swatch));
        const no = bindLocalizedText(button("", "le-link le-link--muted"), "算了");
        no.addEventListener("click", () => { this.deleting = null; this.render(); });
        confirm.append(yes, no);
        item.append(confirm);
      } else {
        const remove = bindLocalizedText(button("", "le-link le-link--muted le-swatch__delete"), "删除");
        bindLocalizedAttribute(remove, "aria-label", "删除 {{v0}} 的「{{v1}}」", { v0: part.name, v1: swatch.label });
        remove.disabled = this.busy;
        remove.addEventListener("click", () => { this.deleting = key; this.render(); });
        item.append(remove);
      }
    }
    return item;
  }

  /** Deletes a saved version; one being worn goes back to the original first. */
  private async removeVersion(part: PartSwatches, swatch: Swatch): Promise<void> {
    this.deleting = null;
    await this.run(async () => {
      if (this.isActive(swatch)) {
        const original = part.swatches.find((item) => item.original);
        if (original) {
          await this.options.select([...swatch.materials.keys()].flatMap((unitId) => {
            const material = original.materials.get(unitId);
            return material ? [{ unitId, materialId: material.id }] : [];
          }));
        }
      }
      await this.options.remove([...swatch.materials.values()].map((material) => material.id));
    });
  }

  private async run(action: () => Promise<void>): Promise<void> {
    this.busy = true;
    this.notice = "";
    this.render();
    try {
      await action();
    } catch (error) {
      this.notice = error instanceof Error ? error.message : String(error);
    } finally {
      this.busy = false;
    }
    this.render();
  }

  private syncPicking(): void {
    this.options.setPicking((drawableIds) => {
      const groups = effectivePartGroups(this.options.manifest, this.options.capability);
      for (const drawable of drawableIds) {
        const owner = groupOfDrawable(groups, drawable);
        if (!owner || !unitsOfGroup(owner, this.options.capability).length) continue;
        this.choose(owner.id);
        this.render();
        return;
      }
    });
  }
}

/**
 * Each part's versions: the original and every saved AI result or color. With
 * `all`, parts that only have their original are listed too.
 */
export function partSwatches(
  manifest: PetManifest,
  capability: ModelEditCapabilityManifest,
  library: TextureMaterialLibrary,
  all = false,
): PartSwatches[] {
  const byUnit = new Map<string, TextureMaterial[]>();
  for (const material of library.materials) {
    const list = byUnit.get(material.textureUnitId) ?? [];
    list.push(material);
    byUnit.set(material.textureUnitId, list);
  }
  return effectivePartGroups(manifest, capability).flatMap((group, index) => {
    const units = unitsOfGroup(group, capability);
    const varied = units.filter((unit) => (byUnit.get(unit.id)?.length ?? 0) > 1);
    if (!units.length || (!all && !varied.length)) return [];
    const swatches = new Map<string, Swatch>();
    for (const unit of all ? units : varied) {
      for (const material of byUnit.get(unit.id) ?? []) {
        const original = material.original ?? material.createdAt === capability.generatedAt;
        const key = original ? "original" : `${material.createdAt}|${material.prompt}`;
        const swatch = swatches.get(key) ?? {
          key,
          label: original ? "原版" : swatchLabel(material),
          ai: !original && material.origin !== "recolor" && Boolean(material.prompt),
          original,
          materials: new Map(),
        };
        swatch.materials.set(unit.id, material);
        swatches.set(key, swatch);
      }
    }
    const ordered = [...swatches.values()].sort((left, right) =>
      Number(right.original) - Number(left.original) || right.key.localeCompare(left.key));
    return [{
      id: group.id, name: group.name, color: partGroupColor(index), drawableIds: group.drawableIds,
      unitIds: units.map((unit) => unit.id), swatches: ordered,
    }];
  });
}

/**
 * Every saved AI result, newest first, as the pieces to put on: in each part it
 * painted, its version where it has one and the original elsewhere.
 */
export function wholeOutfits(parts: readonly PartSwatches[]): Array<{ key: string; label: string; assignments: Array<{ unitId: string; materialId: string }> }> {
  const outfits = new Map<string, { key: string; label: string; assignments: Array<{ unitId: string; materialId: string }> }>();
  for (const part of parts) {
    const original = part.swatches.find((swatch) => swatch.original);
    for (const swatch of part.swatches) {
      if (!swatch.ai) continue;
      const outfit = outfits.get(swatch.key) ?? { key: swatch.key, label: outfitLabel(swatch), assignments: [] };
      for (const unitId of part.unitIds) {
        const material = swatch.materials.get(unitId) ?? original?.materials.get(unitId);
        if (material) outfit.assignments.push({ unitId, materialId: material.id });
      }
      outfits.set(swatch.key, outfit);
    }
  }
  return [...outfits.values()].sort((left, right) => right.key.localeCompare(left.key));
}

/** A result is named by what the person asked for: its leading name when they gave one ("赛博朋克风：…"). */
function outfitLabel(swatch: Swatch): string {
  const prompt = [...swatch.materials.values()][0]?.prompt.replace(/^用户要求：/, "").trim() ?? "";
  const lead = prompt.split(/[：:，,。\n]/)[0]?.trim() ?? "";
  return lead && [...lead].length <= 12 ? lead : swatch.label;
}

function swatchLabel(material: TextureMaterial): string {
  if (material.origin === "recolor") return recolorName(material.prompt);
  const prompt = material.prompt.replace(/\s+/g, " ").trim();
  // Older saves stored the compiled model prompt, which reads as noise.
  if (!prompt || prompt.startsWith("任务") || [...prompt].length > 60) {
    const date = new Date(material.createdAt);
    return Number.isNaN(date.getTime()) ? "AI 版本"
      : `AI 版本 · ${date.getMonth() + 1}/${date.getDate()} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
  }
  return [...prompt].length > 10 ? `${[...prompt].slice(0, 10).join("")}…` : prompt;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(text: string, className: string): HTMLButtonElement {
  const node = el("button", className, text);
  node.type = "button";
  return node;
}
