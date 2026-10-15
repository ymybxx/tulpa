import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import { t } from "../i18n";
import type {
  ModelEditCapabilityManifest,
  ModelPartGroup,
  PetManifest,
  TextureAtlasDelta,
  TextureEditUnit,
} from "../shared/contracts";
import {
  DEFAULT_CHARACTER_BUDGET_USD,
  DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
  drawnClauseIds,
  keptRegionKeys,
  TEXTURE_VISUAL_ATTRIBUTE_CATEGORIES,
  type TextureClauseAction,
  soloTextureEditScale,
  textureEditPlanRegions,
  type CharacterGenerationJob,
  type CharacterGenerationProgress,
  type TextureEditPlanned,
  type TexturePromptOptimizationResult,
  type TextureReferencePromptId,
  type TextureReferenceSelection,
  type TextureVisualAttributeCategory,
  type TextureVisualEditSpec,
} from "../shared/character-generation";
import { pieceSections } from "./piece-names";
import { effectivePartGroups, groupOfDrawable, paintableUnits, partGroupColor, ungroupedUnits, unitsOfGroup } from "../shared/part-groups";
import {
  compileTextureVisualPrompt,
  splitTexturePromptClauses,
  type TextureVisualEditSpecInput,
} from "../../electron/character-generation/visual-edit-spec";
import { textureReferenceDeliveryState } from "./character-wizard-state";
import { openSettings } from "../ui/remembered";
import { PieceLocator } from "./piece-locator";

/** What the user has set up in AI 换装, kept per model while they look elsewhere. */
export interface AiOutfitDraft {
  selection: Array<[string, string[]]>;
  prompt: string;
  references: TextureReferenceSelection[];
  optimization: TexturePromptOptimizationResult | null;
  step: "pick" | "confirm";
}

export interface AiOutfitPanelOptions {
  root: HTMLElement;
  /** The preview stage, where the "several parts here" chooser and 看原版 appear. */
  stage: HTMLElement;
  manifest: PetManifest;
  capability: ModelEditCapabilityManifest;
  imageUrl: (relativePath: string) => string;
  highlight: (drawableIds: readonly string[]) => void;
  /** Dims the rest and zooms the live preview onto these drawables; [] zooms out. */
  focus: (drawableIds: readonly string[]) => void;
  /** Live thumbnails of each group, as the preview shows it now; [] when it can't. */
  capture: (groups: readonly (readonly string[])[], size: number) => HTMLCanvasElement[];
  /** Drawables on screen right now; null when the preview can't tell. */
  shown: () => Set<string> | null;
  /** Another 装扮 pane: switches for removing a part, 换色 for a plain color change. */
  openPane: (pane: "toggles" | "textures") => void;
  setPicking: (handler: ((drawableIds: string[], point: { x: number; y: number }) => void) | null) => void;
  previewDelta: (delta: TextureAtlasDelta) => Promise<void> | void;
  /** Saved results are new 换色 materials; the appearance must be reloaded. */
  saved: (count: number, applied: boolean) => Promise<void>;
  /** Where to start from when the pane was left mid-way. */
  draft?: AiOutfitDraft | null;
}

type Step = "pick" | "pieces" | "understanding" | "confirm" | "working" | "review" | "failed";
type PlanState =
  | { status: "empty" }
  | { status: "planning" }
  | { status: "ok"; plan: TextureEditPlanned }
  | { status: "too-big"; message: string }
  | { status: "error"; message: string };

const LOOSE = "__loose__";
const IDEAS = ["樱粉色", "星空渐变", "黑色皮革", "牛仔布", "和风花纹", "奶油白"];
const CATEGORY_LABELS: Record<TextureVisualAttributeCategory, string> = {
  palette: "配色", rendering: "画风", shading: "明暗", linework: "线条", material: "材质", motif: "图案", tone: "感觉",
};
const ACTION_LABELS: Record<TextureClauseAction, string> = {
  repaint: "会画", recolor: "只换颜色", toggle: "用开关", unsupported: "做不到", unselected: "没选上",
};
const ACTIVE_PHASES = new Set(["draft", "decomposing", "aligning", "base-preview", "completing", "validating"]);

interface Group {
  id: string;
  name: string;
  color: string;
  units: TextureEditUnit[];
  drawableIds: string[];
}

/**
 * 装扮 › AI 换装: pick parts on the model, say the look, confirm how the AI
 * understood it, then try the result on and keep it piece by piece. Kept
 * results become 换色 materials.
 */
export class AiOutfitPanel {
  private step: Step = "pick";
  private readonly selection = new Map<string, Set<string>>();
  private piecesGroup: string | null = null;
  /** Whether the pieces the character is not showing right now are listed. */
  private hiddenPiecesOpen = false;
  /** The part whose pieces are laid out one by one on the try-on step. */
  private reviewPieces: string | null = null;
  private prompt = "";
  private references: TextureReferenceSelection[] = [];
  private plan: PlanState = { status: "empty" };
  private planVersion = 0;
  private planTimer = 0;
  private optimization: TexturePromptOptimizationResult | null = null;
  private editingGroup: string | null = null;
  private job: CharacterGenerationJob | null = null;
  private skippedFor: TexturePromptOptimizationResult | null = null;
  private skipped = new Map<string, { keep: string; label: string }>();
  private progressDetail = "";
  private error = "";
  /** The last 开始设计 could not get the AI's reading; drawing as written is offered instead. */
  private understandFailed = false;
  private stack: { groups: Group[]; x: number; y: number } | null = null;
  private stackLayer: HTMLElement | null = null;
  private compareButton: HTMLButtonElement | null = null;
  private busy = false;
  private locator: PieceLocator;
  /** Whether an image model has a key; null until known. */
  private imageReady: boolean | null = null;
  private readonly unsubscribe: () => void;
  /** Until the job list is back, a drawing or finished job may still be picked up. */
  private resuming = true;

  constructor(private options: AiOutfitPanelOptions) {
    this.locator = new PieceLocator(options.capability.modelId, options.imageUrl);
    this.unsubscribe = window.desktopPet.onCharacterProgress((progress) => void this.onProgress(progress));
    this.restore(options.draft ?? null);
    void this.resume();
    void this.checkImageModel();
  }

  private async checkImageModel(): Promise<void> {
    const configuration = await window.desktopPet.getConfiguration().catch(() => null);
    if (!configuration) return;
    this.imageReady = configuration.imageEditing.openaiCompatibleBaseUrl
      ? configuration.secrets.openaiCompatibleImageApiKey
      : configuration.secrets.falApiKey;
    this.render();
  }

  private imageModelStrip(): HTMLElement | null {
    if (this.imageReady !== false) return null;
    const strip = el("div", "le-strip le-strip--warn");
    const go = bindLocalizedText(button("", "secondary"), "去设置");
    go.addEventListener("click", () => openSettings("ai"));
    strip.append(bindLocalizedText(el("span", "", ""), "还没设置图片模型，画不了。在 设置 › AI 服务 里填好图片接口的地址和 Key"), go);
    return strip;
  }

  update(options: Partial<AiOutfitPanelOptions>): void {
    const modelChanged = options.capability && options.capability.modelId !== this.options.capability.modelId;
    this.options = { ...this.options, ...options };
    if (modelChanged) {
      this.locator = new PieceLocator(this.options.capability.modelId, this.options.imageUrl);
      this.selection.clear();
      this.step = "pick";
      this.plan = { status: "empty" };
      this.job = null;
      void this.resume();
    }
    this.render();
  }

  /** The user's unfinished setup, to hand back when the pane opens again. */
  snapshot(): AiOutfitDraft | null {
    if (!this.selection.size && !this.prompt.trim() && !this.references.length) return null;
    return {
      selection: [...this.selection].map(([groupId, ids]) => [groupId, [...ids]]),
      prompt: this.prompt,
      references: this.references.map((reference) => ({ ...reference })),
      optimization: this.optimization,
      step: this.step === "confirm" && this.optimization ? "confirm" : "pick",
    };
  }

  private restore(draft: AiOutfitDraft | null): void {
    if (!draft) return;
    const units = new Set(paintableUnits(this.options.capability).map((unit) => unit.id));
    for (const [groupId, ids] of draft.selection) {
      const kept = ids.filter((id) => units.has(id));
      if (kept.length) this.selection.set(groupId, new Set(kept));
    }
    this.prompt = draft.prompt;
    this.references = draft.references;
    this.optimization = draft.optimization;
    this.schedulePlan();
    // The understanding belongs to the plan it was made for; confirm again only once that plan is back.
    if (draft.step === "confirm" && this.optimization) this.step = "confirm";
  }

  dispose(): void {
    this.unsubscribe();
    window.clearTimeout(this.planTimer);
    this.options.setPicking(null);
    this.options.highlight([]);
    this.stackLayer?.remove();
    this.compareButton?.remove();
    if (this.job && this.step === "review") void this.showResult(false);
  }

  // ---------- groups and selection ----------

  private groups(): Group[] {
    const { capability, manifest } = this.options;
    const groups = effectivePartGroups(manifest, capability);
    const list: Group[] = groups.map((group: ModelPartGroup, index) => ({
      id: group.id, name: group.name, color: partGroupColor(index),
      units: unitsOfGroup(group, capability), drawableIds: group.drawableIds,
    }));
    const loose = ungroupedUnits(groups, capability);
    if (loose.length) {
      list.push({ id: LOOSE, name: "其他", color: "#9A9AA2", units: loose, drawableIds: loose.flatMap((unit) => unit.drawableIds) });
    }
    return list;
  }

  private groupsAt(drawableIds: readonly string[]): Group[] {
    const groups = this.groups();
    const partGroups = effectivePartGroups(this.options.manifest, this.options.capability);
    const found: Group[] = [];
    for (const drawable of drawableIds) {
      const owner = groupOfDrawable(partGroups, drawable);
      const group = owner
        ? groups.find((item) => item.id === owner.id)
        : groups.find((item) => item.id === LOOSE && item.drawableIds.includes(drawable));
      if (group && !found.includes(group) && group.units.length) found.push(group);
      if (found.length >= 4) break;
    }
    return found;
  }

  private selectedUnitIds(): string[] {
    return [...new Set([...this.selection.values()].flatMap((ids) => [...ids]))];
  }

  private toggleGroup(group: Group): void {
    if (this.selection.has(group.id)) this.selection.delete(group.id);
    else this.selection.set(group.id, new Set(this.onScreen(group).map((unit) => unit.id)));
    this.selectionChanged();
  }

  /**
   * The group's pieces the character shows right now. Another hairstyle's
   * strands would only crowd the canvas; they can still be ticked by hand.
   */
  private onScreen(group: Group): TextureEditUnit[] {
    const shown = this.options.shown();
    const visible = shown ? group.units.filter((unit) => unit.drawableIds.some((id) => shown.has(id))) : [];
    return visible.length ? visible : group.units;
  }

  private selectionChanged(): void {
    this.optimization = null;
    this.schedulePlan();
    this.render();
  }

  private schedulePlan(): void {
    window.clearTimeout(this.planTimer);
    const ids = this.selectedUnitIds();
    const version = ++this.planVersion;
    if (!ids.length) {
      this.plan = { status: "empty" };
      return;
    }
    const oversize = this.oversizePieces(ids);
    if (oversize) {
      this.plan = { status: "too-big", message: oversize };
      return;
    }
    this.plan = { status: "planning" };
    this.planTimer = window.setTimeout(async () => {
      try {
        const result = await window.desktopPet.planCharacterTextureBatch(
          this.options.capability.templateId, ids, DEFAULT_TEXTURE_EDIT_IMAGE_MODEL);
        if (version !== this.planVersion) return;
        this.plan = result.fits ? { status: "ok", plan: result } : { status: "too-big", message: result.reason };
      } catch (error) {
        if (version !== this.planVersion) return;
        this.plan = { status: "error", message: errorText(error) };
      }
      this.render();
    }, 250);
  }

  /** A piece bigger than the largest canvas can't be drawn at all; say which. */
  private oversizePieces(ids: readonly string[]): string | null {
    const units = new Map(this.options.capability.textureUnits.map((unit) => [unit.id, unit]));
    for (const id of ids) {
      const unit = units.get(id);
      if (!unit) continue;
      if (soloTextureEditScale(unit.bounds.width, unit.bounds.height, DEFAULT_TEXTURE_EDIT_IMAGE_MODEL) === null) {
        const group = this.groups().find((item) => this.selection.get(item.id)?.has(id));
        return `「${group?.name ?? "其他"}」里有一块太大（${unit.bounds.width}×${unit.bounds.height}），缩小了也画不了。点这一组的「只改几块」把它去掉`;
      }
    }
    return null;
  }

  // ---------- rendering ----------

  /** Selections follow the current groups: pieces moved elsewhere drop out. */
  private pruneSelection(): void {
    const groups = new Map(this.groups().map((group) => [group.id, new Set(group.units.map((unit) => unit.id))]));
    for (const [groupId, ids] of [...this.selection]) {
      const members = groups.get(groupId);
      const kept = members ? new Set([...ids].filter((id) => members.has(id))) : new Set<string>();
      if (!kept.size) this.selection.delete(groupId);
      else if (kept.size !== ids.size) this.selection.set(groupId, kept);
    }
  }

  render(): void {
    const { root } = this.options;
    // Whatever was hovered may be gone after this redraw, and with it its mouseleave.
    this.options.focus([]);
    if (this.resuming) {
      // Not the first step: it would flash before a finished job comes back.
      root.replaceChildren(bindLocalizedText(el("div", "le-strip le-strip--info", ""), "正在接着上次的地方…"));
      return;
    }
    if (this.step === "pick" || this.step === "pieces") this.pruneSelection();
    root.replaceChildren();
    const head = el("div", "le-pane__head");
    head.append(bindLocalizedText(el("h3", "", ""), "AI 换装"), bindLocalizedText(el("span", "", ""), "只重画颜色和花纹，形状、动作都不变"));
    const steps = el("ol", "le-steps");
    const current = { pick: 0, pieces: 0, understanding: 1, confirm: 1, working: 2, failed: 2, review: 3 }[this.step];
    ["选部位和样子", "确认", "画", "试穿"].forEach((label, index) => {
      const item = bindLocalizedText(el("li", index === current ? "on" : ""), "{{v0}} {{v1}}", () => ({ v0: index + 1, v1: t(label) }));
      if (index === current) item.setAttribute("aria-current", "step");
      steps.append(item);
    });
    head.append(steps);
    root.append(head);
    if (this.step === "pick" || this.step === "pieces") this.renderPick(root);
    else if (this.step === "understanding") root.append(bindLocalizedText(el("div", "le-strip le-strip--info", ""), "AI 正在看要改的部位，理解你的要求…"));
    else if (this.step === "confirm") this.renderConfirm(root);
    else if (this.step === "working" || this.step === "failed") this.renderWorking(root);
    else if (this.step === "review") this.renderReview(root);
    this.syncPicking();
    this.renderStack();
    this.renderCompare();
  }

  private renderPick(root: HTMLElement): void {
    const groups = this.groups();
    const missing = this.imageModelStrip();
    if (missing) root.append(missing);
    const top = el("div", "le-ai__row");
    top.append(bindLocalizedText(el("b", "", ""), "改哪里"), bindLocalizedText(el("span", "le-ai__muted", ""), "在左边点 TA 身上的部位，再点一下去掉"));
    const add = document.createElement("select");
    add.className = "le-ai__add";
    bindLocalizedAttribute(add, "aria-label", "加部位");
    add.append(new Option("+ 加部位", ""));
    for (const group of groups) if (!this.selection.has(group.id)) add.append(new Option(group.name, group.id));
    add.addEventListener("change", () => {
      const group = groups.find((item) => item.id === add.value);
      if (group) this.toggleGroup(group);
    });
    top.append(add);
    root.append(top);

    if (this.step === "pieces" && this.piecesGroup) {
      const group = groups.find((item) => item.id === this.piecesGroup);
      if (group) root.append(this.piecesEditor(group));
    } else {
      const list = el("div", "le-ai__list");
      const selected = groups.filter((group) => this.selection.has(group.id));
      if (!selected.length) list.append(bindLocalizedText(el("p", "le-ai__empty", ""), "还没选。在左边点头发、衣服这些地方试试"));
      for (const group of selected) {
        const chosen = this.selection.get(group.id)!;
        const row = el("div", "le-ai__item");
        const dot = el("span", "le-group__dot");
        dot.style.background = group.color;
        const all = chosen.size === group.units.length;
        const visible = this.onScreen(group);
        const justVisible = !all && chosen.size === visible.length && visible.every((unit) => chosen.has(unit.id));
        const scope = all ? `全部 ${group.units.length} 块`
          : justVisible ? `看得到的 ${chosen.size} 块` : `只改 ${chosen.size} / ${group.units.length} 块`;
        const pieces = bindLocalizedText(button("", "le-link"), (all || justVisible ? "只改几块" : "改"));
        pieces.addEventListener("click", () => { this.piecesGroup = group.id; this.step = "pieces"; this.render(); });
        const remove = button("×", "le-group__delete");
        bindLocalizedAttribute(remove, "aria-label", "不改{{v0}}", { v0: group.name });
        remove.addEventListener("click", () => this.toggleGroup(group));
        row.append(dot, el("b", "", group.name), el("span", "le-ai__muted le-ai__grow", scope), pieces, remove);
        // Pointing at a row only lights its pieces up; zooming in is for the piece grid.
        row.addEventListener("mouseenter", () => this.options.highlight(group.units.filter((unit) => chosen.has(unit.id)).flatMap((unit) => unit.drawableIds)));
        row.addEventListener("mouseleave", () => this.options.highlight([]));
        list.append(row);
      }
      root.append(list);
    }

    const label = bindLocalizedText(el("label", "le-ai__label", ""), "想要的样子");
    const textarea = document.createElement("textarea");
    textarea.rows = 2;
    textarea.id = "ai-outfit-prompt";
    label.htmlFor = textarea.id;
    bindLocalizedAttribute(textarea, "placeholder", "比如：头发换成樱粉色，外套换成白色带金边");
    textarea.value = this.prompt;
    const start = bindLocalizedText(button("", "accent"), "开始设计");
    const syncStart = () => { start.disabled = !(this.plan.status === "ok" && this.prompt.trim()); };
    textarea.addEventListener("input", () => { this.prompt = textarea.value; this.optimization = null; syncStart(); });
    const ideas = el("div", "le-ai__ideas");
    for (const idea of IDEAS) {
      const chip = bindLocalizedText(button("", "le-ai__idea"), idea);
      chip.addEventListener("click", () => {
        this.prompt = this.prompt.trim() ? `${this.prompt.trim()}，${idea}` : idea;
        this.optimization = null;
        this.render();
      });
      ideas.append(chip);
    }
    root.append(label, textarea, ideas);

    const refs = el("div", "le-ai__row");
    refs.append(bindLocalizedText(el("b", "", ""), "参考图"), bindLocalizedText(el("span", "le-ai__muted le-ai__grow", ""), "可选"));
    for (const reference of this.references) {
      const tile = el("span", "le-ai__ref");
      tile.title = reference.name;
      if (reference.previewUrl) {
        const image = document.createElement("img");
        image.src = reference.previewUrl;
        image.alt = reference.name;
        tile.append(image);
      }
      const remove = button("×", "le-piece__remove");
      bindLocalizedAttribute(remove, "aria-label", "去掉参考图 {{v0}}", { v0: reference.name });
      remove.addEventListener("click", () => {
        this.references = this.references.filter((item) => item !== reference);
        this.optimization = null;
        this.render();
      });
      tile.append(remove);
      refs.append(tile);
    }
    if (this.references.length < 5) {
      const pick = button("+", "le-ai__ref le-ai__ref--add");
      bindLocalizedAttribute(pick, "aria-label", "加一张参考图");
      pick.addEventListener("click", async () => {
        const chosen = await window.desktopPet.pickCharacterReference().catch(() => null);
        if (!chosen) return;
        this.references.push({
          id: crypto.randomUUID(), source: "upload", name: chosen.name, description: "", path: chosen.path,
          textureUnitId: null, materialId: null, previewUrl: chosen.previewUrl,
        });
        this.optimization = null;
        this.render();
      });
      refs.append(pick);
    }
    root.append(refs);

    const footer = el("div", "le-ai__footer");
    const note = el("span", "le-ai__muted le-ai__grow");
    const plan = this.plan;
    if (plan.status === "empty") bindLocalizedText(note, "先在左边点要改的部位");
    else if (plan.status === "planning") bindLocalizedText(note, "正在算要画几次…");
    else if (plan.status === "ok") {
      note.append(bindLocalizedText(el("span"), "{{copy}}", () => ({ copy: costLine(plan.plan) })), document.createElement("br"), bindLocalizedText(el("span"), plan.plan.canvases.length > 1
        ? "一张画布放不下，分几次画；后面几张会参照第一张的颜色，也可能有轻微差别"
        : "下一步先看 AI 怎么理解，确认了才开始画"));
    } else {
      note.classList.add("le-ai__warn");
      bindLocalizedText(note, "{{copy}}", () => ({ copy: (plan.status === "too-big" ? plan.message : t("没算出来：{{v0}}", { v0: plan.message })) }));
    }
    if (this.error) {
      const error = el("p", "le-ai__warn", this.error);
      root.append(error);
    }
    syncStart();
    start.addEventListener("click", () => void this.understand());
    footer.append(note);
    // When understanding failed, the error offers drawing as written; the way to do it is right here.
    if (this.understandFailed && plan.status === "ok") {
      const raw = bindLocalizedText(button("", "le-link le-link--muted"), "不用理解，按原话画");
      raw.disabled = this.busy || this.imageReady === false || !this.prompt.trim();
      raw.addEventListener("click", () => void this.startJob(false));
      footer.append(raw);
    }
    footer.append(start);
    root.append(footer);
  }

  private piecesEditor(group: Group): HTMLElement {
    const box = el("div", "le-ai__pieces");
    const head = el("div", "le-ai__row");
    const dot = el("span", "le-group__dot");
    dot.style.background = group.color;
    const done = bindLocalizedText(button("", "secondary"), "好了");
    done.addEventListener("click", () => { this.step = "pick"; this.piecesGroup = null; this.render(); });
    head.append(dot, bindLocalizedText(el("b", "", ""), "{{v0}}里改哪几块", { v0: group.name }), bindLocalizedText(el("span", "le-ai__muted le-ai__grow", ""), "勾上的会重画，也可以在左边点"), done);
    box.append(head);
    const chosen = this.selection.get(group.id) ?? new Set<string>();
    const shown = this.options.shown();
    const sections = pieceSections(group.units, group.name,
      (unit) => !shown || unit.drawableIds.some((id) => shown.has(id)));
    const hiddenOpen = this.hiddenPiecesOpen || sections.every((section) => section.hidden);
    const pieces = sections.filter((section) => !section.hidden || hiddenOpen).flatMap((section) => section.pieces);
    const live = this.options.capture(pieces.map((piece) => piece.unit.drawableIds), 240);
    const shots = new Map(pieces.map((piece, index) => [piece.unit.id, live[index]]));
    const setChosen = (ids: readonly string[], on: boolean) => {
      const set = this.selection.get(group.id) ?? new Set<string>();
      for (const id of ids) if (on) set.add(id); else set.delete(id);
      if (set.size) this.selection.set(group.id, set); else this.selection.delete(group.id);
      this.selectionChanged();
    };
    const list = el("div", "le-ai__piece-list");
    for (const section of sections) {
      const ids = section.pieces.map((piece) => piece.unit.id);
      if (sections.length > 1) {
        const title = el("label", "le-ai__piece-section");
        title.classList.toggle("hidden", Boolean(section.hidden));
        const all = document.createElement("input");
        all.type = "checkbox";
        const count = ids.filter((id) => chosen.has(id)).length;
        all.checked = count === ids.length;
        all.indeterminate = count > 0 && count < ids.length;
        all.addEventListener("change", () => setChosen(ids, all.checked));
        title.append(all, el("b", "", section.name), bindLocalizedText(el("span", "le-ai__muted", ""), "{{v0}} 块", { v0: ids.length }));
        if (section.hidden) {
          title.append(bindLocalizedText(el("span", "le-ai__muted le-ai__grow", ""), "换发型、表情或做动作时才出现"));
          const toggle = bindLocalizedText(button("", "le-link"), (hiddenOpen ? "收起" : "展开看看"));
          toggle.addEventListener("click", (event) => {
            event.preventDefault();
            this.hiddenPiecesOpen = !hiddenOpen;
            this.render();
          });
          title.append(toggle);
        }
        const drawables = section.pieces.flatMap((piece) => piece.unit.drawableIds);
        title.addEventListener("mouseenter", () => this.options.highlight(drawables));
        title.addEventListener("mouseleave", () => this.options.highlight([]));
        list.append(title);
      }
      if (section.hidden && !hiddenOpen) continue;
      const grid = el("div", "le-ai__piece-grid");
      for (const { unit, name } of section.pieces) {
        const item = el("label", "le-ai__piece");
        const check = document.createElement("input");
        check.type = "checkbox";
        check.checked = chosen.has(unit.id);
        check.setAttribute("aria-label", `${section.name} · ${name}`);
        check.addEventListener("change", () => setChosen([unit.id], check.checked));
        item.classList.toggle("on", check.checked);
        const shot = shots.get(unit.id);
        if (shot) {
          shot.className = "le-locator";
          shot.style.width = shot.style.height = "120px";
        }
        item.append(shot ?? this.locator.thumbnail(unit, 120), check, el("span", "", name));
        if (this.oversizePieces([unit.id])) {
          item.classList.add("warn");
          item.append(bindLocalizedText(el("small", "le-ai__warn", ""), "太大，画不了"));
        }
        item.addEventListener("mouseenter", () => this.options.focus(unit.drawableIds));
        item.addEventListener("mouseleave", () => this.options.focus([]));
        grid.append(item);
      }
      list.append(grid);
    }
    box.append(list);
    box.append(bindLocalizedText(el("span", "le-ai__muted", ""), "常一起改的几块，可以在「部位」里单独建一个分组"));
    return box;
  }

  private renderConfirm(root: HTMLElement): void {
    const optimization = this.optimization;
    if (optimization && this.plan.status === "planning") {
      root.append(bindLocalizedText(el("div", "le-strip le-strip--info", ""), "正在接着上次的地方…"));
      return;
    }
    const plan = this.plan.status === "ok" ? this.plan.plan : null;
    if (!optimization || !plan) {
      this.step = "pick";
      this.render();
      return;
    }
    root.append(bindLocalizedText(el("div", "le-strip le-strip--info", ""), "AI 看过要改的部位，这样理解你的要求。哪条不对就点「改」，没问题就开始画"));
    const original = el("div", "le-ai__row");
    const back = bindLocalizedText(button("", "le-link"), "回去改原话");
    back.addEventListener("click", () => { this.step = "pick"; this.render(); });
    original.append(bindLocalizedText(el("span", "le-ai__muted le-ai__grow", ""), "你写的：「{{v0}}」", { v0: optimization.originalPrompt }), back);
    root.append(original);
    const actions = this.actionList(optimization.spec);
    if (actions) root.append(actions);
    const keptBox = this.keptList(optimization, plan);
    if (keptBox) root.append(keptBox);
    const drawn = optimization.spec.regions.some((region) => !region.keep)
      ? drawnClauseIds(optimization.spec, splitTexturePromptClauses(optimization.originalPrompt).map((clause) => clause.id))
      : [];

    const list = el("div", "le-ai__list");
    for (const row of this.understandingRows(optimization.spec, plan)) {
      const item = el("div", "le-ai__item le-ai__item--top");
      const dot = el("span", "le-group__dot");
      dot.style.background = row.color;
      const values = el("div", "le-ai__chips le-ai__grow");
      if (this.editingGroup === row.groupId) {
        for (const attribute of row.attributes) {
          const field = el("label", "le-ai__edit");
          const input = document.createElement("input");
          input.value = attribute.description;
          input.addEventListener("change", () => {
            for (const profile of optimization.spec.styleProfiles) {
              if (row.profileIds.includes(profile.id) && profile.attributes[attribute.category]) {
                profile.attributes[attribute.category] = { description: input.value.trim() || attribute.description };
              }
            }
            optimization.optimizedPrompt = this.compile(optimization.spec, plan);
          });
          field.append(bindLocalizedText(el("b"), CATEGORY_LABELS[attribute.category]), input);
          values.append(field);
        }
      } else if (row.attributes.length) {
        for (const attribute of row.attributes) {
          const chip = el("span", "le-ai__chip");
          chip.append(bindLocalizedText(el("b"), CATEGORY_LABELS[attribute.category]), ` ${attribute.description}`);
          values.append(chip);
        }
      } else {
        values.append(bindLocalizedText(el("span", "le-ai__muted", ""), "保持原样"));
      }
      const edit = bindLocalizedText(button("", "le-link"), (this.editingGroup === row.groupId ? "好了" : "改"));
      edit.addEventListener("click", () => { this.editingGroup = this.editingGroup === row.groupId ? null : row.groupId; this.render(); });
      item.append(dot, el("b", "le-ai__name", row.name), values, edit);
      list.append(item);
    }
    root.append(list);
    const touched = new Set(optimization.spec.styleProfiles.flatMap((profile) => Object.keys(profile.attributes)));
    const kept = TEXTURE_VISUAL_ATTRIBUTE_CATEGORIES.filter((category) => !touched.has(category)).map((category) => CATEGORY_LABELS[category]);
    if (kept.length) root.append(bindLocalizedText(el("p", "le-ai__muted", ""), "{{v0}}保持原样", () => ({ v0: kept.map(label => t(label)).join(", ") })));
    for (const warning of optimization.warnings) root.append(el("div", "le-strip le-strip--warn", warning));
    const footer = el("div", "le-ai__footer");
    const note = el("span", "le-ai__muted le-ai__grow");
    if (drawn.length) {
      note.append(bindLocalizedText(el("span"), "{{copy}}", () => ({ copy: costLine(plan) })), document.createElement("br"), plan.canvases.length > 1
        ? bindLocalizedText(el("span"), "点「开始画」才会调用图片模型 · 分 {{v0}} 次画，后面几张参照第一张的颜色", { v0: plan.canvases.length })
        : bindLocalizedText(el("span"), "点「开始画」才会调用图片模型"));
    } else {
      bindLocalizedText(note, "这次没有需要画的，按上面的提示去做就好");
    }
    const raw = bindLocalizedText(button("", "le-link le-link--muted"), "不用理解，按原话画");
    raw.disabled = this.imageReady === false;
    raw.addEventListener("click", () => void this.startJob(false));
    const missing = this.imageModelStrip();
    if (missing) root.append(missing);
    const go = bindLocalizedText(button("", "accent"), "开始画");
    go.disabled = this.busy || this.imageReady === false || !drawn.length;
    go.addEventListener("click", () => void this.startJob(true));
    footer.append(note, raw, go);
    if (this.error) root.append(el("p", "le-ai__warn", this.error));
    root.append(footer);
  }

  /** Pieces the request doesn't cover: drawn as they are and left off, unless the user wants them drawn too. */
  private keptList(optimization: TexturePromptOptimizationResult, plan: TextureEditPlanned): HTMLElement | null {
    // What the understanding skipped, kept so a piece switched back on can be skipped again.
    if (this.skippedFor !== optimization) {
      this.skippedFor = optimization;
      this.skipped = new Map(optimization.spec.regions.filter((region) => region.keep)
        .map((region) => [region.key, { keep: region.keep!, label: region.keepLabel ?? "" }]));
    }
    if (!this.skipped.size) return null;
    const regionByKey = new Map(optimization.spec.regions.map((region) => [region.key, region]));
    const keyByUnit = new Map(textureEditPlanRegions(plan.canvases).map((region) => [region.textureUnitId, region.key]));
    const setDrawn = (keys: readonly string[], drawn: boolean) => {
      const others = optimization.spec.regions.filter((region) => !region.keep && !keys.includes(region.key));
      const profiles = [...new Set(others.flatMap((region) => region.styleProfileIds))];
      const clauses = [...new Set(others.flatMap((region) => region.clauseIds))];
      for (const key of keys) {
        const region = regionByKey.get(key);
        const skipped = this.skipped.get(key);
        if (!region || !skipped) continue;
        if (drawn) {
          delete region.keep;
          delete region.keepLabel;
          region.styleProfileIds = profiles.length ? profiles : optimization.spec.styleProfiles.map((profile) => profile.id);
          region.clauseIds = clauses;
        } else {
          region.keep = skipped.keep;
          if (skipped.label) region.keepLabel = skipped.label;
          region.styleProfileIds = [];
          region.clauseIds = [];
        }
      }
      optimization.optimizedPrompt = this.compile(optimization.spec, plan);
      this.render();
    };
    const left = [...this.skipped.keys()].filter((key) => regionByKey.get(key)?.keep);
    const box = el("div", "le-ai__pieces le-ai__skipped");
    const head = el("div", "le-ai__row");
    head.append(left.length
      ? bindLocalizedText(el("span", "le-ai__grow", ""), "这 {{v0}} 块这次可以跳过，会保持原样、不换上。想照样画哪块就勾上", { v0: left.length })
      : bindLocalizedText(el("span", "le-ai__grow", ""), "原本可以跳过的 {{v0}} 块都会照样画", { v0: this.skipped.size }));
    const all = bindLocalizedText(button("", "le-link"), left.length ? "全部画上" : "恢复跳过");
    all.addEventListener("click", () => setDrawn([...this.skipped.keys()], left.length > 0));
    head.append(all);
    box.append(head);
    // Hovering a piece says why it can be skipped.
    const why = el("p", "le-ai__muted le-ai__why", "");
    box.append(why);
    const list = el("div", "le-ai__piece-list");
    for (const group of this.groups().filter((item) => this.selection.has(item.id))) {
      const units = group.units.filter((unit) => this.skipped.has(keyByUnit.get(unit.id) ?? ""));
      if (!units.length) continue;
      const title = el("div", "le-ai__piece-section");
      title.append(el("b", "", group.name), bindLocalizedText(el("span", "le-ai__muted", ""), "{{v0}} 块", { v0: units.length }));
      list.append(title);
      const live = this.options.capture(units.map((unit) => unit.drawableIds), 240);
      const grid = el("div", "le-ai__piece-grid");
      units.forEach((unit, index) => {
        const key = keyByUnit.get(unit.id)!;
        const skipped = this.skipped.get(key)!;
        const item = el("label", "le-ai__piece");
        const check = document.createElement("input");
        check.type = "checkbox";
        check.checked = !regionByKey.get(key)?.keep;
        check.addEventListener("change", () => setDrawn([key], check.checked));
        item.classList.toggle("on", check.checked);
        // A faint layer (a highlight, a shadow) can come out empty from the live model; show its texture then.
        const shot = live[index] && !blankCanvas(live[index]!) ? live[index] : undefined;
        if (shot) {
          shot.className = "le-locator";
          shot.style.width = shot.style.height = "120px";
        }
        item.append(shot ?? this.locator.thumbnail(unit, 120), check, el("span", "", skipped.label || unit.label));
        item.addEventListener("mouseenter", () => { this.options.focus(unit.drawableIds); why.textContent = `${skipped.label || unit.label}：${skipped.keep}`; });
        item.addEventListener("mouseleave", () => { this.options.focus([]); why.textContent = ""; });
        grid.append(item);
      });
      list.append(grid);
    }
    box.append(list);
    return box;
  }

  /** Each sentence of the request and what happens to it; only some are drawn. */
  private actionList(spec: TextureVisualEditSpec): HTMLElement | null {
    const plans = spec.actions ?? [];
    if (!plans.length || plans.every((plan) => plan.action === "repaint")) return null;
    const clauses = new Map(splitTexturePromptClauses(this.optimization?.originalPrompt ?? this.prompt).map((clause) => [clause.id, clause.text]));
    const box = el("div", "le-ai__plan");
    for (const plan of plans) {
      const row = el("div", `le-ai__plan-row le-ai__plan-row--${plan.action}`);
      row.append(bindLocalizedText(el("span", "le-ai__plan-tag"), ACTION_LABELS[plan.action]), el("span", "le-ai__grow", `「${clauses.get(plan.clauseId) ?? plan.clauseId}」${plan.note ? ` ${plan.note}` : ""}`));
      if (plan.action === "toggle") {
        const go = bindLocalizedText(button("", "le-link"), "去开关");
        go.addEventListener("click", () => this.options.openPane("toggles"));
        row.append(go);
      } else if (plan.action === "recolor") {
        const go = bindLocalizedText(button("", "le-link"), "用换色");
        bindLocalizedAttribute(go, "title", "只换颜色的话，换色马上就能看到，不用等也不花钱");
        go.addEventListener("click", () => this.options.openPane("textures"));
        row.append(go);
      } else if (plan.action === "unselected") {
        const group = this.groups().find((item) => plan.part && (item.name === plan.part || item.name.includes(plan.part)));
        if (group && !this.selection.has(group.id)) {
          const add = bindLocalizedText(button("", "le-link"), "加上{{v0}}", { v0: group.name });
          add.addEventListener("click", () => {
            this.toggleGroup(group);
            // The understanding was for the old selection.
            this.optimization = null;
            this.step = "pick";
            this.render();
          });
          row.append(add);
        }
      }
      box.append(row);
    }
    return box;
  }

  private understandingRows(spec: TextureVisualEditSpec, plan: TextureEditPlanned) {
    const regions = textureEditPlanRegions(plan.canvases);
    const keyByUnit = new Map(regions.map((region) => [region.textureUnitId, region.key]));
    const groups = this.groups();
    return groups.filter((group) => this.selection.has(group.id)).map((group) => {
      const keys = new Set([...this.selection.get(group.id)!].flatMap((id) => keyByUnit.get(id) ?? []));
      const profileIds = [...new Set(spec.regions.filter((region) => keys.has(region.key)).flatMap((region) => region.styleProfileIds))];
      const attributes: Array<{ category: TextureVisualAttributeCategory; description: string }> = [];
      for (const category of TEXTURE_VISUAL_ATTRIBUTE_CATEGORIES) {
        const descriptions = [...new Set(spec.styleProfiles
          .filter((profile) => profileIds.includes(profile.id))
          .flatMap((profile) => profile.attributes[category]?.description ?? []))];
        if (descriptions.length) attributes.push({ category, description: descriptions.join("；") });
      }
      const chosen = this.selection.get(group.id)!;
      const name = chosen.size === group.units.length ? group.name : `${group.name} · ${chosen.size} 块`;
      return { groupId: group.id, name, color: group.color, profileIds, attributes };
    });
  }

  private renderWorking(root: HTMLElement): void {
    const job = this.job;
    const card = el("div", "le-card");
    if (this.step === "failed") {
      card.append(bindLocalizedText(el("b", "", ""), "这次没画成"), bindLocalizedText(el("p", "le-ai__warn", ""), "{{copy}}", () => ({ copy: (this.error || t("图片模型没有返回可用的结果")) })));
      const actions = el("div", "le-ai__footer");
      const back = bindLocalizedText(button("", "secondary"), "回去改一下");
      back.addEventListener("click", () => { this.job = null; this.step = "pick"; this.error = ""; this.render(); });
      const retry = bindLocalizedText(button("", "accent"), "再试一次");
      retry.disabled = !job;
      retry.addEventListener("click", async () => {
        if (!job) return;
        try {
          this.job = await window.desktopPet.retryCharacterJob(job.id);
          this.step = "working";
          this.error = "";
        } catch (error) {
          this.error = errorText(error);
        }
        this.render();
      });
      actions.append(el("span", "le-ai__grow"), back, retry);
      card.append(actions);
      root.append(card);
      return;
    }
    const names = this.groups().filter((group) => this.selection.has(group.id)).map((group) => group.name);
    card.append(bindLocalizedText(el("b", "", ""), "{{copy}}", () => ({ copy: (names.length ? t("正在画{{v0}}", { v0: names.join("、") }) : t("正在画")) })));
    if (job) card.append(el("span", "le-ai__muted", `「${job.input.originalDescription ?? job.input.description}」`));
    const bar = el("span", "le-ai__bar");
    const fill = el("span", "");
    fill.style.width = `${Math.round(this.progressValue() * 100)}%`;
    bar.append(fill);
    bar.setAttribute("role", "progressbar");
    card.append(bar, bindLocalizedText(el("span", "le-ai__muted", ""), "{{copy}}", () => ({ copy: (this.progressDetail || t("可以先去做别的，画好了会通知你")) })));
    if (job?.paused) {
      const resume = bindLocalizedText(button("", "accent"), "继续");
      resume.addEventListener("click", async () => {
        try { this.job = await window.desktopPet.resumeCharacterJob(job.id); } catch (error) { this.error = errorText(error); }
        this.render();
      });
      card.append(resume);
    }
    const stop = bindLocalizedText(button("", "secondary"), "停下");
    stop.addEventListener("click", async () => {
      if (!job) return;
      await window.desktopPet.cancelCharacterJob(job.id).catch(() => undefined);
      this.job = null;
      this.step = "pick";
      this.render();
    });
    const actions = el("div", "le-ai__footer");
    actions.append(el("span", "le-ai__grow"), stop);
    card.append(actions);
    root.append(card);
  }

  private renderReview(root: HTMLElement): void {
    const job = this.job;
    if (!job) return;
    const results = job.artifacts.textureEdits ?? {};
    root.append(bindLocalizedText(el("div", "le-strip le-strip--ok", ""), "画好了，左边就是新样子。按住「看原版」可以对比"));
    const list = el("div", "le-ai__list");
    const units = new Set(job.input.textureUnitIds ?? []);
    const groups = this.groups().map((group) => ({ group, ids: group.units.map((unit) => unit.id).filter((id) => units.has(id)) }))
      .filter((entry) => entry.ids.length);
    const shown = new Set<string>();
    for (const { group, ids } of groups) {
      const own = ids.filter((id) => !shown.has(id));
      if (!own.length) continue;
      own.forEach((id) => shown.add(id));
      const ok = own.filter((id) => ["review", "accepted"].includes(results[id]?.status ?? ""));
      const failed = own.filter((id) => !ok.includes(id));
      const accepted = ok.filter((id) => results[id]?.status === "accepted");
      const row = el("div", "le-ai__item");
      const first = ok[0] ?? own[0]!;
      const result = results[first];
      const pair = el("span", "le-ai__pair");
      const source = result?.sourcePreviewPath ?? result?.sourcePath;
      const edited = result?.editedPreviewPath ?? result?.editedPatchPath;
      pair.append(thumb(source ? generationUrl(job.id, source) : null), el("span", "le-ai__arrow", "→"), thumb(edited ? generationUrl(job.id, edited) : null));
      const doubtful = ok.flatMap((id) => results[id]?.quality && !results[id]!.quality!.accepted ? results[id]!.quality!.issues : []);
      const partial = accepted.length > 0 && accepted.length < ok.length;
      const note = failed.length && !ok.length
        ? "这块没画好，先保持原样"
        : failed.length ? `${ok.length} 块画好了，${failed.length} 块没画好保持原样`
        : doubtful.length ? `${ok.length} 块 · ${doubtful[0]}，按住「看原版」比一下`
        : partial ? `用了 ${accepted.length} / ${ok.length} 块` : `${ok.length} 块`;
      const text = el("span", "le-ai__grow le-ai__stack");
      text.append(el("b", "", group.name), el("small", (failed.length && !ok.length) || doubtful.length ? "le-ai__warn" : "le-ai__muted", note));
      row.append(pair, text);
      if (ok.length > 1) {
        const open = this.reviewPieces === group.id;
        const pick = bindLocalizedText(button("", "le-link"), (open ? "收起" : "挑几块"));
        pick.addEventListener("click", () => { this.reviewPieces = open ? null : group.id; this.render(); });
        row.append(pick);
      }
      if (ok.length) {
        const on = accepted.length === ok.length;
        row.append(bindLocalizedText(el("span", "le-ai__muted", ""), "用这个"));
        const toggle = button("", "le-switch");
        toggle.setAttribute("role", "switch");
        toggle.setAttribute("aria-checked", String(on));
        bindLocalizedAttribute(toggle, "aria-label", "用新的{{v0}}", { v0: group.name });
        toggle.append(el("span", "le-switch__knob"));
        toggle.disabled = this.busy;
        toggle.addEventListener("click", () => void this.accept(ok, !on));
        row.append(toggle);
      }
      if (failed.length) {
        const redo = bindLocalizedText(button("", "secondary"), "重画这块");
        redo.disabled = this.busy;
        redo.addEventListener("click", () => void this.redo(failed));
        row.append(redo);
      }
      row.addEventListener("mouseenter", () => this.options.highlight(this.options.capability.textureUnits
        .filter((unit) => own.includes(unit.id)).flatMap((unit) => unit.drawableIds)));
      row.addEventListener("mouseleave", () => this.options.highlight([]));
      list.append(row);
      if (this.reviewPieces === group.id && ok.length > 1) list.append(this.reviewPieceGrid(group, own, results));
    }
    root.append(list);
    const acceptedCount = Object.values(results).filter((result) => result.status === "accepted").length;
    const footer = el("div", "le-ai__footer le-ai__footer--stack");
    footer.append(bindLocalizedText(el("span", "le-ai__muted", ""), "保存的都会放进「换色」，随时换回原版。不满意可以改一下要求再来一次"));
    const actions = el("div", "le-ai__row");
    const discard = bindLocalizedText(button("", "le-link le-link--muted"), "不要了");
    discard.addEventListener("click", () => void this.finish(false));
    const again = bindLocalizedText(button("", "secondary"), "改一下再来");
    again.addEventListener("click", () => void this.finish(false, true));
    const store = bindLocalizedText(button("", "secondary"), "只存到换色");
    bindLocalizedAttribute(store, "title", "存下来，先不换上，想穿的时候去「换色」点一下");
    store.disabled = this.busy || acceptedCount === 0;
    store.addEventListener("click", () => void this.finish(true, false, false));
    const keep = bindLocalizedText(button("", "accent"), "{{copy}}", () => ({ copy: (acceptedCount ? t("保存并换上 {{v0}} 块", { v0: acceptedCount }) : t("保存并换上")) }));
    keep.disabled = this.busy || acceptedCount === 0;
    keep.addEventListener("click", () => void this.finish(true));
    actions.append(discard, el("span", "le-ai__grow"), again, store, keep);
    footer.append(actions);
    if (this.error) root.append(el("p", "le-ai__warn", this.error));
    root.append(footer);
  }

  /** The try-on step's pieces of one part, each switched on or off by itself. */
  private reviewPieceGrid(group: Group, ids: readonly string[], results: NonNullable<CharacterGenerationJob["artifacts"]["textureEdits"]>): HTMLElement {
    const box = el("div", "le-ai__pieces");
    const shown = this.options.shown();
    const units = group.units.filter((unit) => ids.includes(unit.id));
    const sections = pieceSections(units, group.name, (unit) => !shown || unit.drawableIds.some((id) => shown.has(id)));
    const pieces = sections.flatMap((section) => section.pieces);
    // As the preview shows them now: switched-on pieces in the new look.
    const live = this.options.capture(pieces.map((piece) => piece.unit.drawableIds), 240);
    const shots = new Map(pieces.map((piece, index) => [piece.unit.id, live[index]]));
    const list = el("div", "le-ai__piece-list");
    for (const section of sections) {
      if (sections.length > 1) {
        const title = el("div", "le-ai__piece-section");
        title.append(el("b", "", section.name));
        list.append(title);
      }
      const grid = el("div", "le-ai__piece-grid");
      for (const { unit, name } of section.pieces) {
        const status = results[unit.id]?.status ?? "";
        const drawn = status === "review" || status === "accepted";
        const item = el("label", "le-ai__piece");
        const check = document.createElement("input");
        check.type = "checkbox";
        check.checked = status === "accepted";
        check.disabled = this.busy || !drawn;
        bindLocalizedAttribute(check, "aria-label", "用新的{{v0}} · {{v1}}", { v0: section.name, v1: name });
        check.addEventListener("change", () => void this.accept([unit.id], check.checked));
        item.classList.toggle("on", check.checked);
        const shot = shots.get(unit.id);
        if (shot) {
          shot.className = "le-locator";
          shot.style.width = shot.style.height = "120px";
        }
        item.append(shot ?? this.locator.thumbnail(unit, 120), check, el("span", "", name));
        if (!drawn) item.append(bindLocalizedText(el("small", "le-ai__warn", ""), "没画成"));
        item.addEventListener("mouseenter", () => this.options.focus(unit.drawableIds));
        item.addEventListener("mouseleave", () => this.options.focus([]));
        grid.append(item);
      }
      list.append(grid);
    }
    box.append(list);
    return box;
  }

  // ---------- preview overlays ----------

  private renderStack(): void {
    this.stackLayer?.remove();
    this.stackLayer = null;
    if (!this.stack || (this.step !== "pick" && this.step !== "pieces")) return;
    const layer = el("div", "le-stack");
    layer.setAttribute("role", "dialog");
    bindLocalizedAttribute(layer, "aria-label", "这里有几个部位");
    layer.style.left = `${Math.max(8, this.stack.x + 12)}px`;
    layer.style.top = `${Math.max(8, this.stack.y - 12)}px`;
    layer.append(bindLocalizedText(el("span", "le-ai__muted", ""), "这里有 {{v0}} 个部位，选要改的", { v0: this.stack.groups.length }));
    for (const group of this.stack.groups) {
      const row = el("label", "le-stack__row");
      const check = document.createElement("input");
      check.type = "checkbox";
      check.checked = this.selection.has(group.id);
      check.addEventListener("change", () => this.toggleGroup(group));
      const dot = el("span", "le-group__dot");
      dot.style.background = group.color;
      row.append(check, dot, el("span", "", group.name));
      row.addEventListener("mouseenter", () => this.options.highlight(group.drawableIds));
      row.addEventListener("mouseleave", () => this.options.highlight([]));
      layer.append(row);
    }
    const close = bindLocalizedText(button("", "le-link"), "好了");
    close.addEventListener("click", () => { this.stack = null; this.options.focus([]); this.renderStack(); });
    layer.append(close);
    this.options.stage.append(layer);
    // Keep the chooser inside the preview, next to where the user clicked.
    const bounds = this.options.stage.getBoundingClientRect();
    const left = Math.min(this.stack.x + 12, bounds.width - layer.offsetWidth - 8);
    const top = Math.min(this.stack.y - 12, bounds.height - layer.offsetHeight - 8);
    layer.style.left = `${Math.max(8, left)}px`;
    layer.style.top = `${Math.max(8, top)}px`;
    this.stackLayer = layer;
  }

  private renderCompare(): void {
    this.compareButton?.remove();
    this.compareButton = null;
    if (this.step !== "review" || !this.job) return;
    const hold = bindLocalizedText(button("", "le-compare"), "按住看原版");
    const down = () => void this.showResult(false);
    const up = () => void this.showResult(true);
    hold.addEventListener("pointerdown", down);
    hold.addEventListener("pointerup", up);
    hold.addEventListener("pointerleave", up);
    this.options.stage.append(hold);
    this.compareButton = hold;
  }

  private syncPicking(): void {
    if (this.step !== "pick" && this.step !== "pieces") {
      this.options.setPicking(null);
      return;
    }
    this.options.setPicking((drawableIds, point) => {
      if (this.step === "pieces" && this.piecesGroup) {
        const group = this.groups().find((item) => item.id === this.piecesGroup);
        const top = drawableIds.find((id) => group?.drawableIds.includes(id));
        if (!group || !top) return;
        const set = this.selection.get(group.id) ?? new Set<string>();
        for (const unit of group.units.filter((item) => item.drawableIds.includes(top))) {
          if (set.has(unit.id)) set.delete(unit.id); else set.add(unit.id);
        }
        if (set.size) this.selection.set(group.id, set); else this.selection.delete(group.id);
        this.selectionChanged();
        return;
      }
      const groups = this.groupsAt(drawableIds);
      if (!groups.length) return;
      if (groups.length === 1) {
        this.stack = null;
        this.toggleGroup(groups[0]!);
      } else {
        this.stack = { groups, x: point.x, y: point.y };
        this.render();
      }
    });
  }

  // ---------- actions ----------

  private referencesForRequest(): TextureReferenceSelection[] {
    return [{
      id: "model-front", source: "model-preview", name: `${this.options.capability.modelName} 正面`, description: "",
      path: null, textureUnitId: null, materialId: null, previewUrl: null,
    }, ...this.references];
  }

  private async understand(): Promise<void> {
    if (this.plan.status !== "ok" || !this.prompt.trim()) return;
    const plan = this.plan.plan;
    this.step = "understanding";
    this.error = "";
    this.understandFailed = false;
    this.stack = null;
    this.render();
    try {
      const result = await window.desktopPet.optimizeCharacterTexturePrompt({
        templateId: this.options.capability.templateId,
        modelName: this.options.capability.modelName,
        textureUnitIds: this.selectedUnitIds(),
        parts: {
          selected: this.groups().filter((group) => this.selection.has(group.id)).map((group) => group.name),
          others: this.groups().filter((group) => !this.selection.has(group.id)).map((group) => group.name),
        },
        prompt: this.prompt.trim(),
        planHash: plan.hash,
        imageModel: plan.plan.imageModel,
        references: this.referencesForRequest(),
      });
      if (result.planned) this.plan = { status: "ok", plan: result.planned };
      this.optimization = result;
      this.step = "confirm";
    } catch (error) {
      this.error = `AI 没理解成：${errorText(error)}。可以直接按原话画`;
      this.understandFailed = true;
      this.optimization = null;
      this.step = "pick";
    }
    this.render();
  }

  /**
   * The prompt for the whole edit, or for one canvas of it: `canvasIndex`
   * keeps only that canvas's regions, so a separate image request is never
   * told about regions that are not on its canvas.
   */
  private compile(spec: TextureVisualEditSpec, plan: TextureEditPlanned, canvasIndex?: number): string {
    const references = this.referencesForRequest();
    const ids = promptIds(references);
    const units = new Map(this.options.capability.textureUnits.map((unit) => [unit.id, unit]));
    const regions = textureEditPlanRegions(plan.canvases)
      .filter((region) => canvasIndex === undefined || region.canvasIndex === canvasIndex);
    const keys = new Set(regions.map((region) => region.key));
    if (canvasIndex !== undefined) spec = { ...spec, regions: spec.regions.filter((region) => keys.has(region.key)) };
    const input: TextureVisualEditSpecInput = {
      originalPrompt: this.prompt.trim(),
      clauses: splitTexturePromptClauses(this.prompt.trim()),
      imageModel: plan.plan.imageModel,
      positionReferenceAvailable: true,
      regions: regions.map((region) => ({ key: region.key, label: units.get(region.textureUnitId)?.label ?? region.label })),
      references: references.map((reference, index) => ({
        selectionId: reference.id, id: ids[index]!, source: reference.source, name: reference.name, description: reference.description,
      })),
    };
    return compileTextureVisualPrompt(spec, input);
  }

  private async startJob(useUnderstanding: boolean): Promise<void> {
    if (this.plan.status !== "ok" || this.busy) return;
    const plan = this.plan.plan;
    const optimization = useUnderstanding ? this.optimization : null;
    const references = this.referencesForRequest();
    const ids = promptIds(references);
    const delivery = textureReferenceDeliveryState(Boolean(optimization), references.map((reference, index) => ({
      selectionId: reference.id, promptId: ids[index]!,
    })), []);
    const deliveryById = new Map(delivery.map((item) => [item.selectionId, item]));
    this.busy = true;
    this.error = "";
    this.render();
    try {
      this.job = await window.desktopPet.createCharacterJob({
        operation: "texture-edit",
        templateId: this.options.capability.templateId,
        description: optimization?.optimizedPrompt ?? this.prompt.trim(),
        originalDescription: this.prompt.trim(),
        optimizedDescription: this.optimization?.optimizedPrompt ?? null,
        promptOptimization: optimization ? "accepted" : this.optimization ? "rejected" : "not-requested",
        promptVisualSpec: optimization?.spec ?? null,
        textureBatchPlanHash: plan.hash,
        ...plan.canvases.length > 1 ? {
          textureCanvasDescriptions: plan.canvases.map((_, index) =>
            optimization ? this.compile(optimization.spec, plan, index) : this.prompt.trim()),
        } : {},
        // Starting after seeing the price is the go-ahead for every canvas it takes.
        ...plan.estimatedUsd > DEFAULT_CHARACTER_BUDGET_USD
          ? { budgetLimitUsd: Math.min(10, Math.ceil((plan.estimatedUsd + 0.05) * 100) / 100) }
          : {},
        textureImageModel: plan.plan.imageModel,
        textureReferences: references.flatMap((reference) => {
          const item = deliveryById.get(reference.id);
          return item?.send ? [{ ...reference, promptId: item.promptId, deliveryUse: item.required ? "required" as const : "supplemental" as const }] : [];
        }),
        referenceImage: null,
        sourceModelId: this.options.capability.modelId,
        editMode: "appearance",
        textureUnitIds: this.selectedUnitIds(),
      });
      this.step = "working";
      this.progressDetail = "";
    } catch (error) {
      this.error = errorText(error);
    } finally {
      this.busy = false;
    }
    this.render();
  }

  private progressValue(): number {
    const results = Object.values(this.job?.artifacts.textureEdits ?? {});
    if (!results.length) return 0.1;
    const done = results.filter((result) => !["pending", "running"].includes(result.status)).length;
    return Math.max(0.1, done / results.length);
  }

  private async onProgress(progress: CharacterGenerationProgress): Promise<void> {
    const job = progress.job;
    if (job.input.operation !== "texture-edit" || job.input.sourceModelId !== this.options.capability.modelId) return;
    if (this.job && progress.jobId !== this.job.id) return;
    if (!this.job && !ACTIVE_PHASES.has(job.phase) && job.phase !== "preview") return;
    this.job = job;
    this.progressDetail = progress.detail;
    if (job.phase === "preview" && this.step !== "review") {
      this.step = "review";
      this.restoreSelection(job);
      await this.acceptAllGood(job);
    } else if (job.phase === "failed") {
      this.step = "failed";
      this.error = progress.detail;
    } else if (job.phase === "cancelled" || job.phase === "saved") {
      if (this.step === "working") { this.step = "pick"; this.job = null; }
    } else if (ACTIVE_PHASES.has(job.phase) && this.step !== "working") {
      this.step = "working";
      this.restoreSelection(job);
    }
    this.render();
  }

  /** A job that was running when the page closed picks up where it was. */
  private async resume(): Promise<void> {
    this.resuming = true;
    const jobs = await window.desktopPet.listCharacterJobs().catch(() => [] as CharacterGenerationJob[]);
    const job = jobs.find((item) => item.input.operation === "texture-edit" && item.input.sourceModelId === this.options.capability.modelId &&
      (ACTIVE_PHASES.has(item.phase) || item.phase === "preview"));
    this.resuming = false;
    if (!job) {
      this.render();
      return;
    }
    this.job = job;
    this.restoreSelection(job);
    this.prompt = job.input.originalDescription ?? job.input.description;
    this.step = job.phase === "preview" ? "review" : "working";
    if (this.step === "review") await this.showResult(true);
    this.render();
  }

  private restoreSelection(job: CharacterGenerationJob): void {
    if (this.selection.size) return;
    const ids = new Set(job.input.textureUnitIds ?? []);
    for (const group of this.groups()) {
      const chosen = group.units.filter((unit) => ids.has(unit.id)).map((unit) => unit.id);
      if (chosen.length) this.selection.set(group.id, new Set(chosen));
    }
  }

  /** Good results start switched on; the user switches off what they don't want. */
  private async acceptAllGood(job: CharacterGenerationJob): Promise<void> {
    // What the understanding said the request leaves alone stays off; the user can still switch it on.
    const spec = job.input.promptVisualSpec;
    const kept = spec && this.plan.status === "ok"
      ? new Set(textureEditPlanRegions(this.plan.plan.canvases).filter((region) => keptRegionKeys(spec).has(region.key)).map((region) => region.textureUnitId))
      : new Set<string>();
    const pending = Object.values(job.artifacts.textureEdits ?? {})
      .filter((result) => result.status === "review" && !kept.has(result.textureUnitId)).map((result) => result.textureUnitId);
    if (pending.length) await this.accept(pending, true);
    else await this.showResult(true);
  }

  private async accept(unitIds: readonly string[], accepted: boolean): Promise<void> {
    const job = this.job;
    if (!job) return;
    this.busy = true;
    this.render();
    try {
      for (const id of unitIds) {
        const change = await window.desktopPet.acceptCharacterTexture(job.id, id, accepted);
        this.job = change.job;
        await this.options.previewDelta(change.delta);
      }
    } catch (error) {
      this.error = errorText(error);
    } finally {
      this.busy = false;
    }
    this.render();
  }

  private async showResult(enabled: boolean): Promise<void> {
    if (!this.job) return;
    try {
      await this.options.previewDelta(await window.desktopPet.setCharacterTexturePreview(this.job.id, enabled));
    } catch {
      // The job may have finished meanwhile; nothing left to show.
    }
  }

  private async finish(keep: boolean, again = false, apply = true): Promise<void> {
    const job = this.job;
    if (!job) return;
    this.busy = true;
    this.render();
    try {
      if (!keep) {
        for (const result of Object.values(job.artifacts.textureEdits ?? {})) {
          if (result.status === "accepted") {
            const change = await window.desktopPet.acceptCharacterTexture(job.id, result.textureUnitId, false);
            await this.options.previewDelta(change.delta);
          }
        }
      }
      await this.showResult(false);
      const count = keep ? Object.values(job.artifacts.textureEdits ?? {}).filter((result) => result.status === "accepted").length : 0;
      await window.desktopPet.saveAcceptedCharacterTexturesAndCancel(job.id, apply);
      this.job = null;
      this.step = "pick";
      if (!again) {
        this.selection.clear();
        this.plan = { status: "empty" };
        this.prompt = keep ? "" : this.prompt;
      } else {
        this.schedulePlan();
      }
      this.optimization = null;
      this.reviewPieces = null;
      if (count) await this.options.saved(count, apply);
    } catch (error) {
      this.error = errorText(error);
    } finally {
      this.busy = false;
    }
    this.render();
  }

  /** Keeps what came out well, then goes back to confirm with only the failed pieces. */
  private async redo(unitIds: readonly string[]): Promise<void> {
    const job = this.job;
    if (!job) return;
    await this.finish(true, true);
    const wanted = new Set(unitIds);
    for (const [groupId, ids] of [...this.selection]) {
      const kept = new Set([...ids].filter((id) => wanted.has(id)));
      if (kept.size) this.selection.set(groupId, kept); else this.selection.delete(groupId);
    }
    this.prompt = job.input.originalDescription ?? job.input.description;
    this.selectionChanged();
  }
}

function promptIds(references: readonly TextureReferenceSelection[]): TextureReferencePromptId[] {
  let visual = 0;
  return references.map((reference) => reference.source === "model-preview" ? "P0" : `R${++visual}` as TextureReferencePromptId);
}

/** A piece's own Part name when the rigger gave one (袜子, 腿环…). */
function costLine(plan: TextureEditPlanned): string {
  const items = plan.canvases.flatMap((canvas) => canvas.items);
  const count = plan.canvases.length;
  const shrunk = items.filter((item) => item.scale < 1);
  // Shrunk but still 1024 px or more on the long side: far beyond what the screen shows.
  const blurred = shrunk.filter((item) => Math.max(item.workWidth, item.workHeight) < 1024).length;
  const note = blurred ? t(" · {{v0}} 块太大，缩小后画，细节会略糊", { v0: blurred })
    : shrunk.length ? t(" · {{v0}} 块纹理比屏幕上用到的大得多，缩小后画，看不出差别", { v0: shrunk.length }) : "";
  return t("{{v0}} 块纹理 · 画 {{v1}} 次 · 约 ${{v2}} · {{v3}} 分钟{{v4}}", { v0: items.length, v1: count, v2: plan.estimatedUsd.toFixed(2), v3: count > 1 ? `${count}–${count * 2}` : "1–2", v4: note });
}

function generationUrl(jobId: string, absolutePath: string): string {
  const normalized = absolutePath.replaceAll("\\", "/");
  return `pet-generation://${encodeURIComponent(jobId)}/${encodeURIComponent(normalized.slice(normalized.lastIndexOf("/") + 1))}`;
}

function thumb(url: string | null): HTMLElement {
  const box = el("span", "le-ai__thumb");
  if (url) {
    const image = document.createElement("img");
    image.src = url;
    image.alt = "";
    box.append(image);
  }
  return box;
}

function errorText(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.replace(/^Error invoking remote method '[^']+': (Error: )?/, "");
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

/** True when almost nothing shows: under 1% of the shot is more than faintly drawn. */
function blankCanvas(canvas: HTMLCanvasElement): boolean {
  const context = canvas.getContext("2d");
  if (!context || !canvas.width || !canvas.height) return false;
  const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
  let shown = 0, samples = 0;
  for (let offset = 3; offset < data.length; offset += 16) {
    samples += 1;
    if (data[offset]! > 48) shown += 1;
  }
  return shown < samples * 0.01;
}
