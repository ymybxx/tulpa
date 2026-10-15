import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import { t } from "../i18n";
import type { ModelEditCapabilityManifest, ModelPartGroup, PetManifest, TextureEditUnit } from "../shared/contracts";
import {
  MAX_PART_GROUP_NAME_LENGTH,
  MAX_PART_GROUPS,
  effectivePartGroups,
  moveDrawablesToGroup,
  moveUnitsToGroup,
  partGroupColor,
  removeUnitsFromGroups,
  ungroupedUnits,
  unitsMatchingName,
  unitsOfGroup,
} from "../shared/part-groups";
import { PieceLocator } from "./piece-locator";

export interface PartRecognitionState {
  running: boolean;
  done: number;
  total: number;
  message: string;
}

export interface PartGroupsPanelOptions {
  root: HTMLElement;
  manifest: PetManifest;
  capability: ModelEditCapabilityManifest;
  imageUrl: (relativePath: string) => string;
  /** The draft changed. */
  changed: () => void;
  highlight: (drawableIds: readonly string[]) => void;
  /** Model clicks while a group is open add or remove the piece under the cursor. */
  setPicking: (handler: ((drawableIds: string[]) => void) | null) => void;
  recognize: () => void;
  stopRecognition: () => void;
}

const UNGROUPED = "__ungrouped__";
const THUMBNAIL_LIMIT = 48;

/**
 * 装扮 › 部位: the model's pieces in named groups that 换色, AI 换装 and layer
 * hiding all select by. Groups start from the automatic grouping and become
 * the user's as soon as anything is changed.
 */
export class PartGroupsPanel {
  private expanded: string | null = null;
  private renaming: string | null = null;
  private query = "";
  private recognition: PartRecognitionState | null = null;
  private locator: PieceLocator;

  constructor(private options: PartGroupsPanelOptions) {
    this.locator = new PieceLocator(options.capability.modelId, options.imageUrl);
  }

  update(options: Partial<PartGroupsPanelOptions>): void {
    const modelChanged = options.capability && options.capability.modelId !== this.options.capability.modelId;
    this.options = { ...this.options, ...options };
    if (modelChanged) this.locator = new PieceLocator(this.options.capability.modelId, this.options.imageUrl);
    this.render();
  }

  setRecognition(state: PartRecognitionState | null): void {
    this.recognition = state;
    this.render();
  }

  dispose(): void {
    this.options.setPicking(null);
    this.options.highlight([]);
  }

  private groups(): ModelPartGroup[] {
    return effectivePartGroups(this.options.manifest, this.options.capability);
  }

  private save(groups: ModelPartGroup[]): void {
    this.options.manifest.partGroups = groups;
    this.options.changed();
    this.render();
  }

  render(): void {
    const { root, capability } = this.options;
    this.options.highlight([]);
    const groups = this.groups();
    const loose = ungroupedUnits(groups, capability);
    if (this.expanded && this.expanded !== UNGROUPED && !groups.some((group) => group.id === this.expanded)) this.expanded = null;
    root.replaceChildren();

    const head = element("div", "le-pane__head");
    head.append(bindLocalizedText(element("h3", "", ""), "部位"), bindLocalizedText(element("span", "", ""), "换色、AI 换装、隐藏图层都按这里的分组来选"));
    const create = bindLocalizedText(button("", "secondary"), "+ 新分组");
    create.disabled = groups.length >= MAX_PART_GROUPS;
    create.addEventListener("click", () => {
      const id = `group-${crypto.randomUUID().slice(0, 8)}`;
      this.expanded = id;
      this.renaming = id;
      this.save([...groups, { id, name: `新分组 ${groups.length + 1}`, drawableIds: [] }]);
    });
    head.append(create);
    root.append(head);

    const recognition = this.recognition;
    if (recognition?.running) {
      const strip = element("div", "le-strip le-strip--info");
      const text = bindLocalizedText(element("span", "", ""), "识图模型正在看：{{v0}} / {{v1}} 块 · 认出来的直接放进对应分组，都能改", { v0: recognition.done, v1: recognition.total });
      const stop = bindLocalizedText(button("", "secondary"), "停下");
      stop.addEventListener("click", () => this.options.stopRecognition());
      strip.append(text, stop);
      root.append(strip);
    } else if (loose.length) {
      const strip = element("div", "le-strip le-strip--info");
      strip.append(bindLocalizedText(element("span", "", ""), "还有 {{v0}} 块没分进任何部位。识图模型可以看着 TA 的样子帮你分，只分这些，已有的分组不动", { v0: loose.length }));
      const go = bindLocalizedText(button("", "secondary"), "让识图模型分");
      go.addEventListener("click", () => this.options.recognize());
      strip.append(go);
      root.append(strip);
    }
    if (recognition && !recognition.running && recognition.message) {
      root.append(element("p", "behavior-help", recognition.message));
    }

    const list = element("div", "le-groups");
    groups.forEach((group, index) => list.append(this.row(group, index, groups)));
    if (loose.length) list.append(this.row({ id: UNGROUPED, name: "没分组", drawableIds: loose.flatMap((unit) => unit.drawableIds) }, -1, groups));
    root.append(list);
    const footer = element("div", "le-groups__footer");
    footer.append(bindLocalizedText(element("p", "behavior-help", ""), "导入时按图层名字先分一遍；模型作者在 Cubism 里写了标签的，按标签分。分错了直接改：展开一组，在左边点 TA 身上把那块加进来或去掉。"));
    if (this.options.manifest.partGroups) {
      const reset = bindLocalizedText(button("", "le-link le-link--muted"), "按名字重新自动分一遍");
      reset.addEventListener("click", () => {
        if (!window.confirm(t("用图层名字重新分组？你改过的分组会被替换，保存前都可以「放弃修改」。"))) return;
        delete this.options.manifest.partGroups;
        this.expanded = null;
        this.options.changed();
        this.render();
      });
      footer.append(reset);
    }
    root.append(footer);
    this.syncPicking(groups);
  }

  private row(group: ModelPartGroup, index: number, groups: ModelPartGroup[]): HTMLElement {
    const { capability } = this.options;
    const ungrouped = group.id === UNGROUPED;
    const units = ungrouped ? ungroupedUnits(groups, capability) : unitsOfGroup(group, capability);
    const open = this.expanded === group.id;
    const wrapper = element("div", "le-group");
    wrapper.classList.toggle("open", open);
    const line = element("div", "le-group__line");
    const dot = element("span", "le-group__dot");
    dot.style.background = ungrouped ? "transparent" : partGroupColor(index);
    if (ungrouped) dot.classList.add("le-group__dot--empty");
    line.append(dot);

    if (this.renaming === group.id && !ungrouped) {
      const input = document.createElement("input");
      input.type = "text";
      input.value = group.name;
      input.maxLength = MAX_PART_GROUP_NAME_LENGTH;
      bindLocalizedAttribute(input, "aria-label", "分组名字");
      const commit = () => {
        if (this.renaming !== group.id) return;
        this.renaming = null;
        const name = input.value.trim();
        if (name && name !== group.name) this.save(groups.map((item) => item.id === group.id ? { ...item, name } : item));
        else this.render();
      };
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") commit();
        if (event.key === "Escape") { this.renaming = null; this.render(); }
      });
      input.addEventListener("blur", commit);
      line.append(input);
      queueMicrotask(() => { input.focus(); input.select(); });
    } else {
      const name = element("span", "le-group__name", group.name);
      name.append(bindLocalizedText(element("small", "", ""), "{{v0}} 块", { v0: units.length }));
      line.append(name);
    }
    const toggle = bindLocalizedText(button("", "le-link"), (open ? "收起" : "展开"));
    toggle.addEventListener("click", () => { this.expanded = open ? null : group.id; this.render(); });
    line.append(toggle);
    if (!ungrouped) {
      const rename = bindLocalizedText(button("", "le-link le-link--muted"), "改名");
      rename.addEventListener("click", () => { this.renaming = group.id; this.render(); });
      const remove = button("×", "le-group__delete");
      bindLocalizedAttribute(remove, "aria-label", "删除分组 {{v0}}", { v0: group.name });
      bindLocalizedAttribute(remove, "title", "删除分组，里面的块变成没分组");
      remove.addEventListener("click", () => this.save(groups.filter((item) => item.id !== group.id)));
      line.append(rename, remove);
    }
    wrapper.append(line);
    wrapper.addEventListener("mouseenter", () => this.options.highlight(group.drawableIds));
    wrapper.addEventListener("mouseleave", () => this.options.highlight([]));
    if (open) wrapper.append(this.detail(group, units, groups, ungrouped));
    return wrapper;
  }

  private detail(group: ModelPartGroup, units: TextureEditUnit[], groups: ModelPartGroup[], ungrouped: boolean): HTMLElement {
    const detail = element("div", "le-group__detail");
    const pieces = element("div", "le-pieces");
    for (const unit of units.slice(0, THUMBNAIL_LIMIT)) {
      const piece = element("span", "le-piece");
      piece.title = unit.label;
      piece.append(this.locator.thumbnail(unit, 64));
      piece.addEventListener("mouseenter", () => this.options.highlight(unit.drawableIds));
      if (!ungrouped) {
        const remove = button("×", "le-piece__remove");
        bindLocalizedAttribute(remove, "aria-label", "把 {{v0}} 移出{{v1}}", { v0: unit.label, v1: group.name });
        remove.addEventListener("click", () => this.save(removeUnitsFromGroups(groups, [unit])));
        piece.append(remove);
      }
      pieces.append(piece);
    }
    if (units.length > THUMBNAIL_LIMIT) pieces.append(bindLocalizedText(element("span", "le-pieces__more", ""), "…共 {{v0}} 块", { v0: units.length }));
    if (!ungrouped) pieces.append(bindLocalizedText(element("span", "le-pieces__pick", ""), "在左边点着加"));
    detail.append(pieces);
    if (ungrouped) {
      detail.append(bindLocalizedText(element("p", "behavior-help", ""), "展开别的分组后，在左边点这些块就能把它们加进去。"));
      return detail;
    }
    const byName = element("div", "le-by-name");
    const label = bindLocalizedText(element("label", "", ""), "按名字加");
    const input = document.createElement("input");
    input.type = "text";
    bindLocalizedAttribute(input, "placeholder", "图层或部件名，比如 hair、袖");
    input.value = this.query;
    const id = `group-name-${group.id}`;
    input.id = id;
    label.htmlFor = id;
    const members = new Set(group.drawableIds);
    const matches = () => unitsMatchingName(this.options.capability, this.query)
      .filter((unit) => !unit.drawableIds.every((drawable) => members.has(drawable)));
    const hint = element("span", "le-by-name__hint", "");
    const add = bindLocalizedText(button("", "secondary"), "加进来");
    const refresh = () => {
      const found = matches();
      bindLocalizedText(hint, "{{copy}}", () => ({ copy: (this.query.trim() ? t("名字里带「{{v0}}」的还有 {{v1}} 块", { v0: this.query.trim(), v1: found.length }) : "") }));
      add.disabled = found.length === 0;
      this.options.highlight(found.flatMap((unit) => unit.drawableIds));
    };
    input.addEventListener("input", () => { this.query = input.value; refresh(); });
    add.addEventListener("click", () => {
      const found = matches();
      this.query = "";
      this.save(moveUnitsToGroup(groups, group.id, found));
    });
    byName.append(label, input, hint, add);
    refresh();
    detail.append(byName);
    return detail;
  }

  private syncPicking(groups: ModelPartGroup[]): void {
    const target = groups.find((group) => group.id === this.expanded);
    if (!target) {
      this.options.setPicking(null);
      return;
    }
    this.options.setPicking((drawableIds) => {
      const top = drawableIds[0];
      if (!top) return;
      const current = this.groups();
      const owner = current.find((group) => group.drawableIds.includes(top));
      // Only the clicked layer moves: a texture can be shared with other parts.
      // Clicking a layer already in the open group takes it out again.
      this.save(owner?.id === target.id
        ? current.map((group) => group.id === target.id ? { ...group, drawableIds: group.drawableIds.filter((id) => id !== top) } : group)
        : moveDrawablesToGroup(current, target.id, [top]));
    });
  }
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(text: string, className: string): HTMLButtonElement {
  const node = element("button", className, text);
  node.type = "button";
  return node;
}
