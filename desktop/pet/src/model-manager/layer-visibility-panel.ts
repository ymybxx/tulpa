import { t } from "../i18n";
import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import type { DefaultAppearanceSettings, ModelLayerGroup, NativeModelLayer } from "../shared/contracts";
import { normalizeLayerIds } from "../shared/default-appearance";

type LayerSettings = Pick<DefaultAppearanceSettings, "hiddenDrawableIds" | "layerGroups">;
export interface LayerVisibilityPanel {
  element: HTMLElement;
  pick(ids: readonly string[]): void;
  dispose(): void;
  reset(): void;
}
const PAGE_SIZE = 40;

export function renderLayerVisibilityPanel(
  initial: LayerSettings,
  layers: readonly NativeModelLayer[],
  changed: (settings: LayerSettings) => void,
  highlight: (ids: readonly string[]) => void,
): LayerVisibilityPanel {
  const root = document.createElement("section"); root.className = "layer-visibility-panel";
  const heading = document.createElement("h4"); bindLocalizedText(heading, "部位显示");
  const help = document.createElement("p"); help.className = "behavior-help";
  bindLocalizedText(help, "取消勾选以隐藏部位；重新勾选恢复模型原本的显示。点击左侧预览可定位图层，悬停列表可高亮，包括手动隐藏的部位。");
  const hidden = new Set(initial.hiddenDrawableIds ?? []);
  let groups = structuredClone(initial.layerGroups ?? []);
  const selected = new Set<string>();
  let picked: Set<string> | null = null, page = 0;
  const byId = new Map(layers.map(layer => [layer.id, layer]));
  for (const id of [...hidden, ...groups.flatMap(group => group.drawableIds)]) {
    if (!byId.has(id)) byId.set(id, { id, name: id, partId: null, partName: "当前模型未找到", visible: false });
  }
  const all = [...byId.values()];
  const groupList = document.createElement("div"); groupList.className = "layer-groups";
  const toolbar = document.createElement("div"); toolbar.className = "layer-toolbar";
  const search = document.createElement("input"); search.type = "search"; bindLocalizedAttribute(search, "placeholder", "搜索部位名称或图层编号");
  bindLocalizedAttribute(search, "aria-label", "搜索部位图层");
  const filter = document.createElement("select"); bindLocalizedAttribute(filter, "aria-label", "部位图层筛选");
  for (const [value, name] of [["all", "全部图层"], ["hidden", "已隐藏"], ["selected", "已选取"]]) filter.add(new Option(name, value));
  const parts = new Map(layers.filter(layer => layer.partId).map(layer => [layer.partId!, layer.partName]));
  for (const [id, name] of parts) filter.add(new Option(name, `part:${id}`));
  const clearPick = bindLocalizedText(button("", () => { picked = null; page = 0; highlight([]); render(); }), "显示全部图层");
  toolbar.append(search, filter, clearPick);
  const selection = document.createElement("div"); selection.className = "layer-selection-actions";
  const count = document.createElement("span");
  const chooseAll = bindLocalizedText(button("", () => { for (const layer of matching()) selected.add(layer.id); highlight([...selected]); render(); }), "选取筛选结果");
  const clearSelection = bindLocalizedText(button("", () => { selected.clear(); highlight([]); render(); }), "清除选取");
  const hideSelection = bindLocalizedText(button("", () => changeVisibility([...selected], false)), "隐藏选中");
  const showSelection = bindLocalizedText(button("", () => changeVisibility([...selected], true)), "显示选中");
  selection.append(count, chooseAll, clearSelection, hideSelection, showSelection);
  const groupEditor = document.createElement("div"); groupEditor.className = "layer-group-editor";
  const groupName = document.createElement("input"); bindLocalizedAttribute(groupName, "placeholder", "分组名称，例如帽子"); groupName.maxLength = 80;
  bindLocalizedAttribute(groupName, "aria-label", "新部位分组名称");
  const createGroup = bindLocalizedText(button("", () => {
    const name = groupName.value.trim();
    if (!name || !selected.size) return;
    groups.push({ id: crypto.randomUUID(), name, drawableIds: normalizeLayerIds([...selected]) });
    groupName.value = ""; save();
  }), "保存为分组");
  groupEditor.append(groupName, createGroup);
  const list = document.createElement("div"); list.className = "model-layer-list";
  const pager = document.createElement("div"); pager.className = "layer-pager";
  const previous = bindLocalizedText(button("", () => { page--; render(); }), "上一页");
  const next = bindLocalizedText(button("", () => { page++; render(); }), "下一页");
  const pageLabel = document.createElement("span"); pager.append(previous, pageLabel, next);
  const reset = bindLocalizedText(button("", () => { hidden.clear(); selected.clear(); highlight([]); save(); }), "恢复原始显示");
  const status = document.createElement("p"); status.className = "behavior-help layer-status"; status.setAttribute("role", "status");
  root.append(heading, help, groupList, toolbar, selection, groupEditor, status, list, pager, reset);

  function matching(): NativeModelLayer[] {
    const query = search.value.trim().toLocaleLowerCase();
    return all.filter(layer => (!picked || picked.has(layer.id)) &&
      (filter.value !== "hidden" || hidden.has(layer.id)) &&
      (filter.value !== "selected" || selected.has(layer.id)) &&
      (!filter.value.startsWith("part:") || layer.partId === filter.value.slice(5)) &&
      (!query || `${layer.name} ${layer.partName} ${layer.id}`.toLocaleLowerCase().includes(query)));
  }
  function save(): void {
    changed({ hiddenDrawableIds: [...hidden], layerGroups: structuredClone(groups) });
    render();
  }
  function changeVisibility(ids: readonly string[], visible: boolean): void {
    for (const id of ids) visible ? hidden.delete(id) : hidden.add(id);
    selected.clear(); highlight([]); save();
  }
  function selectGroup(group: ModelLayerGroup): void {
    selected.clear(); for (const id of group.drawableIds) selected.add(id);
    picked = new Set(group.drawableIds); page = 0; filter.value = "all"; search.value = "";
    highlight(group.drawableIds); render();
  }
  function renderGroups(): void {
    groupList.replaceChildren();
    for (const group of groups) {
      const row = document.createElement("div"); row.className = "model-layer-group";
      const show = document.createElement("input"); show.type = "checkbox";
      const hiddenCount = group.drawableIds.filter(id => hidden.has(id)).length;
      show.checked = hiddenCount === 0; show.indeterminate = hiddenCount > 0 && hiddenCount < group.drawableIds.length;
      bindLocalizedAttribute(show, "aria-label", "显示分组 {{v0}}", { v0: group.name });
      show.addEventListener("change", () => changeVisibility(group.drawableIds, show.checked));
      const name = document.createElement("input"); name.type = "text"; name.value = group.name; name.maxLength = 80;
      bindLocalizedAttribute(name, "aria-label", "重命名分组 {{v0}}", { v0: group.name });
      name.addEventListener("change", () => { group.name = name.value.trim() || group.name; save(); });
      const locate = bindLocalizedText(button("", () => selectGroup(group)), "选取 {{v0}} 个图层", { v0: group.drawableIds.length });
      bindLocalizedAttribute(locate, "aria-label", "选取分组 {{v0}}", { v0: group.name });
      const remove = bindLocalizedText(button("", () => { groups = groups.filter(item => item.id !== group.id); save(); }), "删除分组");
      bindLocalizedAttribute(remove, "aria-label", "删除分组 {{v0}}", { v0: group.name });
      row.append(show, name, locate, remove); groupList.append(row);
      row.addEventListener("mouseenter", () => highlight(group.drawableIds));
      row.addEventListener("mouseleave", () => highlight([]));
    }
  }
  function render(): void {
    const matches = matching();
    const pages = Math.max(1, Math.ceil(matches.length / PAGE_SIZE));
    page = Math.max(0, Math.min(pages - 1, page));
    bindLocalizedText(count, "已选取 {{v0}} 个", { v0: selected.size });
    clearSelection.disabled = hideSelection.disabled = showSelection.disabled = selected.size === 0;
    createGroup.disabled = selected.size === 0 || !groupName.value.trim();
    chooseAll.disabled = matches.length === 0;
    reset.disabled = hidden.size === 0;
    clearPick.hidden = picked === null;
    bindLocalizedText(status, "{{copy}}", () => ({ copy: (layers.length ? t("{{v0}} 个已隐藏 · {{v1}}{{v2}} 个图层", { v0: hidden.size, v1: (picked ? t("此处 ") : t("匹配 ")), v2: matches.length }) : t("模型预览加载后可选择部位。")) }));
    list.replaceChildren();
    for (const layer of matches.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)) {
      const row = document.createElement("div"); row.className = "model-layer-row"; row.classList.toggle("selected", selected.has(layer.id));
      const choose = button(layer.name, () => {
        selected.has(layer.id) ? selected.delete(layer.id) : selected.add(layer.id);
        highlight([...selected]); render();
      });
      choose.className = "model-layer-name"; choose.setAttribute("aria-pressed", String(selected.has(layer.id)));
      bindLocalizedAttribute(choose, "aria-label", "选取图层 {{v0}}", { v0: layer.id });
      const meta = document.createElement("small"); meta.textContent = layer.partName;
      choose.append(meta);
      const label = document.createElement("label"); label.className = "model-layer-show";
      const show = document.createElement("input"); show.type = "checkbox"; show.checked = !hidden.has(layer.id);
      bindLocalizedAttribute(show, "aria-label", "显示图层 {{v0}}", { v0: layer.id });
      show.addEventListener("change", () => changeVisibility([layer.id], show.checked));
      label.append(show, bindLocalizedText(document.createElement("span"), "显示"));
      row.append(choose, label); list.append(row);
      row.addEventListener("mouseenter", () => highlight([layer.id]));
      row.addEventListener("mouseleave", () => highlight([]));
      choose.addEventListener("focus", () => highlight([layer.id]));
      choose.addEventListener("blur", () => highlight([]));
    }
    pageLabel.textContent = `${page + 1} / ${pages}`;
    previous.disabled = page === 0; next.disabled = page + 1 >= pages;
    pager.hidden = pages === 1;
    renderGroups();
  }
  search.addEventListener("input", () => { page = 0; highlight([]); render(); });
  filter.addEventListener("change", () => { page = 0; highlight([]); render(); });
  groupName.addEventListener("input", () => { createGroup.disabled = !selected.size || !groupName.value.trim(); });
  render();
  return {
    element: root,
    pick(ids) {
      picked = new Set(ids); selected.clear(); if (ids[0]) selected.add(ids[0]);
      search.value = ""; filter.value = "all"; page = 0;
      highlight(ids[0] ? [ids[0]] : []); render();
      root.scrollIntoView({ block: "nearest" });
    },
    dispose: () => highlight([]),
    reset: () => { hidden.clear(); groups = []; selected.clear(); highlight([]); save(); },
  };
}

function button(label: string, action: () => void): HTMLButtonElement {
  const value = document.createElement("button"); value.type = "button"; value.textContent = label;
  value.addEventListener("click", action); return value;
}
