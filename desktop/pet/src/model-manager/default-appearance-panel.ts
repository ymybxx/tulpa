import { bindLocalizedAttribute, bindLocalizedText } from "../i18n/dom";
import type { DefaultAppearanceSettings, ModelOutfit, NativeModelLayer, NativeModelParameter, PetManifest } from "../shared/contracts";
import { MAX_OUTFITS, MAX_OUTFIT_NAME_LENGTH, normalizeDefaultAppearance, sameAppearance } from "../shared/default-appearance";
import { renderDefaultParameterPanel } from "./default-parameter-panel";
import { renderLayerVisibilityPanel, type LayerVisibilityPanel } from "./layer-visibility-panel";

export interface DefaultAppearancePanelOptions {
  roots: { suits: HTMLElement; toggles: HTMLElement; params: HTMLElement };
  manifest: PetManifest;
  /** The draft changed; the preview and dirty state should follow. */
  changed: () => void;
  parameters?: readonly NativeModelParameter[];
  layers?: readonly NativeModelLayer[];
  highlight?: (ids: readonly string[]) => void;
  deleteToggle?: (assetId: string) => void;
  /** 「用参数建一个」 opens the 参数 pane, where the values are chosen. */
  openParams?: () => void;
  /** Writes the pinned values as a new model toggle. */
  saveAsToggle?: (name: string, values: Record<string, number>) => Promise<void>;
  /** A whole 套装 was applied, so every pane has to be drawn again. */
  redraw?: () => void;
}

/**
 * 装扮: 套装 on top, then 开关 (the model's own expression files as switches) and
 * 参数 (pinned native parameters, with layer hiding for parts no parameter reaches).
 * Everything is kept by asset or parameter ID.
 */
export function renderDefaultAppearancePanel(options: DefaultAppearancePanelOptions): LayerVisibilityPanel {
  const { roots, manifest, changed } = options;
  const redraw = options.redraw ?? (() => {});
  roots.suits.replaceChildren();
  roots.toggles.replaceChildren();
  roots.params.replaceChildren();
  const initial = normalizeDefaultAppearance(manifest.defaultAppearance, manifest.expressionAssets);
  let ids = initial.expressionAssetIds;
  let parameterValues = initial.parameterValues;
  let hiddenDrawableIds = initial.hiddenDrawableIds;
  let layerGroups = initial.layerGroups;
  const switches = new Map<string, HTMLButtonElement>();
  const current = (): DefaultAppearanceSettings => normalizeDefaultAppearance(
    { expressionAssetIds: ids, parameterValues, hiddenDrawableIds, layerGroups },
    manifest.expressionAssets,
  );

  const suits = renderSuits(roots.suits, manifest, current, redraw, changed);
  const pinBar = document.createElement("div");
  const sync = () => {
    for (const [id, button] of switches) {
      const on = ids.includes(id);
      button.setAttribute("aria-checked", String(on));
      button.closest(".le-toggle")!.classList.toggle("on", on);
    }
    suits.sync();
    syncPinBar();
  };
  const save = () => {
    manifest.defaultAppearance = current();
    sync();
    changed();
  };

  // ---------- 开关 ----------
  const assets = Object.values(manifest.expressionAssets);
  const head = paneHead("开关", "模型作者做好的，点一下就开或关；名字看不懂，点名字就能改");
  const search = document.createElement("input");
  search.type = "search";
  bindLocalizedAttribute(search, "placeholder", "搜索开关");
  bindLocalizedAttribute(search, "aria-label", "搜索开关");
  search.hidden = assets.length <= 9;
  const create = document.createElement("button");
  create.type = "button";
  create.className = "secondary";
  bindLocalizedText(create, "用参数建一个");
  create.addEventListener("click", () => options.openParams?.());
  head.append(search, create);
  const grid = document.createElement("div");
  grid.className = "le-toggles";
  for (const asset of assets) {
    const card = document.createElement("div");
    card.className = "le-toggle";
    card.dataset.search = `${asset.name} ${asset.expression}`.toLocaleLowerCase();
    const name = document.createElement("button");
    name.type = "button";
    name.className = "le-toggle__name";
    name.textContent = asset.name;
    bindLocalizedAttribute(name, "title", "点一下改名");
    bindLocalizedAttribute(name, "aria-label", "给开关 {{v0}} 改名", { v0: asset.name });
    name.addEventListener("click", () => {
      const input = document.createElement("input");
      input.type = "text";
      input.className = "le-toggle__rename";
      input.value = asset.name;
      input.maxLength = 40;
      bindLocalizedAttribute(input, "aria-label", "开关名字");
      let done = false;
      const finish = (commit: boolean) => {
        if (done) return;
        done = true;
        const next = input.value.trim().replace(/\s+/g, " ");
        if (commit && next && next !== asset.name) {
          asset.name = next;
          name.textContent = next;
          button.setAttribute("aria-label", next);
          card.dataset.search = `${next} ${asset.expression}`.toLocaleLowerCase();
          bindLocalizedAttribute(remove, "aria-label", "删除开关 {{v0}}", { v0: next });
          sync();
          changed();
        }
        input.replaceWith(name);
      };
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") finish(true);
        if (event.key === "Escape") finish(false);
      });
      input.addEventListener("blur", () => finish(true));
      name.replaceWith(input);
      input.focus();
      input.select();
    });
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "switch");
    button.setAttribute("aria-label", asset.name);
    button.className = "le-switch";
    button.append(Object.assign(document.createElement("span"), { className: "le-switch__knob" }));
    button.addEventListener("click", () => {
      ids = ids.filter((id) => id !== asset.id);
      if (button.getAttribute("aria-checked") !== "true") ids.push(asset.id);
      save();
    });
    switches.set(asset.id, button);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "le-toggle__delete";
    bindLocalizedAttribute(remove, "aria-label", "删除开关 {{v0}}", { v0: asset.name });
    bindLocalizedAttribute(remove, "title", "删除这个开关");
    remove.textContent = "×";
    remove.addEventListener("click", () => options.deleteToggle?.(asset.id));
    card.append(name, button, remove);
    grid.append(card);
  }
  const noMatch = bindLocalizedText(textNode("p", "models-empty", ""), "没有叫这个名字的开关");
  noMatch.hidden = true;
  search.addEventListener("input", () => {
    const query = search.value.trim().toLocaleLowerCase();
    let shown = 0;
    for (const card of grid.querySelectorAll<HTMLElement>(".le-toggle")) {
      card.hidden = Boolean(query) && !card.dataset.search!.includes(query);
      if (!card.hidden) shown += 1;
    }
    noMatch.hidden = shown > 0;
  });
  if (switches.size === 0) {
    grid.append(bindLocalizedText(textNode("p", "models-empty", ""), "这个模型没有自带开关。可以在「参数」里调好，再存成一个开关。"));
  }
  const hint = bindLocalizedText(textNode("p", "behavior-help", ""), "开着的开关就是 TA 平时的样子，说话时的表情会叠加在上面。同一部位有多个开关时，后打开的优先。「用参数建一个」：挑几个参数、定好数值，存成新开关。");
  roots.toggles.append(head, grid, noMatch, hint);

  // ---------- 参数 ----------
  const parameters = options.parameters ?? [];
  const paramsHead = paneHead("参数", parameters.length ? `${parameters.length} 个 · 拖动就在左边预览` : "预览加载好后显示");
  pinBar.className = "le-pinbar";
  const advanced = renderDefaultParameterPanel(parameterValues, parameters, (values) => {
    parameterValues = values;
    save();
  });
  // In 装扮 the parameters are the pane itself, not a folded extra.
  advanced.element.classList.add("le-params");
  advanced.element.open = true;
  const visibility = renderLayerVisibilityPanel(initial, options.layers ?? [], settings => {
    hiddenDrawableIds = settings.hiddenDrawableIds; layerGroups = settings.layerGroups; save();
  }, options.highlight ?? (() => {}));
  const fallback = document.createElement("details");
  fallback.className = "le-layer-fallback";
  const summary = document.createElement("summary");
  bindLocalizedText(summary, "参数控制不了的部位，在左边预览里直接点选隐藏");
  fallback.append(summary, visibility.element);
  roots.params.append(paramsHead, pinBar, advanced.element, fallback);

  let naming = false;
  function syncPinBar(): void {
    const values = parameterValues ?? {};
    const count = Object.keys(values).length;
    pinBar.hidden = count === 0 || !options.saveAsToggle;
    if (pinBar.hidden || naming) return;
    const text = bindLocalizedText(textNode("span", "", ""), "固定了 {{v0}} 个参数：这就是 TA 平时的样子，也可以存成一个开关", { v0: count });
    const button = document.createElement("button");
    button.type = "button";
    button.className = "accent";
    bindLocalizedText(button, "存成开关");
    button.addEventListener("click", () => {
      naming = true;
      askName(pinBar, "开关名字，比如「戴耳麦」", 20, async (name) => {
        await options.saveAsToggle!(name, { ...values });
      }, () => { naming = false; syncPinBar(); });
    });
    pinBar.replaceChildren(text, button);
  }

  sync();
  return visibility;
}

function renderSuits(
  root: HTMLElement,
  manifest: PetManifest,
  current: () => DefaultAppearanceSettings,
  redraw: () => void,
  changed: () => void,
): { sync: () => void } {
  const label = bindLocalizedText(textNode("span", "le-suits__label", ""), "套装");
  const list = document.createElement("div");
  list.className = "le-suits__list";
  const status = textNode("span", "le-suits__status", "");
  const save = document.createElement("button");
  save.type = "button";
  save.className = "secondary";
  bindLocalizedText(save, "存成套装");
  const outfits = (): ModelOutfit[] => manifest.outfits ?? [];
  const setOutfits = (next: ModelOutfit[]) => {
    if (next.length) manifest.outfits = next;
    else delete manifest.outfits;
  };

  const sync = () => {
    const appearance = current();
    list.replaceChildren();
    for (const outfit of outfits()) {
      const on = sameAppearance(outfit.appearance, appearance);
      const pill = document.createElement("span");
      pill.className = "le-suit";
      pill.classList.toggle("on", on);
      const apply = document.createElement("button");
      apply.type = "button";
      apply.textContent = outfit.name;
      apply.setAttribute("aria-pressed", String(on));
      bindLocalizedAttribute(apply, "title", "换上「{{v0}}」，双击改名", { v0: outfit.name });
      apply.addEventListener("click", () => {
        manifest.defaultAppearance = structuredClone(outfit.appearance);
        changed();
        redraw();
      });
      apply.addEventListener("dblclick", () => {
        askName(root, "套装名字", MAX_OUTFIT_NAME_LENGTH, async (name) => {
          setOutfits(outfits().map((item) => item.id === outfit.id ? { ...item, name } : item));
          changed();
        }, restore, outfit.name);
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "le-suit__delete";
      remove.textContent = "×";
      bindLocalizedAttribute(remove, "aria-label", "删除套装 {{v0}}", { v0: outfit.name });
      remove.addEventListener("click", () => {
        setOutfits(outfits().filter((item) => item.id !== outfit.id));
        changed();
        sync();
      });
      pill.append(apply, remove);
      list.append(pill);
    }
    if (!outfits().length) list.append(bindLocalizedText(textNode("span", "le-suits__empty", ""), "把现在的样子存起来，之后一键换回"));
    bindLocalizedText(status, "当前：{{v0}}", { v0: describeAppearance(appearance, manifest) });
    save.disabled = outfits().length >= MAX_OUTFITS;
  };
  const restore = () => {
    root.replaceChildren(label, list, status, save);
    sync();
  };
  save.addEventListener("click", () => {
    askName(root, "套装名字，比如「上班」", MAX_OUTFIT_NAME_LENGTH, async (name) => {
      setOutfits([...outfits(), { id: crypto.randomUUID(), name, appearance: current() }]);
      changed();
    }, restore);
  });
  restore();
  return { sync };
}

function describeAppearance(appearance: DefaultAppearanceSettings, manifest: PetManifest): string {
  const names = appearance.expressionAssetIds.map((id) => manifest.expressionAssets[id]?.name).filter(Boolean);
  const parts = [
    names.length ? `开着${names.slice(0, 2).join("、")}${names.length > 2 ? ` 等 ${names.length} 个` : ""}` : "",
    Object.keys(appearance.parameterValues ?? {}).length ? `固定 ${Object.keys(appearance.parameterValues!).length} 个参数` : "",
    appearance.hiddenDrawableIds?.length ? `隐藏 ${appearance.hiddenDrawableIds.length} 个图层` : "",
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "模型原样";
}

/** Swaps the container's content for a one-line name field until saved or cancelled. */
function askName(
  root: HTMLElement,
  placeholder: string,
  maxLength: number,
  submit: (name: string) => Promise<void>,
  done: () => void,
  value = "",
): void {
  const form = document.createElement("form");
  form.className = "le-name-form";
  const input = document.createElement("input");
  input.type = "text";
  bindLocalizedAttribute(input, "placeholder", placeholder);
  input.maxLength = maxLength;
  input.value = value;
  input.setAttribute("aria-label", placeholder);
  const ok = document.createElement("button");
  ok.type = "submit";
  ok.className = "accent";
  bindLocalizedText(ok, "保存");
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "secondary";
  bindLocalizedText(cancel, "取消");
  const error = textNode("span", "le-name-form__error", "");
  const previous = [...root.childNodes];
  const close = () => { root.replaceChildren(...previous); done(); };
  cancel.addEventListener("click", close);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const name = input.value.trim();
    if (!name) { bindLocalizedText(error, "起个名字吧"); return; }
    ok.disabled = true;
    void submit(name).then(close, (reason: unknown) => {
      ok.disabled = false;
      error.textContent = reason instanceof Error ? reason.message : String(reason);
    });
  });
  form.append(input, ok, cancel, error);
  root.replaceChildren(form);
  input.focus();
}

function paneHead(title: string, note: string): HTMLElement {
  const head = document.createElement("div");
  head.className = "le-pane__head";
  head.append(bindLocalizedText(textNode("h3", "", ""), title), bindLocalizedText(textNode("span", "", ""), note));
  return head;
}

function textNode<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text: string): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}
