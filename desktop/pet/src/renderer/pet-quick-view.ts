import { t, getLocale, subscribeLocale } from "../i18n";
import type { PetChatApi, PetChatState } from "../shared/pet-chat";
import type { PetState } from "../shared/contracts";
import { QUICK_TILES, QUICK_TILE_KEYS, type QuickPanelView, type QuickTileKey } from "../shared/pet-quick";
import type { SenseKey } from "../shared/pet-senses";
import { stateLabel } from "../chat/model";

const ICONS: Record<QuickTileKey, string> = {
  speaker: "M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4",
  microphone: "M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zM5 11a7 7 0 0 0 14 0M12 18v3",
  screen: "M3 5h18v11H3zM8 20h8M12 16v4",
  audio: "M4 10v4h4l5 4V6L8 10zM16 9a4 4 0 0 1 0 6M19 6a8 8 0 0 1 0 12",
  inner: "M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z",
  companion: "M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18zM8.5 14a4 4 0 0 0 7 0M9 9.5h.01M15 9.5h.01",
  bubbles: "M4 5h16v11H9l-5 4zM8 9h8M8 12h5",
  passthrough: "M5 3l14 7-6 2-2 6z",
  chat: "M3 5h12v8H7l-4 3zM9 16v1h8l4 3V9h-3",
  settings: "M12 9a3 3 0 1 0 0 6a3 3 0 0 0 0-6zM12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3M4.9 19.1L7 17M17 7l2.1-2.1",
};
const PERSON = "M12 12a4 4 0 1 0 0-8a4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0";
const WINDOW = "M3 5h18v14H3zM3 9h18";
const COMPOSE = "M4 5h16v11H9l-5 4zM8 10h8";
const LOOK = "M8 3L3 6l2 4 3-1v12h8V9l3 1 2-4-5-3a4 4 0 0 1-8 0z";
const HIDE = "M3 3l18 18M10.6 10.6a2 2 0 0 0 2.8 2.8M9.9 5.1A10 10 0 0 1 12 5c5 0 9 5 9 7a11 11 0 0 1-2.6 3.4M6.6 6.6C4.4 8 3 10.4 3 12c0 2 4 7 9 7a9 9 0 0 0 4.4-1.2";
const CHEVRON = "M9 6l6 6-6 6";
const ADJUST = "M4 7h10M18 7h2M4 17h4M12 17h8M16 5v4M10 15v4";
const SENSES: readonly QuickTileKey[] = ["screen", "audio", "microphone", "inner"];
const SVG = "http://www.w3.org/2000/svg";

function icon(path: string): SVGSVGElement {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const shape = document.createElementNS(SVG, "path");
  shape.setAttribute("d", path);
  svg.append(shape);
  return svg;
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(className: string, content: Array<Node | string>, onClick: () => void): HTMLButtonElement {
  const node = element("button", className);
  node.type = "button";
  node.append(...content);
  node.addEventListener("click", onClick);
  return node;
}

/** On/off and "needs attention" for one tile, from the panel view and the senses snapshot. */
function tileState(key: QuickTileKey, quick: QuickPanelView, state: PetChatState): { on: boolean; warn: boolean } {
  if (SENSES.includes(key)) {
    const sense = state.senses?.senses.find((item) => item.key === key as SenseKey);
    return { on: Boolean(sense?.on), warn: Boolean(sense && sense.status !== "on" && sense.status !== "off") };
  }
  if (key === "companion") return { on: state.senses ? !state.senses.companionPaused : false, warn: false };
  if (key === "speaker") return { on: quick.speaker, warn: false };
  if (key === "bubbles") return { on: quick.bubbles, warn: false };
  return { on: false, warn: false };
}

/**
 * Right-click on the pet: who TA is, the user's own switch grid, then 聊两句 ·
 * 换形象 · 先藏起来. 编辑面板 turns the grid into remove/add/drag-to-sort.
 */
export function mountQuickPanel(api: PetChatApi): (state: PetChatState) => void {
  const root = document.getElementById("quick") as HTMLElement;
  let looksOpen = false;
  let dragged: QuickTileKey | null = null;
  let rendered = "";
  let latest: PetChatState | null = null;

  const setTiles = (tiles: QuickTileKey[]) => api.quickAction({ type: "tiles", tiles });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && latest?.quick) api.quickAction({ type: "close" });
  });

  function tile(key: QuickTileKey, quick: QuickPanelView, state: PetChatState, mode: "use" | "placed" | "available"): HTMLButtonElement {
    const { title: sourceTitle, kind } = QUICK_TILES[key];
    const title = t(sourceTitle);
    const { on, warn } = tileState(key, quick, state);
    const node = button(`pq-tile${mode === "available" ? " pq-tile--available" : ""}`, [icon(ICONS[key]), element("span", "", title)], () => {
      if (mode === "use") api.quickAction({ type: "tile", key });
      else if (mode === "placed") setTiles(quick.tiles.filter((item) => item !== key));
      else setTiles([...quick.tiles, key]);
    });
    node.dataset.key = key;
    if (kind === "switch" && mode === "use") node.setAttribute("aria-pressed", String(on));
    if (mode === "use" && on) node.classList.add("pq-tile--on");
    if (warn && mode === "use") node.append(element("span", "pq-warn"));
    if (mode !== "use") {
      node.append(element("span", `pq-badge pq-badge--${mode === "placed" ? "remove" : "add"}`, mode === "placed" ? "−" : "+"));
      node.setAttribute("aria-label", t(mode === "placed" ? "移除{{title}}" : "添加{{title}}", { title }));
      node.draggable = true;
      node.addEventListener("dragstart", (event) => { dragged = key; event.dataTransfer?.setData("text/plain", key); });
      node.addEventListener("dragend", () => { dragged = null; });
    }
    if (mode === "placed") {
      node.addEventListener("dragover", (event) => { if (dragged && dragged !== key) event.preventDefault(); });
      node.addEventListener("drop", (event) => {
        event.preventDefault();
        if (!dragged || dragged === key) return;
        const rest = quick.tiles.filter((item) => item !== dragged);
        rest.splice(rest.indexOf(key), 0, dragged);
        setTiles(rest);
      });
    }
    return node;
  }

  function render(state: PetChatState): void {
    const quick = state.quick!;
    root.replaceChildren();
    root.classList.toggle("pq--editing", quick.editing);
    if (quick.editing) {
      const head = element("div", "pq-head pq-head--edit");
      head.append(element("b", "pq-title", t("编辑快捷面板")),
        button("pq-done", [t("完成")], () => api.quickAction({ type: "edit", editing: false })));
      const grid = element("div", "pq-grid");
      grid.append(...quick.tiles.map((key) => tile(key, quick, state, "placed")));
      // Dropping an available tile on the grid's empty space adds it at the end.
      grid.addEventListener("dragover", (event) => { if (dragged) event.preventDefault(); });
      grid.addEventListener("drop", (event) => {
        if (!dragged || event.defaultPrevented) return;
        event.preventDefault();
        setTiles([...quick.tiles.filter((item) => item !== dragged), dragged]);
      });
      if (!quick.tiles.length) grid.append(element("p", "pq-empty", t("都拿掉了，下面点 + 放回来")));
      const available = QUICK_TILE_KEYS.filter((key) => !quick.tiles.includes(key));
      root.append(head, grid);
      if (available.length) {
        const more = element("div", "pq-grid");
        more.append(...available.map((key) => tile(key, quick, state, "available")));
        root.append(element("p", "pq-hint", t("还可以放进来 · 拖动可排序")), more);
      }
      return;
    }

    const head = element("div", "pq-head");
    const who = element("div", "pq-who");
    who.append(element("b", "", quick.character ?? "TA"),
      element("small", "", t(stateLabel(quick.petState as PetState | null, !state.senses?.companionPaused))));
    const avatar = element("span", "pq-avatar");
    avatar.append(icon(PERSON));
    // Labelled like the pill beside the pet: an icon alone said nothing.
    const main = button("pq-main-button", [icon(WINDOW), element("span", "", t("主窗口"))], () => api.quickAction({ type: "main" }));
    head.append(avatar, who, main);
    root.append(head);

    if (quick.tiles.length) {
      const grid = element("div", "pq-grid");
      grid.append(...quick.tiles.map((key) => tile(key, quick, state, "use")));
      root.append(grid);
    }

    const rows = element("div", "pq-rows");
    const current = quick.looks.find((look) => look.id === quick.lookId);
    const lookRow = button("pq-row", [icon(LOOK), element("span", "pq-row__label", t("换形象")), element("small", "", current?.name ?? ""), icon(CHEVRON)], () => {
      looksOpen = !looksOpen;
      render(state);
    });
    lookRow.setAttribute("aria-expanded", String(looksOpen));
    lookRow.disabled = quick.looks.length === 0;
    rows.append(button("pq-row", [icon(COMPOSE), element("span", "pq-row__label", t("聊两句"))], () => api.quickAction({ type: "compose" })), lookRow);
    if (looksOpen) {
      const list = element("div", "pq-looks");
      list.setAttribute("role", "radiogroup");
      list.append(...quick.looks.map((look) => {
        const option = button("pq-look", [element("span", "pq-look__dot"), element("span", "", look.name)], () => api.quickAction({ type: "look", id: look.id }));
        option.setAttribute("role", "radio");
        option.setAttribute("aria-checked", String(look.id === quick.lookId));
        return option;
      }));
      rows.append(list);
    }
    rows.append(button("pq-row", [icon(HIDE), element("span", "pq-row__label", t("先藏起来"))], () => api.quickAction({ type: "hide" })));
    root.append(rows);

    const foot = element("div", "pq-foot");
    foot.append(button("pq-link", [icon(ADJUST), t("编辑面板")], () => { looksOpen = false; api.quickAction({ type: "edit", editing: true }); }),
      button("pq-link", [t("退出")], () => api.quickAction({ type: "quit" })));
    root.append(foot);
  }

  return (state) => {
    latest = state;
    const quick = state.quick;
    root.hidden = !quick;
    if (!quick) { looksOpen = false; rendered = ""; return; }
    const { rect, edge } = quick;
    root.style.left = `${rect.x}px`;
    root.style.width = `${rect.width}px`;
    root.style.maxHeight = `${rect.height}px`;
    // The edge at the click stays put; the panel grows away from it.
    root.style.top = edge === "top" ? `${rect.y}px` : "";
    root.style.bottom = edge === "bottom" ? `${window.innerHeight - rect.y - rect.height}px` : "";
    const signature = JSON.stringify([getLocale(), quick.editing, quick.tiles, quick.speaker, quick.bubbles, quick.character, quick.petState,
      quick.looks, quick.lookId, state.senses]);
    if (signature === rendered) return;
    rendered = signature;
    render(state);
  };
}
