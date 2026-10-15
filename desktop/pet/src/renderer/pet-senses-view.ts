import { t, getLocale, subscribeLocale } from "../i18n";
import "../ui/theme.css";
import "./pet-senses.css";
import { followScene } from "../ui/scene";
import type { PetChatApi, PetChatState, Rect } from "../shared/pet-chat";
import type { SenseKey, SenseProblem, SenseView } from "../shared/pet-senses";

const ICONS: Record<SenseKey, string> = {
  screen: "M3 5h18v11H3zM8 20h8M12 16v4",
  audio: "M4 10v4h4l5 4V6L8 10zM16 9a4 4 0 0 1 0 6M19 6a8 8 0 0 1 0 12",
  microphone: "M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zM5 11a7 7 0 0 0 14 0M12 18v3",
  inner: "M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z",
};
/** Perceiving the world, then 内心: the capsule and panel show the two groups apart. */
const ORDER: SenseKey[] = ["screen", "audio", "microphone", "inner"];
const PAUSE = "M8 5v14M16 5v14";
const PLAY = "M7 5l12 7-12 7z";
const SVG = "http://www.w3.org/2000/svg";

const get = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

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

/**
 * 屏幕感知 / 电脑音频感知 / 麦克风收音 | 内心 above the pet's head. Shown only
 * while the pointer is on the pet; the panel holds the switches, permission
 * fixes and 暂停陪伴.
 */
export function mountSenses(api: PetChatApi): (state: PetChatState) => void {
  const root = get("senses-root");
  followScene(root);
  const capsule = get("senses");
  const slot = get("sense-slot");
  const panel = get("sense-panel");
  const confirm = get("sense-confirm");
  const rows = get("sense-rows");
  const note = get("sense-note");
  const status = get<HTMLButtonElement>("model-status");
  const ownerPrompt = get<HTMLButtonElement>("owner-voice-prompt");
  ownerPrompt.title = t("只加入最近几句没认出的话，确认是你自己说的再点");
  ownerPrompt.addEventListener("click", () => api.senseAction({ type: "owner-voice-add" }));
  let panelState: PetChatState["sensePanel"] = "closed";
  status.addEventListener("click", () => {
    if (status.dataset.state === "failed") api.senseAction({ type: "retry-models" });
  });
  let rendered = "";
  let ownerVoiceMissed = false;
  let sleeping: { wakeWord: string } | undefined;
  const asleepLine = (word: string): string => t("睡着了 · 喊“{{word}}”叫醒我", { word });

  const buttons = ORDER.map((key) => {
    if (key === "inner") capsule.append(element("span", "sense-divider"));
    const button = element("button", "sense-dot-button");
    button.type = "button";
    button.dataset.key = key;
    button.append(icon(ICONS[key]), element("span", "sense-dot"));
    button.addEventListener("click", () => api.senseAction({ type: "panel", panel: panelState === "closed" ? "open" : "closed" }));
    capsule.append(button);
    return button;
  });

  api.onPointer((point) => {
    for (const button of buttons) {
      const box = button.getBoundingClientRect();
      const over = point !== null && point.x >= box.left && point.x < box.right && point.y >= box.top && point.y < box.bottom;
      button.classList.toggle("sense-dot-button--hover", over);
    }
  });

  get("sense-pause").addEventListener("click", () => {
    if (get("sense-pause").dataset.paused === "true") api.senseAction({ type: "resume" });
    else api.senseAction({ type: "panel", panel: "confirm-pause" });
  });
  get("sense-settings").addEventListener("click", () => api.senseAction({ type: "settings" }));
  const usage = get<HTMLButtonElement>("sense-usage");
  usage.title = t("看看每项功能用了多少");
  usage.addEventListener("click", () => api.senseAction({ type: "settings", section: "usage" }));
  get("sense-keep").addEventListener("click", () => api.senseAction({ type: "pause", closeSenses: false }));
  get("sense-close-all").addEventListener("click", () => api.senseAction({ type: "pause", closeSenses: true }));
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || panelState === "closed") return;
    api.senseAction({ type: "panel", panel: panelState === "confirm-pause" ? "open" : "closed" });
  });

  const row = (sense: SenseView): HTMLElement => {
    const item = element("div", `sense-row sense-row--${sense.status}${sense.key === "inner" ? " sense-row--apart" : ""}`);
    const badge = element("span", "sense-row__icon");
    badge.append(icon(ICONS[sense.key]));
    const text = element("span", "sense-row__text");
    const heading = element("span", "sense-row__heading");
    heading.append(element("span", "sense-row__title", t(sense.title)));
    if (sense.fix && sense.status === "waiting") {
      // The badge itself reopens System Settings, which may have been closed.
      const waiting = element("button", "sense-badge sense-badge--waiting sense-badge--button", t(sense.label));
      waiting.type = "button";
      waiting.title = t("打开系统设置");
      waiting.addEventListener("click", () => api.senseAction({ type: "fix", kind: sense.fix! }));
      heading.append(waiting);
    } else if (sense.fix) {
      const fix = element("button", "sense-fix", t("{{label}} · 去打开", { label: t(sense.label) }));
      fix.type = "button";
      fix.addEventListener("click", () => api.senseAction({ type: "fix", kind: sense.fix! }));
      heading.append(fix);
    } else if (sense.status === "partial") {
      // On, but waiting for 内心: offer that, not a status word.
      const enable = element("button", "sense-fix", t("打开内心"));
      enable.type = "button";
      enable.addEventListener("click", () => api.senseAction({ type: "toggle", key: "inner" }));
      heading.append(enable);
    }
    // Plain on/off needs no word: the switch beside it says so.
    const asleep = sense.key === "microphone" && sleeping;
    text.append(heading, element("span", "sense-row__detail", asleep ? asleepLine(asleep.wakeWord) : t(sense.detail)));
    if (sense.key === "microphone" && ownerVoiceMissed) {
      const add = element("button", "sense-owner-add", t("刚才有几句没认出是你 · 加入声纹"));
      add.type = "button";
      add.title = t("只加入最近几句没认出的话，确认是你自己说的再点");
      add.addEventListener("click", () => api.senseAction({ type: "owner-voice-add" }));
      text.append(add);
    }
    const toggle = element("button", "sense-switch");
    toggle.type = "button";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-checked", String(sense.on));
    toggle.setAttribute("aria-label", t(sense.title));
    toggle.append(element("span", "sense-switch__knob"));
    toggle.addEventListener("click", () => api.senseAction({ type: "toggle", key: sense.key }));
    item.append(badge, text, toggle);
    return item;
  };

  const problem = (item: SenseProblem): HTMLElement => {
    const line = element("div", "sense-problem");
    line.append(element("span", "sense-problem__text", t(item.text)));
    const fix = element("button", "sense-fix", t("去设置"));
    fix.type = "button";
    fix.addEventListener("click", () => api.senseAction({ type: "settings", section: item.section }));
    line.append(fix);
    return line;
  };

  const place = (node: HTMLElement, rect: Rect): void => {
    node.style.left = `${rect.x}px`; node.style.top = `${rect.y}px`;
    node.style.width = `${rect.width}px`; node.style.height = `${rect.height}px`;
  };

  return (state) => {
    ownerPrompt.title = t("只加入最近几句没认出的话，确认是你自己说的再点");
    usage.title = t("看看每项功能用了多少");
    const readiness = state.readiness;
    status.hidden = !readiness;
    if (readiness) {
      place(status, state.layout.status);
      status.dataset.state = readiness.state;
      get("model-status-text").textContent = t(readiness.text);
      status.setAttribute("aria-label", t(readiness.detail) ? `${t(readiness.text)}：${t(readiness.detail)}` : t(readiness.text));
      status.title = t(readiness.detail);
      status.disabled = readiness.state !== "failed";
    }
    // Right where TA ignored the owner, not only inside the capsule's panel.
    ownerPrompt.hidden = Boolean(readiness) || !state.senses?.ownerVoiceMissed;
    if (!ownerPrompt.hidden) place(ownerPrompt, state.layout.status);
    const senses = state.senses;
    const visible = Boolean(senses) && state.entryVisible;
    capsule.hidden = !visible;
    panelState = visible ? state.sensePanel : "closed";
    slot.hidden = panelState === "closed";
    if (!senses) return;
    // Shrinks as a whole beside a small TA; CSS zoom scales the box too.
    const hud = state.layout.hud ?? 1;
    capsule.style.zoom = String(hud);
    const box = state.layout.capsule;
    place(capsule, { x: box.x / hud, y: box.y / hud, width: box.width / hud, height: box.height / hud });
    place(slot, state.layout.panel);
    slot.dataset.placement = state.layout.panelPlacement;
    panel.hidden = panelState !== "open";
    confirm.hidden = panelState !== "confirm-pause";
    sleeping = senses.sleeping;
    senses.senses.forEach((sense, index) => {
      const button = buttons[index]!;
      button.dataset.status = sense.status;
      // Asleep TA looks like idle TA: the microphone says what wakes TA.
      const asleep = sense.key === "microphone" && sleeping;
      button.dataset.asleep = String(Boolean(asleep));
      const label = asleep ? asleepLine(asleep.wakeWord) : t(sense.label);
      button.setAttribute("aria-label", `${t(sense.title)}：${label}`);
      // Native tooltips do not show while tulpa is in the background.
      button.dataset.tip = `${t(sense.title)} · ${label}`;
      button.setAttribute("aria-expanded", String(panelState !== "closed"));
    });
    get("sense-title").textContent = senses.companionPaused ? t("陪伴已暂停") : t("陪伴中");
    const pause = get("sense-pause");
    pause.dataset.paused = String(senses.companionPaused);
    get("sense-pause-label").textContent = senses.companionPaused ? t("恢复陪伴") : t("暂停陪伴");
    get("sense-pause-icon").setAttribute("d", senses.companionPaused ? PLAY : PAUSE);
    note.hidden = !state.senseNote;
    note.textContent = state.senseNote;
    usage.hidden = !senses.usage;
    if (senses.usage) {
      const background = senses.usage.background.map((label) => t(label));
      usage.textContent = t("今天用了 {{tokens}} token", { tokens: senses.usage.today })
        + (background.length ? t(" · 后台在用：{{features}}", { features: background.join(getLocale() === "en" ? ", " : "、") }) : "");
    }
    // Pausing stops speaking up only: say what keeps using the model.
    const running = senses.senses.filter((sense) => sense.on && sense.key !== "microphone").map((sense) => t(sense.title));
    get("sense-confirm-background").textContent = running.length
      ? t("只暂停的话，{{features}}还会在后台用模型。", { features: running.join(getLocale() === "en" ? ", " : "、") }) : "";
    ownerVoiceMissed = Boolean(senses.ownerVoiceMissed);
    const problems = senses.problems ?? [];
    const signature = JSON.stringify([getLocale(), senses.senses, ownerVoiceMissed, sleeping, problems]);
    if (signature !== rendered) {
      rendered = signature;
      rows.replaceChildren(...problems.map(problem), ...senses.senses.map(row));
    }
  };
}
