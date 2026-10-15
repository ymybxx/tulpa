import { bindLocalizedText } from "../i18n/dom";
import type { FacialMotionStyle } from "../shared/facial-bindings";


/**
 * The shared frame of a 说话表情 part editor: two cards side by side —
 * 「这个部位」 (name, the model parameter it moves, its values) on the left and
 * 「决策模型怎么判断」 (the steps Jev picks from) on the right — then one footer
 * line with a quiet action on the left and the single 保存 on the right.
 */
export function editorCard(title: string, action: HTMLElement | null, ...children: HTMLElement[]): HTMLElement {
  const card = element("section", "fbe-card");
  const head = element("div", "fbe-card__head");
  head.append(bindLocalizedText(element("h4"), title));
  if (action) head.append(element("span", "fbe-grow"), action);
  card.append(head, ...children);
  return card;
}

/** A small caption above its control. */
export function editorField(caption: string, ...controls: Array<HTMLElement | string>): HTMLElement {
  const field = element("div", "fbe-field");
  field.append(bindLocalizedText(element("span", "fbe-caption"), caption), ...controls);
  return field;
}

export function editorBody(left: HTMLElement, right: HTMLElement): HTMLElement {
  const body = element("div", "fbe-body");
  body.append(left, right);
  return body;
}

export function editorFooter(quiet: HTMLElement | null, feedback: HTMLElement, primary: HTMLElement): HTMLElement {
  const footer = element("div", "fbe-foot");
  if (quiet) footer.append(quiet);
  if (!feedback.className.includes("fbe-feedback")) feedback.className = `${feedback.className} fbe-feedback`.trim();
  footer.append(feedback, primary);
  return footer;
}

export function editorButton(text: string, key: string, className: string, click: () => void): HTMLButtonElement {
  const button = element("button", className);
  button.type = "button"; bindLocalizedText(button, text); button.dataset[key] = "";
  button.addEventListener("click", click);
  return button;
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

const MOTION_CHOICES: ReadonlyArray<{ value: FacialMotionStyle; text: string; hint: string }> = [
  { value: "hold", text: "保持姿势", hint: "移到选中的那一档，这句话说完前一直保持。" },
  { value: "swing", text: "来回摆动", hint: "在平常和选中的那一档之间来回摆几下再停，比如挥手。" },
  { value: "pulse", text: "做一下就回来", hint: "移到选中的那一档，停一下就回到平常，比如点头、耸肩。" },
];

/** How the part moves once Jev picks its step. */
export function motionPicker(value: FacialMotionStyle = "hold"): { element: HTMLElement; read(): FacialMotionStyle } {
  let current = value;
  const wrap = element("div", "fbe-motion");
  const switcher = element("div", "le-seg"); switcher.dataset.bindingMotion = "";
  const hint = element("span", "fbe-note");
  const buttons = MOTION_CHOICES.map(choice => {
    const button = element("button"); button.type = "button"; bindLocalizedText(button, choice.text);
    button.dataset.motion = choice.value;
    button.addEventListener("click", () => { current = choice.value; show(); });
    return button;
  });
  const show = () => {
    for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.motion === current));
    bindLocalizedText(hint, MOTION_CHOICES.find(choice => choice.value === current)!.hint);
  };
  switcher.append(...buttons); wrap.append(switcher, hint); show();
  return { element: wrap, read: () => current };
}
