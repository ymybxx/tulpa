import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import type { IdleChannel, IdlePreset, NaturalIdleSettings } from "../shared/contracts";
import type { ModelParameterDescriptor } from "../shared/lipsync-adapter";
import { IDLE_CHANNELS, idlePresetSettings, normalizeNaturalIdleSettings } from "../shared/natural-idle-settings";

const CHANNEL_LABELS: Record<IdleChannel, string> = { headX: "转头", headY: "抬头低头", headZ: "歪头", bodyX: "身体转向", bodyY: "身体俯仰", bodyZ: "身体侧倾", eyeX: "视线左右", eyeY: "视线上下", eyeLOpen: "左眼开合", eyeROpen: "右眼开合" };

type Choice = "off" | "legacy" | IdlePreset;
const CHOICES: ReadonlyArray<{ id: Choice; name: string; about: string }> = [
  { id: "off", name: "不动", about: "只有模型自带的呼吸和眨眼" },
  { id: "gentle", name: "轻柔", about: "偶尔动一下，幅度小" },
  { id: "natural", name: "自然", about: "时不时转头，看看别处" },
  { id: "active", name: "活跃", about: "动得多，幅度也大" },
];
// Older looks may still use the first version; it stays pickable only while it is the one in use.
const LEGACY = { id: "legacy" as const, name: "原来的晃动", about: "随机转头和歪头" };

/**
 * 待机: what TA does while not talking. One row of choices that apply at
 * once (the preview on the left plays them); fine-tuning for the chosen one
 * below; model-specific fixes folded away at the bottom.
 */
export function renderNaturalIdlePanel(value: NaturalIdleSettings | undefined, parameters: readonly ModelParameterDescriptor[], changed: (value: NaturalIdleSettings) => void): HTMLElement {
  const root = document.createElement("section");
  root.className = "natural-idle-panel";
  let settings = normalizeNaturalIdleSettings(value);
  const update = () => { settings = normalizeNaturalIdleSettings(settings); changed(structuredClone(settings)); };
  const current = (): Choice => !settings.enabled ? "off" : settings.expressive?.preset ?? "legacy";
  const choose = (choice: Choice) => {
    if (choice === current()) return;
    if (choice === "off") settings.enabled = false;
    else if (choice === "legacy") settings = { enabled: true, intensity: settings.intensity, speed: settings.speed };
    else {
      // A new style starts from its own defaults but keeps fixes made for this model.
      const bindings = settings.expressive?.bindings, directions = settings.expressive?.bodyDirections;
      settings = idlePresetSettings(choice);
      if (bindings) settings.expressive!.bindings = bindings;
      if (directions) settings.expressive!.bodyDirections = directions;
    }
    update();
    draw();
  };
  const draw = () => {
    root.replaceChildren();
    const picked = current();
    const choices = document.createElement("div");
    choices.className = "idle-choices";
    for (const choice of picked === "legacy" ? [...CHOICES, LEGACY] : CHOICES) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "idle-choice";
      button.setAttribute("aria-pressed", String(choice.id === picked));
      const name = document.createElement("b");
      bindLocalizedText(name, choice.name);
      const about = document.createElement("small");
      bindLocalizedText(about, choice.about);
      button.append(name, about);
      button.addEventListener("click", () => choose(choice.id));
      choices.append(button);
    }
    root.append(bindLocalizedText(heading(""), "TA 没在说话时"), choices);
    if (picked === "off") return;

    const tune = document.createElement("div");
    tune.className = "idle-tune";
    tune.append(bindLocalizedText(heading(""), "细调"));
    if (!settings.expressive) {
      tune.append(
        slider("晃动多大", settings.intensity, 0, 2, 0.05, "小", "大", n => { settings.intensity = n; update(); }),
        slider("快慢", settings.speed, 0.5, 1.5, 0.05, "慢", "快", n => { settings.speed = n; update(); }),
      );
      root.append(tune);
      return;
    }
    const expressive = settings.expressive;
    tune.append(
      slider("头动多大", settings.intensity, 0, 2, 0.05, "小", "大", n => { settings.intensity = n; update(); }),
      slider("身体动多大", expressive.bodyIntensity, 0, 2, 0.05, "小", "大", n => { expressive.bodyIntensity = n; update(); }),
      slider("眼神多灵活", expressive.eyeActivity, 0, 2, 0.05, "少", "多", n => { expressive.eyeActivity = n; update(); }),
      slider("快慢", settings.speed, 0.5, 1.5, 0.05, "慢", "快", n => { settings.speed = n; update(); }),
      slider("大动作多不多", expressive.largeMotionProbability, 0, 1, 0.05, "少", "多", n => { expressive.largeMotionProbability = n; update(); }),
      slider("发呆多不多", expressive.pauseProbability, 0, 0.8, 0.05, "少", "多", n => { expressive.pauseProbability = n; update(); }),
    );
    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "secondary idle-reset";
    bindLocalizedText(reset, "恢复这个风格的默认值");
    reset.addEventListener("click", () => {
      const bindings = expressive.bindings, directions = expressive.bodyDirections;
      settings = idlePresetSettings(expressive.preset);
      if (bindings) settings.expressive!.bindings = bindings;
      if (directions) settings.expressive!.bodyDirections = directions;
      update();
      draw();
    });
    tune.append(reset);
    root.append(tune, fixes(expressive));
  };

  /** For models whose parts the automatic setup gets wrong; most people never open it. */
  const fixes = (expressive: NonNullable<NaturalIdleSettings["expressive"]>): HTMLElement => {
    const details = document.createElement("details");
    details.className = "idle-fixes";
    const summary = document.createElement("summary");
    bindLocalizedText(summary, "某个部位不动、动反了，或者动得太猛？");
    details.append(summary);
    const intro = document.createElement("p");
    bindLocalizedText(intro, "一般不用改。下面按部位调最大角度；哪个部位不动或方向反了，给它选模型里对应的参数。");
    details.append(intro);
    for (const [key, name] of [["yawRange", "转头最大角度"], ["pitchRange", "抬头低头最大角度"], ["rollRange", "歪头最大角度"]] as const) {
      details.append(slider(name, expressive[key], 0, key === "yawRange" ? 28 : 25, 0.5, "小", "大", n => { expressive[key] = n; update(); }));
    }
    details.append(
      slider("身体跟着头动的延迟", expressive.followDelay, 0.08, 0.8, 0.02, "紧跟", "慢半拍", n => { expressive.followDelay = n; update(); }),
      slider("大动作之间至少隔多久", expressive.reverseInterval, 0, 10, 0.5, "0 秒", "10 秒", n => { expressive.reverseInterval = n; update(); }),
    );
    const blink = document.createElement("input");
    blink.type = "checkbox";
    blink.checked = expressive.blink;
    blink.addEventListener("change", () => { expressive.blink = blink.checked; update(); });
    details.append(field("自己眨眼", blink));
    for (const channel of IDLE_CHANNELS) {
      const select = document.createElement("select");
      select.append(bindLocalizedText(option("auto", ""), "自动找"), bindLocalizedText(option("none", ""), "这个部位不动"), ...parameters.map(p => option(p.id, p.name ? `${p.name} · ${p.id}` : p.id)));
      const bound = expressive.bindings?.[channel];
      if (typeof bound === "string" && !parameters.some(p => p.id === bound)) select.append(bindLocalizedText(option(bound, ""), "{{v0}}（这个模型里没有）", { v0: bound }));
      select.value = bound === undefined ? "auto" : bound === null ? "none" : bound;
      select.addEventListener("change", () => {
        const bindings = expressive.bindings ??= {};
        if (select.value === "auto") delete bindings[channel];
        else bindings[channel] = select.value === "none" ? null : select.value;
        update();
      });
      details.append(field(`${CHANNEL_LABELS[channel]}用哪个参数`, select));
    }
    for (const [axis, name] of [["x", "身体转向"], ["y", "身体俯仰"], ["z", "身体侧倾"]] as const) {
      const direction = document.createElement("select");
      direction.append(bindLocalizedText(option("auto", ""), "自动"), bindLocalizedText(option("1", ""), "正着"), bindLocalizedText(option("-1", ""), "反过来"));
      direction.value = String(expressive.bodyDirections?.[axis] ?? "auto");
      direction.addEventListener("change", () => {
        const values = expressive.bodyDirections ??= {};
        if (direction.value === "auto") delete values[axis];
        else values[axis] = Number(direction.value) as -1 | 1;
        update();
      });
      details.append(field(`${name}的方向`, direction));
    }
    return details;
  };
  draw();
  return root;
}

function heading(text: string): HTMLElement {
  const node = document.createElement("h4");
  node.className = "idle-heading";
  node.textContent = text;
  return node;
}
function option(value: string, text: string): HTMLOptionElement { const node = document.createElement("option"); node.value = value; node.textContent = text; return node; }
function field(text: string, input: HTMLElement): HTMLLabelElement {
  const node = document.createElement("label");
  node.className = "idle-field";
  const name = document.createElement("span");
  bindLocalizedText(name, text);
  node.append(name, input);
  return node;
}
/** A slider with plain words at both ends instead of a number. */
function slider(text: string, value: number, min: number, max: number, step: number, low: string, high: string, changed: (value: number) => void): HTMLElement {
  const range = document.createElement("input");
  range.type = "range";
  range.min = String(min);
  range.max = String(max);
  range.step = String(step);
  range.value = String(value);
  bindLocalizedAttribute(range, "aria-label", text);
  range.addEventListener("input", () => {
    if (Number.isFinite(range.valueAsNumber)) changed(range.valueAsNumber);
  });
  const row = document.createElement("div");
  row.className = "idle-slider";
  const lowText = document.createElement("small");
  bindLocalizedText(lowText, low);
  const highText = document.createElement("small");
  bindLocalizedText(highText, high);
  row.append(lowText, range, highText);
  return field(text, row);
}
