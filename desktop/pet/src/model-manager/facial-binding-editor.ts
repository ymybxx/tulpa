import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import type { NativeModelParameter } from "../shared/contracts";
import { AUTOMATIC_FACIAL_IDS, initialManualBinding, MAX_PART_OUTPUTS, validateDescriptionOverrides, validateNativeBinding, validateParameterSettings, type FacialDecisionRule, type FacialParameterSettings, type ManualParameterBinding } from "../shared/facial-bindings";
import { outputValue, STANDARD_PARAMETERS, toNativeValue, type ParameterBinding } from "../facial-demo/parameter-bindings";
import { facialDecisionEditor, type DecisionColumns } from "./facial-decision-editor";
import { editorBody, editorButton, editorCard, editorField, editorFooter, motionPicker } from "./facial-editor-layout";

/**
 * One part's settings, in the order a person thinks about them: its name,
 * which of the model's parameters it moves — several, as in VTube Studio —
 * and how far each follows, how Jev names its steps (a table of where each
 * step lands, with 「看看」 to try one), and — folded away — the note Jev
 * reads. Whether Jev drives the part stays on the row's switch.
 */
export function facialBindingEditor(options: {
  binding: ParameterBinding;
  native: NativeModelParameter[];
  blocked: string[];
  /** The author's own wiring of this part (their VTube Studio settings), the full reach of each parameter it names. */
  reference?: readonly ManualParameterBinding[];
  save(bindings: ManualParameterBinding[] | null, settings: FacialParameterSettings, description: string, decision: FacialDecisionRule): Promise<void>;
  cancel(): void;
  /** Takes the part off this look's list; it can be added back from 已删除的部位. */
  remove(): Promise<void>;
  /** Shows these model parameter values in the preview. */
  preview(values: Record<string, number>): void;
}): HTMLElement {
  const { binding, native, blocked } = options;
  const definition = STANDARD_PARAMETERS.find(value => value.id === binding.id)!;
  const root = element("div", "facial-binding-editor");
  root.dataset.bindingEditor = binding.id;
  // Untouched, an automatic part stays automatic; any change saves the list below.
  let changed = false;

  const name = input("text"); name.maxLength = 30; name.value = binding.label; name.dataset.bindingLabel = "";
  name.placeholder = definition.label;

  // ① Which parameters, VTube Studio style: each with the values the part's
  // weakest, usual and strongest steps land on.
  const choices = native.filter(parameter => !AUTOMATIC_FACIAL_IDS.has(parameter.id) && !blocked.includes(parameter.id)
    && [parameter.minimum, parameter.maximum].every(Number.isFinite) && parameter.minimum < parameter.maximum);
  const atLower = definition.neutral === definition.min, atUpper = definition.neutral === definition.max;
  const list = element("div", "fbe-outputs");
  const rows: OutputRow[] = [];
  const addButton = element("button", "le-link"); addButton.type = "button"; bindLocalizedText(addButton, "＋ 再接一个参数");
  addButton.dataset.bindingAdd = "";

  // A parameter reaches, at 100%, the author's ends when they wired it, else its own minimum and maximum.
  const reach = (parameter: NativeModelParameter): ManualParameterBinding =>
    options.reference?.find(item => item.parameterId === parameter.id) ?? initialManualBinding(binding.id, parameter);
  function addRow(initial: ManualParameterBinding | null): void {
    const row = outputRow(choices, atLower, atUpper, initial, reach, () => { changed = true; refresh(); }, () => {
      rows.splice(rows.indexOf(row), 1); row.element.remove(); changed = true; refresh();
    });
    rows.push(row); list.append(row.element);
  }
  const initial = binding.manualBindings?.length
    ? binding.outputs.map(output => output.manual!)
    : binding.outputs.map(output => output.manual
      ?? { parameterId: output.parameterId, lowerValue: binding.min, neutralValue: binding.neutral, upperValue: binding.max });
  if (initial.length) initial.forEach(addRow); else addRow(null);
  addButton.addEventListener("click", () => { addRow(null); changed = true; refresh(); });

  // The part at one value: every parameter where it lands.
  const landing = (value: number): Record<string, number> => Object.fromEntries(current().map(output =>
    [output.parameterId, outputValue(value, binding, { parameterId: output.parameterId, manual: output })]));
  // Drag the whole part from its weakest to its strongest step; every parameter follows.
  // A step's 「看看」 below moves it there too, so 跟随多少 is seen on that step.
  const slider = input("range"); slider.dataset.bindingPreview = "";
  slider.min = String(binding.min); slider.max = String(binding.max); slider.step = String((binding.max - binding.min) / 200);
  slider.value = String(binding.neutral);
  const sliderValue = element("span", "fbe-chip");
  const tryRow = element("div", "fbe-try"); tryRow.append(slider, sliderValue);
  const look = (value: number) => {
    slider.value = String(value);
    sliderValue.textContent = format(value);
    options.preview(landing(value));
  };
  slider.addEventListener("input", () => look(Number(slider.value)));

  const multiplier = input("number"); multiplier.min = "0"; multiplier.step = "0.1";
  multiplier.value = String(binding.multiplier); multiplier.dataset.bindingMultiplier = "";
  const amplitude = element("div", "fbe-amp");
  amplitude.append(Object.assign(element("span", "fbe-times"), { textContent: "×" }), multiplier,
    bindLocalizedText(Object.assign(element("span", "fbe-note"), { textContent: "" }), "1 是原样，大于 1 动得更明显"));

  const motion = motionPicker(binding.motion);

  // ② How Jev names the steps, as a table of where each step lands on every parameter.
  const current = (): ManualParameterBinding[] => rows.flatMap(row => row.read() ?? []);
  const columns = (): DecisionColumns => {
    const outputs = current();
    if (!outputs.length) return { names: [], values: value => [toNativeValue(value, binding)] };
    return { names: outputs.map(output => shortName(native.find(item => item.id === output.parameterId), output.parameterId)),
      values: value => outputs.map(output => outputValue(value, binding, { parameterId: output.parameterId, manual: output })) };
  };
  const decision = facialDecisionEditor({ mode: binding.decisionMode, levels: binding.levels, anchors: binding.anchors,
    minimum: binding.min, maximum: binding.max, columns: columns(), look, ...binding.binaryCriteria });
  function refresh() {
    decision.setColumns(columns());
    addButton.disabled = rows.length >= MAX_PART_OUTPUTS;
    for (const row of rows) row.setRemovable(rows.length > 1);
    // Whatever step is showing follows the change, so dragging 跟随多少 is seen at once.
    look(Number(slider.value));
  }

  // ③ The note Jev reads, folded away: the steps above usually say enough.
  const description = element("textarea"); description.rows = 3; description.maxLength = 500;
  description.value = binding.description; description.dataset.bindingDescription = "";
  const more = element("details", "facial-binding-more");
  const summary = element("summary"); bindLocalizedText(summary, "补充说明（一般不用改）");
  more.append(summary, bindLocalizedText(Object.assign(element("p", "facial-binding-hint"), { textContent: "" }), "决策模型判断时会读这段话，说清这个部位代表什么、往哪边是什么意思。"), description);

  const feedback = element("p", "facial-binding-feedback"); feedback.setAttribute("role", "status");
  const buttons: HTMLButtonElement[] = [];
  const read = (): ManualParameterBinding[] | null => {
    const outputs = rows.map(row => row.read());
    if (outputs.some(output => output === undefined)) throw new Error("请给每个参数填好最弱、平常和最强的值");
    const chosen = outputs.filter((output): output is ManualParameterBinding => !!output);
    if (new Set(chosen.map(output => output.parameterId)).size !== chosen.length) throw new Error("同一个参数只能接一次");
    for (const output of chosen) validateNativeBinding(binding.id, output, native, blocked);
    return chosen.length ? chosen : null;
  };
  /** `shipped` puts this model's wording of the part back to the default. */
  async function save(value: ManualParameterBinding[] | null, shipped = false) {
    try {
      if (!multiplier.value.trim()) throw new Error("请填写幅度");
      if (!shipped && description.value.trim()) validateDescriptionOverrides({ [binding.id]: description.value });
      const rule: FacialDecisionRule = shipped ? { mode: "score", levels: [...definition.levels] } : decision.read();
      const label = name.value.trim();
      const settings = validateParameterSettings({ [binding.id]: {
        enabled: binding.enabled, multiplier: Number(multiplier.value),
        ...label && label !== definition.label ? { label } : {},
        ...motion.read() !== (definition.motion ?? "hold") ? { motion: motion.read() } : {},
      } })[binding.id]!;
      for (const control of buttons) control.disabled = true;
      bindLocalizedText(feedback, "正在保存…");
      await options.save(value, settings, shipped ? definition.description : description.value, rule);
    } catch (error) {
      feedback.textContent = error instanceof Error ? error.message : String(error);
      for (const control of buttons) control.disabled = false;
    }
  }
  const restore = binding.manualBindings?.length
    ? bindLocalizedText(editorButton("", "bindingRemove", "le-link le-link--muted", () => { void save(null); }), "恢复自动对应")
    : null;
  const saveButton = bindLocalizedText(editorButton("", "bindingSave", "accent", () => {
    try { void save(changed ? read() : binding.manualBindings ?? null); }
    catch (error) { feedback.textContent = error instanceof Error ? error.message : String(error); }
  }), "保存");
  // Only a part this model has reworded can go back to the default wording.
  const reworded = binding.description !== definition.description || binding.decisionMode !== "score"
    || binding.levels.length !== definition.levels.length || binding.levels.some((level, index) => level !== definition.levels[index]);
  const resetWording = reworded
    ? bindLocalizedText(editorButton("", "bindingResetWording", "le-link le-link--muted", () => {
      try { void save(changed ? read() : binding.manualBindings ?? null, true); }
      catch (error) { feedback.textContent = error instanceof Error ? error.message : String(error); }
    }), "恢复默认说法")
    : null;
  const remove = bindLocalizedText(editorButton("", "bindingDelete", "le-link fbe-danger", () => {
    void options.remove().catch(error => { feedback.textContent = error instanceof Error ? error.message : String(error); });
  }), "删除这个部位");
  buttons.push(saveButton, remove, ...restore ? [restore] : [], ...resetWording ? [resetWording] : []);
  const quiet = element("div", "fbe-actions");
  quiet.append(...restore ? [restore] : [], remove);
  if ((binding.manualBindings?.length ?? 0) > binding.outputs.length) {
    bindLocalizedText(feedback, "有的参数在模型里已经找不到了，保存时会去掉。");
  }

  const parameters = element("div", "fbe-field");
  parameters.append(list, addButton);
  root.append(
    editorBody(
      editorCard("这个部位", null,
        editorField("名字", name),
        editorField("对应模型的哪些参数", parameters),
        editorField("拖一下看看", tryRow),
        editorField("动作幅度", amplitude),
        editorField("怎么动", motion.element)),
      editorCard("决策模型怎么判断", resetWording, decision.element, more),
    ),
    editorFooter(quiet, feedback, saveButton),
  );
  refresh();
  return root;
}

interface OutputRow {
  element: HTMLElement;
  /** The entry, null when no parameter is chosen, undefined when its values are incomplete. */
  read(): ManualParameterBinding | null | undefined;
  setRemovable(removable: boolean): void;
}

/**
 * One model parameter of the part: which one, how far it follows (both ends
 * drawn in toward 平常 or out to its full reach), flip, remove — and, folded
 * away, its three exact values.
 */
function outputRow(
  choices: NativeModelParameter[], atLower: boolean, atUpper: boolean, initial: ManualParameterBinding | null,
  reach: (parameter: NativeModelParameter) => ManualParameterBinding, changed: () => void, remove: () => void,
): OutputRow {
  const root = element("div", "fbe-output"); root.dataset.bindingOutput = "";
  const select = element("select"); select.dataset.bindingParameter = "";
  const placeholder = element("option"); placeholder.value = ""; bindLocalizedText(placeholder, "选一个参数");
  select.append(placeholder);
  for (const parameter of choices) {
    const option = element("option"); option.value = parameter.id;
    option.textContent = `${parameter.name || parameter.id} · ${parameter.id}（${parameter.minimum}～${parameter.maximum}）`;
    select.append(option);
  }
  const reverse = element("button", "le-link"); reverse.type = "button"; bindLocalizedText(reverse, "反过来");
  bindLocalizedAttribute(reverse, "title", "模型的方向和名字相反时用"); reverse.dataset.bindingReverse = "";
  const drop = element("button", "le-link le-link--muted"); drop.type = "button"; bindLocalizedText(drop, "移除");
  drop.dataset.bindingDrop = "";
  const head = element("div", "fbe-row"); head.append(select, reverse, drop);

  const follow = input("range"); follow.min = "0"; follow.max = "100"; follow.step = "1"; follow.dataset.bindingFollow = "";
  const followValue = element("span", "fbe-chip");
  const followRow = element("div", "fbe-try");
  followRow.append(bindLocalizedText(Object.assign(element("span", "fbe-caption"), { textContent: "" }), "跟随多少"), follow, followValue);

  const range = element("div", "fbe-range");
  const fields = (["lowerValue", "neutralValue", "upperValue"] as const).map((key, index) => {
    const label = element("label");
    label.append(Object.assign(element("span"), { textContent: ["最弱", "平常", "最强"][index]! }));
    const field = input("number"); field.step = "any"; field.dataset.bindingValue = key;
    field.addEventListener("input", () => { showFollow(); changed(); });
    if (index === 1 && (atLower || atUpper)) label.hidden = true;
    label.append(field); range.append(label); return field;
  });
  const [lower, neutral, upper] = fields as [HTMLInputElement, HTMLInputElement, HTMLInputElement];
  const exact = element("details", "fbe-exact");
  exact.append(bindLocalizedText(Object.assign(element("summary"), { textContent: "" }), "精确设置"), range);
  root.append(head, followRow, exact);

  const parameter = () => choices.find(item => item.id === select.value);
  /** How far the ends reach, as a share of the parameter's full reach. */
  function share(): number | null {
    const chosen = parameter();
    if (!chosen) return null;
    const full = reach(chosen), span = Math.abs(full.upperValue - full.lowerValue);
    const ends = [lower.value, upper.value].map(Number);
    if (!span || !ends.every(Number.isFinite)) return null;
    return Math.abs(ends[1]! - ends[0]!) / span;
  }
  function showFollow() {
    const value = share();
    follow.disabled = value === null;
    follow.value = String(Math.round(Math.min(1, value ?? 1) * 100));
    followValue.textContent = value === null ? "—" : `${Math.round(value * 100)}%`;
  }
  // Both ends move toward 平常 by the same share, keeping their direction.
  follow.addEventListener("input", () => {
    const chosen = parameter();
    if (!chosen) return;
    const full = reach(chosen), k = Number(follow.value) / 100;
    const reversed = (Number(upper.value) - Number(lower.value)) * (full.upperValue - full.lowerValue) < 0;
    const [low, high] = reversed ? [full.upperValue, full.lowerValue] : [full.lowerValue, full.upperValue];
    const middle = Number(atLower ? lower.value : atUpper ? upper.value : neutral.value);
    const place = (end: number) => String(round(Math.max(chosen.minimum, Math.min(chosen.maximum, middle + (end - middle) * k))));
    if (!atLower) lower.value = place(low);
    if (!atUpper) upper.value = place(high);
    showFollow(); changed();
  });

  const load = (value: ManualParameterBinding | null) => {
    const chosen = parameter();
    for (const field of fields) field.disabled = !chosen;
    reverse.disabled = !chosen;
    if (chosen) {
      const values = value?.parameterId === chosen.id ? value : reach(chosen);
      fields.forEach(field => {
        field.min = String(chosen.minimum); field.max = String(chosen.maximum);
        field.value = String(values[field.dataset.bindingValue as "lowerValue" | "neutralValue" | "upperValue"]);
      });
    }
    showFollow();
  };
  select.value = initial?.parameterId ?? "";
  load(initial);
  select.addEventListener("change", () => { load(null); changed(); });
  reverse.addEventListener("click", () => { [lower.value, upper.value] = [upper.value, lower.value]; showFollow(); changed(); });
  drop.addEventListener("click", remove);
  return {
    element: root,
    read() {
      if (!select.value) return null;
      if ([lower, upper, ...(!atLower && !atUpper ? [neutral] : [])].some(field => !field.value.trim())) return undefined;
      return { parameterId: select.value, lowerValue: Number(lower.value), upperValue: Number(upper.value),
        neutralValue: Number(atLower ? lower.value : atUpper ? upper.value : neutral.value) };
    },
    setRemovable(removable) { drop.hidden = !removable; },
  };
}

/** A column heading: the author's name for the parameter, else its id without the usual prefix. */
function shortName(parameter: NativeModelParameter | undefined, id: string): string {
  return parameter?.name?.trim() || id.replace(/^Param/u, "");
}

function round(value: number): number { return Math.round(value * 1e4) / 1e4 || 0; }

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function input(type: string): HTMLInputElement {
  const node = document.createElement("input"); node.type = type; return node;
}

function format(value: number): string { return String(Number(value.toFixed(2))); }
