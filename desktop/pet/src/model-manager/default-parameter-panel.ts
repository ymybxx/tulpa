import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import { t } from "../i18n";
import type { NativeModelParameter } from "../shared/contracts";

export function renderDefaultParameterPanel(
  initial: Readonly<Record<string, number>> | undefined,
  parameters: readonly NativeModelParameter[],
  changed: (values: Record<string, number>) => void,
): { element: HTMLDetailsElement; clear: () => void } {
  let values = { ...initial };
  const element = document.createElement("details");
  element.className = "default-parameter-panel";
  const summary = document.createElement("summary");
  const count = document.createElement("span");
  summary.append(bindLocalizedText(document.createElement("span"), "高级参数"), count);
  const body = document.createElement("div");
  body.className = "default-parameter-body";
  const help = document.createElement("p");
  help.className = "behavior-help";
  bindLocalizedText(help, "只固定勾选的原生参数，并优先于所选效果和动作。固定眼睛或嘴巴会影响眨眼、口型；取消勾选后恢复模型与动画控制。部分效果会被其他配饰遮挡。");
  const search = document.createElement("input");
  search.type = "search";
  bindLocalizedAttribute(search, "placeholder", "搜索参数名称或 ID");
  bindLocalizedAttribute(search, "aria-label", "搜索模型原生参数");
  const list = document.createElement("div");
  list.className = "default-parameter-list";
  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "secondary";
  bindLocalizedText(clear, "取消全部参数固定");
  const updateCount = () => {
    const size = Object.keys(values).length;
    if (size) bindLocalizedText(count, "已固定 {{v0}} 项", { v0: size });
    else bindLocalizedText(count, "未固定");
    clear.disabled = size === 0;
  };
  const save = () => { updateCount(); changed({ ...values }); };
  const rows = parameters.filter(p => Number.isFinite(p.minimum) && Number.isFinite(p.maximum) && p.minimum <= p.maximum);
  const known = new Set(rows.map(p => p.id));
  // Keep stale settings visible and removable after a model asset is replaced.
  for (const id of Object.keys(values)) if (!known.has(id)) rows.push({
    id, name: `${id}（当前模型不可用）`, minimum: values[id]!, maximum: values[id]!, defaultValue: values[id]!,
  });
  const draw = () => {
    list.replaceChildren();
    const query = search.value.trim().toLocaleLowerCase();
    const matches = rows.filter(p => `${p.name} ${p.id}`.toLocaleLowerCase().includes(query));
    if (!matches.length) {
      const empty = document.createElement("p");
      empty.className = "behavior-help";
      bindLocalizedText(empty, (rows.length ? "没有匹配的参数" : "模型预览加载完成后，可在这里调整原生参数。"));
      list.append(empty);
    }
    for (const parameter of matches) {
      const { id, name, minimum, maximum, defaultValue } = parameter;
      const row = document.createElement("div");
      row.className = "default-parameter-row";
      const label = document.createElement("label");
      label.className = "default-parameter-label";
      const enabled = document.createElement("input");
      enabled.type = "checkbox";
      enabled.checked = Object.hasOwn(values, id);
      bindLocalizedAttribute(enabled, "aria-label", "固定 {{v0}} ({{v1}})", { v0: name, v1: id });
      const copy = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = name;
      const meta = document.createElement("small");
      bindLocalizedText(meta, "{{v0}} · {{v1}} ～ {{v2}} · 默认 {{v3}}", { v0: id, v1: minimum, v2: maximum, v3: defaultValue });
      copy.append(title, meta);
      label.append(enabled, copy);
      const controls = document.createElement("div");
      controls.className = "default-parameter-controls";
      const range = document.createElement("input");
      range.type = "range";
      const number = document.createElement("input");
      number.type = "number";
      const step = maximum > minimum ? Number(((maximum - minimum) / 1000).toPrecision(2)) : 1;
      for (const input of [range, number]) {
        input.min = String(minimum);
        input.max = String(maximum);
        input.step = input === range ? String(step) : "any";
        bindLocalizedAttribute(input, "aria-label", "{{v0}} ({{v1}}){{v2}}", () => ({ v0: name, v1: id, v2: t(input === range ? "滑块" : "数值") }));
      }
      const reset = document.createElement("button");
      reset.type = "button";
      reset.className = "secondary";
      bindLocalizedText(reset, "默认值");
      bindLocalizedAttribute(reset, "aria-label", "恢复 {{v0}} ({{v1}}) 的默认值", { v0: name, v1: id });
      const refresh = () => {
        const active = Object.hasOwn(values, id);
        const value = Math.max(minimum, Math.min(maximum, values[id] ?? defaultValue));
        enabled.checked = active;
        row.classList.toggle("selected", active);
        for (const input of [range, number]) { input.value = String(value); input.disabled = !active || !known.has(id); }
        reset.disabled = !active || !known.has(id);
      };
      const change = (input: HTMLInputElement) => {
        const next = input.valueAsNumber;
        if (!Object.hasOwn(values, id) || !Number.isFinite(next)) return;
        values[id] = Math.max(minimum, Math.min(maximum, next));
        refresh();
        save();
      };
      enabled.addEventListener("change", () => {
        if (enabled.checked) values[id] = defaultValue;
        else delete values[id];
        refresh(); save();
      });
      range.addEventListener("input", () => change(range));
      number.addEventListener("input", () => change(number));
      number.addEventListener("blur", refresh);
      reset.addEventListener("click", () => { values[id] = defaultValue; refresh(); save(); });
      controls.append(range, number, reset);
      row.append(label, controls);
      refresh();
      list.append(row);
    }
  };
  const clearAll = () => { values = {}; save(); if (element.open) draw(); };
  clear.addEventListener("click", clearAll);
  search.addEventListener("input", draw);
  // Hundreds of native controls are built only when this advanced area is open.
  element.addEventListener("toggle", () => { if (element.open) draw(); else list.replaceChildren(); });
  body.append(help, search, list, clear);
  element.append(summary, body);
  updateCount();
  return { element, clear: clearAll };
}
