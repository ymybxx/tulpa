import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import { validateDecisionRule, type FacialDecisionRule } from "../shared/facial-bindings";

/** The model parameters a part moves, as the steps table's columns. */
export interface DecisionColumns {
  /** Column headings; with none, the single value column goes unheaded. */
  names: readonly string[];
  /** Where each parameter lands for a step's part value, one per column. */
  values(value: number): number[];
}

/**
 * How Jev describes a part: a few named steps from one end to the other, or
 * just yes and no. The steps read as a table — one row per step, one column
 * per model parameter — so each step's name and what it does to the model are
 * read together; 「看看」 shows that step in the preview. The numeric range
 * always comes from the selected model.
 */
export function facialDecisionEditor(options: {
  mode: "score" | "binary";
  levels: readonly string[];
  anchors: readonly number[];
  minimum: number;
  maximum: number;
  columns: DecisionColumns;
  /** Shows the part at this value in the preview; without it, rows have no 「看看」. */
  look?(value: number): void;
  falseDescription?: string;
  trueDescription?: string;
}): {
  element: HTMLElement;
  read(): FacialDecisionRule;
  selection(): { mode: "score" | "binary"; levelCount: number };
  /** Re-reads the values each step moves to, e.g. after a parameter changed. */
  setColumns(columns: DecisionColumns): void;
} {
  let columns = options.columns;
  let mode: "score" | "binary" = options.mode;
  const element = document.createElement("div"); element.className = "facial-binding-criteria";
  const switcher = document.createElement("div"); switcher.className = "le-seg"; switcher.dataset.decisionMode = "";
  const modeButton = (value: "score" | "binary", text: string) => {
    const button = document.createElement("button"); button.type = "button"; bindLocalizedText(button, text);
    button.dataset[value === "score" ? "decisionModeScore" : "decisionModeBinary"] = "";
    button.addEventListener("click", () => { mode = value; show(); });
    switcher.append(button);
    return button;
  };
  const scoreButton = modeButton("score", "从弱到强分几档");
  const binaryButton = modeButton("binary", "只有是和否");
  const intro = document.createElement("p"); intro.className = "facial-binding-hint";
  const score = document.createElement("div"); score.className = "facial-decision-steps"; score.dataset.decisionScore = "";
  const binary = document.createElement("div"); binary.className = "facial-decision-steps"; binary.dataset.decisionBinary = "";
  const falseInput = document.createElement("input"); falseInput.type = "text"; falseInput.maxLength = 100;
  falseInput.dataset.decisionFalse = ""; falseInput.value = options.falseDescription ?? options.levels[0] ?? "不表现";
  const trueInput = document.createElement("input"); trueInput.type = "text"; trueInput.maxLength = 100;
  trueInput.dataset.decisionTrue = ""; trueInput.value = options.trueDescription ?? options.levels.at(-1) ?? "表现";
  const spacer = () => Object.assign(document.createElement("span"), { className: "facial-decision-spacer" });
  const layout = (row: HTMLElement, named: boolean) => {
    row.style.gridTemplateColumns = [...named ? ["28px"] : [], "minmax(0, 1fr)",
      ...Array.from({ length: Math.max(1, columns.names.length) }, () => "64px"), ...options.look ? ["44px"] : [], "28px"].join(" ");
    return row;
  };
  const header = (named: boolean) => {
    const row = document.createElement("div"); row.className = "facial-decision-level facial-decision-head";
    if (named) row.append(spacer());
    row.append(bindLocalizedText(Object.assign(document.createElement("span"), { textContent: "" }), "决策模型选的档"));
    for (const name of columns.names) row.append(Object.assign(document.createElement("span"), { className: "facial-decision-column", textContent: name, title: name }));
    if (options.look) row.append(spacer());
    row.append(spacer());
    return layout(row, named);
  };
  const stepRow = (input: HTMLInputElement, value: number, tail?: HTMLElement, prefix?: string) => {
    const row = document.createElement("div"); row.className = "facial-decision-level";
    if (prefix) row.append(Object.assign(document.createElement("b"), { className: "facial-decision-prefix", textContent: prefix }));
    row.append(input);
    const values = columns.values(value);
    for (const target of values.length ? values : [Number.NaN]) {
      row.append(Object.assign(document.createElement("span"), { className: "facial-decision-value",
        textContent: Number.isFinite(target) ? format(target) : "—" }));
    }
    if (options.look) {
      const look = document.createElement("button"); look.type = "button"; look.className = "le-link facial-decision-look";
      bindLocalizedText(look, "看看"); bindLocalizedAttribute(look, "title", "在左边预览这一档"); look.dataset.decisionLook = "";
      look.addEventListener("click", () => options.look!(value));
      row.append(look);
    }
    row.append(tail ?? spacer());
    return layout(row, !!prefix);
  };
  const renderBinary = () => {
    bindLocalizedAttribute(falseInput, "placeholder", "不是的时候怎么说"); bindLocalizedAttribute(trueInput, "placeholder", "是的时候怎么说");
    binary.replaceChildren(
      ...columns.names.length ? [header(true)] : [],
      stepRow(falseInput, options.minimum, undefined, "否"),
      stepRow(trueInput, options.maximum, undefined, "是"),
    );
  };
  const levels = [...options.levels];
  let inputs: HTMLInputElement[] = [];
  const renderScore = () => {
    const anchors = levels.length === options.anchors.length ? options.anchors :
      levels.map((_, index) => index === 0 ? options.minimum : index === levels.length - 1 ? options.maximum
        : options.minimum + (options.maximum - options.minimum) * index / (levels.length - 1));
    const rows: HTMLElement[] = [];
    inputs = levels.map((level, index) => {
      const input = document.createElement("input"); input.type = "text"; input.maxLength = 100;
      input.value = level; input.dataset.decisionLevel = String(index);
      bindLocalizedAttribute(input, "placeholder", (index === 0 ? "最弱时怎么说" : (index === levels.length - 1 ? "最强时怎么说" : "中间怎么说")));
      input.addEventListener("input", () => { levels[index] = input.value; });
      // The two ends define the range; only steps between them can go.
      let remove: HTMLElement | undefined;
      if (index > 0 && index < levels.length - 1) {
        const button = document.createElement("button"); button.type = "button"; button.className = "facial-decision-remove";
        button.textContent = "×"; bindLocalizedAttribute(button, "title", "去掉这一档"); button.dataset.decisionRemove = String(index);
        button.addEventListener("click", () => { levels.splice(index, 1); renderScore(); });
        remove = button;
      }
      const value = Math.max(options.minimum, Math.min(options.maximum, anchors[index] ?? options.minimum));
      rows.push(stepRow(input, value, remove));
      return input;
    });
    const add = document.createElement("button"); add.type = "button"; add.className = "le-link"; bindLocalizedText(add, "+ 加一档");
    add.dataset.decisionAdd = ""; add.disabled = levels.length >= 10;
    add.addEventListener("click", () => { levels.splice(levels.length - 1, 0, ""); renderScore(); });
    score.replaceChildren(...columns.names.length ? [header(false)] : [], ...rows, add);
  };
  const show = () => {
    score.hidden = mode !== "score"; binary.hidden = mode !== "binary";
    scoreButton.setAttribute("aria-pressed", String(mode === "score"));
    binaryButton.setAttribute("aria-pressed", String(mode === "binary"));
    bindLocalizedText(intro, (mode === "score" ? "TA 每说一句话，决策模型会从下面选一档最贴切的，模型参数就移到那一行的值。" : "TA 每说一句话，决策模型判断是或否，模型参数就移到那一行的值。"));
  };
  renderScore(); renderBinary(); show();
  element.append(switcher, intro, score, binary);
  return {
    element,
    selection: () => ({ mode, levelCount: inputs.length }),
    read: () => validateDecisionRule(mode === "binary"
      ? { mode: "binary", falseDescription: falseInput.value, trueDescription: trueInput.value }
      : { mode: "score", levels: inputs.map(input => input.value) }),
    setColumns: (next) => { columns = next; renderScore(); renderBinary(); },
  };
}

function format(value: number): string { return String(Number(value.toFixed(3))); }
