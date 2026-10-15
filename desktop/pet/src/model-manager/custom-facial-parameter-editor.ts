import { t } from "../i18n";
import { bindLocalizedAttribute, bindLocalizedText } from "../i18n/dom";
import type { NativeModelParameter } from "../shared/contracts";
import { AUTOMATIC_FACIAL_IDS, validateCustomParameters, type CustomFacialParameter } from "../shared/facial-bindings";
import { STANDARD_PARAMETERS } from "../facial-demo/parameter-bindings";
import type { FacialParameterSuggestion } from "../shared/facial-test";
import { facialDecisionEditor } from "./facial-decision-editor";
import { editorBody, editorButton, editorCard, editorField, editorFooter, motionPicker } from "./facial-editor-layout";

type Stage = "parameter" | "confirm" | "running" | "form";

/**
 * A part Jev drives through one of the model's own parameters, set up in
 * steps: pick the parameter (drag to see what it moves), then either have the
 * AI fill the wording — after confirming how the part is split, since the AI
 * works to that — or write it yourself. The AI only fills the form; nothing is
 * saved until the person saves. An existing part opens on its form, with the
 * AI one click away at the top.
 */
export function customFacialParameterEditor(options: {
  value?: CustomFacialParameter;
  native: NativeModelParameter[];
  blocked: string[];
  usedIds: string[];
  preview(id: string, value: number): void;
  suggest(id: string, mode: "score" | "binary", levelCount: number): Promise<FacialParameterSuggestion>;
  save(value: CustomFacialParameter): Promise<void>;
  remove?(): Promise<void>;
  close(): void;
}): HTMLElement {
  const standard = new Set(STANDARD_PARAMETERS.map(parameter => parameter.id));
  const used = new Set(options.usedIds.filter(id => id !== options.value?.id));
  const candidates = options.native.filter(parameter =>
    !standard.has(parameter.id) && !AUTOMATIC_FACIAL_IDS.has(parameter.id)
    && !options.blocked.includes(parameter.id) && !used.has(parameter.id)
    && Number.isFinite(parameter.minimum) && Number.isFinite(parameter.maximum)
    && parameter.minimum < parameter.maximum);
  const root = element("div", "facial-binding-editor");
  root.dataset.customParameterEditor = options.value?.id ?? "new";
  let stage: Stage = options.value ? "form" : "parameter";

  // The parameter, shared by every step.
  const select = element("select"); select.dataset.customParameterId = "";
  for (const parameter of candidates) {
    const option = element("option"); option.value = parameter.id;
    option.textContent = `${parameter.name || parameter.id} · ${parameter.id}（${parameter.minimum}～${parameter.maximum}）`;
    select.append(option);
  }
  select.value = options.value?.id ?? candidates[0]?.id ?? "";
  select.disabled = !!options.value || candidates.length === 0;
  const selected = () => candidates.find(parameter => parameter.id === select.value);
  const slider = input("range"); slider.dataset.customParameterPreview = "";
  const previewValue = element("span", "fbe-chip");
  const tryRow = element("div", "fbe-try"); tryRow.append(slider, previewValue);
  const updateSlider = () => {
    const parameter = selected(); slider.disabled = !parameter;
    if (!parameter) return;
    slider.min = String(parameter.minimum); slider.max = String(parameter.maximum);
    slider.step = String((parameter.maximum - parameter.minimum) / 200);
    slider.value = String(parameter.defaultValue); previewValue.textContent = format(parameter.defaultValue);
  };
  slider.addEventListener("input", () => {
    previewValue.textContent = format(Number(slider.value));
    options.preview(select.value, Number(slider.value));
  });

  // The form.
  const label = input("text"); label.maxLength = 100;
  label.dataset.customParameterLabel = ""; label.value = options.value?.label ?? candidates[0]?.name ?? candidates[0]?.id ?? "";
  const description = element("textarea"); description.rows = 3; description.maxLength = 500;
  description.dataset.customParameterDescription = "";
  bindLocalizedAttribute(description, "placeholder", "这个参数表达什么，往两头分别是什么意思？");
  description.value = options.value?.description ?? "";
  const decisionHost = element("div");
  const createDecision = (mode = options.value?.mode ?? "score", levels = options.value?.levels ?? ["最小", "中间", "最大"],
    falseDescription = options.value?.falseDescription, trueDescription = options.value?.trueDescription) => {
    const parameter = selected();
    const minimum = parameter?.minimum ?? 0, maximum = parameter?.maximum ?? 1;
    return facialDecisionEditor({ mode, levels,
      anchors: levels.map((_, index) => index === 0 ? minimum : index === levels.length - 1 ? maximum
        : minimum + (maximum - minimum) * index / (levels.length - 1)),
      minimum, maximum, columns: { names: [], values: value => [value] }, falseDescription, trueDescription });
  };
  let decision = createDecision(); decisionHost.append(decision.element);
  // Whether Jev drives the part lives on the row's switch; a new part starts on.
  const enabled = options.value?.enabled ?? true;
  const multiplier = input("number"); multiplier.min = "0"; multiplier.max = "100"; multiplier.step = "0.1";
  multiplier.value = String(options.value?.multiplier ?? 1); multiplier.dataset.customParameterMultiplier = "";
  const amplitude = element("div", "fbe-amp");
  amplitude.append(Object.assign(element("span", "fbe-times"), { textContent: "×" }), multiplier,
    bindLocalizedText(Object.assign(element("span", "fbe-note"), { textContent: "" }), "1 是原样，大于 1 动得更明显"));
  const motion = motionPicker(options.value?.motion);
  const more = element("details", "facial-binding-more");
  more.append(bindLocalizedText(Object.assign(element("summary"), { textContent: "" }), "补充说明"),
    bindLocalizedText(Object.assign(element("p", "facial-binding-hint"), { textContent: "" }), "决策模型判断时会读这段话，说清这个参数代表什么、往两头分别是什么意思。"),
    description);
  more.open = !options.value?.description;
  const feedback = element("p", "facial-binding-feedback"); feedback.setAttribute("role", "status");
  let revision = 0;
  root.addEventListener("input", () => { revision++; });
  root.addEventListener("change", () => { revision++; });

  select.addEventListener("change", () => {
    label.value = selected()?.name || select.value;
    let previous;
    try { previous = decision.read(); }
    catch { previous = { mode: "score" as const, levels: [...(options.value?.levels ?? ["最小", "中间", "最大"])] }; }
    decision = createDecision(previous.mode, previous.mode === "score" ? previous.levels : options.value?.levels ?? ["最小", "中间", "最大"],
      previous.mode === "binary" ? previous.falseDescription : undefined,
      previous.mode === "binary" ? previous.trueDescription : undefined);
    decisionHost.replaceChildren(decision.element);
    updateSlider();
    // The confirm step names the parameter the AI will try.
    if (stage === "confirm") render();
  });

  // What the AI works to, confirmed before it runs.
  let aiMode: "score" | "binary" = options.value?.mode ?? "score";
  let aiLevels = Math.max(2, Math.min(10, options.value?.levels.length ?? 3));

  const suggestButton = element("button", "secondary"); suggestButton.type = "button";
  suggestButton.dataset.customParameterSuggest = "";
  suggestButton.addEventListener("click", () => { stage = "confirm"; render(); });
  const suggest = async () => {
    const parameter = selected();
    if (!parameter) return;
    // The AI answers to what was confirmed; the form only changes once it has.
    const selection = { mode: aiMode, levelCount: aiMode === "score" ? aiLevels : 2 };
    const requestedRevision = revision;
    stage = "running"; render();
    try {
      const suggestion = await options.suggest(parameter.id, selection.mode, selection.levelCount);
      if (!root.isConnected) return;
      stage = "form";
      if (requestedRevision !== revision || select.value !== parameter.id) {
        bindLocalizedText(feedback, "识别期间改了内容，这次结果没有填进去，可以再让 AI 填一次。");
        return;
      }
      if (suggestion.parameterId !== parameter.id || suggestion.mode !== selection.mode) throw new Error("AI 给的结果和所选参数不符");
      if (suggestion.confidence === "uncertain") {
        bindLocalizedText(feedback, "{{copy}}", () => ({ copy: (suggestion.note || t("AI 看画面没能确定这个参数的作用，请自己填一下。")) }));
        return;
      }
      label.value = suggestion.label;
      description.value = suggestion.description;
      decision = createDecision(suggestion.mode, suggestion.mode === "score" ? suggestion.levels! :
        options.value?.levels ?? ["最小", "中间", "最大"], suggestion.falseDescription, suggestion.trueDescription);
      decisionHost.replaceChildren(decision.element);
      more.open = true;
      bindLocalizedText(feedback, "{{copy}}", () => ({ copy: (suggestion.note || t("AI 填好了下面的内容。拖一下看看对不对，没问题就保存。")) }));
    } catch (error) {
      stage = "form";
      if (root.isConnected) feedback.textContent = error instanceof Error ? error.message : String(error);
    } finally {
      if (root.isConnected) render();
    }
  };

  const save = async () => {
    try {
      if (!selected()) throw new Error("当前模型没有可添加的原生参数");
      if (!multiplier.value.trim()) throw new Error("请填写幅度");
      const rule = decision.read();
      const value = validateCustomParameters([{ id: select.value, label: label.value,
        description: description.value,
        levels: rule.mode === "score" ? rule.levels : options.value?.levels ?? ["最小", "中间", "最大"],
        mode: rule.mode,
        ...(rule.mode === "binary" ? { falseDescription: rule.falseDescription, trueDescription: rule.trueDescription } : {}),
        enabled, multiplier: Number(multiplier.value), motion: motion.read() }])[0]!;
      bindLocalizedText(feedback, "正在保存…");
      await options.save(value);
    } catch (error) { feedback.textContent = error instanceof Error ? error.message : String(error); }
  };

  function render(): void {
    const parameter = selected();
    const parameterName = parameter ? parameter.name || parameter.id : "";
    if (stage === "parameter") {
      // Step 1: which parameter; then how to fill it in.
      bindLocalizedText(Object.assign(suggestButton, { textContent: "", className: "accent", disabled: !parameter }), "✦ 让 AI 帮我填");
      const manual = bindLocalizedText(editorButton("", "customParameterManual", "secondary", () => { stage = "form"; render(); }), "自己填");
      const card = editorCard("新的部位 · 先选模型参数", null,
        bindLocalizedText(Object.assign(element("p", "fbe-note"), { textContent: "" }), (candidates.length ? "选一个模型参数，拖一下看它动的是哪里。" : "这个模型没有可以添加的参数了（没被物理或呼吸占用的都已经用上）。")),
        editorField("对应模型的哪个参数", select),
        editorField("拖一下看看", tryRow));
      const buttons = element("div", "fbe-actions"); buttons.append(manual, suggestButton);
      root.replaceChildren(card, editorFooter(
        bindLocalizedText(editorButton("", "customParameterClose", "le-link le-link--muted", options.close), "算了"), feedback, buttons));
      return;
    }
    if (stage === "confirm" || stage === "running") {
      // Step 2: what the AI works to, confirmed before it runs.
      const split = element("div", "le-seg");
      const splitButton = (value: "score" | "binary", text: string) => {
        const control = editorButton(text, value === "score" ? "aiModeScore" : "aiModeBinary", "", () => { aiMode = value; render(); });
        control.setAttribute("aria-pressed", String(aiMode === value)); control.disabled = stage === "running";
        split.append(control);
      };
      splitButton("score", "从弱到强分几档");
      splitButton("binary", "只有是和否");
      const splitRow = element("div", "fbe-row"); splitRow.append(split);
      if (aiMode === "score") {
        const count = element("select"); count.dataset.aiLevelCount = ""; count.disabled = stage === "running";
        count.className = "fbe-count";
        for (let value = 2; value <= 10; value += 1) {
          const option = element("option"); option.value = String(value); bindLocalizedText(option, "{{v0}} 档", { v0: value }); count.append(option);
        }
        count.value = String(aiLevels);
        count.addEventListener("change", () => { aiLevels = Number(count.value); });
        splitRow.append(count);
      }
      const card = editorCard("✦ 让 AI 帮我填", null,
        bindLocalizedText(Object.assign(element("p", "fbe-note"), { textContent: "" }), "AI 会试动「{{v0}}」、对照画面看它控制什么，然后填好名字、每一档怎么说和补充说明。只是填进表格，你检查后再保存。", { v0: parameterName }),
        ...options.value ? [editorField("对应模型的哪个参数", Object.assign(element("b", "fbe-value"), { textContent: parameterName }))]
          : [editorField("对应模型的哪个参数", select), editorField("拖一下看看", tryRow)],
        editorField("怎么分", splitRow));
      const start = stage === "running"
        ? bindLocalizedText(Object.assign(element("span", "fbe-note"), { textContent: "" }), "AI 正在试动参数、看画面…")
        : bindLocalizedText(editorButton("", "customParameterSuggestStart", "accent", () => { void suggest(); }), "开始");
      root.replaceChildren(card, editorFooter(
        stage === "running" ? null : bindLocalizedText(editorButton("", "customParameterSuggestCancel", "le-link le-link--muted", () => { stage = options.value || revision ? "form" : "parameter"; render(); }), "算了"),
        feedback, start));
      return;
    }
    // Step 3: the form, filled by hand or by the AI.
    bindLocalizedText(Object.assign(suggestButton, { textContent: "", className: "secondary fbe-small", disabled: !parameter }), (options.value ? "✦ 让 AI 重新填" : "✦ 让 AI 帮我填"));
    const saveButton = bindLocalizedText(editorButton("", "customParameterSave", "accent", () => { void save(); }), "保存");
    saveButton.disabled = !parameter;
    const remove = options.remove
      ? bindLocalizedText(editorButton("", "customParameterDelete", "le-link fbe-danger", () => {
        void options.remove!().catch(error => { feedback.textContent = error instanceof Error ? error.message : String(error); });
      }), "删除这个部位")
      : null;
    root.replaceChildren(
      editorBody(
        editorCard("这个部位", null,
          editorField("名字", label),
          editorField("对应模型的哪个参数", select),
          editorField("拖一下看看", tryRow),
          editorField("动作幅度", amplitude),
          editorField("怎么动", motion.element)),
        // The AI fills what this card holds, so its button lives here.
        editorCard("决策模型怎么判断", suggestButton, decisionHost, more),
      ),
      editorFooter(remove, feedback, saveButton),
    );
  }

  updateSlider();
  render();
  return root;
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function input(type: string): HTMLInputElement {
  const node = document.createElement("input"); node.type = type; return node;
}

function format(value: number): string { return String(Number(value.toFixed(2))); }
