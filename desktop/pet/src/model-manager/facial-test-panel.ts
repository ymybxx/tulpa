import { t } from "../i18n";
import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import type { ModelPackage } from "../shared/contracts";
import type { Face } from "../shared/facial-demo";
import type { ModelManagerPreview } from "./preview";
import { JevTestPlayer } from "../facial-demo/jev-player";
import { buildBindings, type ParameterBinding } from "../facial-demo/parameter-bindings";
import type { FacialTestModelInfo } from "../shared/facial-test";
import { vtsPartBindings } from "../shared/vts-model-settings";
import { facialBindingEditor } from "./facial-binding-editor";
import { STANDARD_PARAMETERS } from "../facial-demo/parameter-bindings";
import { customFacialParameterEditor } from "./custom-facial-parameter-editor";
import { AutoSetupJob, facialAutoSetupPanel } from "./facial-auto-setup-panel";
import type { CustomFacialParameter, FacialParameterSettings } from "../shared/facial-bindings";

/** The editor owns the existing preview; this panel owns only a temporary test session. */
export class FacialTestPanel {
  private readonly player: JevTestPlayer;
  private version = 0;
  private preparing = false;
  private active = false;
  private model: ModelPackage | null = null;
  private preview: ModelManagerPreview | null = null;
  private bindings: ParameterBinding[] = [];
  private info: FacialTestModelInfo | null = null;
  private editing: HTMLElement | null = null;
  /** Hides the 一键配置 card; its job carries on. */
  private detachAutoSetup: (() => void) | null = null;
  /** Outlives the card and page visits, until saved, dismissed or cancelled. */
  private autoSetupJob: AutoSetupJob | null = null;
  private resolved: { active: ParameterBinding[]; all: ParameterBinding[] } | null = null;
  private readonly text: HTMLTextAreaElement;
  private readonly run: HTMLButtonElement;
  private readonly stopButton: HTMLButtonElement;
  private readonly status: HTMLElement;
  private readonly sentence: HTMLElement;
  private readonly values: HTMLElement;
  private readonly support: HTMLElement;
  private readonly zoom: HTMLInputElement;
  private readonly zoomValue: HTMLElement;
  private get hasEnabledParameters(): boolean { return this.bindings.some(binding => binding.enabled); }

  constructor(root: HTMLElement, private readonly beforeStart: () => Promise<void>) {
    this.text = root.querySelector<HTMLTextAreaElement>("textarea")!;
    this.run = root.querySelector<HTMLButtonElement>("[data-facial-run]")!;
    this.stopButton = root.querySelector<HTMLButtonElement>("[data-facial-stop]")!;
    this.status = root.querySelector<HTMLElement>("[data-facial-status]")!;
    this.sentence = root.querySelector<HTMLElement>("[data-facial-sentence]")!;
    this.values = root.querySelector<HTMLElement>("[data-facial-values]")!;
    this.support = root.querySelector<HTMLElement>("[data-facial-support]")!;
    this.zoom = root.querySelector<HTMLInputElement>("[data-facial-zoom]")!;
    this.zoomValue = root.querySelector<HTMLElement>("[data-facial-zoom-value]")!;
    root.querySelector<HTMLButtonElement>("[data-facial-pan-reset]")!.addEventListener("click", () => {
      if (this.active) this.preview?.resetFacialTestPan();
    });
    this.zoom.addEventListener("input", () => {
      if (!this.active) return;
      this.preview?.setFacialTestZoom(Number(this.zoom.value));
      this.zoomValue.textContent = `${Number(this.zoom.value).toFixed(1)}×`;
    });
    this.player = new JevTestPlayer(window.desktopPet.facialTest, {
      status: (message, running) => {
        this.status.textContent = message;
        this.run.disabled = running || this.preparing || !!this.editing || !this.active || !this.preview || !this.hasEnabledParameters;
        this.stopButton.disabled = !running && !this.preparing;
      },
      sentence: text => { this.sentence.textContent = text; },
      values: face => this.renderValues(face),
    });
    this.run.addEventListener("click", () => { void this.start(); });
    this.stopButton.addEventListener("click", () => this.stop());
  }

  /** Setting a new context invalidates preparation and any in-flight speech immediately. */
  setContext(active: boolean, model: ModelPackage | null, preview: ModelManagerPreview | null): void {
    if (this.active === active && this.model?.id === model?.id && this.preview === preview) return;
    this.stop();
    this.preview?.setFacialTestZoom(1);
    this.active = active; this.model = model; this.preview = preview;
    // The whole body first, like everywhere a look is shown; pinch to look at the face.
    this.zoom.value = "1"; this.zoomValue.textContent = "1.0×";
    if (active) preview?.setFacialTestZoom(1);
    this.bindings = [];
    this.info = null;
    this.values.textContent = "";
    this.support.replaceChildren();
    bindLocalizedText(this.sentence, "写一句话，点「朗读并测试」，左边的 TA 会边说边做表情。");
    this.run.disabled = true;
    bindLocalizedText(this.status, (active && model && preview ? "只在左边的预览里试，不会动到桌面上的 TA。" : "先选一个形象。"));
    if (active && model && preview) void this.prepareSupport(model, preview);
  }

  private async prepareSupport(model: ModelPackage, preview: ModelManagerPreview): Promise<void> {
    const version = this.version;
    try {
      const info = await window.desktopPet.facialTest.prepare(model.id);
      if (version !== this.version || !this.active || this.preview !== preview || preview.displayedModelId() !== model.id) return;
      this.info = info;
      const resolved = this.resolveBindings(info, preview);
      this.renderSupport(resolved);
      this.run.disabled = !resolved.active.length;
      // Back on the page while a run is going or waiting to be looked at.
      if (this.autoSetupJob?.modelId === model.id) this.openAutoSetup();
      if (!resolved.active.length) {
        this.run.disabled = true;
        bindLocalizedText(this.status, (this.bindings.length ? "当前未开启任何参数的自动生成，请在下方开启至少一项。" : "当前模型没有可直接控制的参数，可以在下方手动绑定或新增自定义参数。"));
      }
    } catch (error) {
      if (version !== this.version) return;
      this.run.disabled = true;
      bindLocalizedText(this.status, "无法准备表情测试：{{v0}}", { v0: error instanceof Error ? error.message : String(error) });
    }
  }

  private async start(): Promise<void> {
    if (this.editing || this.preparing || this.player.running || !this.active || !this.model || !this.preview) return;
    const text = this.text.value.trim();
    if (!text) { bindLocalizedText(this.status, "先输入想让模型朗读的文字。"); this.text.focus(); return; }
    const model = this.model, preview = this.preview, version = ++this.version;
    this.preparing = true; this.run.disabled = true; this.stopButton.disabled = false;
    bindLocalizedText(this.status, "正在准备当前模型…");
    try {
      await this.beforeStart();
      if (version !== this.version) return;
      const info = await window.desktopPet.facialTest.prepare(model.id);
      if (version !== this.version || !this.active || this.preview !== preview || preview.displayedModelId() !== model.id) return;
      this.info = info;
      const resolved = this.resolveBindings(info, preview);
      this.renderSupport(resolved);
      this.preparing = false;
      await this.player.start(model.id, text, this.bindings, preview, info.parameterEffects);
    } catch (error) {
      if (version === this.version) this.stop(`测试失败：${error instanceof Error ? error.message : String(error)}`);
    } finally {
      if (version === this.version) {
        this.preparing = false;
        this.run.disabled = this.player.running || !this.hasEnabledParameters;
        this.stopButton.disabled = !this.player.running;
      }
    }
  }

  stop(message = "已停止"): void {
    this.version += 1; this.preparing = false;
    this.closeEditor();
    this.player.stop(message);
  }

  destroy(): void { this.stop(); this.active = false; this.player.destroy(); this.preview?.setFacialTestZoom(1); }

  private closeEditor(): void {
    if (!this.editing) return;
    if (this.detachAutoSetup) {
      this.detachAutoSetup(); this.detachAutoSetup = null;
      this.preview?.setFacialTestZoom(Number(this.zoom.value));
    }
    const toggle = this.editing.parentElement?.querySelector<HTMLButtonElement>("[data-bind-standard]");
    if (toggle) bindLocalizedText(toggle, "调整");
    this.editing.remove(); this.editing = null;
    this.preview?.endFacialTest();
    this.run.disabled = !this.active || !this.hasEnabledParameters;
  }

  private editBinding(row: HTMLElement, binding: ParameterBinding): void {
    if (!this.model || !this.preview || !this.info) return;
    this.stop("正在设置参数");
    const model = this.model, preview = this.preview, info = this.info, version = this.version;
    const native = preview.nativeParameters();
    const editor = facialBindingEditor({ binding, native, blocked: info.physicsOutputParameterIds,
      reference: vtsPartBindings(info.vtsSettings ?? [], native, info.physicsOutputParameterIds)[binding.id],
      preview: values => {
        if (this.version !== version || !this.active) return;
        preview.beginFacialTest({ mouth: null, eyes: [] });
        preview.applyFacialTest(values);
      },
      cancel: () => this.stop("绑定设置已关闭"),
      remove: () => this.saveControls(binding, { ...ownLabel(binding), enabled: false, multiplier: binding.multiplier, removed: true }),
      save: async (bindingValue, settings, description, decision) => {
        const standard = STANDARD_PARAMETERS.find(value => value.id === binding.id)!;
        const saved = await window.desktopPet.facialTest.saveBinding(model.id, binding.id, bindingValue, settings,
          description.trim() === standard.description ? "" : description, decision);
        if (version !== this.version || this.preview !== preview || !this.active) return;
        this.stop("参数设置已保存。");
        this.info = saved;
        this.renderSupport(this.resolveBindings(saved, preview));
        this.run.disabled = !this.hasEnabledParameters;
        if (this.bindings.length && !this.hasEnabledParameters) bindLocalizedText(this.status, "当前未开启任何参数的自动生成，请在下方开启至少一项。");
      },
    });
    this.editing = editor; row.append(editor); this.run.disabled = true;
    editor.scrollIntoView({ block: "nearest" });
  }

  /** 一键配置 sits above the parts it may add to; the preview shows the whole body while it is open. */
  private openAutoSetup(): void {
    if (!this.model || !this.preview || !this.info || !this.resolved) return;
    this.stop("");
    const model = this.model, preview = this.preview, info = this.info, version = this.version;
    const all = this.resolved.all;
    const job = this.autoSetupJob?.modelId === model.id ? this.autoSetupJob : null;
    const context = job?.context ?? {
      used: [...new Set([
        ...all.filter(binding => binding.controllable).flatMap(binding => binding.outputs.map(output => output.parameterId)),
        ...all.flatMap(binding => (binding.manualBindings ?? []).map(output => output.parameterId)),
        ...(info.customParameters ?? []).map(value => value.id),
      ])],
      open: all.filter(binding => binding.group !== "custom" && !binding.controllable && !binding.removed
        && STANDARD_PARAMETERS.find(value => value.id === binding.id)?.owner === "jev").map(binding => binding.id),
      customSlots: Math.max(0, 16 - (info.customParameters?.length ?? 0)),
    };
    const panel = facialAutoSetupPanel({
      context, native: preview.nativeParameters(),
      job: () => this.autoSetupJob?.modelId === model.id ? this.autoSetupJob : null,
      start: () => { this.autoSetupJob = new AutoSetupJob(model.id, context); },
      dismiss: () => {
        this.autoSetupJob?.cancel(); this.autoSetupJob = null;
        this.stop("已关闭一键配置");
      },
      show: (id, value) => {
        if (this.version !== version || !this.active) return;
        preview.beginFacialTest({ mouth: null, eyes: [] });
        preview.applyFacialTest(id ? { [id]: value! } : {});
      },
      save: async proposal => {
        let saved = info;
        for (const entry of proposal.standard) {
          saved = await window.desktopPet.facialTest.saveBinding(model.id, entry.standardId, entry.binding, { enabled: true, multiplier: 1 });
        }
        for (const entry of proposal.custom) saved = await window.desktopPet.facialTest.saveCustomParameter(model.id, entry.value);
        if (version !== this.version || this.preview !== preview || !this.active) return;
        const count = proposal.standard.length + proposal.custom.length;
        this.autoSetupJob = null;
        this.stop(`已保存 ${count} 项，可以在下面逐个调整。`);
        this.info = saved;
        this.renderSupport(this.resolveBindings(saved, preview));
        this.run.disabled = !this.hasEnabledParameters;
      },
    });
    preview.setFacialTestZoom(1);
    this.detachAutoSetup = panel.detach;
    this.editing = panel.element; this.support.prepend(panel.element); this.run.disabled = true;
    panel.element.scrollIntoView({ block: "nearest" });
  }

  private renderValues(face: Face): void {
    this.values.textContent = this.bindings.map(binding => `${binding.label} ${(face[binding.id] ?? binding.neutral).toFixed(2)}`).join("  ·  ");
  }

  private resolveBindings(info: FacialTestModelInfo, preview: ModelManagerPreview) {
    return buildBindings(preview.nativeParameters(), info.blushParameter,
      info.physicsOutputParameterIds, info.manualBindings, info.parameterSettings,
      info.descriptions, info.customParameters, info.criteria, info.vtsSettings);
  }

  private editCustom(row: HTMLElement, value?: CustomFacialParameter): void {
    if (!this.model || !this.preview || !this.info) return;
    this.stop("正在设置自定义参数");
    const model = this.model, preview = this.preview, info = this.info, version = this.version;
    const editor = customFacialParameterEditor({
      value, native: preview.nativeParameters(), blocked: info.physicsOutputParameterIds,
      usedIds: [...(info.customParameters?.map(item => item.id) ?? []),
        ...this.bindings.filter(binding => binding.group !== "custom")
          .flatMap(binding => binding.outputs.map(output => output.parameterId))],
      preview: (id, target) => {
        if (this.version !== version || !this.active) return;
        preview.beginFacialTest({ mouth: null, eyes: [] });
        preview.applyFacialTest({ [id]: target });
      },
      suggest: (id, mode, levelCount) => window.desktopPet.facialTest.suggestParameter(model.id, id, mode, levelCount),
      close: () => this.stop("参数设置已关闭"),
      save: async custom => {
        const saved = await window.desktopPet.facialTest.saveCustomParameter(model.id, custom);
        if (version !== this.version || this.preview !== preview || !this.active) return;
        this.stop("自定义参数已保存。"); this.info = saved;
        this.renderSupport(this.resolveBindings(saved, preview));
      },
      remove: value ? async () => {
        const saved = await window.desktopPet.facialTest.deleteCustomParameter(model.id, value.id);
        if (version !== this.version || this.preview !== preview || !this.active) return;
        this.stop("自定义参数已删除。"); this.info = saved;
        this.renderSupport(this.resolveBindings(saved, preview));
      } : undefined,
    });
    this.editing = editor; row.append(editor); this.run.disabled = true;
    editor.scrollIntoView({ block: "nearest" });
  }

  /** Turning a part on or off, or deleting it, keeps its binding and judgement as they are. */
  /** The row's name turns into a text box; Enter or leaving it saves, Esc keeps the old name. */
  private renameInline(name: HTMLButtonElement, binding: ParameterBinding): void {
    const field = document.createElement("input"); field.type = "text"; field.maxLength = 30;
    field.className = "le-param__rename"; field.value = binding.label;
    let done = false;
    const finish = (keep: boolean) => {
      if (done) return;
      done = true;
      const value = field.value.trim();
      field.replaceWith(name);
      if (!keep || !value || value === binding.label) return;
      name.textContent = value;
      if (binding.group === "custom") {
        const custom = this.info?.customParameters?.find(item => item.id === binding.id);
        if (custom && this.model) {
          void window.desktopPet.facialTest.saveCustomParameter(this.model.id, { ...custom, label: value }).then(saved => {
            this.info = saved; if (this.preview) this.renderSupport(this.resolveBindings(saved, this.preview));
          });
        }
        return;
      }
      const standard = STANDARD_PARAMETERS.find(item => item.id === binding.id);
      void this.saveControls(binding, {
        enabled: binding.enabled, multiplier: binding.multiplier,
        ...value !== standard?.label ? { label: value } : {},
      });
    };
    field.addEventListener("keydown", (event) => {
      if (event.key === "Enter") finish(true);
      if (event.key === "Escape") finish(false);
    });
    field.addEventListener("blur", () => finish(true));
    name.replaceWith(field);
    field.focus();
    field.select();
  }

  private async saveCustom(value: CustomFacialParameter): Promise<void> {
    if (!this.model || !this.preview) return;
    const model = this.model, preview = this.preview;
    this.stop("正在保存…");
    try {
      const saved = await window.desktopPet.facialTest.saveCustomParameter(model.id, value);
      if (this.model !== model || this.preview !== preview || !this.active) return;
      this.info = saved;
      this.renderSupport(this.resolveBindings(saved, preview));
      bindLocalizedText(this.status, "已保存。");
      this.run.disabled = !this.hasEnabledParameters;
    } catch (error) {
      bindLocalizedText(this.status, "保存失败：{{v0}}", { v0: error instanceof Error ? error.message : String(error) });
    }
  }

  private async saveControls(binding: ParameterBinding, controls: FacialParameterSettings): Promise<void> {
    if (!this.model || !this.preview) return;
    const model = this.model, preview = this.preview;
    this.stop("正在保存…");
    try {
      const saved = await window.desktopPet.facialTest.saveBinding(model.id, binding.id, binding.manualBindings ?? null, controls);
      if (this.model !== model || this.preview !== preview || !this.active) return;
      this.info = saved;
      this.renderSupport(this.resolveBindings(saved, preview));
      bindLocalizedText(this.status, "{{copy}}", () => ({ copy: (controls.removed ? t("已删除「{{v0}}」", { v0: binding.label }) : t("已保存。")) }));
      this.run.disabled = !this.hasEnabledParameters;
    } catch (error) {
      bindLocalizedText(this.status, "保存失败：{{v0}}", { v0: error instanceof Error ? error.message : String(error) });
    }
  }

  /**
   * One row per part Jev can drive: what it maps to, its multiplier, a switch
   * for whether Jev drives it, 调整 and delete. Parts the model animates by
   * itself (breathing, physics) are not listed; one hint line says so.
   */
  private renderSupport(resolved: { active: ParameterBinding[]; all: ParameterBinding[] }): void {
    this.resolved = resolved;
    this.bindings = resolved.all.filter(binding => binding.controllable);
    const jevDriven = resolved.all.filter(binding => binding.group === "custom"
      || STANDARD_PARAMETERS.find(value => value.id === binding.id)?.owner === "jev");
    const row = (binding: ParameterBinding): HTMLElement => {
      const element = document.createElement("div"); element.className = "le-param";
      element.dataset.standardId = binding.id;
      const text = document.createElement("span"); text.className = "le-param__text";
      const name = document.createElement("button"); name.type = "button"; name.className = "le-param__name";
      if (STANDARD_PARAMETERS.some(parameter => parameter.id === binding.id && parameter.label === binding.label)) bindLocalizedText(name, binding.label);
      else name.textContent = binding.label;
      bindLocalizedAttribute(name, "title", "点一下改名字");
      name.addEventListener("click", () => this.renameInline(name, binding));
      const mapping = document.createElement("small");
      const targets = binding.outputs.map(output =>
        `${output.parameterId}${output.manual && output.manual.lowerValue > output.manual.upperValue ? "（反向）" : ""}`);
      bindLocalizedText(mapping, "{{copy}}", () => ({ copy: (targets.length ? t("对应 {{v0}}{{v1}}", { v0: targets.join("、"), v1: (binding.fromVts ? t("（按模型自带的 VTS 设置）") : "") }) : t("还没对应参数，点「调整」选一个")) }));
      // Found but not Jev's to move (physics drives it, or its range is fixed): say why.
      if (!binding.controllable && (binding.manualBindings?.length || binding.outputs.length)) mapping.textContent = binding.reason;
      text.append(name, mapping);
      // The switch is the decision model's; the voice's intonation still moves the head a little.
      if (binding.id === "ParamAngleY" && binding.controllable) {
        const voice = document.createElement("small"); voice.dataset.voiceNote = "";
        bindLocalizedText(voice, "关掉后，说话时头只会随语调轻微起伏");
        text.append(voice);
      }
      const scale = document.createElement("span"); scale.className = "le-param__scale";
      scale.textContent = `${binding.multiplier.toFixed(1)}×`;
      scale.classList.toggle("changed", binding.multiplier !== 1);
      const toggle = document.createElement("button"); toggle.type = "button"; toggle.className = "le-switch";
      toggle.setAttribute("role", "switch"); bindLocalizedAttribute(toggle, "aria-label", "决策模型控制 {{v0}}", { v0: binding.label });
      toggle.setAttribute("aria-checked", String(binding.enabled)); bindLocalizedAttribute(toggle, "title", "决策模型控制");
      toggle.append(Object.assign(document.createElement("span"), { className: "le-switch__knob" }));
      toggle.addEventListener("click", () => {
        // A custom part keeps its switch on itself; only standard parts have settings rows.
        const custom = binding.group === "custom" ? this.info?.customParameters?.find(value => value.id === binding.id) : undefined;
        if (custom) void this.saveCustom({ ...custom, enabled: !custom.enabled });
        else void this.saveControls(binding, { ...ownLabel(binding), enabled: !binding.enabled, multiplier: binding.multiplier });
      });
      const adjust = document.createElement("button"); adjust.type = "button"; adjust.className = "secondary";
      bindLocalizedText(adjust, "调整");
      adjust.dataset.bindStandard = binding.id;
      adjust.addEventListener("click", () => {
        // The same button folds the settings away again.
        if (this.editing?.parentElement === element) {
          this.stop("已收起");
          return;
        }
        if (binding.group === "custom") this.editCustom(element, this.info?.customParameters?.find(value => value.id === binding.id));
        else this.editBinding(element, binding);
        if (this.editing?.parentElement === element) bindLocalizedText(adjust, "收起");
      });
      // Deleting a part lives in its 调整, next to the other rarely used actions.
      element.append(text, scale, toggle, adjust);
      return element;
    };
    const rows = jevDriven.filter(binding => !binding.removed).map(row);
    const removed = jevDriven.filter(binding => binding.removed);
    const extras: HTMLElement[] = [];
    if (removed.length) {
      const details = document.createElement("details"); details.className = "le-removed";
      const summary = document.createElement("summary"); bindLocalizedText(summary, "已删除的部位（{{v0}}）", { v0: removed.length });
      details.append(summary, ...removed.map(binding => {
        const element = document.createElement("div"); element.className = "le-param";
        const name = document.createElement("span"); name.className = "le-param__text";
        if (STANDARD_PARAMETERS.some(parameter => parameter.id === binding.id && parameter.label === binding.label)) bindLocalizedText(name, binding.label);
        else name.textContent = binding.label;
        const restore = document.createElement("button"); restore.type = "button"; restore.className = "secondary"; bindLocalizedText(restore, "加回来");
        restore.addEventListener("click", () => void this.saveControls(binding, { enabled: true, multiplier: binding.multiplier }));
        element.append(name, restore);
        return element;
      }));
      extras.push(details);
    }
    const add = document.createElement("div"); add.className = "le-param le-param--add";
    const hint = document.createElement("span"); bindLocalizedText(hint, "呼吸、眨眼、口型和头发物理由模型自己控制，不在这里调");
    const button = document.createElement("button"); button.type = "button"; button.className = "secondary";
    button.dataset.addCustomParameter = ""; bindLocalizedText(button, "添加自定义部位");
    button.addEventListener("click", () => this.editCustom(add)); add.append(hint, button);
    const auto = document.createElement("div"); auto.className = "le-param le-param--add le-param--auto";
    // A part the model's physics already moves is taken care of, not missing.
    const unmatched = jevDriven.filter(binding => binding.group !== "custom" && !binding.controllable && !binding.removed
      && binding.owner === "jev").length;
    auto.append(Object.assign(document.createElement("span"), { textContent: unmatched
      ? `还有 ${unmatched} 个部位没对应上。让 AI 看一遍模型，补上它们，再加上有用的表情和动作`
      : "让 AI 看一遍模型，找出还没用上的表情和动作" }));
    const autoButton = document.createElement("button"); autoButton.type = "button"; autoButton.className = "accent";
    const job = this.autoSetupJob?.modelId === this.model?.id ? this.autoSetupJob : null;
    autoButton.dataset.autoSetupOpen = ""; bindLocalizedText(autoButton, (!job ? "AI 一键配置" : (job.running ? "查看进度" : "查看结果")));
    autoButton.addEventListener("click", () => this.openAutoSetup());
    auto.append(autoButton);
    this.support.replaceChildren(auto, ...rows, ...extras, add);
  }
}

/** A renamed standard part keeps its name through switch and delete saves. */
function ownLabel(binding: ParameterBinding): { label?: string } {
  const standard = STANDARD_PARAMETERS.find(item => item.id === binding.id);
  return standard && binding.label !== standard.label ? { label: binding.label } : {};
}
