import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import { t } from "../i18n";
import type {
  ModelEditCapabilityManifest,
  ModelEditPartCapability,
  ModelPackage,
  TextureMaterial,
  TextureAtlasDelta,
  TextureEditUnit,
} from "../shared/contracts";
import {
  DEFAULT_TEXTURE_EDIT_IMAGE_MODEL,
  TEXTURE_EDIT_REGION_COLORS,
  type CharacterCompletionKind,
  type CharacterGenerationJob,
  type CharacterGenerationProgress,
  type CharacterReferenceSelection,
  type MasterTemplateSummary,
  type TextureEditBatchPlan,
  type TextureEditImageModel,
  type TextureReferenceDeliveryUse,
  type TextureReferencePromptId,
  type TexturePromptOptimizationState,
  type TextureReferenceSelection,
  type TextureVisualEditSpec,
  planTextureEditBatch,
  textureEditBatchPromptRegions,
  textureReferenceLabels,
} from "../shared/character-generation";
import {
  characterPreviewModelId,
  characterWizardStep,
  textureAdjustmentReturnState,
  textureReferenceDeliveryState,
  texturePromptReview,
  texturePromptSubmissionChoice,
} from "./character-wizard-state";
import { initialTextureFilterValue, textureUnitsForPart } from "./view-model";
import { VirtualGrid } from "./virtual-grid";
import { TextureLayerSelection } from "./texture-layer-selection";

const ALL_MATERIALS_REPAIR_ID = "__all-materials__";
const REALIGN_TO_TEMPLATE_ID = "__realign-to-template__";

const COMPLETION_OPTIONS: Array<{
  id: CharacterCompletionKind;
  label: string;
  detail: string;
  selected: boolean;
}> = [
  { id: "eyes", label: "闭眼眼皮与睫毛", detail: "仅当眨眼不自然或缺素材", selected: false },
  { id: "mouth", label: "口腔、牙齿与舌头", detail: "仅当张嘴时缺少内部素材", selected: false },
  { id: "back-hair", label: "被遮挡的后发", detail: "仅当转头或摆动时露空", selected: false },
  { id: "face-base", label: "脸部底层", detail: "仅当转头时五官或头发下方露空", selected: false },
  { id: "body-hidden", label: "身体与四肢遮挡层", detail: "仅当动作时衣服下方露空", selected: false },
  { id: "clothing-hidden", label: "服装遮挡层", detail: "仅当头发、手臂移动后服装缺失", selected: false },
];

interface CharacterWizardDependencies {
  models(): ModelPackage[];
  previewModel(id: string): Promise<void>;
  previewTextureDelta(delta: TextureAtlasDelta): Promise<void>;
  alignmentGuideUrl(modelId: string): Promise<string>;
  prepareModelReference(id: string): Promise<void>;
  refreshModels(id?: string): Promise<void>;
  texturePreviewUrl(unit: TextureEditUnit): string;
  textureMaterials(textureUnitId: string): TextureMaterial[];
  attachPreview(target: HTMLElement, modelId: string, pick?: (ids: string[]) => void): Promise<void>;
  highlightLayers(ids: readonly string[]): void;
  parkPreview(): void;
  restorePreview(): Promise<void>;
  setPreviewPaused(value: boolean): void;
}

interface TextureBatchDraft {
  capability: ModelEditCapabilityManifest;
  textureUnitIds: string[];
  originalPrompt: string;
  finalPrompt: string;
  optimizationSourcePrompt: string | null;
  optimizedPrompt: string | null;
  optimizationState: TexturePromptOptimizationState;
  optimizationWarnings: string[];
  optimizationSpec: TextureVisualEditSpec | null;
  references: TextureReferenceSelection[];
  optionalReferenceOverrides: string[];
  plan: TextureEditBatchPlan;
}

export class CharacterWizard {
  private current: CharacterGenerationJob | null = null;
  private currentAlignmentGuideUrl: string | null = null;
  private templates: MasterTemplateSummary[] = [];
  private reference: CharacterReferenceSelection | null = null;
  private capability: ModelEditCapabilityManifest | null = null;
  private requestedPartId: string | null = null;
  private textureSelection: {
    templateId: string;
    ids: string[];
    prompt: string;
    imageModel: TextureEditImageModel;
    references: TextureReferenceSelection[];
  } | null = null;
  private textureBatchDraft: TextureBatchDraft | null = null;
  private progress = 0;
  private detail = "";
  private textureVirtualGrid: VirtualGrid<TextureEditUnit> | null = null;
  private previewRequested = false;
  private previewAttachmentEpoch = 0;
  private pendingTexturePreview: { jobId: string; delta: TextureAtlasDelta } | null = null;

  constructor(private readonly dependencies: CharacterWizardDependencies) {
    required("character-wizard-close").addEventListener("click", () => void this.close());
    window.desktopPet.onCharacterProgress((value) => void this.update(value));
  }

  async open(capability?: ModelEditCapabilityManifest, partId?: string): Promise<void> {
    if (!capability) {
      throw new Error("AI 创建角色和旧部位修改功能已移除，请先预检现有模型的可编辑纹理");
    }
    if (this.capability?.templateId !== capability?.templateId) {
      this.textureSelection = null;
      this.textureBatchDraft = null;
    }
    this.capability = capability ?? null;
    this.currentAlignmentGuideUrl = await this.dependencies.alignmentGuideUrl(capability.modelId);
    this.requestedPartId = partId ?? null;
    bindLocalizedText(required("character-wizard-title"), "AI 高保真修改纹理");
    required("character-wizard").hidden = false;
    this.dependencies.setPreviewPaused(true);
    const jobs = await window.desktopPet.listCharacterJobs();
    this.templates = [];
    this.current = jobs.find((job) => {
      if (["saved", "cancelled"].includes(job.phase)) return false;
      return job.input.operation === "texture-edit" && job.templateId === capability.templateId;
    }) ?? null;
    if (
      this.current?.outputModelId &&
      ["base-preview", "preview"].includes(this.current.phase)
    ) {
      await this.dependencies.previewModel(this.current.outputModelId);
    }
    this.render();
  }

  async close(): Promise<void> {
    this.previewAttachmentEpoch += 1;
    this.pendingTexturePreview = null;
    this.textureVirtualGrid?.destroy();
    this.textureVirtualGrid = null;
    const job = this.current;
    required("character-wizard").hidden = true;
    this.dependencies.setPreviewPaused(true);
    this.dependencies.parkPreview();
    try {
      if (job?.input.operation === "texture-edit" && job.phase === "preview") {
        await this.dependencies.previewTextureDelta(
          await window.desktopPet.setCharacterTexturePreview(job.id, false),
        );
      }
    } catch (error) {
      console.warn("恢复正式素材预览失败", error);
    } finally {
      try {
        await this.dependencies.restorePreview();
      } catch (error) {
        console.warn("恢复模型管理预览失败", error);
      }
      this.dependencies.setPreviewPaused(false);
    }
  }

  private async update(update: CharacterGenerationProgress): Promise<void> {
    if (!this.current || update.jobId !== this.current.id) return;
    this.current = update.job;
    this.progress = update.progress;
    this.detail = update.detail;
    if (["base-preview", "preview"].includes(update.job.phase) && update.job.outputModelId) {
      await this.dependencies.refreshModels(update.job.outputModelId);
      await this.dependencies.previewModel(update.job.outputModelId);
    }
    this.render();
  }

  private render(): void {
    this.previewAttachmentEpoch += 1;
    const root = required("character-wizard-body");
    this.textureVirtualGrid?.destroy();
    this.textureVirtualGrid = null;
    this.previewRequested = false;
    this.dependencies.parkPreview();
    this.dependencies.setPreviewPaused(true);
    root.replaceChildren();
    try {
      if (!this.current) {
        if (this.textureBatchDraft) {
          this.renderTextureBatchConfirmation(root, this.textureBatchDraft);
          return;
        }
        this.renderInput(root);
        return;
      }
      this.renderCost(root, this.current);
      if (this.current.input.operation === "texture-edit") {
        if (this.current.paused && this.current.pauseReason === "budget") this.renderBudget(root, this.current);
        else if (this.current.phase === "preview") this.renderTextureResults(root, this.current);
        else if (this.current.phase === "saved") this.renderDone(root, this.current);
        else if (["failed", "cancelled"].includes(this.current.phase)) this.renderFailure(root, this.current);
        else this.renderProcessing(root, this.current);
        return;
      }
      if (["eye-restyle", "part-restyle"].includes(this.current.input.operation ?? "")) {
        if (this.current.paused && this.current.pauseReason === "budget") {
          this.renderBudget(root, this.current);
        } else if (this.current.phase === "preview") {
          this.renderRestylePreview(root, this.current);
        } else if (this.current.phase === "saved") {
          this.renderDone(root, this.current);
        } else if (["failed", "cancelled"].includes(this.current.phase)) {
          this.renderFailure(root, this.current);
        } else {
          this.renderProcessing(root, this.current);
        }
        return;
      }
      const step = characterWizardStep(this.current);
      if (step === "choose-candidate") this.renderCandidates(root, this.current);
      else if (step === "budget") this.renderBudget(root, this.current);
      else if (step === "parts") this.renderParts(root, this.current);
      else if (step === "confirm-base-preview") this.renderBasePreview(root, this.current);
      else if (step === "confirm-preview") this.renderPreview(root, this.current);
      else if (step === "done") this.renderDone(root, this.current);
      else if (step === "failed") this.renderFailure(root, this.current);
      else this.renderProcessing(root, this.current);
    } finally {
      if (!this.previewRequested) {
        void this.dependencies.restorePreview()
          .catch((error) => console.warn("释放 AI 动画预览失败", error));
      }
    }
  }

  private renderInput(root: HTMLElement): void {
    if (this.capability) {
      this.renderPartInput(root, this.capability);
      return;
    }
    root.append(bindLocalizedText(text("p", "character-intro", ""), "只修改原模型左右虹膜、瞳孔和高光。眼型、眼距、眼皮、睫毛、脸部、头发、身体、动作和物理全部保持原样。"));
    if (this.templates.length === 0) {
      const empty = bindLocalizedText(text("div", "character-preview-confirm", ""), "还没有安装自有正式母版。开发样例不会随产品分发，请安装包含 master.template.json 的母版文件夹。");
      const install = bindLocalizedText(button("", async () => {
        const value = await window.desktopPet.importMasterTemplate();
        if (value) {
          this.templates = (await window.desktopPet.listMasterTemplates())
            .filter((template) => template.coverage === "full-body");
        }
        this.render();
      }), "安装母版文件夹");
      empty.append(install);
      root.append(empty);
      return;
    }
    const form = document.createElement("form");
    form.className = "character-form";
    const template = document.createElement("select");
    for (const item of this.templates) {
      template.append(new Option(`${item.name} · ${item.slotCount} 个槽位`, item.id));
    }
    const mao = this.templates.find((item) => item.id.includes("mao"));
    if (mao) template.value = mao.id;
    const description = document.createElement("textarea");
    bindLocalizedAttribute(description, "placeholder", "例如：清澈的淡粉色虹膜，深色瞳孔，细腻放射纹理和柔和星形高光");
    description.maxLength = 300;
    const submit = bindLocalizedText(button("", async () => undefined, "submit"), "生成眼睛预览 · 预计约 $0.030");
    form.append(
      labelled("基础模型", template),
      labelled("眼睛描述", description),
      submit,
    );
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      void this.start(template.value, description.value, "");
    });
    root.append(form);
  }

  private renderPartInput(root: HTMLElement, capability: ModelEditCapabilityManifest): void {
    if (!this.currentAlignmentGuideUrl) throw new Error("当前模型正面参考图尚未准备好");
    root.append(bindLocalizedText(text("p", "character-intro", ""), "点击模型选择要修改的部位，多次点击可以累加选择。悬停右侧纹理可在模型上高亮；共用同一纹理的部位会一起修改。下一步会将所选纹理放入一张画布联合生成。"));
    if (capability.textureUnits.length === 0) {
      root.append(bindLocalizedText(text("div", "character-error", ""), "当前模型没有扫描到可编辑的唯一 UV 纹理素材。"));
      return;
    }
    const form = document.createElement("form");
    form.className = "character-form";
    const remembered = this.textureSelection?.templateId === capability.templateId
      ? this.textureSelection
      : null;
    const selected = new Set<string>(remembered?.ids ?? []);
    const references = (remembered?.references ?? []).map((reference) => ({
      ...reference,
      description: reference.source === "upload" ? reference.description : "",
    }));
    const imageModel = DEFAULT_TEXTURE_EDIT_IMAGE_MODEL;
    const layerSelection = new TextureLayerSelection(capability.textureUnits);
    let locatedUnits: TextureEditUnit[] | null = null;
    const previewEpoch = this.previewAttachmentEpoch;
    const previewIsCurrent = () => previewEpoch === this.previewAttachmentEpoch && !required("character-wizard").hidden && this.current === null;
    let previewReady = false;
    const highlight = (ids: readonly string[]) => {
      if (previewReady && previewIsCurrent()) this.dependencies.highlightLayers(ids);
    };
    const highlightSelected = () => highlight(layerSelection.layers(selected));
    const locator = document.createElement("section");
    locator.className = "texture-person-locator";
    const locatorHost = text("div", "texture-selection-preview-host", "");
    const locatorHint = bindLocalizedText(text("small", "", ""), "正在加载模型，加载后可点击选择部位。");
    locatorHint.setAttribute("role", "status");
    const clearLocatorFilter = bindLocalizedText(button("", () => {
      locatedUnits = null;
      bindLocalizedText(locatorHint, "点击模型选中最前方的可编辑部位；重叠部位可在右侧勾选。");
      clearLocatorFilter.hidden = true;
      highlightSelected();
      renderUnits();
    }), "显示全部纹理");
    clearLocatorFilter.className = "secondary";
    clearLocatorFilter.hidden = true;
    const locatorStatus = document.createElement("div");
    locatorStatus.className = "texture-person-locator-status";
    locatorStatus.append(locatorHint, clearLocatorFilter);
    locator.append(bindLocalizedText(text("h3", "", ""), "点击模型选择部位"), locatorHost, locatorStatus);
    const filter = document.createElement("select");
    bindLocalizedAttribute(filter, "aria-label", "AI 编辑部位筛选");
    const largeModel = capability.textureUnits.length > 120;
    if (largeModel) filter.append(new Option("请选择人物部位", "__select__"));
    filter.append(new Option("全部纹理", ""));
    for (const item of capability.parts) filter.append(new Option(item.label, item.id));
    filter.value = initialTextureFilterValue(this.requestedPartId);
    const grid = document.createElement("div");
    grid.className = "texture-unit-grid";
    const description = document.createElement("textarea");
    bindLocalizedAttribute(description, "placeholder", "例如：去掉袜口并输出真实透明边缘；或改为干净的淡粉色赛璐璐纹理，保留原线稿与阴影");
    description.maxLength = 300;
    description.value = remembered?.prompt ?? "";
    const feedback = text("div", "character-error", this.detail);
    feedback.hidden = !this.detail;
    const summary = bindLocalizedText(text("p", "texture-selection-summary", ""), "尚未选择纹理，不会产生请求");
    let clearSelection: HTMLButtonElement | null = null;
    const planPreview = document.createElement("section");
    planPreview.className = "texture-batch-live-preview";
    const referenceEditor = document.createElement("section");
    referenceEditor.className = "texture-reference-editor";
    const submit = bindLocalizedText(button("", async () => undefined, "submit"), "请先选择纹理");
    submit.disabled = true;
    let currentPlan: TextureEditBatchPlan | null = null;
    let planRevision = 0;
    const setFeedback = (message: string): void => {
      this.detail = message;
      feedback.textContent = message;
      feedback.hidden = !message;
    };
    const rememberSelection = (): void => {
      this.textureSelection = {
        templateId: capability.templateId,
        ids: [...selected],
        prompt: description.value,
        imageModel,
        references: references.map((reference) => ({ ...reference })),
      };
    };
    const renderReferences = (): void => {
      referenceEditor.replaceChildren(
        bindLocalizedText(text("h3", "", ""), "参考图片（可选）"),
        bindLocalizedText(text("p", "character-intro", ""), "参考图会作为独立图片发送，不占用编辑画布。正面图只定位，模型素材按名称引用，上传图片可以补充一句可选说明。"),
      );
      const actions = document.createElement("div");
      actions.className = "texture-reference-actions";
      const addModel = bindLocalizedText(button("", () => {
        if (references.some((reference) => reference.source === "model-preview")) return;
        references.push({
          id: createReferenceId(),
          source: "model-preview",
          name: `${capability.modelName} 当前正面图`,
          description: "",
          path: null,
          textureUnitId: null,
          materialId: null,
          previewUrl: this.currentAlignmentGuideUrl,
        });
        rememberSelection();
        renderReferences();
      }), "添加当前人物正面图");
      addModel.disabled = references.length >= 6 || references.some((reference) =>
        reference.source === "model-preview"
      );
      const addUpload = bindLocalizedText(button("", async () => {
        const selectedReference = await window.desktopPet.pickCharacterReference();
        if (!selectedReference || references.length >= 6) return;
        references.push({
          id: createReferenceId(),
          source: "upload",
          name: selectedReference.name,
          description: "",
          path: selectedReference.path,
          textureUnitId: null,
          materialId: null,
          previewUrl: selectedReference.previewUrl,
        });
        rememberSelection();
        renderReferences();
      }), "上传参考图");
      addUpload.disabled = references.length >= 6;
      actions.append(addModel, addUpload);
      referenceEditor.append(actions);
      if (references.length === 0) {
        referenceEditor.append(bindLocalizedText(text("p", "character-intro", ""), "未添加参考图：图片模型只依据当前纹理和最终提示词修改。"));
        return;
      }
      const grid = document.createElement("div");
      grid.className = "texture-reference-grid";
      const referenceLabels = textureReferenceLabels(references);
      references.forEach((reference, index) => {
        const card = document.createElement("article");
        card.className = "texture-reference-card";
        const image = document.createElement("img");
        image.src = reference.previewUrl ?? "";
        image.alt = reference.name;
        const descriptionInput = document.createElement("input");
        descriptionInput.type = "text";
        descriptionInput.maxLength = 300;
        bindLocalizedAttribute(descriptionInput, "placeholder", "可选：一句话说明希望参考什么");
        descriptionInput.value = reference.description;
        descriptionInput.addEventListener("input", () => {
          reference.description = descriptionInput.value;
          rememberSelection();
        });
        card.append(
          text("strong", "", `${referenceLabels[index]} · ${reference.name}`),
          image,
          ...(reference.source === "upload"
            ? [labelled("参考说明（可选）", descriptionInput)]
            : []),
          bindLocalizedText(button("", () => {
            references.splice(index, 1);
            rememberSelection();
            renderReferences();
          }), "移除"),
        );
        grid.append(card);
      });
      referenceEditor.append(
        grid,
        bindLocalizedText(text("small", "", ""), "已添加 {{v0}}/6 张参考图。定位图不参与外观参考，其他图片的 R1、R2 顺序会固定写入提示词和图片请求。", { v0: references.length }),
      );
    };
    const addMaterialReference = (
      unit: TextureEditUnit,
      material: TextureMaterial,
    ): void => {
      if (
        references.length >= 6 ||
        references.some((reference) => reference.materialId === material.id)
      ) return;
      references.push({
        id: createReferenceId(),
        source: "texture-unit",
        name: `${unit.label} · ${material.name}`,
        description: "",
        path: null,
        textureUnitId: unit.id,
        materialId: material.id,
        previewUrl: material.imageUrl,
      });
      rememberSelection();
      renderReferences();
    };
    const chooseMaterialReference = (unit: TextureEditUnit): void => {
      const materials = this.dependencies.textureMaterials(unit.id);
      if (materials.length === 0) {
        setFeedback(`${unit.label} 暂无可用素材`);
        return;
      }
      if (materials.length === 1) {
        addMaterialReference(unit, materials[0]!);
        return;
      }
      document.querySelector(".texture-reference-material-dialog")?.remove();
      const dialog = document.createElement("dialog");
      dialog.className = "texture-variant-dialog texture-reference-material-dialog";
      const header = document.createElement("header");
      header.append(
        bindLocalizedText(text("div", "", ""), "{{v0}} · 选择一个参考素材", { v0: unit.label }),
        button("×", () => dialog.close()),
      );
      const choices = document.createElement("div");
      choices.className = "texture-variant-grid";
      for (const material of materials) {
        const choice = document.createElement("article");
        choice.className = "texture-variant-choice";
        const image = document.createElement("img");
        image.src = material.imageUrl;
        image.alt = `${unit.label} ${material.name}`;
        const alreadyAdded = references.some((reference) => reference.materialId === material.id);
        const add = bindLocalizedText(button("", () => {
          addMaterialReference(unit, material);
          dialog.close();
        }), (alreadyAdded ? "✓ 已作为参考" : "添加为参考"));
        add.disabled = alreadyAdded || references.length >= 6;
        choice.append(
          image,
          text("strong", "", material.name),
          material.prompt
            ? text("p", "texture-material-prompt", material.prompt)
            : bindLocalizedText(text("small", "", ""), (material.active ? "当前使用" : "历史素材")),
          add,
        );
        choices.append(choice);
      }
      dialog.append(
        header,
        bindLocalizedText(text("p", "texture-variant-help", ""), "只添加为参考，不会切换当前人物正在使用的素材。"),
        choices,
      );
      dialog.addEventListener("close", () => dialog.remove());
      document.body.append(dialog);
      dialog.showModal();
    };
    const updateSummary = (): void => {
      const count = selected.size;
      if (count === 0) {
        bindLocalizedText(summary, "尚未选择纹理，不会产生请求");
        bindLocalizedText(submit, "请先选择纹理");
      } else if (currentPlan) {
        bindLocalizedText(summary, "已选择 {{v0}} 个纹理 · {{v1}} · {{v2}} · 1 次图片请求 · 预计 ${{v3}}", { v0: count, v1: textureImageModelLabel(currentPlan.imageModel), v2: currentPlan.imageSize, v3: currentPlan.estimatedUsd.toFixed(3) });
        bindLocalizedText(submit, "继续确认 {{v0}} 个纹理", { v0: count });
      } else {
        bindLocalizedText(summary, "已选择 {{v0}} 个纹理 · 正在计算联合画布", { v0: count });
        bindLocalizedText(submit, "正在排版…");
      }
      submit.disabled = currentPlan === null;
      if (clearSelection) clearSelection.disabled = count === 0;
    };
    const refreshPlan = async (): Promise<void> => {
      const revision = ++planRevision;
      const ids = [...selected];
      currentPlan = null;
      rememberSelection();
      updateSummary();
      if (ids.length === 0) {
        planPreview.replaceChildren(bindLocalizedText(text("p", "character-intro", ""), "勾选纹理后，这里会立即显示实际联合画布。"));
        setFeedback("");
        return;
      }
      planPreview.replaceChildren(
        bindLocalizedText(text("h3", "", ""), "联合画布实时预览"),
        bindLocalizedText(text("p", "character-intro", ""), "正在为 {{v0}} 个纹理计算最大整数放大倍数和排版…", { v0: ids.length }),
      );
      try {
        // Live layout is a pure rectangle-packing calculation. Keep it in the
        // renderer so rapid checkbox changes never queue model I/O, material
        // locks or alignment-guide generation in the Electron main process.
        // The main process recomputes the same deterministic hash immediately
        // before optimization and paid generation.
        const selectedUnits = ids.map((id) => capability.textureUnits.find((unit) => unit.id === id))
          .filter((unit): unit is TextureEditUnit => Boolean(unit));
        if (selectedUnits.length !== ids.length) throw new Error("选择中包含已失效的纹理");
        const result = planTextureEditBatch(selectedUnits.map((unit) => ({
          id: unit.id,
          label: unit.label,
          width: unit.bounds.width,
          height: unit.bounds.height,
        })), imageModel);
        if (revision !== planRevision) return;
        if (!result.fits) {
          const labels = result.textureUnitIds.map((id) =>
            capability.textureUnits.find((unit) => unit.id === id)?.label ?? id
          );
          planPreview.replaceChildren(
            bindLocalizedText(text("h3", "", ""), "当前选择无法放入一张画布"),
            text("p", "character-error", `${result.reason}：${labels.join("、")}`),
          );
          setFeedback("请取消至少一个纹理；当前勾选和输入内容已经保留。");
          updateSummary();
          return;
        }
        currentPlan = result.plan;
        setFeedback("");
        planPreview.replaceChildren(
          bindLocalizedText(text("h3", "", ""), "联合画布实时预览"),
          createTextureBatchPreview(
            capability,
            result.plan,
            this.dependencies.texturePreviewUrl,
          ),
          bindLocalizedText(text("p", "character-intro", ""), "{{v0}} · {{v1}} · 每个纹理保持单独生成时的最大整数倍 · 16px 间距 · 只产生 1 次图片请求。", { v0: textureImageModelLabel(result.plan.imageModel), v1: result.plan.imageSize }),
        );
        updateSummary();
      } catch (error) {
        if (revision !== planRevision) return;
        planPreview.replaceChildren(
          bindLocalizedText(text("h3", "", ""), "联合画布暂时无法计算"),
          text("p", "character-error", error instanceof Error ? error.message : String(error)),
        );
        setFeedback("排版检查失败；当前勾选和输入内容已经保留，可以重试勾选或稍后再试。");
        updateSummary();
      }
    };
    const renderUnits = (): void => {
      this.textureVirtualGrid?.destroy();
      this.textureVirtualGrid = null;
      grid.replaceChildren();
      const units = filter.value === "__select__" ? [] : locatedUnits
        ? locatedUnits.filter(unit => !filter.value || unit.semanticPartIds.includes(filter.value))
        : textureUnitsForPart(capability, filter.value);
      if (units.length === 0) {
        grid.append(bindLocalizedText(text("p", "character-intro", ""), (filter.value === "__select__" ? "此模型包含大量纹理，请先选择人物部位；已有勾选会保持不变。" : "此筛选下没有纹理；筛选不会改变已经勾选的内容。")));
        return;
      }
      this.textureVirtualGrid = new VirtualGrid(grid, {
        minColumnWidth: 180,
        rowHeight: 308,
        gap: 12,
        maxRenderedItems: 60,
      });
      this.textureVirtualGrid.set(units, (unit) => {
        const card = document.createElement("label");
        card.className = "texture-unit-card";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        bindLocalizedAttribute(checkbox, "aria-label", "选择纹理 {{v0}}", { v0: unit.label });
        card.dataset.textureUnitId = unit.id;
        checkbox.checked = selected.has(unit.id);
        checkbox.addEventListener("change", () => {
          if (checkbox.checked) selected.add(unit.id); else selected.delete(unit.id);
          card.classList.toggle("selected", checkbox.checked);
          highlightSelected();
          void refreshPlan();
        });
        const image = document.createElement("img");
        image.src = this.dependencies.texturePreviewUrl(unit);
        image.alt = unit.label;
        image.loading = "lazy";
        image.decoding = "async";
        card.append(
          checkbox,
          image,
          text("strong", "", unit.label),
          bindLocalizedText(text("small", "", ""), "{{v0}}×{{v1}} · {{v2}} 个图层共用", { v0: unit.bounds.width, v1: unit.bounds.height, v2: unit.drawableIds.length }),
        );
        const motionNames = textureUnitMotionNames(
          unit.revealedByMotionIds,
          this.dependencies.models().find((model) => model.id === capability.modelId),
        );
        if (unit.revealedByMotionIds.length > 0) {
          card.append(bindLocalizedText(text("small", "texture-action-visibility", ""), "{{copy}}", () => ({ copy: (motionNames.length > 0 ? t("仅在动作中显示：{{v0}}", { v0: motionNames.join("、") }) : t("仅在动作中显示")) })));
        }
        const materialCount = this.dependencies.textureMaterials(unit.id).length;
        const referenceAction = document.createElement("button");
        referenceAction.type = "button";
        referenceAction.className = "texture-unit-reference-button";
        bindLocalizedText(referenceAction, "{{copy}}", () => ({ copy: (materialCount > 1 ? t("作为参考 · 选择 {{v0}} 个素材之一", { v0: materialCount }) : t("添加为参考")) }));
        referenceAction.disabled = materialCount === 0 || references.length >= 6;
        referenceAction.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          chooseMaterialReference(unit);
        });
        card.append(referenceAction);
        card.classList.toggle("selected", checkbox.checked);
        card.addEventListener("mouseenter", () => highlight(unit.drawableIds));
        card.addEventListener("mouseleave", highlightSelected);
        card.addEventListener("focusin", () => highlight(unit.drawableIds));
        card.addEventListener("focusout", event => {
          if (!(event.relatedTarget instanceof Node) || !card.contains(event.relatedTarget)) highlightSelected();
        });
        return card;
      });
    };
    filter.addEventListener("change", renderUnits);
    const pickLayers = (ids: string[]) => {
      if (!previewIsCurrent()) return;
      const pick = layerSelection.pick(ids);
      locatedUnits = pick.matched;
      for (const unit of pick.front) selected.add(unit.id);
      filter.value = "";
      bindLocalizedText(locatorHint, "{{copy}}", () => ({ copy: (pick.front.length ? t("已选中 {{v0}}；此处还有 {{v1}} 个重叠纹理可选。", { v0: pick.front.map(unit => unit.label).join("、"), v1: pick.matched.length - pick.front.length }) : t("此处没有可编辑纹理，已有选择保持不变。")) }));
      clearLocatorFilter.hidden = false;
      highlightSelected();
      renderUnits();
      void refreshPlan();
    };
    renderUnits();
    description.addEventListener("input", () => {
      rememberSelection();
      if (description.value.trim()) setFeedback("");
    });
    clearSelection = bindLocalizedText(button("", () => {
      selected.clear();
      highlightSelected();
      renderUnits();
      void refreshPlan();
    }), "清空选中");
    clearSelection.className = "secondary";
    clearSelection.disabled = selected.size === 0;
    const selectionActions = document.createElement("div");
    selectionActions.className = "texture-selection-actions";
    selectionActions.append(summary, clearSelection);
    renderReferences();
    const selectionDetails = text("div", "texture-selection-details", "");
    selectionDetails.append(labelled("按部位筛选", filter), grid, selectionActions);
    const selectionLayout = text("div", "texture-selection-layout", "");
    selectionLayout.append(locator, selectionDetails);
    form.append(
      selectionLayout,
      planPreview,
      referenceEditor,
      labelled("统一修改要求（联合生成，拆回后仍可分别验收）", description),
      bindLocalizedText(text("p", "character-intro", ""), "统一使用 Image 2.5，支持大画布、换色、材质和透明删除。需要去掉内容时直接写在修改要求里；未选区域、模型网格和动作保持不变。"),
      feedback,
      submit,
    );
    this.previewRequested = true;
    void this.dependencies.attachPreview(locatorHost, capability.modelId, pickLayers).then(() => {
      if (!previewIsCurrent() || !locatorHost.isConnected) return;
      previewReady = true;
      bindLocalizedText(locatorHint, "点击模型选中最前方的可编辑部位；重叠部位可在右侧勾选。");
      highlightSelected();
    }).catch(error => {
      if (previewIsCurrent()) bindLocalizedText(locatorHint, "模型预览加载失败：{{v0}}。仍可在右侧选择纹理。", { v0: error instanceof Error ? error.message : String(error) });
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      rememberSelection();
      const finalPrompt = description.value.trim();
      if (!finalPrompt) {
        setFeedback("请输入具体的外观修改要求；已选纹理不会丢失。");
        description.focus();
        return;
      }
      if (!currentPlan) {
        setFeedback("当前选择还没有可用的联合画布，请等待排版完成或减少选择。");
        return;
      }
      this.detail = "";
      this.textureBatchDraft = {
        capability,
        textureUnitIds: [...selected],
        originalPrompt: finalPrompt,
        finalPrompt,
        optimizationSourcePrompt: null,
        optimizedPrompt: null,
        optimizationState: "not-requested",
        optimizationWarnings: [],
        optimizationSpec: null,
        references: references.map((reference) => ({ ...reference })),
        optionalReferenceOverrides: references.map((reference) => reference.id),
        plan: currentPlan,
      };
      this.render();
    });
    root.append(form);
    void refreshPlan();
  }

  private renderTextureBatchConfirmation(root: HTMLElement, draft: TextureBatchDraft): void {
    const { capability, plan } = draft;
    root.append(
      bindLocalizedText(text("h3", "", ""), "确认联合纹理排版"),
      bindLocalizedText(text("p", "character-intro", ""), "所选 {{v0}} 个纹理保持各自单独生成时的最大整数放大倍数；不旋转、不缩小。确认后只产生 1 次图片请求，预计 ${{v1}}。", { v0: plan.items.length, v1: plan.estimatedUsd.toFixed(3) }),
    );
    const preview = createTextureBatchPreview(
      capability,
      plan,
      this.dependencies.texturePreviewUrl,
    );
    const referenceSummary = createTextureReferenceSummary(draft.references);
    const prompt = document.createElement("textarea");
    prompt.maxLength = 12_000;
    prompt.value = draft.finalPrompt;
    prompt.addEventListener("input", () => {
        if (this.textureBatchDraft === draft) {
          draft.finalPrompt = prompt.value;
        }
    });
    const optimization = document.createElement("section");
    optimization.className = "texture-prompt-optimization";
    const optimize = bindLocalizedText(button("", async () => {
      optimize.disabled = true;
      bindLocalizedText(optimize, "正在优化…");
      draft.optimizedPrompt = null;
      draft.optimizationSpec = null;
      draft.optimizationWarnings = [];
      draft.optimizationState = "not-requested";
      try {
        const result = await window.desktopPet.optimizeCharacterTexturePrompt({
          templateId: capability.templateId,
          modelName: capability.modelName,
          textureUnitIds: draft.textureUnitIds,
          prompt: draft.finalPrompt.trim(),
          planHash: plan.hash,
          imageModel: plan.imageModel,
          references: draft.references,
        });
        if (result.plan) draft.plan = result.plan;
        draft.optimizationSourcePrompt = result.originalPrompt;
        draft.optimizedPrompt = result.optimizedPrompt;
        draft.optimizationSpec = result.spec;
        draft.optimizationWarnings = result.warnings ?? [];
        draft.optionalReferenceOverrides = draft.references.map((reference) => reference.id);
        draft.optimizationState = "not-requested";
        this.detail = "";
      } catch (error) {
        this.detail = `${readableError(error)}；可以保留原文继续生成`;
      }
      this.render();
    }), "优化提示词");
    optimization.append(
      bindLocalizedText(text("p", "character-intro", ""), "优化只在主动点击时调用现有 LLM，不会调用图片模型，也不计入图像任务费用。"),
      optimize,
    );
    if (draft.optimizedPrompt) {
      if (draft.optimizationSpec) {
        const review = texturePromptReview(draft.optimizationSpec);
        const visualReview = document.createElement("section");
        visualReview.className = "texture-visual-review";
        visualReview.append(bindLocalizedText(text("h3", "", ""), "识别出的视觉方向"));
        const attributes = document.createElement("div");
        attributes.className = "texture-visual-attribute-grid";
        for (const attribute of review.attributes) {
          const card = document.createElement("article");
          card.className = "texture-visual-attribute";
          card.append(
            text("strong", "", `${attribute.profileId} · ${attribute.category}`),
            text("p", "", attribute.description),
            text("small", "", attribute.scope),
          );
          attributes.append(card);
        }
        visualReview.append(attributes, bindLocalizedText(text("h3", "", ""), "区域应用关系"));
        const regionList = document.createElement("div");
        regionList.className = "texture-visual-region-list";
        for (const region of review.regions) {
          const details = [
            region.profileIds.length > 0 ? `视觉 ${region.profileIds.join("、")}` : "",
            region.clauseIds.length > 0 ? `用户要求 ${region.clauseIds.join("、")}` : "",
          ].filter(Boolean).join(" · ");
          regionList.append(text("p", "", `${region.key}（${region.role}）：${details}`));
        }
        visualReview.append(regionList);
        optimization.append(visualReview);
      }
      const comparison = document.createElement("div");
      comparison.className = "texture-prompt-comparison";
      const optimizationSource = draft.optimizationSourcePrompt ?? draft.originalPrompt;
      const promptChoice = texturePromptSubmissionChoice(
        draft.optimizedPrompt,
        draft.optimizationState,
      );
      comparison.append(
        labelled("优化前", readOnlyText(optimizationSource)),
        labelled("优化建议", readOnlyText(draft.optimizedPrompt)),
      );
      const useOptimized = bindLocalizedText(button("", async () => {
          draft.finalPrompt = draft.optimizedPrompt ?? optimizationSource;
          draft.optimizationState = "accepted";
          this.render();
        }), (promptChoice === "optimized" ? "✓ 已采用优化版" : "采用优化版"));
      useOptimized.disabled = promptChoice === "optimized";
      const keepOriginal = bindLocalizedText(button("", async () => {
          draft.optimizationState = "rejected";
          this.render();
        }), (promptChoice === "current" ? "✓ 已使用当前提示词" : "继续使用当前提示词"));
      keepOriginal.disabled = promptChoice === "current";
      const choiceStatus = bindLocalizedText(text("p", promptChoice === "pending"
          ? "texture-prompt-choice texture-prompt-choice-pending"
          : "texture-prompt-choice texture-prompt-choice-ready", ""), (promptChoice === "pending" ? "优化建议尚未采用。请先明确选择提交优化版，或继续使用当前提示词。" : (promptChoice === "optimized" ? "本次将提交优化版；上方最终提示词可继续手动调整。" : "本次将提交上方当前提示词，不会采用优化建议。")));
      comparison.append(useOptimized, keepOriginal, choiceStatus);
      optimization.append(comparison);
    }
    if (draft.optimizationWarnings.length > 0) {
      const warnings = document.createElement("div");
      warnings.className = "texture-prompt-warnings";
      warnings.append(bindLocalizedText(text("strong", "", ""), "参考图与修改风险提示"));
      for (const warning of draft.optimizationWarnings) {
        warnings.append(text("p", "", warning));
      }
      optimization.append(warnings);
    }
    const referenceDelivery = this.textureReferenceDeliverySection(draft);
    const actions = document.createElement("div");
    actions.className = "texture-batch-actions";
    const promptChoice = texturePromptSubmissionChoice(
      draft.optimizedPrompt,
      draft.optimizationState,
    );
    const submitBatch = bindLocalizedText(button("", async () => this.startTextureBatch(draft)), (promptChoice === "pending" ? "请先选择提示词版本" : (promptChoice === "optimized" ? "使用优化版并批量生成" : "使用当前提示词并批量生成")));
    submitBatch.disabled = promptChoice === "pending";
    actions.append(
      bindLocalizedText(button("", async () => {
        if (this.textureSelection?.templateId === capability.templateId) {
          this.textureSelection.prompt = draft.originalPrompt;
        }
        this.textureBatchDraft = null;
        this.detail = "";
        this.render();
      }), "返回调整选择"),
      submitBatch,
    );
    root.append(
      preview,
      bindLocalizedText(text("p", "character-intro", ""), "{{v0}} · 画布 {{v1}} · {{v2}} 个纹理区域 · 纹理之间及画布边缘至少 {{v3}}px 间距。参考图全部独立发送，不占用编辑画布。", { v0: textureImageModelLabel(plan.imageModel), v1: plan.imageSize, v2: plan.items.length, v3: plan.gutter }),
      referenceSummary,
      labelled("最终提交提示词（仍可手动编辑）", prompt),
      optimization,
      referenceDelivery,
    );
    if (this.detail) root.append(text("div", "character-error", this.detail));
    root.append(actions);
  }

  private textureReferenceDeliverySection(draft: TextureBatchDraft): HTMLElement {
    const section = document.createElement("section");
    section.className = "texture-reference-delivery";
    const useOptimizedPrompt = texturePromptSubmissionChoice(
      draft.optimizedPrompt,
      draft.optimizationState,
    ) === "optimized";
    const references = draft.references.map((reference, index) => ({
      selectionId: reference.id,
      promptId: texturePromptIdsForReferences(draft.references)[index]!,
    }));
    const states = textureReferenceDeliveryState(
      useOptimizedPrompt,
      references,
      draft.optionalReferenceOverrides,
    );
    const stateById = new Map(states.map((state) => [state.selectionId, state]));
    section.append(
      bindLocalizedText(text("h3", "", ""), "最终发送的参考图"),
      bindLocalizedText(text("p", "character-intro", ""), (useOptimizedPrompt ? "已选择的参考图默认发送；你可以手动取消不需要的图片。" : "本次未采用优化版，所有已选择的参考图都会继续发送。")),
    );
    if (draft.references.length === 0) {
      section.append(bindLocalizedText(text("small", "", ""), "本次没有额外参考图。"));
      return section;
    }
    const list = document.createElement("div");
    list.className = "texture-reference-delivery-list";
    for (const [index, reference] of draft.references.entries()) {
      const state = stateById.get(reference.id);
      if (!state) continue;
      const row = document.createElement("label");
      row.className = "texture-reference-delivery-row";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = state.send;
      checkbox.disabled = state.locked;
      checkbox.addEventListener("change", () => {
        const overrides = new Set(draft.optionalReferenceOverrides);
        if (checkbox.checked) overrides.add(state.selectionId);
        else overrides.delete(state.selectionId);
        draft.optionalReferenceOverrides = [...overrides];
      });
      const status = state.send ? "将发送" : "不发送";
      row.append(
        checkbox,
        text("strong", "", `${references[index]!.promptId} · ${reference.name}`),
        text("small", "", status),
      );
      list.append(row);
    }
    section.append(list);
    return section;
  }

  private async startTextureBatch(draft: TextureBatchDraft): Promise<void> {
    if (texturePromptSubmissionChoice(draft.optimizedPrompt, draft.optimizationState) === "pending") {
      this.detail = "请先选择采用优化版，或继续使用当前提示词，再开始批量生成";
      return this.render();
    }
    const finalPrompt = draft.finalPrompt.trim();
    if (!finalPrompt) {
      this.detail = "最终提示词不能为空";
      return this.render();
    }
    const useOptimizedPrompt = texturePromptSubmissionChoice(
      draft.optimizedPrompt,
      draft.optimizationState,
    ) === "optimized";
    const promptIds = texturePromptIdsForReferences(draft.references);
    const delivery = textureReferenceDeliveryState(
      useOptimizedPrompt,
      draft.references.map((reference, index) => ({
        selectionId: reference.id,
        promptId: promptIds[index]!,
      })),
      draft.optionalReferenceOverrides,
    );
    const deliveryBySelectionId = new Map(delivery.map((item) => [item.selectionId, item]));
    const textureReferences = draft.references.flatMap((reference) => {
      const item = deliveryBySelectionId.get(reference.id);
      if (!item?.send) return [];
      return [{
        ...reference,
        promptId: item.promptId,
        deliveryUse: item.required
          ? "required" as TextureReferenceDeliveryUse
          : "supplemental" as TextureReferenceDeliveryUse,
      }];
    });
    try {
      this.current = await window.desktopPet.createCharacterJob({
        operation: "texture-edit",
        templateId: draft.capability.templateId,
        description: finalPrompt,
        originalDescription: draft.originalPrompt,
        optimizedDescription: draft.optimizedPrompt,
        promptOptimization: draft.optimizationState,
        promptVisualSpec: draft.optimizationState === "accepted" ? draft.optimizationSpec : null,
        textureBatchPlanHash: draft.plan.hash,
        textureImageModel: draft.plan.imageModel,
        textureReferences,
        referenceImage: null,
        sourceModelId: draft.capability.modelId,
        editMode: "appearance",
        textureUnitIds: draft.textureUnitIds,
      });
    } catch (error) {
      this.detail = error instanceof Error ? error.message : String(error);
      return this.render();
    }
    this.textureBatchDraft = null;
    this.progress = 0;
    this.detail = `正在准备 ${draft.textureUnitIds.length} 个纹理的联合画布，只会提交 1 次图片请求…`;
    this.render();
  }

  private async start(templateId: string, description: string, sourceModelId: string): Promise<void> {
    if (!description.trim()) {
      this.detail = "请输入角色描述";
      return this.render();
    }
    if (sourceModelId) {
      await this.dependencies.prepareModelReference(sourceModelId);
      this.reference = null;
    }
    this.reference = null;
    this.current = await window.desktopPet.createCharacterJob({
      operation: "eye-restyle",
      templateId,
      description,
      referenceImage: null,
      sourceModelId: sourceModelId || null,
    });
    this.progress = 0;
    this.detail = "正在提交生成任务…";
    this.render();
  }

  private renderCandidates(root: HTMLElement, job: CharacterGenerationJob): void {
    root.append(
      bindLocalizedText(text("h3", "", ""), "第一次确认：选择角色定稿"),
      bindLocalizedText(text("p", "character-intro", ""), "本批角色描述：{{v0}}", { v0: job.input.description }),
    );
    const grid = document.createElement("div");
    grid.className = "character-candidates";
    for (const candidate of job.candidates) {
      const card = document.createElement("div");
      card.className = "character-candidate";
      const image = document.createElement("img");
      image.src = candidate.previewUrl ?? `pet-generation://${job.id}/${candidate.id}.png`;
      bindLocalizedAttribute(image, "alt", "候选角色 {{v0}}", { v0: candidate.id });
      const select = bindLocalizedText(button("", async () => {
        this.current = await window.desktopPet.selectCharacterCandidate(job.id, candidate.id);
        this.progress = 0;
        this.render();
      }), (candidate.compatible ? "选择并生成网格投影预览 · 预计 $0.040" : "候选图无效"));
      select.toggleAttribute("disabled", !candidate.compatible);
      card.classList.toggle("character-candidate-rejected", !candidate.compatible);
      card.append(image);
      if (!candidate.compatible) {
        card.append(bindLocalizedText(text("p", "character-candidate-reason", ""), "{{copy}}", () => ({ copy: (candidate.rejectionReason ?? t("未通过母版兼容性检查")) })));
      }
      card.append(select);
      grid.append(card);
    }
    const estimate = job.input.referenceImage ? "0.140" : "0.080";
    root.append(
      grid,
      bindLocalizedText(button("", async () => {
        this.current = await window.desktopPet.regenerateCharacterCandidates(job.id);
        this.progress = 0;
        this.detail = "正在使用姿态骨架重新生成角色外观…";
        this.render();
      }), "这批外观不符合描述，重新生成 4 张 · 预计不超过 ${{v0}}", { v0: estimate }),
      this.taskActions(job),
    );
  }

  private renderProcessing(root: HTMLElement, job: CharacterGenerationJob): void {
    const progressDetail = this.detail;
    if (job.input.operation === "texture-edit") {
      const values = Object.values(job.artifacts.textureEdits ?? {});
      const completed = values.filter((item) => ["review", "accepted", "failed", "cancelled"].includes(item.status)).length;
      const running = values.find((item) => item.status === "running");
      const unit = running
        ? this.capability?.textureUnits.find((item) => item.id === running.textureUnitId)
        : null;
      const details = document.createElement("div");
      details.className = "character-processing-details";
      const imageModel = job.input.textureImageModel ?? DEFAULT_TEXTURE_EDIT_IMAGE_MODEL;
      details.append(
        bindLocalizedText(text("strong", "", ""), "联合纹理处理 {{v0}}/{{v1}}", { v0: completed, v1: values.length || job.input.textureUnitIds?.length || 0 }),
        bindLocalizedText(text("span", "", ""), "{{copy}}", () => ({ copy: progressDetail || (unit ? t("正在处理：{{v0}}", { v0: unit.label }) : t("正在准备联合纹理编辑请求…")) })),
      );
      root.append(
        bindLocalizedText(text("h3", "", ""), "正在使用 {{v0}} 生成", { v0: textureImageModelLabel(imageModel) }),
        details,
        progressBar(this.progress),
        bindLocalizedText(text("p", "character-intro", ""), "所选纹理共用一张画布和一次请求；返回后仍会按纹理独立裁切、检查和验收。"),
        this.taskActions(job, true),
      );
      return;
    }
    if (job.input.operation === "eye-restyle" || job.input.operation === "part-restyle") {
      const label = this.editPart(job)?.label ?? (job.input.operation === "eye-restyle" ? "眼睛" : "所选部位");
      const details = document.createElement("div");
      details.className = "character-processing-details";
      details.append(
        bindLocalizedText(text("strong", "", ""), "正在安全修改{{v0}}", { v0: label }),
        bindLocalizedText(text("span", "", ""), "{{copy}}", () => ({ copy: (this.detail || t("正在恢复任务状态…")) })),
      );
      root.append(
        bindLocalizedText(text("h3", "", ""), "仅处理{{v0}}的原有可见像素", { v0: label }),
        details,
        progressBar(this.progress),
        bindLocalizedText(text("p", "character-intro", ""), "程序会强制还原编辑区以外的所有像素，并保留原网格边缘、透明度、UV 和动作。"),
        this.taskActions(job, true),
      );
      return;
    }
    const template = this.templates.find((item) => item.id === job.templateId);
    const materialCount = Object.keys(job.artifacts.completionSheets).length;
    const count = template?.slotCount ?? null;
    const summary = job.phase === "decomposing"
      ? job.artifacts.normalizedImage === null
        ? "正在将五官、关节和全身比例对齐到母版"
        : job.artifacts.subjectImage === null
          ? "正在提取对齐人物的透明边缘"
          : `正在核对坐标并投影到${count === null ? "母版" : ` ${count} 个网格`}`
      : job.phase === "completing"
        ? `局部补全素材 ${materialCount}/${job.artifacts.requestedCompletions.length}`
      : phaseProgressLabel(job.phase);
    const details = document.createElement("div");
    details.className = "character-processing-details";
    details.append(
      text("strong", "", summary),
      bindLocalizedText(text("span", "", ""), "{{copy}}", () => ({ copy: (this.detail || t("正在恢复任务状态…")) })),
    );
    root.append(
      bindLocalizedText(text("h3", "", ""), phaseLabel(job.phase)),
      details,
      progressBar(this.progress),
      bindLocalizedText(text("p", "character-intro", ""), "可以关闭此窗口，任务会在后台继续；重新打开后会继续显示当前步骤。"),
      this.taskActions(job, true),
    );
  }

  private renderBudget(root: HTMLElement, job: CharacterGenerationJob): void {
    const pending = job.budget.pendingOperation?.estimatedUsd ?? 0;
    const suggested = Math.min(10, Math.max(job.budget.limitUsd + 0.25, job.budget.estimatedUsd + pending));
    const input = document.createElement("input");
    input.type = "number";
    input.min = String(job.budget.limitUsd + 0.01);
    input.max = "10";
    input.step = "0.05";
    input.value = suggested.toFixed(2);
    const box = document.createElement("div");
    box.className = "character-preview-confirm";
    box.append(
      bindLocalizedText(text("h3", "", ""), "达到费用上限，任务已在调用前暂停"),
      bindLocalizedText(text("p", "character-intro", ""), "当前上限 ${{v0}}，已累计预计 ${{v1}}。只有你确认后才会继续扣费。", { v0: job.budget.limitUsd.toFixed(2), v1: job.budget.estimatedUsd.toFixed(2) }),
      labelled("新的硬上限（USD）", input),
      bindLocalizedText(button("", async () => {
        this.current = await window.desktopPet.approveCharacterBudget(job.id, Number(input.value));
        this.render();
      }), "确认新上限并继续"),
    );
    root.append(box, this.taskActions(job));
  }

  private renderParts(root: HTMLElement, job: CharacterGenerationJob): void {
    root.append(bindLocalizedText(text("h3", "", ""), "发现需要返工的部件"));
    const issues = document.createElement("div");
    issues.className = "character-issues";
    for (const issue of job.issues) {
      const row = text("div", "character-issue", issue.message);
      if (issue.slotId) row.append(bindLocalizedText(button("", async () => {
        this.current = await window.desktopPet.regenerateCharacterPart(job.id, issue.slotId!);
        this.render();
      }), "只重生成此部件"));
      issues.append(row);
    }
    const materialIssueCount = new Set(job.issues.map((issue) => issue.slotId).filter(Boolean)).size;
    root.append(issues);
    if (materialIssueCount > 1) {
      root.append(
        bindLocalizedText(text("p", "character-intro", ""), "这批四类语义蒙版不可继续使用。可以保留已选角色和标准姿势，直接使用完整人物透明图生成基础 Live2D 网格预览，不再拆分四类大蒙版。"),
        bindLocalizedText(button("", async () => {
          this.current = await window.desktopPet.regenerateCharacterPart(job.id, ALL_MATERIALS_REPAIR_ID);
          this.render();
        }), "使用原图生成基础预览"),
      );
    }
    root.append(this.taskActions(job));
  }

  private renderBasePreview(root: HTMLElement, job: CharacterGenerationJob): void {
    const layout = document.createElement("div");
    layout.className = "character-confirm-layout";
    const preview = this.livePreview("基础网格动画预览");
    const box = text("div", "character-preview-confirm", "");
    const templateName = this.templates.find((item) => item.id === job.templateId)?.name ?? job.templateId;
    box.append(
      bindLocalizedText(text("h3", "", ""), "第二次确认：检查基础网格对齐"),
      bindLocalizedText(text("p", "character-intro", ""), "这是“{{v0}}”固定网格的换皮技术验证：完整透明人物图已按每个 Drawable 的中立姿势三角网格反投影到纹理图集。请先检查静止外观、转头和四肢跟随；闭眼、张嘴内部和被遮挡区域属于下一步局部补全。", { v0: templateName }),
    );
    for (const issue of job.issues.filter((item) => item.code === "projection-coverage-warning")) {
      box.append(text("p", "character-candidate-reason", issue.message));
    }
    const selected = new Set<CharacterCompletionKind>();
    const options = document.createElement("div");
    options.className = "character-issues";
    const proceed = button("", async () => {
      this.current = await window.desktopPet.completeCharacterModel(job.id, [...selected]);
      this.render();
    });
    const updateLabel = (): void => {
      const cost = selected.size * 0.035;
      bindLocalizedText(proceed, "{{copy}}", () => ({ copy: (selected.size === 0 ? t("对齐正常，不补全并生成最终预览（不增加费用）") : t("确认对齐并生成 {{v0}} 组局部素材 · 预计 ${{v1}}", { v0: selected.size, v1: cost.toFixed(3) })) }));
    };
    for (const option of COMPLETION_OPTIONS) {
      const row = document.createElement("label");
      row.className = "character-issue";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = option.selected;
      if (checkbox.checked) selected.add(option.id);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) selected.add(option.id);
        else selected.delete(option.id);
        updateLabel();
      });
      row.append(checkbox, bindLocalizedText(text("span", "", ""), "{{v0}} · {{v1}}", () => ({ v0: t(option.label), v1: t(option.detail) })));
      options.append(row);
    }
    updateLabel();
    box.append(
      bindLocalizedText(button("", async () => {
        const restarted = await window.desktopPet.regenerateCharacterPart(job.id, REALIGN_TO_TEMPLATE_ID);
        await this.refreshAfterPreviewRemoval(job, restarted);
        this.current = restarted;
        this.progress = 0;
        this.detail = "正在重新生成受母版几何约束的候选…";
        this.render();
      }), "发现错位：重新生成母版姿势候选 · 预计不超过 ${{v0}}", { v0: job.input.referenceImage ? "0.140" : "0.080" }),
      bindLocalizedText(text("p", "character-intro", ""), "只勾选基础动画中确实缺失的素材。所有补全结果保持与原图相同画布，且只会绑定到对应的固定 Drawable 槽位。"),
      options,
      proceed,
    );
    layout.append(preview.card, box);
    root.append(layout, this.taskActions(job));
    this.attachLivePreview(preview.host, job);
  }

  private renderPreview(root: HTMLElement, job: CharacterGenerationJob): void {
    const layout = document.createElement("div");
    layout.className = "character-confirm-layout";
    const preview = this.livePreview("局部补全动画预览");
    const box = text("div", "character-preview-confirm", "");
    box.append(
      bindLocalizedText(text("h3", "", ""), "第三次确认：检查局部补全后的 Live2D 预览"),
      bindLocalizedText(text("p", "character-intro", ""), "左侧预览会自动播放待机、眨眼、呼吸和口型。请确认转头、眼嘴、遮挡和动作没有明显异常，再保存到普通模型库。"),
    );
    const slot = document.createElement("select");
    for (const slotId of Object.keys(job.artifacts.slotTextures)) slot.append(new Option(slotId, slotId));
    box.append(
      labelled("有问题时只返工这个部件", slot),
      bindLocalizedText(button("", async () => {
        const restarted = await window.desktopPet.regenerateCharacterPart(job.id, slot.value);
        await this.refreshAfterPreviewRemoval(job, restarted);
        this.current = restarted;
        this.render();
      }), "重生成所选部件"),
      bindLocalizedText(button("", async () => {
        this.current = await window.desktopPet.confirmCharacterModel(job.id);
        await this.dependencies.refreshModels(job.outputModelId ?? undefined);
        this.render();
      }), "确认并保存为普通模型"),
    );
    layout.append(preview.card, box);
    root.append(layout, this.taskActions(job));
    this.attachLivePreview(preview.host, job);
  }

  private renderTextureResults(root: HTMLElement, job: CharacterGenerationJob): void {
    const results = Object.values(job.artifacts.textureEdits ?? {});
    const acceptedCount = acceptedTextureResultCount(job);
    root.append(
      bindLocalizedText(text("h3", "", ""), "逐个检查纹理结果"),
      bindLocalizedText(text("p", "character-intro", ""), "共 {{v0}} 个独立结果，已接受 {{v1}} 个。自动检测只提供参考，最终是否采用完全由你决定；只有明确接受的纹理才会保存到当前模型的素材库。", { v0: results.length, v1: acceptedCount }),
    );
    const grid = document.createElement("div");
    grid.className = "texture-result-grid";
    for (const result of results) {
      const unit = this.capability?.textureUnits.find((item) => item.id === result.textureUnitId);
      const card = document.createElement("section");
      card.className = `texture-result-card texture-result-${result.status}`;
      const heading = document.createElement("div");
      heading.className = "texture-result-heading";
      heading.append(
        text("strong", "", unit?.label ?? result.textureUnitId),
        bindLocalizedText(text("span", "texture-result-status", ""), textureStatusLabel(result.status)),
      );
      const images = document.createElement("div");
      images.className = "texture-result-images";
      const sourcePreviewPath = result.sourcePreviewPath ?? result.sourcePath;
      const editedPreviewPath = result.editedPreviewPath ?? result.editedPatchPath;
      if (sourcePreviewPath) {
        images.append(textureImage("原始纹理", generationAssetUrl(job.id, sourcePreviewPath)));
      }
      if (editedPreviewPath) {
        images.append(textureImage("AI 结果", generationAssetUrl(job.id, editedPreviewPath)));
      } else {
        images.append(textureImagePlaceholder(textureResultPlaceholder(result.status)));
      }
      const details = bindLocalizedText(text("small", "texture-result-detail", ""), "{{copy}}", () => ({ copy: (unit ? t("图集 {{v0}} · {{v1}}×{{v2}} · {{v3}} 个 ArtMesh 共用", { v0: unit.atlasIndex, v1: unit.bounds.width, v2: unit.bounds.height, v3: unit.drawableIds.length }) : result.textureUnitId) }));
      card.append(heading, images, details);
      if (result.quality) {
        const quality = result.quality;
        card.append(bindLocalizedText(text("small", quality.accepted ? "texture-quality-ok" : "texture-quality-warning", ""), "{{copy}}", () => ({ copy: (quality.accepted ? t("自动检查未发现明显问题（仅供参考） · 修改像素 {{v0}}% · 透明度修改 {{v1}}% · 透明面积减少 {{v2}}% · 清晰度 {{v3}}%", { v0: (quality.changedPixelRatio * 100).toFixed(1), v1: (quality.alphaChangedPixelRatio * 100).toFixed(1), v2: (quality.removedAlphaRatio * 100).toFixed(1), v3: (quality.sharpnessRatio * 100).toFixed(0) }) : t("自动检查提醒（仅供参考，不影响接受）：{{v0}}", { v0: quality.issues.join("；") })) })));
      }
      if (result.error) card.append(text("p", "character-error", result.error));
      const actions = document.createElement("div");
      actions.className = "character-actions";
      if (result.status === "review") {
        actions.append(bindLocalizedText(button("", async () => {
          const change = await window.desktopPet.acceptCharacterTexture(job.id, result.textureUnitId, true);
          this.current = change.job;
          this.pendingTexturePreview = await deliverAcceptedTexturePreviewChange(
            change.job,
            change.delta,
            (delta) => this.dependencies.previewTextureDelta(delta),
          );
          this.render();
        }), "接受此纹理"));
      }
      if (result.status === "accepted") {
        actions.append(bindLocalizedText(button("", async () => {
          const change = await window.desktopPet.acceptCharacterTexture(job.id, result.textureUnitId, false);
          this.current = change.job;
          this.pendingTexturePreview = await deliverAcceptedTexturePreviewChange(
            change.job,
            change.delta,
            (delta) => this.dependencies.previewTextureDelta(delta),
          );
          this.render();
        }), "取消采用"));
      }
      card.append(actions);
      grid.append(card);
    }
    root.append(grid);
    const adjustment = textureAdjustmentReturnState(job);
    const returnToAdjustment = async (): Promise<void> => {
      const remembered = this.textureSelection?.templateId === job.templateId
        ? this.textureSelection
        : null;
      await window.desktopPet.saveAcceptedCharacterTexturesAndCancel(job.id);
      this.textureSelection = {
        templateId: job.templateId,
        ids: adjustment.selectedTextureUnitIds,
        prompt: adjustment.originalPrompt,
        imageModel: adjustment.imageModel,
        references: remembered?.references.map((reference) => ({ ...reference })) ?? [],
      };
      this.textureBatchDraft = null;
      this.current = null;
      this.progress = 0;
      this.detail = "";
      this.render();
    };
    if (acceptedCount > 0) {
      const layout = document.createElement("div");
      layout.className = "character-confirm-layout texture-final-preview";
      const preview = this.livePreview("已接受纹理的模型动画预览");
      const box = text("div", "character-preview-confirm", "");
      box.append(
        bindLocalizedText(text("h3", "", ""), "保存 {{v0}} 个已采用纹理并返回调整", { v0: acceptedCount }),
        bindLocalizedText(text("p", "character-intro", ""), "已采用纹理会保存到当前模型素材库并立即应用；其他结果会丢弃。返回后保留原始提示词和参考图，并默认只勾选尚未采用的纹理。"),
        bindLocalizedText(button("", returnToAdjustment), "保存已采用结果并返回调整"),
      );
      layout.append(preview.card, box);
      root.append(layout);
      this.attachLivePreview(preview.host, job);
    } else {
      const box = text("section", "character-preview-confirm texture-return-adjustment", "");
      box.append(
        bindLocalizedText(text("h3", "", ""), "返回调整"),
        bindLocalizedText(text("p", "character-intro", ""), "当前没有采用任何结果。返回后模型不会改变，原选择、原始提示词和参考图都会保留。"),
        bindLocalizedText(button("", returnToAdjustment), "取消任务并返回调整"),
      );
      root.append(box);
    }
  }

  private renderRestylePreview(root: HTMLElement, job: CharacterGenerationJob): void {
    const part = this.editPart(job);
    const label = part?.label ?? (job.input.operation === "eye-restyle" ? "眼睛" : "所选部位");
    const regenerateId = job.input.operation === "eye-restyle" ? "eyes" : job.input.editPartId ?? "";
    const layout = document.createElement("div");
    layout.className = "character-confirm-layout";
    const preview = this.livePreview(`${label}动画安全预览`);
    const box = text("div", "character-preview-confirm", "");
    box.append(
      bindLocalizedText(text("h3", "", ""), "确认{{v0}}效果", { v0: label }),
      bindLocalizedText(text("p", "character-intro", ""), "请检查待机、转头、眨眼、口型、表情和动作。除{{v0}}对应的预检网格外，其余模型贴图均直接复制自原模型。", { v0: label }),
      bindLocalizedText(button("", async () => {
        const restarted = await window.desktopPet.regenerateCharacterPart(job.id, regenerateId);
        await this.refreshAfterPreviewRemoval(job, restarted);
        this.current = restarted;
        this.progress = 0;
        this.detail = `正在重新生成${label}…`;
        this.render();
      }), "不满意，重新生成{{v0}}", { v0: label }),
      bindLocalizedText(button("", async () => {
        this.current = await window.desktopPet.confirmCharacterModel(job.id);
        await this.dependencies.refreshModels(job.outputModelId ?? undefined);
        this.render();
      }), "确认并保存为新模型"),
    );
    layout.append(preview.card, box);
    root.append(layout, this.taskActions(job));
    this.attachLivePreview(preview.host, job);
  }

  private livePreview(title: string): { card: HTMLElement; host: HTMLElement } {
    const card = text("section", "character-live-preview", "");
    const host = text("div", "character-live-preview-host", "");
    card.append(
      text("h3", "", title),
      host,
      bindLocalizedText(text("p", "character-intro", ""), "预览会自动播放转头、眨眼、呼吸和啊、诶、衣、哦、乌的口型；请观察运动过程中是否露空、错位或变形。"),
    );
    return { card, host };
  }

  private attachLivePreview(host: HTMLElement, job: CharacterGenerationJob): void {
    const modelId = characterPreviewModelId(job);
    if (!modelId) {
      console.warn("AI 动画预览缺少任务模型");
      return;
    }
    const epoch = this.previewAttachmentEpoch;
    this.previewRequested = true;
    void this.dependencies.attachPreview(host, modelId).then(async () => {
      if (!this.isCurrentPreviewAttachment(host, job.id, epoch)) return;
      let delta: TextureAtlasDelta | null = null;
      const pending = this.pendingTexturePreview;
      if (pending?.jobId === job.id) {
        delta = pending.delta;
      } else if (
        job.input.operation === "texture-edit" &&
        job.phase === "preview" &&
        Object.values(job.artifacts.textureEdits ?? {}).some((item) => item.status === "accepted")
      ) {
        delta = await window.desktopPet.setCharacterTexturePreview(job.id, true);
      }
      if (!this.isCurrentPreviewAttachment(host, job.id, epoch)) return;
      if (delta) await this.dependencies.previewTextureDelta(delta);
      if (!this.isCurrentPreviewAttachment(host, job.id, epoch)) return;
      if (this.pendingTexturePreview === pending) this.pendingTexturePreview = null;
      this.dependencies.setPreviewPaused(false);
    }).catch((error) => console.warn("恢复 AI 动画预览失败", error));
  }

  private isCurrentPreviewAttachment(
    host: HTMLElement,
    jobId: string,
    epoch: number,
  ): boolean {
    return (
      epoch === this.previewAttachmentEpoch &&
      host.isConnected &&
      !required("character-wizard").hidden &&
      this.current?.id === jobId
    );
  }

  private renderFailure(root: HTMLElement, job: CharacterGenerationJob): void {
    root.append(
      bindLocalizedText(text("h3", "character-error", ""), (job.phase === "cancelled" ? "任务已取消" : "任务失败")),
      ...job.issues.map((issue) => text("p", "character-error", issue.message)),
    );
    if (job.phase === "failed") root.append(bindLocalizedText(button("", async () => {
      this.current = await window.desktopPet.retryCharacterJob(job.id);
      this.render();
    }), "重试"));
    root.append(bindLocalizedText(button("", async () => {
      this.current = null;
      this.render();
    }), "新建部位修改"));
  }

  private renderDone(root: HTMLElement, job: CharacterGenerationJob): void {
    const label = job.input.operation === "texture-edit"
      ? "纹理"
      : this.editPart(job)?.label ?? (job.input.operation === "eye-restyle" ? "眼睛" : "角色");
    const textureEdit = job.input.operation === "texture-edit";
    root.append(
      bindLocalizedText(text("h3", "", ""), "{{v0}}修改已保存", { v0: label }),
      bindLocalizedText(text("p", "character-intro", ""), "{{copy}}", () => ({ copy: (textureEdit ? t("修改已保存为当前模型下的独立素材并应用。可以在“换装”中按部位筛选并随时切换任意素材；不会再为这次修改新增模型。") : t("新模型保留原来的动作、表情、物理和口型，只替换了{{v0}}的安全纹理区域；透明轮廓只可能缩小，不会向外扩大。", { v0: label })) })),
      bindLocalizedText(button("", async () => this.close()), "完成"),
    );
  }

  private editPart(job: CharacterGenerationJob): ModelEditPartCapability | null {
    const id = job.input.editPartId;
    return id ? this.capability?.parts.find((part) => part.id === id) ?? null : null;
  }

  private async refreshAfterPreviewRemoval(
    previous: CharacterGenerationJob,
    next: CharacterGenerationJob,
  ): Promise<void> {
    if (!previous.outputModelId || next.outputModelId === previous.outputModelId) return;
    await this.dependencies.refreshModels(next.input.sourceModelId ?? undefined);
  }

  private renderCost(root: HTMLElement, job: CharacterGenerationJob): void {
    const row = document.createElement("div");
    row.className = "character-cost";
    row.append(
      bindLocalizedText(text("span", "", ""), "本任务预计累计 ${{v0}}", { v0: job.budget.estimatedUsd.toFixed(3) }),
      bindLocalizedText(text("span", "", ""), "本任务硬上限 ${{v0}}", { v0: job.budget.limitUsd.toFixed(2) }),
    );
    root.append(row);
  }

  private clearTextureEditorState(): void {
    this.textureSelection = null;
    this.textureBatchDraft = null;
    this.requestedPartId = null;
    this.reference = null;
    this.progress = 0;
    this.detail = "";
  }

  private taskActions(job: CharacterGenerationJob, canPause = false): HTMLElement {
    const row = document.createElement("div");
    row.className = "character-actions";
    if (canPause && !job.paused) row.append(bindLocalizedText(button("", async () => {
      this.current = await window.desktopPet.pauseCharacterJob(job.id);
      this.render();
    }), "暂停"));
    if (job.paused && job.pauseReason === "user") row.append(bindLocalizedText(button("", async () => {
      this.current = await window.desktopPet.resumeCharacterJob(job.id);
      this.render();
    }), "继续"));
    if (!['saved', 'cancelled'].includes(job.phase)) row.append(bindLocalizedText(button("", async () => {
      if (!window.confirm(t("取消任务并清理中间素材？"))) return;
      if (job.input.operation === "texture-edit" && job.phase === "preview") {
        await this.dependencies.previewTextureDelta(
          await window.desktopPet.setCharacterTexturePreview(job.id, false),
        );
      }
      const cancelled = await window.desktopPet.cancelCharacterJob(job.id);
      await this.refreshAfterPreviewRemoval(job, cancelled);
      if (job.input.operation === "texture-edit") this.clearTextureEditorState();
      this.current = cancelled;
      this.render();
    }), "取消任务"));
    return row;
  }
}

export function acceptedTextureResultCount(job: CharacterGenerationJob): number {
  return Object.values(job.artifacts.textureEdits ?? {})
    .filter((item) => item.status === "accepted").length;
}

/**
 * A final unaccept removes the live-preview host on the next render. Apply its
 * reset snapshot before that happens; snapshots that still contain accepted
 * results are consumed after the replacement host is attached.
 */
export async function deliverAcceptedTexturePreviewChange(
  job: CharacterGenerationJob,
  delta: TextureAtlasDelta,
  applyImmediately: (delta: TextureAtlasDelta) => Promise<void>,
): Promise<{ jobId: string; delta: TextureAtlasDelta } | null> {
  if (acceptedTextureResultCount(job) > 0) return { jobId: job.id, delta };
  await applyImmediately(delta);
  return null;
}

export function textureUnitMotionNames(
  motionIds: string[],
  model: ModelPackage | undefined,
): string[] {
  if (!model) return [];
  return motionIds.flatMap((id) => {
    const asset = model.manifest.motionAssets[id];
    return asset ? [asset.name] : [];
  });
}

function progressBar(value: number): HTMLElement {
  const root = document.createElement("div");
  root.className = "character-progress";
  const fill = document.createElement("span");
  fill.style.width = `${Math.max(2, Math.min(100, value * 100))}%`;
  root.append(fill);
  return root;
}

function textureImageModelLabel(model: TextureEditImageModel): string {
  if (model === "gpt-image-1.5") return "Image 1.5（历史任务）";
  if (model === "gpt-image-2") return "Image 2（历史任务）";
  return "Image 2.5";
}

function partOptionLabel(part: ModelEditPartCapability): string {
  return `${part.label} · ${part.drawableIds.length} 个 ArtMesh`;
}

function partDescriptionPlaceholder(part: ModelEditPartCapability | undefined): string {
  if (!part) return "描述希望修改的颜色、纹理、明暗或妆容";
  if (part.group === "iris") return "例如：淡粉色虹膜、深色瞳孔、细腻放射纹理和柔和星形高光";
  if (part.group.startsWith("hair")) return "例如：把原有发丝改成深栗色，保留原轮廓、明暗和线稿结构";
  if (part.group === "clothing") return "例如：改为低饱和淡粉配色和细小针织纹理，保留原款式与轮廓";
  if (part.group === "face") return "例如：肤色更通透自然，增加非常轻微的暖粉腮红，不改变五官和脸型";
  return `例如：调整${part.label}的颜色与表面纹理，保持原轮廓和结构`;
}

function modelAssetUrl(modelId: string, relativePath: string, revision?: string): string {
  const relative = relativePath.replaceAll("\\", "/").replace(/^\/+/, "");
  const base = `pet-model://${encodeURIComponent(modelId)}/${relative.split("/").map(encodeURIComponent).join("/")}`;
  return revision ? `${base}?revision=${encodeURIComponent(revision)}` : base;
}

function generationAssetUrl(jobId: string, absolutePath: string): string {
  const normalized = absolutePath.replaceAll("\\", "/");
  const name = normalized.slice(normalized.lastIndexOf("/") + 1);
  return `pet-generation://${encodeURIComponent(jobId)}/${encodeURIComponent(name)}`;
}

function textureImage(label: string, src: string): HTMLElement {
  const figure = document.createElement("figure");
  const image = document.createElement("img");
  image.src = src;
  bindLocalizedAttribute(image, "alt", label);
  figure.append(image, bindLocalizedText(text("small", "", ""), label));
  return figure;
}

function textureImagePlaceholder(label: string): HTMLElement {
  const figure = document.createElement("figure");
  const placeholder = bindLocalizedText(text("div", "texture-result-placeholder", ""), label);
  placeholder.setAttribute("role", "status");
  figure.append(placeholder, bindLocalizedText(text("small", "", ""), "AI 结果"));
  return figure;
}

function createReferenceId(): string {
  return `reference-${crypto.randomUUID()}`;
}

function texturePromptIdsForReferences(
  references: readonly TextureReferenceSelection[],
): TextureReferencePromptId[] {
  let visualIndex = 0;
  return references.map((reference) => reference.source === "model-preview"
    ? "P0"
    : `R${++visualIndex}` as TextureReferencePromptId);
}

function createTextureReferenceSummary(
  references: readonly TextureReferenceSelection[],
): HTMLElement {
  const section = document.createElement("section");
  section.className = "texture-reference-editor";
  section.append(
    bindLocalizedText(text("h3", "", ""), "独立参考图 · {{v0}} 张", { v0: references.length }),
    bindLocalizedText(text("p", "character-intro", ""), (references.length > 0 ? "图片模型接收顺序固定为：第 1 张编辑画布，随后依次为 R1、R2……" : "本次没有额外参考图。")),
  );
  if (references.length === 0) return section;
  const grid = document.createElement("div");
  grid.className = "texture-reference-grid";
  const referenceLabels = textureReferenceLabels(references);
  references.forEach((reference, index) => {
    const card = document.createElement("article");
    card.className = "texture-reference-card";
    const image = document.createElement("img");
    image.src = reference.previewUrl ?? "";
    image.alt = reference.name;
    card.append(
      text("strong", "", `${referenceLabels[index]} · ${reference.name}`),
      image,
      reference.source === "model-preview"
        ? bindLocalizedText(text("small", "", ""), "只用于确认部位位置，不参与外观参考")
        : reference.source === "texture-unit"
          ? bindLocalizedText(text("small", "", ""), "模型素材参考")
          : reference.description.trim()
            ? bindLocalizedText(text("small", "", ""), "说明：{{v0}}", { v0: reference.description.trim() })
            : bindLocalizedText(text("small", "", ""), "用户上传图片（无补充说明）"),
    );
    grid.append(card);
  });
  section.append(grid);
  return section;
}

function createTextureBatchPreview(
  capability: ModelEditCapabilityManifest,
  plan: TextureEditBatchPlan,
  texturePreviewUrl: (unit: TextureEditUnit) => string,
): HTMLDivElement {
  const wrapper = document.createElement("div");
  wrapper.className = "texture-batch-preview";
  const preview = document.createElement("div");
  preview.className = "texture-batch-layout";
  preview.style.aspectRatio = `${plan.canvasWidth} / ${plan.canvasHeight}`;
  preview.style.maxWidth = `${textureBatchPreviewMaxWidth(plan.canvasWidth, plan.canvasHeight)}px`;
  const regions = textureEditBatchPromptRegions(plan);
  const regionByTextureId = new Map(regions.map((region, index) => [
    region.textureUnitId,
    { ...region, index },
  ]));
  const legend = document.createElement("div");
  legend.className = "texture-batch-legend";
  for (const item of plan.items) {
    const unit = capability.textureUnits.find((candidate) => candidate.id === item.textureUnitId);
    if (!unit) continue;
    const region = regionByTextureId.get(item.textureUnitId);
    if (!region) continue;
    const color = textureRegionCssColor(region.index);
    const cell = document.createElement("div");
    cell.className = "texture-batch-cell";
    cell.style.borderColor = color;
    placeBatchElement(cell, {
      x: item.x,
      y: item.y,
      width: item.workWidth,
      height: item.workHeight,
    }, plan);
    const image = document.createElement("img");
    image.src = texturePreviewUrl(unit);
    image.alt = unit.label;
    cell.title = `${region.key} · ${unit.label} · ${item.scale}×`;
    cell.setAttribute("aria-label", cell.title);
    cell.append(image);
    preview.append(cell);
    const legendItem = document.createElement("div");
    legendItem.className = "texture-batch-legend-item";
    const legendKey = text("strong", "texture-batch-legend-key", region.key);
    legendKey.style.backgroundColor = color;
    legendItem.append(
      legendKey,
      text("span", "", `${unit.label} · ${item.scale}×`),
    );
    legend.append(legendItem);
  }
  wrapper.append(preview, legend);
  return wrapper;
}

export function textureBatchPreviewMaxWidth(
  canvasWidth: number,
  canvasHeight: number,
  maxWidth = 760,
  maxHeight = 620,
): number {
  return Math.min(maxWidth, maxHeight * canvasWidth / canvasHeight);
}

function textureRegionCssColor(index: number): string {
  const [red, green, blue] = TEXTURE_EDIT_REGION_COLORS[index % TEXTURE_EDIT_REGION_COLORS.length]!;
  return `rgb(${red}, ${green}, ${blue})`;
}

function placeBatchElement(
  element: HTMLElement,
  bounds: { x: number; y: number; width: number; height: number },
  plan: TextureEditBatchPlan,
): void {
  element.style.left = `${bounds.x / plan.canvasWidth * 100}%`;
  element.style.top = `${bounds.y / plan.canvasHeight * 100}%`;
  element.style.width = `${bounds.width / plan.canvasWidth * 100}%`;
  element.style.height = `${bounds.height / plan.canvasHeight * 100}%`;
}

function readOnlyText(value: string): HTMLTextAreaElement {
  const element = document.createElement("textarea");
  element.value = value;
  element.readOnly = true;
  return element;
}

function textureResultPlaceholder(
  status: NonNullable<CharacterGenerationJob["artifacts"]["textureEdits"]>[string]["status"],
): string {
  if (status === "running") return "正在生成…";
  if (status === "pending") return "等待生成";
  if (status === "cancelled") return "已取消";
  return "未生成结果";
}

function textureStatusLabel(status: NonNullable<CharacterGenerationJob["artifacts"]["textureEdits"]>[string]["status"]): string {
  return {
    pending: "等待",
    running: "生成中",
    review: "待确认",
    accepted: "已接受",
    failed: "失败",
    cancelled: "已取消",
  }[status];
}

function phaseLabel(phase: CharacterGenerationJob["phase"]): string {
  return {
    draft: "正在生成角色候选",
    decomposing: "正在生成基础网格预览",
    aligning: "正在按 Cubism 网格烘焙",
    "base-preview": "基础网格预览已准备好",
    completing: "正在生成局部缺失素材",
    validating: "正在检查全身动作与遮挡",
    preview: "预览已准备好",
    saved: "已保存",
    failed: "生成失败",
    cancelled: "已取消",
  }[phase];
}

function phaseProgressLabel(phase: CharacterGenerationJob["phase"]): string {
  if (phase === "aligning") return "正在将全身素材反向烘焙到 Drawable 网格";
  if (phase === "completing") return "正在按固定槽位生成局部缺失素材";
  if (phase === "validating") return "正在检查眨眼、口型、四肢、服装和遮挡";
  return "正在处理";
}

function labelled(label: string, control: HTMLElement): HTMLLabelElement {
  const root = document.createElement("label");
  root.append(bindLocalizedText(document.createElement("span"), label), control);
  return root;
}

function button(
  label: string,
  action: () => void | Promise<void>,
  type: "button" | "submit" = "button",
): HTMLButtonElement {
  const element = document.createElement("button");
  element.type = type;
  element.textContent = label;
  if (type === "button") element.addEventListener("click", () => void action());
  return element;
}

function text(tag: "div" | "section" | "p" | "h3" | "small" | "span" | "strong", className: string, content: string): HTMLElement {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = content;
  return element;
}

function readableError(error: unknown): string {
  return error instanceof Error && error.message.trim() ? error.message : String(error);
}

function required<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`缺少界面元素 #${id}`);
  return element as T;
}
