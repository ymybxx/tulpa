import { t } from "../i18n";
import { bindLocalizedText, bindLocalizedAttribute } from "../i18n/dom";
import "./styles.css";
import { EmotionSuggestionsPanel } from "./emotion-suggestions";
import { FacialTestPanel } from "./facial-test-panel";
import { targetUsesStatePlayback } from "../shared/behavior-target";

import type {
  BehaviorCandidate,
  ModelAppearanceSnapshot,
  ModelEditCapabilityManifest,
  ModelEditPartCapability,
  ModelImportReport,
  ModelPackage,
  PartRecognitionUpdate,
  TextureAtlasDelta,
  PetManifest,
  TextureEditUnit,
  TextureMaterialChangeResult,
  TextureMaterialLibrary,
  TextureMaterial,
} from "../shared/contracts";
import { resolveLipSyncAdapter } from "../shared/lipsync-adapter";
import { renderNaturalIdlePanel } from "./natural-idle-panel";
import { renderDefaultAppearancePanel } from "./default-appearance-panel";
import { PartGroupsPanel, type PartRecognitionState } from "./part-groups-panel";
import { AiOutfitPanel, type AiOutfitDraft } from "./ai-outfit-panel";
import { openSettings, recall, recallValue, remember } from "../ui/remembered";
import { decisionReady } from "../shared/decision-model";
import { TextureSwatchesPanel } from "./texture-swatches-panel";
import { effectivePartGroups, looseDrawables, moveDrawablesToGroup, ungroupedUnits } from "../shared/part-groups";
import {
  createDraft,
  deleteBehaviorCandidate,
  hasUnsavedChanges,
  isDraftValid,
  renameAsset,
  saveBehaviorCandidate,
  toManifest,
  type AssetKind,
  type BehaviorKind,
  type ModelDraft,
} from "./draft";
import type { ModelManagerPreview } from "./preview";
import { loadPreviewConstructor } from "./preview-loader";
import {
  buildBehaviorRows,
} from "./view-model";
import { CharacterWizard } from "./character-wizard";
import { VirtualGrid } from "./virtual-grid";
import {
  modelManagerTabKeepsPreview,
} from "./preview-resource-policy";
import { loadCurrentModelSelection } from "./model-selection-load";
import {
  canSaveCandidate,
  candidateFromEditor,
  createCandidateEditorState,
  moveCandidateAsset as moveEditorCandidateAsset,
  setCandidateAssetSelected,
  type CandidateEditorState,
} from "./candidate-editor-state";

const LIP_TEST_VOWELS = ["a", "e", "i", "o", "u"] as const;

type OutfitPane = "toggles" | "params" | "textures" | "parts" | "ai";
const LOOK_EDITOR_TABS = ["pools", "facial", "defaults", "idle", "display"] as const;
const OUTFIT_PANES = ["toggles", "params", "textures", "parts", "ai"] as const;
const LOOK_EDITOR_COPY: Record<string, { subtitle: string; latin: string; footer: string }> = {
  pools: { subtitle: "什么状态、什么心情，用哪些动作和表情", latin: "BEHAVIOR", footer: "改动先在左边预览，保存后桌面上的 TA 才会变" },
  facial: { subtitle: "说话时决策模型逐句判断表情，这里调每个部位怎么动", latin: "FACIAL", footer: "设置跟着这个形象保存，换形象不会带过去" },
  defaults: { subtitle: "开关、参数、换色，组合好存成套装", latin: "OUTFIT", footer: "改动先在左边预览，保存后桌面上的 TA 才会变" },
  idle: { subtitle: "没在说话时 TA 的小动作", latin: "IDLE", footer: "选了左边马上动给你看，保存后桌面上的 TA 才会变" },
  display: { subtitle: "显示与口型", latin: "MORE", footer: "形象文件只保存在这台电脑上" },
};
type LipTestVowel = (typeof LIP_TEST_VOWELS)[number];
/** The Live2D vowels are あいうえお, not English letter names. */
const LIP_VOWEL_LABELS: Record<LipTestVowel, string> = { a: "啊 A", e: "诶 E", i: "衣 I", o: "哦 O", u: "乌 U" };

export class ModelManager {
  private preview: ModelManagerPreview | null = null;
  private previewCreation: Promise<ModelManagerPreview> | null = null;
  private models: ModelPackage[] = [];
  private activeId: string | null = null;
  private current: ModelPackage | null = null;
  private draft: ModelDraft | null = null;
  private editCapability: ModelEditCapabilityManifest | null = null;
  private appearanceLoading = false;
  private textureLibrary: TextureMaterialLibrary | null = null;
  private appearanceRequestVersion = 0;
  private initialized = false;
  private active = false;
  private activationVersion = 0;
  private activation: Promise<void> | null = null;
  private previewGeneration = 0;
  private previewRecovery: Promise<void> | null = null;
  private selectionEpoch = 0;
  private requestedModelId: string | null = null;
  private displayedModelId: string | null = null;
  private previewLoadStatus: "idle" | "loading" | "ready" | "failed" = "idle";
  private lipTestTimer: number | null = null;
  private lipTestMicrophoneActive = false;
  private lipTestMicrophoneStarting = false;
  private lipTestMicrophoneVersion = 0;
  private lipTestMicrophoneButton: HTMLButtonElement | null = null;
  private readonly characterWizard: CharacterWizard;
  private readonly previewStage = required("models-preview-stage");
  private layerPanel: ReturnType<typeof renderDefaultAppearancePanel> | null = null;
  private partGroupsPanel: PartGroupsPanel | null = null;
  private partPick: ((drawableIds: string[]) => void) | null = null;
  private requestedEditId: string | null = null;
  private aiOutfitPanel: AiOutfitPanel | null = null;
  /** Unfinished AI 换装 setups by model, kept while the user looks at other panes or pages. */
  private readonly aiOutfitDrafts = new Map<string, AiOutfitDraft>();
  private aiOutfitModelId: string | null = null;
  private swatchesPanel: TextureSwatchesPanel | null = null;
  private swatchPick: ((drawableIds: string[]) => void) | null = null;
  private aiPick: ((drawableIds: string[], point: { x: number; y: number }) => void) | null = null;
  private partRecognition: (PartRecognitionState & { modelId: string }) | null = null;
  private readonly previewHome = this.previewStage.parentElement;
  private readonly previewHomeNextSibling = this.previewStage.nextSibling;
  private materialVirtualGrid: VirtualGrid<TextureEditUnit> | null = null;
  private modelVirtualGrid: VirtualGrid<ModelPackage> | null = null;
  private activeModelTab = "pools";
  private outfitPane: OutfitPane = "toggles";
  private previewResumeEpoch = 0;
  private candidateEditor: CandidateEditorState | null = null;
  /** Behavior rows the person opened, kept open across re-renders. */
  private readonly openBehaviorRows = new Set<string>();
  private readonly facialTest = new FacialTestPanel(required("model-tab-facial"), async () => {
    if (this.lipTestTimer !== null || this.lipTestMicrophoneActive || this.lipTestMicrophoneStarting) {
      this.stopLipTest();
      await this.preview?.stopMicrophoneLipTest();
    }
  });
  private readonly emotionSuggestions = new EmotionSuggestionsPanel(required("model-emotion-suggestions"), {
    draft: () => this.draft,
    changed: () => { this.renderBehaviorViews(); this.renderDirty(); },
    preview: async (suggestion) => {
      await this.requirePreview().previewCandidate(suggestion.candidate, targetUsesStatePlayback(suggestion.target) ? "state" : "transient");
    },
    save: () => this.save(),
  });

  constructor() {
    this.bind();
    window.desktopPet.partRecognition.onUpdate((update) => this.applyPartRecognition(update));
    window.desktopPet.onCommand((command) => {
      if (command.type === "texture-materials-updated") {
        void this.applyTextureMaterialChange(command.change);
      } else if (
        command.type === "texture-compaction-warning" &&
        command.modelId === this.current?.id
      ) {
        this.showReport({
          status: "warning",
          checks: [],
          warnings: [{ code: "texture-cache", message: command.message, file: null }],
          errors: [],
        });
      }
    });
    this.characterWizard = new CharacterWizard({
      models: () => this.models,
      previewModel: async (id) => this.previewGeneratedModel(id),
      previewTextureDelta: async (delta) => this.preview?.updateTextureAtlases(delta),
      alignmentGuideUrl: async (id) => (await window.desktopPet.getModelAlignmentGuide(id)).url,
      refreshModels: async (id) => this.refresh(id),
      prepareModelReference: async (id) => this.prepareModelReference(id),
      texturePreviewUrl: (unit) => this.texturePreviewUrl(unit),
      textureMaterials: (textureUnitId) => this.textureLibrary?.materials
        .filter((material) => material.textureUnitId === textureUnitId)
        .sort((left, right) => right.createdAt.localeCompare(left.createdAt)) ?? [],
      attachPreview: (target, modelId, pick) => this.attachPreview(target, modelId, pick),
      highlightLayers: (ids) => this.preview?.highlightLayers(ids),
      parkPreview: () => this.parkPreview(),
      restorePreview: () => this.restorePreview(),
      setPreviewPaused: (value) => this.preview?.setPaused(value),
    });
  }

  private async attachPreview(target: HTMLElement, modelId: string, pick?: (ids: string[]) => void): Promise<void> {
    this.facialTest.setContext(false, null, null);
    const model = this.models.find((item) => item.id === modelId);
    if (!model) throw new Error(`任务预览模型不存在：${modelId}`);
    this.preview?.setLayerPickHandler(null);
    target.append(this.previewStage);
    const existing = await this.ensurePreview();
    const preview = existing.displayedModelId() === modelId
      ? existing
      : await this.resumePreviewModel(model, false);
    if (
      !target.isConnected ||
      this.previewStage.parentElement !== target ||
      preview.displayedModelId() !== modelId
    ) return;
    if (pick) preview.settle();
    preview.setLayerPickHandler(pick ?? null);
    preview.setPaused(false);
    requestAnimationFrame(() => preview.resize());
  }

  private parkPreview(): void {
    this.preview?.setLayerPickHandler(null);
    this.previewResumeEpoch += 1;
    if (this.previewHome && this.previewStage.parentElement !== this.previewHome) {
      this.previewHome.insertBefore(this.previewStage, this.previewHomeNextSibling);
    }
    requestAnimationFrame(() => this.preview?.resize());
  }

  private async restorePreview(): Promise<void> {
    this.parkPreview();
    await this.resumeCurrentPreview();
    this.syncLayerPicking();
    this.syncFacialTest();
  }

  private async previewGeneratedModel(id: string): Promise<void> {
    if (this.current?.id !== id) await this.refresh(id);
    this.preview?.startValidationSequence();
  }

  private async prepareModelReference(id: string): Promise<void> {
    const model = this.models.find((item) => item.id === id);
    if (!model) throw new Error("参考模型不存在");
    await this.select(model, false);
    const preview = this.requirePreview();
    preview.reset();
    const saved = await window.desktopPet.saveModelPreview(id, await preview.capturePng());
    this.replaceModel(saved);
  }

  /** The next time the editor opens (or now, if open) it edits this look instead of the worn one. */
  edit(modelId: string): void {
    if (this.active && this.initialized) {
      if (this.current?.id !== modelId && this.mayDiscard()) void this.refresh(modelId);
      return;
    }
    this.requestedEditId = modelId;
  }

  async enter(): Promise<void> {
    this.emotionSuggestions.activate();
    if (this.active) {
      this.preview?.resize();
      return this.activation ?? Promise.resolve();
    }
    this.active = true;
    const version = ++this.activationVersion;
    const activation = this.activate(version).finally(() => {
      if (this.activation === activation) this.activation = null;
    });
    this.activation = activation;
    return activation;
  }

  leave(): void {
    this.facialTest.setContext(false, null, null);
    this.emotionSuggestions.deactivate();
    if (!this.active && !this.preview) return;
    this.active = false;
    this.stopLipTest();
    this.activationVersion += 1;
    this.selectionEpoch += 1;
    this.requestedModelId = null;
    this.displayedModelId = null;
    this.previewLoadStatus = "idle";
    this.materialVirtualGrid?.destroy();
    this.materialVirtualGrid = null;
    this.modelVirtualGrid?.destroy();
    this.modelVirtualGrid = null;
    this.editCapability = null;
    this.textureLibrary = null;
    // A group being put together survives the window being hidden for a while;
    // only a preview that was playing stops.
    if (this.candidateEditor?.previewing) {
      this.candidateEditor.previewing = false;
      this.renderBehaviorViews();
    }
    this.releasePreview();
  }

  destroy(): void {
    // The window is closing: keep an unfinished AI 换装 for when it reopens.
    this.parkAiOutfit();
    this.facialTest.destroy();
    this.emotionSuggestions.deactivate();
    this.active = false;
    this.stopLipTest();
    this.activationVersion += 1;
    this.selectionEpoch += 1;
    this.requestedModelId = null;
    this.displayedModelId = null;
    this.previewLoadStatus = "idle";
    this.materialVirtualGrid?.destroy();
    this.materialVirtualGrid = null;
    this.modelVirtualGrid?.destroy();
    this.modelVirtualGrid = null;
    this.editCapability = null;
    this.textureLibrary = null;
    this.candidateEditor = null;
    this.releasePreview();
  }

  private bind(): void {
    required("models-import-folder").addEventListener("click", () => void this.importModel("folder"));
    required("models-import-zip").addEventListener("click", () => void this.importModel("zip"));
    required("models-preview-reset").addEventListener("click", () => {
      this.facialTest.stop(); this.stopLipTest();
      remember("look-editor-preview-mode", "static");
      this.preview?.setPreferredMode(false);
      this.preview?.reset();
    });
    required("models-preview-animate").addEventListener("click", () => {
      this.facialTest.stop(); this.stopLipTest();
      remember("look-editor-preview-mode", "animated");
      this.preview?.setPreferredMode(true);
      this.preview?.playAwakeIdle();
    });
    required("models-save").addEventListener("click", () => void this.save());
    required("models-discard").addEventListener("click", () => this.discard());
    required("model-use").addEventListener("click", () => void this.useCurrent());
    required("model-delete").addEventListener("click", () => void this.deleteCurrent());
    required("models-ai-create").addEventListener("click", () => void this.openAiEditor());
    required("model-edit-scan").addEventListener("click", () => void this.loadAppearance(true));
    required<HTMLInputElement>("model-name").addEventListener("input", (event) => {
      if (!this.draft) return;
      this.draft.manifest.name = (event.currentTarget as HTMLInputElement).value;
      this.renderDirty();
    });
    for (const button of document.querySelectorAll<HTMLButtonElement>("[data-model-tab]")) {
      button.addEventListener("click", () => void this.showTab(button.dataset.modelTab ?? "pools"));
    }
    for (const button of document.querySelectorAll<HTMLButtonElement>("[data-outfit-pane]")) {
      button.addEventListener("click", () => void this.showOutfitPane(button.dataset.outfitPane as OutfitPane));
    }
    required("facial-connect-jev").addEventListener("click", () => openSettings("jev"));
    window.addEventListener("resize", () => this.preview?.resize());
  }

  private async refresh(preferredId?: string, preferActive = false): Promise<void> {
    const [models, active] = await Promise.all([
      window.desktopPet.listModels(),
      window.desktopPet.getSelectedModel(),
    ]);
    this.models = models;
    this.activeId = active?.id ?? null;
    this.renderList();
    const id = preferredId ?? (preferActive ? this.activeId : this.current?.id ?? this.activeId) ?? models[0]?.id;
    const next = models.find((model) => model.id === id) ?? models[0] ?? null;
    if (next) await this.select(next, false);
    else this.showEmpty();
  }

  private async select(model: ModelPackage, confirm = true): Promise<void> {
    if (confirm && !this.mayDiscard()) return;
    remember("look-editor-model", model.id);
    this.facialTest.setContext(false, null, null);
    this.stopLipTest();
    const modelId = model.id;
    const epoch = ++this.selectionEpoch;
    const previewEpoch = ++this.previewResumeEpoch;
    this.requestedModelId = modelId;
    this.previewLoadStatus = "loading";
    this.current = model;
    this.draft = createDraft(model);
    this.emotionSuggestions.resetDraft();
    this.candidateEditor = null;
    this.resetAppearance();
    this.renderList();
    this.renderEditor();
    if (this.usesAppearance()) {
      void this.loadAppearance(false);
    }
    this.showPreviewLoading(model.name);
    try {
      const loaded = await loadCurrentModelSelection({
        loadBundle: () => window.desktopPet.getModelRenderBundle(modelId),
        loadPreview: async (bundle) => (await this.ensurePreview()).load(bundle),
        isCurrent: () => (
          epoch === this.selectionEpoch &&
          previewEpoch === this.previewResumeEpoch &&
          this.current?.id === modelId
        ),
      });
      if (!loaded.current) return;
      const displayed = loaded.result;
      if (displayed !== modelId) throw new Error("预览加载已被更新的模型选择取代");
      this.displayedModelId = displayed;
      this.previewLoadStatus = "ready";
      this.showPreviewReady(model.name);
    } catch (error) {
      if (epoch !== this.selectionEpoch || this.current?.id !== modelId) return;
      this.previewLoadStatus = "failed";
      this.showPreviewFailure(model.name, error);
      this.showError(error);
    }
  }

  private async importModel(kind: "zip" | "folder"): Promise<void> {
    if (!this.mayDiscard()) return;
    const result = await window.desktopPet.importModel(kind);
    if (!result) return;
    this.showReport(result.report);
    if (!result.model) return;
    this.activeId = result.model.id;
    this.models = await window.desktopPet.listModels();
    await this.select(result.model, false);
    try {
      const saved = await window.desktopPet.saveModelPreview(
        result.model.id,
        await this.requirePreview().capturePng(),
      );
      this.replaceModel(saved);
    } catch (error) {
      this.showError(error);
    }
  }

  private async save(): Promise<boolean> {
    if (
      !this.current ||
      !this.draft ||
      !hasUnsavedChanges(this.draft) ||
      !isDraftValid(this.draft)
    ) return false;
    try {
      const model = await window.desktopPet.saveManifest(
        this.current.id,
        toManifest(this.draft),
        this.emotionSuggestions.adoptions(),
      );
      this.emotionSuggestions.resetDraft();
      this.replaceModel(model);
      await this.select(model, false);
      this.showMessage("模型配置已保存并应用");
      return true;
    } catch (error) {
      this.showError(error);
      return false;
    }
  }

  private discard(): void {
    if (!this.current) return;
    this.draft = createDraft(this.current);
    this.emotionSuggestions.resetDraft();
    this.candidateEditor = null;
    this.renderEditor();
    this.preview?.settle();
  }

  private async useCurrent(): Promise<void> {
    if (!this.current) return;
    try {
      const model = await window.desktopPet.selectModel(this.current.id);
      this.activeId = model.id;
      this.replaceModel(model);
      this.renderList();
    } catch (error) {
      this.showError(error);
    }
  }

  private async deleteCurrent(): Promise<void> {
    if (!this.current || !window.confirm(t("删除模型“{{v0}}”及其所有本机素材？", { v0: this.current.name }))) return;
    try {
      const result = await window.desktopPet.deleteModel(this.current.id);
      this.current = null;
      this.draft = null;
      await this.refresh(result.selectedModel.id);
    } catch (error) {
      this.showError(error);
    }
  }

  private async deleteAsset(kind: AssetKind, id: string): Promise<void> {
    if (!this.current || !this.mayDiscard()) return;
    const asset = kind === "motion"
      ? this.current.manifest.motionAssets[id]
      : this.current.manifest.expressionAssets[id];
    if (!asset || !window.confirm(t("删除“{{v0}}”？用到它的表情动作组合也会一起删除。", { v0: asset.name }))) return;
    try {
      const model = await window.desktopPet.deleteModelAsset(this.current.id, kind, id);
      this.replaceModel(model);
      await this.select(model, false);
    } catch (error) {
      this.showError(error);
    }
  }

  /** 存成开关: the pinned values become a model toggle that is switched on in their place. */
  private async createToggle(name: string, values: Record<string, number>): Promise<void> {
    if (!this.current || !this.draft) return;
    const edited = structuredClone(this.draft.manifest);
    const { model, assetId } = await window.desktopPet.createModelToggle(this.current.id, name, values);
    this.replaceModel(model);
    const appearance = edited.defaultAppearance ?? { expressionAssetIds: [] };
    const parameterValues = Object.fromEntries(Object.entries(appearance.parameterValues ?? {})
      .filter(([id]) => !Object.hasOwn(values, id)));
    edited.expressionAssets = Object.fromEntries(Object.entries(model.manifest.expressionAssets)
      .map(([id, asset]) => [id, edited.expressionAssets[id] ?? asset]));
    edited.defaultAppearance = { ...appearance, expressionAssetIds: [...appearance.expressionAssetIds, assetId], parameterValues };
    // The preview has to load the new expression file; unsaved edits carry over.
    await this.select(model, false);
    if (this.current?.id !== model.id || !this.draft) return;
    this.draft.manifest = edited;
    this.renderEditor();
    await this.showOutfitPane("toggles");
    this.showMessage(`已建好开关「${model.manifest.expressionAssets[assetId]?.name ?? name}」，保存后生效`);
  }

  /** Asks the configured picture-reading model to group the pieces no group holds yet. */
  private async recognizeParts(): Promise<void> {
    const draft = this.draft;
    const capability = this.editCapability;
    if (!draft || !capability) return;
    const groups = effectivePartGroups(draft.manifest, capability);
    const loose = ungroupedUnits(groups, capability);
    if (!loose.length) return;
    this.partRecognition = { modelId: capability.modelId, running: true, done: 0, total: loose.length, message: "" };
    this.partGroupsPanel?.setRecognition(this.partRecognition);
    try {
      await window.desktopPet.partRecognition.start(capability.modelId, groups.map((group) => group.name), loose.map((unit) => unit.id));
    } catch (error) {
      this.partRecognition = { modelId: capability.modelId, running: false, done: 0, total: loose.length, message: errorMessage(error) };
      this.partGroupsPanel?.setRecognition(this.partRecognition);
    }
  }

  /** Recognized pieces land in the draft like any hand edit, so they can be changed or discarded. */
  private applyPartRecognition(update: PartRecognitionUpdate): void {
    const draft = this.draft;
    const capability = this.editCapability;
    if (!draft || !capability || update.modelId !== draft.manifest.id || capability.modelId !== update.modelId) return;
    if (update.assignments.length) {
      const units = new Map(capability.textureUnits.map((unit) => [unit.id, unit]));
      let groups = effectivePartGroups(draft.manifest, capability);
      for (const assignment of update.assignments) {
        const unit = units.get(assignment.unitId);
        if (!unit) continue;
        let group = groups.find((item) => item.name === assignment.groupName);
        if (!group) {
          group = { id: `group-${crypto.randomUUID().slice(0, 8)}`, name: assignment.groupName, drawableIds: [] };
          groups = [...groups, group];
        }
        groups = moveDrawablesToGroup(groups, group.id, looseDrawables(groups, unit));
      }
      draft.manifest.partGroups = groups;
      this.renderDirty();
    }
    this.partRecognition = { modelId: update.modelId, running: update.running, done: update.done, total: update.total, message: update.message };
    if (this.partGroupsPanel) this.partGroupsPanel.setRecognition(this.partRecognition);
  }

  private renderList(): void {
    const root = required("models-list");
    const previousScrollTop = root.scrollTop;
    this.modelVirtualGrid?.destroy();
    this.modelVirtualGrid = null;
    root.replaceChildren();
    required("models-count").textContent = String(this.models.length);
    required("models-empty").hidden = this.models.length > 0;
    const renderModel = (model: ModelPackage): HTMLElement => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "model-card";
      button.classList.toggle("selected", model.id === this.current?.id);
      const preview = model.previewUrl
        ? image(model.previewUrl, model.name)
        : textElement("span", "model-card-placeholder", "LIVE2D");
      const copy = document.createElement("span");
      copy.className = "model-card-copy";
      const statuses = [
        model.id === this.activeId ? "当前桌宠" : null,
        model.id === this.current?.id ? "正在编辑" : null,
      ].filter((value): value is string => Boolean(value));
      copy.append(
        textElement("strong", "", model.name),
        bindLocalizedText(textElement("small", "", ""), "{{copy}}", () => ({ copy: (statuses.length > 0 ? statuses.join(" · ") : t("点击编辑")) })),
      );
      button.append(preview, copy);
      button.addEventListener("click", () => void this.select(model));
      return button;
    };
    if (this.models.length > 40) {
      this.modelVirtualGrid = new VirtualGrid(root, {
        minColumnWidth: 260,
        rowHeight: 176,
        gap: 10,
        maxRenderedItems: 40,
      });
      this.modelVirtualGrid.set(this.models, renderModel);
      root.scrollTop = previousScrollTop;
    } else {
      root.append(...this.models.map(renderModel));
    }
  }

  private showPreviewLoading(name: string): void {
    bindLocalizedText(required("models-preview-title"), "正在加载：{{v0}}", { v0: name });
    const placeholder = required("models-preview-placeholder");
    bindLocalizedText(placeholder, "正在加载 {{v0}}…\n大型模型会在首次绘制完成后显示", { v0: name });
    placeholder.hidden = false;
  }

  private showPreviewReady(name: string): void {
    bindLocalizedText(required("models-preview-title"), "实时预览：{{v0}}", { v0: name });
    required("models-preview-placeholder").hidden = true;
    this.syncDefaultAppearancePreview();
    this.renderDefaultAppearance();
    this.syncFacialTest();
    // Live thumbnails need the loaded preview.
    this.aiOutfitPanel?.render();
  }

  private showPreviewFailure(name: string, error: unknown): void {
    bindLocalizedText(required("models-preview-title"), "加载失败：{{v0}}", { v0: name });
    const fallback = this.displayedModelId
      ? this.models.find((model) => model.id === this.displayedModelId)?.name
      : null;
    const placeholder = required("models-preview-placeholder");
    bindLocalizedText(placeholder, "{{copy}}", () => ({ copy: (fallback ? t("{{v0}} 加载失败，当前仍保留 {{v1}}\n{{v2}}", { v0: name, v1: fallback, v2: errorMessage(error) }) : t("{{v0}} 加载失败\n{{v1}}", { v0: name, v1: errorMessage(error) })) }));
    placeholder.hidden = false;
  }

  private renderEditor(): void {
    const draft = this.draft;
    required("models-editor-empty").hidden = draft !== null;
    required("models-editor").hidden = draft === null;
    if (!draft) return;
    this.emotionSuggestions.setModel(draft.manifest.id);
    required<HTMLInputElement>("model-name").value = draft.manifest.name;
    bindLocalizedText(required("le-title"), "编辑「{{v0}}」", { v0: draft.manifest.name });
    const copy = LOOK_EDITOR_COPY[this.activeModelTab];
    if (copy) bindLocalizedText(required("le-footer-note"), this.footerNote(copy.footer));
    this.renderBehaviors("state", "model-state-behaviors");
    this.renderBehaviors("emotion", "model-emotion-behaviors");
    this.renderDisplay();
    this.renderDefaultAppearance();
    this.syncDefaultAppearancePreview();
    this.renderEditCapabilities();
    this.renderTextureMaterials();
    this.partGroupsPanel?.dispose();
    this.partGroupsPanel = null;
    this.renderPartGroups();
    this.renderAiOutfit();
    this.renderDirty();
  }

  private resetAppearance(): void {
    this.appearanceRequestVersion += 1;
    this.editCapability = null;
    this.textureLibrary = null;
    this.appearanceLoading = false;
  }

  private syncDefaultAppearancePreview(): void {
    if (!this.draft || this.preview?.displayedModelId() !== this.draft.manifest.id) return;
    this.preview.configureDefaultAppearance(this.draft.manifest.defaultAppearance);
  }

  private renderDefaultAppearance(): void {
    this.layerPanel?.dispose();
    this.layerPanel = null;
    if (!this.draft) return;
    const parameters = this.preview?.displayedModelId() === this.draft.manifest.id
      ? this.preview.nativeParameters() : [];
    const layers = this.preview?.displayedModelId() === this.draft.manifest.id ? this.preview.nativeLayers() : [];
    this.layerPanel = renderDefaultAppearancePanel({
      roots: { suits: required("model-default-suits"), toggles: required("model-default-toggles"), params: required("model-default-appearance") },
      manifest: this.draft.manifest,
      changed: () => {
        this.syncDefaultAppearancePreview();
        this.renderDirty();
      },
      parameters,
      layers,
      highlight: ids => this.preview?.highlightLayers(ids),
      deleteToggle: id => void this.deleteAsset("expression", id),
      openParams: () => void this.showOutfitPane("params"),
      saveAsToggle: (name, values) => this.createToggle(name, values),
      redraw: () => this.renderDefaultAppearance(),
    });
    this.syncLayerPicking();
  }

  private syncLayerPicking(): void {
    const active = required("character-wizard").hidden && this.previewStage.parentElement === this.previewHome &&
      this.activeModelTab === "defaults" && (this.outfitPane === "toggles" || this.outfitPane === "params") &&
      this.draft && this.preview?.displayedModelId() === this.draft.manifest.id;
    const parts = required("character-wizard").hidden && this.previewStage.parentElement === this.previewHome &&
      this.activeModelTab === "defaults" && this.outfitPane === "parts" && this.partPick;
    const ai = required("character-wizard").hidden && this.previewStage.parentElement === this.previewHome &&
      this.activeModelTab === "defaults" && this.outfitPane === "ai" && this.aiPick;
    const swatches = required("character-wizard").hidden && this.previewStage.parentElement === this.previewHome &&
      this.activeModelTab === "defaults" && this.outfitPane === "textures" && this.swatchPick;
    this.preview?.setLayerPickHandler(active ? ids => void this.pickLayers(ids) : parts ? this.partPick : ai ? this.aiPick
      : swatches ? this.swatchPick : null);
  }

  /** 装扮 › AI 换装 selects by the draft's part groups and applies results to the preview. */
  private renderAiOutfit(): void {
    const root = document.getElementById("model-ai-outfit");
    if (!root) return;
    const capability = this.editCapability;
    const draft = this.draft;
    const open = this.outfitPane === "ai" && this.activeModelTab === "defaults";
    if (!open || !draft || !capability || capability.modelId !== draft.manifest.id) {
      // Hidden for a moment (full screen, another Space), the look reloads its layers;
      // an unfinished AI 换装 for that look waits for them instead of starting over.
      const reloading = open && (!draft || !capability) && (draft?.manifest.id ?? this.current?.id) === this.aiOutfitModelId;
      if (reloading && this.aiOutfitPanel) return;
      this.parkAiOutfit();
      if (this.outfitPane === "ai") {
        root.replaceChildren(bindLocalizedText(textElement("p", "behavior-help", ""), (this.appearanceLoading ? "正在读取这个形象的图层…" : "还没读取到这个形象的图层")));
      }
      return;
    }
    const options = {
      root,
      stage: this.previewStage,
      manifest: draft.manifest,
      capability,
      imageUrl: (relative: string) => modelAssetUrl(capability.modelId, relative),
      highlight: (ids: readonly string[]) => this.preview?.highlightLayers(ids),
      focus: (ids: readonly string[]) => this.preview?.focusLayers(ids),
      capture: (groups: readonly (readonly string[])[], size: number) =>
        this.preview && this.preview.displayedModelId() === capability.modelId ? this.preview.capturePieces(groups, size) : [],
      shown: () => this.preview && this.preview.displayedModelId() === capability.modelId ? this.preview.shownLayerIds() : null,
      openPane: (pane: "toggles" | "textures") => void this.showOutfitPane(pane),
      setPicking: (handler: ((drawableIds: string[], point: { x: number; y: number }) => void) | null) => {
        this.aiPick = handler;
        this.syncLayerPicking();
      },
      previewDelta: async (delta: TextureAtlasDelta) => { await this.preview?.updateTextureAtlases(delta); },
      saved: async (count: number, applied: boolean) => {
        await this.loadAppearance(false);
        this.showMessage(applied
          ? `已换上 ${count} 块，也放进了「换色」，随时换回原版`
          : `已把 ${count} 块存进「换色」，想穿的时候去那里点一下`);
      },
    };
    if (this.aiOutfitPanel && this.aiOutfitModelId !== capability.modelId) this.parkAiOutfit();
    if (this.aiOutfitPanel) this.aiOutfitPanel.update(options);
    else {
      this.aiOutfitPanel = new AiOutfitPanel({ ...options, draft: this.aiOutfitDrafts.get(capability.modelId) ?? storedAiDraft(capability.modelId) });
      this.aiOutfitModelId = capability.modelId;
      this.aiOutfitPanel.render();
    }
  }

  /** Closes the AI 换装 pane but keeps what the user set up for its model. */
  private parkAiOutfit(): void {
    const panel = this.aiOutfitPanel;
    if (!panel) return;
    const snapshot = panel.snapshot();
    if (this.aiOutfitModelId) {
      if (snapshot) this.aiOutfitDrafts.set(this.aiOutfitModelId, snapshot);
      else this.aiOutfitDrafts.delete(this.aiOutfitModelId);
      // Stored without the AI's understanding: it belongs to a plan that may change.
      remember(`ai-outfit:${this.aiOutfitModelId}`, snapshot ? JSON.stringify({ ...snapshot, optimization: null, step: "pick" }) : "");
    }
    panel.dispose();
    this.aiOutfitPanel = null;
    this.aiOutfitModelId = null;
  }

  /** 装扮 › 部位 needs the texture scan; it edits the draft's part groups. */
  private renderPartGroups(): void {
    const root = document.getElementById("model-part-groups");
    if (!root) return;
    const capability = this.editCapability;
    const draft = this.draft;
    if (this.outfitPane !== "parts" || this.activeModelTab !== "defaults" || !draft || !capability || capability.modelId !== draft.manifest.id) {
      this.partGroupsPanel?.dispose();
      this.partGroupsPanel = null;
      if (this.outfitPane === "parts") {
        root.replaceChildren(bindLocalizedText(textElement("p", "behavior-help", ""), (this.appearanceLoading ? "正在读取这个形象的图层…" : "还没读取到这个形象的图层")));
      }
      return;
    }
    const options = {
      root,
      manifest: draft.manifest,
      capability,
      imageUrl: (relative: string) => modelAssetUrl(capability.modelId, relative),
      changed: () => this.renderDirty(),
      highlight: (ids: readonly string[]) => this.preview?.highlightLayers(ids),
      setPicking: (handler: ((drawableIds: string[]) => void) | null) => {
        this.partPick = handler;
        this.syncLayerPicking();
      },
      recognize: () => void this.recognizeParts(),
      stopRecognition: () => void window.desktopPet.partRecognition.cancel(capability.modelId),
    };
    if (this.partGroupsPanel) this.partGroupsPanel.update(options);
    else {
      this.partGroupsPanel = new PartGroupsPanel(options);
      this.partGroupsPanel.setRecognition(this.partRecognition?.modelId === capability.modelId ? this.partRecognition : null);
    }
  }

  /** Clicking the model picks the part under the cursor for hiding, from 开关 or 参数. */
  private async pickLayers(ids: string[]): Promise<void> {
    if (this.outfitPane !== "params") await this.showOutfitPane("params");
    const fallback = document.querySelector<HTMLDetailsElement>("#model-default-appearance .le-layer-fallback");
    if (fallback && !fallback.open) fallback.open = true;
    this.layerPanel?.pick(ids);
    fallback?.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  private async loadAppearance(
    forceScan: boolean,
  ): Promise<ModelAppearanceSnapshot | null> {
    const model = this.current;
    if (!model) return null;
    const version = ++this.appearanceRequestVersion;
    this.editCapability = null;
    this.textureLibrary = null;
    this.appearanceLoading = true;
    this.renderEditCapabilities();
    this.renderTextureMaterials();
    try {
      if (forceScan) {
        const available = await window.desktopPet.listModels();
        if (this.current?.id !== model.id || version !== this.appearanceRequestVersion) return null;
        if (!available.some((candidate) => candidate.id === model.id)) {
          await this.refresh(this.activeId ?? undefined);
          this.showMessage("临时预览模型已被清理，已切回可用模型，请重新点击 AI 修改");
          return null;
        }
      }
      const snapshot = await window.desktopPet.getModelAppearance(model.id, forceScan);
      if (this.current?.id !== model.id || version !== this.appearanceRequestVersion) return null;
      if (snapshot.modelId !== model.id) return null;
      this.replaceModel(snapshot.model);
      this.editCapability = snapshot.capability;
      this.textureLibrary = snapshot.materials;
      return snapshot;
    } catch (error) {
      if (this.current?.id === model.id && version === this.appearanceRequestVersion) this.showError(error);
      return null;
    } finally {
      if (this.current?.id === model.id && version === this.appearanceRequestVersion) {
        this.appearanceLoading = false;
        this.renderEditCapabilities();
        this.renderTextureMaterials();
        this.renderPartGroups();
        this.renderAiOutfit();
      }
    }
  }

  /** 装扮 › 换色: versions of each part as swatches, by the draft's part groups. */
  private renderTextureMaterials(): void {
    const root = document.getElementById("model-texture-materials");
    const summary = document.getElementById("model-appearance-summary");
    if (!root || !summary) return;
    this.materialVirtualGrid?.destroy();
    this.materialVirtualGrid = null;
    bindLocalizedText(summary, "点左边 TA 身上的部位，只看那里能换的");
    const capability = this.editCapability;
    const library = this.textureLibrary;
    const draft = this.draft;
    const model = this.current;
    if (this.outfitPane !== "textures" || this.activeModelTab !== "defaults" || !model || !capability || !library || !draft ||
      capability.modelId !== draft.manifest.id) {
      this.swatchesPanel?.dispose();
      this.swatchesPanel = null;
      root.replaceChildren();
      if (this.outfitPane === "textures" && this.appearanceLoading) {
        root.append(bindLocalizedText(textElement("p", "behavior-help", ""), "正在读取这个形象的纹理…"));
      }
      return;
    }
    const options = {
      root,
      manifest: draft.manifest,
      capability,
      library,
      highlight: (ids: readonly string[]) => this.preview?.highlightLayers(ids),
      setPicking: (handler: ((drawableIds: string[]) => void) | null) => {
        this.swatchPick = handler;
        this.syncLayerPicking();
      },
      select: async (assignments: Array<{ unitId: string; materialId: string }>) => {
        if (assignments.length === 1) {
          const [only] = assignments;
          await this.applyTextureMaterialChange(await window.desktopPet.selectTextureMaterial(model.id, only!.unitId, only!.materialId));
        } else if (assignments.length) {
          await this.applyTextureMaterialChange(await window.desktopPet.selectTextureMaterials(model.id,
            assignments.map(({ unitId, materialId }) => ({ textureUnitId: unitId, materialId }))));
        }
      },
      remove: async (materialIds: string[]) => {
        for (const materialId of materialIds) {
          await this.applyTextureMaterialChange(await window.desktopPet.deleteTextureMaterial(model.id, materialId));
        }
      },
      previewColor: async (unitIds: string[], hex: string | null) => {
        const delta = await window.desktopPet.previewTextureRecolor(model.id, unitIds, hex);
        if (delta && this.current?.id === model.id) await this.preview?.updateTextureAtlases(delta);
      },
      saveColor: async (unitIds: string[], hex: string, wear: boolean) => {
        await this.applyTextureMaterialChange(await window.desktopPet.saveTextureRecolor(model.id, unitIds, hex, wear));
      },
      shown: () => this.preview && this.preview.displayedModelId() === model.id ? this.preview.shownLayerIds() : null,
      capture: (groups: readonly (readonly string[])[], size: number) =>
        this.preview && this.preview.displayedModelId() === model.id ? this.preview.capturePieces(groups, size) : [],
      focus: (ids: readonly string[]) => this.preview?.focusLayers(ids),
      openAi: () => void this.showOutfitPane("ai"),
    };
    if (this.swatchesPanel) this.swatchesPanel.update(options);
    else {
      this.swatchesPanel = new TextureSwatchesPanel(options);
      this.swatchesPanel.render();
    }
  }

  private async applyAppearanceSnapshot(snapshot: ModelAppearanceSnapshot): Promise<void> {
    if (this.current?.id !== snapshot.modelId) return;
    const requestVersion = ++this.appearanceRequestVersion;
    const previewEpoch = ++this.previewResumeEpoch;
    this.replaceModel(snapshot.model);
    this.current = snapshot.model;
    this.draft = createDraft(snapshot.model);
    this.emotionSuggestions.resetDraft();
    this.editCapability = snapshot.capability;
    this.textureLibrary = snapshot.materials;
    this.appearanceLoading = false;
    this.renderEditor();
    const preview = this.preview;
    if (!preview) return;
    await loadCurrentModelSelection({
      loadBundle: () => window.desktopPet.getModelRenderBundle(snapshot.model.id),
      loadPreview: (bundle) => preview.load(bundle),
      isCurrent: () => (
        requestVersion === this.appearanceRequestVersion &&
        previewEpoch === this.previewResumeEpoch &&
        this.current?.id === snapshot.model.id &&
        this.preview === preview
      ),
    });
  }

  private async applyTextureMaterialChange(change: TextureMaterialChangeResult): Promise<void> {
    if (this.current?.id !== change.delta.modelId) return;
    if (this.textureLibrary && change.library.revision < this.textureLibrary.revision) return;
    this.textureLibrary = change.library;
    this.renderTextureMaterials();
    await this.preview?.updateTextureAtlases(change.delta);
  }

  private texturePreviewUrl(unit: TextureEditUnit): string {
    const activeId = this.textureLibrary?.activeAssignments[unit.id];
    const material = activeId
      ? this.textureLibrary?.materials.find((item) => item.id === activeId)
      : null;
    if (!material) throw new Error(`纹理 ${unit.label} 没有当前素材`);
    return material.imageUrl;
  }

  private async openAiEditor(partId?: string): Promise<void> {
    if (!this.current) {
      this.showError(new Error("请先选择一个 Live2D 模型"));
      return;
    }
    this.facialTest.stop();
    const capability = this.editCapability ?? (await this.loadAppearance(false))?.capability;
    if (!capability) return;
    await this.characterWizard.open(capability, partId);
  }

  private renderEditCapabilities(): void {
    const summary = required("model-edit-summary");
    const root = required("model-edit-parts");
    const scan = required<HTMLButtonElement>("model-edit-scan");
    scan.disabled = this.appearanceLoading || !this.current;
    root.replaceChildren();
    if (!this.current) {
      bindLocalizedText(summary, "请先选择模型");
      return;
    }
    if (this.appearanceLoading) {
      bindLocalizedText(summary, "正在读取 Moc、纹理 alpha、ArtMesh 与 UV…");
      return;
    }
    const capability = this.editCapability;
    if (!capability) {
      bindLocalizedText(summary, "尚未预检。扫描只读取本机模型，不会调用 AI 或产生费用。");
      return;
    }
    const supported = capability.parts.filter((part) => part.operations.appearance.supported);
    bindLocalizedText(summary, "{{copy}}", () => ({ copy: (t("识别到 {{v0}} 个部位，其中 {{v1}} 个可安全修改外观。", { v0: capability.parts.length, v1: supported.length }) + t("当前严格保留原轮廓与透明度；轮廓变化需完成全动作扫描后才会开放。")) }));
    for (const part of capability.parts) root.append(this.editPartRow(part));
  }

  private editPartRow(part: ModelEditPartCapability): HTMLElement {
    const row = document.createElement("article");
    row.className = "model-edit-part";
    const copy = document.createElement("div");
    const slack = part.operations.contour.staticSlack;
    copy.append(
      textElement("strong", "", part.label),
      bindLocalizedText(textElement("small", "", ""), "{{copy}}", () => ({ copy: (part.operations.appearance.supported ? ((t("{{v0}} 个 ArtMesh · 可改颜色、纹理和明暗 · ", { v0: part.drawableIds.length }) + t("中立网格余量 上 {{v0}} / 右 {{v1}} / ", { v0: slack.top.pixels.toFixed(1), v1: slack.right.pixels.toFixed(1) })) + t("下 {{v0}} / 左 {{v1}} px（仅诊断）", { v0: slack.bottom.pixels.toFixed(1), v1: slack.left.pixels.toFixed(1) })) : t("{{v0}} 个 ArtMesh · 当前不允许 AI 修改", { v0: part.drawableIds.length })) })),
      bindLocalizedText(textElement("small", "model-edit-limit", ""), "轮廓修改：{{v0}}", { v0: part.operations.contour.reason }),
    );
    const edit = bindLocalizedText(smallButton("", () => void this.openAiEditor(part.id)), "AI 修改此外观");
    edit.disabled = !part.operations.appearance.supported;
    row.append(copy, edit);
    return row;
  }

  private renderBehaviors(kind: BehaviorKind, rootId: string): void {
    if (!this.draft) return;
    const root = required(rootId);
    root.replaceChildren();
    for (const row of buildBehaviorRows(this.draft.manifest, kind)) {
      const details = document.createElement("details");
      details.className = "behavior-row";
      const rowId = `${kind}:${row.key}`;
      const editing = this.candidateEditor?.kind === kind && this.candidateEditor.key === row.key;
      details.open = editing || this.openBehaviorRows.has(rowId);
      details.addEventListener("toggle", () => {
        if (details.open) this.openBehaviorRows.add(rowId);
        else this.openBehaviorRows.delete(rowId);
      });
      const summary = document.createElement("summary");
      summary.append(
        bindLocalizedText(textElement("strong", "", ""), row.label),
        row.candidates.length
          ? bindLocalizedText(textElement("span", "", ""), "{{v0}} 组", { v0: row.candidates.length })
          : bindLocalizedText(textElement("span", "", ""), "默认姿态"),
      );
      const body = document.createElement("div");
      body.className = "behavior-candidates";
      row.candidates.forEach((candidate, index) => {
        // The group being edited is edited where it is.
        if (editing && this.candidateEditor!.candidateId === candidate.id) {
          body.append(this.candidateEditorPanel(row.label, index + 1));
          return;
        }
        const card = document.createElement("article");
        card.className = "behavior-candidate";
        const copy = document.createElement("div");
        const describe = this.candidateDescription(candidate);
        copy.append(
          bindLocalizedText(textElement("strong", "", ""), "第 {{v0}} 组", { v0: index + 1 }),
          bindLocalizedText(textElement("small", "", ""), "{{copy}}", () => ({ copy: describe() })),
        );
        const actions = document.createElement("div");
        actions.className = "behavior-candidate-actions";
        actions.append(
          bindLocalizedText(smallButton("", () => void this.preview?.previewCandidate(candidate).catch((error) => this.showError(error))), "预览"),
          bindLocalizedText(smallButton("", () => {
            this.openCandidateEditor(kind, row.key, candidate);
            this.renderBehaviorViews();
            this.revealCandidateEditor();
          }), "编辑"),
          bindLocalizedText(smallButton("", () => {
            if (!this.draft || !window.confirm(t("删除{{v0}}的第 {{v1}} 组？", { v0: t(row.label), v1: index + 1 }))) return;
            deleteBehaviorCandidate(this.draft, kind, row.key, candidate.id);
            if (this.candidateEditor?.candidateId === candidate.id) this.candidateEditor = null;
            this.renderBehaviorViews();
            this.renderDirty();
          }, "danger"), "删除"),
        );
        card.append(copy, actions);
        body.append(card);
      });
      const adding = editing && !row.candidates.some((candidate) => candidate.id === this.candidateEditor!.candidateId);
      if (adding) body.append(this.candidateEditorPanel(row.label));
      else {
        body.append(bindLocalizedText(smallButton("", () => {
          this.openCandidateEditor(kind, row.key);
          this.renderBehaviorViews();
          this.revealCandidateEditor();
        }, "secondary"), "＋ 添加一组"));
      }
      details.append(summary, body);
      root.append(details);
    }
  }

  private openCandidateEditor(
    kind: BehaviorKind,
    key: string,
    candidate?: BehaviorCandidate,
  ): void {
    this.candidateEditor = createCandidateEditorState(
      kind,
      key,
      candidate?.id ?? crypto.randomUUID(),
      candidate,
    );
  }

  /**
   * Editing one group: what it plays, in order, pinned on top with preview
   * and save; below, every motion and expression in a fixed order to pick
   * from, so picking never moves the list.
   */
  private candidateEditorPanel(label: string, group?: number): HTMLElement {
    const editor = this.candidateEditor!;
    const panel = document.createElement("section");
    panel.className = "candidate-editor";
    const bar = document.createElement("div");
    bar.className = "candidate-bar";
    const head = document.createElement("div");
    head.className = "candidate-bar__head";
    head.append(group === undefined
      ? bindLocalizedText(textElement("h4", "", ""), "{{v0}} · 新的一组", () => ({ v0: t(label) }))
      : bindLocalizedText(textElement("h4", "", ""), "{{v0}} · 第 {{v1}} 组", () => ({ v0: t(label), v1: group })));
    const hasAssets = editor.motionAssetIds.length + editor.expressionAssetIds.length > 0;
    const preview = bindLocalizedText(smallButton("", () => void this.previewCandidateEditor()), (editor.previewing ? "正在预览…" : "▶ 预览这一组"));
    preview.disabled = !hasAssets || editor.previewing;
    const save = bindLocalizedText(smallButton("", () => {
      if (!this.draft || !this.candidateEditor) return;
      const current = this.candidateEditor;
      if (!canSaveCandidate(current)) return;
      saveBehaviorCandidate(this.draft, current.kind, current.key, this.editorCandidate(current));
      this.emotionSuggestions.editedCandidate(this.editorCandidate(current).id);
      this.candidateEditor = null;
      this.renderBehaviorViews();
      this.renderDirty();
    }, "accent"), "保存");
    save.disabled = !canSaveCandidate(editor);
    const cancel = bindLocalizedText(smallButton("", () => {
      this.candidateEditor = null;
      this.preview?.settle();
      this.renderBehaviorViews();
    }, "secondary"), "取消");
    const actions = document.createElement("div");
    actions.className = "candidate-bar__actions";
    actions.append(preview, save, cancel);
    head.append(actions);
    const chosen = document.createElement("div");
    chosen.className = "candidate-chosen";
    const chips = [
      ...editor.motionAssetIds.map((id) => this.candidateChip("motion", id, editor.motionAssetIds)),
      ...editor.expressionAssetIds.map((id) => this.candidateChip("expression", id, editor.expressionAssetIds)),
    ].filter((chip): chip is HTMLElement => chip !== null);
    if (chips.length) chosen.append(...chips);
    else chosen.append(bindLocalizedText(textElement("span", "candidate-chosen__empty", ""), "在下面点选动作和表情。先播动作，再叠表情"));
    bar.append(head, chosen);
    panel.append(bar, this.candidateLibrary("motion"), this.candidateLibrary("expression"));
    return panel;
  }

  /** Bring a just-opened group editor into view; the page does not move otherwise. */
  private revealCandidateEditor(): void {
    document.querySelector(".candidate-editor")?.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  /** One picked motion or expression in the group, with its place in the order. */
  private candidateChip(kind: AssetKind, assetId: string, ids: string[]): HTMLElement | null {
    const assets = kind === "motion" ? this.draft!.manifest.motionAssets : this.draft!.manifest.expressionAssets;
    const asset = assets[assetId];
    if (!asset) return null;
    const index = ids.indexOf(assetId);
    const chip = document.createElement("span");
    chip.className = `candidate-chip candidate-chip--${kind}`;
    const earlier = smallButton("‹", () => this.moveCandidateAsset(kind, assetId, -1), "candidate-chip__move");
    bindLocalizedAttribute(earlier, "title", "往前");
    earlier.disabled = index <= 0;
    const later = smallButton("›", () => this.moveCandidateAsset(kind, assetId, 1), "candidate-chip__move");
    bindLocalizedAttribute(later, "title", "往后");
    later.disabled = index >= ids.length - 1;
    const remove = smallButton("×", () => {
      const current = this.candidateEditor;
      if (current && setCandidateAssetSelected(current, kind, assetId, false)) this.renderBehaviorViews();
    }, "candidate-chip__remove");
    bindLocalizedAttribute(remove, "title", "不要这个");
    chip.append(
      bindLocalizedText(textElement("span", "candidate-chip__kind", ""), "{{v0}} {{v1}}", () => ({ v0: t(kind === "motion" ? "动作" : "表情"), v1: index + 1 })),
      textElement("b", "", asset.name),
      ...(ids.length > 1 ? [earlier, later] : []),
      remove,
    );
    return chip;
  }

  /** Every motion or expression, in a fixed order; clicking one picks or drops it. */
  private candidateLibrary(kind: AssetKind): HTMLElement {
    const editor = this.candidateEditor;
    const selectedIds = kind === "motion" ? editor?.motionAssetIds ?? [] : editor?.expressionAssetIds ?? [];
    const section = document.createElement("section");
    section.className = "candidate-library";
    const title = kind === "motion" ? "动作" : "表情";
    const assets = Object.values(kind === "motion" ? this.draft!.manifest.motionAssets : this.draft!.manifest.expressionAssets);
    section.append(bindLocalizedText(textElement("h5", "", ""), "{{v0}} · {{v1}} 个", () => ({ v0: t(title), v1: assets.length })));
    if (!assets.length) {
      section.append(bindLocalizedText(textElement("p", "candidate-empty", ""), "这个形象没有{{v0}}", () => ({ v0: t(title) })));
      return section;
    }
    const grid = document.createElement("div");
    grid.className = "candidate-grid";
    for (const asset of assets) {
      const order = selectedIds.indexOf(asset.id);
      const tile = document.createElement("div");
      tile.className = `candidate-tile${order >= 0 ? " selected" : ""}`;
      const pick = document.createElement("button");
      pick.type = "button";
      pick.className = "candidate-tile__pick";
      pick.setAttribute("aria-pressed", String(order >= 0));
      pick.title = asset.file;
      const copy = document.createElement("span");
      copy.className = "candidate-tile__copy";
      copy.append(textElement("b", "candidate-asset-name", asset.name));
      pick.append(
        order >= 0 ? textElement("span", "candidate-tile__order", String(order + 1)) : textElement("span", "candidate-tile__box", ""),
        copy,
      );
      pick.addEventListener("click", () => {
        const current = this.candidateEditor;
        if (current && setCandidateAssetSelected(current, kind, asset.id, order < 0)) this.renderBehaviorViews();
      });
      const play = smallButton("▶", () => {
        void this.previewAsset(kind, asset.id, () => this.candidateEditor === editor && !editor?.previewing)
          .catch((error) => this.showError(error));
      }, "candidate-tile__icon");
      bindLocalizedAttribute(play, "title", "单独预览「{{v0}}」", { v0: asset.name });
      play.disabled = !!editor?.previewing;
      const rename = smallButton("✎", () => {
        // The name becomes a text field; it cannot sit inside the pick button.
        pick.replaceWith(copy);
        this.beginInlineAssetRename(tile, copy, kind, asset.id);
      }, "candidate-tile__icon");
      bindLocalizedAttribute(rename, "title", "改名字");
      tile.append(pick, play, rename);
      grid.append(tile);
    }
    section.append(grid);
    return section;
  }

  private beginInlineAssetRename(
    row: HTMLElement,
    copy: HTMLElement,
    kind: AssetKind,
    assetId: string,
  ): void {
    if (!this.draft) return;
    const asset = kind === "motion"
      ? this.draft.manifest.motionAssets[assetId]
      : this.draft.manifest.expressionAssets[assetId];
    if (!asset) return;
    const input = document.createElement("input");
    input.className = "candidate-name-input";
    input.maxLength = 80;
    input.value = asset.name;
    let finished = false;
    const finish = (save: boolean): void => {
      if (finished) return;
      finished = true;
      if (save) {
        try {
          renameAsset(this.draft!, kind, assetId, input.value);
          this.renderDirty();
        } catch (error) {
          this.showError(error);
        }
      }
      // A display-name edit never changes the visual revision of this candidate.
      this.renderBehaviorViews();
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        finish(true);
      } else if (event.key === "Escape") {
        event.preventDefault();
        finish(false);
      }
    });
    input.addEventListener("blur", () => finish(true));
    copy.replaceChildren(input, textElement("small", "", asset.file));
    row.classList.add("renaming");
    input.focus();
    input.select();
  }

  private moveCandidateAsset(kind: AssetKind, assetId: string, delta: -1 | 1): void {
    const editor = this.candidateEditor;
    if (!editor) return;
    if (moveEditorCandidateAsset(editor, kind, assetId, delta)) this.renderBehaviorViews();
  }

  private async previewCandidateEditor(): Promise<void> {
    const editor = this.candidateEditor;
    if (!editor || editor.previewing) return;
    editor.previewing = true;
    this.renderBehaviorViews();
    try {
      await this.requirePreview().previewCandidate(this.editorCandidate(editor));
    } catch (error) {
      this.showError(error);
    } finally {
      if (this.candidateEditor === editor) {
        editor.previewing = false;
        this.renderBehaviorViews();
      }
    }
  }


  private editorCandidate(editor: CandidateEditorState): BehaviorCandidate {
    return candidateFromEditor(editor);
  }

  private candidateDescription(candidate: BehaviorCandidate): () => string {
    const manifest = this.draft!.manifest;
    const motions = candidate.motionAssetIds.map((id) => manifest.motionAssets[id]?.name ?? id);
    const expressions = candidate.expressionAssetIds.map((id) => manifest.expressionAssets[id]?.name ?? id);
    return () => [
      motions.length ? t("动作：{{v0}}", { v0: motions.join(" → ") }) : null,
      expressions.length ? t("表情：{{v0}}", { v0: expressions.join(" → ") }) : null,
    ].filter((value): value is string => Boolean(value)).join("；");
  }

  private renderBehaviorViews(): void {
    // Rebuilding the lists must not move what the person is looking at.
    const scroller = required("model-state-behaviors").closest<HTMLElement>(".le-body");
    const top = scroller?.scrollTop ?? 0;
    this.renderBehaviors("state", "model-state-behaviors");
    this.renderBehaviors("emotion", "model-emotion-behaviors");
    this.emotionSuggestions.refresh();
    if (scroller) scroller.scrollTop = top;
  }

  private renderDisplay(): void {
    if (!this.draft || !this.current) return;
    this.stopLipTest();
    const root = required("model-display-fields");
    root.replaceChildren();
    const manifest = this.draft.manifest;
    required("model-idle-fields").replaceChildren(this.naturalIdlePanel());
    const runtime = this.cubismRuntimeStatusPanel();
    const fields: HTMLElement[] = [
      ...(runtime ? [runtime] : []),
      this.lipSyncStatusPanel(),
      percentSlider("嘴张多大", manifest.lipsync.maxOpen, (value) => {
        manifest.lipsync.maxOpen = value;
        this.preview?.configureLipSync(manifest.lipsync);
      }),
      this.lipSyncAdvancedPanel(),
    ];
    fields.push(this.lipTestPanel());
    root.append(...fields);
    this.preview?.configureLipSync(manifest.lipsync);
    this.preview?.configureNaturalIdle(manifest.naturalIdle);
    for (const input of root.querySelectorAll("input")) {
      input.addEventListener("input", () => this.renderDirty());
    }
  }

  private naturalIdlePanel(): HTMLElement {
    return renderNaturalIdlePanel(this.draft!.manifest.naturalIdle, this.current!.capabilities.parameters ?? [], (settings) => {
      this.draft!.manifest.naturalIdle = settings;
      this.preview?.configureNaturalIdle(settings);
      this.renderDirty();
    });
  }

  /** Only when the model has something the person should know; its format version is not that. */
  private cubismRuntimeStatusPanel(): HTMLElement | null {
    const warnings = this.current!.capabilities.warnings ?? [];
    if (warnings.length === 0) return null;
    const section = document.createElement("section");
    section.className = "lip-adapter-status";
    const copy = document.createElement("div");
    copy.append(bindLocalizedText(textElement("strong", "", ""), "这个形象要注意的地方"));
    section.append(copy);
    for (const warning of warnings) {
      section.append(textElement("small", "model-edit-limit", warning));
    }
    return section;
  }

  private lipSyncStatusPanel(): HTMLElement {
    const section = document.createElement("section");
    section.className = "lip-adapter-status";
    const value = this.draft!.manifest.lipsync;
    const title = value.mode === "viseme"
      ? "嘴型跟着字音变"
      : value.mode === "open-form"
        ? "嘴巴跟着声音开合"
        : "还没配好嘴型";
    const detail = value.needsConfirmation
      ? "这个形象的嘴型没认全，在下面「嘴型细调」里确认一次。"
      : value.source === "manual"
        ? "用的是你手动调好的嘴型。"
        : value.mode === "viseme"
          ? "说话时会按 a、e、i、o、u 的口型动嘴。"
          : "说话时嘴巴会跟着声音一张一合。";
    const copy = document.createElement("div");
    copy.append(bindLocalizedText(textElement("strong", "", ""), title), bindLocalizedText(textElement("span", "", ""), detail));
    section.append(copy);
    if (value.needsConfirmation && this.lipSyncMappingComplete()) {
      section.append(bindLocalizedText(smallButton("", () => {
        if (!this.draft) return;
        this.draft.manifest.lipsync.source = "manual";
        this.draft.manifest.lipsync.needsConfirmation = false;
        this.preview?.configureLipSync(this.draft.manifest.lipsync);
        this.renderDisplay();
        this.renderDirty();
      }), "就用这套嘴型"));
    }
    return section;
  }

  private lipSyncAdvancedPanel(): HTMLElement {
    const details = document.createElement("details");
    details.className = "lip-adapter-advanced";
    details.open = this.draft!.manifest.lipsync.needsConfirmation;
    details.append(bindLocalizedText(textElement("summary", "", ""), "嘴型细调"));
    const body = document.createElement("div");
    const value = this.draft!.manifest.lipsync;
    const mode = selectField(
      "嘴巴怎么动",
      value.mode === "viseme" ? "viseme" : "open-form",
      [
        { value: "open-form", label: "跟着声音开合" },
        { value: "viseme", label: "按字音变嘴型" },
      ],
      (next) => {
        if (!this.draft || (next !== "open-form" && next !== "viseme")) return;
        const lipsync = this.draft.manifest.lipsync;
        lipsync.mode = next;
        lipsync.source = "manual";
        if (next === "open-form") {
          const automatic = resolveLipSyncAdapter(this.current?.capabilities ?? {});
          lipsync.mouthOpenParameter = lipsync.mouthOpenParameter ?? automatic.mouthOpenParameter;
          lipsync.mouthFormParameter = lipsync.mouthFormParameter ?? automatic.mouthFormParameter;
          lipsync.jawParameter = lipsync.jawParameter ?? automatic.jawParameter;
          lipsync.puckerWidenParameter = lipsync.puckerWidenParameter ?? automatic.puckerWidenParameter;
          lipsync.funnelParameter = lipsync.funnelParameter ?? automatic.funnelParameter;
        } else {
          const automatic = resolveLipSyncAdapter(this.current?.capabilities ?? {});
          if (automatic.mode === "viseme") lipsync.vowelParameters = automatic.vowelParameters;
        }
        this.refreshLipSyncDraft();
        this.renderDisplay();
      },
    );
    body.append(mode);
    const options = this.lipSyncParameterOptions();
    if (value.mode === "viseme") {
      for (const vowel of LIP_TEST_VOWELS) {
        body.append(selectField(
          `${LIP_VOWEL_LABELS[vowel]} 的嘴型`,
          value.vowelParameters[vowel] ?? "",
          options,
          (next) => {
            if (!this.draft) return;
            this.draft.manifest.lipsync.vowelParameters[vowel] = next || null;
            this.draft.manifest.lipsync.source = "manual";
            this.refreshLipSyncDraft();
          },
        ));
      }
    } else {
      body.append(
        selectField(
          "嘴巴开合",
          value.mouthOpenParameter ?? "",
          options,
          (next) => {
            if (!this.draft) return;
            this.draft.manifest.lipsync.mouthOpenParameter = next || null;
            this.draft.manifest.lipsync.source = "manual";
            this.refreshLipSyncDraft();
          },
        ),
        selectField(
          "嘴形变化（可不选）",
          value.mouthFormParameter ?? "",
          options,
          (next) => {
            if (!this.draft) return;
            this.draft.manifest.lipsync.mouthFormParameter = next || null;
            this.draft.manifest.lipsync.source = "manual";
            this.refreshLipSyncDraft();
          },
        ),
        ...([
          ["jawParameter", "下巴（可不选）"],
          ["puckerWidenParameter", "嘟嘴与咧嘴（可不选）"],
          ["funnelParameter", "嘴收圆（可不选）"],
        ] as const).map(([key, label]) => selectField(label, value[key] ?? "", options, (next) => {
          if (!this.draft) return;
          this.draft.manifest.lipsync[key] = next || null;
          this.draft.manifest.lipsync.source = "manual";
          this.refreshLipSyncDraft();
        })),
      );
    }
    if (value.source === "manual") {
      body.append(bindLocalizedText(smallButton("", () => {
        if (!this.draft || !this.current) return;
        const maxOpen = this.draft.manifest.lipsync.maxOpen;
        this.draft.manifest.lipsync = {
          ...resolveLipSyncAdapter(this.current.capabilities),
          maxOpen,
        };
        this.preview?.configureLipSync(this.draft.manifest.lipsync);
        this.renderDisplay();
        this.renderDirty();
      }), "恢复自动适配"));
    }
    details.append(body);
    return details;
  }

  private lipSyncParameterOptions(): Array<{ value: string; label: string }> {
    const parameters = this.current?.capabilities.parameters ?? [];
    const declared = this.current?.capabilities.lipSyncParameters ?? [];
    const byId = new Map(parameters.map((item) => [item.id, item]));
    return [
      { value: "", label: "不使用" },
      ...[...new Set([...parameters.map((item) => item.id), ...declared])].map((id) => {
        const parameter = byId.get(id);
        return {
          value: id,
          label: parameter && parameter.name !== id
            ? `${parameter.name}（${id}）`
            : id,
        };
      }),
    ];
  }

  private lipSyncMappingComplete(): boolean {
    const value = this.draft!.manifest.lipsync;
    return value.mode === "viseme"
      ? Object.values(value.vowelParameters).every(Boolean)
      : value.mode === "open-form" && Boolean(value.mouthOpenParameter);
  }

  private refreshLipSyncDraft(): void {
    if (!this.draft) return;
    this.draft.manifest.lipsync.needsConfirmation = !this.lipSyncMappingComplete();
    this.preview?.configureLipSync(this.draft.manifest.lipsync);
    this.renderDirty();
  }

  private lipTestPanel(): HTMLElement {
    const panel = document.createElement("section");
    panel.className = "lip-test-panel";
    const title = bindLocalizedText(textElement("h3", "", ""), "试试嘴型");
    const strength = document.createElement("input");
    strength.type = "range";
    strength.min = "0";
    strength.max = "1";
    strength.step = "0.01";
    strength.value = "1";
    const strengthValue = textElement("span", "", "100%");
    strength.addEventListener("input", () => {
      strengthValue.textContent = `${Math.round(Number(strength.value) * 100)}%`;
    });
    const range = document.createElement("label");
    range.className = "lip-test-strength";
    range.append(bindLocalizedText(document.createElement("span"), "声音大小"), strength, strengthValue);
    const actions = document.createElement("div");
    actions.className = "lip-test-actions";
    const preview = (vowel: LipTestVowel) => {
      this.previewLipVowel(vowel, Number(strength.value));
    };
    for (const vowel of LIP_TEST_VOWELS) actions.append(smallButton(LIP_VOWEL_LABELS[vowel], () => preview(vowel)));
    const microphone = bindLocalizedText(smallButton("", () => {
      void this.toggleMicrophoneLipTest(microphone);
    }), "用麦克风试");
    this.lipTestMicrophoneButton = microphone;
    const loop = bindLocalizedText(smallButton("", () => {
      if (this.lipTestTimer !== null) {
        this.stopLipTest();
        bindLocalizedText(loop, "轮流播放");
        return;
      }
      let index = 0;
      preview(LIP_TEST_VOWELS[index]!);
      this.lipTestTimer = window.setInterval(() => {
        index = (index + 1) % LIP_TEST_VOWELS.length;
        preview(LIP_TEST_VOWELS[index]!);
      }, 420);
      bindLocalizedText(loop, "停止轮流");
    }), "轮流播放");
    actions.append(microphone, loop, bindLocalizedText(smallButton("", () => {
      this.stopLipTest();
      bindLocalizedText(loop, "轮流播放");
    }), "停止并闭嘴"));
    panel.append(
      title,
      bindLocalizedText(textElement("p", "", ""), "对着麦克风说话，或点下面的啊、诶、衣、哦、乌，看看左边的嘴型对不对。只影响左边的预览。"),
      range,
      actions,
    );
    return panel;
  }

  private previewLipVowel(vowel: LipTestVowel, strength: number): void {
    if (!this.draft) return;
    if (this.lipTestMicrophoneActive) this.stopMicrophoneLipTest();
    this.preview?.configureLipSync(this.draft.manifest.lipsync);
    this.preview?.previewLipShape({
      // Keep raw volume at zero so fallback models without independent vowel
      // parameters still reveal the vowel-specific opening weights.
      volume: 0,
      a: vowel === "a" ? strength : 0,
      e: vowel === "e" ? strength : 0,
      i: vowel === "i" ? strength : 0,
      o: vowel === "o" ? strength : 0,
      u: vowel === "u" ? strength : 0,
    });
  }

  private stopLipTest(): void {
    if (this.lipTestTimer !== null) {
      window.clearInterval(this.lipTestTimer);
      this.lipTestTimer = null;
    }
    this.stopMicrophoneLipTest();
    this.preview?.stopLipPreview();
  }

  private async toggleMicrophoneLipTest(button: HTMLButtonElement): Promise<void> {
    if (!this.preview) return;
    this.facialTest.stop();
    if (this.lipTestMicrophoneActive) {
      this.stopMicrophoneLipTest();
      return;
    }
    this.stopLipTest();
    const version = ++this.lipTestMicrophoneVersion;
    const preview = this.preview;
    this.lipTestMicrophoneStarting = true;
    button.disabled = true;
    bindLocalizedText(button, "正在准备麦克风…");
    try {
      // TA's listening holds the microphone; it pauses until the test stops.
      await window.desktopPet.borrowMicrophone();
      if (version !== this.lipTestMicrophoneVersion || this.preview !== preview) return;
      preview.configureLipSync(this.draft!.manifest.lipsync);
      await preview.startMicrophoneLipTest();
      if (version !== this.lipTestMicrophoneVersion || this.preview !== preview) return;
      this.lipTestMicrophoneActive = true;
      bindLocalizedText(button, "停止麦克风测试");
    } catch (error) {
      if (version === this.lipTestMicrophoneVersion) {
        this.showError(error);
        bindLocalizedText(button, "用麦克风试");
        void window.desktopPet.returnMicrophone();
      }
    } finally {
      if (version === this.lipTestMicrophoneVersion) this.lipTestMicrophoneStarting = false;
      button.disabled = false;
    }
  }

  private stopMicrophoneLipTest(): void {
    this.lipTestMicrophoneVersion += 1;
    if (!this.lipTestMicrophoneActive && !this.lipTestMicrophoneStarting) return;
    this.lipTestMicrophoneStarting = false;
    this.lipTestMicrophoneActive = false;
    if (this.lipTestMicrophoneButton) {
      bindLocalizedText(this.lipTestMicrophoneButton, "用麦克风试");
    }
    void this.preview?.stopMicrophoneLipTest();
    void window.desktopPet.returnMicrophone();
  }

  private renderDirty(): void {
    const dirty = this.draft ? hasUnsavedChanges(this.draft) : false;
    const valid = this.draft ? isDraftValid(this.draft) : false;
    required("models-dirty").hidden = !dirty;
    required<HTMLButtonElement>("models-save").disabled = !dirty || !valid;
    required<HTMLButtonElement>("models-discard").disabled = !dirty;
  }

  private async showTab(tab: string): Promise<void> {
    if (!modelManagerTabKeepsPreview(tab)) return;
    if (tab === this.activeModelTab) return;
    if (tab !== "facial") this.facialTest.setContext(false, null, null);
    if (this.activeModelTab === "idle") {
      // Back to the preview mode the person chose, after 待机 showed it moving.
      const animated = recall("look-editor-preview-mode", ["static", "animated"] as const, "static") === "animated";
      this.preview?.setPreferredMode(animated);
      if (!animated) this.preview?.reset();
    }
    this.activeModelTab = tab;
    remember("look-editor-tab", tab);
    this.syncLayerPicking();
    if (tab !== "defaults") {
      this.renderPartGroups();
      this.renderAiOutfit();
      this.renderTextureMaterials();
    }
    if (tab !== "display") this.stopLipTest();
    if (tab === "defaults") this.preview?.settle();
    // Idle can only be seen moving: show it moving here, without changing the remembered choice.
    if (tab === "idle") {
      this.preview?.setPreferredMode(true);
      this.preview?.playAwakeIdle();
    }
    for (const button of document.querySelectorAll<HTMLButtonElement>("[data-model-tab]")) {
      button.classList.toggle("active", button.dataset.modelTab === tab);
    }
    for (const name of LOOK_EDITOR_TABS) {
      required(`model-tab-${name}`).hidden = name !== tab;
    }
    const copy = LOOK_EDITOR_COPY[tab];
    if (copy) {
      bindLocalizedText(required("le-subtitle"), copy.subtitle);
      required("le-latin").textContent = copy.latin;
      bindLocalizedText(required("le-footer-note"), this.footerNote(copy.footer));
    }
    required("models-editor-actions").hidden = tab === "facial";
    if (tab === "facial") void this.syncJevNotice();
    this.syncFacialTest();
    if (tab === "defaults") await this.showOutfitPane(this.outfitPane);
  }

  /** Speech expressions need a decision model (百炼 or Jev); the settings still save without one. */
  private async syncJevNotice(): Promise<void> {
    const configuration = await window.desktopPet.getConfiguration().catch(() => null);
    required("facial-jev-missing").hidden = configuration !== null && decisionReady(configuration,
      (id) => configuration.secrets.modelApiKeys?.[id] === true, configuration.secrets.jevApiKey === true);
  }

  /** 装扮 is split into 开关 / 参数 / 换色 / AI 换装; the last two need the texture scan. */
  private async showOutfitPane(pane: OutfitPane): Promise<void> {
    this.outfitPane = pane;
    remember("look-editor-pane", pane);
    for (const button of document.querySelectorAll<HTMLButtonElement>("[data-outfit-pane]")) {
      button.classList.toggle("active", button.dataset.outfitPane === pane);
    }
    for (const element of document.querySelectorAll<HTMLElement>(".le-pane")) {
      element.hidden = element.dataset.pane !== pane;
    }
    const needsScan = this.usesAppearance() && (!this.editCapability || !this.textureLibrary);
    // The first paint already says the layers are loading, not that they are missing.
    if (needsScan) this.appearanceLoading = true;
    this.syncLayerPicking();
    this.renderPartGroups();
    this.renderAiOutfit();
    if (needsScan) await this.loadAppearance(false);
    this.renderTextureMaterials();
    this.renderPartGroups();
    this.renderAiOutfit();
  }

  private usesAppearance(): boolean {
    return this.activeModelTab === "defaults" && (this.outfitPane === "textures" || this.outfitPane === "parts" || this.outfitPane === "ai");
  }

  private showEmpty(): void {
    this.facialTest.setContext(false, null, null);
    this.layerPanel?.dispose(); this.layerPanel = null;
    this.preview?.setLayerPickHandler(null);
    this.current = null;
    this.draft = null;
    this.resetAppearance();
    this.renderList();
    this.renderEditor();
    required("models-preview-placeholder").hidden = false;
  }

  private replaceModel(model: ModelPackage): void {
    const index = this.models.findIndex((candidate) => candidate.id === model.id);
    if (index >= 0) this.models[index] = model;
    else this.models.push(model);
    this.current = model;
    this.renderList();
  }

  /** A look TA isn't wearing only shows on the desktop once it is worn. */
  private footerNote(tabNote: string): string {
    return this.current && this.activeId && this.current.id !== this.activeId
      ? "这个形象现在没在用：改好保存后，在 角色 › 形象 里换上才会在桌面看到"
      : tabNote;
  }

  private mayDiscard(): boolean {
    return !this.draft || !hasUnsavedChanges(this.draft) || window.confirm(t("放弃尚未保存的模型修改？"));
  }

  private showReport(report: ModelImportReport): void {
    const root = required("models-report");
    root.hidden = false;
    root.className = `models-report ${report.status}`;
    root.replaceChildren(...[
      ...report.errors,
      ...report.warnings,
      ...report.checks,
    ].map((issue) => textElement("p", "", `${issue.message}${issue.file ? ` · ${issue.file}` : ""}`)));
  }

  private showMessage(message: string): void {
    this.showReport({
      status: "valid",
      checks: [{ code: "message", message, file: null }],
      warnings: [],
      errors: [],
    });
  }

  private showError(error: unknown): void {
    this.showReport({
      status: "invalid",
      checks: [],
      warnings: [],
      errors: [{
        code: "ui-error",
        message: error instanceof Error ? error.message : String(error),
        file: null,
      }],
    });
  }

  private async ensurePreview(): Promise<ModelManagerPreview> {
    if (this.preview) return this.preview;
    if (this.previewCreation) return this.previewCreation;
    const lifecycle = this.previewGeneration;
    const creation = loadPreviewConstructor(Boolean(window.Live2DCubismCore)).then(
      (PreviewConstructor) => {
        if (lifecycle !== this.previewGeneration) {
          throw new Error("Live2D 预览创建已被取消");
        }
        const generation = ++this.previewGeneration;
        const preview = new PreviewConstructor(
          required<HTMLCanvasElement>("models-preview-canvas"),
          required("models-preview-stage"),
          (error) => this.schedulePreviewRecovery(generation, error),
          (animated) => {
            required("models-preview-reset").setAttribute("aria-pressed", String(!animated));
            required("models-preview-animate").setAttribute("aria-pressed", String(animated));
            bindLocalizedText(required("models-preview-mode"), (animated ? "动起来看：TA 平时待机的样子，也能单独试某个状态、动作或口型。" : "定住看：停下动作、眨眼和晃动，方便看清参数改了哪里。只影响左边的预览。"));
          },
        );
        preview.setPreferredMode(recall("look-editor-preview-mode", ["static", "animated"] as const, "static") === "animated");
        this.preview = preview;
        return preview;
      },
    ).finally(() => {
      if (this.previewCreation === creation) this.previewCreation = null;
    });
    this.previewCreation = creation;
    return creation;
  }

  private async resumeCurrentPreview(): Promise<ModelManagerPreview> {
    const model = this.current;
    if (!model) return this.ensurePreview();
    return this.resumePreviewModel(model, true);
  }

  private async resumePreviewModel(
    model: ModelPackage,
    announce: boolean,
  ): Promise<ModelManagerPreview> {
    const epoch = ++this.previewResumeEpoch;
    const preview = await this.ensurePreview();
    if (epoch !== this.previewResumeEpoch || preview !== this.preview) return preview;
    const switchingModel = preview.displayedModelId() !== model.id;
    if (announce && switchingModel) this.showPreviewLoading(model.name);
    const bundle = await window.desktopPet.getModelRenderBundle(model.id);
    if (epoch !== this.previewResumeEpoch || preview !== this.preview) return preview;
    const displayed = await preview.load(bundle);
    if (epoch !== this.previewResumeEpoch || preview !== this.preview) return preview;
    if (displayed !== model.id) throw new Error("预览加载已被更新的模型选择取代");
    this.syncDefaultAppearancePreview();
    this.displayedModelId = displayed;
    this.previewLoadStatus = "ready";
    if (announce && switchingModel) this.showPreviewReady(model.name);
    preview.resize();
    return preview;
  }

  private async previewAsset(kind: "motion" | "expression", id: string, isCurrent: () => boolean = () => true): Promise<void> {
    if (!isCurrent()) return;
    const preview = await this.resumeCurrentPreview();
    if (!isCurrent()) return;
    if (kind === "motion") preview.playMotion(id);
    else preview.showExpression(id);
  }

  private schedulePreviewRecovery(generation: number, error: Error): void {
    if (
      !this.active ||
      generation !== this.previewGeneration ||
      this.previewRecovery
    ) return;
    const recovery = this.recoverPreview(error).finally(() => {
      if (this.previewRecovery === recovery) this.previewRecovery = null;
    });
    this.previewRecovery = recovery;
  }

  private async recoverPreview(cause: Error): Promise<void> {
    const model = this.current;
    this.releasePreview(true);
    if (!this.active || !model) return;
    try {
      await nextFrame();
      const preview = await this.ensurePreview();
      const previewEpoch = ++this.previewResumeEpoch;
      const loaded = await loadCurrentModelSelection({
        loadBundle: () => window.desktopPet.getModelRenderBundle(model.id),
        loadPreview: (bundle) => preview.load(bundle),
        isCurrent: () => (
          this.active &&
          previewEpoch === this.previewResumeEpoch &&
          this.current?.id === model.id &&
          this.preview === preview
        ),
      });
      if (!loaded.current) return;
      preview.resize();
      required("models-preview-placeholder").hidden = true;
      this.showMessage("Live2D 预览渲染器已自动恢复");
    } catch (error) {
      this.showError(new Error(
        `Live2D 预览自动恢复失败：${errorMessage(error)}（原始错误：${cause.message}）`,
      ));
    }
  }

  private async activate(version: number): Promise<void> {
    try {
      const preview = await this.ensurePreview();
      if (!this.isCurrentActivation(version)) {
        if (this.preview === preview) this.releasePreview();
        return;
      }
      const requested = this.requestedEditId;
      this.requestedEditId = null;
      if (!this.initialized) {
        // A fresh window picks up the look, tab and 装扮 pane the user last had open.
        const last = requested ?? localModelChoice();
        await this.refresh(last ?? undefined, !last);
        this.initialized = true;
        this.outfitPane = recall("look-editor-pane", OUTFIT_PANES, "toggles");
        const tab = recall("look-editor-tab", LOOK_EDITOR_TABS, "pools");
        if (tab !== this.activeModelTab) await this.showTab(tab);
      } else if (requested && requested !== this.current?.id && this.mayDiscard()) {
        await this.refresh(requested);
      } else if (this.current && this.draft && (hasUnsavedChanges(this.draft) || this.candidateEditor)) {
        const model = this.current;
        const previewEpoch = ++this.previewResumeEpoch;
        this.showPreviewLoading(model.name);
        const loaded = await loadCurrentModelSelection({
          loadBundle: () => window.desktopPet.getModelRenderBundle(model.id),
          loadPreview: (bundle) => preview.load(bundle),
          isCurrent: () => (
            this.isCurrentActivation(version) &&
            previewEpoch === this.previewResumeEpoch &&
            this.current?.id === model.id &&
            this.preview === preview
          ),
        });
        if (!loaded.current) return;
        const displayed = loaded.result;
        if (displayed === model.id) {
          this.displayedModelId = displayed;
          this.previewLoadStatus = "ready";
          this.showPreviewReady(model.name);
        }
      } else {
        await this.refresh(undefined, true);
      }
      if (!this.isCurrentActivation(version)) {
        if (this.preview === preview) this.releasePreview();
        return;
      }
      preview.resize();
    } catch (error) {
      if (this.isCurrentActivation(version)) this.showError(error);
    }
  }

  private isCurrentActivation(version: number): boolean {
    return this.active && version === this.activationVersion;
  }

  private releasePreview(replaceCanvas = true): void {
    this.facialTest.setContext(false, null, null);
    this.layerPanel = null;
    const preview = this.preview;
    this.preview = null;
    this.previewCreation = null;
    this.previewResumeEpoch += 1;
    this.previewGeneration += 1;
    try {
      preview?.destroy();
    } catch (error) {
      console.warn("释放 Live2D 预览失败", error);
    }
    if (replaceCanvas) replacePreviewCanvas();
  }

  private requirePreview(): ModelManagerPreview {
    if (!this.preview) throw new Error("Live2D 预览尚未就绪");
    return this.preview;
  }

  private syncFacialTest(): void {
    const ready = this.active && this.activeModelTab === "facial" && this.previewLoadStatus === "ready"
      && !!this.current && this.preview?.displayedModelId() === this.current.id
      && this.previewStage.parentElement === this.previewHome && required("character-wizard").hidden;
    this.facialTest.setContext(ready, ready ? this.current : null, ready ? this.preview : null);
  }
}

function storedAiDraft(modelId: string): AiOutfitDraft | null {
  try {
    const value = JSON.parse(recallValue(`ai-outfit:${modelId}`) || "null") as AiOutfitDraft | null;
    return value && Array.isArray(value.selection) && typeof value.prompt === "string" && Array.isArray(value.references) ? value : null;
  } catch {
    return null;
  }
}

/** The look last open in the editor, if the user had one. */
function localModelChoice(): string | null {
  return recallValue("look-editor-model") || null;
}

function replacePreviewCanvas(): HTMLCanvasElement {
  const current = required<HTMLCanvasElement>("models-preview-canvas");
  const replacement = document.createElement("canvas");
  replacement.id = current.id;
  replacement.className = current.className;
  replacement.setAttribute("aria-label", current.getAttribute("aria-label") ?? "Live2D 实时预览");
  current.replaceWith(replacement);
  return replacement;
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** A 0–1 amount as a slider with its percent, like the other sliders here. */
function percentSlider(label: string, value: number, update: (value: number) => void): HTMLLabelElement {
  const input = document.createElement("input");
  input.type = "range";
  input.min = "0";
  input.max = "1";
  input.step = "0.01";
  input.value = String(value);
  const shown = textElement("span", "", `${Math.round(value * 100)}%`);
  input.addEventListener("input", () => {
    const next = Number(input.value);
    shown.textContent = `${Math.round(next * 100)}%`;
    update(next);
  });
  const row = document.createElement("label");
  row.className = "lip-test-strength lip-strength-row";
  row.append(bindLocalizedText(document.createElement("span"), label), input, shown);
  return row;
}

function textField(
  label: string,
  value: string,
  update: (value: string) => void,
): HTMLLabelElement {
  const input = document.createElement("input");
  input.value = value;
  input.addEventListener("input", () => update(input.value.trim()));
  return labeled(label, input);
}

function selectField(
  label: string,
  value: string,
  options: Array<{ value: string; label: string }>,
  update: (value: string) => void,
): HTMLLabelElement {
  const select = document.createElement("select");
  for (const option of options) {
    const element = document.createElement("option");
    element.value = option.value;
    // Only these app-owned choices are translated; model parameter names stay intact.
    if (["", "open-form", "viseme"].includes(option.value)) bindLocalizedText(element, option.label);
    else element.textContent = option.label;
    select.append(element);
  }
  select.value = value;
  select.addEventListener("change", () => update(select.value));
  return labeled(label, select);
}

function labeled(text: string, input: HTMLElement): HTMLLabelElement {
  const label = document.createElement("label");
  label.className = "display-field";
  label.append(bindLocalizedText(document.createElement("span"), text), input);
  return label;
}

function smallButton(
  label: string,
  action: () => void,
  className = "",
): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.addEventListener("click", action);
  return button;
}

function image(source: string, alt: string): HTMLImageElement {
  const result = document.createElement("img");
  result.src = source;
  result.alt = alt;
  result.loading = "lazy";
  result.decoding = "async";
  return result;
}


function modelAssetUrl(modelId: string, relativePath: string, revision?: string): string {
  const relative = relativePath.replace(/^\/+/, "");
  const base = `pet-model://${encodeURIComponent(modelId)}/${relative.split("/").map(encodeURIComponent).join("/")}`;
  return revision ? `${base}?revision=${encodeURIComponent(revision)}` : base;
}


function textElement<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text: string,
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function required<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`缺少模型管理元素 #${id}`);
  return element as T;
}
