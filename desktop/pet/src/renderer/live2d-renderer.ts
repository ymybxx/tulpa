import type {
  BehaviorCandidate,
  ModelPackage,
  ModelRenderBundle,
  NativeModelParameter,
  NativeModelLayer,
  PetEmotion,
  PetBehaviorState,
  PetState,
  TextureAtlasDelta,
} from "../shared/contracts";
import { live2DFrameRate } from "./frame-rate";
import { LatestTaskQueue } from "./latest-task-queue";
import { CubismModelInstance, type CubismDisplayMetrics } from "./cubism-model-instance";
import { CubismRuntimeHost } from "./cubism-runtime-host";
import {
  Live2DBehaviorPlayback,
  type StateBehaviorSelection,
} from "./live2d-behavior-playback";
import {
  CLOSED_MOUTH,
  type LipShape,
  type PetModelBounds,
  type PetRenderer,
} from "./pet-renderer";
import { PoolSelector } from "./pool-selector";
import { CubismWebGLOffscreenManager } from "@cubism/rendering/cubismoffscreenmanager";
import { adaptiveTextureDimension, supportedTextureDimensions } from "./texture-detail";
import {
  assertTextureStreamTransitionHasState,
  shouldApplyTextureSnapshot,
} from "./texture-stream-order";
import { TargetFrameScheduler } from "./target-frame-scheduler";
import type { FacialDemoSnapshot } from "./facial-parameter-controller";
import type { FacialMotionBindings, FacialOverlayBindings } from "../shared/facial-bindings";

export const GRAPHICS_HEALTH_SAMPLE_INTERVAL_MS = 1_000;

export class GraphicsHealthCheckCadence {
  private lastCheckedAt: number | null = null;

  shouldCheck(timestampMs: number, mode: "sampled" | "immediate"): boolean {
    if (
      mode === "sampled"
      && this.lastCheckedAt !== null
      && timestampMs >= this.lastCheckedAt
      && timestampMs - this.lastCheckedAt < GRAPHICS_HEALTH_SAMPLE_INTERVAL_MS
    ) return false;
    this.lastCheckedAt = timestampMs;
    return true;
  }
}

export interface Live2DRendererOptions {
  onFatalError?: (error: Error) => void;
  /** Screenshot previews need this; the always-on desktop pet deliberately does not. */
  preserveDrawingBuffer?: boolean;
  maxTextureDimension?: number;
  /** Desktop pet only: hot-switches atlas detail without rebuilding the model. */
  adaptiveTextureDimensions?: readonly number[];
  maxAdaptiveTextureDimension?: number;
  onGpuDiagnostics?: (value: Live2DGpuDiagnostics) => void;
}

export interface Live2DGpuDiagnostics {
  modelId: string | null;
  atlasCount: number;
  atlasDimensions: Array<{ width: number; height: number }>;
  offscreenContainerCount: number;
  texturesCreated: number;
  texturesDeleted: number;
}

/** Official Cubism 5 Web renderer shared by the pet and model manager. */
export class Live2DRenderer implements PetRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly modelLoads = new LatestTaskQueue<ModelRenderBundle>();
  private readonly poolSelector = new PoolSelector();
  private readonly behaviorPlayback: Live2DBehaviorPlayback;
  private model: CubismModelInstance | null = null;
  private currentPackage: ModelPackage | null = null;
  private targetLip: LipShape = CLOSED_MOUTH;
  private state: PetState = "unbound";
  private activeState: PetState | null = null;
  private activeEmotion: PetEmotion | null = null;
  private selectedEmotionCandidate: BehaviorCandidate | null = null;
  private previewingState = false;
  private editingPreview = false;
  private facialDemoActive = false;
  private speechFacialEnabled = false;
  private facialOverlayState: { wasEditing: boolean; releasing: boolean } | null = null;
  private textureUpdateTail: Promise<void> = Promise.resolve();
  private textureStreamId: string | null = null;
  private textureStreamRevision = 0;
  private destroyed = false;
  private fatalError: Error | null = null;
  private restoringContext = false;
  private contextGeneration = 0;
  private manuallyPaused = false;
  private expressiveIdleEnabled = false;
  private readonly frameScheduler: TargetFrameScheduler;
  private lastUpdateAt = 0;
  private metrics: CubismDisplayMetrics;
  private texturesCreated = 0;
  private texturesDeleted = 0;
  private textureDetailTimer: number | null = null;
  private desiredTextureDimension: number | null = null;
  private adaptiveTextureDimensions: number[];
  private readonly gpuMaximumTextureDimension: number;
  private modelOffset = { x: 0, y: 0 };
  private modelZoom = 1;
  private readonly graphicsHealthCheckCadence = new GraphicsHealthCheckCadence();

  private readonly onAnimationFrame = (timestamp: number): void => {
    if (this.destroyed || document.hidden || this.manuallyPaused) return;
    if (this.fatalError || this.restoringContext || !this.isVisible()) return;
    const fps = this.currentFrameRate();
    const delta = this.lastUpdateAt > 0 ? (timestamp - this.lastUpdateAt) / 1000 : 1 / fps;
    this.lastUpdateAt = timestamp;
    this.render(delta, timestamp);
  };

  private readonly onVisibilityChange = (): void => {
    this.lastUpdateAt = 0;
    if (document.hidden) {
      this.frameScheduler.pause();
      return;
    }
    if (!this.manuallyPaused && !this.editingPreview) this.frameScheduler.resume();
  };

  private readonly onContextLost = (event: Event): void => {
    event.preventDefault();
    this.contextGeneration += 1;
    this.restoringContext = true;
    this.lastUpdateAt = 0;
  };

  private readonly onContextRestored = (): void => {
    const model = this.model;
    const generation = this.contextGeneration;
    if (this.destroyed) return;
    if (!model) {
      this.restoringContext = false;
      return;
    }
    void this.enqueueTextureMutation(async () => {
      if (this.destroyed || generation !== this.contextGeneration) return;
      if (model !== this.model) {
        this.restoringContext = false;
        this.render(1 / 60);
        return;
      }
      await model.restoreWebGL(this.metrics);
      if (
        this.destroyed ||
        generation !== this.contextGeneration ||
        model !== this.model
      ) return;
      this.drainGraphicsErrors();
      this.restoringContext = false;
      this.render(1 / 60);
    })
      .catch((error) => {
        if (this.destroyed || generation !== this.contextGeneration) return;
        this.restoringContext = false;
        if (model !== this.model) {
          this.render(1 / 60);
          return;
        }
        this.reportFatal(asError(error, "Live2D WebGL 恢复失败"));
      });
  };

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly viewport: Window | HTMLElement = window,
    private readonly options: Live2DRendererOptions = {},
  ) {
    this.gl = CubismRuntimeHost.createContext(canvas, options.preserveDrawingBuffer ?? false);
    const gpuMaximum = Number(this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE));
    this.gpuMaximumTextureDimension = Number.isFinite(gpuMaximum) && gpuMaximum > 0
      ? gpuMaximum : 4096;
    this.adaptiveTextureDimensions = options.adaptiveTextureDimensions?.length
      ? supportedTextureDimensions(
        options.maxAdaptiveTextureDimension ?? Number.POSITIVE_INFINITY,
        this.gpuMaximumTextureDimension,
        options.adaptiveTextureDimensions,
      ) : [];
    this.metrics = this.measure();
    this.applyCanvasSize();
    this.behaviorPlayback = new Live2DBehaviorPlayback(() => this.model);
    this.frameScheduler = new TargetFrameScheduler(
      this.currentFrameRate(),
      this.onAnimationFrame,
    );
    canvas.addEventListener("webglcontextlost", this.onContextLost);
    canvas.addEventListener("webglcontextrestored", this.onContextRestored);
    document.addEventListener("visibilitychange", this.onVisibilityChange);
    if (!document.hidden) this.frameScheduler.resume();
  }

  async loadModel(value: ModelRenderBundle): Promise<void> {
    return this.modelLoads.enqueue(value, (next, isCurrent) => this.enqueueTextureMutation(async () => {
      if (this.destroyed || !isCurrent()) return;
      if (this.model && this.currentPackage?.url === next.model.url) {
        const behaviorsChanged = behaviorConfigurationChanged(
          this.currentPackage.manifest.behaviors,
          next.model.manifest.behaviors,
        );
        if (shouldApplyTextureSnapshot(
          this.textureStreamId,
          this.textureStreamRevision,
          next.appearance,
        )) {
          assertTextureStreamTransitionHasState(this.textureStreamId, next.appearance);
          await this.model.replaceTextureSnapshot(
            next.appearance,
            () => !this.destroyed && isCurrent() && this.model !== null,
          );
          this.checkGraphicsHealth("immediate");
          this.textureStreamId = next.appearance.streamId;
          this.textureStreamRevision = next.appearance.revision;
        }
        if (this.destroyed || !isCurrent()) return;
        this.currentPackage = next.model;
        this.configureNaturalIdle(next.model.manifest.naturalIdle);
        this.model.configureDefaultAppearance(next.model.manifest.defaultAppearance);
        if (behaviorsChanged) this.refreshBehaviorConfiguration();
        if (this.editingPreview) this.render(0);
        this.emitGpuDiagnostics();
        return;
      }
      const candidate = await CubismModelInstance.load(this.gl, next, this.metrics, {
        maxTextureDimension: this.initialTextureDimension(),
        onTextureCreated: () => { this.texturesCreated += 1; },
        onTextureDeleted: () => { this.texturesDeleted += 1; },
      });
      if (this.destroyed || !isCurrent()) {
        candidate.release();
        return;
      }
      candidate.setViewOffset(this.modelOffset.x, this.modelOffset.y);
      candidate.setViewZoom(this.modelZoom);
      const previous = this.model;
      const previousPackage = this.currentPackage;
      const previousExpressiveIdle = this.expressiveIdleEnabled;
      try {
        if (this.facialDemoActive) this.endFacialDemo();
        this.clearFacialOverlay();
        this.clear();
        candidate.draw();
        this.checkGraphicsHealth("immediate");
        this.model = candidate;
        this.currentPackage = next.model;
        this.expressiveIdleEnabled = !!next.model.manifest.naturalIdle?.enabled && !!next.model.manifest.naturalIdle.expressive;
        this.textureStreamId = next.appearance.streamId;
        this.textureStreamRevision = next.appearance.revision;
        this.behaviorPlayback.cancel();
        this.poolSelector.clear();
        this.activeState = null;
        this.activeEmotion = null;
        this.selectedEmotionCandidate = null;
        if (this.editingPreview) this.resetEditingPreview();
        else {
          this.setState("standby");
          this.setEmotion("neutral");
        }
        if (previous) {
          previous.release();
          this.checkGraphicsHealth("immediate");
        }
        this.emitGpuDiagnostics();
        this.scheduleTextureDetailUpdate();
      } catch (error) {
        candidate.release();
        this.model = previous;
        this.currentPackage = previousPackage;
        this.expressiveIdleEnabled = previousExpressiveIdle;
        this.syncFrameRate();
        if (previous) this.render(1 / 60);
        throw error;
      }
    }));
  }

  async updateTextureAtlases(delta: TextureAtlasDelta): Promise<void> {
    return this.enqueueTextureMutation(() => this.replaceTextureAtlases(delta));
  }

  displayedModelId(): string | null {
    return this.currentPackage?.id ?? null;
  }

  setState(value: PetState): void {
    if (this.facialDemoActive) return;
    this.state = value;
    this.model?.setSpeaking(value === "speaking");
    this.syncFrameRate();
    if (value === this.activeState) return;
    this.activeState = value;
    if (value === "speaking") {
      this.behaviorPlayback.setSpeaking(true);
      if (!this.speechFacialEnabled && !this.behaviorPlayback.hasSpeakingEmotion() && this.selectedEmotionCandidate) {
        this.behaviorPlayback.playTransient(
          this.selectedEmotionCandidate,
          `speaking-emotion:${this.activeEmotion ?? "neutral"}`,
        );
      }
      return;
    }
    const selection = this.selectStateCandidate(value);
    if (value === "acknowledging" && selection.candidate) {
      this.behaviorPlayback.setSpeaking(false);
      this.behaviorPlayback.playAcknowledgement(selection.candidate);
      return;
    }
    this.behaviorPlayback.setState(
      selection.candidate,
      selection.rotate ? () => this.selectStateCandidate(value) : undefined,
    );
    if (value === "standby" || value === "unbound") {
      this.behaviorPlayback.cancelAcknowledgement();
    }
    this.behaviorPlayback.setSpeaking(false);
  }

  setEmotion(value: PetEmotion): void {
    if (this.facialDemoActive) return;
    if (value === this.activeEmotion) return;
    this.activeEmotion = value;
    if (this.speechFacialEnabled) return;
    this.playEmotionCandidate(value);
    this.syncFrameRate();
  }

  playClick(): void {
    if (this.facialDemoActive) return;
    const manifest = this.currentPackage?.manifest;
    const candidate = this.pickCandidate("state:click", manifest?.behaviors.states.click ?? []);
    if (candidate) {
      this.behaviorPlayback.playClick(candidate);
      this.syncFrameRate();
    }
  }

  setLipShape(value: LipShape): void {
    if (this.facialDemoActive) return;
    this.targetLip = value;
    this.model?.setLipShape(value);
    this.syncFrameRate();
  }

  configureLipSync(value: ModelPackage["manifest"]["lipsync"]): void {
    this.model?.configureLipSync(value);
  }

  configureNaturalIdle(value: ModelPackage["manifest"]["naturalIdle"]): void {
    this.model?.configureNaturalIdle(value);
    this.expressiveIdleEnabled = !!value?.enabled && !!value.expressive;
    this.syncFrameRate();
  }

  configureDefaultAppearance(value: ModelPackage["manifest"]["defaultAppearance"]): void {
    this.model?.configureDefaultAppearance(value);
    if (this.editingPreview) this.render(0);
  }

  nativeParameters(): NativeModelParameter[] {
    return this.model?.nativeParameters() ?? [];
  }

  setSpeechFacialEnabled(enabled: boolean): void {
    if (enabled === this.speechFacialEnabled) return;
    this.speechFacialEnabled = enabled;
    if (enabled) {
      this.selectedEmotionCandidate = null;
      this.behaviorPlayback.cancelEmotion();
    } else {
      this.playEmotionCandidate(this.activeEmotion ?? "neutral");
    }
  }

  /** Isolated demo mode: authored editing appearance plus explicitly requested inputs. */
  beginFacialDemo(motion?: FacialMotionBindings): void {
    if (this.facialDemoActive) return;
    if (!this.model || this.destroyed) throw new Error("请先加载 Live2D 模型");
    this.clearFacialOverlay();
    this.behaviorPlayback.cancel();
    this.model.beginFacialDemo(motion);
    this.facialDemoActive = true;
    this.editingPreview = false;
    this.previewingState = false;
    this.activeState = null;
    this.activeEmotion = null;
    this.selectedEmotionCandidate = null;
    this.targetLip = CLOSED_MOUTH;
    this.lastUpdateAt = 0;
    this.syncFrameRate();
    this.render(0);
    if (!this.manuallyPaused && !document.hidden) this.frameScheduler.resume();
  }

  setFacialDemoTargets(targets: Record<string, number>): void {
    if (this.facialDemoActive || this.facialOverlayState) this.model?.setFacialDemoTargets(targets);
  }

  setFacialDemoSpeechLevel(level: number): void {
    if (this.facialDemoActive || this.facialOverlayState) this.model?.setFacialDemoSpeechLevel(level);
  }

  facialDemoSnapshot(): FacialDemoSnapshot {
    return this.model?.facialDemoSnapshot() ?? { active: false, values: {}, targets: {} };
  }

  endFacialDemo(): void {
    if (!this.facialDemoActive) return;
    this.model?.endFacialDemo();
    this.facialDemoActive = false;
    this.editingPreview = true;
    this.lastUpdateAt = 0;
    this.frameScheduler.pause();
    this.resetEditingPreview();
  }

  beginFacialOverlay(bindings: FacialOverlayBindings): void {
    if (!this.model || this.destroyed) throw new Error("请先加载 Live2D 模型");
    if (this.facialDemoActive) this.endFacialDemo();
    const wasEditing = this.facialOverlayState?.wasEditing ?? this.editingPreview;
    this.model.beginFacialOverlay(bindings, !wasEditing);
    this.facialOverlayState = { wasEditing, releasing: false };
    this.editingPreview = false;
    if (wasEditing) this.lastUpdateAt = 0;
    this.syncFrameRate();
    if (!this.manuallyPaused && !document.hidden) this.frameScheduler.resume();
  }

  endFacialOverlay(): void {
    if (!this.facialOverlayState || this.facialOverlayState.releasing) return;
    this.facialOverlayState.releasing = true;
    this.model?.endFacialOverlay();
  }

  private clearFacialOverlay(): void {
    if (!this.facialOverlayState) return;
    this.model?.endFacialOverlay(true);
    this.facialOverlayState = null;
  }

  nativeLayers(): NativeModelLayer[] { return this.model?.nativeLayers() ?? []; }
  highlightLayers(ids: readonly string[]): void {
    this.model?.highlightLayers(ids);
    if (this.editingPreview || this.manuallyPaused) this.render(0);
  }
  pickLayers(x: number, y: number): string[] { return this.model?.pickLayers(x, y, this.metrics) ?? []; }
  shownLayerIds(): Set<string> | null { return this.model?.shownLayerIds() ?? null; }
  /** Where what the character shows now is on screen, in CSS pixels. */
  shownBounds(): { x: number; y: number; width: number; height: number } | null {
    return this.model?.visibleScreenBounds(this.metrics) ?? null;
  }

  layerBounds(ids: readonly string[]): { x: number; y: number; width: number; height: number } | null {
    return this.model?.layerScreenBounds(ids, this.metrics) ?? null;
  }

  /**
   * Renders each group of drawables alone (the rest darkened) and crops a
   * square around it from the live canvas, so thumbnails match exactly what
   * the preview shows: toggles, parameters, swapped textures and hidden layers.
   */
  capturePieces(groups: readonly (readonly string[])[], size: number): HTMLCanvasElement[] {
    const model = this.model;
    if (!model || this.destroyed || this.gl.isContextLost()) return [];
    const ratio = this.canvas.width / Math.max(1, this.metrics.cssWidth);
    // The whole character as it looks now, for the small locator in the corner.
    model.highlightLayers([]);
    this.render(0);
    const body = model.visibleScreenBounds(this.metrics);
    const overview = document.createElement("canvas");
    if (body) {
      overview.height = Math.max(1, Math.round(size * 0.4));
      overview.width = Math.max(1, Math.round(overview.height * body.width / Math.max(1, body.height)));
      overview.getContext("2d")!.drawImage(this.canvas, body.x * ratio, body.y * ratio, body.width * ratio, body.height * ratio,
        0, 0, overview.width, overview.height);
    }
    const shots = groups.map((ids) => {
      const out = document.createElement("canvas");
      out.width = out.height = size;
      const bounds = model.layerScreenBounds(ids, this.metrics);
      if (!bounds) return out;
      // Only these pieces, so ones behind the head or the body are not covered.
      model.highlightLayers(ids, "solo");
      this.render(0);
      const side = Math.max(48, Math.max(bounds.width, bounds.height) * 1.12);
      const left = bounds.x + bounds.width / 2 - side / 2;
      const top = bounds.y + bounds.height / 2 - side / 2;
      // Read back in the same task as the draw, before the frame is presented.
      const context = out.getContext("2d")!;
      context.drawImage(this.canvas, left * ratio, top * ratio, side * ratio, side * ratio, 0, 0, size, size);
      if (body && overview.width > 1) drawLocator(context, overview, body, bounds, size);
      return out;
    });
    model.highlightLayers([]);
    this.render(0);
    return shots;
  }

  setGaze(x: number, y: number, source: "pointer" | "ambient" = "pointer"): void {
    if (this.facialDemoActive) return;
    this.model?.setGaze(x, y, source);
  }

  getModelBounds(): PetModelBounds | null {
    return this.model?.getBounds() ?? null;
  }

  setModelOffset(x: number, y: number): void {
    if (x === this.modelOffset.x && y === this.modelOffset.y) return;
    this.modelOffset = { x, y };
    this.model?.setViewOffset(x, y);
    if (this.model) this.render(0);
  }

  setModelZoom(zoom: number): void {
    if (zoom === this.modelZoom) return;
    this.modelZoom = zoom;
    this.model?.setViewZoom(zoom);
    if (this.model) this.render(0);
    this.scheduleTextureDetailUpdate();
  }

  setTextureMaximumDimension(maximum: number): void {
    if (!this.options.adaptiveTextureDimensions?.length) return;
    const next = supportedTextureDimensions(
      maximum, this.gpuMaximumTextureDimension,
      this.options.adaptiveTextureDimensions,
    );
    if (next.join(",") === this.adaptiveTextureDimensions.join(",")) return;
    this.adaptiveTextureDimensions = next;
    this.scheduleTextureDetailUpdate();
  }

  isPointInDragArea(x: number, y: number): boolean {
    return this.model?.containsDraggablePoint(x, y, this.metrics) ?? false;
  }

  cancelGeneration(): void {
    this.behaviorPlayback.cancelEmotion();
    this.activeEmotion = null;
    this.selectedEmotionCandidate = null;
    this.setLipShape(CLOSED_MOUTH);
  }

  previewMotionAsset(id: string): void {
    if (this.facialDemoActive) return;
    if (this.currentPackage?.manifest.motionAssets[id]) {
      this.behaviorPlayback.playTransient(
        { id: `preview-motion:${id}`, motionAssetIds: [id], expressionAssetIds: [] },
        `preview:${id}`,
      );
    }
  }

  previewExpressionAsset(id: string): void {
    if (this.facialDemoActive) return;
    if (this.currentPackage?.manifest.expressionAssets[id]) {
      this.behaviorPlayback.playTransient(
        { id: `preview-expression:${id}`, motionAssetIds: [], expressionAssetIds: [id] },
        `preview:${id}`,
      );
    }
  }

  async previewBehaviorCandidate(candidate: BehaviorCandidate, mode: "transient" | "state" = "transient"): Promise<void> {
    if (this.facialDemoActive) throw new Error("请先结束面部参数 Demo");
    const model = this.model;
    if (this.previewingState) this.resetPreview();
    if (mode === "state") {
      this.setState("standby");
      this.behaviorPlayback.cancelTransient();
    }
    const started = mode === "state"
      ? this.behaviorPlayback.setState(candidate)
      : this.behaviorPlayback.playTransient(candidate, `preview:${candidate.id}`);
    if (!model || !started) {
      throw new Error("行为候选无法启动");
    }
    this.previewingState = mode === "state";
    this.render(1 / 60);
    if (this.fatalError) throw this.fatalError;
    this.checkGraphicsHealth("immediate");
    await nextPaint();
    if (model !== this.model || this.destroyed) throw new Error("预览模型已经切换");
  }

  resetPreview(): void {
    this.clearFacialOverlay();
    if (this.facialDemoActive) { this.endFacialDemo(); return; }
    if (this.editingPreview) { this.resetEditingPreview(); return; }
    this.previewingState = false;
    this.behaviorPlayback.cancelTransient();
    const state = this.state;
    const emotion = this.activeEmotion ?? "neutral";
    this.activeState = null;
    this.activeEmotion = null;
    this.setState(state);
    this.setEmotion(emotion);
  }

  /** Keeps model editing independent of the real pet's state and animations. */
  setEditingPreview(value: boolean): void {
    this.clearFacialOverlay();
    if (this.facialDemoActive) {
      if (value) this.endFacialDemo();
      return;
    }
    if (this.editingPreview === value) return;
    this.editingPreview = value;
    this.lastUpdateAt = 0;
    if (value) {
      this.frameScheduler.pause();
      this.resetEditingPreview();
    } else if (!this.manuallyPaused && !document.hidden) {
      this.frameScheduler.resume();
    }
  }

  private resetEditingPreview(): void {
    this.behaviorPlayback.cancel();
    this.previewingState = false;
    this.activeState = null;
    this.activeEmotion = null;
    this.selectedEmotionCandidate = null;
    this.state = "unbound";
    this.model?.setSpeaking(false);
    this.setLipShape(CLOSED_MOUTH);
    this.render(0);
  }

  resize(): void {
    this.metrics = this.measure();
    this.applyCanvasSize();
    this.model?.resize(this.metrics);
    this.render(1 / 60);
    this.scheduleTextureDetailUpdate();
  }

  setPaused(value: boolean): void {
    if (this.manuallyPaused === value) return;
    this.manuallyPaused = value;
    this.lastUpdateAt = 0;
    if (value) {
      this.frameScheduler.pause();
      return;
    }
    if (!document.hidden && !this.editingPreview) this.frameScheduler.resume();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.frameScheduler.destroy();
    if (this.textureDetailTimer !== null) window.clearTimeout(this.textureDetailTimer);
    this.modelLoads.invalidate();
    this.behaviorPlayback.destroy();
    this.canvas.removeEventListener("webglcontextlost", this.onContextLost);
    this.canvas.removeEventListener("webglcontextrestored", this.onContextRestored);
    document.removeEventListener("visibilitychange", this.onVisibilityChange);
    const ownedModel = this.model;
    this.model = null;
    this.currentPackage = null;
    void Promise.all([
      this.modelLoads.whenIdle(),
      this.textureUpdateTail.catch(() => undefined),
    ]).finally(() => {
      try {
        ownedModel?.release();
      } finally {
        CubismWebGLOffscreenManager.getInstance().removeContext(this.gl);
        this.gl.getExtension("WEBGL_lose_context")?.loseContext();
        this.emitGpuDiagnostics();
      }
    });
  }

  private async replaceTextureAtlases(delta: TextureAtlasDelta): Promise<void> {
    const model = this.model;
    if (
      !model ||
      this.destroyed ||
      this.currentPackage?.id !== delta.modelId ||
      !shouldApplyTextureSnapshot(this.textureStreamId, this.textureStreamRevision, delta)
    ) return;
    assertTextureStreamTransitionHasState(this.textureStreamId, delta);
    if (delta.atlases.length === 0 && delta.patchSets.length === 0) {
      this.textureStreamId = delta.streamId;
      this.textureStreamRevision = delta.revision;
      return;
    }
    await model.replaceTextureSnapshot(
      delta,
      () => this.model === model && !this.destroyed,
    );
    if (this.model !== model || this.destroyed) return;
    this.textureStreamId = delta.streamId;
    this.textureStreamRevision = delta.revision;
    this.render(1 / 60);
    this.emitGpuDiagnostics();
  }

  private enqueueTextureMutation(operation: () => Promise<void>): Promise<void> {
    const update = this.textureUpdateTail.then(operation);
    this.textureUpdateTail = update.catch(() => undefined);
    return update;
  }

  private initialTextureDimension(): number | undefined {
    const levels = this.adaptiveTextureDimensions;
    return levels.length > 0 ? Math.min(...levels) : this.options.maxTextureDimension;
  }

  private scheduleTextureDetailUpdate(): void {
    const levels = this.adaptiveTextureDimensions;
    const model = this.model;
    if (!levels?.length || !model || this.destroyed) return;
    const bounds = model.getBounds();
    if (!bounds) return;
    const density = this.metrics.pixelWidth / Math.max(1, this.metrics.cssWidth);
    this.desiredTextureDimension = adaptiveTextureDimension(
      Math.max(bounds.width, bounds.height) * density,
      model.textureMaximumDimension(),
      levels,
    );
    if (this.textureDetailTimer !== null) window.clearTimeout(this.textureDetailTimer);
    this.textureDetailTimer = window.setTimeout(() => {
      this.textureDetailTimer = null;
      const target = this.desiredTextureDimension;
      if (!target) return;
      const currentModel = this.model;
      const update = this.textureUpdateTail.then(async () => {
        if (!currentModel || currentModel !== this.model || this.destroyed) return;
        await currentModel.setTextureMaximumDimension(target);
        if (currentModel !== this.model || this.destroyed) return;
        this.render(1 / 60);
        this.emitGpuDiagnostics();
        if (this.desiredTextureDimension !== target) this.scheduleTextureDetailUpdate();
      });
      this.textureUpdateTail = update.catch((error) => {
        console.warn("切换 Live2D 纹理清晰度失败", error);
      });
    }, 320);
  }

  private emitGpuDiagnostics(): void {
    if (!this.options.onGpuDiagnostics) return;
    const model = this.model?.gpuDiagnostics();
    this.options.onGpuDiagnostics({
      modelId: this.currentPackage?.id ?? null,
      atlasCount: model?.atlasCount ?? 0,
      atlasDimensions: model?.atlasDimensions ?? [],
      offscreenContainerCount: model?.offscreenContainerCount ?? 0,
      texturesCreated: this.texturesCreated,
      texturesDeleted: this.texturesDeleted,
    });
  }

  private pickCandidate(
    key: string,
    candidates: readonly BehaviorCandidate[],
  ): BehaviorCandidate | null {
    const id = this.poolSelector.pick(key, candidates.map((candidate) => candidate.id));
    return id ? candidates.find((candidate) => candidate.id === id) ?? null : null;
  }

  private selectStateCandidate(value: PetState): StateBehaviorSelection {
    if (!isBehaviorState(value)) return { candidate: null, rotate: false };
    const candidates = this.currentPackage?.manifest.behaviors.states[value] ?? [];
    return {
      candidate: this.pickCandidate(`state:${value}`, candidates),
      rotate: candidates.length > 1,
    };
  }

  private refreshBehaviorConfiguration(): void {
    const emotion = this.activeEmotion;
    this.behaviorPlayback.cancel();
    this.poolSelector.clear();
    this.activeState = null;
    this.selectedEmotionCandidate = emotion
      ? this.pickCandidate(
          `emotion:${emotion}`,
          this.currentPackage?.manifest.behaviors.emotions[emotion] ?? [],
        )
      : null;
    this.setState(this.state);
  }

  private playEmotionCandidate(value: PetEmotion): void {
    if (this.speechFacialEnabled) return;
    const manifest = this.currentPackage?.manifest;
    const candidate = this.pickCandidate(
      `emotion:${value}`,
      manifest?.behaviors.emotions[value] ?? [],
    );
    this.selectedEmotionCandidate = candidate;
    if (candidate) this.behaviorPlayback.playTransient(candidate, `emotion:${value}`);
    else this.behaviorPlayback.cancelEmotion();
  }

  private render(delta: number, frameTimestamp?: number): void {
    if (this.destroyed || this.fatalError || this.gl.isContextLost()) return;
    try {
      this.clear();
      if (this.editingPreview) this.model?.updateEditingPreview();
      else this.model?.update(delta);
      this.model?.draw();
      if (this.facialOverlayState?.releasing && !this.model?.hasFacialOverlay()) {
        this.editingPreview = this.facialOverlayState.wasEditing;
        this.facialOverlayState = null;
        if (this.editingPreview) this.frameScheduler.pause();
        this.syncFrameRate();
      }
      this.checkGraphicsHealth(
        frameTimestamp === undefined ? "immediate" : "sampled",
        frameTimestamp,
      );
    } catch (error) {
      this.reportFatal(asError(error, "Live2D 预览渲染失败"));
    }
  }

  private currentFrameRate(): number {
    if (this.facialDemoActive || this.facialOverlayState) return 60;
    return live2DFrameRate(
      this.state,
      this.targetLip.volume,
      this.behaviorPlayback.hasTransient(),
      this.expressiveIdleEnabled,
    );
  }

  private syncFrameRate(): void {
    this.frameScheduler.setFrameRate(this.currentFrameRate());
  }

  private clear(): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.viewport(0, 0, this.metrics.pixelWidth, this.metrics.pixelHeight);
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.STENCIL_BUFFER_BIT);
  }

  private measure(): CubismDisplayMetrics {
    const cssWidth = Math.max(1, this.viewport instanceof Window
      ? this.viewport.innerWidth
      : this.viewport.clientWidth);
    const cssHeight = Math.max(1, this.viewport instanceof Window
      ? this.viewport.innerHeight
      : this.viewport.clientHeight);
    const density = Math.min(window.devicePixelRatio || 1, 2);
    return {
      cssWidth,
      cssHeight,
      pixelWidth: Math.max(1, Math.round(cssWidth * density)),
      pixelHeight: Math.max(1, Math.round(cssHeight * density)),
    };
  }

  private applyCanvasSize(): void {
    if (this.canvas.width !== this.metrics.pixelWidth) this.canvas.width = this.metrics.pixelWidth;
    if (this.canvas.height !== this.metrics.pixelHeight) this.canvas.height = this.metrics.pixelHeight;
    this.canvas.style.width = `${this.metrics.cssWidth}px`;
    this.canvas.style.height = `${this.metrics.cssHeight}px`;
  }

  private isVisible(): boolean {
    if (document.hidden) return false;
    if (this.viewport instanceof Window) return true;
    return this.viewport.isConnected && this.viewport.getClientRects().length > 0;
  }

  private assertGraphicsHealthy(): void {
    const error = this.gl.getError();
    if (error !== this.gl.NO_ERROR) throw new Error(`Cubism WebGL 错误 0x${error.toString(16)}`);
  }

  private checkGraphicsHealth(
    mode: "sampled" | "immediate",
    timestampMs = performance.now(),
  ): void {
    if (!this.graphicsHealthCheckCadence.shouldCheck(timestampMs, mode)) return;
    this.assertGraphicsHealthy();
  }

  private drainGraphicsErrors(): void {
    for (let count = 0; count < 32 && this.gl.getError() !== this.gl.NO_ERROR; count += 1) {
      // Context restoration may leave errors from invalidated pre-restore
      // objects. They are historical once every GPU resource is rebuilt.
    }
  }

  private reportFatal(error: Error): void {
    if (this.destroyed || this.fatalError) return;
    this.fatalError = error;
    console.error(error);
    this.options.onFatalError?.(error);
  }
}

function asError(error: unknown, fallback: string): Error {
  return error instanceof Error ? error : new Error(error ? String(error) : fallback);
}

function behaviorConfigurationChanged(
  previous: ModelPackage["manifest"]["behaviors"],
  next: ModelPackage["manifest"]["behaviors"],
): boolean {
  return JSON.stringify(previous) !== JSON.stringify(next);
}

function isBehaviorState(value: PetState): value is Exclude<PetBehaviorState, "click"> {
  return value === "standby" ||
    value === "acknowledging" ||
    value === "listening" ||
    value === "thinking";
}

function nextPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}

type ScreenRect = { x: number; y: number; width: number; height: number };

/** The whole character, small, in the bottom-right corner, with a box where the piece is. */
function drawLocator(context: CanvasRenderingContext2D, overview: HTMLCanvasElement, body: ScreenRect, piece: ScreenRect, size: number): void {
  const margin = size * 0.03;
  const x = size - overview.width - margin;
  const y = size - overview.height - margin;
  context.save();
  context.fillStyle = "rgba(12, 15, 24, 0.82)";
  context.beginPath();
  context.roundRect(x - 3, y - 3, overview.width + 6, overview.height + 6, size * 0.03);
  context.fill();
  context.globalAlpha = 0.7;
  context.drawImage(overview, x, y);
  context.globalAlpha = 1;
  const k = overview.width / Math.max(1, body.width);
  context.strokeStyle = "rgb(242, 193, 78)";
  context.lineWidth = Math.max(1.5, size / 110);
  context.strokeRect(x + (piece.x - body.x) * k, y + (piece.y - body.y) * k, Math.max(3, piece.width * k), Math.max(3, piece.height * k));
  context.restore();
}
