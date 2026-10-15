import { bindLocalizedText } from "../i18n/dom";
import type {
  BehaviorCandidate,
  ModelPackage,
  ModelRenderBundle,
  NativeModelParameter,
  NativeModelLayer,
  PetManifest,
  TextureAtlasDelta,
} from "../shared/contracts";
import type { LipShape } from "../renderer/pet-renderer";
import { Live2DRenderer } from "../renderer/live2d-renderer";
import { characterSnapshot } from "../renderer/preview-snapshot";
import { PreviewValidationSequence } from "./preview-validation";
import { MicrophoneLipSyncTest } from "./microphone-lipsync-test";
import type { FacialDemoSnapshot } from "../renderer/facial-parameter-controller";
import type { FacialMotionBindings, FacialOverlayBindings } from "../shared/facial-bindings";
import { isPetScaleWheel } from "../renderer/model-pointer-gesture";

const MIN_VIEW_ZOOM = 0.6;
const MAX_VIEW_ZOOM = 6;

export class ModelManagerPreview {
  private readonly renderer: Live2DRenderer;
  private readonly validation: PreviewValidationSequence;
  private readonly microphoneLipSync: MicrophoneLipSyncTest;
  private animated = false;
  /** The mode the person picked with 静态调试 / 动态预览; automatic resets come back to it. */
  private preferAnimated = false;
  private facialTestPreviousAnimation: boolean | null = null;
  private facialTestOverlay = false;
  /** The face framing 说话表情 asked for; 1 when none. The view zoom does the work. */
  private facialTestZoom = 1;
  /**
   * Pinch or mouse wheel zooms about the cursor, like the desktop pet; while
   * zoomed, a two-finger scroll or a drag moves the view. The renderer zooms
   * the model itself, so it stays sharp and clicks still land on the right piece.
   */
  private viewZoom = 1;
  private viewOffset = { x: 0, y: 0 };
  private viewDrag: { pointerId: number; x: number; y: number; moved: boolean } | null = null;
  private suppressClick = false;
  private readonly zoomReset: HTMLButtonElement | null;
  private readonly onViewWheel = (event: WheelEvent) => {
    if (!this.current) return;
    if (isPetScaleWheel(event)) {
      event.preventDefault();
      const deltaY = event.ctrlKey ? event.deltaY : Math.sign(event.deltaY) * 12;
      const point = this.canvasPoint(event);
      this.zoomViewAt(this.viewZoom * Math.exp(-deltaY * 0.01), point.x, point.y);
    } else if (this.viewZoom !== 1) {
      event.preventDefault();
      this.moveView(-event.deltaX, -event.deltaY);
    }
  };
  private readonly viewPointer = {
    pointerdown: (event: PointerEvent) => {
      if (this.viewZoom === 1 || !event.isPrimary || event.button !== 0) return;
      this.viewDrag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
    },
    pointermove: (event: PointerEvent) => {
      const drag = this.viewDrag;
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (event.buttons === 0) { this.viewDrag = null; return; }
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      // A short wobble is still a click on a piece.
      if (!drag.moved && Math.hypot(dx, dy) < 4) return;
      if (!drag.moved) {
        drag.moved = true;
        this.canvas.setPointerCapture(event.pointerId);
        this.viewport.classList.add("preview-view-dragging");
      }
      drag.x = event.clientX;
      drag.y = event.clientY;
      this.moveView(dx, dy);
    },
    pointerup: (event: PointerEvent) => {
      const drag = this.viewDrag;
      if (!drag || event.pointerId !== drag.pointerId) return;
      this.viewDrag = null;
      this.viewport.classList.remove("preview-view-dragging");
      // The click that ends a drag must not pick a piece.
      if (drag.moved) this.suppressClick = true;
    },
  };
  private layerPick: ((ids: string[], point: { x: number; y: number }) => void) | null = null;
  private readonly onLayerPick = (event: MouseEvent) => {
    if (this.suppressClick) {
      this.suppressClick = false;
      return;
    }
    if (!this.layerPick) return;
    const bounds = this.canvas.getBoundingClientRect();
    const point = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
    this.layerPick(this.renderer.pickLayers(point.x, point.y), point);
  };

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly viewport: HTMLElement,
    onFatalError?: (error: Error) => void,
    private readonly onAnimationChanged: (animated: boolean) => void = () => {},
  ) {
    this.renderer = new Live2DRenderer(canvas, viewport, {
      onFatalError,
      preserveDrawingBuffer: true,
      maxTextureDimension: 2048,
      onGpuDiagnostics: (value) => console.debug("[Live2D preview GPU]", value),
    });
    this.microphoneLipSync = new MicrophoneLipSyncTest((shape) =>
      this.renderer.setLipShape(shape)
    );
    this.renderer.setEditingPreview(true);
    this.zoomReset = this.viewport.ownerDocument?.createElement("button") ?? null;
    if (this.zoomReset) {
      this.zoomReset.type = "button";
      this.zoomReset.className = "preview-zoom-reset";
      this.zoomReset.hidden = true;
      this.zoomReset.addEventListener("click", () => this.resetView());
      this.viewport.append(this.zoomReset);
    }
    this.viewport.addEventListener?.("wheel", this.onViewWheel, { passive: false });
    for (const type of Object.keys(this.viewPointer) as Array<keyof typeof this.viewPointer>) {
      this.canvas.addEventListener?.(type, this.viewPointer[type]);
    }
    this.validation = new PreviewValidationSequence({
      setState: (value) => this.renderer.setState(value),
      setGaze: (x, y) => this.renderer.setGaze(x, y),
      setLipShape: (value) => this.renderer.setLipShape(value),
      playFirstMotion: () => {
        const id = Object.keys(this.current?.manifest.motionAssets ?? {})[0];
        if (id) this.renderer.previewMotionAsset(id);
      },
      reset: () => this.reset(),
    });
  }

  private current: ModelPackage | null = null;

  async load(bundle: ModelRenderBundle): Promise<string | null> {
    if (bundle.model.id !== this.current?.id) this.resetView();
    this.endFacialTest();
    this.highlightLayers([]);
    this.validation.cancel();
    this.setAnimated(false);
    await this.renderer.loadModel(bundle);
    const displayed = this.renderer.displayedModelId();
    if (displayed === bundle.model.id) {
      this.current = bundle.model;
      if (this.preferAnimated) this.playAwakeIdle();
    }
    return displayed;
  }

  displayedModelId(): string | null {
    return this.renderer.displayedModelId();
  }

  async updateTextureAtlases(delta: TextureAtlasDelta): Promise<void> {
    await this.renderer.updateTextureAtlases(delta);
  }

  startValidationSequence(): void {
    this.reset();
    this.setAnimated(true);
    this.validation.start();
  }

  playAwakeIdle(): void {
    this.reset();
    this.setAnimated(true);
    this.renderer.setState("listening");
  }

  private setAnimated(value: boolean): void {
    this.animated = value;
    this.renderer.setEditingPreview(!value);
    this.onAnimationChanged(value);
  }

  playMotion(id: string): void {
    this.reset();
    this.setAnimated(true);
    this.renderer.previewMotionAsset(id);
  }

  showExpression(id: string): void {
    this.reset();
    this.setAnimated(true);
    this.renderer.previewExpressionAsset(id);
  }

  async previewCandidate(candidate: BehaviorCandidate, mode: "transient" | "state" = "transient"): Promise<void> {
    this.reset();
    if (mode === "state") await this.stopMicrophoneLipTest();
    this.setAnimated(true);
    await this.renderer.previewBehaviorCandidate(candidate, mode);
  }

  configureLipSync(value: PetManifest["lipsync"]): void {
    this.renderer.configureLipSync(value);
  }

  configureNaturalIdle(value: PetManifest["naturalIdle"]): void {
    this.renderer.configureNaturalIdle(value);
  }

  configureDefaultAppearance(value: PetManifest["defaultAppearance"]): void {
    this.validation.cancel();
    // The desktop pet wears its default appearance while moving too; only stop if the person chose still.
    if (!this.preferAnimated) this.setAnimated(false);
    this.renderer.configureDefaultAppearance(value);
  }

  nativeParameters(): NativeModelParameter[] {
    return this.renderer.nativeParameters();
  }

  beginFacialTest(motion?: FacialMotionBindings): void {
    if (this.facialTestPreviousAnimation !== null) return;
    this.validation.cancel();
    this.facialTestPreviousAnimation = this.animated;
    this.renderer.beginFacialDemo(motion);
    this.onAnimationChanged(true);
  }

  beginFacialSpeech(bindings: FacialOverlayBindings): void {
    if (this.facialTestPreviousAnimation !== null) return;
    this.validation.cancel();
    this.facialTestPreviousAnimation = this.animated;
    this.facialTestOverlay = true;
    this.renderer.beginFacialOverlay(bindings);
    this.onAnimationChanged(true);
  }

  applyFacialTest(targets: Record<string, number>): void {
    this.renderer.setFacialDemoTargets(targets);
  }

  /** The speaking mouth, from the same lip sync TA speaks with on the desktop. */
  setFacialTestLip(value: LipShape): void {
    this.renderer.setLipShape(value);
  }

  facialTestSnapshot(): FacialDemoSnapshot {
    return this.renderer.facialDemoSnapshot();
  }

  endFacialTest(): void {
    if (this.facialTestPreviousAnimation === null) return;
    const animated = this.facialTestPreviousAnimation;
    this.facialTestPreviousAnimation = null;
    this.renderer.setFacialDemoSpeechLevel(0);
    if (this.facialTestOverlay) {
      this.facialTestOverlay = false;
      this.renderer.endFacialOverlay();
      this.onAnimationChanged(animated);
      return;
    }
    this.renderer.endFacialDemo();
    this.setAnimated(animated);
    if (animated) this.renderer.setState("listening");
  }

  /**
   * Frames the face at `scale` with the same zoom the wheel uses, so the
   * person can keep zooming and moving from there.
   */
  setFacialTestZoom(scale: number): void {
    this.facialTestZoom = Number.isFinite(scale) ? Math.max(1, Math.min(4, scale)) : 1;
    this.resetView();
    if (this.facialTestZoom === 1 || !this.current) return;
    // What is drawn, not the model's full box: hidden long hair or effects would put the "face" at the chest.
    const bounds = this.renderer.shownBounds() ?? this.renderer.getModelBounds();
    if (!bounds) return;
    const faceX = bounds.x + bounds.width / 2;
    const faceY = bounds.y + bounds.height * 0.13;
    this.zoomViewAt(this.facialTestZoom, faceX, faceY);
    // The face a little above the middle, with room for the mouth and chin.
    this.moveView(this.canvas.clientWidth / 2 - faceX, this.canvas.clientHeight * 0.38 - faceY);
  }

  /** Back to the face framing after the person moved the view. */
  resetFacialTestPan(): void {
    this.setFacialTestZoom(this.facialTestZoom);
  }

  private canvasPoint(event: MouseEvent): { x: number; y: number } {
    const bounds = this.canvas.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  }

  /** Zooms keeping the point under the cursor where it is. */
  private zoomViewAt(zoom: number, x: number, y: number): void {
    const next = Math.max(MIN_VIEW_ZOOM, Math.min(MAX_VIEW_ZOOM, zoom));
    if (next === this.viewZoom) return;
    // The renderer zooms about the model's center; shift so the cursor's point stays put.
    const bounds = this.renderer.getModelBounds();
    const ratio = next / this.viewZoom;
    if (bounds) {
      this.viewOffset = {
        x: this.viewOffset.x + (1 - ratio) * (x - (bounds.x + bounds.width / 2)),
        y: this.viewOffset.y + (1 - ratio) * (y - (bounds.y + bounds.height / 2)),
      };
    }
    this.viewZoom = Math.abs(next - 1) < 0.02 ? 1 : next;
    if (this.viewZoom === 1) this.viewOffset = { x: 0, y: 0 };
    this.applyView();
  }

  private moveView(dx: number, dy: number): void {
    this.viewOffset = { x: this.viewOffset.x + dx, y: this.viewOffset.y + dy };
    this.applyView();
  }

  resetView(): void {
    if (this.viewZoom === 1 && !this.viewOffset.x && !this.viewOffset.y) return;
    this.viewZoom = 1;
    this.viewOffset = { x: 0, y: 0 };
    this.applyView();
  }

  private applyView(): void {
    this.renderer.setModelZoom(this.viewZoom);
    this.renderer.setModelOffset(this.viewOffset.x, this.viewOffset.y);
    if (this.zoomReset) {
      this.zoomReset.hidden = this.viewZoom === 1 && !this.viewOffset.x && !this.viewOffset.y;
      bindLocalizedText(this.zoomReset, "{{v0}}% · 复位", { v0: Math.round(this.viewZoom * 100) });
    }
    this.viewport.classList.toggle("preview-view-zoomed", this.viewZoom !== 1);
  }

  nativeLayers(): NativeModelLayer[] { return this.renderer.nativeLayers(); }
  highlightLayers(ids: readonly string[]): void { this.renderer.highlightLayers(ids); }

  private focusReset = 0;
  /**
   * Dims everything but `ids` and zooms the preview onto them; an empty list
   * zooms back out shortly after, so moving between tiles doesn't bounce.
   */
  focusLayers(ids: readonly string[]): void {
    window.clearTimeout(this.focusReset);
    this.renderer.highlightLayers(ids);
    // The person set their own zoom; pointing at a piece only lights it up.
    if (this.viewZoom !== 1) return;
    const bounds = ids.length ? this.renderer.layerBounds(ids) : null;
    if (!bounds) {
      this.focusReset = window.setTimeout(() => {
        this.canvas.style.transition = "transform 220ms ease";
        this.canvas.style.transform = "";
      }, 160);
      return;
    }
    const width = this.canvas.clientWidth, height = this.canvas.clientHeight;
    const scale = Math.max(1, Math.min(4, Math.min(width / Math.max(1, bounds.width), height / Math.max(1, bounds.height)) * 0.5));
    const x = width / 2 - (bounds.x + bounds.width / 2) * scale;
    const y = height / 2 - (bounds.y + bounds.height / 2) * scale;
    this.canvas.style.transition = "transform 220ms ease";
    this.canvas.style.transformOrigin = "0 0";
    this.canvas.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  }

  /** Drawables on screen right now, or null before a model is shown. */
  shownLayerIds(): Set<string> | null {
    return this.renderer.shownLayerIds();
  }

  /** Thumbnails of each group as the preview shows it now. */
  capturePieces(groups: readonly (readonly string[])[], size: number): HTMLCanvasElement[] {
    return this.renderer.capturePieces(groups, size);
  }
  /** `point` is the click position within the preview canvas, in CSS pixels. */
  setLayerPickHandler(handler: ((ids: string[], point: { x: number; y: number }) => void) | null): void {
    this.canvas.removeEventListener("click", this.onLayerPick);
    this.layerPick = handler;
    this.canvas.classList.toggle("layer-picking", !!handler);
    if (handler) this.canvas.addEventListener("click", this.onLayerPick);
    else this.highlightLayers([]);
  }

  previewLipShape(value: LipShape): void {
    this.reset();
    this.setAnimated(true);
    this.renderer.setState("speaking");
    this.renderer.setLipShape(value);
  }

  async startMicrophoneLipTest(): Promise<void> {
    this.reset();
    this.setAnimated(true);
    this.renderer.setState("speaking");
    try {
      await this.microphoneLipSync.start();
    } catch (error) {
      this.renderer.setLipShape({ volume: 0, a: 0, e: 0, i: 0, o: 0, u: 0 });
      this.setAnimated(false);
      throw error;
    }
  }

  async stopMicrophoneLipTest(): Promise<void> {
    this.renderer.setLipShape({ volume: 0, a: 0, e: 0, i: 0, o: 0, u: 0 });
    this.returnToPreferredMode();
    await this.microphoneLipSync.stop();
  }

  stopLipPreview(): void {
    this.renderer.setLipShape({ volume: 0, a: 0, e: 0, i: 0, o: 0, u: 0 });
    this.returnToPreferredMode();
  }

  /** After a test ends: idle again if the person picked 动态预览, otherwise still. */
  private returnToPreferredMode(): void {
    if (this.preferAnimated) {
      this.renderer.setState("listening");
      this.setAnimated(true);
    } else {
      this.setAnimated(false);
    }
  }

  setPreferredMode(animated: boolean): void {
    this.preferAnimated = animated;
  }

  /** A clean pose in the mode the person picked, rather than always static. */
  settle(): void {
    if (this.preferAnimated) this.playAwakeIdle();
    else this.reset();
  }

  reset(): void {
    this.endFacialTest();
    this.validation.cancel();
    this.setAnimated(false);
    this.renderer.resetPreview();
  }

  resize(): void {
    this.renderer.resize();
    if (this.facialTestZoom !== 1) this.setFacialTestZoom(this.facialTestZoom);
  }

  setPaused(value: boolean): void {
    this.renderer.setPaused(value);
  }

  async capturePng(): Promise<ArrayBuffer> {
    await nextPaint();
    const blob = await characterSnapshot(this.canvas);
    if (!blob) throw new Error("生成模型预览图失败");
    return blob.arrayBuffer();
  }

  destroy(): void {
    this.setFacialTestZoom(1);
    this.endFacialTest();
    this.setLayerPickHandler(null);
    this.validation.cancel();
    void this.microphoneLipSync.stop();
    this.renderer.destroy();
  }
}

function nextPaint(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}
