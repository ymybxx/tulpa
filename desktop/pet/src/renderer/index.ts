import { localizeStatic } from "../i18n/dom";
import "./styles.css";
import type { Rect } from "../shared/pet-chat";

import type { ModelRenderBundle, PetControlEvent, PetPanelPlacement, PetSnapshotEvent } from "../shared/contracts";
import {
  pointInsideInteractionControl,
  unlockControlPosition,
} from "../shared/interaction-lock";
import { RealtimeLipSync } from "./audio-lipsync";
import type { LipShape, PetRenderer } from "./pet-renderer";
import { PetScheduler } from "./scheduler";
import { SpeechFacialRuntime } from "./speech-facial";
import {
  petScaleWheelStep,
  modelCursorForHover,
  modelPointerCompletesClick,
  modelPointerMoved,
  petModelScaleFromPinch,
  petContentZoom,
} from "./model-pointer-gesture";
import {
  ModelBundleArrivalGate,
  runCurrentModelBundleArrival,
} from "./model-bundle-arrival";
import { PET_TEXTURE_DETAIL_LEVELS } from "./texture-detail";
import { petWindowScale } from "../shared/pet-view";
import { petWindowSize, placePet, resizePetBounds } from "../../electron/window-drag";

const canvas = required<HTMLCanvasElement>("pet-canvas");
// TA's panel inside the stage window; mouse positions are measured from it.
const panelElement = required<HTMLElement>("pet-panel");
let panel = { x: 0, y: 0, width: 0, height: 0 };
let panelOffset = { x: 0, y: 0 };
const unlockInteraction = required<HTMLButtonElement>("unlock-interaction");
let renderer: PetRenderer;
let scheduler: PetScheduler;
let speechFacial: SpeechFacialRuntime;
let lipSync: RealtimeLipSync;
const emptyState = required<HTMLElement>("empty-state");
const errorState = required<HTMLElement>("error-state");
let lastPointerAt = 0;
let petModelScale = 1;
let scaleSaveTimer: number | null = null;
let scaleGestureActive = false;
let pointerWithinWindow = false;
let interactionLocked = false;
let chatBounds: Rect | null = null;
let chatHovered = false;
// Where the wheel is taking TA, and where it is on screen now: each frame moves
// part of the way, so a mouse wheel's big notches glide instead of jumping.
let targetScale = 1;
let scalePoint = { x: 0, y: 0 };
let scaleFrame: number | null = null;
let scaleFrameAt = 0;
let lastChatGeometry = "";
const modelBundleArrivals = new ModelBundleArrivalGate();
let modelDrag: {
  pointerId: number;
  startScreenX: number;
  startScreenY: number;
  moving: boolean;
} | null = null;

void initialize();

async function initialize(): Promise<void> {
  const [scale, modelOffset, locked, textureMaximum, placement] = await Promise.all([
    window.desktopPet.getPetModelScale(),
    window.desktopPet.getPetModelOffset(),
    window.desktopPet.getInteractionLocked(),
    window.desktopPet.getPetTextureMaximum(),
    window.desktopPet.getPetPanel(),
  ]);
  // Laid out before the renderer measures it.
  if (placement) placePanel(placement);
  renderer = await createRenderer(textureMaximum);
  petModelScale = scale;
  interactionLocked = locked;
  renderer.setModelOffset?.(modelOffset.x, modelOffset.y);
  renderer.setModelZoom?.(petContentZoom(scale));
  applyInteractionLockState(interactionLocked);
  speechFacial = new SpeechFacialRuntime(renderer, window.desktopPet.speechFacial);
  scheduler = new PetScheduler(renderer, speechFacial);
  lipSync = new RealtimeLipSync((shape) => renderer.setLipShape(shape));
  // Prepare the worklet before the first TTS packet. Lazy initialization used
  // to route short sentences through the volume-only ParamA fallback.
  void lipSync.initialize().catch(() => undefined);
  // Capture before subscribing: every pushed selection observed from this
  // point invalidates the asynchronous initial snapshot request below.
  const initialModelEpoch = modelBundleArrivals.capture();
  bindUi();
  bindPetEvents();
  const selected = await runCurrentModelBundleArrival(
    modelBundleArrivals,
    initialModelEpoch,
    () => window.desktopPet.getSelectedModelRenderBundle(),
  );
  if (selected.current) {
    emptyState.hidden = selected.value !== null;
    if (selected.value) await loadModel(selected.value, initialModelEpoch);
  }
  if (!window.Live2DCubismCore) {
    showError(new Error("缺少 live2dcubismcore.min.js"));
  }
  window.desktopPet.onPetPanel(applyPanel);
  window.addEventListener("beforeunload", () => {
    renderer.destroy();
    void lipSync.close();
  });
  scheduleAmbientGaze();
}

async function createRenderer(textureMaximum: number): Promise<PetRenderer> {
  if (!window.Live2DCubismCore) return new MissingCoreRenderer();
  try {
    const { Live2DRenderer } = await import("./live2d-renderer");
    return new Live2DRenderer(canvas, panelElement, {
      adaptiveTextureDimensions: PET_TEXTURE_DETAIL_LEVELS,
      maxAdaptiveTextureDimension: textureMaximum,
    });
  } catch (error) {
    showError(error);
    return new MissingCoreRenderer();
  }
}

function bindPetEvents(): void {
  window.desktopPet.onConnection((connected) => {
    if (!connected) { lipSync.stop(); speechFacial.release(); }
    else void speechFacial.refresh().catch(error => console.warn("同步桌宠表情配置失败", error));
  });
  window.desktopPet.onPetWindowVisibility((visible) => {
    renderer.setPaused(!visible);
  });
  window.desktopPet.onControl(handleControl);
  window.desktopPet.onAudio((frame) => {
    if (scheduler.acceptsAudio(frame.generation)) lipSync.push(frame);
  });
  window.desktopPet.onCommand((command) => {
    if (command.type === "open-settings") window.desktopPet.showMainView("models");
    else if (command.type === "model-selected") {
      const epoch = modelBundleArrivals.advance();
      void loadModel(command.bundle, epoch);
    }
    else if (command.type === "facial-bindings-changed") {
      void speechFacial.refresh(command.modelId).catch(error => console.warn("更新桌宠表情绑定失败", error));
    }
    else if (command.type === "pet-model-scale-changed") {
      // Mid-gesture the page is ahead of the app; its own scale stands.
      if (scaleGestureActive) return;
      petModelScale = command.scale;
      renderer.setModelZoom?.(petContentZoom(command.scale));
      refreshChatBounds();
      window.desktopPet.requestPointerRefresh();
    }
    else if (command.type === "pet-model-offset-changed") {
      panelOffset = { ...command.offset };
      renderer.setModelOffset?.(command.offset.x, command.offset.y);
      refreshChatBounds();
      window.desktopPet.requestPointerRefresh();
    }
    else if (command.type === "pet-texture-maximum-changed") {
      renderer.setTextureMaximumDimension?.(command.maximum);
    }
    else if (command.type === "pet-pointer-position") {
      updatePointerPosition(command.x, command.y);
    }
    else if (command.type === "interaction-lock-changed") {
      applyInteractionLockState(command.locked);
    }
    else if (command.type === "texture-materials-updated") {
      void renderer.updateTextureAtlases(command.change.delta).catch(showError);
    } else if (command.type === "texture-compaction-warning") {
      console.warn(command.message);
    }
  });
}

function handleControl(event: PetControlEvent): void {
  scheduler.apply(event);
  if (event.type === "pet.snapshot") {
    const events = (event as PetSnapshotEvent).payload.events;
    for (const cached of [...events].sort((left, right) => left.sequence - right.sequence)) {
      handleAudioControl(cached);
    }
    return;
  }
  handleAudioControl(event);
}

function handleAudioControl(event: PetControlEvent): void {
  if (event.type === "pet.audio.start" && event.generation && scheduler.acceptsAudio(event.generation)) {
    const sampleRate = Number(event.payload.sampleRate);
    lipSync.start(event.generation, Number.isFinite(sampleRate) ? sampleRate : 16_000);
  } else if (event.type === "pet.audio.pause") {
    lipSync.pause(event.generation);
  } else if (
    event.type === "pet.audio.stop" ||
    event.type === "pet.generation.cancel"
  ) {
    lipSync.stop(event.generation);
  }
}

function bindUi(): void {
  unlockInteraction.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  unlockInteraction.addEventListener("pointerleave", () => {
    if (interactionLocked) window.desktopPet.setInteractionUnlockHover(false);
  });
  unlockInteraction.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    applyInteractionLockState(false);
    window.desktopPet.unlockInteraction();
  });
  canvas.addEventListener("pointerdown", (event) => {
    if (interactionLocked) return;
    if (!event.isPrimary || event.button !== 0) return;
    const point = inPanel(event.clientX, event.clientY);
    if (!renderer.isPointInDragArea(point.x, point.y)) return;
    pointerWithinWindow = true;
    event.preventDefault();
    window.desktopPet.setMousePassthrough(false);
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.remove("drag-hover");
    canvas.classList.add("dragging");
    applyCursor("grabbing");
    modelDrag = {
      pointerId: event.pointerId,
      startScreenX: event.screenX,
      startScreenY: event.screenY,
      moving: false,
    };
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!modelDrag || modelDrag.pointerId !== event.pointerId) return;
    if (!modelDrag.moving) {
      if (!modelPointerMoved(
        { x: modelDrag.startScreenX, y: modelDrag.startScreenY },
        { x: event.screenX, y: event.screenY },
      )) return;
      modelDrag.moving = true;
      applyCursor("grabbing");
      window.desktopPet.startWindowDrag(
        modelDrag.startScreenX, modelDrag.startScreenY, renderer.getModelBounds(),
      );
    }
    window.desktopPet.moveWindowDrag(event.screenX, event.screenY);
  });
  const endModelDrag = (event: PointerEvent, cancelled = false) => {
    if (!modelDrag || modelDrag.pointerId !== event.pointerId) return;
    const completed = modelDrag;
    modelDrag = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    if (completed.moving) {
      window.desktopPet.endWindowDrag();
    } else if (modelPointerCompletesClick(completed.moving, cancelled)) {
      scheduler.interact();
    }
    canvas.classList.remove("dragging");
    const point = inPanel(event.clientX, event.clientY);
    setDragHover(renderer.isPointInDragArea(point.x, point.y));
    if (!pointerWithinWindow) window.desktopPet.setMousePassthrough(true);
    window.desktopPet.requestPointerRefresh();
  };
  canvas.addEventListener("pointerup", (event) => endModelDrag(event));
  canvas.addEventListener("pointercancel", (event) => endModelDrag(event, true));
  canvas.addEventListener("wheel", (event) => {
    if (interactionLocked) return;
    // A pinch neither scales TA nor zooms its page.
    if (event.ctrlKey) { event.preventDefault(); return; }
    const step = petScaleWheelStep(event);
    if (step === 0) return;
    if (!scaleGestureActive) {
      const point = inPanel(event.clientX, event.clientY);
      if (!renderer.isPointInDragArea(point.x, point.y)) return;
      pointerWithinWindow = true;
      scaleGestureActive = true;
      window.desktopPet.setMousePassthrough(false);
    }
    event.preventDefault();
    scalePet(step, event.clientX, event.clientY);
  }, { passive: false });

  document.addEventListener("pointermove", (event) => {
    if (interactionLocked) return;
    const point = inPanel(event.clientX, event.clientY);
    updatePointerPosition(point.x, point.y);
  });
  document.addEventListener("mousemove", (event) => {
    if (!interactionLocked) return;
    const point = inPanel(event.clientX, event.clientY);
    updatePointerPosition(point.x, point.y);
  });
  document.addEventListener("mouseleave", () => {
    // macOS may send mouseleave during activation/Space changes even while
    // the cursor remains on the pet. Check the actual cursor before releasing.
    if (window.desktopPet.requestPointerRefresh()) return;
    pointerWithinWindow = false;
    reportChatHover(false);
    setDragHover(false);
    if (!interactionLocked) {
      if (!modelDrag && !scaleGestureActive) window.desktopPet.setMousePassthrough(true);
      return;
    }
    unlockInteraction.classList.remove("visible");
    window.desktopPet.setInteractionUnlockHover(false);
  });
  document.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    window.desktopPet.showContextMenu();
  });
}

function placePanel(next: PetPanelPlacement): boolean {
  const resized = next.width !== panel.width || next.height !== panel.height;
  panel = { x: next.x, y: next.y, width: next.width, height: next.height };
  panelOffset = { ...next.offset };
  panelElement.style.left = `${next.x}px`;
  panelElement.style.top = `${next.y}px`;
  panelElement.style.width = `${next.width}px`;
  panelElement.style.height = `${next.height}px`;
  return resized;
}

/**
 * The panel moved or changed size (dragging, scaling). Its place, its size and
 * TA's picture all change in this one frame of this one page: no window moves,
 * so nothing shows a stale frame in the wrong place.
 */
function applyPanel(next: PetPanelPlacement): void {
  const resized = placePanel(next);
  renderer.setModelOffset?.(next.offset.x, next.offset.y);
  if (resized) renderer.resize();
  refreshChatBounds();
  if (unlockInteraction.classList.contains("visible")) positionUnlockControl();
}

/** A page position (CSS pixels in the stage) as a position in TA's panel. */
function inPanel(clientX: number, clientY: number): { x: number; y: number } {
  return { x: clientX - panel.x, y: clientY - panel.y };
}

/** How long TA takes to catch up most of the way to where the wheel sent it. */
const SCALE_EASE_MS = 45;

/**
 * Scales TA by a zoom-wheel step (negative grows). The step moves the target;
 * every frame TA eases toward it, so scaling glides at the display's rate
 * whether the steps come from a trackpad or a notched wheel.
 */
function scalePet(deltaY: number, clientX: number, clientY: number): void {
  if (scaleFrame === null) {
    targetScale = petModelScale;
    scaleFrameAt = performance.now();
  }
  targetScale = petModelScaleFromPinch(targetScale, deltaY);
  scalePoint = { x: clientX, y: clientY };
  if (scaleFrame === null) scaleFrame = requestAnimationFrame(easeScale);
}

function easeScale(now: number): void {
  const elapsed = Math.max(0, Math.min(50, now - scaleFrameAt));
  scaleFrameAt = now;
  // Eased in log space: growing and shrinking feel the same.
  const current = Math.log(petModelScale), target = Math.log(targetScale);
  const next = Math.abs(target - current) < 0.002
    ? target
    : current + (target - current) * (1 - Math.exp(-elapsed / SCALE_EASE_MS));
  applyScale(Math.exp(next), scalePoint.x, scalePoint.y);
  scaleFrame = next === target ? null : requestAnimationFrame(easeScale);
}

/**
 * Scales TA here, in this frame, by the same rules the app uses: the panel
 * grows around the pointer and stays on the display, the model keeps its
 * place in it. The app only follows (HUD, saving); a round trip for every
 * step made some frames miss the step and the next one jump.
 */
function applyScale(scale: number, clientX: number, clientY: number): void {
  const previous = petModelScale;
  petModelScale = scale;
  const before = petWindowScale(previous), after = petWindowScale(petModelScale);
  if (Math.abs(after - before) > 1e-6) {
    const ratio = after / before;
    const resized = resizePetBounds(panel, petWindowSize(after), { x: clientX, y: clientY });
    const offset = { x: panelOffset.x * ratio, y: panelOffset.y * ratio };
    // The stage covers the display's work area, so that area is the whole page.
    const placement = placePet(
      { x: resized.x + offset.x, y: resized.y + offset.y }, resized, null,
      { x: 0, y: 0, width: window.innerWidth, height: window.innerHeight },
    );
    applyPanel({ ...resized, ...placement.window, offset: placement.offset });
  }
  renderer.setModelZoom?.(petContentZoom(petModelScale));
  window.desktopPet.previewPetModelScale(petModelScale, { ...panel, offset: { ...panelOffset } });
  if (scaleSaveTimer !== null) window.clearTimeout(scaleSaveTimer);
  scaleSaveTimer = window.setTimeout(() => {
    scaleSaveTimer = null;
    scaleGestureActive = false;
    if (!pointerWithinWindow) window.desktopPet.setMousePassthrough(true);
    window.desktopPet.requestPointerRefresh();
    void window.desktopPet.setPetModelScale(petModelScale).catch(showError);
  }, 180);
}

/** `x`, `y`: the pointer in TA's panel. */
function updatePointerPosition(x: number, y: number): void {
  lastPointerAt = performance.now();
  pointerWithinWindow = x >= 0 && y >= 0 && x < panel.width && y < panel.height;
  if (!interactionLocked && (modelDrag || scaleGestureActive)) {
    window.desktopPet.setMousePassthrough(false);
    return;
  }
  const overModel = pointerWithinWindow && renderer.isPointInDragArea(x, y);
  reportChatHover(overModel && !interactionLocked);
  if (interactionLocked) {
    if (overModel) {
      positionUnlockControl();
      unlockInteraction.classList.add("visible");
    } else if (!pointerWithinWindow) {
      unlockInteraction.classList.remove("visible");
    }
    const overUnlock = unlockInteraction.classList.contains("visible") &&
      pointInsideInteractionControl({ x: x + panel.x, y: y + panel.y }, unlockInteraction.getBoundingClientRect());
    window.desktopPet.setInteractionUnlockHover(overUnlock);
    return;
  }
  const target = pointerWithinWindow ? document.elementFromPoint(x + panel.x, y + panel.y) : null;
  const overControls = Boolean(target?.closest("#empty-state, #error-state"));
  setDragHover(overModel);
  window.desktopPet.setMousePassthrough(!overControls && !overModel);
  if (pointerWithinWindow) {
    renderer.setGaze(
      (x / Math.max(1, panel.width) - 0.5) * 2,
      -(y / Math.max(1, panel.height) - 0.5) * 2,
    );
  }
}

function refreshChatBounds(): void {
  // Capture once after resize/load, not once per Live2D animation frame.
  requestAnimationFrame(() => {
    chatBounds = renderer.getModelBounds();
    reportChatHover(chatHovered);
  });
}

function reportChatHover(hovered: boolean): void {
  chatHovered = hovered;
  const geometry = { bounds: chatBounds, hovered };
  const signature = JSON.stringify(geometry);
  if (signature === lastChatGeometry) return;
  lastChatGeometry = signature;
  window.desktopPet.reportChatGeometry(geometry);
}

function applyInteractionLockState(value: boolean): void {
  interactionLocked = value;
  canvas.classList.remove("drag-hover", "dragging");
  applyCursor("default");
  unlockInteraction.classList.remove("visible");
  if (!value) window.desktopPet.setInteractionUnlockHover(false);
  window.desktopPet.requestPointerRefresh();
}

function setDragHover(active: boolean): void {
  if (interactionLocked || modelDrag || scaleGestureActive) active = false;
  canvas.classList.toggle("drag-hover", active);
  if (!canvas.classList.contains("dragging")) {
    applyCursor(modelCursorForHover(active));
  }
}

let systemCursor: "default" | "grabbing" | "pointer" = "default";

function applyCursor(cursor: "default" | "grabbing" | "pointer"): void {
  canvas.style.cursor = cursor;
  document.documentElement.style.cursor = cursor;
  document.body.style.cursor = cursor;
  // A hand is set again on every move over the pet, as another app may have
  // changed the cursor since; an arrow only on leaving the model within the
  // window, never over another app's window where it would replace its cursor.
  if (cursor === "default" && (cursor === systemCursor || !pointerWithinWindow)) {
    systemCursor = cursor;
    return;
  }
  systemCursor = cursor;
  window.desktopPet.setPetCursor(cursor);
}

function positionUnlockControl(): void {
  const modelBounds = renderer.getModelBounds();
  if (!modelBounds) return;
  // A shrunken pet's window can be narrower than the labelled pill; then only the padlock shows.
  unlockInteraction.classList.remove("compact");
  if (unlockInteraction.offsetWidth + 16 > panel.width) unlockInteraction.classList.add("compact");
  const position = unlockControlPosition(
    modelBounds,
    {
      width: unlockInteraction.offsetWidth || 34,
      height: unlockInteraction.offsetHeight || 34,
    },
    { width: panel.width, height: panel.height },
  );
  unlockInteraction.style.left = `${position.left}px`;
  unlockInteraction.style.top = `${position.top}px`;
}

async function loadModel(bundle: ModelRenderBundle, epoch: number): Promise<void> {
  speechFacial.clear();
  try {
    const loaded = await runCurrentModelBundleArrival(
      modelBundleArrivals,
      epoch,
      () => renderer.loadModel(bundle),
    );
    if (!loaded.current) return;
    void speechFacial.configure(bundle.model.id).catch(error => console.warn("准备桌宠表情失败", error));
    scheduler.replay();
    emptyState.hidden = true;
    errorState.hidden = true;
    refreshChatBounds();
    window.desktopPet.requestPointerRefresh();
  } catch (error) {
    if (modelBundleArrivals.isCurrent(epoch)) showError(error);
  }
}

function scheduleAmbientGaze(): void {
  const update = () => {
    if (performance.now() - lastPointerAt > 1_500) {
      renderer.setGaze(randomBetween(-0.35, 0.35), randomBetween(-0.18, 0.22), "ambient");
    }
    window.setTimeout(update, randomBetween(2_200, 4_800));
  };
  window.setTimeout(update, 2_000);
}

function showError(error: unknown): void {
  const detail = error instanceof Error ? error.message : String(error);
  errorState.textContent = detail.includes("live2dcubismcore")
    ? "缺少 Cubism Core，请打开 App 设置重新准备 Live2D 资源。"
    : detail;
  errorState.hidden = false;
}

function required<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`缺少界面元素 #${id}`);
  return element as T;
}

function randomBetween(minimum: number, maximum: number): number {
  return minimum + Math.random() * (maximum - minimum);
}

class MissingCoreRenderer implements PetRenderer {
  async loadModel(): Promise<void> {
    throw new Error("缺少 live2dcubismcore.min.js");
  }

  async updateTextureAtlases(): Promise<void> {}

  setState(): void {}
  setEmotion(): void {}
  playClick(): void {}
  setLipShape(_value: LipShape): void {}
  setGaze(): void {}
  getModelBounds(): null {
    return null;
  }
  isPointInDragArea(): boolean {
    return false;
  }
  cancelGeneration(): void {}
  resize(): void {}
  setPaused(): void {}
  destroy(): void {}
}

localizeStatic(document);
