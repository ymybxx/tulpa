export interface PointerPoint {
  x: number;
  y: number;
}

export const MODEL_DRAG_THRESHOLD = 4;

export function modelCursorForHover(active: boolean): "pointer" | "default" {
  return active ? "pointer" : "default";
}

export {
  MAX_PET_MODEL_SCALE,
  MAX_PET_WINDOW_SCALE,
  MIN_PET_MODEL_SCALE,
  normalizePetModelScale,
  petModelScaleFromPinch,
  petWindowScale,
  petContentZoom,
} from "../shared/pet-view";

export function modelPointerMoved(
  start: PointerPoint,
  current: PointerPoint,
  threshold = MODEL_DRAG_THRESHOLD,
): boolean {
  return Math.hypot(current.x - start.x, current.y - start.y) >= threshold;
}

export function modelPointerCompletesClick(moved: boolean, cancelled: boolean): boolean {
  return !moved && !cancelled;
}

/**
 * How far a wheel event on the desktop pet scales it, as a zoom-wheel step
 * (negative grows). TA is scaled by a two-finger scroll or a mouse wheel,
 * which reach it whichever app is in front. A trackpad pinch (Chromium's
 * ctrl+wheel) does not scale it: macOS sends pinches only to the active
 * app, and making tulpa active pulls the person to the desktop its main
 * window is on.
 */
export function petScaleWheelStep(event: { ctrlKey: boolean; deltaMode: number; deltaY: number }): number {
  if (event.ctrlKey) return 0;
  if (event.deltaMode !== 0 || Math.abs(event.deltaY) >= 50) return Math.sign(event.deltaY) * 12;
  // Trackpad scrolls come as many small pixel steps.
  return event.deltaY * 0.35;
}

export function isPetScaleWheel(event: {
  ctrlKey: boolean;
  deltaMode: number;
  deltaY: number;
}): boolean {
  // Chromium exposes a trackpad pinch as ctrl+pixel-wheel. A traditional
  // notched mouse wheel reports line/page deltas. Plain pixel scrolling is a
  // two-finger trackpad scroll and must not resize the pet.
  return event.ctrlKey || event.deltaMode !== 0 || Math.abs(event.deltaY) >= 50;
}
