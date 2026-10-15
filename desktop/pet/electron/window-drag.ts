export interface ScreenPoint {
  x: number;
  y: number;
}

export interface ScreenBounds extends ScreenPoint {
  width: number;
  height: number;
}

const ELECTRON_MIN_COORDINATE = -2_147_483_648;
const ELECTRON_MAX_COORDINATE = 2_147_483_647;

const MIN_VISIBLE_FRACTION = 0.2;
const MIN_VISIBLE_PIXELS = 96;

export interface PetPlacement {
  window: ScreenPoint;
  offset: ScreenPoint;
}

/**
 * The one rule for every pet move. `target` is where the panel would sit if
 * the model had no offset. The panel goes as close to it as the display
 * allows (centred when larger than the display); the model shifts inside the
 * panel for the rest, so it can hang past a screen edge. At least 20% of the
 * model (96px minimum) stays visible on each axis so it can be grabbed again.
 * `model` is the model rectangle in panel coordinates without offset; when
 * it is unknown only the panel is placed.
 */
export function placePet(
  target: ScreenPoint,
  panel: { width: number; height: number },
  model: ScreenBounds | null,
  display: ScreenBounds,
): PetPlacement {
  const axis = (
    wanted: number, size: number, modelStart: number | undefined, modelSize: number,
    start: number, length: number,
  ): [number, number] => {
    const window = electronCoordinate(size > length
      ? start + (length - size) / 2
      : Math.max(start, Math.min(start + length - size, wanted)));
    let offset = wanted - window;
    if (modelStart !== undefined) {
      const visibleStart = Math.max(window, start);
      const visibleEnd = Math.min(window + size, start + length);
      const keep = Math.min(
        modelSize,
        visibleEnd - visibleStart,
        Math.max(MIN_VISIBLE_PIXELS, modelSize * MIN_VISIBLE_FRACTION),
      );
      const at = window + modelStart + offset;
      offset += Math.max(visibleStart + keep - modelSize, Math.min(visibleEnd - keep, at)) - at;
    }
    return [window, offset];
  };
  const [x, offsetX] = axis(target.x, panel.width, model?.x, model?.width ?? 0, display.x, display.width);
  const [y, offsetY] = axis(target.y, panel.height, model?.y, model?.height ?? 0, display.y, display.height);
  return { window: { x, y }, offset: { x: offsetX, y: offsetY } };
}

/** The pet panel at scale 1; every other size is this times the window scale. */
export const PET_WINDOW_BASE = { width: 480, height: 720 } as const;

/**
 * Panel size for a window scale. Always derived from the scale itself: sizing
 * each pinch step from the previous (rounded) size let slow pinches round the
 * change away, so the window stopped shrinking while the scale kept falling.
 */
export function petWindowSize(windowScale: number): { width: number; height: number } {
  return {
    width: Math.max(1, Math.round(PET_WINDOW_BASE.width * windowScale)),
    height: Math.max(1, Math.round(PET_WINDOW_BASE.height * windowScale)),
  };
}

/** Resize the panel so the anchor keeps its relative spot inside it. */
export function resizePetBounds(
  current: ScreenBounds,
  size: { width: number; height: number },
  anchor: ScreenPoint = {
    x: current.x + current.width / 2,
    y: current.y + current.height / 2,
  },
): ScreenBounds {
  const anchorX = (anchor.x - current.x) / Math.max(1, current.width);
  const anchorY = (anchor.y - current.y) / Math.max(1, current.height);
  return {
    x: electronCoordinate(anchor.x - anchorX * size.width),
    y: electronCoordinate(anchor.y - anchorY * size.height),
    width: size.width,
    height: size.height,
  };
}

/**
 * The scale that matches how big a saved panel shows the pet. Panels saved by
 * the old step-by-step resizing may have drifted from their scale; keeping what
 * the person sees (and not the stale number) lets pinching work again at once.
 */
export function petScaleShownBy(
  size: { width: number; height: number },
  modelScale: number,
  windowScale: number,
): number {
  if (!(size.width > 0) || !(size.height > 0) || !(windowScale > 0)) return modelScale;
  const expected = petWindowSize(windowScale);
  // A panel squeezed into a sliver (dragged by its edge) says nothing about scale.
  if (size.width < expected.width * 0.3 || size.height < expected.height * 0.3) return modelScale;
  if (Math.abs(expected.width - size.width) <= 2 && Math.abs(expected.height - size.height) <= 2) {
    return modelScale;
  }
  const shown = Math.min(size.width / PET_WINDOW_BASE.width, size.height / PET_WINDOW_BASE.height);
  return modelScale * shown / windowScale;
}

function electronCoordinate(value: number): number {
  const rounded = Math.round(value);
  const clamped = Math.max(
    ELECTRON_MIN_COORDINATE,
    Math.min(ELECTRON_MAX_COORDINATE, rounded),
  );
  return Object.is(clamped, -0) ? 0 : clamped;
}
