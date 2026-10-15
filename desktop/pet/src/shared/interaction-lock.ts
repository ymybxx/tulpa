interface Point {
  x: number;
  y: number;
}

interface Bounds {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

interface Size {
  width: number;
  height: number;
}

interface ModelBounds extends Size {
  x: number;
  y: number;
}

export function pointInsideInteractionControl(point: Point, bounds: Bounds): boolean {
  return point.x >= bounds.left && point.x <= bounds.right &&
    point.y >= bounds.top && point.y <= bounds.bottom;
}

/**
 * Where the 恢复点击 control goes: beside TA (right, then left), else above its
 * head, and over TA only when the window has no room anywhere else.
 */
export function unlockControlPosition(
  model: ModelBounds,
  control: Size,
  viewport: Size,
  gap = 8,
): { left: number; top: number } {
  const maximumLeft = Math.max(gap, viewport.width - control.width - gap);
  const maximumTop = Math.max(gap, viewport.height - control.height - gap);
  const clampTop = (top: number) => Math.round(Math.max(gap, Math.min(maximumTop, top)));
  const clampLeft = (left: number) => Math.round(Math.max(gap, Math.min(maximumLeft, left)));
  const besideTop = clampTop(model.y + gap);
  const right = model.x + model.width + gap;
  if (right <= maximumLeft) return { left: Math.round(right), top: besideTop };
  const left = model.x - control.width - gap;
  if (left >= gap) return { left: Math.round(left), top: besideTop };
  const above = model.y - control.height - gap;
  if (above >= gap) {
    return { left: clampLeft(model.x + (model.width - control.width) / 2), top: Math.round(above) };
  }
  return { left: clampLeft(model.x + model.width - control.width - gap), top: besideTop };
}
