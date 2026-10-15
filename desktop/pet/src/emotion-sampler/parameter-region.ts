/** Screenshots of one parameter at several values, enlarged where they differ. */

export interface Region { x: number; y: number; width: number; height: number }

/**
 * The part of the picture that differs between the parameter's values, padded
 * and widened to the canvas's shape. Found on a small copy, so a few stray
 * pixels (anti-aliasing) are ignored. Nothing visible changing falls back to
 * the upper middle, where faces usually are.
 */
export function changedRegion(shots: readonly HTMLCanvasElement[], width: number, height: number): Region {
  const fallback = { x: width / 3, y: 0, width: width / 3, height: height / 3 };
  const scale = 4;
  const w = Math.max(1, Math.round(width / scale)), h = Math.max(1, Math.round(height / scale));
  const pixels = shots.map(shot => {
    const small = document.createElement("canvas"); small.width = w; small.height = h;
    const context = small.getContext("2d");
    if (!context) return null;
    context.drawImage(shot, 0, 0, w, h);
    return context.getImageData(0, 0, w, h).data;
  });
  if (pixels.length < 2 || pixels.some(data => !data)) return fallback;
  let left = w, top = h, right = -1, bottom = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const at = (y * w + x) * 4;
      const changed = pixels.some(data => pixels.some(other =>
        Math.abs(data![at]! - other![at]!) + Math.abs(data![at + 1]! - other![at + 1]!) + Math.abs(data![at + 2]! - other![at + 2]!) > 36));
      if (!changed) continue;
      left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y);
    }
  }
  if (right < 0) return fallback;
  return fitRegion({ x: left * scale, y: top * scale, width: (right - left + 1) * scale, height: (bottom - top + 1) * scale }, width, height);
}

/**
 * Pads a changed box and keeps it between tall 2:3 and wide 2:1, so a wide
 * change (both forearms) is not stretched into a tall crop of the whole body.
 * Never smaller than a fifth of the picture, never outside it.
 */
export function fitRegion(box: Region, width: number, height: number): Region {
  let w = Math.max(box.width * 1.3, width / 5), h = Math.max(box.height * 1.3, height / 5);
  if (w / h < 2 / 3) w = h * 2 / 3;
  if (w / h > 2) h = w / 2;
  w = Math.min(w, width); h = Math.min(h, height);
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  return { x: Math.max(0, Math.min(width - w, cx - w / 2)), y: Math.max(0, Math.min(height - h, cy - h / 2)), width: w, height: h };
}

/** The whole pose on the left and the changing part, enlarged as far as it fits, on the right. */
export function sideBySide(shot: HTMLCanvasElement, detail: Region): string {
  const scale = Math.min(shot.width * 1.5 / detail.width, shot.height / detail.height);
  const w = Math.round(detail.width * scale), h = Math.round(detail.height * scale);
  const panel = document.createElement("canvas");
  panel.width = shot.width + w; panel.height = shot.height;
  const context = panel.getContext("2d");
  if (!context) throw new Error("无法生成参数对照画面");
  context.fillStyle = "rgb(230, 235, 232)"; context.fillRect(0, 0, panel.width, panel.height);
  context.drawImage(shot, 0, 0);
  context.drawImage(shot, detail.x, detail.y, detail.width, detail.height, shot.width, Math.round((shot.height - h) / 2), w, h);
  // Mark the enlarged area on the whole pose so the two halves read together.
  context.strokeStyle = "rgba(220, 60, 60, 0.9)"; context.lineWidth = 4;
  context.strokeRect(detail.x, detail.y, detail.width, detail.height);
  return panel.toDataURL("image/jpeg", 0.85);
}
