import type { ModelRenderBundle } from "../shared/contracts";
import { loadCubismCore } from "./cubism-core";
import { characterSnapshot } from "./preview-snapshot";

/**
 * A 形象 card thumbnail: the model in its default pose, full body, framed like
 * every other card. Drawn off screen on a throwaway renderer, so it works for
 * looks that never went through import or onboarding (e.g. copied in).
 */
export async function renderLookThumbnail(bundle: ModelRenderBundle): Promise<ArrayBuffer | null> {
  await loadCubismCore();
  const { Live2DRenderer } = await import("./live2d-renderer");
  const viewport = document.createElement("div");
  // Off screen but laid out: the renderer fits the body to the viewport's size.
  viewport.style.cssText = "position:fixed;left:-10000px;top:0;width:480px;height:720px;pointer-events:none;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;height:100%;display:block;";
  viewport.append(canvas);
  document.body.append(viewport);
  const renderer = new Live2DRenderer(canvas, viewport, { maxTextureDimension: 2048, preserveDrawingBuffer: true });
  try {
    await renderer.loadModel(bundle);
    // The default pose is drawn synchronously; no frame wait (rAF stops while the window is hidden).
    renderer.setEditingPreview(true);
    const blob = await characterSnapshot(canvas);
    return blob ? await blob.arrayBuffer() : null;
  } finally {
    renderer.destroy();
    viewport.remove();
  }
}
