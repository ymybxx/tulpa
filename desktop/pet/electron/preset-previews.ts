import path from "node:path";

/**
 * Card thumbnails for built-in 形象, rendered ahead of time from the real
 * Live2D renderer and shipped with the app so first-run users never wait.
 * Keyed by the model3 file name of the bundled sample.
 */
const PRESET_PREVIEWS: Record<string, string> = {
  "Mao.model3.json": "mao-preview.png",
};

export function presetPreviewFile(modelUrl: string, appPath: string, packaged: boolean): string | null {
  const fileName = decodeURIComponent(modelUrl.split(/[?#]/u)[0]!.split("/").pop() ?? "");
  const preview = PRESET_PREVIEWS[fileName];
  if (!preview) return null;
  // Vite copies public-runtime/ into dist-renderer/ for packaged builds.
  return path.join(appPath, packaged ? "dist-renderer" : "public-runtime", "presets", preview);
}
