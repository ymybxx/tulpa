const CORE_URL = "pet-assets://runtime/live2dcubismcore.min.js";

/**
 * Loads the Live2D core if the page has none yet. During onboarding it is
 * downloaded after main.html's own script tag already failed.
 */
export function loadCubismCore(): Promise<void> {
  if (window.Live2DCubismCore) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${CORE_URL}?t=${Date.now()}`;
    script.onload = () => window.Live2DCubismCore ? resolve() : reject(new Error("Live2D 核心加载失败"));
    script.onerror = () => reject(new Error("Live2D 核心加载失败"));
    document.head.append(script);
  });
}
