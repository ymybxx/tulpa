export type Scene = "day" | "dusk" | "night";

/** System dark mode always wins; otherwise the local hour picks the sky. */
export function sceneFor(hour: number, prefersDark: boolean): Scene {
  if (prefersDark) return "night";
  if (hour >= 17 && hour < 19) return "dusk";
  if (hour >= 19 || hour < 6) return "night";
  return "day";
}

/** "auto" follows the clock and system dark mode; otherwise the user picked a scene. */
export type ScenePreference = "auto" | Scene;

const PREFERENCE_KEY = "fish-memory.scene";
const PREFERENCE_EVENT = "fish-memory:scene-preference";

export function scenePreference(): ScenePreference {
  try {
    const value = localStorage.getItem(PREFERENCE_KEY);
    return value === "day" || value === "dusk" || value === "night" ? value : "auto";
  } catch {
    return "auto";
  }
}

export function setScenePreference(value: ScenePreference): void {
  try {
    if (value === "auto") localStorage.removeItem(PREFERENCE_KEY);
    else localStorage.setItem(PREFERENCE_KEY, value);
  } catch { /* a blocked store keeps the automatic scene */ }
  window.dispatchEvent(new Event(PREFERENCE_EVENT));
}

export function currentScene(): Scene {
  const preference = scenePreference();
  if (preference !== "auto") return preference;
  const prefersDark = typeof matchMedia === "function"
    && matchMedia("(prefers-color-scheme: dark)").matches;
  return sceneFor(new Date().getHours(), prefersDark);
}

/** Keeps `element[data-scene]` in step with the clock and the system appearance. */
export function followScene(element: HTMLElement): () => void {
  const apply = () => { element.dataset.scene = currentScene(); };
  apply();
  const media = typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : null;
  media?.addEventListener("change", apply);
  window.addEventListener(PREFERENCE_EVENT, apply);
  window.addEventListener("storage", apply);
  const timer = window.setInterval(apply, 5 * 60_000);
  return () => {
    media?.removeEventListener("change", apply);
    window.removeEventListener(PREFERENCE_EVENT, apply);
    window.removeEventListener("storage", apply);
    window.clearInterval(timer);
  };
}
