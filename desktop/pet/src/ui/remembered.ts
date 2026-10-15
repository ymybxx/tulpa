/**
 * Where the user last was on a page (a settings section, a character tab), so
 * reopening the page returns there. Kept in memory for the session and in
 * localStorage as a convenience; losing it only means starting at the top.
 */
const memory = new Map<string, string>();

export function remember(key: string, value: string): void {
  memory.set(key, value);
  try {
    localStorage.setItem(`fish-memory:${key}`, value);
  } catch {
    // Storage can be unavailable; the in-memory copy still covers this session.
  }
}

export function recall<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  const value = recallValue(key);
  return allowed.includes(value as T) ? value as T : fallback;
}

/** A free-form remembered value, such as an ID; null when there is none. */
export function recallValue(key: string): string | null {
  const value = memory.get(key);
  if (value !== undefined) return value;
  try {
    return localStorage.getItem(`fish-memory:${key}`);
  } catch {
    return null;
  }
}

/** Opens settings on a specific section, e.g. from a "去设置" link elsewhere. */
export function openSettings(section: "ai" | "jev" | "companion" | "permissions" | "general" | "advanced"): void {
  remember("settings-section", section);
  window.desktopPet.showMainView("settings");
}

/** Opens 角色 on one of its tabs, e.g. 声音. */
export function openCharacter(tab: "profile" | "voice" | "memory"): void {
  remember("character-tab", tab);
  window.desktopPet.showMainView("character");
}

/**
 * Opens 形象 where the user left it: the list of looks, or the editor still
 * open on the look being edited.
 */
export function openLook(): void {
  window.desktopPet.showMainView(recall("look-page", ["looks", "models"] as const, "looks"));
}
