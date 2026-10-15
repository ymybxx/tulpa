const PREVIEW_TABS = new Set(["pools", "assets", "defaults", "appearance", "idle", "display", "facial", "edit"]);
const APPEARANCE_TABS = new Set(["appearance", "edit"]);

/** Heavy WebGL preview is retained only where it is directly useful. */
export function modelManagerTabKeepsPreview(tab: string): boolean {
  return PREVIEW_TABS.has(tab);
}

export function modelManagerTabUsesAppearance(tab: string): boolean {
  return APPEARANCE_TABS.has(tab);
}
