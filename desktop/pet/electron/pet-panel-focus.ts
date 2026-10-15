import type { BrowserWindow, WebContents } from "electron";
import type { PetApplicationActivation } from "./pet-application-activation";

type PanelWindow = Pick<BrowserWindow,
  "isDestroyed" | "isVisible" | "isFocused" | "getNativeWindowHandle"
> & { webContents: Pick<WebContents, "isDevToolsOpened"> };

/** Temporarily lends a macOS panel input focus while the pet is interactive. */
export class PetPanelFocus {
  private acquiredFocus = false;
  private previousApplication: object | null = null;
  private menuOpen = false;
  private hovered = false;

  constructor(
    private readonly window: PanelWindow,
    private readonly application: PetApplicationActivation,
  ) {}

  setHovered(hovered: boolean): void {
    this.hovered = hovered;
    if (this.menuOpen) return;
    const window = this.window;
    if (window.isDestroyed() || !window.isVisible() || window.webContents.isDevToolsOpened()) {
      this.forgetFocus();
      return;
    }
    if (hovered) {
      if (!window.isFocused()) {
        // BrowserWindow.focus() also orders the window front on macOS. A pet
        // already visible in this Space only needs input, not window ordering.
        if (!this.application.makeKeyWindow(window.getNativeWindowHandle())) return;
        this.acquiredFocus = true;
      }
      // AppKit sends magnify gestures to the active application. A key panel
      // and document.hasFocus() alone do not make that application active.
      // Keep application activation for gesture delivery, after making this
      // panel key without reordering windows. Native restoration is Space-bound.
      if (!this.application.isActive()) {
        this.previousApplication = this.application.acquire();
      }
    } else {
      const previous = this.previousApplication;
      const releaseFocus = this.acquiredFocus || previous !== null;
      this.previousApplication = null;
      this.acquiredFocus = false;
      if (previous && window.isFocused()) this.application.restore(previous);
      if (releaseFocus && window.isFocused()) {
        this.application.resignKeyWindow(window.getNativeWindowHandle());
      }
    }
  }

  setMenuOpen(open: boolean): void {
    this.menuOpen = open;
    if (!open) this.setHovered(this.hovered);
  }

  forgetFocus(): void {
    this.acquiredFocus = false;
    this.previousApplication = null;
  }
}
