import { EventEmitter } from "node:events";
import type { BrowserWindow, Rectangle } from "electron";

import type { PetPanelPlacement } from "../src/shared/contracts";

type StageWindow = Pick<BrowserWindow,
  "isDestroyed" | "isVisible" | "getBounds" | "setBounds" | "on" | "removeListener"
> & { webContents: Pick<BrowserWindow["webContents"], "send" | "isDestroyed"> };

/**
 * TA's panel, drawn inside a window that covers the whole display and never
 * moves or resizes while TA is dragged or scaled. Moving or resizing a
 * window shows its previous frame in the wrong place until the page draws
 * again, so a panel that was a window of its own shook while being scaled.
 * Here the panel is a box in the page: its size and TA's picture change in
 * the same frame.
 *
 * To the rest of the app this behaves like the old panel window: bounds in
 * screen points, `move` / `resize` events, and `show` / `hide` from the
 * stage window.
 */
export class PetStage {
  private panel: Rectangle;
  private readonly events = new EventEmitter();

  constructor(
    private readonly window: StageWindow,
    panel: Rectangle,
    /** The display area a panel centred at this point is drawn on. */
    private readonly displayAt: (point: { x: number; y: number }) => Rectangle,
    /** The model's current offset inside the panel. */
    private readonly offset: () => { x: number; y: number },
  ) {
    this.panel = rounded(panel);
    this.fitStage();
  }

  getBounds(): Rectangle {
    return { ...this.panel };
  }

  /** The stage window's own bounds (the whole display). */
  getStageBounds(): Rectangle {
    return this.window.getBounds();
  }

  /**
   * `fromPage`: the page already placed the panel itself (scaling), so it is
   * not sent back; only the rest of the app follows.
   */
  setBounds(bounds: Rectangle, fromPage = false): void {
    const next = rounded(bounds);
    const moved = next.x !== this.panel.x || next.y !== this.panel.y;
    const resized = next.width !== this.panel.width || next.height !== this.panel.height;
    if (!moved && !resized) return;
    this.panel = next;
    if (!fromPage) {
      this.fitStage();
      this.publish();
    }
    if (moved) this.events.emit("move");
    if (resized) this.events.emit("resize");
  }

  setPosition(x: number, y: number): void {
    this.setBounds({ ...this.panel, x, y });
  }

  /** Sends the panel's place to the page (after a load, too). */
  publish(): void {
    if (this.window.isDestroyed() || this.window.webContents.isDestroyed()) return;
    this.window.webContents.send("pet:panel", this.placement());
  }

  placement(): PetPanelPlacement {
    const stage = this.window.getBounds();
    return {
      x: this.panel.x - stage.x, y: this.panel.y - stage.y, width: this.panel.width, height: this.panel.height,
      offset: { ...this.offset() },
    };
  }

  /** Re-covers the display, e.g. after display changes. */
  refit(reposition?: (display: Rectangle) => void): void {
    if (this.window.isDestroyed()) return;
    const center = { x: this.panel.x + this.panel.width / 2, y: this.panel.y + this.panel.height / 2 };
    // A removed/resized display can strand the panel even after the native
    // stage moves. Reuse the model-aware placement rule before publishing.
    reposition?.(this.displayAt(center));
    if (this.fitStage()) this.publish();
  }

  isDestroyed(): boolean {
    return this.window.isDestroyed();
  }

  isVisible(): boolean {
    return this.window.isVisible();
  }

  on(event: "move" | "resize" | "show" | "hide", listener: () => void): this {
    if (event === "move" || event === "resize") this.events.on(event, listener);
    else this.window.on(event as "show", listener);
    return this;
  }

  removeListener(event: "move" | "resize" | "show" | "hide", listener: () => void): this {
    if (event === "move" || event === "resize") this.events.removeListener(event, listener);
    else this.window.removeListener(event as "show", listener);
    return this;
  }

  /**
   * The stage covers the display the panel's centre is on. It moves only when
   * the panel crosses to another display; returns whether it did.
   */
  private fitStage(): boolean {
    if (this.window.isDestroyed()) return false;
    const center = { x: this.panel.x + this.panel.width / 2, y: this.panel.y + this.panel.height / 2 };
    const area = this.displayAt(center);
    const current = this.window.getBounds();
    if (current.x === area.x && current.y === area.y && current.width === area.width && current.height === area.height) return false;
    this.window.setBounds(area, false);
    return true;
  }
}

function rounded(bounds: Rectangle): Rectangle {
  return { x: Math.round(bounds.x), y: Math.round(bounds.y), width: Math.max(1, Math.round(bounds.width)), height: Math.max(1, Math.round(bounds.height)) };
}
