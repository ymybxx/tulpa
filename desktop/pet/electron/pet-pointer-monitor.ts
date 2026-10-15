import type { BrowserWindow, Point, WebContents } from "electron";
import type { PetMouseMonitor } from "./pet-application-activation";

type PointerWindow = Pick<BrowserWindow, "isDestroyed" | "isVisible" | "getBounds"> & {
  webContents: Pick<WebContents, "isDestroyed" | "send">;
};

/** Restores hit testing when fullscreen apps receive mouse movement instead. */
export class PetPointerMonitor {
  private running = false;
  private wasInside = false;

  constructor(
    private readonly window: PointerWindow,
    private readonly source: PetMouseMonitor,
    private readonly cursor: () => Point,
    private readonly onPointer?: (point: Point) => void,
  ) {}

  get active(): boolean { return this.running; }

  start(): void {
    if (this.running || this.window.isDestroyed()) return;
    this.running = this.source.startMouseMonitor(() => this.refresh(false));
    this.refresh();
  }

  stop(): void {
    if (!this.running) return;
    this.running = false;
    this.wasInside = false;
    this.source.stopMouseMonitor();
  }

  refresh(force = true): void {
    const window = this.window;
    if ((!this.running && !force) || window.isDestroyed() || !window.isVisible() || window.webContents.isDestroyed()) return;
    const point = this.cursor();
    this.onPointer?.(point);
    const bounds = window.getBounds();
    const x = point.x - bounds.x;
    const y = point.y - bounds.y;
    const inside = x >= 0 && y >= 0 && x < bounds.width && y < bounds.height;
    // Avoid renderer IPC for movement elsewhere on the desktop. One final
    // outside position clears hover; the actual model hit test stays in renderer.
    if (force || inside || this.wasInside) {
      window.webContents.send("pet:command", { type: "pet-pointer-position", x, y });
    }
    this.wasInside = inside;
  }
}
