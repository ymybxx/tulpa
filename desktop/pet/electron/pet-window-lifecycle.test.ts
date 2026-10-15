import { EventEmitter } from "node:events";
import { describe, expect, it, vi } from "vitest";

import { PetWindowLifecycle } from "./pet-window-lifecycle";

class Window extends EventEmitter {
  visible = true;
  destroyed = false;
  readonly model = { id: "existing-pet", scale: 0.44 };
  isVisible() { return this.visible; }
  isDestroyed() { return this.destroyed; }
  showInactive() { this.visible = true; this.emit("show"); }
  hide() { this.visible = false; this.emit("hide"); }
  destroy() { this.destroyed = true; this.visible = false; this.emit("closed"); }
  close() {
    let prevented = false;
    this.emit("close", { preventDefault: () => { prevented = true; } });
    if (!prevented) this.destroy();
  }
}

function fixture() {
  let current: Window | null = new Window();
  let quitting = false;
  const changed = vi.fn();
  const create = vi.fn(async () => {
    current = new Window();
    lifecycle.bind(current);
    current.showInactive();
  });
  const lifecycle = new PetWindowLifecycle({
    current: () => current,
    create,
    quitting: () => quitting,
    visibilityChanged: changed,
  });
  lifecycle.bind(current);
  return {
    lifecycle, create, changed,
    current: () => current!,
    clear: () => { current = null; },
    quit: () => { quitting = true; },
  };
}

describe("desktop pet window lifecycle", () => {
  it("hides on close and restores the same loaded pet from the tray or Dock", async () => {
    const f = fixture();
    const window = f.current();
    const model = window.model;
    window.close();
    expect(window.isDestroyed()).toBe(false);
    expect(window.isVisible()).toBe(false);
    expect(f.changed).toHaveBeenCalledTimes(1);
    await f.lifecycle.show();
    expect(f.current()).toBe(window);
    expect(f.current().model).toBe(model);
    expect(window.isVisible()).toBe(true);
    expect(f.create).not.toHaveBeenCalled();
    expect(f.changed).toHaveBeenCalledTimes(2);
  });

  it("allows a real quit and never recreates a window while quitting", async () => {
    const f = fixture();
    f.quit();
    f.current().close();
    expect(f.current().isDestroyed()).toBe(true);
    await f.lifecycle.show();
    await f.lifecycle.toggle();
    expect(f.create).not.toHaveBeenCalled();
    expect(f.changed).not.toHaveBeenCalled();
  });

  it.each(["missing", "destroyed"])("recreates a %s pet when shown", async (state) => {
    const f = fixture();
    const previous = f.current();
    if (state === "missing") f.clear();
    else previous.destroy();
    await f.lifecycle.toggle();
    expect(f.create).toHaveBeenCalledTimes(1);
    expect(f.current()).not.toBe(previous);
    expect(f.current().isVisible()).toBe(true);
  });

  it("coalesces simultaneous restore requests into one creation", async () => {
    const f = fixture();
    f.clear();
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    f.create.mockImplementationOnce(() => gate);
    const first = f.lifecycle.show();
    const second = f.lifecycle.toggle();
    expect(second).toBe(first);
    await Promise.resolve();
    expect(f.create).toHaveBeenCalledTimes(1);
    release();
    await Promise.all([first, second]);
  });

  it("allows restoration to retry after a failed creation", async () => {
    const f = fixture();
    f.clear();
    f.create.mockRejectedValueOnce(new Error("creation failed"));
    await expect(f.lifecycle.show()).rejects.toThrow("creation failed");
    await f.lifecycle.show();
    expect(f.create).toHaveBeenCalledTimes(2);
    expect(f.current().isVisible()).toBe(true);
  });

  it("does not begin a queued creation after quitting starts", async () => {
    const f = fixture();
    f.clear();
    const show = f.lifecycle.show();
    f.quit();
    await show;
    expect(f.create).not.toHaveBeenCalled();
  });
});
