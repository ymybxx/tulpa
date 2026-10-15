import { describe, expect, it, vi } from "vitest";
import { PetPointerMonitor } from "./pet-pointer-monitor";

function fixture() {
  let movement: () => void = () => undefined;
  let point = { x: -500, y: 40 };
  const window = {
    isDestroyed: vi.fn(() => false),
    isVisible: vi.fn(() => true),
    getBounds: vi.fn(() => ({ x: -200, y: 30, width: 300, height: 400 })),
    webContents: { isDestroyed: vi.fn(() => false), send: vi.fn() },
  };
  const source = {
    startMouseMonitor: vi.fn((callback: () => void) => { movement = callback; return true; }),
    stopMouseMonitor: vi.fn(),
  };
  const monitor = new PetPointerMonitor(window, source, () => point);
  return { window, source, monitor, move: (x: number, y: number) => { point = { x, y }; movement(); } };
}

describe("native pet pointer recovery", () => {
  it("restores renderer hit testing from other-app movement using current window coordinates", () => {
    const { window, monitor, move } = fixture();
    monitor.start();
    window.webContents.send.mockClear();
    move(-120, 70);
    expect(window.webContents.send).toHaveBeenLastCalledWith("pet:command", {
      type: "pet-pointer-position", x: 80, y: 40,
    });
    window.getBounds.mockReturnValue({ x: -100, y: 50, width: 300, height: 400 });
    move(-20, 90);
    expect(window.webContents.send).toHaveBeenLastCalledWith("pet:command", {
      type: "pet-pointer-position", x: 80, y: 40,
    });
  });

  it("sends one leave position and avoids renderer work for movement elsewhere", () => {
    const { window, monitor, move } = fixture();
    monitor.start();
    window.webContents.send.mockClear();
    move(-400, 50);
    move(-350, 60);
    expect(window.webContents.send).not.toHaveBeenCalled();
    move(-150, 80);
    move(100, 80);
    move(400, 80);
    expect(window.webContents.send).toHaveBeenCalledTimes(2);
    expect(window.webContents.send).toHaveBeenLastCalledWith("pet:command", {
      type: "pet-pointer-position", x: 300, y: 50,
    });
  });

  it("stops on hide/close and ignores queued callbacks until restarted", () => {
    const { window, source, monitor, move } = fixture();
    monitor.start();
    monitor.start();
    expect(source.startMouseMonitor).toHaveBeenCalledTimes(1);
    monitor.stop();
    monitor.stop();
    window.webContents.send.mockClear();
    move(-150, 80);
    expect(window.webContents.send).not.toHaveBeenCalled();
    expect(source.stopMouseMonitor).toHaveBeenCalledTimes(1);
    monitor.start();
    expect(window.webContents.send).toHaveBeenCalledTimes(1);
    window.isDestroyed.mockReturnValue(true);
    move(-140, 80);
    expect(window.webContents.send).toHaveBeenCalledTimes(1);
  });

  it("refreshes a stationary cursor after loading or a spurious renderer mouseleave", () => {
    const { window, monitor, move } = fixture();
    monitor.start();
    move(-120, 70);
    window.webContents.send.mockClear();
    monitor.refresh();
    expect(window.webContents.send).toHaveBeenCalledWith("pet:command", {
      type: "pet-pointer-position", x: 80, y: 40,
    });
    window.isVisible.mockReturnValue(false);
    monitor.refresh();
    move(-130, 75);
    expect(window.webContents.send).toHaveBeenCalledTimes(1);
  });

  it("can still verify renderer mouseleave if the OS monitor could not start", () => {
    const { window, source, monitor, move } = fixture();
    source.startMouseMonitor.mockReturnValue(false);
    monitor.start();
    window.webContents.send.mockClear();
    move(-120, 70);
    expect(window.webContents.send).not.toHaveBeenCalled();
    monitor.refresh();
    expect(window.webContents.send).toHaveBeenCalledTimes(1);
  });
});
