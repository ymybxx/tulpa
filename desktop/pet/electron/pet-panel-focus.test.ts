import { describe, expect, it, vi } from "vitest";

import { PetPanelFocus } from "./pet-panel-focus";

function fixture(initialFocus = false, initialActive = initialFocus) {
  let focused = initialFocus;
  let active = initialActive;
  const token = {};
  const window = {
    isDestroyed: vi.fn(() => false),
    isVisible: vi.fn(() => true),
    isFocused: vi.fn(() => focused),
    focus: vi.fn(() => { focused = true; }),
    blur: vi.fn(() => { focused = false; }),
    getNativeWindowHandle: vi.fn(() => Buffer.alloc(8)),
    webContents: { isDevToolsOpened: vi.fn(() => false) },
  };
  const application = {
    isActive: vi.fn(() => active),
    makeKeyWindow: vi.fn(() => { focused = true; return true; }),
    acquire: vi.fn(() => { active = true; return token; }),
    restore: vi.fn(() => { active = false; focused = false; return true; }),
    resignKeyWindow: vi.fn(() => { focused = false; return true; }),
  };
  const focus = new PetPanelFocus(window, application);
  return { window, application, token, focus, deactivate: () => {
    active = false;
    focused = false;
    focus.forgetFocus();
  } };
}

describe("pet panel input focus", () => {
  it("does not activate the app when its panel cannot take input in the current Space", () => {
    const { window, focus, application } = fixture();
    application.makeKeyWindow.mockReturnValue(false);
    focus.setHovered(true);
    focus.setHovered(false);
    expect(application.makeKeyWindow).toHaveBeenCalledWith(window.getNativeWindowHandle());
    expect(window.focus).not.toHaveBeenCalled();
    expect(application.acquire).not.toHaveBeenCalled();
    expect(application.restore).not.toHaveBeenCalled();
    expect(application.resignKeyWindow).not.toHaveBeenCalled();
  });

  it("releases panel input when native restoration rejects an old Space", () => {
    const { window, focus, application, token } = fixture();
    application.restore.mockImplementation(() => false);
    focus.setHovered(true);
    focus.setHovered(false);
    focus.setHovered(false);
    expect(application.restore).toHaveBeenCalledTimes(1);
    expect(application.restore).toHaveBeenCalledWith(token);
    expect(application.resignKeyWindow).toHaveBeenCalledTimes(1);
    expect(application.resignKeyWindow).toHaveBeenCalledWith(window.getNativeWindowHandle());
    expect(window.isFocused()).toBe(false);
  });

  it("acquires focus on the pet and releases it when leaving or locking", () => {
    const { window, focus, application, token } = fixture();
    focus.setHovered(false);
    expect(window.focus).not.toHaveBeenCalled();
    focus.setHovered(true);
    focus.setHovered(true);
    expect(application.makeKeyWindow).toHaveBeenCalledTimes(1);
    expect(window.focus).not.toHaveBeenCalled();
    focus.setHovered(false);
    focus.setHovered(false);
    expect(application.acquire).toHaveBeenCalledTimes(1);
    expect(application.restore).toHaveBeenCalledTimes(1);
    expect(application.restore).toHaveBeenCalledWith(token);
    expect(window.isFocused()).toBe(false);
  });

  it("reacquires focus after switching to another application", () => {
    const { window, focus, application, deactivate } = fixture();
    focus.setHovered(true);
    deactivate();
    focus.setHovered(true);
    expect(application.makeKeyWindow).toHaveBeenCalledTimes(2);
    expect(window.focus).not.toHaveBeenCalled();
    expect(window.isFocused()).toBe(true);
    focus.setHovered(false);
    expect(window.isFocused()).toBe(false);
  });

  it("activates the application even when its panel already has key focus", () => {
    const { window, focus, application, token } = fixture(true, false);
    focus.setHovered(true);
    expect(window.focus).not.toHaveBeenCalled();
    expect(application.acquire).toHaveBeenCalledTimes(1);
    focus.setHovered(false);
    expect(application.restore).toHaveBeenCalledTimes(1);
    expect(application.restore).toHaveBeenCalledWith(token);
  });

  it("keeps a previously active application active after temporary window focus", () => {
    const { window, focus, application } = fixture(false, true);
    focus.setHovered(true);
    focus.setHovered(false);
    expect(application.acquire).not.toHaveBeenCalled();
    expect(application.restore).not.toHaveBeenCalled();
    expect(application.resignKeyWindow).toHaveBeenCalledTimes(1);
    expect(window.blur).not.toHaveBeenCalled();
  });

  it("releases a stale key panel without ordering it out while activation settles", () => {
    const { window, focus, application, token } = fixture(true, false);
    application.restore.mockImplementation(() => true);
    focus.setHovered(true);
    focus.setHovered(false);
    expect(application.restore).toHaveBeenCalledWith(token);
    expect(application.resignKeyWindow).toHaveBeenCalledWith(window.getNativeWindowHandle());
    expect(window.isFocused()).toBe(false);
    expect(window.blur).not.toHaveBeenCalled();
  });

  it("keeps native menus active when the pointer leaves the pet for a menu item", () => {
    const { focus, application, token } = fixture();
    focus.setHovered(true);
    focus.setMenuOpen(true);
    focus.setHovered(false);
    expect(application.restore).not.toHaveBeenCalled();
    focus.setMenuOpen(false);
    expect(application.restore).toHaveBeenCalledWith(token);
  });

  it("does not reactivate an old app when a menu action opened another window", () => {
    const { focus, application } = fixture();
    focus.setHovered(true);
    focus.setMenuOpen(true);
    focus.setHovered(false);
    focus.forgetFocus();
    focus.setMenuOpen(false);
    expect(application.restore).not.toHaveBeenCalled();
  });

  it("does not restore a stale application after the user switches elsewhere", () => {
    const { focus, application, deactivate } = fixture();
    focus.setHovered(true);
    deactivate();
    focus.setHovered(false);
    expect(application.restore).not.toHaveBeenCalled();
  });

  it("does not release focus that was already owned before hovering", () => {
    const { window, focus } = fixture(true);
    focus.setHovered(true);
    focus.setHovered(false);
    expect(window.focus).not.toHaveBeenCalled();
    expect(window.blur).not.toHaveBeenCalled();
  });

  it("forgets an old hover lease when focus moves elsewhere", () => {
    const { window, focus } = fixture();
    focus.setHovered(true);
    window.blur();
    focus.forgetFocus();
    window.focus();
    window.blur.mockClear();
    focus.setHovered(false);
    expect(window.blur).not.toHaveBeenCalled();
  });

  it.each(["hidden", "destroyed", "devtools"] as const)("does not focus or blur a %s window", (state) => {
    const { window, focus } = fixture();
    if (state === "hidden") window.isVisible.mockReturnValue(false);
    if (state === "destroyed") window.isDestroyed.mockReturnValue(true);
    if (state === "devtools") window.webContents.isDevToolsOpened.mockReturnValue(true);
    focus.setHovered(true);
    focus.setHovered(false);
    expect(window.focus).not.toHaveBeenCalled();
    expect(window.blur).not.toHaveBeenCalled();
  });

  it("clears a previous lease when the window hides", () => {
    const { window, focus } = fixture();
    focus.setHovered(true);
    window.isVisible.mockReturnValue(false);
    focus.setHovered(false);
    window.isVisible.mockReturnValue(true);
    focus.setHovered(false);
    expect(window.blur).not.toHaveBeenCalled();
  });
});
