import { describe, expect, it } from "vitest";

import type { DesktopMainWindowState } from "../shared/contracts";
import { applyMainView, type MainViewElements } from "./view";

const CHAT_URL = "http://127.0.0.1:49152";

describe("applyMainView", () => {
  it("shows only the selected settings view", () => {
    const elements = fakeElements();

    applyMainView(elements, { view: "settings", chatUrl: null });

    expect(elements.settings.hidden).toBe(false);
    expect(elements.chat.hidden).toBe(true);
    expect(elements.models.hidden).toBe(true);
    expect(elements.settingsButton.selected).toBe(true);
    expect(elements.chatButton.selected).toBe(false);
  });

  it("shows only the model view and selects its navigation button", () => {
    const elements = fakeElements();

    applyMainView(elements, { view: "models", chatUrl: null });

    expect(elements.models.hidden).toBe(false);
    expect(elements.chat.hidden).toBe(true);
    expect(elements.settings.hidden).toBe(true);
    expect(elements.modelsButton.selected).toBe(true);
  });

  it("shows chat loading until a URL is available", () => {
    const elements = fakeElements();

    applyMainView(elements, { view: "chat", chatUrl: null });

    expect(elements.chat.hidden).toBe(false);
    expect(elements.settings.hidden).toBe(true);
    expect(elements.loading.hidden).toBe(false);
    expect(elements.frame.srcAssignments).toEqual([]);
  });

  it("loads a ready chat URL only once", () => {
    const elements = fakeElements();
    const ready: DesktopMainWindowState = { view: "chat", chatUrl: CHAT_URL };

    applyMainView(elements, ready);
    applyMainView(elements, ready);

    expect(elements.loading.hidden).toBe(true);
    expect(elements.frame.srcAssignments).toEqual([CHAT_URL]);
  });

  it("keeps an opened document when URL updates are deferred while hidden or in another view", () => {
    const elements = fakeElements();
    const ready: DesktopMainWindowState = { view: "chat", chatUrl: CHAT_URL };
    applyMainView(elements, ready);

    for (const view of ["chat", "settings", "models", "plugins"] as const) {
      applyMainView(elements, { view, chatUrl: null });
      applyMainView(elements, ready);
    }

    expect(elements.frame.getAttribute("src")).toBe(CHAT_URL);
    expect(elements.frame.srcAssignments).toEqual([CHAT_URL]);
  });

  it("navigates again when the runtime provides a different chat URL", () => {
    const elements = fakeElements();
    const nextUrl = "http://127.0.0.1:49153";
    applyMainView(elements, { view: "chat", chatUrl: CHAT_URL });
    applyMainView(elements, { view: "chat", chatUrl: nextUrl });
    expect(elements.frame.srcAssignments).toEqual([CHAT_URL, nextUrl]);
  });
});

function fakeElements(): MainViewElements & {
  chatButton: FakeButton;
  settingsButton: FakeButton;
  modelsButton: FakeButton;
  frame: FakeFrame;
} {
  return {
    chat: { hidden: false },
    settings: { hidden: false },
    models: { hidden: false },
    loading: { hidden: true },
    chatButton: new FakeButton(),
    settingsButton: new FakeButton(),
    modelsButton: new FakeButton(),
    frame: new FakeFrame(),
  };
}

class FakeButton {
  selected = false;
  classList = {
    toggle: (_name: string, selected: boolean) => {
      this.selected = selected;
    },
  };

  setAttribute(name: string, value: string): void {
    if (name === "aria-selected") this.selected = value === "true";
  }
}

class FakeFrame {
  private source: string | null = null;
  readonly srcAssignments: string[] = [];

  getAttribute(name: string): string | null {
    return name === "src" ? this.source : null;
  }

  setAttribute(name: string, value: string): void {
    if (name !== "src") return;
    this.source = value;
    this.srcAssignments.push(value);
  }
}
