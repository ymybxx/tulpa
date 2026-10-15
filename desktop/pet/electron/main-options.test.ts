import { describe, expect, it } from "vitest";

import {
  configureMainProcess,
  shouldThrottleRenderer,
} from "./main-options";

describe("configureMainProcess", () => {
  it("uses Chromium's mock keychain on macOS", () => {
    const switches: string[] = [];

    configureMainProcess(
      { appendSwitch: (value) => switches.push(value) },
      "darwin",
    );

    expect(switches).toEqual(["use-mock-keychain"]);
  });

  it("leaves other platforms' credential backends unchanged", () => {
    const switches: string[] = [];

    configureMainProcess(
      { appendSwitch: (value) => switches.push(value) },
      "win32",
    );

    expect(switches).toEqual([]);
  });
});

describe("shouldThrottleRenderer", () => {
  it("keeps the desktop pet live while throttling the hidden main window", () => {
    expect(shouldThrottleRenderer("pet")).toBe(false);
    expect(shouldThrottleRenderer("main")).toBe(true);
  });
});
