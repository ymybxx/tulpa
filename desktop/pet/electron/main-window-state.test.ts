import { describe, expect, it } from "vitest";

import type { DesktopRuntimeStatus } from "../src/shared/contracts";
import { resolveMainWindowState } from "./main-window-state";

const startingStatus: DesktopRuntimeStatus = {
  phase: "starting-api",
  detail: "正在启动",
  apiUrl: null,
  voiceEnabled: true,
};
const stoppedStatus: DesktopRuntimeStatus = {
  phase: "stopped",
  detail: "尚未启动",
  apiUrl: null,
  voiceEnabled: true,
};
const failedStatus: DesktopRuntimeStatus = {
  phase: "failed",
  detail: "启动失败",
  apiUrl: null,
  voiceEnabled: true,
};
const readyStatus: DesktopRuntimeStatus = {
  phase: "ready",
  detail: "运行正常",
  apiUrl: "http://127.0.0.1:49152",
  voiceEnabled: true,
};
const readyWithoutVoiceStatus: DesktopRuntimeStatus = {
  ...readyStatus,
  detail: "运行正常，语音已关闭",
  voiceEnabled: false,
};

describe("resolveMainWindowState", () => {
  it("keeps the requested chat view while a configured runtime starts", () => {
    expect(resolveMainWindowState("chat", true, startingStatus)).toEqual({
      view: "chat",
      chatUrl: null,
    });
  });

  it("sends unconfigured users to onboarding instead of chat", () => {
    expect(resolveMainWindowState("chat", false, stoppedStatus).view).toBe("onboarding");
  });

  it("still opens settings or models on request before setup", () => {
    expect(resolveMainWindowState("settings", false, stoppedStatus).view).toBe("settings");
    expect(resolveMainWindowState("models", false, stoppedStatus).view).toBe("models");
  });

  it("leaves onboarding once configuration is complete", () => {
    expect(resolveMainWindowState("onboarding", true, readyStatus).view).toBe("chat");
  });

  it("forces settings when the runtime fails", () => {
    expect(resolveMainWindowState("chat", true, failedStatus).view).toBe("settings");
  });

  it("shows models without requiring a configured Python runtime", () => {
    expect(resolveMainWindowState("models", false, failedStatus)).toEqual({
      view: "models",
      chatUrl: null,
    });
  });

  it("exposes the chat URL once the API is ready", () => {
    expect(resolveMainWindowState("chat", true, readyStatus)).toEqual({
      view: "chat",
      chatUrl: "http://127.0.0.1:49152/?nativeAgentPlayback=1",
    });
    expect(resolveMainWindowState("settings", true, startingStatus).chatUrl).toBeNull();
  });

  it("keeps chat available while the authenticated API waits for audio or retries it", () => {
    for (const detail of ["正在准备本地语音模型…", "LiveKit 本地音频未收到麦克风数据，正在重试"]) {
      expect(resolveMainWindowState("chat", true, {
        ...readyStatus,
        phase: "starting-companion",
        detail,
      })).toEqual({
        view: "chat",
        chatUrl: "http://127.0.0.1:49152/?nativeAgentPlayback=1",
      });
    }
  });

  it.each(["starting-api", "restarting", "stopped", "failed"] as const)(
    "does not expose a stale API URL while %s",
    (phase) => {
      expect(resolveMainWindowState("chat", true, { ...readyStatus, phase }).chatUrl).toBeNull();
    },
  );

  it("does not expose a chat URL when audio status has no API URL", () => {
    expect(resolveMainWindowState("chat", true, {
      ...readyStatus,
      phase: "starting-companion",
      apiUrl: null,
    }).chatUrl).toBeNull();
  });

  it("marks only a voice-enabled embedded chat as native Agent playback", () => {
    expect(resolveMainWindowState("chat", true, readyStatus).chatUrl).toBe(
      "http://127.0.0.1:49152/?nativeAgentPlayback=1",
    );
    expect(resolveMainWindowState("chat", true, readyWithoutVoiceStatus).chatUrl).toBe(
      "http://127.0.0.1:49152",
    );
  });

  it("honors an explicit settings request while ready", () => {
    expect(resolveMainWindowState("settings", true, readyStatus)).toEqual({
      view: "settings",
      chatUrl: "http://127.0.0.1:49152/?nativeAgentPlayback=1",
    });
  });

  it("allows chat again after runtime recovery", () => {
    expect(resolveMainWindowState("chat", true, failedStatus).view).toBe("settings");
    expect(resolveMainWindowState("chat", true, readyStatus).view).toBe("chat");
  });
});
