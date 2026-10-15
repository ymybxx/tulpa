import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  DesktopConfigurationInput,
  DesktopConfigurationView,
} from "../src/shared/contracts";
import { experimentalConfiguration, runtimeApiConfiguration, runtimeConfigurationAction, webSearchOn } from "./runtime-config-impact";
import { supportsSystemAudio } from "./system-audio-support";

vi.mock("./system-audio-support", () => ({ supportsSystemAudio: vi.fn(() => true) }));
beforeEach(() => { vi.mocked(supportsSystemAudio).mockReturnValue(true); });

describe("runtimeConfigurationAction", () => {
  it("hot-applies log retention without restarting voice or the API", () => {
    const previous = configuration();
    const next = configuration({ llmLogRetentionDays: 30 });
    expect(runtimeApiConfiguration(previous, {}).llm_log_retention_days).toBe(7);
    expect(runtimeApiConfiguration(next, {}).llm_log_retention_days).toBe(30);
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    expect(runtimeConfigurationAction(previous, next, input(next), { previous: {}, next: {} })).toBe("configure-api");
  });
  it("hot-applies facial waiting without restarting audio or enabling experimental routing", () => {
    const previous = configuration();
    const next = configuration({ speechFacialWaitForReady: true });
    expect(runtimeApiConfiguration(previous, {}).speech_facial_wait_for_ready).toBe(false);
    expect(runtimeApiConfiguration(next, {}).speech_facial_wait_for_ready).toBe(true);
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    expect(runtimeConfigurationAction(next, previous, input(previous))).toBe("configure-api");
    expect(runtimeConfigurationAction(previous, next, input(next), {
      previous: {}, next: {},
    })).toBe("configure-api");
    expect(next.experimentalCompanion?.enabled ?? false).toBe(false);
    const explicitDefault = configuration({ speechFacialWaitForReady: false });
    expect(runtimeConfigurationAction(previous, explicitDefault, input(explicitDefault))).toBe("none");
  });
  it("hot switches experimental mode without changing ordinary routing settings", () => {
    const previous = configuration();
    const next = configuration({ experimentalCompanion: { enabled: true, model: "jev-1.13.0", wakeSeconds: 5 } });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    expect(runtimeConfigurationAction(next, previous, input(previous))).toBe("configure-api");
    expect(next.proactiveCompanionEnabled).toBe(previous.proactiveCompanionEnabled);
    expect(next.modelAssignments).toEqual(previous.modelAssignments);
  });
  it("gives each Jev judgement its own switch; picking replies follows Jev until chosen", () => {
    const proactiveOff = configuration({ proactiveCompanionEnabled: false });
    const withJev = experimentalConfiguration(proactiveOff, { jevApiKey: "jev-key" } as never);
    expect(withJev).toMatchObject({ enabled: false, chat_pick_enabled: true, speech_facial_enabled: true });
    expect(experimentalConfiguration(proactiveOff, {} as never).chat_pick_enabled).toBe(false);
    const chosen = configuration({ experimentalCompanion: { model: "jev-1.13.0", wakeSeconds: 5, chatPickEnabled: false, speechFacialEnabled: false } });
    expect(experimentalConfiguration(chosen, { jevApiKey: "jev-key" } as never))
      .toMatchObject({ chat_pick_enabled: false, speech_facial_enabled: false });
    expect(runtimeConfigurationAction(proactiveOff, chosen, input(chosen))).toBe("configure-api");
  });
  it("hot-applies thought settings without restarting the host", () => {
    const previous = configuration({ experimentalCompanion: { enabled: true, model: "jev-1.13.0", wakeSeconds: 5, thoughtsEnabled: true, thoughtSeconds: 2 } });
    const next = configuration({ experimentalCompanion: { ...previous.experimentalCompanion!, thoughtsEnabled: false, thoughtSeconds: 4 } });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    expect(next.modelAssignments).toEqual(previous.modelAssignments);
    expect(next.experimentalCompanion?.wakeSeconds).toBe(5);
  });
  it("hot-applies 内心 switched from the pet's panel", () => {
    const previous = configuration();
    const next = configuration({ perception: { ...previous.perception, innerLifeEnabled: false } });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
  });

  it("hot-applies 联网查资料 and a reasoning model that changes whether it can search", () => {
    const previous = configuration();
    const off = configuration({ webSearchEnabled: false });
    expect(runtimeConfigurationAction(previous, off, input(off))).toBe(
      webSearchOn(previous) ? "configure-api" : "none",
    );
    const ark = configuration({
      models: { ...previous.models, text: previous.models.text.map((item) => ({ ...item, baseUrl: "https://ark.cn-beijing.volces.com/api/v3" })) },
    });
    expect(webSearchOn(ark)).toBe(true);
    const arkOff = { ...ark, webSearchEnabled: false };
    expect(webSearchOn(arkOff)).toBe(false);
    expect(runtimeConfigurationAction(ark, arkOff, input(arkOff))).toBe("configure-api");
  });

  it("hot-applies Codex completion notifications independently of Task and voice", () => {
    const previous = configuration({ codexNotificationsEnabled: false });
    const next = configuration({ codexNotificationsEnabled: true });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    expect(runtimeConfigurationAction(next, previous, input(previous))).toBe("configure-api");
    expect(next.agent).toEqual(previous.agent);
  });
  it("hot-applies companionship without restarting capture or voice", () => {
    const previous = configuration({ proactiveCompanionEnabled: false });
    const next = configuration({ proactiveCompanionEnabled: true });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    expect(runtimeConfigurationAction(next, previous, input(previous))).toBe("configure-api");
    expect(next.perception).toEqual(previous.perception);
  });
  it("restarts only Companion when owner recognition is toggled", () => {
    const previous = configuration();
    const next = configuration({ audio: { ...previous.audio, ownerRecognitionEnabled: true } });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("restart-companion");
  });
  it("does not restart services for image, login or unchanged settings", () => {
    const previous = configuration();
    const next = configuration({
      launchAtLogin: true,
      imageEditing: { openaiCompatibleBaseUrl: "https://images.example.com/v1" },
    });
    expect(runtimeConfigurationAction(previous, next, input(next, {
      openaiCompatibleImageApiKey: "new-image-key",
    }))).toBe("none");
  });

  it("changes only the pet renderer when the texture ceiling changes", () => {
    const previous = configuration();
    const next = configuration({ rendering: { maxTextureDimension: 8192 } });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("none");
  });

  it("does not restart Python services for perception-only settings", () => {
    const previous = configuration();
    const next = configuration({
      perception: { operationPerceptionEnabled: true, screenPerceptionEnabled: true, audioPerceptionEnabled: true },
    });

    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("none");
  });

  it("starts or stops media recognition independently when microphone voice is off", () => {
    const previous = configuration({ voiceEnabled: false });
    const next = configuration({ voiceEnabled: false,
      perception: { ...previous.perception, audioPerceptionEnabled: true },
    });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("restart-companion");
    expect(runtimeConfigurationAction(next, previous, input(previous))).toBe("restart-companion");
  });

  it("only restarts the companion for microphone and voice processing settings", () => {
    const previous = configuration();
    const next = configuration({
      audio: { ...previous.audio, inputDevice: "microphone-2" },
    });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe(
      "restart-companion",
    );

    const livekit = configuration({
      advanced: { ...previous.advanced, audioBackend: "livekit" },
    });
    expect(runtimeConfigurationAction(previous, livekit, input(livekit))).toBe(
      "restart-companion",
    );
  });

  it("hot-applies selected model changes", () => {
    const previous = configuration();
    const next = configuration({
      models: { ...previous.models, text: [{ ...previous.models.text[0]!, model: "new-model" }] },
    });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe(
      "configure-api",
    );
  });

  it("hot-applies voice draft availability and restarts only Companion", () => {
    const previous = configuration();
    const next = configuration({
      advanced: { ...previous.advanced, voiceDraftEnabled: true },
    });

    expect(runtimeConfigurationAction(previous, next, input(next))).toBe(
      "configure-api-and-companion",
    );
  });

  it("keeps API and memory alive when voice availability changes", () => {
    const previous = configuration();
    const next = configuration({ voiceEnabled: false });

    expect(runtimeConfigurationAction(previous, next, input(next))).toBe(
      "restart-companion",
    );

  });

  it("hot-applies active Agent topology and credentials", () => {
    const previous = configuration();
    previous.agent.enabled = true;
    const next = configuration({
      agent: { ...previous.agent, url: "http://127.0.0.1:8642/p/coder" },
    });

    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    expect(runtimeConfigurationAction(previous, previous, input(previous, {
      agentApiKey: "new-secret",
    }))).toBe("configure-api");
    expect(runtimeConfigurationAction(previous, previous, input(previous, {
      agentApiKey: null,
    }))).toBe("configure-api");
  });

  it("toggles Agent admission without restarting the API, plugins or Companion", () => {
    const previous = configuration();
    const next = configuration({ agent: { ...previous.agent, enabled: true } });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    expect(runtimeConfigurationAction(next, previous, input(previous))).toBe("configure-api");
    const keys = { internalApiKey: "internal", agentApiKey: "agent" };
    expect(runtimeConfigurationAction(previous, next, input(next), {
      previous: keys, next: keys,
    })).toBe("configure-api");
  });

  it("starts only once after initial configuration", () => {
    const previous = configuration({ configured: false });
    const next = configuration();
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("start");
  });
});

function configuration(
  overrides: Partial<DesktopConfigurationView> = {},
): DesktopConfigurationView {
  return {
    version: 2,
    configured: true,
    voiceEnabled: true,
    launchAtLogin: false,
    models: { text: [{ id: "text-default", name: "Test", baseUrl: "https://example.com/v1", model: "model" }], voice: [{ id: "voice-default", name: "Voice", provider: "auto" }] },
    modelAssignments: { chat: "text-default", reasoning: "text-default", speech: "voice-default" },
    imageEditing: { openaiCompatibleBaseUrl: null },
    homeAssistant: { enabled: false, url: "http://127.0.0.1:8123" },
    perception: {
      operationPerceptionEnabled: false,
      screenPerceptionEnabled: false,
      audioPerceptionEnabled: false,
    },
    rendering: { maxTextureDimension: 4096 },
    audio: { wakeWord: "小鱼", inputDevice: null, outputDevice: null, ownerRecognitionEnabled: false },
    advanced: {
      audioBackend: "sounddevice",
      asrModel: "sensevoice",
      asrThreads: 2,
      vadThreshold: 0.25,
      vadMinSilenceMs: 350,
      vadMinSpeechMs: 80,
      smartTurnEnabled: true,
      smartTurnThreshold: 0.5,
      aecDelayMode: "auto",
      aecDelayMs: 50,
      diagnosticsEnabled: false,
      voiceDraftEnabled: false,
    },
    agent: {
      enabled: false,
      url: "http://127.0.0.1:8642",
      runConcurrency: 4,
      runPollIntervalSeconds: 2,
      httpTimeoutSeconds: 15,
      sseReadTimeoutSeconds: 45,
      replyContextTtlSeconds: 30,
    },
    secrets: {
      modelApiKeys: { "text-default": true, "voice-default": false },
      homeAssistantToken: false,
      openaiCompatibleImageApiKey: false,
      falApiKey: false,
      agentApiKey: false,
    },
    ...overrides,
  };
}

function input(
  view: DesktopConfigurationView,
  secrets: NonNullable<DesktopConfigurationInput["secrets"]> = {},
): DesktopConfigurationInput {
  const { secrets: _flags, ...configuration } = view;
  return { ...configuration, secrets };
}

describe("configuration dependency boundaries", () => {
  it("does not refresh services when the same key is entered again or model aliases are identical", () => {
    const previous = configuration();
    const selected = previous.models.text[0]!;
    const alias = { ...selected, id: "alias", name: "another label" };
    const next = configuration({ models: { ...previous.models, text: [selected, alias] },
      modelAssignments: { ...previous.modelAssignments, reasoning: "alias" } });
    const keys = { internalApiKey: "internal", modelApiKeys: { "text-default": "same", alias: "same" } };
    expect(runtimeConfigurationAction(previous, next, input(next, { modelApiKeys: { alias: "same" } }),
      { previous: keys, next: keys })).toBe("none");
    expect(runtimeConfigurationAction(previous, previous, input(previous, { modelApiKeys: { "text-default": "same" } }),
      { previous: keys, next: keys })).toBe("none");
    expect(runtimeConfigurationAction(previous, previous, input(previous), {
      previous: keys, next: { ...keys, modelApiKeys: { ...keys.modelApiKeys, "text-default": "changed" } },
    })).toBe("configure-api");
  });

  it("hot-applies speech provider/key changes without reloading microphone or memory", () => {
    const previous = configuration();
    const next = configuration({ models: { ...previous.models,
      voice: [{ ...previous.models.voice[0]!, provider: "local" }] } });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    expect(runtimeConfigurationAction(previous, previous, input(previous, {
      modelApiKeys: { "voice-default": "new-key" },
    }))).toBe("configure-api");
  });

  it("does not restart for inactive integration settings", () => {
    const previous = configuration();
    const next = configuration({
      homeAssistant: { enabled: false, url: "http://home.invalid:8123" },
      agent: { ...previous.agent, url: "http://agent.invalid", runConcurrency: 7 },
    });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
  });

  it("does not reload audio for parameters that are not in use", () => {
    const previous = configuration({ voiceEnabled: false });
    const next = configuration({ voiceEnabled: false,
      audio: { ...previous.audio, inputDevice: "other", wakeWord: "new word" },
      advanced: { ...previous.advanced, asrThreads: 4, vadThreshold: .7 },
    });
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("none");
    const automatic = configuration();
    const unusedDelay = configuration({ advanced: { ...automatic.advanced, aecDelayMs: 400 } });
    expect(runtimeConfigurationAction(automatic, unusedDelay, input(unusedDelay))).toBe("none");
    const fixed = configuration({ advanced: { ...automatic.advanced, aecDelayMode: "fixed", aecDelayMs: 400 } });
    expect(runtimeConfigurationAction(automatic, fixed, input(fixed))).toBe("restart-companion");
  });

  it("only refreshes media ASR when its thread count changes with voice off", () => {
    const previous = configuration({ voiceEnabled: false });
    previous.perception.audioPerceptionEnabled = true;
    const next = structuredClone(previous);
    next.advanced.asrThreads = 4;
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("restart-companion");
    next.advanced = previous.advanced;
    next.audio.outputDevice = "other";
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("none");
  });

  it("does not refresh unavailable media ASR on older systems", () => {
    vi.mocked(supportsSystemAudio).mockReturnValue(false);
    const previous = configuration({ voiceEnabled: false });
    previous.perception.audioPerceptionEnabled = true;
    const next = structuredClone(previous);
    next.advanced.asrThreads += 1;
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("none");
  });

  it("hot-applies Agent scheduling/HTTP timeouts, with only voice refreshed for reply TTL", () => {
    const previous = configuration();
    previous.agent.enabled = true;
    for (const field of ["runConcurrency", "runPollIntervalSeconds", "httpTimeoutSeconds", "sseReadTimeoutSeconds"] as const) {
      const next = structuredClone(previous);
      next.agent[field] += 1;
      expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
    }
    const next = structuredClone(previous);
    next.agent.replyContextTtlSeconds += 1;
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api-and-companion");
  });

  it("keeps memory running when voice preflight is changed while voice is off", () => {
    const previous = configuration({ voiceEnabled: false });
    const next = structuredClone(previous);
    next.advanced.voiceDraftEnabled = true;
    expect(runtimeConfigurationAction(previous, next, input(next))).toBe("configure-api");
  });

  it("does not rebuild an active Home integration for an identical token", () => {
    const previous = configuration();
    previous.homeAssistant.enabled = true;
    const keys = { internalApiKey: "internal", homeAssistantToken: "same" };
    expect(runtimeConfigurationAction(previous, previous, input(previous, { homeAssistantToken: "same" }),
      { previous: keys, next: keys })).toBe("none");
  });
});
