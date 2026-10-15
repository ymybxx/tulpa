import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { assertPrivateFile } from "../test-support/private-file";

import { afterEach, describe, expect, it } from "vitest";

import { platformAudioBackend } from "./audio-backend";
import { AppConfigStore } from "./app-config";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { recursive: true, force: true }),
    ),
  );
});

describe("AppConfigStore", () => {
  it("shares one internal key across concurrent first-start callers", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const values = await Promise.all(Array.from({ length: 8 }, () => store.runtimeSecrets()));
    expect(new Set(values.map(value => value.internalApiKey)).size).toBe(1);
    expect((await new AppConfigStore(root).runtimeSecrets()).internalApiKey).toBe(values[0]!.internalApiKey);
  });
  it("defaults old profiles to seven days of logs and persists custom retention", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-log-retention-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const initial = await store.load();
    expect(initial.llmLogRetentionDays).toBe(7);
    const { llmLogRetentionDays: _days, ...legacy } = initial;
    await writeFile(path.join(root, "config.json"), JSON.stringify(legacy));
    expect((await store.load()).llmLogRetentionDays).toBe(7);
    await store.save({ ...initial, llmLogRetentionDays: 30 });
    expect((await new AppConfigStore(root).load()).llmLogRetentionDays).toBe(30);
    for (const invalid of [0, -1, 366, 1.5, Number.NaN, "7", null]) {
      await expect(store.save({ ...initial, llmLogRetentionDays: invalid as number })).rejects.toThrow("日志保留天数");
      expect((await store.load()).llmLogRetentionDays).toBe(30);
    }
  });
  it("uses a detected texture ceiling for legacy settings and persists manual overrides", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-texture-quality-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const initial = await store.load();
    expect([2048, 4096, 8192]).toContain(initial.rendering.maxTextureDimension);

    const { rendering: _rendering, ...legacy } = initial;
    await writeFile(path.join(root, "config.json"), JSON.stringify(legacy));
    expect((await store.load()).rendering.maxTextureDimension)
      .toBe(initial.rendering.maxTextureDimension);

    await store.save({ ...initial, rendering: { maxTextureDimension: 2048 } });
    expect((await new AppConfigStore(root).load()).rendering.maxTextureDimension).toBe(2048);
    await store.save({ ...initial, rendering: { maxTextureDimension: 8192 } });
    expect((await new AppConfigStore(root).load()).rendering.maxTextureDimension).toBe(8192);
  });

  it("defaults facial waiting off and persists it independently of experimental mode", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-facial-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const initial = await store.load();
    expect(initial.speechFacialWaitForReady).toBe(false);
    for (const enabled of [true, false]) {
      await store.save({ ...initial, speechFacialWaitForReady: enabled });
      const reloaded = await new AppConfigStore(root).load();
      expect(reloaded.speechFacialWaitForReady).toBe(enabled);
      expect(reloaded.proactiveCompanionEnabled).toBe(false);
      expect(reloaded.audio).toEqual(initial.audio);
    }
    for (const invalid of [undefined, "true"]) {
      await writeFile(path.join(root, "config.json"), JSON.stringify({
        ...initial, speechFacialWaitForReady: invalid,
      }));
      expect((await store.load()).speechFacialWaitForReady).toBe(false);
    }
  });
  it("runs 主动陪伴 without Jev and keeps the optional Jev credential out of public config", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-jev-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const initial = await store.load();
    expect(initial.proactiveCompanionEnabled).toBe(false);
    await store.save({ ...initial, proactiveCompanionEnabled: true, secrets: {} });
    expect((await store.load()).proactiveCompanionEnabled).toBe(true);
    await store.save({ ...initial, proactiveCompanionEnabled: true, secrets: { jevApiKey: "jev-test-canary" } });
    const reloaded = await store.load();
    expect(reloaded.proactiveCompanionEnabled).toBe(true);
    expect(reloaded.experimentalCompanion).toEqual({ model: "jev-1.13.0", wakeSeconds: 5, thoughtsEnabled: true, thoughtSeconds: 2, speechFacialEnabled: true });
    expect(reloaded.secrets.jevApiKey).toBe(true);
    expect(JSON.stringify(reloaded)).not.toContain("jev-test-canary");
    expect(await readFile(path.join(root, "config.json"), "utf8")).not.toContain("jev-test-canary");
    await store.save({ ...reloaded, proactiveCompanionEnabled: false, secrets: {} });
    expect((await store.load()).secrets.jevApiKey).toBe(true);
    await store.save({ ...reloaded, proactiveCompanionEnabled: true, secrets: { jevApiKey: null } });
    const removed = await store.load();
    expect(removed.proactiveCompanionEnabled).toBe(true);
    expect(removed.secrets.jevApiKey).toBe(false);
  });
  it("turns the old Jev experimental switch into 主动陪伴", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-jev-migration-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const initial = await store.load();
    await store.save({ ...initial, secrets: { jevApiKey: "jev-test-canary" } });
    const { secrets: _secrets, ...persisted } = initial;
    for (const [experimental, proactive, expected] of [[true, false, true], [false, true, false]] as const) {
      await writeFile(path.join(root, "config.json"), JSON.stringify({
        ...persisted, proactiveCompanionEnabled: proactive,
        experimentalCompanion: { enabled: experimental, model: "jev-1.13.0", wakeSeconds: 5 },
      }));
      expect((await store.load()).proactiveCompanionEnabled).toBe(expected);
    }
  });
  it("keeps 主动陪伴 on while no Jev key is stored", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-jev-missing-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const { secrets: _secrets, ...persisted } = await store.load();
    await writeFile(path.join(root, "config.json"), JSON.stringify({ ...persisted, proactiveCompanionEnabled: true }));
    expect((await store.load()).proactiveCompanionEnabled).toBe(true);
  });
  it("persists independent thought settings and validates their interval", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-thought-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const initial = await store.load();
    expect(initial.experimentalCompanion?.thoughtsEnabled).toBe(true);
    expect(initial.experimentalCompanion?.thoughtSeconds).toBe(2);
    const experimentalCompanion = { ...initial.experimentalCompanion!, thoughtsEnabled: false, thoughtSeconds: 7 };
    await store.save({ ...initial, experimentalCompanion });
    expect((await store.load()).experimentalCompanion).toEqual(experimentalCompanion);
    await expect(store.save({ ...initial, experimentalCompanion: { ...experimentalCompanion, thoughtSeconds: 0 } })).rejects.toThrow("后台思考间隔");
    expect((await store.load()).experimentalCompanion).toEqual(experimentalCompanion);
    // Picking replies stays unset (it follows Jev) until the user flips it.
    expect(experimentalCompanion.chatPickEnabled).toBeUndefined();
    const chosen = { ...experimentalCompanion, chatPickEnabled: false, speechFacialEnabled: false };
    await store.save({ ...initial, experimentalCompanion: chosen });
    expect((await store.load()).experimentalCompanion).toEqual(chosen);
  });
  it("keeps 外部消息 off by default and validates its port", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-gateway-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const initial = await store.load();
    expect(initial.gateway).toEqual({ enabled: false, port: 17821 });
    await store.save({ ...initial, gateway: { enabled: true, port: 18000 } });
    expect((await store.load()).gateway).toEqual({ enabled: true, port: 18000 });
    await expect(store.save({ ...initial, gateway: { enabled: true, port: 80 } })).rejects.toThrow("端口");
    expect((await store.load()).gateway).toEqual({ enabled: true, port: 18000 });
  });

  it("keeps companionship opt-in and persists it without enabling perception", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const initial = await store.load();
    expect(initial.proactiveCompanionEnabled).toBe(false);
    await store.save({ ...initial, proactiveCompanionEnabled: true });
    const reloaded = await store.load();
    expect(reloaded.proactiveCompanionEnabled).toBe(true);
    expect(reloaded.perception).toEqual(initial.perception);
  });
  it("keeps owner recognition off for new, missing and invalid legacy settings", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const current = await store.load();
    expect(current.audio.ownerRecognitionEnabled).toBe(false);
    const { ownerRecognitionEnabled: _enabled, ...legacyAudio } = current.audio;
    await writeFile(path.join(root, "config.json"), JSON.stringify({ ...current, audio: legacyAudio }));
    expect((await store.load()).audio.ownerRecognitionEnabled).toBe(false);
    await writeFile(path.join(root, "config.json"), JSON.stringify({
      ...current, audio: { ...legacyAudio, ownerRecognitionEnabled: "true" },
    }));
    expect((await store.load()).audio.ownerRecognitionEnabled).toBe(false);
  });

  it("validates the model/profile before persisting an enabled setting", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const unvalidated = new AppConfigStore(root);
    const current = await unvalidated.load();
    const input = { ...current, audio: { ...current.audio, ownerRecognitionEnabled: true } };
    await expect(unvalidated.save(input)).rejects.toThrow("请先准备模型");
    const rejected = new AppConfigStore(root, async (value) => {
      expect(value.audio.ownerRecognitionEnabled).toBe(true);
      throw new Error("音频后端不同，请重新录入");
    });
    await expect(rejected.save(input)).rejects.toThrow("重新录入");
    expect((await rejected.load()).audio.ownerRecognitionEnabled).toBe(false);
    const valid = new AppConfigStore(root, async () => {});
    expect((await valid.save(input)).audio.ownerRecognitionEnabled).toBe(true);
    expect((await valid.load()).audio.ownerRecognitionEnabled).toBe(true);
  });

  it("picks the voice backend from the platform and ignores stored choices", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const current = await store.load();
    expect(current.advanced.audioBackend).toBe(platformAudioBackend());
    for (const stored of ["sounddevice", "livekit", "unknown", undefined]) {
      await writeFile(path.join(root, "config.json"), JSON.stringify({
        ...current, audio: { ...current.audio, wakeWord: "旧配置唤醒词" },
        advanced: { ...current.advanced, audioBackend: stored },
      }));
      const loaded = await store.load();
      expect(loaded.advanced.audioBackend).toBe(platformAudioBackend());
      expect(loaded.audio.wakeWord).toBe("旧配置唤醒词");
    }
  });

  it.each(["paraformer", "zipformer"])(
    "migrates the legacy %s ASR selection to SenseVoice",
    async (legacyModel) => {
      const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
      temporaryDirectories.push(root);
      const store = new AppConfigStore(root);
      const current = await store.load();
      await writeFile(
        path.join(root, "config.json"),
        JSON.stringify({
          ...current,
          advanced: { ...current.advanced, asrModel: legacyModel },
        }),
      );

      expect((await store.load()).advanced.asrModel).toBe("sensevoice");
    },
  );

  it("follows the system default devices wherever LiveKit is the backend", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const loaded = await store.load();
    const saved = await store.save({ ...loaded, audio: { ...loaded.audio, inputDevice: "USB Mic", outputDevice: "USB Speaker" } });
    const expected = platformAudioBackend() === "livekit" ? null : "USB Mic";
    expect(saved.audio.inputDevice).toBe(expected);
  });

  it("enables voice draft preflight by default and falls back to enabled for non-boolean values", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);

    expect((await store.load()).advanced.voiceDraftEnabled).toBe(true);
    await writeFile(
      path.join(root, "config.json"),
      JSON.stringify({
        ...await store.load(),
        advanced: {
          ...(await store.load()).advanced,
          voiceDraftEnabled: "true",
        },
      }),
    );

    expect((await store.load()).advanced.voiceDraftEnabled).toBe(true);
  });

  it("always listens with turn detection and voice drafts, even over an old opt-out", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const loaded = await store.load();
    await writeFile(
      path.join(root, "config.json"),
      JSON.stringify({
        ...loaded,
        advanced: { ...loaded.advanced, voiceDraftEnabled: false, smartTurnEnabled: false },
      }),
    );
    const reloaded = await store.load();
    expect(reloaded.advanced.voiceDraftEnabled).toBe(true);
    expect(reloaded.advanced.smartTurnEnabled).toBe(true);
  });

  it("keeps the Agent API key out of config and exposes only its presence", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const loaded = await store.load();

    const saved = await store.save({
      ...loaded,
      agent: { ...loaded.agent, url: "http://127.0.0.1:8642/p/coder" },
      secrets: { agentApiKey: "private-agent-key" },
    });

    expect(saved.secrets.agentApiKey).toBe(true);
    expect(await readFile(path.join(root, "config.json"), "utf8")).not.toContain("private-agent-key");
    expect(await readFile(path.join(root, "config.json"), "utf8")).not.toContain("agentApiKey");
    expect(await readFile(path.join(root, "secrets.json"), "utf8")).toContain("private-agent-key");
    await assertPrivateFile(path.join(root, "secrets.json"));
    expect(saved.advanced.voiceDraftEnabled).toBe(true);
  });

  it("keeps the Agent API key when a later save omits the secret field", async () => {
    const store = await createStore();
    const loaded = await store.load();
    await store.save({
      ...loaded,
      secrets: { agentApiKey: "private-agent-key" },
    });

    const { secrets: _secretFlags, ...configuration } = await store.load();
    const saved = await store.save(configuration);

    expect(saved.secrets.agentApiKey).toBe(true);
    expect((await store.runtimeSecrets()).agentApiKey).toBe("private-agent-key");
  });

  it("explicitly clears the saved Agent API key", async () => {
    const store = await createStore();
    const loaded = await store.load();
    await store.save({
      ...loaded,
      secrets: { agentApiKey: "private-agent-key" },
    });
    const { secrets: _flags, ...configuration } = await store.load();

    const saved = await store.save({
      ...configuration,
      secrets: { agentApiKey: null },
    });

    expect(saved.secrets.agentApiKey).toBe(false);
    expect((await store.runtimeSecrets()).agentApiKey).toBeUndefined();
  });

  it("fills in Agent defaults for older desktop configuration files", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const { agent: _agent, secrets: _secrets, ...legacy } = await store.load();
    await writeFile(path.join(root, "config.json"), JSON.stringify(legacy));

    expect((await store.load()).agent).toEqual({
      enabled: false,
      url: "http://127.0.0.1:8642",
      runConcurrency: 4,
      runPollIntervalSeconds: 2,
      httpTimeoutSeconds: 15,
      sseReadTimeoutSeconds: 45,
      replyContextTtlSeconds: 30,
    });
  });

  it("keeps environment perception disabled for configs without computer history", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const { perception: _perception, secrets: _secrets, ...legacy } = await store.load();
    await writeFile(path.join(root, "config.json"), JSON.stringify(legacy));

    expect((await store.load()).perception).toEqual({
      operationPerceptionEnabled: false,
      screenPerceptionEnabled: false,
      audioPerceptionEnabled: false,
      excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
    });
  });

  it("never lets TA see itself on screen, even over an old opt-out", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const initial = await store.load();
    expect(initial.perception.excludeFishMemoryFromScreenCapture).toBe(true);
    await store.save({
      ...initial,
      perception: { ...initial.perception, excludeFishMemoryFromScreenCapture: false },
    });
    const reloaded = await new AppConfigStore(root).load();
    expect(reloaded.perception.excludeFishMemoryFromScreenCapture).toBe(true);
    expect(reloaded.perception.screenPerceptionEnabled).toBe(false);
  });

  it("migrates an old audiovisual opt-in into independent screen and audio inputs", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const current = await store.load();
    await writeFile(path.join(root, "config.json"), JSON.stringify({
      ...current, perception: {
        operationPerceptionEnabled: false, screenPerceptionEnabled: false,
        audiovisualPerceptionEnabled: true,
      },
    }));
    expect((await store.load()).perception).toEqual({
      operationPerceptionEnabled: false, screenPerceptionEnabled: true, audioPerceptionEnabled: true,
      excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
    });
    await writeFile(path.join(root, "config.json"), JSON.stringify({
      ...current, perception: {
        operationPerceptionEnabled: false, screenPerceptionEnabled: false,
        audiovisualPerceptionEnabled: true, audioPerceptionEnabled: false,
      },
    }));
    expect((await store.load()).perception).toEqual({
      operationPerceptionEnabled: false, screenPerceptionEnabled: false, audioPerceptionEnabled: false,
      excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
    });
  });

  it("migrates only an explicit legacy computer history enable setting", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const current = await store.load();

    await writeFile(path.join(root, "config.json"), JSON.stringify({
      ...current,
      perception: {
        workspaceEnabled: true,
        screenEnabled: true,
        accessibilityEnabled: true,
        cameraEnabled: true,
        computerHistoryEnabled: true,
        visionModel: "vision-model",
        enhancedVisionEnabled: true,
        proactivityProfile: "lively",
      },
    }));
    expect((await store.load()).perception).toEqual({
      operationPerceptionEnabled: true,
      screenPerceptionEnabled: true,
      audioPerceptionEnabled: false,
      excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
    });

    await writeFile(path.join(root, "config.json"), JSON.stringify({
      ...current,
      perception: {
        workspaceEnabled: true,
        screenEnabled: true,
        accessibilityEnabled: true,
        cameraEnabled: true,
        computerHistoryEnabled: false,
        visionModel: "vision-model",
        enhancedVisionEnabled: true,
        proactivityProfile: "lively",
      },
    }));
    expect((await store.load()).perception).toEqual({
      operationPerceptionEnabled: false,
      screenPerceptionEnabled: false,
      audioPerceptionEnabled: false,
      excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
    });
  });

  it("persists only the current perception switches after migration", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const current = await store.load();
    await writeFile(path.join(root, "config.json"), JSON.stringify({
      ...current,
      perception: {
        workspaceEnabled: true,
        screenEnabled: true,
        accessibilityEnabled: true,
        cameraEnabled: true,
        computerHistoryEnabled: true,
        visionModel: "vision-model",
        enhancedVisionEnabled: true,
        proactivityProfile: "lively",
      },
    }));

    const migrated = await store.load();
    const { secrets: _flags, ...configuration } = migrated;
    await store.save(configuration);

    const persisted = JSON.parse(
      await readFile(path.join(root, "config.json"), "utf8"),
    ) as { perception: Record<string, unknown> };
    expect(persisted.perception).toEqual({
      operationPerceptionEnabled: true,
      screenPerceptionEnabled: true,
      audioPerceptionEnabled: false,
      excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
    });
  });

  it("prefers the new environment switch over stale legacy fields", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const current = await store.load();
    await writeFile(path.join(root, "config.json"), JSON.stringify({
      ...current,
      perception: {
        environmentPerceptionEnabled: false,
        computerHistoryEnabled: true,
      },
    }));

    expect((await store.load()).perception).toEqual({
      operationPerceptionEnabled: false,
      screenPerceptionEnabled: false,
      audioPerceptionEnabled: false,
      excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
    });
  });

  it("migrates an enabled environment switch without opting into audiovisual generation", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const current = await store.load();
    await writeFile(path.join(root, "config.json"), JSON.stringify({
      ...current,
      perception: { environmentPerceptionEnabled: true },
    }));
    expect((await store.load()).perception).toEqual({
      operationPerceptionEnabled: true,
      screenPerceptionEnabled: true,
      audioPerceptionEnabled: false,
      excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
    });
  });

  it.each(Array.from({ length: 8 }, (_value, mask) => mask))(
    "preserves independent perception combination %i over stale migration fields", async (mask) => {
      const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
      temporaryDirectories.push(root);
      const store = new AppConfigStore(root);
      const current = await store.load();
      const perception = {
        operationPerceptionEnabled: Boolean(mask & 1),
        screenPerceptionEnabled: Boolean(mask & 2),
        audioPerceptionEnabled: Boolean(mask & 4),
        excludeFishMemoryFromScreenCapture: true, innerLifeEnabled: true,
      };
      await writeFile(path.join(root, "config.json"), JSON.stringify({
        ...current,
        perception: { environmentPerceptionEnabled: true, ...perception },
      }));
      const { secrets: _flags, ...loaded } = await store.load();
      expect(loaded.perception).toEqual(perception);
      await store.save(loaded);
      const persisted = JSON.parse(await readFile(path.join(root, "config.json"), "utf8"));
      expect(persisted.perception).toEqual(perception);
    },
  );

  it("drops obsolete Agent target and credential-map fields", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const current = await store.load();
    await writeFile(
      path.join(root, "config.json"),
      JSON.stringify({
        ...current,
        agent: {
          ...current.agent,
          defaultTargetId: "c95e8b6e-bc3b-4211-aeaa-8c39f49d90b8",
        },
      }),
    );
    await writeFile(
      path.join(root, "secrets.json"),
      JSON.stringify({
        version: 2,
        values: {
          modelApiKeys: { "text-default": "llm-secret" },
          agentCredentials: { "hermes:local": "obsolete-agent-key" },
        },
      }),
    );

    const loaded = await store.load();
    const runtimeSecrets = await store.runtimeSecrets();

    expect(loaded.agent).not.toHaveProperty("defaultTargetId");
    expect(loaded).not.toHaveProperty("agentCredentialRefs");
    expect(runtimeSecrets).not.toHaveProperty("agentCredentials");
    const { secrets: _flags, ...configuration } = loaded;
    await store.save(configuration);
    expect(await readFile(path.join(root, "config.json"), "utf8")).not.toContain(
      "defaultTargetId",
    );
    expect(await readFile(path.join(root, "secrets.json"), "utf8")).not.toContain(
      "obsolete-agent-key",
    );
  });

  it("requires the LLM secret before onboarding can complete", async () => {
    const store = await createStore();
    const value = await store.load();

    await expect(store.save({ ...value, configured: true })).rejects.toThrow(
      /默认文字模型.*API Key/,
    );
  });

  it("does not create a secrets file before values are saved", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);

    const loaded = await new AppConfigStore(root).load();

    expect(loaded.configured).toBe(false);
    await expect(readFile(path.join(root, "secrets.json"), "utf8")).rejects.toMatchObject({
      code: "ENOENT",
    });
  });

  it("drops unknown fields left by older desktop configurations", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const { secrets: _secretFlags, ...configuration } = await store.load();
    await writeFile(
      path.join(root, "config.json"),
      JSON.stringify({
        ...configuration,
        acceptedLive2dTerms: false,
        secrets: {},
      }),
    );

    const loaded = await store.load();

    expect(loaded).not.toHaveProperty("acceptedLive2dTerms");
  });

  it("persists ordinary values separately from local secrets", async () => {
    const store = await createStore();
    const value = await store.load();
    const saved = await store.save({
      ...value,
      configured: true,
      models: { ...value.models, text: [{ ...value.models.text[0]!, baseUrl: "https://example.com/v1", model: "test-model" }] },
      secrets: { modelApiKeys: { "text-default": "secret-value" } },
    });

    expect(saved.configured).toBe(true);
    expect(saved.secrets.modelApiKeys["text-default"]).toBe(true);
    expect((await store.runtimeSecrets()).modelApiKeys?.["text-default"]).toBe("secret-value");
  });

  it("stores secrets in a user-only local file", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const value = await store.load();

    await store.save({
      ...value,
      configured: true,
      secrets: { modelApiKeys: { "text-default": "local-secret" } },
    });

    const secretsPath = path.join(root, "secrets.json");
    expect(JSON.parse(await readFile(secretsPath, "utf8"))).toEqual({
      version: 2,
      values: { modelApiKeys: { "text-default": "local-secret" } },
    });
    await assertPrivateFile(secretsPath);
  });

  it("keeps the fal key only in the user-only secrets file", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const value = await store.load();

    const saved = await store.save({
      ...value,
      secrets: { falApiKey: "fal-local-secret" },
    });

    expect(saved.secrets.falApiKey).toBe(true);
    expect((await store.runtimeSecrets()).falApiKey).toBe("fal-local-secret");
    expect(await readFile(path.join(root, "config.json"), "utf8")).not.toContain(
      "fal-local-secret",
    );
    await assertPrivateFile(path.join(root, "secrets.json"));
  });

  it("stores the OpenAI-compatible image key separately from ordinary configuration", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const store = new AppConfigStore(root);
    const value = await store.load();

    const saved = await store.save({
      ...value,
      imageEditing: {
        openaiCompatibleBaseUrl: "https://image.example.test/v1",
      },
      secrets: { openaiCompatibleImageApiKey: "sk-image-secret" },
    });

    expect(saved.imageEditing).toEqual({
      openaiCompatibleBaseUrl: "https://image.example.test/v1",
    });
    expect(saved.secrets.openaiCompatibleImageApiKey).toBe(true);
    expect((await store.runtimeSecrets()).openaiCompatibleImageApiKey).toBe("sk-image-secret");
    expect(await readFile(path.join(root, "config.json"), "utf8")).not.toContain("sk-image-secret");
  });

  it("drops legacy user-configurable image model and quality fields", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    await writeFile(path.join(root, "config.json"), JSON.stringify({
      ...await new AppConfigStore(root).load(),
      imageEditing: {
        openaiCompatibleBaseUrl: "http://127.0.0.1:8080/v1",
        openaiCompatibleModel: "gpt-image-2",
        openaiCompatibleQuality: "medium",
      },
    }));

    const loaded = await new AppConfigStore(root).load();

    expect(loaded.imageEditing).toEqual({
      openaiCompatibleBaseUrl: "http://127.0.0.1:8080/v1",
    });
  });

  it("requires the dedicated image key when an OpenAI-compatible image endpoint is enabled", async () => {
    const store = await createStore();
    const value = await store.load();

    await expect(store.save({
      ...value,
      imageEditing: {
        ...value.imageEditing,
        openaiCompatibleBaseUrl: "https://images.example.com/v1",
      },
    })).rejects.toThrow(/图片 API Key/);
  });

  it("creates one stable internal API key", async () => {
    const store = await createStore();

    const first = await store.runtimeSecrets();
    const second = await store.runtimeSecrets();

    expect(first.internalApiKey).toMatch(/^fm_/);
    expect(second.internalApiKey).toBe(first.internalApiKey);
  });

  it("returns to onboarding for legacy encrypted secrets without Keychain access", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
    temporaryDirectories.push(root);
    const initialStore = new AppConfigStore(root);
    const { secrets: _secretFlags, ...configuration } = await initialStore.load();
    await writeFile(
      path.join(root, "config.json"),
      JSON.stringify({ ...configuration, configured: true }),
    );
    await writeFile(
      path.join(root, "secrets.json"),
      JSON.stringify({
        version: 1,
        payload: Buffer.from("encrypted:legacy-secrets").toString("base64"),
      }),
    );
    const store = new AppConfigStore(root);

    const loaded = await store.load();

    expect(loaded.configured).toBe(false);
    expect(loaded.secrets.modelApiKeys["text-default"]).toBe(false);
  });
});

async function createStore(): Promise<AppConfigStore> {
  const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
  temporaryDirectories.push(root);
  return new AppConfigStore(root);
}

it("does not spawn owner-profile validation again for unrelated saves", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "fish-memory-config-test-"));
  temporaryDirectories.push(root);
  let validations = 0;
  const store = new AppConfigStore(root, async () => { validations += 1; });
  const initial = await store.load();
  const enabled = await store.save({ ...initial, audio: { ...initial.audio, ownerRecognitionEnabled: true } });
  expect(validations).toBe(1);
  const unchanged = await store.save({ ...enabled, launchAtLogin: !enabled.launchAtLogin });
  expect(validations).toBe(1);
  await store.save({ ...unchanged, advanced: { ...unchanged.advanced, audioBackend: "sounddevice" } });
  expect(validations).toBe(1);
});
