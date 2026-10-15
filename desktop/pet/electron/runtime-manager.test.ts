import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import net from "node:net";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import type { ChildProcess } from "node:child_process";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { DesktopConfigurationView, DesktopRuntimeStatus } from "../src/shared/contracts";
import { resolveMainWindowState } from "./main-window-state";
import {
  buildRuntimeEnvironment,
  companionReadinessFromLog,
  companionSpeechActivityFromLog,
  companionStartupFailureFromLog,
  ensureAssistantBinding,
  liveKitAudioHelperExecutable,
  modelStatusDirectory,
  prepareRuntimeLog,
  RuntimeManager,
  sharedAsrSocketPath,
} from "./runtime-manager";
import type { CompanionSpeechActivityEvent } from "./runtime-manager";
import type { ProcessSnapshot, RuntimeProcessController } from "./runtime-process-registry";
import type { PerceptionManagerStatus } from "./perception-manager";
import { supportsSystemAudio } from "./system-audio-support";

vi.mock("./system-audio-support", () => ({ supportsSystemAudio: vi.fn(() => true) }));
beforeEach(() => { vi.mocked(supportsSystemAudio).mockReturnValue(true); });

const temporaryDirectories: string[] = [];

afterEach(async () => {
  delete process.env.FISH_MEMORY_TEST_LEAK;
  delete process.env.FISH_MEMORY_AGENT_URL;
  delete process.env.FISH_MEMORY_AGENT_API_KEY;
  delete process.env.FISH_MEMORY_AGENT_CREDENTIALS_FILE;
  delete process.env.FISH_MEMORY_AGENT_TASKS_ENABLED;
  delete process.env.FISH_MEMORY_AGENT_RUN_POLL_INTERVAL_SECONDS;
  delete process.env.FISH_MEMORY_AGENT_HTTP_TIMEOUT_SECONDS;
  delete process.env.FISH_MEMORY_AGENT_SSE_READ_TIMEOUT_SECONDS;
  delete process.env.FISH_MEMORY_COMPANION_AUDIO_BACKEND;
  delete process.env.FISH_MEMORY_LIVEKIT_AUDIO_HELPER;
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { recursive: true, force: true }),
    ),
  );
});

describe("buildRuntimeEnvironment", () => {
  it("isolates the Windows PostgreSQL binaries and registry in the app profile", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "tulpa-中文-profile-"));
    temporaryDirectories.push(userData);
    vi.stubGlobal("process", { ...process, platform: "win32",
      env: { ...process.env, TULPA_PG0_HOME: "another-application" } });
    try {
      const environment = await buildRuntimeEnvironment(
        configuration(), { internalApiKey: "fm_internal" }, userData, null,
      );
      expect(environment.TULPA_PG0_HOME).toBe(path.join(userData, "data", "pg0-runtime"));
      expect(environment.FISH_MEMORY_DATABASE_DATA_DIR).toBe(path.join(userData, "data", "postgres"));
    } finally {
      vi.unstubAllGlobals();
    }
  });
  it("passes the configured log retention to runtime startup", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-retention-runtime-test-"));
    temporaryDirectories.push(userData);
    const configured = { ...configuration(), llmLogRetentionDays: 30 };
    const environment = await buildRuntimeEnvironment(configured, { internalApiKey: "fm_internal" }, userData, null);
    expect(environment.FISH_MEMORY_LLM_LOG_RETENTION_DAYS).toBe("30");
  });
  it.each([false, true])("passes facial wait mode %s to API startup", async (enabled) => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-facial-runtime-test-"));
    temporaryDirectories.push(userData);
    const config = configuration();
    config.speechFacialWaitForReady = enabled;
    const environment = await buildRuntimeEnvironment(
      config, { internalApiKey: "fm_internal" }, userData, null,
    );
    expect(environment.FISH_MEMORY_SPEECH_FACIAL_WAIT_FOR_READY).toBe(String(enabled));
  });
  it("searches the web only where the reasoning model can and the switch is on", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    const config = configuration();
    config.models.text[0]!.baseUrl = "https://ark.cn-beijing.volces.com/api/v3";
    const on = await buildRuntimeEnvironment(config, { internalApiKey: "fm_internal" }, userData, null);
    expect(on.FISH_MEMORY_ANSWER_ENABLE_SEARCH).toBe("true");
    config.webSearchEnabled = false;
    const off = await buildRuntimeEnvironment(config, { internalApiKey: "fm_internal" }, userData, null);
    expect(off.FISH_MEMORY_ANSWER_ENABLE_SEARCH).toBe("false");
  });
  it("downloads embeddings through the userData Hugging Face cache", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    process.env.FISH_MEMORY_TEST_LEAK = "must-not-be-inherited";

    const enabled = configuration();
    enabled.advanced.voiceDraftEnabled = true;
    process.env.FISH_MEMORY_COMPANION_AUDIO_BACKEND = "livekit";
    process.env.FISH_MEMORY_LIVEKIT_AUDIO_HELPER = "/tmp/stale-helper";
    const environment = await buildRuntimeEnvironment(
      enabled,
      { internalApiKey: "fm_internal", modelApiKeys: { "text-default": "llm-secret" } },
      userData,
      "http://127.0.0.1:12345",
    );

    expect(environment.FISH_MEMORY_EMBEDDING_LOCAL_MODEL_DIR).toBeUndefined();
    expect(environment.HF_HOME).toBe(
      path.join(userData, "models-runtime", "cache", "huggingface"),
    );
    expect(environment.FISH_MEMORY_TEST_LEAK).toBeUndefined();
    expect(environment.FISH_MEMORY_API_URL).toBe("http://127.0.0.1:12345");
    // example.com cannot search: 联网查资料 stays off rather than pretend.
    expect(environment.FISH_MEMORY_ANSWER_ENABLE_SEARCH).toBe("false");
    expect(environment.FISH_MEMORY_EMBEDDING_BATCH_SIZE).toBe("8");
    expect(environment.FISH_MEMORY_LLM_LOG_RETENTION_DAYS).toBe("7");
    expect(environment.TOKENIZERS_PARALLELISM).toBe("false");
    expect(environment.FISH_MEMORY_COMPANION_VOICE_DRAFT_ENABLED).toBe("true");
    expect(environment.FISH_MEMORY_SPEECH_FACIAL_WAIT_FOR_READY).toBe("false");
    expect(environment.FISH_MEMORY_COMPANION_AUDIO_BACKEND).toBe("sounddevice");
    expect(environment.FISH_MEMORY_ASR_MODEL).toBeUndefined();
    expect(environment.FISH_MEMORY_ASR_MODEL_DIR).toBe(
      path.join(userData, "models-runtime", "asr", "sensevoice"),
    );
    expect(environment.FISH_MEMORY_OWNER_VOICE_ENABLED).toBe("false");
    expect(environment.FISH_MEMORY_OWNER_VOICE_DIR).toBe(path.join(userData, "data", "owner-voice"));
    expect(environment.FISH_MEMORY_SHARED_ASR_SOCKET).toBe(
      sharedAsrSocketPath(path.join(userData, "data")),
    );
    expect(environment.FISH_MEMORY_LIVEKIT_AUDIO_HELPER).toBeUndefined();
  });

  it("follows the inherited network but keeps loopback runtime traffic direct", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    const inherited = process.env.NO_PROXY;
    const inheritedProxy = process.env.HTTP_PROXY;
    process.env.NO_PROXY = "api.apiyi.com,localhost";
    process.env.HTTP_PROXY = "http://127.0.0.1:7890";
    try {
      const environment = await buildRuntimeEnvironment(
        configuration(), { internalApiKey: "fm_internal" }, userData, "http://127.0.0.1:12345",
      );
      expect(environment.HTTP_PROXY).toBe("http://127.0.0.1:7890");
      expect(environment.NO_PROXY).toBe("api.apiyi.com,localhost,127.0.0.1,::1");
      expect(environment.no_proxy).toBe(environment.NO_PROXY);
    } finally {
      if (inheritedProxy === undefined) delete process.env.HTTP_PROXY;
      else process.env.HTTP_PROXY = inheritedProxy;
      if (inherited === undefined) delete process.env.NO_PROXY;
      else process.env.NO_PROXY = inherited;
    }
  });

  it("keeps the same audiovisual ASR endpoint when voice is disabled", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    const disabled = configuration();
    disabled.voiceEnabled = false;

    const environment = await buildRuntimeEnvironment(
      disabled,
      { internalApiKey: "fm_internal", modelApiKeys: { "text-default": "llm-secret" } },
      userData,
      "http://127.0.0.1:12345",
    );

    expect(environment.FISH_MEMORY_SHARED_ASR_SOCKET).toBe(
      sharedAsrSocketPath(path.join(userData, "data")),
    );
  });

  it("passes the pinned local helper only for the experimental LiveKit backend", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    const enabled = configuration();
    enabled.advanced.audioBackend = "livekit";
    enabled.audio.ownerRecognitionEnabled = true;

    const environment = await buildRuntimeEnvironment(
      enabled,
      { internalApiKey: "fm_internal", modelApiKeys: { "text-default": "llm-secret" } },
      userData,
      null,
      "/Applications/Fish Memory.app/Contents/Resources/audio/helper",
    );

    expect(environment.FISH_MEMORY_COMPANION_AUDIO_BACKEND).toBe("livekit");
    expect(environment.FISH_MEMORY_OWNER_VOICE_ENABLED).toBe("true");
    expect(environment.FISH_MEMORY_LIVEKIT_AUDIO_HELPER).toBe(
      "/Applications/Fish Memory.app/Contents/Resources/audio/helper",
    );
  });

  it("resolves the macOS packaged and development helper executables", () => {
    const common = {
      configStore: {} as never,
      userDataPath: "/tmp/fish-memory-user",
      authenticateBrowser: async () => undefined,
    };

    vi.stubGlobal("process", { ...process, platform: "darwin", arch: "arm64" });
    try {
      expect(liveKitAudioHelperExecutable({
        ...common,
        projectRoot: "/Users/example/fish-memory",
        resourcesPath: "/Applications/Fish Memory.app/Contents/Resources",
        packaged: true,
      })).toBe(
        path.join("/Applications/Fish Memory.app/Contents/Resources/audio",
          "FishMemoryAudioHelper.app/Contents/MacOS/fish-memory-livekit-audio-helper"),
      );
      expect(liveKitAudioHelperExecutable({
        ...common,
        projectRoot: "/Users/example/fish-memory",
        resourcesPath: "/unused",
        packaged: false,
      })).toBe(
        path.join("/Users/example/fish-memory/desktop/pet/runtime/mac-arm64",
          "FishMemoryAudioHelper.app/Contents/MacOS/fish-memory-livekit-audio-helper"),
      );
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("replaces inherited Agent settings with the secure desktop configuration", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    process.env.FISH_MEMORY_AGENT_URL = "https://leaked.example.test";
    process.env.FISH_MEMORY_AGENT_API_KEY = "leaked-key";
    process.env.FISH_MEMORY_AGENT_TASKS_ENABLED = "false";
    process.env.FISH_MEMORY_AGENT_RUN_POLL_INTERVAL_SECONDS = "999";
    process.env.FISH_MEMORY_AGENT_HTTP_TIMEOUT_SECONDS = "998";
    process.env.FISH_MEMORY_AGENT_SSE_READ_TIMEOUT_SECONDS = "997";
    const configured = configuration();
    configured.agent = {
      enabled: true,
      url: "http://127.0.0.1:8642/p/coder",
      runConcurrency: 6,
      runPollIntervalSeconds: 3,
      httpTimeoutSeconds: 16,
      sseReadTimeoutSeconds: 46,
      replyContextTtlSeconds: 31,
    };

    const environment = await buildRuntimeEnvironment(
      configured,
      { internalApiKey: "fm_internal", modelApiKeys: { "text-default": "llm-secret" }, agentApiKey: "private-agent-key" },
      userData,
      null,
    );

    // Prevent a parent process from disabling Agent tasks after Desktop enabled them.
    expect(environment.FISH_MEMORY_AGENT_TASKS_ENABLED).toBe("true");
    expect(environment.FISH_MEMORY_AGENT_URL).toBe("http://127.0.0.1:8642/p/coder");
    expect(environment.FISH_MEMORY_AGENT_API_KEY).toBe("private-agent-key");
    expect(environment.FISH_MEMORY_AGENT_CREDENTIALS_FILE).toBe(
      path.join(userData, "agent-credentials.json"),
    );
    // Prevent stale inherited concurrency from overriding the Desktop scheduler setting.
    expect(environment.FISH_MEMORY_AGENT_RUN_CONCURRENCY).toBe("6");
    // Prevent a stale inherited poll interval from delaying queued Run pickup.
    expect(environment.FISH_MEMORY_AGENT_RUN_POLL_INTERVAL_SECONDS).toBe("3");
    // Prevent an inherited HTTP timeout from changing dispatch timeout behavior.
    expect(environment.FISH_MEMORY_AGENT_HTTP_TIMEOUT_SECONDS).toBe("16");
    // Prevent an inherited SSE timeout from changing task progress latency behavior.
    expect(environment.FISH_MEMORY_AGENT_SSE_READ_TIMEOUT_SECONDS).toBe("46");
    // Prevent a stale inherited TTL from changing voice reply-context lifetime.
    expect(environment.FISH_MEMORY_AGENT_REPLY_CONTEXT_TTL_SECONDS).toBe("31");
  });

  it("does not inherit stale Agent secrets when none is configured", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    process.env.FISH_MEMORY_AGENT_API_KEY = "inherited-key";

    const environment = await buildRuntimeEnvironment(
      configuration(),
      { internalApiKey: "fm_internal", modelApiKeys: { "text-default": "llm-secret" } },
      userData,
      null,
    );

    expect(environment.FISH_MEMORY_AGENT_API_KEY).toBeUndefined();
    expect(environment.FISH_MEMORY_AGENT_URL).toBe("http://127.0.0.1:8642");
  });

  it("reuses only the pinned complete INT8 embedding snapshot", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    const snapshot = path.join(
      userData,
      "models-runtime",
      "cache",
      "huggingface",
      "hub",
      "models--intfloat--multilingual-e5-small",
      "snapshots",
      "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    );
    const requiredFiles = [
      "onnx/model_qint8_avx512_vnni.onnx",
      "config.json",
      "sentencepiece.bpe.model",
      "special_tokens_map.json",
      "tokenizer.json",
      "tokenizer_config.json",
    ];
    await Promise.all(requiredFiles.map(async (file) => {
      const target = path.join(snapshot, file);
      await mkdir(path.dirname(target), { recursive: true });
      await writeFile(target, file);
    }));

    const environment = await buildRuntimeEnvironment(
      configuration(),
      { internalApiKey: "fm_internal", modelApiKeys: { "text-default": "llm-secret" } },
      userData,
      "http://127.0.0.1:12345",
    );

    expect(environment.FISH_MEMORY_EMBEDDING_LOCAL_MODEL_DIR).toBe(snapshot);
  });

  it("does not silently reuse a FP32-only embedding snapshot", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    const snapshot = path.join(
      userData,
      "models-runtime",
      "cache",
      "huggingface",
      "hub",
      "models--intfloat--multilingual-e5-small",
      "snapshots",
      "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    );
    const requiredFiles = [
      "onnx/model.onnx",
      "config.json",
      "sentencepiece.bpe.model",
      "special_tokens_map.json",
      "tokenizer.json",
      "tokenizer_config.json",
    ];
    await Promise.all(requiredFiles.map(async (file) => {
      const target = path.join(snapshot, file);
      await mkdir(path.dirname(target), { recursive: true });
      await writeFile(target, file);
    }));

    const environment = await buildRuntimeEnvironment(
      configuration(),
      { internalApiKey: "fm_internal", modelApiKeys: { "text-default": "llm-secret" } },
      userData,
      "http://127.0.0.1:12345",
    );

    expect(environment.FISH_MEMORY_EMBEDDING_LOCAL_MODEL_DIR).toBeUndefined();
  });
});

describe("ensureAssistantBinding", () => {
  it("binds the first existing fish when the desktop database is unbound", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const fetcher = async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.endsWith("/v1/fishes/assistant") && !init?.method) {
        return Response.json({ fishId: null });
      }
      if (url.endsWith("/v1/fishes")) {
        return Response.json({ items: [{ fishId: "fish-1" }] });
      }
      return Response.json({ fishId: "fish-1" });
    };

    await expect(
      ensureAssistantBinding("http://127.0.0.1:1234", "secret", fetcher),
    ).resolves.toBe(true);
    expect(requests).toHaveLength(3);
    expect(requests[2]).toMatchObject({
      url: "http://127.0.0.1:1234/v1/fishes/fish-1/assistant",
      init: {
        method: "PUT",
        headers: { Authorization: "Bearer secret" },
      },
    });
  });

  it("does not replace an existing assistant binding", async () => {
    const fetcher = async () => Response.json({ fishId: "fish-2" });

    await expect(
      ensureAssistantBinding("http://127.0.0.1:1234", "secret", fetcher),
    ).resolves.toBe(true);
  });

  it("reports that voice must wait when no fish exists", async () => {
    let request = 0;
    const fetcher = async () => {
      request += 1;
      return request === 1
        ? Response.json({ fishId: null })
        : Response.json({ items: [] });
    };

    await expect(
      ensureAssistantBinding("http://127.0.0.1:1234", "secret", fetcher),
    ).resolves.toBe(false);
  });
});

describe("companionReadinessFromLog", () => {
  it("distinguishes an active microphone listener from an unbound companion", () => {
    expect(companionReadinessFromLog("已绑定助手 Fish abc，唤醒词为“小鱼”")).toBe(
      "bound",
    );
    expect(companionReadinessFromLog("当前没有绑定助手 Fish，请先创建一个 Fish")).toBe(
      "unbound",
    );
    expect(companionReadinessFromLog("Smart Turn 已启用")).toBeNull();
  });
});

describe("companionStartupFailureFromLog", () => {
  it("turns a stalled or missing LiveKit helper into an actionable status", () => {
    expect(companionStartupFailureFromLog(
      "RuntimeError: LiveKit audio helper did not deliver its first frame within 5 seconds",
    )?.text).toContain("没有声音");
    expect(companionStartupFailureFromLog(
      "LiveKit audio helper is missing or not executable: /missing/helper",
    )?.kind).toBe("audio");
    expect(companionStartupFailureFromLog(
      "RuntimeError: LiveKit audio helper failed: ... Code=705 \"Unknown(Microphone permission is not granted)\"",
    )).toEqual({ kind: "microphone", text: "麦克风没有权限，TA 听不到你说话" });
    expect(companionStartupFailureFromLog("Smart Turn 已启用")).toBeNull();
  });
});

describe("companion speech activity", () => {
  const runId = "91b1f58b-c368-4216-b2be-b9967a0cdecd";
  const trace = (event: "speech_start" | "speech_end", generation: number) => (
    `2026-09-04 INFO voice_trace event=${event} `
    + JSON.stringify({ runId, atMs: 12.3, speechGeneration: generation })
  );

  it("parses only structured speech boundary traces", () => {
    expect(companionSpeechActivityFromLog(trace("speech_start", 7))).toEqual({
      active: true,
      runId,
      speechGeneration: 7,
    });
    expect(companionSpeechActivityFromLog(trace("speech_end", 7))).toEqual({
      active: false,
      runId,
      speechGeneration: 7,
    });
    expect(companionSpeechActivityFromLog("voice_trace event=speech_start {}"))
      .toBeNull();
    expect(companionSpeechActivityFromLog("ordinary speech_start text")).toBeNull();
  });

  it("reassembles chunks, rejects stale END, and resets on Companion stop", async () => {
    const { manager, userData } = await logTestManager();
    manager["updateStatus"]("ready", "ready", true, "http://127.0.0.1:49152");
    const events: CompanionSpeechActivityEvent[] = [];
    manager.on("companion-speech-activity", (event) => events.push(event));
    const start = `${trace("speech_start", 7)}\n`;
    manager["forwardLog"]("companion", start.slice(0, 45));
    expect(events).toEqual([]);
    manager["forwardLog"]("companion", start.slice(45));
    manager["forwardLog"]("companion", start);
    manager["forwardLog"]("companion", `${trace("speech_end", 6)}\n`);
    expect(events).toEqual([{
      active: true,
      reason: "speech_start",
      runId,
      speechGeneration: 7,
    }]);

    manager["resetCompanionSpeechActivity"]();
    expect(events.at(-1)).toEqual({
      active: false,
      reason: "companion_stopped",
      runId,
      speechGeneration: 7,
    });
    await vi.waitFor(async () => {
      const log = await readFile(path.join(userData, "logs", "runtime.log"), "utf8");
      expect(log.match(/\[companion\]/gu)).toHaveLength(4);
    });
  });

  it("ignores late speech traces from a replaced Companion process", async () => {
    const { manager, userData } = await logTestManager();
    manager["updateStatus"]("ready", "ready", true, "http://127.0.0.1:49152");
    const events: CompanionSpeechActivityEvent[] = [];
    manager.on("companion-speech-activity", (event) => events.push(event));
    const replaced = {} as ChildProcess;
    const current = {} as ChildProcess;
    manager["companionProcess"] = current;

    manager["forwardLog"]("companion", `${trace("speech_start", 7)}\n`, replaced);
    expect(events).toEqual([]);

    manager["forwardLog"]("companion", `${trace("speech_start", 8)}\n`, current);
    expect(events).toEqual([{
      active: true,
      reason: "speech_start",
      runId,
      speechGeneration: 8,
    }]);
    await vi.waitFor(async () => {
      const log = await readFile(path.join(userData, "logs", "runtime.log"), "utf8");
      expect(log.match(/\[companion\]/gu)).toHaveLength(2);
    });
  });
});

describe("prepareRuntimeLog", () => {
  it("keeps only a complete bounded tail after a runaway log", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-log-test-"));
    temporaryDirectories.push(userData);
    const logPath = path.join(userData, "logs", "runtime.log");
    await mkdir(path.dirname(logPath), { recursive: true });
    const line = `${"x".repeat(1_023)}\n`;
    await writeFile(logPath, `${line.repeat(8_193)}tail-marker\n`);

    await prepareRuntimeLog(logPath);

    const retained = await readFile(logPath);
    expect(retained.byteLength).toBeLessThanOrEqual(4 * 1_024 * 1_024);
    expect(retained.toString("utf8").endsWith("tail-marker\n")).toBe(true);
  });

  it("bounds serialized writes during one RuntimeManager lifetime", async () => {
    const { manager, userData } = await logTestManager();
    const completeLine = `${"x".repeat(1_023)}\n`;
    const chunk = completeLine.repeat(1_024);
    for (let index = 0; index < 8; index += 1) {
      manager["persistRuntimeLog"](chunk);
    }
    manager["persistRuntimeLog"](
      `${completeLine.repeat(1_023)}${"latest-marker".padEnd(1_023, "y")}\n`,
    );

    await manager["runtimeLogWrite"];

    const retained = await readFile(path.join(userData, "logs", "runtime.log"));
    expect(retained.byteLength).toBeLessThanOrEqual(4 * 1_024 * 1_024);
    expect(retained.toString("utf8")).toContain("latest-marker");
    expect(manager["runtimeLogBytes"]).toBe(retained.byteLength);
  });
});

describe("RuntimeManager voice process lifecycle", () => {
  it.each(["microphone", "playback"] as const)("toggles %s without restarting the other audio direction", async (kind) => {
    const { manager } = await logTestManager();
    let saved = { ...configuration(), microphoneEnabled: true, playbackEnabled: true };
    manager["options"].configStore.load = async () => saved;
    manager["options"].configStore.save = async (value) => {
      saved = { ...saved, ...value } as typeof saved;
      return saved;
    };
    const api = {} as ChildProcess, companion = {} as ChildProcess;
    manager["apiProcess"] = api;
    manager["companionProcess"] = companion;
    manager["updateStatus"]("ready", "ready", true, "http://127.0.0.1:49152");
    const originalFetch = globalThis.fetch;
    const request = vi.fn(async () => new Response('{"enabled":false}', { status: 200 }));
    globalThis.fetch = request;
    try {
      if (kind === "microphone") await manager.setMicrophoneEnabled(false);
      else await manager.setPlaybackEnabled(false);
      expect(manager["apiProcess"]).toBe(api);
      expect(manager["companionProcess"]).toBe(companion);
      expect(saved.voiceEnabled).toBe(true);
      expect(saved.microphoneEnabled).toBe(kind !== "microphone");
      expect(saved.playbackEnabled).toBe(kind !== "playback");
      expect(request.mock.calls[0]?.[0]).toBe(`http://127.0.0.1:49152/internal/runtime/${kind}`);
      expect(manager.currentStatus().microphoneEnabled).toBe(saved.microphoneEnabled);
      expect(manager.currentStatus().playbackEnabled).toBe(saved.playbackEnabled);
    } finally { globalThis.fetch = originalFetch; }
  });
  it("does not start media ASR on a system without playback audio support", async () => {
    vi.mocked(supportsSystemAudio).mockReturnValue(false);
    const { manager } = await logTestManager();
    const configured = configuration();
    configured.voiceEnabled = false;
    configured.perception.audioPerceptionEnabled = true;
    manager["processRegistry"].stopRole = async () => [];
    const spawn = vi.fn(() => ({} as ChildProcess));
    manager["spawnRole"] = spawn;
    await manager["restartCompanionNow"](configured);
    expect(spawn).not.toHaveBeenCalled();
    expect(manager["environmentAsrProcess"]).toBeNull();
  });

  it("hands media recognition to a microphone-free process without restarting the API", async () => {
    const { manager } = await logTestManager();
    const configured = configuration();
    configured.voiceEnabled = false;
    configured.perception.audioPerceptionEnabled = true;
    const api = {} as ChildProcess;
    const companion = {} as ChildProcess;
    manager["apiProcess"] = api;
    manager["companionProcess"] = companion;
    manager["updateStatus"]("ready", "ready", true, "http://127.0.0.1:49152");
    const operations: string[] = [];
    manager["processRegistry"].stopRole = async (role) => {
      operations.push(`stop:${role}`);
      return [];
    };
    const background = {} as ChildProcess;
    manager["spawnRole"] = (role, env) => {
      operations.push(`start:${role}`);
      expect(env.FISH_MEMORY_SHARED_ASR_SOCKET).toMatch(/(?:shared-asr|fish-memory-asr-[0-9a-f]{12})\.sock$/u);
      expect(role).toBe("environment-asr");
      return background;
    };

    await manager["restartCompanionNow"](configured);
    expect(operations).toEqual(["stop:environment-asr", "stop:companion", "start:environment-asr"]);
    expect(manager["apiProcess"]).toBe(api);
    expect(manager["companionProcess"]).toBeNull();
    expect(manager["environmentAsrProcess"]).toBe(background);
    expect(manager.currentStatus().voiceEnabled).toBe(false);
    // Late readiness from the retired Companion must not reopen microphone UI.
    manager["forwardLog"]("companion", "助手 Fish 绑定已变更: None -> fish", companion);
    expect(manager.currentStatus().voiceEnabled).toBe(false);

    configured.perception.audioPerceptionEnabled = false;
    operations.length = 0;
    await manager["restartCompanionNow"](configured);
    expect(operations).toEqual(["stop:environment-asr", "stop:companion"]);
    expect(manager["environmentAsrProcess"]).toBeNull();
    expect(manager["apiProcess"]).toBe(api);
  });

  it.each([true, false])("cancels enrollment without restarting the API, restoring voice intent %s", async (voiceEnabled) => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    const configured = { ...configuration(), voiceEnabled };
    const save = vi.fn(async () => configured);
    const authenticateBrowser = vi.fn(async () => {});
    const manager = new RuntimeManager({
      configStore: { load: async () => configured, save, runtimeSecrets: async () => ({ internalApiKey: "test" }) },
      userDataPath: userData, projectRoot: process.cwd(), resourcesPath: "/unused", packaged: true,
      authenticateBrowser, processController: fakeProcessController([]),
    });
    const api = {} as ChildProcess;
    manager["apiProcess"] = api;
    manager["updateStatus"]("ready", "ready", voiceEnabled, "http://127.0.0.1:49152");
    const restartCompanion = vi.fn(async () => {});
    manager["restartCompanionNow"] = restartCompanion;
    const actions: string[] = [];
    manager["spawnRole"] = (role, environment, args = []) => {
      expect(role).toBe("owner-voice");
      expect(environment.FISH_MEMORY_OWNER_VOICE_DIR).toBe(path.join(userData, "data", "owner-voice"));
      expect(args).toContain(path.join(userData, "data", "owner-voice"));
      const child = Object.assign(new EventEmitter(), { stdout: new EventEmitter(), stderr: new EventEmitter() }) as unknown as ChildProcess;
      const action = args[1];
      actions.push(action);
      queueMicrotask(() => {
        if (action === "prepare") {
          child.stdout!.emit("data", Buffer.from(JSON.stringify({ event: "result", status: {
            enrolled: false, modelReady: true, modelId: "cam++", createdAt: null, detail: "ready",
          } }) + "\n"));
          child.emit("close", 0);
        } else child.stdout!.emit("data", Buffer.from('{"event":"recording","durationSeconds":15}\n'));
      });
      return child;
    };
    const enrollment = manager.enrollOwnerVoice();
    const rejected = expect(enrollment).rejects.toThrow("取消");
    await vi.waitFor(() => expect(manager.ownerVoice.currentState().phase).toBe("recording"));
    expect(() => manager.setVoiceEnabled(!voiceEnabled)).toThrow("处理中");
    await expect(manager.withConfigurationChange(async () => undefined)).rejects.toThrow("处理中");
    await manager.ownerVoice.cancel();
    await rejected;
    expect(actions).toEqual(["prepare", "enroll"]);
    expect(manager["apiProcess"]).toBe(api);
    expect(manager.currentStatus().apiUrl).toBe("http://127.0.0.1:49152");
    expect(restartCompanion).toHaveBeenCalledTimes(voiceEnabled ? 1 : 0);
    expect(save).not.toHaveBeenCalled();
    expect(authenticateBrowser).not.toHaveBeenCalled();
  });

  it("persists only changed, sanitized perception state to runtime.log", async () => {
    const { manager, userData } = await logTestManager();
    const status: PerceptionManagerStatus = {
      phase: "starting", detail: "private-window-canary", crashCount: 0, helperPid: null,
      modules: { workspace: false, screen: true, accessibility: false, camera: false },
      capabilities: null,
    };
    manager.recordPerceptionStatus(status);
    manager.recordPerceptionStatus(status);
    await vi.waitFor(async () => {
      const log = await readFile(path.join(userData, "logs", "runtime.log"), "utf8");
      expect(log.match(/perception_status/gu)).toHaveLength(1);
      expect(log).not.toContain("private-window-canary");
    });
  });

  it("does not resume Companion when application shutdown supersedes enrollment", async () => {
    const { manager } = await logTestManager();
    manager["apiProcess"] = {} as ChildProcess;
    manager["updateStatus"]("ready", "ready", true, "http://127.0.0.1:49152");
    manager["processRegistry"].stopRole = async () => [];
    const restart = vi.fn(async () => {});
    manager["restartCompanionNow"] = restart;
    const restore = await manager["suspendCompanionAudio"]("主人声纹处理中，语音伴侣暂时暂停");
    manager["shutdownRequested"] = true;
    await restore();
    expect(restart).not.toHaveBeenCalled();
  });
  it("pauses Companion while the mouth test borrows the microphone and resumes it once", async () => {
    const { manager } = await logTestManager();
    manager["apiProcess"] = {} as ChildProcess;
    manager["companionProcess"] = {} as ChildProcess;
    manager["updateStatus"]("ready", "ready", true, "http://127.0.0.1:49152");
    manager["processRegistry"].stopRole = async () => [];
    const restart = vi.fn(async () => {});
    manager["restartCompanionNow"] = restart;
    await manager.borrowMicrophone();
    await manager.borrowMicrophone();
    expect(manager["companionProcess"]).toBeNull();
    expect(manager.currentStatus().detail).toBe("正在试嘴型，暂时不听你说话");
    await manager.returnMicrophone();
    await manager.returnMicrophone();
    expect(restart).toHaveBeenCalledTimes(1);
  });

  it("leaves a stopped Companion alone when the mouth test borrows the microphone", async () => {
    const { manager } = await logTestManager();
    const restart = vi.fn(async () => {});
    manager["restartCompanionNow"] = restart;
    await manager.borrowMicrophone();
    await manager.returnMicrophone();
    expect(restart).not.toHaveBeenCalled();
  });
  it.each([
    "RuntimeError: LiveKit audio helper did not deliver its first frame within 5 seconds",
    "LiveKit audio helper is missing or not executable: /missing/helper",
    "Unable to start LiveKit audio helper",
  ].flatMap((failure) => (["starting-companion", "ready"] as const)
    .map((phase) => [phase, failure] as const)))(
    "keeps authenticated chat available through audio recovery from %s: %s",
    async (phase, failure) => {
      const { manager, userData, authenticateBrowser } = await logTestManager();
      const apiUrl = "http://127.0.0.1:49152";
      // startNow reaches this phase only after authenticateBrowser has resolved.
      manager["updateStatus"](phase, "语音状态", true, apiUrl);
      const statuses: DesktopRuntimeStatus[] = [];
      manager.on("status", (status: DesktopRuntimeStatus) => statuses.push(status));

      manager["forwardLog"]("companion", failure);
      expect(manager.currentStatus()).toMatchObject({
        phase: "starting-companion",
        apiUrl,
        voiceEnabled: true,
      });
      expect(manager.currentStatus().detail).toMatch(/重试/u);
      expect(manager.currentStatus().voiceIssue?.kind).toBe("audio");
      expect(resolveMainWindowState("chat", true, manager.currentStatus())).toEqual({
        view: "chat",
        chatUrl: `${apiUrl}/?nativeAgentPlayback=1`,
      });

      const recovered = "已绑定助手 Fish test，唤醒词为“小鱼”";
      manager["forwardLog"]("companion", recovered);
      expect(statuses.map((status) => status.phase)).toEqual(["starting-companion", "ready"]);
      // Neither state clears the browser authorization in handleRuntimeStatus.
      expect(resolveMainWindowState("chat", true, manager.currentStatus()).chatUrl).toBe(
        `${apiUrl}/?nativeAgentPlayback=1`,
      );
      expect(authenticateBrowser).not.toHaveBeenCalled();

      // forwardLog persists asynchronously; let those writes finish before cleanup.
      await vi.waitFor(async () => {
        const log = await readFile(path.join(userData, "logs", "runtime.log"), "utf8");
        expect(log).toContain(failure);
        expect(log).toContain(recovered);
      });
    },
  );

  it.each(["stopped", "restarting", "starting-api", "failed"] as const)(
    "ignores late Companion state logs while the API is %s",
    async (phase) => {
      const { manager, userData } = await logTestManager();
      manager["updateStatus"](phase, "API unavailable", true, "http://127.0.0.1:49152");
      const statuses: DesktopRuntimeStatus[] = [];
      manager.on("status", (status: DesktopRuntimeStatus) => statuses.push(status));
      const logs = [
        "RuntimeError: LiveKit audio helper did not deliver its first frame within 5 seconds",
        "已绑定助手 Fish test，唤醒词为“小鱼”",
        "当前没有绑定助手 Fish",
        "模型准备中",
      ];
      manager["forwardLog"]("companion", logs.join("\n"));
      expect(statuses).toEqual([]);
      expect(manager.currentStatus().phase).toBe(phase);
      expect(resolveMainWindowState("chat", true, manager.currentStatus()).chatUrl).toBeNull();
      await vi.waitFor(async () => {
        const log = await readFile(path.join(userData, "logs", "runtime.log"), "utf8");
        for (const line of logs) expect(log).toContain(line);
      });
    },
  );

  it("ignores late Companion readiness while shutting down a ready API", async () => {
    const { manager, userData } = await logTestManager();
    manager["updateStatus"]("ready", "运行正常", true, "http://127.0.0.1:49152");
    manager["stopping"] = true;
    const statuses: DesktopRuntimeStatus[] = [];
    manager.on("status", (status: DesktopRuntimeStatus) => statuses.push(status));
    const log = "已绑定助手 Fish test，唤醒词为“小鱼”";
    manager["forwardLog"]("companion", log);
    expect(statuses).toEqual([]);
    await vi.waitFor(async () => {
      expect(await readFile(path.join(userData, "logs", "runtime.log"), "utf8")).toContain(log);
    });
  });

  it("keeps a Companion's readiness that arrives while the API is still starting", async () => {
    // The Companion starts alongside the API; it may finish loading first.
    const { manager } = await logTestManager();
    manager["updateStatus"]("starting-api", "正在启动本地记忆服务…", true, "http://127.0.0.1:49152");
    const statuses: DesktopRuntimeStatus[] = [];
    manager.on("status", (status: DesktopRuntimeStatus) => statuses.push(status));
    manager["forwardLog"]("companion", "已绑定助手 Fish test，唤醒词为“小星”");
    // Not announced before the API is up…
    expect(statuses).toEqual([]);
    // …but remembered, so the API's readiness can go straight to ready.
    expect(manager["earlyCompanionReadiness"]).toBe("bound");
  });

  it("stops a recorded Companion without a process handle even when voice is already disabled", async () => {
    const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
    temporaryDirectories.push(userData);
    const resourcesPath = "/Applications/Fish Memory.app/Contents/Resources";
    const executable = path.join(
      resourcesPath,
      "runtime",
      "fish-memory-runtime",
      `fish-memory-runtime${process.platform === "win32" ? ".exe" : ""}`,
    );
    const controller = fakeProcessController([
      { pid: 77, parentPid: process.pid, command: `${executable} companion` },
    ]);
    const disabled = { ...configuration(), voiceEnabled: false };
    const manager = new RuntimeManager({
      configStore: {
        load: async () => disabled,
        save: async () => disabled,
        runtimeSecrets: async () => ({
          internalApiKey: "fm_internal",
          modelApiKeys: { "text-default": "llm-secret" },
        }),
      },
      userDataPath: userData,
      projectRoot: process.cwd(),
      resourcesPath,
      packaged: true,
      authenticateBrowser: async () => undefined,
      processController: controller,
    });
    manager["processRegistry"].track(77, "companion", executable, ["companion"]);

    await manager.setVoiceEnabled(false);

    expect(controller.stopped).toEqual([77]);
    expect(manager.currentStatus().voiceEnabled).toBe(false);
  });
});

async function logTestManager() {
  const userData = await mkdtemp(path.join(os.tmpdir(), "fish-memory-runtime-test-"));
  temporaryDirectories.push(userData);
  const configured = configuration();
  const authenticateBrowser = vi.fn(async () => undefined);
  const manager = new RuntimeManager({
    configStore: {
      load: async () => configured,
      save: async () => configured,
      runtimeSecrets: async () => ({ internalApiKey: "fm_internal" }),
    },
    userDataPath: userData,
    projectRoot: process.cwd(),
    resourcesPath: "/unused",
    packaged: true,
    authenticateBrowser,
  });
  return { manager, userData, authenticateBrowser };
}

describe("runtime startup failure cleanup", () => {
  it.each(["api", "companion"] as const)("cleans up when %s launch throws synchronously", async (failedRole) => {
    const { manager, authenticateBrowser } = await logTestManager();
    const server = {
      unref() {}, once() {},
      listen(_port: number, _host: string, listening: () => void) { queueMicrotask(listening); },
      address: () => ({ port: 49152 }),
      close: (closed: () => void) => closed(),
    } as unknown as net.Server;
    const createServer = vi.spyOn(net, "createServer").mockReturnValue(server);
    vi.stubGlobal("process", { ...process, platform: "win32" });
    const stdin = new PassThrough();
    const api = Object.assign(new EventEmitter(), {
      pid: 1234, exitCode: null as number | null, stdin,
    }) as unknown as ChildProcess;
    let shutdownInput = "";
    stdin.on("data", (data) => { shutdownInput += data.toString(); });
    stdin.on("finish", () => { api.exitCode = 0; api.emit("exit", 0); });
    const failure = new Error("无法管理 Windows 子进程");
    const spawn = vi.fn((role: string) => {
      if (role === failedRole) throw failure;
      return api;
    });
    manager["spawnRole"] = spawn;
    manager["prepareProcessRegistry"] = async () => {};
    manager["processRegistry"].stopRole = async () => [];
    const clear = vi.spyOn(manager["processRegistry"], "clear").mockResolvedValue(undefined);
    try {
      await expect(manager.start()).rejects.toBe(failure);
      expect(spawn.mock.calls.map(([role]) => role)).toEqual(failedRole === "api" ? ["api"] : ["api", "companion"]);
      expect(shutdownInput).toBe(failedRole === "companion" ? "__FISH_MEMORY_SHUTDOWN__\n" : "");
      if (failedRole === "companion") expect(api.exitCode).toBe(0);
      expect(clear).toHaveBeenCalledOnce();
      expect(manager["apiProcess"]).toBeNull();
      expect(manager["companionProcess"]).toBeNull();
      expect(manager.currentStatus()).toMatchObject({ phase: "failed", detail: failure.message, apiUrl: null });
      expect(authenticateBrowser).not.toHaveBeenCalled();
    } finally {
      createServer.mockRestore();
      vi.unstubAllGlobals();
    }
  });
});

describe("retry model preparation", () => {
  const apiUrl = "http://127.0.0.1:49152";
  const alive = () => ({ exitCode: null } as ChildProcess);
  function gate<T = void>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((done) => { resolve = done; });
    return { promise, resolve };
  }

  it("marks the API ready only after its readiness check and browser authentication finish", async () => {
    const { manager, authenticateBrowser } = await logTestManager();
    const server = {
      unref() {}, once() {},
      listen(_port: number, _host: string, listening: () => void) { queueMicrotask(listening); },
      address: () => ({ port: 49152 }),
      close: (closed: () => void) => closed(),
    } as unknown as net.Server;
    const createServer = vi.spyOn(net, "createServer").mockReturnValue(server);
    const ready = gate<Response>(), authenticated = gate();
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(() => ready.promise);
    manager["prepareProcessRegistry"] = async () => {};
    manager["options"].packaged = false;
    manager["spawnRole"] = () => alive();
    manager["bindAssistant"] = async () => {};
    authenticateBrowser.mockImplementation(() => authenticated.promise);
    try {
      const start = manager.start();
      await vi.waitFor(() => expect(fetch).toHaveBeenCalledOnce());
      expect(manager["apiProcess"]).not.toBeNull();
      expect(manager.currentStatus().apiUrl).toBe(apiUrl);
      expect(manager["apiReady"]).toBe(false);
      ready.resolve(new Response("{}", { status: 200 }));
      await vi.waitFor(() => expect(authenticateBrowser).toHaveBeenCalledOnce());
      expect(manager["apiReady"]).toBe(false);
      authenticated.resolve();
      await start;
      expect(manager["apiReady"]).toBe(true);
    } finally {
      fetch.mockRestore();
      createServer.mockRestore();
    }
  });

  it.each([false, true])("restarts the entire runtime when API readiness is absent (process exists: %s)", async (hasProcess) => {
    const { manager } = await logTestManager();
    manager["apiProcess"] = hasProcess ? alive() : null;
    manager["updateStatus"](hasProcess ? "starting-api" : "failed", "embedding download failed", true, hasProcess ? apiUrl : null);
    const stop = vi.fn(async () => {});
    const start = vi.fn(async () => {});
    const voice = vi.fn(async () => {});
    manager["stopNow"] = stop;
    manager["startNow"] = start;
    manager["restartCompanionNow"] = voice;
    manager["failureTimes"] = [Date.now(), Date.now(), Date.now(), Date.now()];

    await manager.retryModelPreparation();
    expect(stop).toHaveBeenCalledOnce();
    expect(start).toHaveBeenCalledWith(manager["lifecycleVersion"]);
    expect(voice).not.toHaveBeenCalled();
    expect(manager["failureTimes"]).toEqual([]);
  });

  it.each(["ready", "starting-companion"] as const)("preserves the healthy API and audio switches in %s", async (phase) => {
    const { manager, authenticateBrowser } = await logTestManager();
    const api = alive();
    const configured = { ...configuration(), microphoneEnabled: false, playbackEnabled: true };
    manager["apiProcess"] = api;
    manager["apiReady"] = true;
    manager["updateStatus"](phase, "audio retry", true, apiUrl);
    manager["options"].configStore.load = async () => configured;
    const save = vi.fn(async () => configured);
    manager["options"].configStore.save = save;
    const stop = vi.fn(async () => {});
    const voice = vi.fn(async () => {});
    manager["stopNow"] = stop;
    manager["restartCompanionNow"] = voice;

    await manager.retryModelPreparation();
    expect(voice).toHaveBeenCalledOnce();
    expect(voice).toHaveBeenCalledWith(configured);
    expect(manager["apiProcess"]).toBe(api);
    expect(stop).not.toHaveBeenCalled();
    expect(save).not.toHaveBeenCalled();
    expect(authenticateBrowser).not.toHaveBeenCalled();
  });

  it("restores browser authorization after Companion exhausts automatic recovery without restarting the API", async () => {
    const { manager, authenticateBrowser } = await logTestManager();
    const api = alive(), companion = alive();
    manager["apiProcess"] = api;
    manager["apiReady"] = true;
    manager["companionProcess"] = companion;
    manager["updateStatus"]("ready", "ready", true, apiUrl);
    manager["failureTimes"] = [Date.now(), Date.now(), Date.now()];
    manager["handleUnexpectedExit"]("companion", companion, "code 1");
    expect(manager.currentStatus().phase).toBe("failed");
    expect(manager["apiReady"]).toBe(true);
    const stop = vi.fn(async () => {});
    const voice = vi.fn(async () => { expect(authenticateBrowser).toHaveBeenCalledOnce(); });
    manager["stopNow"] = stop;
    manager["restartCompanionNow"] = voice;

    await manager.retryModelPreparation();
    expect(authenticateBrowser).toHaveBeenCalledOnce();
    expect(authenticateBrowser).toHaveBeenCalledWith(apiUrl, "fm_internal");
    expect(voice).toHaveBeenCalledOnce();
    expect(stop).not.toHaveBeenCalled();
    expect(manager["apiProcess"]).toBe(api);
    expect(manager["failureTimes"]).toEqual([]);
  });

  it("clears readiness when the API crashes", async () => {
    const { manager } = await logTestManager();
    const api = alive();
    manager["apiProcess"] = api;
    manager["apiReady"] = true;
    manager["failureTimes"] = [Date.now(), Date.now(), Date.now()];
    manager["handleUnexpectedExit"]("api", api, "code 1");
    expect(manager["apiReady"]).toBe(false);
    expect(manager["apiProcess"]).toBeNull();
    const start = vi.fn(async () => {});
    manager["stopNow"] = async () => {};
    manager["startNow"] = start;
    await manager.retryModelPreparation();
    expect(start).toHaveBeenCalledOnce();
  });

  it("retries only environment ASR when voice is disabled", async () => {
    const { manager } = await logTestManager();
    const configured = { ...configuration(), voiceEnabled: false, microphoneEnabled: false, playbackEnabled: false };
    configured.perception.audioPerceptionEnabled = true;
    manager["options"].configStore.load = async () => configured;
    const save = vi.fn(async () => configured);
    manager["options"].configStore.save = save;
    const api = alive();
    manager["apiProcess"] = api;
    manager["apiReady"] = true;
    manager["updateStatus"]("ready", "ready", false, apiUrl);
    manager["processRegistry"].stopRole = async () => [];
    const spawn = vi.fn(() => alive());
    manager["spawnRole"] = spawn;

    await manager.retryModelPreparation();
    expect(spawn).toHaveBeenCalledOnce();
    expect(spawn.mock.calls[0]?.[0]).toBe("environment-asr");
    expect(manager["apiProcess"]).toBe(api);
    expect(manager["companionProcess"]).toBeNull();
    expect(manager.currentStatus().voiceEnabled).toBe(false);
    expect(save).not.toHaveBeenCalled();
  });

  it("coalesces clicks and chooses the failed layer after earlier queued work finishes", async () => {
    const { manager, userData } = await logTestManager();
    const statusFile = path.join(modelStatusDirectory(userData), "asr.json");
    await mkdir(path.dirname(statusFile), { recursive: true });
    await writeFile(statusFile, '{"state":"failed"}');
    const pending = gate();
    manager["operation"] = pending.promise;
    const voice = vi.fn(async () => {});
    const stop = vi.fn(async () => {});
    manager["restartCompanionNow"] = voice;
    manager["stopNow"] = stop;
    const first = manager.retryModelPreparation();
    const second = manager.retryModelPreparation();
    expect(first).toBe(second);
    expect(await readFile(statusFile, "utf8")).toContain("failed");
    manager["apiProcess"] = alive();
    manager["apiReady"] = true;
    manager["updateStatus"]("ready", "ready", true, apiUrl);
    pending.resolve();
    await first;
    expect(voice).toHaveBeenCalledOnce();
    expect(stop).not.toHaveBeenCalled();
    expect(manager["modelPreparationRetry"]).toBeNull();
    await expect(readFile(statusFile, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("allows another attempt after a failed retry", async () => {
    const { manager } = await logTestManager();
    manager["stopNow"] = async () => {};
    const start = vi.fn().mockRejectedValueOnce(new Error("offline")).mockResolvedValue(undefined);
    manager["startNow"] = start;
    await expect(manager.retryModelPreparation()).rejects.toThrow("offline");
    await manager.retryModelPreparation();
    expect(start).toHaveBeenCalledTimes(2);
  });

  it("lets stop supersede a queued retry and ignores further clicks after shutdown", async () => {
    const { manager, userData } = await logTestManager();
    const statusFile = path.join(modelStatusDirectory(userData), "embeddings.json");
    await mkdir(path.dirname(statusFile), { recursive: true });
    await writeFile(statusFile, '{"state":"failed"}');
    const pending = gate();
    manager["operation"] = pending.promise;
    const stop = vi.fn(async () => {});
    const start = vi.fn(async () => {});
    manager["stopNow"] = stop;
    manager["startNow"] = start;
    const retry = manager.retryModelPreparation();
    await manager.stop();
    pending.resolve();
    await retry;
    await manager.retryModelPreparation();
    expect(stop).toHaveBeenCalledOnce();
    expect(start).not.toHaveBeenCalled();
    expect(await readFile(statusFile, "utf8")).toContain("failed");
  });

  it("keeps failure reports if owner voice becomes busy before the retry reaches the queue", async () => {
    const { manager, userData } = await logTestManager();
    const statusFile = path.join(modelStatusDirectory(userData), "embeddings.json");
    await mkdir(path.dirname(statusFile), { recursive: true });
    await writeFile(statusFile, '{"state":"failed"}');
    const pending = gate();
    manager["operation"] = pending.promise;
    const retry = manager.retryModelPreparation();
    const rejected = expect(retry).rejects.toThrow("busy");
    const idle = vi.spyOn(manager.ownerVoice, "assertIdle").mockImplementation(() => { throw new Error("busy"); });
    try {
      pending.resolve();
      await rejected;
      expect(await readFile(statusFile, "utf8")).toContain("failed");
    } finally { idle.mockRestore(); }
  });

  it("does not restart after stop interrupts retry teardown", async () => {
    const { manager } = await logTestManager();
    const teardown = gate();
    const stop = vi.fn().mockImplementationOnce(() => teardown.promise).mockResolvedValue(undefined);
    const start = vi.fn(async () => {});
    manager["stopNow"] = stop;
    manager["startNow"] = start;
    const retry = manager.retryModelPreparation();
    await vi.waitFor(() => expect(stop).toHaveBeenCalledOnce());
    await manager.stop();
    teardown.resolve();
    await retry;
    expect(start).not.toHaveBeenCalled();
  });

  it("does not spawn Companion if shutdown occurs while binding it", async () => {
    const { manager } = await logTestManager();
    manager["apiProcess"] = alive();
    manager["apiReady"] = true;
    manager["updateStatus"]("ready", "ready", true, apiUrl);
    manager["processRegistry"].stopRole = async () => [];
    manager["processRegistry"].clear = async () => {};
    const binding = gate();
    const bind = vi.fn(() => binding.promise);
    manager["bindAssistant"] = bind;
    const spawn = vi.fn(() => alive());
    manager["spawnRole"] = spawn;
    const retry = manager.retryModelPreparation();
    await vi.waitFor(() => expect(bind).toHaveBeenCalledOnce());
    await manager.stop();
    binding.resolve();
    await retry;
    expect(spawn).not.toHaveBeenCalled();
    expect(manager["apiReady"]).toBe(false);
    expect(manager.currentStatus().phase).toBe("stopped");
  });
});

function configuration(): DesktopConfigurationView {
  return {
    version: 2,
    configured: true,
    voiceEnabled: true,
    launchAtLogin: false,
    models: { text: [{ id: "text-default", name: "Test", baseUrl: "https://example.com/v1", model: "test-model" }], voice: [{ id: "voice-default", name: "Voice", provider: "auto" }] },
    modelAssignments: { chat: "text-default", reasoning: "text-default", speech: "voice-default" },
    imageEditing: { openaiCompatibleBaseUrl: null },
    homeAssistant: { enabled: false, url: "http://127.0.0.1:8123" },
    perception: {
      operationPerceptionEnabled: false,
      screenPerceptionEnabled: false,
      audioPerceptionEnabled: false,
    },
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
  };
}

function fakeProcessController(
  processes: ProcessSnapshot[],
): RuntimeProcessController & { stopped: number[] } {
  const stopped: number[] = [];
  return {
    stopped,
    list: async () => processes,
    stop: async (pid) => {
      stopped.push(pid);
    },
  };
}

describe("live API configuration", () => {
  it("keeps API/voice processes and browser identity on success and retries failures without restarting", async () => {
    const { manager } = await logTestManager();
    await manager["runtimeLogWrite"];
    const api = {} as ChildProcess;
    const voice = {} as ChildProcess;
    manager["apiProcess"] = api;
    manager["companionProcess"] = voice;
    manager["updateStatus"]("ready", "ready", true, "http://127.0.0.1:49152");
    const before = manager.currentStatus();
    const fetch = vi.spyOn(globalThis, "fetch");
    try {
      fetch.mockResolvedValueOnce(new Response("{}", { status: 500 }));
      await expect(manager.configureApi(configuration())).rejects.toThrow("再次点击保存重试");
      expect(manager.hasPendingApiConfiguration).toBe(true);
      expect(manager["apiProcess"]).toBe(api);
      expect(manager["companionProcess"]).toBe(voice);
      fetch.mockResolvedValueOnce(new Response('{"revision":1}'));
      await manager.configureApi(configuration());
      expect(manager.hasPendingApiConfiguration).toBe(false);
      expect(manager["apiProcess"]).toBe(api);
      expect(manager["companionProcess"]).toBe(voice);
      expect(manager.currentStatus()).toEqual({ ...before, microphoneEnabled: true, playbackEnabled: true });
      expect(fetch.mock.calls[1]?.[0]).toBe("http://127.0.0.1:49152/internal/runtime/configuration");
      expect(JSON.parse(String(fetch.mock.calls[1]?.[1]?.body))).toMatchObject({
        models: { chat: { model: "test-model" }, reasoning: { model: "test-model" } },
      });
    } finally { fetch.mockRestore(); }
  });
});

describe("sharedAsrSocketPath", () => {
  it("keeps the socket beside the data when the path is short", () => {
    expect(sharedAsrSocketPath("/Users/a/data", "/tmp")).toBe(path.join("/Users/a/data", "shared-asr.sock"));
  });

  it("moves a too-long socket path to a short, stable per-profile name", () => {
    const longRoot = `/Users/${"x".repeat(90)}/data`;
    const socket = sharedAsrSocketPath(longRoot, "/var/folders/ab/T");
    expect(path.dirname(socket)).toBe(path.normalize("/var/folders/ab/T"));
    expect(path.basename(socket)).toMatch(/^fish-memory-asr-[0-9a-f]{12}\.sock$/u);
    expect(sharedAsrSocketPath(longRoot, "/var/folders/ab/T")).toBe(socket);
    expect(sharedAsrSocketPath(`${longRoot}2`, "/var/folders/ab/T")).not.toBe(socket);
  });
});

describe("embeddedDatabaseName", () => {
  it("gives each data directory its own stable instance", async () => {
    const { embeddedDatabaseName } = await import("./runtime-manager");
    const trial = embeddedDatabaseName("/Users/me/Library/Application Support/tulpa/data");
    expect(trial).toMatch(/^tulpa-[0-9a-f]{10}$/);
    expect(embeddedDatabaseName("/Users/me/Library/Application Support/tulpa/data")).toBe(trial);
    expect(embeddedDatabaseName("/tmp/test-profile/data")).not.toBe(trial);
  });
});
