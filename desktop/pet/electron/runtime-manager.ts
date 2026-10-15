import { createCharacter, type PendingCharacterStore } from "./character-profile";
import type { PendingCharacter } from "../src/shared/contracts";
import { spawn, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import type { PerceptionManagerStatus } from "./perception-manager";
import { perceptionStatusDiagnostic } from "./perception-status-diagnostic";
import { access, appendFile, mkdir, open, stat, writeFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";

import type {
  AudioDevice,
  DesktopConfiguration,
  DesktopConfigurationView,
  DesktopRuntimeStatus,
} from "../src/shared/contracts";
import { AppConfigStore, type SecretValues } from "./app-config";
import { cloudVoiceKey, voiceModel } from "./model-configuration";
import { clearModelStatus } from "./model-readiness";
import { supportsSystemAudio } from "./system-audio-support";
import { experimentalConfiguration, runtimeApiConfiguration, webSearchOn } from "./runtime-config-impact";
import { attachProcessJob, stopProcessTree } from "./platform/process-tree";
import { frozenRuntimeExecutable, runtimeAssetsRoot } from "./platform/runtime-paths";
import { OwnerVoiceManager, runOwnerVoiceCommand } from "./owner-voice-manager";
import {
  RuntimeProcessRegistry,
  type RuntimeProcessController,
  type RuntimeProcessRole,
} from "./runtime-process-registry";

interface RuntimeManagerOptions {
  configStore: Pick<AppConfigStore, "load" | "save" | "runtimeSecrets">;
  userDataPath: string;
  projectRoot: string;
  resourcesPath: string;
  packaged: boolean;
  authenticateBrowser: (apiUrl: string, apiKey: string) => Promise<void>;
  processController?: RuntimeProcessController;
  /** The character created during onboarding, written once the API is up. */
  pendingCharacter?: Pick<PendingCharacterStore, "pending" | "clear">;
  onCharacterCreated?: (characterId: string, character: PendingCharacter) => Promise<void>;
}

export interface CompanionSpeechActivityEvent {
  active: boolean;
  reason: "speech_start" | "speech_end" | "companion_stopped";
  runId: string | null;
  speechGeneration: number | null;
}

interface CompanionSpeechTraceEvent {
  active: boolean;
  runId: string;
  speechGeneration: number;
}

const RUNTIME_LOG_MAX_BYTES = 8 * 1024 * 1024;
const RUNTIME_LOG_RETAIN_BYTES = 4 * 1024 * 1024;

export class RuntimeManager extends EventEmitter {
  private apiProcess: ChildProcess | null = null;
  /** A process and URL exist before /ready; voice failures must not erase API readiness. */
  private apiReady = false;
  private modelPreparationRetry: Promise<void> | null = null;
  private companionProcess: ChildProcess | null = null;
  /**
   * The Companion now starts alongside the API so its models load meanwhile.
   * A readiness it reports before the API is up is kept here until then.
   */
  private earlyCompanionReadiness: "bound" | "unbound" | null = null;
  private environmentAsrProcess: ChildProcess | null = null;
  private stopping = false;
  private readonly expectedStops = new WeakSet<ChildProcess>();
  private readonly handledFailures = new WeakSet<ChildProcess>();
  private failureTimes: number[] = [];
  private lifecycleVersion = 0;
  private operation: Promise<void> = Promise.resolve();
  private processRegistryReady = false;
  private processRegistryPreparation: Promise<void> | null = null;
  private readonly processRegistry: RuntimeProcessRegistry;
  readonly ownerVoice: OwnerVoiceManager;
  private configurationBusy = false;
  private apiConfigurationPending = false;
  private ownerAudioSuspended = false;
  private microphoneLoan: (() => Promise<void>) | null = null;
  private shutdownRequested = false;
  private companionVoiceTraceRemainder = "";
  private activeCompanionSpeech: CompanionSpeechTraceEvent | null = null;
  private latestCompanionSpeechTrace: CompanionSpeechTraceEvent | null = null;
  private lastPerceptionDiagnostic: string | null = null;
  private readonly runtimeLogPath: string;
  private runtimeLogWrite: Promise<void>;
  private runtimeLogBytes: number | null = null;
  private status: DesktopRuntimeStatus = {
    phase: "stopped",
    detail: "尚未启动",
    apiUrl: null,
    voiceEnabled: true,
  };

  constructor(private readonly options: RuntimeManagerOptions) {
    super();
    this.runtimeLogPath = path.join(options.userDataPath, "logs", "runtime.log");
    this.runtimeLogWrite = prepareRuntimeLog(this.runtimeLogPath)
      .then((bytes) => {
        this.runtimeLogBytes = bytes;
      })
      .catch(() => undefined);
    const executable = frozenRuntimeExecutable(options);
    this.processRegistry = new RuntimeProcessRegistry(
      path.join(options.userDataPath, "runtime", "processes.json"),
      // Development profiles share one binary, so only stop recorded children.
      options.packaged ? executable : null,
      options.processController,
    );
    this.ownerVoice = new OwnerVoiceManager({
      configuration: () => this.options.configStore.load(),
      exclusive: (operation) => this.enqueueValue(operation),
      command: async (action, configuration, signal, onProgress, createdAt) => {
        await this.prepareProcessRegistry();
        const env = await buildRuntimeEnvironment(
          configuration, await this.options.configStore.runtimeSecrets(),
          this.options.userDataPath, this.status.apiUrl, liveKitAudioHelperExecutable(this.options), bundledModelsDirectory(this.options),
        );
        signal.throwIfAborted();
        const child = this.spawnRole("owner-voice", env, [
          "--action", action, "--directory", path.join(this.options.userDataPath, "data", "owner-voice"),
          "--duration", "15",
          ...(createdAt ? ["--created-at", createdAt] : []),
        ]);
        return runOwnerVoiceCommand(child, action, signal, onProgress);
      },
      suspendCompanion: () => this.suspendCompanionAudio("主人声纹处理中，语音伴侣暂时暂停"),
      disableRecognition: async () => {
        const { secrets: _flags, ...configuration } = await this.options.configStore.load();
        if (configuration.audio.ownerRecognitionEnabled) {
          await this.options.configStore.save({
            ...configuration, audio: { ...configuration.audio, ownerRecognitionEnabled: false },
          });
        }
      },
    });
  }

  currentStatus(): DesktopRuntimeStatus {
    return { ...this.status };
  }

  async codexNotificationStatus(): Promise<{ enabled: boolean; error: string | null; pending?: number }> {
    if (!this.status.apiUrl || !this.apiProcess) return { enabled: false, error: "请先启动宠物服务" };
    try {
      const secrets = await this.options.configStore.runtimeSecrets();
      const response = await fetch(`${this.status.apiUrl}/internal/runtime/codex-notifications`, {
        headers: { Authorization: `Bearer ${secrets.internalApiKey}` },
        signal: AbortSignal.timeout(3_000),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch {
      return { enabled: false, error: "暂时无法读取 Codex 通知状态" };
    }
  }

  get hasPendingApiConfiguration(): boolean {
    return this.apiConfigurationPending;
  }

  async configureApi(configuration: DesktopConfigurationView): Promise<void> {
    const apiUrl = this.status.apiUrl;
    if (!apiUrl || !this.apiProcess) return;
    const secrets = await this.options.configStore.runtimeSecrets();
    this.apiConfigurationPending = true;
    try {
      const response = await fetch(`${apiUrl}/internal/runtime/configuration`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${secrets.internalApiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(runtimeApiConfiguration(configuration, secrets)),
        signal: AbortSignal.timeout(5_000),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.apiConfigurationPending = false;
      this.status = { ...this.status, microphoneEnabled: configuration.microphoneEnabled ?? configuration.voiceEnabled, playbackEnabled: configuration.playbackEnabled ?? configuration.voiceEnabled };
      this.emit("status", this.currentStatus());
    } catch {
      throw new Error("设置已保存，但尚未应用到本地服务，请再次点击保存重试。");
    }
  }

  recordPerceptionStatus(status: PerceptionManagerStatus): void {
    const diagnostic = perceptionStatusDiagnostic(status);
    if (diagnostic === this.lastPerceptionDiagnostic) return;
    this.lastPerceptionDiagnostic = diagnostic;
    this.forwardLog("perception", diagnostic);
  }

  start(): Promise<void> {
    this.ownerVoice.assertIdle();
    this.shutdownRequested = false;
    const version = this.lifecycleVersion;
    return this.enqueue(() => this.startNow(version));
  }

  restart(): Promise<void> {
    this.ownerVoice.assertIdle();
    return this.enqueue(async () => {
      const version = ++this.lifecycleVersion;
      this.updateStatus("restarting", "正在应用新配置…");
      await this.stopNow();
      await this.startNow(version);
    });
  }

  async stop(): Promise<void> {
    this.shutdownRequested = true;
    this.lifecycleVersion += 1;
    await this.ownerVoice.cancel();
    await this.stopNow();
  }

  async withConfigurationChange<T>(operation: () => Promise<T>): Promise<T> {
    this.ownerVoice.assertIdle();
    if (this.configurationBusy) throw new Error("设置正在保存，请稍候");
    this.configurationBusy = true;
    try { return await this.enqueueValue(operation); }
    finally { this.configurationBusy = false; }
  }

  enrollOwnerVoice(append = false) {
    if (this.shutdownRequested) throw new Error("App 正在退出，不能开始录入");
    if (this.configurationBusy) throw new Error("请等待设置保存完成后再录入");
    return this.ownerVoice.enroll(append);
  }

  deleteOwnerVoice(createdAt?: string) {
    if (this.shutdownRequested) throw new Error("App 正在退出");
    if (this.configurationBusy) throw new Error("请等待设置保存完成后再删除声纹");
    return this.ownerVoice.delete(createdAt);
  }

  validateOwnerVoiceConfiguration(configuration: DesktopConfiguration): Promise<void> {
    return this.ownerVoice.validate(configuration);
  }

  /**
   * The companion holds the microphone with voice processing, which leaves any
   * other capture in the app silent. The mouth test borrows it the way owner
   * voice enrollment does: the companion stops and comes back afterwards.
   */
  borrowMicrophone(): Promise<void> {
    this.ownerVoice.assertIdle();
    return this.enqueue(async () => {
      if (this.microphoneLoan || !this.companionProcess) return;
      this.microphoneLoan = await this.suspendCompanionAudio("正在试嘴型，暂时不听你说话");
    });
  }

  returnMicrophone(): Promise<void> {
    return this.enqueue(async () => {
      const restore = this.microphoneLoan;
      this.microphoneLoan = null;
      if (restore) await restore();
    });
  }

  private async suspendCompanionAudio(detail: string): Promise<() => Promise<void>> {
    const previous = this.currentStatus();
    const version = this.lifecycleVersion;
    const shouldResume = Boolean(this.apiProcess && previous.apiUrl && previous.voiceEnabled);
    const companion = this.companionProcess;
    this.ownerAudioSuspended = true;
    if (companion) this.expectedStops.add(companion);
    this.companionProcess = null;
    this.resetCompanionSpeechActivity();
    const restore = async () => {
      this.ownerAudioSuspended = false;
      if (this.shutdownRequested || version !== this.lifecycleVersion) return;
      if (shouldResume) {
        if (!this.apiProcess || !this.status.apiUrl) throw new Error("本地服务已停止，语音需随服务重启恢复");
        await this.restartCompanionNow(await this.options.configStore.load());
      } else if (this.apiProcess) {
        this.updateStatus("ready", "桌宠已就绪，语音已关闭", false);
      }
    };
    try {
      await stopProcessTree(companion);
      await this.processRegistry.stopRole("companion");
      // Keep the authenticated API and its chat URL available throughout enrollment.
      if (this.apiProcess) this.updateStatus("ready", detail, previous.voiceEnabled);
      return restore;
    } catch (error) { await restore(); throw error; }
  }

  setVoiceEnabled(enabled: boolean): Promise<void> {
    this.ownerVoice.assertIdle();
    if (this.configurationBusy) throw new Error("设置正在保存，请稍候");
    return this.enqueue(async () => {
      let configuration = await this.options.configStore.load();
      if (enabled !== configuration.voiceEnabled) {
        const { secrets: _secretFlags, ...persisted } = configuration;
        await this.options.configStore.save({ ...persisted, voiceEnabled: enabled });
        configuration = await this.options.configStore.load();
      } else if (enabled && this.companionProcess) {
        return;
      }
      await this.restartCompanionNow(configuration);
    });
  }

  setMicrophoneEnabled(enabled: boolean): Promise<void> {
    return this.setAudioEnabled("microphone", enabled);
  }

  setPlaybackEnabled(enabled: boolean): Promise<void> {
    return this.setAudioEnabled("playback", enabled);
  }

  private setAudioEnabled(kind: "microphone" | "playback", enabled: boolean): Promise<void> {
    this.ownerVoice.assertIdle();
    if (this.configurationBusy) throw new Error("设置正在保存，请稍候");
    return this.enqueue(async () => {
      const previous = await this.options.configStore.load();
      const { secrets: _flags, ...persisted } = previous;
      const microphoneEnabled = kind === "microphone" ? enabled : previous.microphoneEnabled ?? previous.voiceEnabled;
      const playbackEnabled = kind === "playback" ? enabled : previous.playbackEnabled ?? previous.voiceEnabled;
      await this.options.configStore.save({ ...persisted, microphoneEnabled, playbackEnabled,
        voiceEnabled: microphoneEnabled || playbackEnabled });
      const configuration = await this.options.configStore.load();
      if (this.status.apiUrl) {
        const secrets = await this.options.configStore.runtimeSecrets();
        const response = await fetch(`${this.status.apiUrl}/internal/runtime/${kind}`, {
          method: "PUT", headers: { Authorization: `Bearer ${secrets.internalApiKey}`, "Content-Type": "application/json" },
          body: JSON.stringify({ enabled }), signal: AbortSignal.timeout(5_000),
        });
        if (!response.ok) throw new Error("音频开关已保存，但尚未应用，请重试。");
      }
      if (previous.voiceEnabled !== configuration.voiceEnabled || !this.companionProcess) {
        await this.restartCompanionNow(configuration);
      }
      this.status = { ...this.status, microphoneEnabled, playbackEnabled };
      this.emit("status", this.currentStatus());
    });
  }

  restartCompanion(): Promise<void> {
    this.ownerVoice.assertIdle();
    return this.enqueue(async () => {
      await this.restartCompanionNow(await this.options.configStore.load());
    });
  }

  /** Retry the failed layer without changing the user's audio switches. */
  retryModelPreparation(): Promise<void> {
    if (this.shutdownRequested) return Promise.resolve();
    if (this.modelPreparationRetry) return this.modelPreparationRetry;
    this.ownerVoice.assertIdle();
    const requestedVersion = this.lifecycleVersion;
    const operation = this.enqueue(async () => {
      if (this.shutdownRequested || requestedVersion !== this.lifecycleVersion) return;
      this.ownerVoice.assertIdle();
      await clearModelStatus(modelStatusDirectory(this.options.userDataPath));
      if (this.shutdownRequested || requestedVersion !== this.lifecycleVersion) return;
      this.failureTimes = [];
      if (!this.apiReady || !this.apiProcess || this.apiProcess.exitCode != null || !this.status.apiUrl) {
        const version = ++this.lifecycleVersion;
        this.updateStatus("restarting", "正在重新准备本地服务…");
        await this.stopNow();
        if (this.shutdownRequested || version !== this.lifecycleVersion) return;
        await this.startNow(version);
        return;
      }

      const configuration = await this.options.configStore.load();
      if (this.shutdownRequested || requestedVersion !== this.lifecycleVersion) return;
      // The main window revokes its API authorization on a global failure,
      // including when only Companion exhausted its automatic restarts.
      if (!["ready", "starting-companion"].includes(this.status.phase)) {
        const secrets = await this.options.configStore.runtimeSecrets();
        if (this.shutdownRequested || requestedVersion !== this.lifecycleVersion || !this.status.apiUrl) return;
        await this.options.authenticateBrowser(this.status.apiUrl, secrets.internalApiKey);
        if (this.shutdownRequested || requestedVersion !== this.lifecycleVersion) return;
      }
      await this.restartCompanionNow(configuration);
    });
    const retry = operation.finally(() => {
      if (this.modelPreparationRetry === retry) this.modelPreparationRetry = null;
    });
    this.modelPreparationRetry = retry;
    return retry;
  }

  async listAudioDevices(): Promise<AudioDevice[]> {
    this.ownerVoice.assertIdle();
    const configuration = await this.options.configStore.load();
    const secrets = await this.options.configStore.runtimeSecrets();
    const env = await buildRuntimeEnvironment(
      configuration,
      secrets,
      this.options.userDataPath,
      this.status.apiUrl,
      liveKitAudioHelperExecutable(this.options), bundledModelsDirectory(this.options),
    );
    return new Promise((resolve, reject) => {
      const child = this.spawnRole("audio-devices", env, ["--log-level", "ERROR"]);
      let stdout = "";
      let stderr = "";
      child.stdout?.on("data", (value) => (stdout += value.toString()));
      child.stderr?.on("data", (value) => (stderr += value.toString()));
      child.once("error", reject);
      child.once("exit", (code) => {
        if (code !== 0) {
          reject(new Error(stderr.trim() || "无法读取音频设备"));
          return;
        }
        try {
          resolve(JSON.parse(stdout) as AudioDevice[]);
        } catch {
          reject(new Error("音频设备返回格式不正确"));
        }
      });
    });
  }

  private async startNow(version: number): Promise<void> {
    if (this.shutdownRequested || version !== this.lifecycleVersion) return;
    if (this.apiProcess) return;
    await this.prepareProcessRegistry();
    const configuration = await this.options.configStore.load();
    if (!configuration.configured) return;
    const secrets = await this.options.configStore.runtimeSecrets();
    const port = await findFreePort();
    const apiUrl = `http://127.0.0.1:${port}`;
    const env = await buildRuntimeEnvironment(
      configuration,
      secrets,
      this.options.userDataPath,
      apiUrl,
      liveKitAudioHelperExecutable(this.options), bundledModelsDirectory(this.options),
    );
    if (this.shutdownRequested || version !== this.lifecycleVersion) return;
    this.status = { ...this.status, microphoneEnabled: configuration.microphoneEnabled ?? configuration.voiceEnabled, playbackEnabled: configuration.playbackEnabled ?? configuration.voiceEnabled };
    env.FISH_MEMORY_PORT = String(port);
    env.FISH_PLUGIN_ROOT = path.join(this.options.userDataPath, "plugins");
    if (frozenRuntimeExecutable(this.options)) env.FISH_PLUGIN_ASSETS = path.join(runtimeAssetsRoot(this.options), "fish-plugin-tools");
    this.stopping = false;
    this.updateStatus("starting-api", "正在启动本地记忆服务…", configuration.voiceEnabled, apiUrl);
    this.apiReady = false;
    try {
      this.apiProcess = this.spawnRole("api", env);
      // Voice models load while the API starts; the Companion retries until the API answers.
      this.earlyCompanionReadiness = null;
      if (configuration.voiceEnabled) this.companionProcess = this.spawnRole("companion", env);
      await waitUntilReady(apiUrl, secrets.internalApiKey, this.apiProcess);
      if (version !== this.lifecycleVersion) return;
      this.apiConfigurationPending = false;
      await this.bindAssistant(apiUrl, secrets.internalApiKey);
      if (version !== this.lifecycleVersion) return;
      await this.options.authenticateBrowser(apiUrl, secrets.internalApiKey);
      if (version !== this.lifecycleVersion) return;
      this.apiReady = true;
      if (configuration.voiceEnabled) {
        const early = this.earlyCompanionReadiness;
        this.earlyCompanionReadiness = null;
        if (early === "bound") this.updateStatus("ready", "桌宠和语音已就绪", true, apiUrl);
        else if (early === "unbound") this.updateStatus("ready", "桌宠已就绪；首次创建 Fish 后语音会自动开启", true, apiUrl);
        else this.updateStatus("starting-companion", "正在准备本地语音模型…", true, apiUrl);
      } else {
        this.startEnvironmentAsr(configuration, env);
        this.updateStatus("ready", "桌宠已就绪，语音已关闭", false, apiUrl);
      }
    } catch (error) {
      await this.stopNow();
      if (version !== this.lifecycleVersion) return;
      this.updateStatus("failed", error instanceof Error ? error.message : String(error));
      throw error;
    }
  }

  private async stopNow(): Promise<void> {
    this.stopping = true;
    this.apiReady = false;
    const companion = this.companionProcess;
    const api = this.apiProcess;
    if (companion) this.expectedStops.add(companion);
    if (api) this.expectedStops.add(api);
    this.companionProcess = null;
    this.apiProcess = null;
    this.resetCompanionSpeechActivity();
    await stopProcessTree(companion);
    await this.stopEnvironmentAsr();
    await stopProcessTree(api);
    await this.processRegistry.clear();
    this.stopping = false;
    this.updateStatus("stopped", "已停止", false, null);
  }

  private async restartCompanionNow(
    configuration: DesktopConfigurationView,
  ): Promise<void> {
    const version = this.lifecycleVersion;
    if (this.shutdownRequested) return;
    this.status = { ...this.status, microphoneEnabled: configuration.microphoneEnabled ?? configuration.voiceEnabled,
      playbackEnabled: configuration.playbackEnabled ?? configuration.voiceEnabled };
    const companion = this.companionProcess;
    if (companion) this.expectedStops.add(companion);
    this.companionProcess = null;
    this.resetCompanionSpeechActivity();
    if (!configuration.voiceEnabled) {
      this.updateStatus(
        this.status.phase,
        "正在关闭语音伴侣…",
        false,
      );
    }
    await stopProcessTree(companion);
    await this.stopEnvironmentAsr();
    const orphaned = await this.processRegistry.stopRole("companion");
    if (this.shutdownRequested || version !== this.lifecycleVersion) return;
    if (orphaned.length && !orphaned.includes(companion?.pid ?? -1)) {
      this.emit("log", {
        role: "runtime",
        line: `关闭语音时额外清理了 ${orphaned.length} 个残留 Companion 进程`,
      });
    }

    if (!this.apiProcess || !this.status.apiUrl) {
      this.updateStatus(
        this.status.phase,
        this.status.detail,
        configuration.voiceEnabled,
      );
      return;
    }

    const secrets = await this.options.configStore.runtimeSecrets();
    const env = await buildRuntimeEnvironment(
      configuration,
      secrets,
      this.options.userDataPath,
      this.status.apiUrl,
      liveKitAudioHelperExecutable(this.options), bundledModelsDirectory(this.options),
    );
    if (this.shutdownRequested || version !== this.lifecycleVersion) return;
    if (!configuration.voiceEnabled) {
      this.startEnvironmentAsr(configuration, env);
      this.updateStatus("ready", "桌宠已就绪，语音已关闭", false);
      return;
    }

    this.updateStatus("starting-companion", "正在应用语音配置…", true);
    await this.bindAssistant(this.status.apiUrl, secrets.internalApiKey);
    if (this.shutdownRequested || version !== this.lifecycleVersion) return;
    this.companionProcess = this.spawnRole("companion", env);
  }

  private async bindAssistant(apiUrl: string, apiKey: string): Promise<void> {
    const character = await this.options.pendingCharacter?.pending();
    if (!character) {
      await ensureAssistantBinding(apiUrl, apiKey);
      return;
    }
    const characterId = await createCharacter(apiUrl, apiKey, character);
    // Clear before the callback so a failing callback cannot create a duplicate
    // character on the next start.
    await this.options.pendingCharacter?.clear();
    await this.options.onCharacterCreated?.(characterId, character);
  }

  private startEnvironmentAsr(
    configuration: DesktopConfigurationView,
    env: NodeJS.ProcessEnv,
  ): void {
    if (configuration.perception.audioPerceptionEnabled && supportsSystemAudio()) {
      // This role serves the same socket without opening a microphone, VAD,
      // wake-word or TTS stack. Its one model loads only on received audio.
      this.environmentAsrProcess = this.spawnRole("environment-asr", env);
    }
  }

  private async stopEnvironmentAsr(): Promise<void> {
    const process = this.environmentAsrProcess;
    if (process) this.expectedStops.add(process);
    this.environmentAsrProcess = null;
    await stopProcessTree(process);
    await this.processRegistry.stopRole("environment-asr");
  }

  private spawnRole(
    role: RuntimeProcessRole,
    env: NodeJS.ProcessEnv,
    extraArguments: string[] = [],
  ): ChildProcess {
    const executable = frozenRuntimeExecutable(this.options);
    const command = executable ?? "uv";
    const args = executable
      ? [role, ...extraArguments]
      : [
          "run",
          "--extra",
          "voice",
          "python",
          "-m",
          "fish_memory.desktop_runtime",
          role,
          ...extraArguments,
        ];
    const child = spawn(command, args, {
      cwd: executable ? this.options.userDataPath : this.options.projectRoot,
      env,
      detached: process.platform !== "win32",
      windowsHide: true,
      stdio: [process.platform === "win32" ? "pipe" : "ignore", "pipe", "pipe"],
    });
    attachProcessJob(child);
    this.processRegistry.track(child.pid, role, command, args);
    if (role !== "owner-voice") {
      child.stdout?.on("data", (value) => this.forwardLog(role, value.toString(), child));
      child.stderr?.on("data", (value) => this.forwardLog(role, value.toString(), child));
    } else {
      child.stderr?.on("data", () => {}); // CLI exposes only validated status JSONL to the UI.
    }
    child.once("error", (error) => {
      this.handleUnexpectedExit(role, child, error.message);
    });
    child.once("exit", (code, signal) => {
      this.processRegistry.forget(child.pid);
      const reason = code === null ? signal ?? "unknown" : `code ${code}`;
      this.handleUnexpectedExit(role, child, reason);
    });
    return child;
  }

  private handleUnexpectedExit(
    role: RuntimeProcessRole,
    child: ChildProcess,
    reason: string,
  ): void {
    if (
      role === "audio-devices" || role === "owner-voice" ||
      this.stopping ||
      this.expectedStops.has(child) ||
      this.handledFailures.has(child)
    ) return;
    this.handledFailures.add(child);
    if (role === "api" && this.apiProcess === child) {
      this.apiProcess = null;
      this.apiReady = false;
    }
    if (role === "companion" && this.companionProcess === child) {
      this.companionProcess = null;
      this.resetCompanionSpeechActivity();
    }
    if (role === "environment-asr" && this.environmentAsrProcess === child) {
      this.environmentAsrProcess = null;
    }
    const now = Date.now();
    this.failureTimes = this.failureTimes.filter((timestamp) => now - timestamp < 120_000);
    this.failureTimes.push(now);
    if (this.failureTimes.length > 3) {
      this.updateStatus("failed", `${role} 连续异常退出 (${reason})，请在设置中查看诊断`);
      return;
    }
    this.updateStatus(
      "restarting",
      `${role === "api" ? "本地服务" : "语音服务"}异常退出，正在自动恢复 (${this.failureTimes.length}/3)…`,
    );
    void this.ownerVoice.cancel().then(() => (
      role === "environment-asr" ? this.restartCompanion() : this.restart()
    )).catch((error) => {
      this.updateStatus("failed", error instanceof Error ? error.message : String(error));
    });
  }

  private forwardLog(
    role: string,
    value: string,
    sourceProcess?: ChildProcess,
  ): void {
    const currentCompanionLog = role !== "companion"
      || sourceProcess === undefined
      || sourceProcess === this.companionProcess;
    if (
      role === "companion"
      && currentCompanionLog
      && !this.stopping
      && !this.ownerAudioSuspended
      && ["starting-companion", "ready"].includes(this.status.phase)
    ) {
      this.observeCompanionVoiceTrace(value);
    } else if (role === "companion" && currentCompanionLog) {
      this.companionVoiceTraceRemainder = "";
    }
    const records: string[] = [];
    for (const line of value.split(/\r?\n/).filter(Boolean)) {
      const record = `${new Date().toISOString()} [${role}] ${line}\n`;
      records.push(record);
      this.emit("log", { role, line });
      // Started alongside the API: hold its readiness until the API is up.
      if (role === "companion" && currentCompanionLog && !this.stopping && this.status.phase === "starting-api") {
        const early = companionReadinessFromLog(line);
        if (early) this.earlyCompanionReadiness = early;
        continue;
      }
      // A stopped/replaced Companion can still flush stdout. Its late logs must
      // not announce an authenticated API before the next startup reaches it.
      if (role === "companion" && (
        !currentCompanionLog
        || this.stopping
        || this.ownerAudioSuspended
        || !["starting-companion", "ready"].includes(this.status.phase)
      )) continue;
      const companionReadiness = role === "companion"
        ? companionReadinessFromLog(line)
        : null;
      const companionStartupFailure = role === "companion"
        ? companionStartupFailureFromLog(line)
        : null;
      if (companionReadiness === "bound") {
        this.updateStatus("ready", "桌宠和语音已就绪", true);
      } else if (companionReadiness === "unbound") {
        this.updateStatus(
          "ready",
          "桌宠已就绪；首次创建 Fish 后语音会自动开启",
          true,
        );
      } else if (companionStartupFailure) {
        // Companion retries its audio stream in-process; the authenticated API
        // is still healthy. A global failure would clear the chat credentials
        // without startNow running again to restore them after audio recovers.
        this.updateStatus("starting-companion", companionStartupFailure.text, true, this.status.apiUrl, companionStartupFailure);
      } else if (role === "companion" && line.includes("模型准备中")) {
        this.updateStatus("starting-companion", line, true);
      } else if (
        line.includes("Embedding 模型下载进度") ||
        line.includes("Local embedding model")
      ) {
        this.updateStatus("starting-api", line, this.status.voiceEnabled);
      }
    }
    if (records.length) this.persistRuntimeLog(records.join(""));
  }

  private persistRuntimeLog(value: string): void {
    const write = this.runtimeLogWrite.then(async () => {
      if (this.runtimeLogBytes === null) {
        this.runtimeLogBytes = await prepareRuntimeLog(this.runtimeLogPath);
      }
      await appendFile(this.runtimeLogPath, value);
      this.runtimeLogBytes += Buffer.byteLength(value);
      if (this.runtimeLogBytes > RUNTIME_LOG_MAX_BYTES) {
        this.runtimeLogBytes = await retainRuntimeLogTail(
          this.runtimeLogPath,
          this.runtimeLogBytes,
        );
      }
    });
    this.runtimeLogWrite = write.catch(() => undefined);
  }

  private observeCompanionVoiceTrace(value: string): void {
    const combined = this.companionVoiceTraceRemainder + value;
    const lines = combined.split(/\r?\n/);
    this.companionVoiceTraceRemainder = lines.pop() ?? "";
    for (const line of lines) this.applyCompanionSpeechTrace(line);
    const completeTail = companionSpeechActivityFromLog(
      this.companionVoiceTraceRemainder,
    );
    if (completeTail) {
      this.companionVoiceTraceRemainder = "";
      this.applyCompanionSpeechTraceEvent(completeTail);
    } else if (this.companionVoiceTraceRemainder.length > 32_768) {
      this.companionVoiceTraceRemainder = "";
    }
  }

  private applyCompanionSpeechTrace(line: string): void {
    const event = companionSpeechActivityFromLog(line);
    if (event) this.applyCompanionSpeechTraceEvent(event);
  }

  private applyCompanionSpeechTraceEvent(event: CompanionSpeechTraceEvent): void {
    const active = this.activeCompanionSpeech;
    const latest = this.latestCompanionSpeechTrace;
    if (
      latest?.runId === event.runId
      && (
        event.speechGeneration < latest.speechGeneration
        || (
          event.speechGeneration === latest.speechGeneration
          && (!latest.active || latest.active === event.active)
        )
      )
    ) return;
    if (event.active) {
      if (
        active?.runId === event.runId
        && active.speechGeneration === event.speechGeneration
      ) return;
      this.activeCompanionSpeech = event;
      this.latestCompanionSpeechTrace = event;
      this.emit("companion-speech-activity", {
        ...event,
        reason: "speech_start",
      } satisfies CompanionSpeechActivityEvent);
      return;
    }
    if (
      !active
      || active.runId !== event.runId
      || active.speechGeneration !== event.speechGeneration
    ) return;
    this.activeCompanionSpeech = null;
    this.latestCompanionSpeechTrace = event;
    this.emit("companion-speech-activity", {
      ...event,
      reason: "speech_end",
    } satisfies CompanionSpeechActivityEvent);
  }

  private resetCompanionSpeechActivity(): void {
    this.companionVoiceTraceRemainder = "";
    const active = this.activeCompanionSpeech;
    this.latestCompanionSpeechTrace = null;
    if (!active) return;
    this.activeCompanionSpeech = null;
    this.emit("companion-speech-activity", {
      active: false,
      reason: "companion_stopped",
      runId: active.runId,
      speechGeneration: active.speechGeneration,
    } satisfies CompanionSpeechActivityEvent);
  }

  private updateStatus(
    phase: DesktopRuntimeStatus["phase"],
    detail: string,
    voiceEnabled = this.status.voiceEnabled,
    apiUrl: string | null = this.status.apiUrl,
    voiceIssue: CompanionVoiceIssue | null = null,
  ): void {
    // Any other status change means the audio problem is over or no longer relevant.
    this.status = { ...this.status, phase, detail, apiUrl, voiceEnabled, voiceIssue };
    this.emit("status", this.currentStatus());
  }

  private enqueue(operation: () => Promise<void>): Promise<void> {
    const next = this.operation.then(operation, operation);
    this.operation = next.catch(() => undefined);
    return next;
  }

  private enqueueValue<T>(operation: () => Promise<T>): Promise<T> {
    const next = this.operation.then(operation, operation);
    this.operation = next.then(() => undefined, () => undefined);
    return next;
  }

  private async prepareProcessRegistry(): Promise<void> {
    if (this.processRegistryReady) return;
    if (this.processRegistryPreparation) return this.processRegistryPreparation;
    const preparation = (async () => {
      const stopped = await this.processRegistry.beginSession();
      this.processRegistryReady = true;
      if (stopped.length) {
        this.emit("log", {
          role: "runtime",
          line: `已清理 ${stopped.length} 个上次异常退出后残留的后台进程`,
        });
      }
    })();
    this.processRegistryPreparation = preparation;
    try { await preparation; }
    finally { if (this.processRegistryPreparation === preparation) this.processRegistryPreparation = null; }
  }
}

/** Models shipped inside the app (wake word, VAD, speech, TTS vocoder, espeak data, Live2D). */
export function bundledModelsDirectory(options: Pick<RuntimeManagerOptions, "packaged" | "resourcesPath" | "projectRoot">): string {
  return path.join(runtimeAssetsRoot(options), "models");
}

export function liveKitAudioHelperExecutable(options: RuntimeManagerOptions): string {
  const appRoot = options.packaged
    ? path.join(options.resourcesPath, "audio")
    : runtimeAssetsRoot(options);
  return path.join(
    appRoot,
    "FishMemoryAudioHelper.app",
    "Contents",
    "MacOS",
    "fish-memory-livekit-audio-helper",
  );
}

type Fetcher = (
  input: string | URL | Request,
  init?: RequestInit,
) => Promise<Response>;

export async function ensureAssistantBinding(
  apiUrl: string,
  apiKey: string,
  fetcher: Fetcher = fetch,
): Promise<boolean> {
  const headers = { Authorization: `Bearer ${apiKey}` };
  const assistantResponse = await fetcher(`${apiUrl}/v1/fishes/assistant`, {
    headers,
  });
  await requireSuccessfulResponse(assistantResponse, "读取助手 Fish 绑定失败");
  const assistant = await assistantResponse.json() as { fishId?: unknown };
  if (typeof assistant.fishId === "string" && assistant.fishId) return true;

  const fishesResponse = await fetcher(`${apiUrl}/v1/fishes`, { headers });
  await requireSuccessfulResponse(fishesResponse, "读取 Fish 列表失败");
  const fishes = await fishesResponse.json() as {
    items?: Array<{ fishId?: unknown }>;
  };
  const fishId = fishes.items?.find(
    (item) => typeof item.fishId === "string" && item.fishId,
  )?.fishId;
  if (typeof fishId !== "string") return false;

  const bindResponse = await fetcher(
    `${apiUrl}/v1/fishes/${encodeURIComponent(fishId)}/assistant`,
    { method: "PUT", headers },
  );
  await requireSuccessfulResponse(bindResponse, "绑定助手 Fish 失败");
  return true;
}

export function companionReadinessFromLog(
  line: string,
): "bound" | "unbound" | null {
  if (line.includes("当前没有绑定助手 Fish")) return "unbound";
  if (line.includes("已绑定助手 Fish") && line.includes("唤醒词")) return "bound";
  return null;
}

export function companionSpeechActivityFromLog(
  line: string,
): CompanionSpeechTraceEvent | null {
  const match = line.match(
    /(?:^|\s)voice_trace event=(speech_start|speech_end) (\{.*\})\s*$/u,
  );
  if (!match) return null;
  let payload: unknown;
  try {
    payload = JSON.parse(match[2]!);
  } catch {
    return null;
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null;
  const value = payload as Record<string, unknown>;
  if (
    typeof value.runId !== "string"
    || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu
      .test(value.runId)
    || !Number.isSafeInteger(value.speechGeneration)
    || (value.speechGeneration as number) < 0
  ) return null;
  return {
    active: match[1] === "speech_start",
    runId: value.runId,
    speechGeneration: value.speechGeneration as number,
  };
}

export interface CompanionVoiceIssue {
  kind: "microphone" | "audio";
  text: string;
}

/** Audio failures the companion keeps retrying, worded for the pet's status chip. */
export function companionStartupFailureFromLog(line: string): CompanionVoiceIssue | null {
  if (line.includes("Microphone permission is not granted")) {
    return { kind: "microphone", text: "麦克风没有权限，TA 听不到你说话" };
  }
  if (line.includes("LiveKit audio helper did not deliver its first frame")) {
    return { kind: "audio", text: "麦克风没有声音进来，正在重试" };
  }
  if (
    line.includes("LiveKit audio helper is missing or not executable")
    || line.includes("Unable to start LiveKit audio helper")
    || line.includes("LiveKit audio helper failed")
  ) {
    return { kind: "audio", text: "语音没能启动，正在重试" };
  }
  return null;
}

async function requireSuccessfulResponse(
  response: Response,
  message: string,
): Promise<void> {
  if (!response.ok) throw new Error(`${message} (${response.status})`);
}

export async function buildRuntimeEnvironment(
  configuration: DesktopConfiguration,
  secrets: SecretValues & { internalApiKey: string },
  userDataPath: string,
  apiUrl: string | null,
  livekitAudioHelper?: string,
  bundledModels?: string,
): Promise<NodeJS.ProcessEnv> {
  const modelRoutes = runtimeApiConfiguration(configuration, secrets).models;
  const selectedVoice = voiceModel(configuration);
  // Characters pick 通义 or 本机 themselves; the 通义 key goes along either way.
  const voiceApiKey = cloudVoiceKey(configuration, secrets.modelApiKeys);
  const modelRoot = path.join(userDataPath, "models-runtime");
  const cacheRoot = path.join(modelRoot, "cache");
  const dataRoot = path.join(userDataPath, "data");
  const logsRoot = path.join(userDataPath, "logs");
  await Promise.all(
    [cacheRoot, modelRoot, dataRoot, logsRoot].map((directory) =>
      mkdir(directory, { recursive: true }),
    ),
  );
  const cachedEmbeddingSnapshot = await findCachedEmbeddingSnapshot(cacheRoot);
  const inheritedEnvironment = Object.fromEntries(
    Object.entries(process.env).filter(([key]) => !key.startsWith("FISH_MEMORY_")),
  );
  const env: NodeJS.ProcessEnv = {
    ...inheritedEnvironment,
    PYTHONUTF8: "1",
    PYTHONUNBUFFERED: "1",
    ...(process.platform === "win32" ? {
      FISH_MEMORY_PARENT_CONTROL: "stdin",
      TULPA_PG0_HOME: path.join(dataRoot, "pg0-runtime"),
    } : {}),
    FISH_MEMORY_DATABASE_URL: `pg0://${embeddedDatabaseName(dataRoot)}`,
    FISH_MEMORY_DATABASE_DATA_DIR: path.join(dataRoot, "postgres"),
    FISH_MEMORY_API_KEY: secrets.internalApiKey,
    FISH_MEMORY_COMPANION_MICROPHONE_ENABLED: String(configuration.microphoneEnabled ?? configuration.voiceEnabled),
    FISH_MEMORY_COMPANION_PLAYBACK_ENABLED: String(configuration.playbackEnabled ?? configuration.voiceEnabled),
    FISH_MEMORY_SPEECH_FACIAL_WAIT_FOR_READY: String(configuration.speechFacialWaitForReady ?? false),
    FISH_MEMORY_OWNER_VOICE_ENABLED: String(configuration.audio.ownerRecognitionEnabled ?? false),
    FISH_MEMORY_OWNER_VOICE_DIR: path.join(dataRoot, "owner-voice"),
    FISH_MEMORY_HOST: "127.0.0.1",
    FISH_MEMORY_DESKTOP_MODE: "true",
    FISH_MEMORY_DESKTOP_PET_ENABLED: "true",
    FISH_MEMORY_LLM_PROVIDER: "openai",
    FISH_MEMORY_LLM_API_KEY: modelRoutes.chat.apiKey,
    FISH_MEMORY_LLM_MODEL: modelRoutes.chat.model,
    FISH_MEMORY_LLM_BASE_URL: modelRoutes.chat.baseUrl,
    FISH_MEMORY_DESKTOP_MODEL_ROUTES: JSON.stringify(modelRoutes),
    FISH_MEMORY_EMBEDDING_BATCH_SIZE: "8",
    FISH_MEMORY_ANSWER_ENABLE_SEARCH: String(webSearchOn(configuration)),
    FISH_MEMORY_EXPERIMENTAL_COMPANION: JSON.stringify(experimentalConfiguration(configuration, secrets)),
    FISH_MEMORY_CODEX_NOTIFICATIONS_ENABLED: String(configuration.codexNotificationsEnabled ?? false),
    FISH_MEMORY_CODEX_NOTIFICATIONS_STATE_FILE: path.join(dataRoot, "codex-notifications.json"),
    FISH_MEMORY_TTS_PROVIDER: selectedVoice?.provider ?? "local",
    FISH_MEMORY_HOME_ASSISTANT_ENABLED: String(configuration.homeAssistant.enabled),
    FISH_MEMORY_HOME_ASSISTANT_URL: configuration.homeAssistant.url,
    FISH_MEMORY_WAKE_WORD: configuration.audio.wakeWord,
    FISH_MEMORY_ASR_NUM_THREADS: String(configuration.advanced.asrThreads),
    FISH_MEMORY_COMPANION_VAD_THRESHOLD: String(configuration.advanced.vadThreshold),
    FISH_MEMORY_COMPANION_VAD_MIN_SILENCE_MS: String(configuration.advanced.vadMinSilenceMs),
    FISH_MEMORY_COMPANION_VAD_MIN_SPEECH_MS: String(configuration.advanced.vadMinSpeechMs),
    FISH_MEMORY_COMPANION_SMART_TURN_ENABLED: String(configuration.advanced.smartTurnEnabled),
    FISH_MEMORY_COMPANION_SMART_TURN_THRESHOLD: String(configuration.advanced.smartTurnThreshold),
    FISH_MEMORY_COMPANION_AEC_DELAY_MODE: configuration.advanced.aecDelayMode,
    FISH_MEMORY_COMPANION_AEC_DELAY_MS: String(configuration.advanced.aecDelayMs),
    FISH_MEMORY_COMPANION_AUDIO_BACKEND: configuration.advanced.audioBackend,
    FISH_MEMORY_COMPANION_VOICE_DRAFT_ENABLED: String(
      configuration.advanced.voiceDraftEnabled,
    ),
    FISH_MEMORY_AUDIO_DIAGNOSTICS_ENABLED: String(configuration.advanced.diagnosticsEnabled),
    FISH_MEMORY_LLM_LOG_RETENTION_DAYS: String(configuration.llmLogRetentionDays ?? 7),
    FISH_MEMORY_AGENT_TASKS_ENABLED: String(configuration.agent.enabled),
    FISH_MEMORY_AGENT_URL: configuration.agent.url,
    FISH_MEMORY_AGENT_CREDENTIALS_FILE: path.join(userDataPath, "agent-credentials.json"),
    FISH_MEMORY_AGENT_RUN_CONCURRENCY: String(configuration.agent.runConcurrency),
    FISH_MEMORY_AGENT_RUN_POLL_INTERVAL_SECONDS: String(
      configuration.agent.runPollIntervalSeconds,
    ),
    FISH_MEMORY_AGENT_HTTP_TIMEOUT_SECONDS: String(configuration.agent.httpTimeoutSeconds),
    FISH_MEMORY_AGENT_SSE_READ_TIMEOUT_SECONDS: String(
      configuration.agent.sseReadTimeoutSeconds,
    ),
    FISH_MEMORY_AGENT_REPLY_CONTEXT_TTL_SECONDS: String(
      configuration.agent.replyContextTtlSeconds,
    ),
    FISH_MEMORY_AUDIO_DIAGNOSTICS_DIR: path.join(logsRoot, "audio-diagnostics"),
    FISH_MEMORY_KWS_MODEL_DIR: path.join(modelRoot, "kws"),
    FISH_MEMORY_ASR_MODEL_DIR: path.join(modelRoot, "asr", "sensevoice"),
    FISH_MEMORY_COMPANION_VAD_MODEL_DIR: path.join(modelRoot, "vad"),
    FISH_MEMORY_TTS_LOCAL_MODEL_DIR: path.join(modelRoot, "tts"),
    HF_HOME: path.join(cacheRoot, "huggingface"),
    TOKENIZERS_PARALLELISM: "false",
  };
  // API lifetime outlives voice toggles. Both voice states must use this same
  // endpoint, otherwise toggling off disconnects media ASR until API restart.
  env.FISH_MEMORY_SHARED_ASR_SOCKET = sharedAsrSocketPath(dataRoot);
  // Speech models: app bundle first, then the mainland mirror, then the original host.
  // Each process reports preparation progress here for the desktop to show.
  env.FISH_MEMORY_MODEL_STATUS_DIR = modelStatusDirectory(userDataPath);
  if (bundledModels) env.FISH_MEMORY_BUNDLED_MODELS_DIR = bundledModels;
  else delete env.FISH_MEMORY_BUNDLED_MODELS_DIR;
  if (configuration.advanced.audioBackend === "livekit" && livekitAudioHelper) {
    env.FISH_MEMORY_LIVEKIT_AUDIO_HELPER = livekitAudioHelper;
  }
  if (cachedEmbeddingSnapshot) {
    env.FISH_MEMORY_EMBEDDING_LOCAL_MODEL_DIR = cachedEmbeddingSnapshot;
  }
  if (apiUrl) {
    env.FISH_MEMORY_API_URL = apiUrl;
    env.FISH_MEMORY_PET_API_URL = apiUrl;
  }
  if (voiceApiKey) env.FISH_MEMORY_TTS_API_KEY = voiceApiKey;
  else delete env.FISH_MEMORY_TTS_API_KEY;
  if (secrets.homeAssistantToken) {
    env.FISH_MEMORY_HOME_ASSISTANT_TOKEN = secrets.homeAssistantToken;
  } else {
    delete env.FISH_MEMORY_HOME_ASSISTANT_TOKEN;
  }
  if (secrets.agentApiKey) {
    env.FISH_MEMORY_AGENT_API_KEY = secrets.agentApiKey;
  } else {
    delete env.FISH_MEMORY_AGENT_API_KEY;
  }
  if (configuration.audio.inputDevice) {
    env.FISH_MEMORY_COMPANION_INPUT_DEVICE = configuration.audio.inputDevice;
  } else {
    delete env.FISH_MEMORY_COMPANION_INPUT_DEVICE;
  }
  if (configuration.audio.outputDevice) {
    env.FISH_MEMORY_COMPANION_OUTPUT_DEVICE = configuration.audio.outputDevice;
  } else {
    delete env.FISH_MEMORY_COMPANION_OUTPUT_DEVICE;
  }
  // Network follows whatever tulpa was started with. Runtimes talk to each
  // other on loopback; an inherited proxy may answer those with 502.
  const bypass = new Set(
    [env.NO_PROXY, env.no_proxy, LOOPBACK_NO_PROXY]
      .flatMap((value) => value?.split(",") ?? [])
      .map((host) => host.trim())
      .filter(Boolean),
  );
  env.NO_PROXY = [...bypass].join(",");
  env.no_proxy = env.NO_PROXY;
  return env;
}

const EMBEDDING_SNAPSHOT_FILES = [
  "onnx/model_qint8_avx512_vnni.onnx",
  "config.json",
  "sentencepiece.bpe.model",
  "special_tokens_map.json",
  "tokenizer.json",
  "tokenizer_config.json",
] as const;
const EMBEDDING_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3";

async function findCachedEmbeddingSnapshot(cacheRoot: string): Promise<string | null> {
  const snapshot = path.join(
    cacheRoot,
    "huggingface",
    "hub",
    "models--intfloat--multilingual-e5-small",
    "snapshots",
    EMBEDDING_MODEL_REVISION,
  );
  try {
    await Promise.all(
      EMBEDDING_SNAPSHOT_FILES.map((file) => access(path.join(snapshot, file))),
    );
    return snapshot;
  } catch {
    // Ignore an incomplete snapshot left by an interrupted first download.
    return null;
  }
}

export async function prepareRuntimeLog(logPath: string): Promise<number> {
  await mkdir(path.dirname(logPath), { recursive: true });
  let size: number;
  try {
    size = (await stat(logPath)).size;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return 0;
    throw error;
  }
  if (size <= RUNTIME_LOG_MAX_BYTES) return size;
  return retainRuntimeLogTail(logPath, size);
}

async function retainRuntimeLogTail(logPath: string, size: number): Promise<number> {
  const retain = Math.min(size, RUNTIME_LOG_RETAIN_BYTES);
  const buffer = Buffer.allocUnsafe(retain);
  const handle = await open(logPath, "r");
  let bytesRead = 0;
  try {
    while (bytesRead < retain) {
      const result = await handle.read(
        buffer,
        bytesRead,
        retain - bytesRead,
        size - retain + bytesRead,
      );
      if (result.bytesRead === 0) break;
      bytesRead += result.bytesRead;
    }
  } finally {
    await handle.close();
  }
  const captured = buffer.subarray(0, bytesRead);
  const firstLineBreak = captured.indexOf(0x0a);
  const tail = firstLineBreak >= 0
    ? captured.subarray(firstLineBreak + 1)
    : captured;
  await writeFile(logPath, tail);
  return tail.byteLength;
}

async function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        reject(new Error("无法分配本地服务端口"));
        return;
      }
      server.close((error) => (error ? reject(error) : resolve(address.port)));
    });
  });
}

async function waitUntilReady(
  apiUrl: string,
  apiKey: string,
  process: ChildProcess,
): Promise<void> {
  const deadline = Date.now() + 10 * 60_000;
  while (Date.now() < deadline) {
    if (process.exitCode !== null) {
      throw new Error(`本地记忆服务提前退出 (${process.exitCode})`);
    }
    try {
      const response = await fetch(`${apiUrl}/ready`, {
        headers: { Authorization: `Bearer ${apiKey}` },
      });
      if (response.ok) return;
    } catch {
      // Startup includes PostgreSQL and embedding initialization.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("本地记忆服务启动超时");
}

const LOOPBACK_NO_PROXY = "127.0.0.1,localhost,::1";

/** The runtime refuses socket paths over 100 bytes (sun_path is 104 on macOS). */
const MAX_SOCKET_PATH_BYTES = 100;

/**
 * Keeps the socket next to the data it serves, unless a long home or profile
 * path would push it past the limit and stop the runtime from starting; then a
 * short per-profile name in the temp directory is used instead.
 */
export function sharedAsrSocketPath(dataRoot: string, temporaryDirectory = os.tmpdir()): string {
  const beside = path.join(dataRoot, "shared-asr.sock");
  if (Buffer.byteLength(beside) <= MAX_SOCKET_PATH_BYTES) return beside;
  const id = createHash("sha256").update(dataRoot).digest("hex").slice(0, 12);
  return path.join(temporaryDirectory, `fish-memory-asr-${id}.sock`);
}

export function modelStatusDirectory(userDataPath: string): string {
  return path.join(userDataPath, "data", "model-status");
}

/**
 * The embedded database's instance name. Each data directory gets its own, so
 * two profiles (a trial copy and a test profile) never share one registration.
 */
export function embeddedDatabaseName(dataRoot: string): string {
  return `tulpa-${createHash("sha256").update(path.resolve(dataRoot)).digest("hex").slice(0, 10)}`;
}
