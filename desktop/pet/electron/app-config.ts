import { randomBytes } from "node:crypto";
import { mkdir, readFile } from "node:fs/promises";
import { totalmem } from "node:os";
import path from "node:path";
import { platformAudioBackend } from "./audio-backend";
import { readPrivateText, writePrivateText } from "./private-file";

import type {
  DesktopConfiguration,
  DesktopConfigurationInput,
  DesktopConfigurationView,
} from "../src/shared/contracts";
import {
  petTextureMaximumOrDefault,
  suggestedPetTextureMaximum,
} from "../src/shared/texture-quality";
import { BAILIAN_WORKSPACE_PATTERN } from "../src/shared/decision-model";

import {
  DEFAULT_MODEL_ASSIGNMENTS, DEFAULT_MODEL_CATALOG,
  textModel, validateAssignments, validateModels, voiceModel,
} from "./model-configuration";

export interface SecretValues {
  internalApiKey?: string;
  modelApiKeys?: Record<string, string>;
  homeAssistantToken?: string;
  openaiCompatibleImageApiKey?: string;
  falApiKey?: string;
  agentApiKey?: string;
  jevApiKey?: string;
}

const SECRET_VALUE_KEYS = [
  "internalApiKey",
  "homeAssistantToken",
  "openaiCompatibleImageApiKey",
  "falApiKey",
  "agentApiKey",
  "jevApiKey",
] as const satisfies readonly (keyof SecretValues)[];

interface StoredSecrets {
  version: 2;
  values: SecretValues;
}

export const DEFAULT_GATEWAY_PORT = 17821;

const DEFAULT_CONFIGURATION: DesktopConfiguration = {
  version: 2,
  configured: false,
  llmLogRetentionDays: 7,
  voiceEnabled: true,
  microphoneEnabled: true,
  playbackEnabled: true,
  speechFacialWaitForReady: false,
  proactiveCompanionEnabled: false,
  experimentalCompanion: { model: "jev-1.13.0", wakeSeconds: 5, thoughtsEnabled: true, thoughtSeconds: 2, speechFacialEnabled: true },
  codexNotificationsEnabled: false,
  webSearchEnabled: true,
  gateway: { enabled: false, port: DEFAULT_GATEWAY_PORT },
  launchAtLogin: false,
  models: DEFAULT_MODEL_CATALOG,
  modelAssignments: DEFAULT_MODEL_ASSIGNMENTS,
  imageEditing: {
    openaiCompatibleBaseUrl: null,
  },
  homeAssistant: {
    enabled: false,
    url: "http://127.0.0.1:8123",
  },
  perception: {
    operationPerceptionEnabled: false,
    screenPerceptionEnabled: false,
    audioPerceptionEnabled: false,
    // TA does not see its own windows: it would narrate them and not know itself.
    excludeFishMemoryFromScreenCapture: true,
    innerLifeEnabled: true,
  },
  rendering: { maxTextureDimension: suggestedPetTextureMaximum(totalmem()) },
  audio: {
    wakeWord: "小星",
    inputDevice: null,
    outputDevice: null,
    ownerRecognitionEnabled: false,
  },
  advanced: {
    audioBackend: platformAudioBackend(),
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
    voiceDraftEnabled: true,
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
};

export class AppConfigStore {
  private readonly configPath: string;
  private readonly secretsPath: string;
  private secretsPromise: Promise<SecretValues> | undefined;
  private runtimeSecretsPromise: Promise<Required<Pick<SecretValues, "internalApiKey">> & SecretValues> | undefined;

  constructor(
    private readonly userDataPath: string,
    private readonly validateOwnerVoice?: (configuration: DesktopConfiguration) => Promise<void>,
  ) {
    this.configPath = path.join(userDataPath, "config.json");
    this.secretsPath = path.join(userDataPath, "secrets.json");
  }

  async load(): Promise<DesktopConfigurationView> {
    const [configuration, secrets] = await Promise.all([
      this.readConfiguration(),
      this.readSecrets(),
    ]);
    return {
      ...configuration,
      configured: configuration.configured && hasRequiredSecrets(configuration, secrets),
      secrets: {
        jevApiKey: Boolean(secrets.jevApiKey),
        modelApiKeys: Object.fromEntries(
          [...configuration.models.text, ...configuration.models.voice]
            .map((model) => [model.id, Boolean(secrets.modelApiKeys?.[model.id])]),
        ),
        homeAssistantToken: Boolean(secrets.homeAssistantToken),
        openaiCompatibleImageApiKey: Boolean(secrets.openaiCompatibleImageApiKey),
        falApiKey: Boolean(secrets.falApiKey),
        agentApiKey: Boolean(secrets.agentApiKey),
      },
    };
  }

  async save(input: DesktopConfigurationInput): Promise<DesktopConfigurationView> {
    const configuration = validateConfiguration(input);
    const previous = await this.readConfiguration();
    if (configuration.audio.ownerRecognitionEnabled) {
      if (!this.validateOwnerVoice) throw new Error("请先准备模型并录入当前音频后端的主人声纹");
      if (!previous.audio.ownerRecognitionEnabled ||
        previous.advanced.audioBackend !== configuration.advanced.audioBackend) {
        await this.validateOwnerVoice(configuration);
      }
    }
    const secrets = await this.readSecrets();
    secrets.modelApiKeys = Object.fromEntries(
      [...configuration.models.text, ...configuration.models.voice].flatMap((model) => {
        const update = input.secrets?.modelApiKeys?.[model.id];
        const key = update === null ? undefined
          : typeof update === "string" && update.trim() ? update.trim()
          : secrets.modelApiKeys?.[model.id];
        return key ? [[model.id, key]] : [];
      }),
    );
    updateSecret(
      secrets,
      "homeAssistantToken",
      input.secrets?.homeAssistantToken,
    );
    updateSecret(
      secrets,
      "openaiCompatibleImageApiKey",
      input.secrets?.openaiCompatibleImageApiKey,
    );
    updateSecret(secrets, "falApiKey", input.secrets?.falApiKey);
    if (input.secrets?.jevApiKey === null) delete secrets.jevApiKey;
    else updateSecret(secrets, "jevApiKey", input.secrets?.jevApiKey);
    if (input.secrets?.agentApiKey === null) {
      delete secrets.agentApiKey;
    } else {
      updateSecret(secrets, "agentApiKey", input.secrets?.agentApiKey);
    }
    if (configuration.configured) {
      for (const module of ["chat", "reasoning"] as const) {
        const selected = textModel(configuration, module);
        if (!secrets.modelApiKeys[selected.id]) throw new Error(`请填写“${selected.name}”的 API Key`);
      }
    }
    if (configuration.homeAssistant.enabled && !secrets.homeAssistantToken) {
      throw new Error("启用 Home Assistant 时必须填写 Token");
    }
    const selectedVoice = voiceModel(configuration);
    if (selectedVoice?.provider === "dashscope" && !secrets.modelApiKeys[selectedVoice.id]) {
      throw new Error(`请填写“${selectedVoice.name}”的 TTS API Key`);
    }
    if (
      configuration.imageEditing.openaiCompatibleBaseUrl &&
      !secrets.openaiCompatibleImageApiKey
    ) {
      throw new Error("配置 OpenAI 兼容图片地址时必须填写图片 API Key");
    }
    await mkdir(this.userDataPath, { recursive: true });
    await Promise.all([
      atomicJsonWrite(this.configPath, configuration),
      this.writeSecrets(secrets),
    ]);
    return this.load();
  }

  async runtimeSecrets(): Promise<Required<Pick<SecretValues, "internalApiKey">> & SecretValues> {
    const pending = this.runtimeSecretsPromise ??= this.prepareRuntimeSecrets();
    try { return await pending; }
    finally { if (this.runtimeSecretsPromise === pending) this.runtimeSecretsPromise = undefined; }
  }

  private async prepareRuntimeSecrets(): Promise<Required<Pick<SecretValues, "internalApiKey">> & SecretValues> {
    const secrets = await this.readSecrets();
    if (!secrets.internalApiKey) {
      secrets.internalApiKey = `fm_${randomBytes(32).toString("base64url")}`;
      await this.writeSecrets(secrets);
    }
    return secrets as Required<Pick<SecretValues, "internalApiKey">> & SecretValues;
  }

  private async readConfiguration(): Promise<DesktopConfiguration> {
    try {
      const value = JSON.parse(await readFile(this.configPath, "utf8")) as unknown;
      return validateConfiguration(value);
    } catch {
      return structuredClone(DEFAULT_CONFIGURATION);
    }
  }

  private async readSecrets(): Promise<SecretValues> {
    if (!this.secretsPromise) {
      this.secretsPromise = this.readSecretsFromDisk();
    }
    const pending = this.secretsPromise;
    try {
      const secrets = await pending;
      return { ...secrets, modelApiKeys: { ...secrets.modelApiKeys } };
    } catch (error) {
      if (this.secretsPromise === pending) this.secretsPromise = undefined;
      throw error;
    }
  }

  private async readSecretsFromDisk(): Promise<SecretValues> {
    let stored: unknown;
    try {
      stored = JSON.parse(await readPrivateText(this.secretsPath)) as unknown;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return {};
      throw error;
    }
    if (!isRecord(stored) || stored.version !== 2 || !isRecord(stored.values)) {
      return {};
    }
    const values = stored.values as Record<string, unknown>;
    const sanitized: SecretValues = {};
    for (const key of SECRET_VALUE_KEYS) {
      const value = values[key];
      if (typeof value === "string" && value.trim()) {
        sanitized[key] = value.trim();
      }
    }
    if (isRecord(values.modelApiKeys)) {
      sanitized.modelApiKeys = Object.fromEntries(
        Object.entries(values.modelApiKeys).filter((entry): entry is [string, string] =>
          typeof entry[1] === "string" && Boolean(entry[1].trim()),
        ).map(([id, key]) => [id, key.trim()]),
      );
    }
    return sanitized;
  }

  private async writeSecrets(value: SecretValues): Promise<void> {
    await mkdir(this.userDataPath, { recursive: true });
    await atomicJsonWrite(this.secretsPath, {
      version: 2,
      values: value,
    } satisfies StoredSecrets);
    this.secretsPromise = Promise.resolve({ ...value });
  }
}

function hasRequiredSecrets(
  configuration: DesktopConfiguration,
  secrets: SecretValues,
): boolean {
  if (!configuration.models.text.length) return false;
  for (const module of ["chat", "reasoning"] as const) {
    if (!secrets.modelApiKeys?.[textModel(configuration, module).id]) return false;
  }
  const selectedVoice = voiceModel(configuration);
  if (selectedVoice?.provider === "dashscope" && !secrets.modelApiKeys?.[selectedVoice.id]) return false;
  if (configuration.homeAssistant.enabled && !secrets.homeAssistantToken) return false;
  return true;
}

function validateConfiguration(value: unknown): DesktopConfiguration {
  if (!isRecord(value)) return structuredClone(DEFAULT_CONFIGURATION);
  const candidate = {
    ...structuredClone(DEFAULT_CONFIGURATION),
    ...value,
    models: validateModels(value.models),
    imageEditing: {
      ...DEFAULT_CONFIGURATION.imageEditing,
      ...(isRecord(value.imageEditing) ? value.imageEditing : {}),
    },
    homeAssistant: {
      ...DEFAULT_CONFIGURATION.homeAssistant,
      ...(isRecord(value.homeAssistant) ? value.homeAssistant : {}),
    },
    perception: normalizedPerceptionConfiguration(value.perception),
    rendering: {
      maxTextureDimension: petTextureMaximumOrDefault(
        isRecord(value.rendering) ? value.rendering.maxTextureDimension : undefined,
        DEFAULT_CONFIGURATION.rendering.maxTextureDimension,
      ),
    },
    audio: {
      ...DEFAULT_CONFIGURATION.audio,
      ...(isRecord(value.audio) ? value.audio : {}),
    },
    advanced: {
      ...DEFAULT_CONFIGURATION.advanced,
      ...(isRecord(value.advanced) ? value.advanced : {}),
    },
    agent: {
      ...DEFAULT_CONFIGURATION.agent,
      ...(isRecord(value.agent) ? value.agent : {}),
    },
  } as DesktopConfiguration;
  const merged: DesktopConfiguration = {
    version: 2,
    configured: candidate.configured,
    llmLogRetentionDays: value.llmLogRetentionDays === undefined ? 7 : value.llmLogRetentionDays as number,
    voiceEnabled: candidate.voiceEnabled,
    microphoneEnabled: candidate.microphoneEnabled,
    playbackEnabled: candidate.playbackEnabled,
    speechFacialWaitForReady: booleanOrDefault(candidate.speechFacialWaitForReady, false),
    proactiveCompanionEnabled: candidate.proactiveCompanionEnabled,
    experimentalCompanion: {
      ...decisionChoice(value.experimentalCompanion),
      model: requiredText(isRecord(value.experimentalCompanion) ? value.experimentalCompanion.model ?? "jev-1.13.0" : "jev-1.13.0", "Jev 模型"),
      wakeSeconds: isRecord(value.experimentalCompanion) && typeof value.experimentalCompanion.wakeSeconds === "number"
        ? value.experimentalCompanion.wakeSeconds : 5,
      thoughtsEnabled: booleanOrDefault(isRecord(value.experimentalCompanion) ? value.experimentalCompanion.thoughtsEnabled : true, true),
      thoughtSeconds: isRecord(value.experimentalCompanion) && typeof value.experimentalCompanion.thoughtSeconds === "number"
        ? value.experimentalCompanion.thoughtSeconds : 2,
      // Unset until the user flips it: then picking replies follows whether Jev is connected.
      ...(isRecord(value.experimentalCompanion) && typeof value.experimentalCompanion.chatPickEnabled === "boolean"
        ? { chatPickEnabled: value.experimentalCompanion.chatPickEnabled } : {}),
      speechFacialEnabled: booleanOrDefault(isRecord(value.experimentalCompanion) ? value.experimentalCompanion.speechFacialEnabled : true, true),
    },
    codexNotificationsEnabled: candidate.codexNotificationsEnabled ?? false,
    webSearchEnabled: booleanOrDefault(value.webSearchEnabled, true),
    gateway: {
      enabled: booleanOrDefault(isRecord(value.gateway) ? value.gateway.enabled : false, false),
      port: isRecord(value.gateway) && value.gateway.port !== undefined ? value.gateway.port as number : DEFAULT_GATEWAY_PORT,
    },
    launchAtLogin: candidate.launchAtLogin,
    models: candidate.models,
    modelAssignments: validateAssignments(value.modelAssignments, candidate.models),
    imageEditing: {
      openaiCompatibleBaseUrl: candidate.imageEditing.openaiCompatibleBaseUrl,
    },
    homeAssistant: { ...candidate.homeAssistant },
    perception: { ...candidate.perception },
    rendering: { ...candidate.rendering },
    audio: { ...candidate.audio },
    advanced: { ...candidate.advanced },
    agent: {
      enabled: candidate.agent.enabled,
      url: candidate.agent.url,
      runConcurrency: candidate.agent.runConcurrency,
      runPollIntervalSeconds: candidate.agent.runPollIntervalSeconds,
      httpTimeoutSeconds: candidate.agent.httpTimeoutSeconds,
      sseReadTimeoutSeconds: candidate.agent.sseReadTimeoutSeconds,
      replyContextTtlSeconds: candidate.agent.replyContextTtlSeconds,
    },
  };
  merged.configured = booleanOrDefault(merged.configured, false);
  if (!Number.isInteger(merged.llmLogRetentionDays) || merged.llmLogRetentionDays! < 1 || merged.llmLogRetentionDays! > 365) {
    throw new Error("日志保留天数需为 1 至 365 的整数");
  }
  if (merged.experimentalCompanion && (!Number.isFinite(merged.experimentalCompanion.wakeSeconds) ||
    merged.experimentalCompanion.wakeSeconds < 1 || merged.experimentalCompanion.wakeSeconds > 3600)) {
    throw new Error("实验模式唤醒间隔需在 1 至 3600 秒之间");
  }
  const thoughtSeconds = merged.experimentalCompanion?.thoughtSeconds ?? 2;
  if (!Number.isFinite(thoughtSeconds) || thoughtSeconds < 1 || thoughtSeconds > 3600) {
    throw new Error("后台思考间隔需在 1 至 3600 秒之间");
  }
  merged.voiceEnabled = booleanOrDefault(merged.voiceEnabled, true);
  merged.microphoneEnabled = booleanOrDefault(value.microphoneEnabled, merged.voiceEnabled);
  merged.playbackEnabled = booleanOrDefault(value.playbackEnabled, merged.voiceEnabled);
  // Older files kept Jev on its own "experimental" switch; that switch is now 主动陪伴.
  merged.proactiveCompanionEnabled = isRecord(value.experimentalCompanion)
    && typeof value.experimentalCompanion.enabled === "boolean"
    ? value.experimentalCompanion.enabled
    : booleanOrDefault(merged.proactiveCompanionEnabled, false);
  merged.codexNotificationsEnabled = booleanOrDefault(merged.codexNotificationsEnabled, false);
  const gatewayPort = merged.gateway!.port;
  if (!Number.isInteger(gatewayPort) || gatewayPort < 1024 || gatewayPort > 65535) {
    throw new Error("外部消息的端口需在 1024 至 65535 之间");
  }
  merged.launchAtLogin = booleanOrDefault(merged.launchAtLogin, false);
  // The network proxy setting was removed; tulpa follows the system's own.
  delete (merged as { downloadProxy?: unknown }).downloadProxy;
  merged.imageEditing.openaiCompatibleBaseUrl = optionalHttpUrl(
    merged.imageEditing.openaiCompatibleBaseUrl,
    "OpenAI 兼容图片 Base URL",
  );
  merged.homeAssistant.enabled = booleanOrDefault(merged.homeAssistant.enabled, false);
  merged.audio.wakeWord = requiredText(merged.audio.wakeWord, "唤醒词");
  merged.audio.inputDevice = optionalText(merged.audio.inputDevice);
  merged.audio.outputDevice = optionalText(merged.audio.outputDevice);
  merged.audio.ownerRecognitionEnabled = booleanOrDefault(
    merged.audio.ownerRecognitionEnabled,
    false,
  );
  merged.homeAssistant.url = validHttpUrl(
    merged.homeAssistant.url,
    "Home Assistant 地址",
  );
  // Chosen by platform, never by the user; stored values from older versions are ignored.
  merged.advanced.audioBackend = platformAudioBackend();
  if (merged.advanced.audioBackend === "livekit") {
    // LiveKit follows the system default microphone and speaker.
    merged.audio.inputDevice = null;
    merged.audio.outputDevice = null;
  }
  range(merged.advanced.asrThreads, 1, 32, "ASR 线程数");
  merged.advanced.asrModel = oneOf(
    merged.advanced.asrModel,
    ["sensevoice"],
    "sensevoice",
  );
  range(merged.advanced.vadThreshold, 0, 1, "VAD 阈值");
  range(merged.advanced.vadMinSilenceMs, 10, 5_000, "VAD 静音时长");
  range(merged.advanced.vadMinSpeechMs, 10, 5_000, "VAD 人声时长");
  range(merged.advanced.smartTurnThreshold, 0, 1, "Smart Turn 阈值");
  // Turn detection and voice drafts are how TA listens; they are not options.
  merged.advanced.smartTurnEnabled = true;
  merged.advanced.aecDelayMode = oneOf(
    merged.advanced.aecDelayMode,
    ["auto", "fixed"],
    "auto",
  );
  range(merged.advanced.aecDelayMs, 0, 2_000, "AEC 延迟");
  merged.advanced.diagnosticsEnabled = booleanOrDefault(
    merged.advanced.diagnosticsEnabled,
    false,
  );
  merged.advanced.voiceDraftEnabled = true;
  merged.agent.enabled = booleanOrDefault(merged.agent.enabled, false);
  merged.agent.url = validHttpUrl(merged.agent.url, "Agent URL");
  positiveInteger(merged.agent.runConcurrency, "Agent 并发数");
  positiveNumber(merged.agent.runPollIntervalSeconds, "Agent 轮询间隔");
  positiveNumber(merged.agent.httpTimeoutSeconds, "Agent HTTP 超时");
  positiveNumber(merged.agent.sseReadTimeoutSeconds, "Agent SSE 超时");
  positiveNumber(merged.agent.replyContextTtlSeconds, "Agent 回复上下文 TTL");
  return merged;
}

type SecretScalarKey = Exclude<keyof SecretValues, "internalApiKey" | "modelApiKeys">;

function updateSecret(
  secrets: SecretValues,
  key: SecretScalarKey,
  value: unknown,
): void {
  if (value === undefined || typeof value !== "string") return;
  const normalized = value.trim();
  if (normalized) secrets[key] = normalized;
  else delete secrets[key];
}

function validHttpUrl(value: unknown, label: string): string {
  const text = requiredText(value, label);
  let parsed: URL;
  try {
    parsed = new URL(text);
  } catch {
    throw new Error(`${label}格式不正确`);
  }
  if (!new Set(["http:", "https:"]).has(parsed.protocol)) {
    throw new Error(`${label}必须使用 http 或 https`);
  }
  return parsed.toString().replace(/\/$/, "");
}

function optionalHttpUrl(value: unknown, label: string): string | null {
  if (value === null || value === undefined || value === "") return null;
  return validHttpUrl(value, label);
}

function requiredText(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label}不能为空`);
  return value.trim();
}

function optionalText(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function booleanOrDefault(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function normalizedPerceptionConfiguration(
  value: unknown,
): DesktopConfiguration["perception"] {
  if (!isRecord(value)) return { ...DEFAULT_CONFIGURATION.perception };
  const legacy = typeof value.environmentPerceptionEnabled === "boolean"
    ? value.environmentPerceptionEnabled
    : value.computerHistoryEnabled === true;
  const legacyAudiovisual = typeof value.audioPerceptionEnabled !== "boolean"
    && value.audiovisualPerceptionEnabled === true;
  return {
    operationPerceptionEnabled: booleanOrDefault(value.operationPerceptionEnabled, legacy),
    screenPerceptionEnabled: booleanOrDefault(value.screenPerceptionEnabled, legacy)
      || legacyAudiovisual,
    audioPerceptionEnabled: booleanOrDefault(value.audioPerceptionEnabled, legacyAudiovisual),
    // TA never sees itself or tulpa's windows on screen.
    excludeFishMemoryFromScreenCapture: true,
    innerLifeEnabled: booleanOrDefault(value.innerLifeEnabled, true),
  };
}

function oneOf<const T extends readonly string[]>(
  value: unknown,
  choices: T,
  fallback: T[number],
): T[number] {
  return typeof value === "string" && choices.includes(value) ? value : fallback;
}

function range(value: unknown, minimum: number, maximum: number, label: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${label}必须在 ${minimum} 到 ${maximum} 之间`);
  }
}

function positiveInteger(value: unknown, label: string): void {
  if (!Number.isInteger(value) || (value as number) < 1) {
    throw new Error(`${label}必须是正整数`);
  }
}

function positiveNumber(value: unknown, label: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label}必须大于零`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}


async function atomicJsonWrite(file: string, value: unknown): Promise<void> {
  await writePrivateText(file, `${JSON.stringify(value, null, 2)}\n`);
}

/** 决策模型 settings: which one, and the 百炼业务空间 ID it needs; anything unusable is dropped. */
function decisionChoice(value: unknown): { decisionProvider?: "bailian" | "jev"; bailianWorkspaceId?: string } {
  if (!isRecord(value)) return {};
  const provider = value.decisionProvider === "bailian" || value.decisionProvider === "jev" ? value.decisionProvider : undefined;
  const workspace = typeof value.bailianWorkspaceId === "string" ? value.bailianWorkspaceId.trim() : "";
  if (workspace && !BAILIAN_WORKSPACE_PATTERN.test(workspace)) throw new Error("百炼业务空间 ID 格式不对，应形如 ws-xxxx 或 llm-xxxx");
  return { ...provider ? { decisionProvider: provider } : {}, ...workspace ? { bailianWorkspaceId: workspace } : {} };
}
