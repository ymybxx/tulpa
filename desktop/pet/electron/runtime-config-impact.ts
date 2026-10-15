import type {
  DesktopConfiguration,
  DesktopConfigurationInput,
  DesktopConfigurationView,
} from "../src/shared/contracts";
import { bailianKeyModelId, bailianWorkspaceId, chatPickEnabled, decisionConnection, decisionProvider } from "../src/shared/decision-model";
import { webSearchSupported } from "../src/shared/web-search";
import type { SecretValues } from "./app-config";
import { cloudVoiceKey, textModel, voiceModel } from "./model-configuration";
import { supportsSystemAudio } from "./system-audio-support";

export type RuntimeConfigurationAction =
  | "none"
  | "start"
  | "stop"
  | "configure-api"
  | "configure-api-and-companion"
  | "restart-companion"
  | "restart-runtime";

export interface ConfigurationSecrets {
  previous: SecretValues;
  next: SecretValues;
}

/** Wire representation shared by initial startup and live API updates. */
export function runtimeApiConfiguration(
  configuration: DesktopConfiguration,
  secrets: SecretValues,
) {
  const endpoint = (module: "chat" | "reasoning") => {
    const selected = textModel(configuration, module);
    return {
      baseUrl: selected.baseUrl,
      model: selected.model,
      apiKey: secrets.modelApiKeys?.[selected.id] ?? "",
    };
  };
  const voice = voiceModel(configuration);
  return {
    models: { chat: endpoint("chat"), reasoning: endpoint("reasoning") },
    speech: {
      provider: voice?.provider ?? "local",
      api_key: cloudVoiceKey(configuration, secrets.modelApiKeys) ?? null,
    },
    voice_draft_enabled: configuration.advanced.voiceDraftEnabled,
    llm_log_retention_days: configuration.llmLogRetentionDays ?? 7,
    speech_facial_wait_for_ready: configuration.speechFacialWaitForReady ?? false,
    experimental_companion: experimentalConfiguration(configuration, secrets),
    codex_notifications_enabled: configuration.codexNotificationsEnabled ?? false,
    web_search_enabled: webSearchOn(configuration),
    microphone_enabled: configuration.microphoneEnabled ?? configuration.voiceEnabled,
    playback_enabled: configuration.playbackEnabled ?? configuration.voiceEnabled,
    agent_tuning: configuration.agent.enabled ? agentTuning(configuration) : null,
    agent: {
      enabled: configuration.agent.enabled,
      url: configuration.agent.url,
      api_key: secrets.agentApiKey ?? null,
    },
    forget_agent_credentials: !configuration.agent.enabled && !secrets.agentApiKey,
  };
}

/** Only settings consumed at process construction can require a restart. */
export function runtimeConfigurationAction(
  previous: DesktopConfigurationView,
  next: DesktopConfigurationView,
  input: DesktopConfigurationInput,
  secrets?: ConfigurationSecrets,
): RuntimeConfigurationAction {
  if (previous.configured && !next.configured) return "stop";
  if (!previous.configured && next.configured) return "start";
  if (!next.configured) return "none";

  // Home Assistant currently constructs its owners at API startup.
  // Editing an inactive integration has no effect on the running services.
  if (
    ((previous.homeAssistant.enabled || next.homeAssistant.enabled) && (
      changed(previous.homeAssistant, next.homeAssistant) ||
      secretChanged("homeAssistantToken", input, secrets)
    ))
  ) return "restart-runtime";

  const apiChanged = secrets
    ? changed(runtimeApiConfiguration(previous, secrets.previous), runtimeApiConfiguration(next, secrets.next))
    : changed(selectedModels(previous), selectedModels(next)) ||
      (previous.llmLogRetentionDays ?? 7) !== (next.llmLogRetentionDays ?? 7) ||
      previous.microphoneEnabled !== next.microphoneEnabled ||
      previous.playbackEnabled !== next.playbackEnabled ||
      Boolean(previous.speechFacialWaitForReady) !== Boolean(next.speechFacialWaitForReady) ||
      previous.advanced.voiceDraftEnabled !== next.advanced.voiceDraftEnabled ||
      Boolean(previous.proactiveCompanionEnabled) !== Boolean(next.proactiveCompanionEnabled) ||
      JSON.stringify(previous.experimentalCompanion) !== JSON.stringify(next.experimentalCompanion) ||
      // What the service's companion settings are built from elsewhere, such
      // as 内心 in perception.
      changed(experimentalConfiguration(previous, {}), experimentalConfiguration(next, {})) ||
      input.secrets?.jevApiKey !== undefined ||
      Boolean(previous.codexNotificationsEnabled) !== Boolean(next.codexNotificationsEnabled) ||
      webSearchOn(previous) !== webSearchOn(next) ||
      previous.agent.enabled !== next.agent.enabled || previous.agent.url !== next.agent.url ||
      secretChanged("agentApiKey", input, secrets) ||
      (next.agent.enabled && changed(agentTuning(previous), agentTuning(next))) ||
      (!next.agent.enabled && input.secrets?.agentApiKey === null) ||
      Object.values(next.modelAssignments).some((id) => id &&
        input.secrets?.modelApiKeys?.[id] !== undefined) ||
      // A 通义 key typed on a character page belongs to no assignment.
      next.models.voice.some((item) => input.secrets?.modelApiKeys?.[item.id] !== undefined) ||
      // The 百炼 decision model borrows a saved Bailian key that may serve no brain.
      decisionKeyChanged(next, input);
  const companionChanged = previous.voiceEnabled !== next.voiceEnabled ||
    changed(companionConfiguration(previous), companionConfiguration(next));

  if (apiChanged) return companionChanged ? "configure-api-and-companion" : "configure-api";
  if (companionChanged) return "restart-companion";
  return "none";
}

function companionConfiguration(configuration: DesktopConfigurationView): object {
  if (!configuration.voiceEnabled) {
    return configuration.perception.audioPerceptionEnabled && supportsSystemAudio()
      ? { mediaAsr: true, threads: configuration.advanced.asrThreads }
      : { mediaAsr: false };
  }
  const advanced = configuration.advanced;
  return {
    audio: configuration.audio,
    agentReplyContextTtl: configuration.agent.replyContextTtlSeconds,
    advanced: {
      ...advanced,
      aecDelayMs: advanced.aecDelayMode === "fixed" ? advanced.aecDelayMs : null,
      smartTurnThreshold: advanced.smartTurnEnabled ? advanced.smartTurnThreshold : null,
    },
  };
}

/** 联网查资料 as the service gets it: switched on, and possible with the reasoning model. */
export function webSearchOn(configuration: DesktopConfiguration): boolean {
  if (configuration.webSearchEnabled === false) return false;
  try {
    return webSearchSupported(textModel(configuration, "reasoning").baseUrl);
  } catch {
    return false;
  }
}

export function experimentalConfiguration(configuration: DesktopConfiguration, secrets: SecretValues) {
  // The decision model (百炼 or Jev) is optional: connected, it judges; otherwise the reasoning model does.
  // Each judgement has its own switch; `enabled` is only 主动陪伴.
  const decision = decisionConnection(configuration, secrets);
  return {
    enabled: Boolean(configuration.proactiveCompanionEnabled),
    chat_pick_enabled: chatPickEnabled(configuration, decision !== null),
    speech_facial_enabled: configuration.experimentalCompanion?.speechFacialEnabled ?? true,
    api_key: decision?.apiKey ?? null,
    model: decision?.model ?? configuration.experimentalCompanion?.model ?? "jev-1.13.0",
    ...decision ? { endpoint: decision.endpoint } : {},
    wake_seconds: configuration.experimentalCompanion?.wakeSeconds ?? 5,
    thoughts_enabled: (configuration.experimentalCompanion?.thoughtsEnabled ?? true)
      && configuration.perception.innerLifeEnabled !== false,
    thought_seconds: configuration.experimentalCompanion?.thoughtSeconds ?? 2,
  };
}

function selectedModels(configuration: DesktopConfigurationView): object {
  const endpoint = (module: "chat" | "reasoning") => {
    const { id, baseUrl, model } = textModel(configuration, module);
    return { id, baseUrl, model };
  };
  const voice = voiceModel(configuration);
  return { chat: endpoint("chat"), reasoning: endpoint("reasoning"), voice: voice ? { id: voice.id, provider: voice.provider } : null };
}

function agentTuning(configuration: DesktopConfiguration) {
  const agent = configuration.agent;
  return {
    concurrency: agent.runConcurrency,
    poll_seconds: agent.runPollIntervalSeconds,
    http_timeout_seconds: agent.httpTimeoutSeconds,
    sse_read_timeout_seconds: agent.sseReadTimeoutSeconds,
    reply_context_ttl_seconds: agent.replyContextTtlSeconds,
  };
}

function changed(left: object, right: object): boolean {
  return JSON.stringify(left) !== JSON.stringify(right);
}

function secretChanged(
  name: "homeAssistantToken" | "agentApiKey",
  input: DesktopConfigurationInput,
  secrets?: ConfigurationSecrets,
): boolean {
  if (secrets) return secrets.previous[name] !== secrets.next[name];
  const value = input.secrets?.[name];
  return value === null || (typeof value === "string" && value.trim().length > 0);
}

/** Whether this save replaces the key the 百炼 decision model borrows. */
function decisionKeyChanged(next: DesktopConfigurationView, input: DesktopConfigurationInput): boolean {
  const has = (id: string) => next.secrets.modelApiKeys?.[id] === true || Boolean(input.secrets?.modelApiKeys?.[id]);
  if (!bailianWorkspaceId(next) || decisionProvider(next, has, next.secrets.jevApiKey === true) !== "bailian") return false;
  const borrowed = bailianKeyModelId(next, has);
  return borrowed !== null && input.secrets?.modelApiKeys?.[borrowed] !== undefined;
}
