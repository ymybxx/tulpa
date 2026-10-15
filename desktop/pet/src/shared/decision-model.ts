import type { DesktopConfiguration } from "./contracts";

/**
 * 决策模型: the model that makes TA's quick judgements — when to speak, how to
 * answer, which thought to keep, and the face for each spoken sentence. Bailian's
 * decision model and Jev take the same request (`state` plus typed questions) at
 * a different address; without either, the reasoning model judges (no faces).
 */
export type DecisionProvider = "bailian" | "jev";

export const JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
export const JEV_DEFAULT_MODEL = "jev-1.13.0";
export const BAILIAN_DECISION_MODEL = "decision-model-preview";
/** Bailian's decision model runs in 北京 and 新加坡; keys and workspaces are per region, and ours are 北京. */
export const BAILIAN_DECISION_REGION = "cn-beijing";
export const BAILIAN_WORKSPACE_PATTERN = /^[a-z0-9][a-z0-9-]{2,62}$/i;

export function bailianDecisionEndpoint(workspaceId: string): string {
  return `https://${workspaceId}.${BAILIAN_DECISION_REGION}.maas.aliyuncs.com/compatible-mode/v1/systemone`;
}

/** A Bailian (DashScope) address, old shared or per-workspace. */
export function isBailianUrl(baseUrl: string): boolean {
  try {
    const host = new URL(baseUrl).hostname.toLowerCase();
    return host === "dashscope.aliyuncs.com" || host.endsWith(".dashscope.aliyuncs.com") || host.endsWith(".maas.aliyuncs.com");
  } catch {
    return false;
  }
}

type DecisionSettings = Pick<DesktopConfiguration, "experimentalCompanion" | "models">;

/** The saved model whose key Bailian's decision model uses: a Bailian text model, else the 通义 voice. */
export function bailianKeyModelId(configuration: Pick<DesktopConfiguration, "models">, hasKey: (id: string) => boolean): string | null {
  const text = configuration.models.text.find(model => isBailianUrl(model.baseUrl) && hasKey(model.id));
  if (text) return text.id;
  return configuration.models.voice.find(model => model.provider === "dashscope" && hasKey(model.id))?.id ?? null;
}

export function bailianWorkspaceId(configuration: Pick<DesktopConfiguration, "experimentalCompanion">): string | null {
  const value = configuration.experimentalCompanion?.bailianWorkspaceId?.trim();
  return value && BAILIAN_WORKSPACE_PATTERN.test(value) ? value : null;
}

/**
 * Which decision model judges. The person's choice when they made one;
 * otherwise Bailian when it can work, Jev when only its key is set, and
 * Bailian as the default to set up.
 */
export function decisionProvider(configuration: DecisionSettings, hasKey: (id: string) => boolean, jevKey: boolean): DecisionProvider {
  const chosen = configuration.experimentalCompanion?.decisionProvider;
  if (chosen === "bailian" || chosen === "jev") return chosen;
  if (bailianWorkspaceId(configuration) && bailianKeyModelId(configuration, hasKey)) return "bailian";
  return jevKey ? "jev" : "bailian";
}

/** Whether the chosen decision model has everything it needs. */
export function decisionReady(configuration: DecisionSettings, hasKey: (id: string) => boolean, jevKey: boolean): boolean {
  return decisionProvider(configuration, hasKey, jevKey) === "jev"
    ? jevKey
    : Boolean(bailianWorkspaceId(configuration) && bailianKeyModelId(configuration, hasKey));
}

export interface DecisionConnection {
  provider: DecisionProvider;
  endpoint: string;
  model: string;
  apiKey: string;
}

/** Where and how the runtime calls the decision model, or null when it cannot. */
export function decisionConnection(
  configuration: DecisionSettings,
  secrets: { jevApiKey?: string; modelApiKeys?: Record<string, string> },
): DecisionConnection | null {
  const keys = secrets.modelApiKeys ?? {};
  const hasKey = (id: string) => Boolean(keys[id]?.trim());
  const jevKey = secrets.jevApiKey?.trim();
  const provider = decisionProvider(configuration, hasKey, Boolean(jevKey));
  if (provider === "jev") {
    return jevKey ? { provider, endpoint: JEV_ENDPOINT, model: configuration.experimentalCompanion?.model || JEV_DEFAULT_MODEL, apiKey: jevKey } : null;
  }
  const workspace = bailianWorkspaceId(configuration);
  const keyModel = bailianKeyModelId(configuration, hasKey);
  return workspace && keyModel
    ? { provider, endpoint: bailianDecisionEndpoint(workspace), model: BAILIAN_DECISION_MODEL, apiKey: keys[keyModel]!.trim() }
    : null;
}

/** 聊天时挑怎么回: the person's choice, or on exactly when a decision model is connected. */
export function chatPickEnabled(configuration: Pick<DesktopConfiguration, "experimentalCompanion">, connected: boolean): boolean {
  return configuration.experimentalCompanion?.chatPickEnabled ?? connected;
}
