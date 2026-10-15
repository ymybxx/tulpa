import type {
  DesktopConfiguration,
  DesktopModelAssignments,
  DesktopModelCatalog,
  DesktopTextModel,
  DesktopVoiceModel,
} from "../src/shared/contracts";

export const DEFAULT_MODEL_CATALOG: DesktopModelCatalog = {
  text: [{
    id: "text-default", name: "默认文字模型",
    baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1", model: "qwen-plus",
  }],
  voice: [{ id: "voice-default", name: "默认语音服务", provider: "auto" }],
};

export const DEFAULT_MODEL_ASSIGNMENTS: DesktopModelAssignments = {
  chat: "text-default", reasoning: "text-default", speech: "voice-default",
};

export function validateModels(value: unknown): DesktopModelCatalog {
  if (value === undefined) return structuredClone(DEFAULT_MODEL_CATALOG);
  if (!record(value) || !Array.isArray(value.text) || !Array.isArray(value.voice)) {
    throw new Error("模型配置必须包含文字和语音模型列表");
  }
  const ids = new Set<string>();
  function identity(item: unknown): { id: string; name: string } {
    if (!record(item)) throw new Error("模型配置无效");
    const id = required(item.id, "模型 ID");
    if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/.test(id) || Object.hasOwn(Object.prototype, id)) {
      throw new Error("模型 ID 无效");
    }
    if (ids.has(id)) throw new Error("模型 ID 不能重复");
    ids.add(id);
    return { id, name: required(item.name, "模型名称") };
  }
  return {
    text: value.text.map((raw: unknown): DesktopTextModel => {
      const item = modelRecord(raw);
      const info = identity(item);
      const url = new URL(required(item.baseUrl, "API Base URL"));
      if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) {
        throw new Error("模型地址必须是 HTTP(S) 地址，密钥请填写在 API Key 中");
      }
      return { ...info, baseUrl: url.href.replace(/\/$/, ""), model: required(item.model, "模型名称 / ID") };
    }),
    voice: value.voice.map((raw: unknown): DesktopVoiceModel => {
      const item = modelRecord(raw);
      const info = identity(item);
      const provider = item.provider;
      if (provider !== "auto" && provider !== "dashscope" && provider !== "local") {
        throw new Error("语音服务类型无效");
      }
      return { ...info, provider };
    }),
  };
}

export function validateAssignments(value: unknown, models: DesktopModelCatalog): DesktopModelAssignments {
  const selected = record(value) ? value : {};
  function resolve(key: keyof DesktopModelAssignments, options: { id: string }[]): string | null {
    if (!options.length) return null;
    if (options.length === 1 || selected[key] == null || selected[key] === "") return options[0]!.id;
    if (!options.some((model) => model.id === selected[key])) throw new Error("所选模块模型不存在，请重新选择");
    return selected[key] as string;
  }
  return {
    chat: resolve("chat", models.text) ?? "",
    reasoning: resolve("reasoning", models.text) ?? "",
    speech: resolve("speech", models.voice),
  };
}

export function textModel(configuration: DesktopConfiguration, module: "chat" | "reasoning"): DesktopTextModel {
  const model = configuration.models.text.find((item) => item.id === configuration.modelAssignments[module]);
  if (!model) throw new Error("请先配置文字模型并选择模块模型");
  return model;
}

export function voiceModel(configuration: DesktopConfiguration): DesktopVoiceModel | undefined {
  return configuration.models.voice.find((item) => item.id === configuration.modelAssignments.speech);
}

/**
 * The 通义 key every character's cloud voice speaks with. Characters choose
 * 通义 or 本机 themselves, so the key reaches the service whichever voice
 * service is the default: the default's own key first, then any cloud one,
 * then the chat service's key when that is 千问 — one 阿里云百炼 key serves both.
 */
export function cloudVoiceKey(configuration: DesktopConfiguration, keys: Record<string, string> | undefined): string | undefined {
  const selected = voiceModel(configuration);
  const candidates = [selected, ...configuration.models.voice].filter((item): item is DesktopVoiceModel => item !== undefined && item.provider !== "local");
  const own = candidates.map((item) => keys?.[item.id]).find((key) => Boolean(key));
  if (own) return own;
  const bailian = configuration.models.text.find((item) => isBailian(item.baseUrl) && keys?.[item.id]);
  return bailian ? keys?.[bailian.id] : undefined;
}

/** 阿里云百炼 (DashScope) — its keys also run 通义 speech. */
export function isBailian(baseUrl: string): boolean {
  try {
    return new URL(baseUrl).hostname.endsWith("dashscope.aliyuncs.com");
  } catch {
    return false;
  }
}

function required(value: unknown, label: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`请填写${label}`);
  return value.trim();
}

function modelRecord(value: unknown): Record<string, unknown> {
  if (!record(value)) throw new Error("模型配置无效");
  return value;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
