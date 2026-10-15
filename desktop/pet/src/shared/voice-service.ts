import type { DesktopConfiguration, DesktopVoiceModel } from "./contracts";

/** Where a 通义 key typed on a character page is kept: the existing cloud service, or a new one. */
export function cloudVoiceService(configuration: Pick<DesktopConfiguration, "models">): { models: DesktopConfiguration["models"]; id: string } {
  const existing = configuration.models.voice.find((item) => item.provider === "dashscope")
    ?? configuration.models.voice.find((item) => item.provider === "auto");
  if (existing) return { models: configuration.models, id: existing.id };
  const added: DesktopVoiceModel = { id: "voice-dashscope", name: "通义语音", provider: "dashscope" };
  return { models: { ...configuration.models, voice: [...configuration.models.voice, added] }, id: added.id };
}
