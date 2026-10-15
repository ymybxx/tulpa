import type { CharacterGenerationProvider } from "../../src/shared/character-generation";
import type { DesktopConfiguration } from "../../src/shared/contracts";
import { FalCharacterGenerationProvider } from "./fal-provider";
import { OpenAICompatibleImageProvider } from "./openai-compatible-image-provider";

export interface ImageProviderSecrets {
  openaiCompatibleImageApiKey?: string;
  falApiKey?: string;
}

export interface ImageProviderConstructors {
  openai(options: ConstructorParameters<typeof OpenAICompatibleImageProvider>[0]): CharacterGenerationProvider;
  fal(apiKey: string): CharacterGenerationProvider;
}

const DEFAULT_CONSTRUCTORS: ImageProviderConstructors = {
  openai: (options) => new OpenAICompatibleImageProvider(options),
  fal: (apiKey) => new FalCharacterGenerationProvider(apiKey),
};

/**
 * Selects exactly one image transport. An OpenAI-compatible Base URL is an
 * explicit override: fal is neither inspected nor used as an implicit
 * fallback when that URL is configured.
 */
export function createConfiguredImageProvider(
  configuration: Pick<DesktopConfiguration, "imageEditing">,
  secrets: ImageProviderSecrets,
  constructors: ImageProviderConstructors = DEFAULT_CONSTRUCTORS,
): CharacterGenerationProvider {
  const baseUrl = configuration.imageEditing.openaiCompatibleBaseUrl;
  if (baseUrl) {
    const apiKey = secrets.openaiCompatibleImageApiKey?.trim();
    if (!apiKey) throw new Error("请先在设置中填写 OpenAI 兼容图片 API Key");
    return constructors.openai({
      baseUrl,
      apiKey,
    });
  }
  const falApiKey = secrets.falApiKey?.trim();
  if (!falApiKey) throw new Error("请先在设置中填写 fal API Key");
  return constructors.fal(falApiKey);
}
