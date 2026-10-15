import { describe, expect, it, vi } from "vitest";

import type { CharacterGenerationProvider } from "../../src/shared/character-generation";
import { createConfiguredImageProvider } from "./image-provider-factory";

const placeholder = {} as CharacterGenerationProvider;

describe("createConfiguredImageProvider", () => {
  it("uses only the OpenAI-compatible provider when its Base URL is configured", () => {
    const openai = vi.fn(() => placeholder);
    const fal = vi.fn(() => placeholder);

    const result = createConfiguredImageProvider({ imageEditing: {
      openaiCompatibleBaseUrl: "http://127.0.0.1:8080/v1",
    } }, {
      openaiCompatibleImageApiKey: "image-key",
      falApiKey: "fal-key-that-must-be-ignored",
    }, { openai, fal });

    expect(result).toBe(placeholder);
    expect(openai).toHaveBeenCalledWith({
      baseUrl: "http://127.0.0.1:8080/v1",
      apiKey: "image-key",
    });
    expect(fal).not.toHaveBeenCalled();
  });

  it("does not fall back to fal when the OpenAI image key is missing", () => {
    const openai = vi.fn(() => placeholder);
    const fal = vi.fn(() => placeholder);

    expect(() => createConfiguredImageProvider({ imageEditing: {
      openaiCompatibleBaseUrl: "https://images.example.com/v1",
    } }, { falApiKey: "fal-key" }, { openai, fal })).toThrow(/OpenAI 兼容图片 API Key/);
    expect(openai).not.toHaveBeenCalled();
    expect(fal).not.toHaveBeenCalled();
  });

  it("uses fal only when no OpenAI-compatible Base URL is configured", () => {
    const openai = vi.fn(() => placeholder);
    const fal = vi.fn(() => placeholder);

    createConfiguredImageProvider({ imageEditing: {
      openaiCompatibleBaseUrl: null,
    } }, { falApiKey: "fal-key" }, { openai, fal });

    expect(fal).toHaveBeenCalledWith("fal-key");
    expect(openai).not.toHaveBeenCalled();
  });
});
