import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import {
  resolveTextureEditImageModel,
  type CharacterGenerationProvider,
  type CharacterProviderStatus,
} from "../../src/shared/character-generation";

type Fetcher = (input: string, init?: RequestInit) => Promise<Response>;

export const OPENAI_COMPATIBLE_TEXTURE_QUALITY = "high";

export interface OpenAICompatibleImageProviderOptions {
  baseUrl: string;
  apiKey: string;
}

interface OpenAICompatibleImageProviderDependencies {
  fetcher: Fetcher;
  readLocalFile(file: string): Promise<Uint8Array>;
  requestId(): string;
}

/**
 * OpenAI-compatible implementation of the current masked UV texture editor.
 * It intentionally does not fall back to fal: choosing this provider is an
 * explicit configuration decision made by the user.
 */
export class OpenAICompatibleImageProvider implements CharacterGenerationProvider {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly fetcher: Fetcher;
  private readonly readLocalFile: OpenAICompatibleImageProviderDependencies["readLocalFile"];
  private readonly requestId: OpenAICompatibleImageProviderDependencies["requestId"];

  constructor(
    options: OpenAICompatibleImageProviderOptions,
    dependencies: Partial<OpenAICompatibleImageProviderDependencies> = {},
  ) {
    this.baseUrl = normalizeBaseUrl(options.baseUrl);
    this.apiKey = options.apiKey.trim();
    if (!this.apiKey) throw new Error("请先配置 OpenAI 兼容图片 API Key");
    this.fetcher = dependencies.fetcher ?? fetch;
    this.readLocalFile = dependencies.readLocalFile ?? readFile;
    this.requestId = dependencies.requestId ?? randomUUID;
  }

  async editMaskedTexture(
    input: Parameters<CharacterGenerationProvider["editMaskedTexture"]>[0],
  ): ReturnType<CharacterGenerationProvider["editMaskedTexture"]> {
    const startedAt = Date.now();
    const imageModel = resolveTextureEditImageModel(input.imageModel);
    throwIfAborted(input.signal);
    const requestId = this.requestId();
    const referencePaths = input.referenceImagePaths ?? [];
    const [imageBytes, maskBytes, ...referenceBytes] = await Promise.all([
      this.readLocalFile(input.imagePath),
      this.readLocalFile(input.maskPath),
      ...referencePaths.map((file) => this.readLocalFile(file)),
    ]);
    throwIfAborted(input.signal);
    // The Image API accepts the selected image model directly, including
    // large canvases, without requiring a second text model to call a tool.
    const endpointUrl = `${this.baseUrl}/images/edits`;
    const endpoint = `openai-compatible:${imageModel}/images/edits`;
    const form = new FormData();
    form.append("model", imageModel);
    form.append("prompt", input.prompt);
    const multipartImageField = referencePaths.length > 0 ? "image[]" : "image";
    form.append(multipartImageField, imageBlob(input.imagePath, imageBytes), path.basename(input.imagePath));
    for (let index = 0; index < referencePaths.length; index += 1) {
      const file = referencePaths[index]!;
      form.append(multipartImageField, imageBlob(file, referenceBytes[index]!), path.basename(file));
    }
    form.append("size", input.imageSize);
    form.append("quality", OPENAI_COMPATIBLE_TEXTURE_QUALITY);
    form.append("output_format", "png");
    form.append("background", "transparent");
    form.append(
      "mask",
      imageBlob(input.maskPath, maskBytes),
      path.basename(input.maskPath),
    );
    await input.onSubmitted?.(endpoint, requestId);
    input.onStatus?.(status("running", startedAt, "OpenAI 兼容图片服务正在生成"));
    const response = await this.fetcher(endpointUrl, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.apiKey}` },
      body: form,
      signal: input.signal,
    });
    const body = await response.text();
    const parsed = parseJson(body);
    if (!response.ok) {
      throw new Error(openAIErrorMessage(parsed, response.status));
    }
    const url = imageResultUrl(parsed);
    input.onStatus?.(status("completed", startedAt, null));
    return { endpoint, requestId, urls: [url] };
  }

  generateCandidates(): ReturnType<CharacterGenerationProvider["generateCandidates"]> {
    return removedFeature();
  }

  editImage(): ReturnType<CharacterGenerationProvider["editImage"]> {
    return removedFeature();
  }

  inpaintImage(): ReturnType<CharacterGenerationProvider["inpaintImage"]> {
    return removedFeature();
  }

  decomposeImage(): ReturnType<CharacterGenerationProvider["decomposeImage"]> {
    return removedFeature();
  }

  removeBackground(): ReturnType<CharacterGenerationProvider["removeBackground"]> {
    return removedFeature();
  }

  segmentImage(): ReturnType<CharacterGenerationProvider["segmentImage"]> {
    return removedFeature();
  }

  analyzeImages(): ReturnType<CharacterGenerationProvider["analyzeImages"]> {
    return removedFeature();
  }

  async cancel(): Promise<void> {
    // The caller owns the AbortSignal passed to fetch; aborting that signal is
    // the only portable cancellation mechanism for OpenAI-compatible servers.
  }
}

function normalizeBaseUrl(value: string): string {
  const text = value.trim().replace(/\/+$/, "");
  let parsed: URL;
  try {
    parsed = new URL(text);
  } catch {
    throw new Error("OpenAI 兼容图片 Base URL 格式不正确");
  }
  if (!new Set(["http:", "https:"]).has(parsed.protocol)) {
    throw new Error("OpenAI 兼容图片 Base URL 必须使用 http 或 https");
  }
  return text;
}

function imageBlob(filePath: string, bytes: Uint8Array): Blob {
  return new Blob([Buffer.from(bytes)], {
    type: mimeType(filePath),
  });
}

function imageResultUrl(value: unknown): string {
  const root = record(value);
  const data = root.data;
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error("OpenAI 兼容图片服务没有返回图片");
  }
  const image = record(data[0]);
  if (typeof image.b64_json === "string" && image.b64_json.trim()) {
    return `data:image/png;base64,${image.b64_json.trim()}`;
  }
  if (typeof image.url === "string" && /^https:\/\//i.test(image.url)) {
    return image.url;
  }
  throw new Error("OpenAI 兼容图片服务返回的图片格式无效");
}

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return { message: value.trim() };
  }
}

function openAIErrorMessage(value: unknown, statusCode: number): string {
  if (typeof value === "object" && value !== null) {
    const root = value as Record<string, unknown>;
    const error = typeof root.error === "object" && root.error !== null
      ? root.error as Record<string, unknown>
      : null;
    const message = typeof error?.message === "string"
      ? error.message
      : typeof root.message === "string"
        ? root.message
        : null;
    if (message?.trim()) return `OpenAI 兼容图片请求失败：${message.trim()}`;
  }
  return `OpenAI 兼容图片请求失败 (${statusCode})`;
}

function status(
  stage: CharacterProviderStatus["stage"],
  startedAt: number,
  message: string | null,
): CharacterProviderStatus {
  return {
    stage,
    queuePosition: null,
    message,
    elapsedMs: Date.now() - startedAt,
  };
}

function removedFeature<T>(): Promise<T> {
  return Promise.reject(new Error("AI 创建角色和旧部位修改功能已移除，请使用纹理编辑"));
}

function record(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("OpenAI 兼容图片服务返回格式无效");
  }
  return value as Record<string, unknown>;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Cancelled", "AbortError");
}

function mimeType(filePath: string): string {
  switch (path.extname(filePath).toLowerCase()) {
    case ".jpg":
    case ".jpeg": return "image/jpeg";
    case ".webp": return "image/webp";
    default: return "image/png";
  }
}
