import { File } from "node:buffer";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { createFalClient } from "@fal-ai/client";

import {
  resolveTextureEditImageModel,
  type CharacterGenerationProvider,
  type CharacterProviderStatus,
} from "../../src/shared/character-generation";

/** fal is now only the fallback transport for the current UV texture editor. */
export const FAL_ENDPOINTS = {
  textureEdit: "openai/gpt-image-2.5/sunburst/edit",
} as const;

interface FalQueueResult {
  data: unknown;
  requestId: string;
}

interface FalQueueStatus {
  status: "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED";
  queue_position?: number;
  logs?: Array<{ message?: string }>;
  metrics?: { inference_time?: number };
}

export interface FalClientPort {
  storage: {
    upload(file: File, options?: { lifecycle?: { expiresIn?: "1h" } }): Promise<string>;
  };
  queue: {
    submit(endpoint: string, options: {
      input: Record<string, unknown>;
      storageSettings?: { expiresIn: "1h" };
    }): Promise<{ request_id: string }>;
    subscribeToStatus(
      endpoint: string,
      options: {
        requestId: string;
        abortSignal?: AbortSignal;
        logs?: boolean;
        onQueueUpdate?: (status: FalQueueStatus) => void;
      },
    ): Promise<unknown>;
    result(endpoint: string, options: {
      requestId: string;
      abortSignal?: AbortSignal;
    }): Promise<FalQueueResult>;
    cancel(endpoint: string, options: { requestId: string }): Promise<void>;
  };
}

interface FalProviderDependencies {
  createClient(credentials: string): FalClientPort;
  readLocalFile(file: string): Promise<Uint8Array>;
}

export class FalCharacterGenerationProvider implements CharacterGenerationProvider {
  private readonly client: FalClientPort;
  private readonly readLocalFile: FalProviderDependencies["readLocalFile"];

  constructor(credentials: string, dependencies: Partial<FalProviderDependencies> = {}) {
    if (!credentials.trim()) throw new Error("请先配置 fal API Key");
    this.client = dependencies.createClient?.(credentials) ??
      createFalClient({ credentials }) as unknown as FalClientPort;
    this.readLocalFile = dependencies.readLocalFile ?? readFile;
  }

  async editMaskedTexture(
    input: Parameters<CharacterGenerationProvider["editMaskedTexture"]>[0],
  ): ReturnType<CharacterGenerationProvider["editMaskedTexture"]> {
    resolveTextureEditImageModel(input.imageModel);
    const endpoint = FAL_ENDPOINTS.textureEdit;
    const imageSize = parseImageSize(input.imageSize);
    const referenceUrls = await Promise.all(
      (input.referenceImagePaths ?? []).map((file) => this.upload(file)),
    );
    const request: Record<string, unknown> = {
      prompt: input.prompt,
      image_urls: [await this.upload(input.imagePath), ...referenceUrls],
      mask_url: await this.upload(input.maskPath),
      image_size: imageSize,
      quality: "high",
      num_images: 1,
      output_format: "png",
      background: "transparent",
    };
    const result = await this.run(endpoint, request, input.signal, input.onSubmitted, input.onStatus);
    return {
      endpoint,
      requestId: result.requestId,
      urls: imageUrlsFrom(result.data),
    };
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

  async cancel(endpoint: string, requestId: string): Promise<void> {
    // Allow cancellation of requests submitted before the model migration.
    if (![FAL_ENDPOINTS.textureEdit, "fal-ai/gpt-image-1.5/edit", "openai/gpt-image-2/edit"].includes(endpoint)) return;
    await this.client.queue.cancel(endpoint, { requestId });
  }

  private async upload(filePath: string): Promise<string> {
    const bytes = await this.readLocalFile(filePath);
    const file = new File([Buffer.from(bytes)], path.basename(filePath), {
      type: mimeType(filePath),
    });
    return this.client.storage.upload(file, { lifecycle: { expiresIn: "1h" } });
  }

  private async run(
    endpoint: string,
    input: Record<string, unknown>,
    signal: AbortSignal | undefined,
    onSubmitted: ((endpoint: string, requestId: string) => void | Promise<void>) | undefined,
    onStatus: ((status: CharacterProviderStatus) => void) | undefined,
  ): Promise<FalQueueResult> {
    try {
      const startedAt = Date.now();
      throwIfAborted(signal);
      const submitted = await this.client.queue.submit(endpoint, {
        input,
        storageSettings: { expiresIn: "1h" },
      });
      await onSubmitted?.(endpoint, submitted.request_id);
      onStatus?.({
        stage: "queued",
        queuePosition: null,
        message: null,
        elapsedMs: Date.now() - startedAt,
      });
      await this.client.queue.subscribeToStatus(endpoint, {
        requestId: submitted.request_id,
        abortSignal: signal,
        logs: true,
        onQueueUpdate: (status) => onStatus?.(providerStatus(status, startedAt)),
      });
      throwIfAborted(signal);
      const result = await this.client.queue.result(endpoint, {
        requestId: submitted.request_id,
        abortSignal: signal,
      });
      throwIfAborted(signal);
      return result;
    } catch (error) {
      if (signal?.aborted) throw error;
      throw new Error(falErrorMessage(error), { cause: error });
    }
  }
}

function parseImageSize(value: string): { width: number; height: number } {
  const match = /^(\d+)x(\d+)$/.exec(value);
  if (!match) throw new Error(`图片画布尺寸无效：${value}`);
  return { width: Number(match[1]), height: Number(match[2]) };
}

function removedFeature<T>(): Promise<T> {
  return Promise.reject(new Error("AI 创建角色和旧部位修改功能已移除，请使用纹理编辑"));
}

function imageUrlsFrom(value: unknown): string[] {
  const images = record(value).images;
  if (!Array.isArray(images)) throw new Error("fal 返回的图片列表无效");
  const urls = images.map((image) => {
    const url = record(image).url;
    if (typeof url !== "string" || !url.startsWith("https://")) {
      throw new Error("fal 返回了无效的临时图片地址");
    }
    return url;
  });
  if (urls.length === 0) throw new Error("fal 没有返回图片");
  return urls;
}

function falErrorMessage(error: unknown): string {
  if (typeof error !== "object" || error === null) return String(error);
  const value = error as Record<string, unknown>;
  const body = typeof value.body === "object" && value.body !== null
    ? value.body as Record<string, unknown>
    : null;
  const details = Array.isArray(body?.detail) ? body.detail : [];
  const messages = details.flatMap((detail) => {
    if (typeof detail !== "object" || detail === null) return [];
    const item = detail as Record<string, unknown>;
    const location = Array.isArray(item.loc)
      ? item.loc.filter((part): part is string | number =>
        typeof part === "string" || typeof part === "number"
      ).filter((part) => part !== "body").join(".")
      : "请求参数";
    const context = typeof item.ctx === "object" && item.ctx !== null
      ? item.ctx as Record<string, unknown>
      : null;
    if (item.type === "string_too_long" && typeof context?.max_length === "number") {
      return [`${location || "请求参数"} 超过 ${context.max_length} 个字符的接口上限`];
    }
    return typeof item.msg === "string" && item.msg.trim()
      ? [`${location || "请求参数"}：${item.msg.trim()}`]
      : [];
  });
  if (messages.length > 0) return `fal 参数校验失败：${messages.join("；")}`;
  return typeof value.message === "string" && value.message.trim()
    ? value.message.trim()
    : "fal 请求失败";
}

function providerStatus(status: FalQueueStatus, startedAt: number): CharacterProviderStatus {
  const lastMessage = status.logs?.map((log) => log.message?.trim())
    .filter((message): message is string => Boolean(message)).at(-1) ?? null;
  return {
    stage: status.status === "IN_QUEUE"
      ? "queued"
      : status.status === "COMPLETED"
        ? "completed"
        : "running",
    queuePosition: status.status === "IN_QUEUE" && Number.isInteger(status.queue_position)
      ? status.queue_position!
      : null,
    message: lastMessage,
    elapsedMs: Date.now() - startedAt,
  };
}

function record(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("fal 返回格式无效");
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
