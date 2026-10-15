import { describe, expect, it, vi } from "vitest";

import {
  FAL_ENDPOINTS,
  FalCharacterGenerationProvider,
  type FalClientPort,
} from "./fal-provider";

function client(): FalClientPort {
  return {
    storage: { upload: vi.fn(async () => "https://storage.example/input.png") },
    queue: {
      submit: vi.fn(async () => ({ request_id: "request-1" })),
      subscribeToStatus: vi.fn(async () => ({ status: "COMPLETED" })),
      result: vi.fn(async () => ({
        data: { images: [{ url: "https://cdn.example/result.png" }] },
        requestId: "request-1",
      })),
      cancel: vi.fn(async () => undefined),
    },
  };
}

describe("FalCharacterGenerationProvider", () => {
  it("uses the Image 2.5 endpoint and its mask_url schema with transparent output", async () => {
    const mock = client();
    const provider = new FalCharacterGenerationProvider("fal-key", {
      createClient: () => mock,
      readLocalFile: async () => new Uint8Array([1, 2, 3]),
    });
    const submitted = vi.fn();

    const result = await provider.editMaskedTexture({
      prompt: "只修改已选 Live2D 纹理",
      imagePath: "/tmp/work.png",
      maskPath: "/tmp/mask.png",
      imageSize: "1536x1024",
      imageModel: "gpt-image-2.5-sunburst",
      referenceImagePaths: ["/tmp/style.png"],
      onSubmitted: submitted,
    });

    expect(mock.queue.submit).toHaveBeenCalledTimes(1);
    expect(mock.queue.submit).toHaveBeenCalledWith(
      FAL_ENDPOINTS.textureEdit,
      expect.objectContaining({
        input: {
          prompt: "只修改已选 Live2D 纹理",
          image_urls: ["https://storage.example/input.png", "https://storage.example/input.png"],
          mask_url: "https://storage.example/input.png",
          image_size: { width: 1536, height: 1024 },
          quality: "high",
          background: "transparent",
          num_images: 1,
          output_format: "png",
        },
      }),
    );
    expect(submitted).toHaveBeenCalledWith(FAL_ENDPOINTS.textureEdit, "request-1");
    expect(result).toEqual({
      endpoint: FAL_ENDPOINTS.textureEdit,
      requestId: "request-1",
      urls: ["https://cdn.example/result.png"],
    });
  });

  it.each(["gpt-image-1.5", "gpt-image-2"] as const)("upgrades %s to Image 2.5 with both a large canvas and transparency", async (legacyModel) => {
    const mock = client();
    const provider = new FalCharacterGenerationProvider("fal-key", {
      createClient: () => mock,
      readLocalFile: async () => new Uint8Array([1, 2, 3]),
    });

    await provider.editMaskedTexture({
      prompt: "只修改已选 Live2D 纹理",
      imagePath: "/tmp/work.png",
      maskPath: "/tmp/mask.png",
      imageSize: "2048x1152",
      imageModel: legacyModel,
    });

    expect(mock.queue.submit).toHaveBeenCalledWith(
      FAL_ENDPOINTS.textureEdit,
      expect.objectContaining({
        input: expect.objectContaining({
          image_size: { width: 2048, height: 1152 },
          quality: "high",
          output_format: "png",
        }),
      }),
    );
    const input = vi.mocked(mock.queue.submit).mock.calls[0]![1].input;
    expect(input.background).toBe("transparent");
    expect(input).not.toHaveProperty("mask_image_url");
    expect(input).not.toHaveProperty("input_fidelity");
  });

  it("rejects removed AI-character and old part-edit operations without sending requests", async () => {
    const mock = client();
    const provider = new FalCharacterGenerationProvider("fal-key", {
      createClient: () => mock,
    });

    await expect(provider.generateCandidates()).rejects.toThrow(/功能已移除/);
    await expect(provider.editImage()).rejects.toThrow(/功能已移除/);
    await expect(provider.inpaintImage()).rejects.toThrow(/功能已移除/);
    await expect(provider.removeBackground()).rejects.toThrow(/功能已移除/);
    expect(mock.queue.submit).not.toHaveBeenCalled();
  });

  it("does not forward cancellation for removed endpoints", async () => {
    const mock = client();
    const provider = new FalCharacterGenerationProvider("fal-key", {
      createClient: () => mock,
    });

    await provider.cancel("fal-ai/removed", "request-1");
    expect(mock.queue.cancel).not.toHaveBeenCalled();
    await provider.cancel(FAL_ENDPOINTS.textureEdit, "request-1");
    expect(mock.queue.cancel).toHaveBeenCalledWith(FAL_ENDPOINTS.textureEdit, {
      requestId: "request-1",
    });
    for (const endpoint of ["fal-ai/gpt-image-1.5/edit", "openai/gpt-image-2/edit"]) {
      await provider.cancel(endpoint, "legacy-request");
      expect(mock.queue.cancel).toHaveBeenCalledWith(endpoint, { requestId: "legacy-request" });
    }
  });
});
