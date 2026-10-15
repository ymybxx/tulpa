import { describe, expect, it, vi } from "vitest";

import { OpenAICompatibleImageProvider } from "./openai-compatible-image-provider";

describe("OpenAICompatibleImageProvider", () => {
  it("posts large-canvas Image 2.5 edits with transparency through the same endpoint", async () => {
    const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
      expect(url).toBe("https://image.example.test/v1/images/edits");
      expect(init?.headers).toEqual({
        Authorization: "Bearer sk-image",
      });
      const form = init?.body as FormData;
      expect(form.get("model")).toBe("gpt-image-2.5-sunburst");
      expect(form.get("prompt")).toBe("把帽子改成粉色小熊风格");
      expect(form.get("size")).toBe("3840x2160");
      expect(form.get("quality")).toBe("high");
      expect(form.get("output_format")).toBe("png");
      expect(form.get("background")).toBe("transparent");
      expect(form.get("input_fidelity")).toBeNull();
      const images = form.getAll("image[]") as File[];
      expect(images.map((file) => file.name)).toEqual(["work.png", "style-reference.png"]);
      expect(new Uint8Array(await images[0]!.arrayBuffer())).toEqual(new Uint8Array([1, 2, 3]));
      expect(new Uint8Array(await images[1]!.arrayBuffer())).toEqual(new Uint8Array([7, 8, 9]));
      expect(new Uint8Array(await (form.get("mask") as File).arrayBuffer())).toEqual(new Uint8Array([4, 5, 6]));
      return new Response(JSON.stringify({
        data: [{ b64_json: Buffer.from("result").toString("base64") }],
      }), { status: 200, headers: { "content-type": "application/json" } });
    });
    const submitted = vi.fn();
    const provider = new OpenAICompatibleImageProvider({
      baseUrl: "https://image.example.test/v1/",
      apiKey: "sk-image",
    }, {
      fetcher,
      readLocalFile: async (file) => new Uint8Array(
        file.includes("reference") ? [7, 8, 9] : file.includes("mask") ? [4, 5, 6] : [1, 2, 3],
      ),
      requestId: () => "local-request-1",
    });

    const result = await provider.editMaskedTexture({
      prompt: "把帽子改成粉色小熊风格",
      imagePath: "/tmp/work.png",
      maskPath: "/tmp/mask.png",
      imageSize: "3840x2160",
      imageModel: "gpt-image-2.5-sunburst",
      referenceImagePaths: ["/tmp/style-reference.png"],
      onSubmitted: submitted,
    });

    expect(fetcher).toHaveBeenCalledWith(
      "https://image.example.test/v1/images/edits",
      expect.objectContaining({
        method: "POST",
      }),
    );
    expect(submitted).toHaveBeenCalledWith(
      "openai-compatible:gpt-image-2.5-sunburst/images/edits",
      "local-request-1",
    );
    expect(result).toEqual({
      endpoint: "openai-compatible:gpt-image-2.5-sunburst/images/edits",
      requestId: "local-request-1",
      urls: [`data:image/png;base64,${Buffer.from("result").toString("base64")}`],
    });
  });

  it.each(["gpt-image-1.5", "gpt-image-2"] as const)("upgrades legacy %s requests to transparent Image 2.5 edits", async (legacyModel) => {
    const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
      expect(url).toBe("https://images.example.com/v1/images/edits");
      const form = init?.body as FormData;
      expect(form.get("model")).toBe("gpt-image-2.5-sunburst");
      expect(form.getAll("image[]")).toHaveLength(2);
      expect(form.getAll("image[]")[0]).toBeInstanceOf(Blob);
      expect(form.get("mask")).toBeInstanceOf(Blob);
      expect(form.get("background")).toBe("transparent");
      expect(form.get("input_fidelity")).toBeNull();
      return new Response(JSON.stringify({
        data: [{ b64_json: Buffer.from("transparent-result").toString("base64") }],
      }), { status: 200, headers: { "content-type": "application/json" } });
    });
    const provider = new OpenAICompatibleImageProvider({
      baseUrl: "https://images.example.com/v1",
      apiKey: "sk-image",
    }, {
      fetcher,
      readLocalFile: async () => new Uint8Array([1]),
    });

    await provider.editMaskedTexture({
      prompt: "去掉袜子并输出真实透明度",
      imagePath: "/tmp/work.png",
      maskPath: "/tmp/mask.png",
      imageSize: "1024x1536",
      imageModel: legacyModel,
      referenceImagePaths: ["/tmp/skin-reference.png"],
    });
  });

  it("surfaces an OpenAI-compatible error without falling back to another provider", async () => {
    const provider = new OpenAICompatibleImageProvider({
      baseUrl: "https://images.example.com/v1",
      apiKey: "sk-image",
    }, {
      fetcher: async (url) => {
        expect(url).toBe("https://images.example.com/v1/images/edits");
        return new Response(JSON.stringify({
        error: { message: "model is unavailable" },
        }), { status: 503 });
      },
      readLocalFile: async () => new Uint8Array([1]),
    });

    await expect(provider.editMaskedTexture({
      prompt: "edit",
      imagePath: "/tmp/work.png",
      maskPath: "/tmp/mask.png",
      imageSize: "1024x1024",
      imageModel: "gpt-image-2.5-sunburst",
    })).rejects.toThrow(/model is unavailable/);
  });
});
