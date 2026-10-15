import { describe, expect, it, vi } from "vitest";

import { snapshotTexturePreviewAssets } from "./texture-preview-assets";

describe("snapshotTexturePreviewAssets", () => {
  it("gives every preview revision immutable image and mask paths", async () => {
    const snapshotImage = vi.fn(async (_jobId: string, _source: string, name: string) =>
      `/job/${name}.png`
    );
    const artifacts = { snapshotImage };

    const first = await snapshotTexturePreviewAssets(artifacts, {
      jobId: "job-1",
      textureUnitId: "body",
      revision: 1,
      imagePath: "/mutable/body.png",
      maskPath: "/mutable/mask.png",
    });
    const second = await snapshotTexturePreviewAssets(artifacts, {
      jobId: "job-1",
      textureUnitId: "body",
      revision: 2,
      imagePath: "/mutable/body.png",
      maskPath: "/mutable/mask.png",
    });

    expect(first).toEqual({
      imagePath: "/job/preview-image-230d8358dc8e8890-1.png",
      maskPath: "/job/preview-mask-230d8358dc8e8890-1.png",
    });
    expect(second).toEqual({
      imagePath: "/job/preview-image-230d8358dc8e8890-2.png",
      maskPath: "/job/preview-mask-230d8358dc8e8890-2.png",
    });
    expect(second).not.toEqual(first);
    expect(snapshotImage).toHaveBeenCalledTimes(4);
  });

  it("rejects an invalid resource revision", async () => {
    await expect(snapshotTexturePreviewAssets({ snapshotImage: vi.fn() }, {
      jobId: "job-1",
      textureUnitId: "body",
      revision: 0,
      imagePath: "/mutable/body.png",
      maskPath: "/mutable/mask.png",
    })).rejects.toThrow("revision 无效");
  });
});
