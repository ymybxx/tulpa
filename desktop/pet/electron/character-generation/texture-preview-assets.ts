import { createHash } from "node:crypto";

export interface TexturePreviewArtifactSnapshotter {
  snapshotImage(jobId: string, sourcePath: string, name: string): Promise<string>;
}

export async function snapshotTexturePreviewAssets(
  artifacts: TexturePreviewArtifactSnapshotter,
  input: {
    jobId: string;
    textureUnitId: string;
    revision: number;
    imagePath: string;
    maskPath: string;
  },
): Promise<{ imagePath: string; maskPath: string }> {
  if (!Number.isSafeInteger(input.revision) || input.revision <= 0) {
    throw new Error("纹理预览资源 revision 无效");
  }
  const textureUnitId = input.textureUnitId.trim();
  if (!textureUnitId) throw new Error("纹理预览资源缺少纹理单元 ID");
  const unitKey = createHash("sha256").update(textureUnitId).digest("hex").slice(0, 16);
  const suffix = `${unitKey}-${input.revision}`;
  const [imagePath, maskPath] = await Promise.all([
    artifacts.snapshotImage(input.jobId, input.imagePath, `preview-image-${suffix}`),
    artifacts.snapshotImage(input.jobId, input.maskPath, `preview-mask-${suffix}`),
  ]);
  return { imagePath, maskPath };
}
