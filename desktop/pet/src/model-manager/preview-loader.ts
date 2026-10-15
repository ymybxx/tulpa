import type { ModelManagerPreview } from "./preview";

export function loadPreviewConstructor(
  coreAvailable: boolean,
): Promise<typeof ModelManagerPreview>;
export function loadPreviewConstructor<T>(
  coreAvailable: boolean,
  load: () => Promise<{ ModelManagerPreview: T }>,
): Promise<T>;
export async function loadPreviewConstructor<T>(
  coreAvailable: boolean,
  load: () => Promise<{ ModelManagerPreview: T }> = () => import("./preview") as Promise<{ ModelManagerPreview: T }>,
): Promise<T> {
  if (!coreAvailable) {
    throw new Error("Live2D 运行资源尚未准备，仍可管理模型；保存设置后即可实时预览");
  }
  return (await load()).ModelManagerPreview;
}
