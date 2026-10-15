import type { OpenDialogOptions } from "electron";
import type { DesktopUiLocale } from "../src/shared/contracts";

/** "any" lets one picker take either a model folder or its .zip. */
export function modelDialogOptions(kind: "zip" | "folder" | "any", locale: DesktopUiLocale = "zh-CN"): OpenDialogOptions {
  const t = (zh: string, en: string): string => locale === "en" ? en : zh;
  if (kind === "folder") return { properties: ["openDirectory"] };
  const zip = { name: t("Live2D 模型文件夹或 ZIP", "Live2D model folder or ZIP"), extensions: ["zip"] };
  return kind === "any"
    ? { title: t("选一个 Live2D 模型文件夹或 .zip", "Choose a Live2D model folder or .zip"), properties: ["openFile", "openDirectory"], filters: [zip] }
    : { properties: ["openFile"], filters: [{ name: t("Live2D 模型 ZIP", "Live2D model ZIP"), extensions: ["zip"] }] };
}

export function referenceImageDialogOptions(locale: DesktopUiLocale = "zh-CN"): OpenDialogOptions {
  return {
    properties: ["openFile"],
    filters: [{ name: locale === "en" ? "Reference image" : "参考图片", extensions: ["png", "jpg", "jpeg", "webp"] }],
  };
}
