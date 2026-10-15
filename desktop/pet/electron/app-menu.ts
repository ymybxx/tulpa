import type { MenuItemConstructorOptions } from "electron";
import type { DesktopUiLocale } from "../src/shared/contracts";

/**
 * The menu bar follows the interface language instead of Electron's default;
 * only what a user of this app needs is kept. The 编辑 menu must stay: it is
 * what makes copy and paste work in text fields on macOS.
 */
export function appMenuTemplate(appName: string, openSettings: () => void, locale: DesktopUiLocale = "zh-CN"): MenuItemConstructorOptions[] {
  const t = (zh: string, en: string): string => locale === "en" ? en : zh;
  return [
    {
      label: appName,
      submenu: [
        { role: "about", label: t(`关于 ${appName}`, `About ${appName}`) },
        { type: "separator" },
        { label: t("设置…", "Settings…"), accelerator: "CommandOrControl+,", click: openSettings },
        { type: "separator" },
        { role: "hide", label: t(`隐藏 ${appName}`, `Hide ${appName}`) },
        { role: "hideOthers", label: t("隐藏其他", "Hide Others") },
        { role: "unhide", label: t("全部显示", "Show All") },
        { type: "separator" },
        { role: "quit", label: t(`退出 ${appName}`, `Quit ${appName}`) },
      ],
    },
    {
      label: t("编辑", "Edit"),
      submenu: [
        { role: "undo", label: t("撤销", "Undo") },
        { role: "redo", label: t("重做", "Redo") },
        { type: "separator" },
        { role: "cut", label: t("剪切", "Cut") },
        { role: "copy", label: t("拷贝", "Copy") },
        { role: "paste", label: t("粘贴", "Paste") },
        { role: "selectAll", label: t("全选", "Select All") },
      ],
    },
    {
      label: t("窗口", "Window"),
      submenu: [
        { role: "minimize", label: t("最小化", "Minimize") },
        { role: "close", label: t("关闭窗口", "Close Window") },
      ],
    },
  ];
}
