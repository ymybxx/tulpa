import { t } from "./index";

/** Store app copy and its values, then translate when rendering the current locale. */
export type UiMessage = string | { key: string; values?: Record<string, string | number> };

export function messageText(message: UiMessage): string {
  return typeof message === "string" ? t(message) : t(message.key, message.values);
}
