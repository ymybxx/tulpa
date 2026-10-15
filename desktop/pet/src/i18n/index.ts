import { createInstance } from "i18next";
import type { DesktopUiLocale } from "../shared/contracts";
import { settingsMessages } from "./messages/settings";
import { primaryMessages } from "./messages/primary";
import { desktopMessages } from "./messages/desktop";
import { editorMessages } from "./messages/editor";

export const messages: Record<string, string> = { ...desktopMessages, ...settingsMessages, ...primaryMessages, ...editorMessages };
const engine = createInstance();
void engine.init({
  lng: "zh-CN", fallbackLng: false, supportedLngs: ["zh-CN", "en"],
  resources: { en: { translation: messages }, "zh-CN": { translation: {} } },
  initImmediate: false, keySeparator: false, nsSeparator: false,
  interpolation: { escapeValue: false },
});

let locale: DesktopUiLocale = "zh-CN";
const listeners = new Set<() => void>();
export const getLocale = (): DesktopUiLocale => locale;

/** Translate only app-owned copy. Never pass chat, prompts or editable user data here. */
export function t(source: string, values?: Record<string, string | number>): string {
  return engine.t(source, { ...values, lng: locale, defaultValue: source });
}

export function subscribeLocale(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function applyLocale(value: DesktopUiLocale): void {
  if (value !== "en" && value !== "zh-CN") return;
  if (typeof document !== "undefined") document.documentElement.lang = value;
  if (locale === value) return;
  locale = value;
  for (const listener of listeners) listener();
}

type LocaleBridge = {
  getUiLocale(): Promise<DesktopUiLocale>;
  onUiLocale(callback: (value: DesktopUiLocale) => void): () => void;
  setUiLocale?(value: DesktopUiLocale): Promise<void>;
};

let bridge: LocaleBridge | undefined;
let initialized: Promise<void> | undefined;
/** Subscribe before reading, so a delayed initial read cannot undo a newer broadcast. */
export function initializeLocale(): Promise<void> {
  if (initialized) return initialized;
  if (typeof window === "undefined") return Promise.resolve();
  const candidates = window as unknown as { desktopPet?: LocaleBridge; petChat?: LocaleBridge };
  bridge = candidates.desktopPet ?? candidates.petChat;
  if (!bridge?.getUiLocale || !bridge.onUiLocale) return Promise.resolve();
  let revision = 0;
  const stop = bridge.onUiLocale((value) => { revision += 1; applyLocale(value); });
  window.addEventListener("beforeunload", stop, { once: true });
  const initialRevision = revision;
  initialized = bridge.getUiLocale().then((value) => {
    if (revision === initialRevision) applyLocale(value);
  }).catch(() => undefined);
  return initialized;
}

export async function changeLocale(value: DesktopUiLocale): Promise<void> {
  if (value !== "en" && value !== "zh-CN") throw new Error("Unsupported UI language");
  await initializeLocale();
  if (!bridge?.setUiLocale) throw new Error(t("暂时无法保存语言设置"));
  // The main process persists first and broadcasts the committed preference.
  await bridge.setUiLocale(value);
}

void initializeLocale();
