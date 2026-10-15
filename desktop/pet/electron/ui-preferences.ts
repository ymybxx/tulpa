import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import type { DesktopUiLocale } from "../src/shared/contracts";
import { LatestAtomicFileWriter } from "./latest-atomic-file-writer";

/** Display preferences stay separate from the AI/runtime configuration. */
export class UiPreferencesStore {
  private readonly file: string;
  private readonly writer: LatestAtomicFileWriter;
  private pending: Promise<void> = Promise.resolve();
  private locale: DesktopUiLocale = "zh-CN";

  constructor(private readonly userDataPath: string) {
    this.file = path.join(userDataPath, "ui-preferences.json");
    this.writer = new LatestAtomicFileWriter(this.file);
  }

  get uiLocale(): DesktopUiLocale { return this.locale; }

  async load(): Promise<void> {
    try {
      const value: unknown = JSON.parse(await readFile(this.file, "utf8"));
      this.locale = value && typeof value === "object" && "locale" in value && value.locale === "en"
        ? "en" : "zh-CN";
    } catch (error) {
      if (!(error instanceof SyntaxError) && (error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      this.locale = "zh-CN";
    }
  }

  async setLocale(value: unknown): Promise<void> {
    if (value !== "zh-CN" && value !== "en") throw new Error("Unsupported interface language");
    // Complete changes in request order, including persistence and notification.
    const write = this.pending.then(async () => {
      if (this.locale === value) return;
      await mkdir(this.userDataPath, { recursive: true });
      await this.writer.write(JSON.stringify({ version: 1, locale: value }));
      this.locale = value;
    });
    this.pending = write.catch(() => undefined);
    await write;
  }
}

interface UiContents {
  mainFrame: unknown;
  send(channel: string, value: DesktopUiLocale): void;
  isDestroyed(): boolean;
}
interface UiLocaleIpcEvent { sender: unknown; senderFrame: unknown }

/** The main settings window writes; every existing and future window can read. */
export function registerUiLocaleIpc(options: {
  handle(channel: string, handler: (event: UiLocaleIpcEvent, value?: unknown) => unknown): void;
  store: UiPreferencesStore;
  mainContents(): UiContents | undefined;
  allContents(): UiContents[];
  refreshMenus(): void;
}): void {
  options.handle("desktop:ui-locale:get", () => options.store.uiLocale);
  options.handle("desktop:ui-locale:set", async (event, value) => {
    const main = options.mainContents();
    if (!main || event.sender !== main || event.senderFrame !== main.mainFrame) {
      throw new Error("Interface language can only be changed from the main window");
    }
    await options.store.setLocale(value);
    options.refreshMenus();
    for (const contents of options.allContents()) {
      if (!contents.isDestroyed()) contents.send("desktop:ui-locale", options.store.uiLocale);
    }
  });
}
