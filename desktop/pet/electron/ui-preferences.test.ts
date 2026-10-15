import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { registerUiLocaleIpc, UiPreferencesStore } from "./ui-preferences";

const directories: string[] = [];
async function directory(): Promise<string> {
  const value = await mkdtemp(path.join(tmpdir(), "tulpa-ui-locale-"));
  directories.push(value);
  return value;
}
afterEach(async () => { await Promise.all(directories.splice(0).map((dir) => rm(dir, { recursive: true, force: true }))); });

describe("interface language preferences", () => {
  it("defaults to Chinese for existing installs, missing files and invalid values", async () => {
    const dir = await directory();
    const store = new UiPreferencesStore(dir);
    await store.load();
    expect(store.uiLocale).toBe("zh-CN");
    for (const value of ['{"locale":"fr"}', '{"locale":false}', 'null', '{broken']) {
      await writeFile(path.join(dir, "ui-preferences.json"), value);
      await store.load();
      expect(store.uiLocale).toBe("zh-CN");
    }
  });

  it("persists the selection without changing runtime configuration or character prompts", async () => {
    const dir = await directory();
    const configuration = '{"configured":true,"audio":{"wakeWord":"小星"}}';
    await writeFile(path.join(dir, "config.json"), configuration);
    const store = new UiPreferencesStore(dir);
    await store.setLocale("en");
    const reopened = new UiPreferencesStore(dir);
    await reopened.load();
    expect(reopened.uiLocale).toBe("en");
    expect(await readFile(path.join(dir, "config.json"), "utf8")).toBe(configuration);
    await expect(store.setLocale("fr")).rejects.toThrow("Unsupported interface language");
    expect(store.uiLocale).toBe("en");
  });

  it("keeps rapid toggles in order on disk and recovers after a write failure", async () => {
    const dir = await directory();
    const store = new UiPreferencesStore(dir);
    await Promise.all([store.setLocale("en"), store.setLocale("zh-CN"), store.setLocale("en")]);
    expect(JSON.parse(await readFile(path.join(dir, "ui-preferences.json"), "utf8")).locale).toBe("en");
    const target = path.join(dir, "ui-preferences.json");
    await rm(target);
    const { mkdir } = await import("node:fs/promises");
    await mkdir(target);
    await expect(store.setLocale("zh-CN")).rejects.toThrow();
    expect(store.uiLocale).toBe("en");
    await rm(target, { recursive: true });
    await store.setLocale("zh-CN");
    expect(store.uiLocale).toBe("zh-CN");
  });

  it("notifies all live windows after saving and restricts changes to the main frame", async () => {
    const dir = await directory();
    const store = new UiPreferencesStore(dir);
    const handlers = new Map<string, (event: any, value?: unknown) => unknown>();
    const contents = () => ({ mainFrame: {}, send: vi.fn(), isDestroyed: () => false });
    const main = contents(); const pet = contents(); const hud = contents();
    const closed = { ...contents(), isDestroyed: () => true };
    const refreshMenus = vi.fn();
    registerUiLocaleIpc({
      handle: (channel, handler) => { handlers.set(channel, handler); }, store,
      mainContents: () => main, allContents: () => [main, pet, hud, closed], refreshMenus,
    });
    const set = handlers.get("desktop:ui-locale:set")!;
    await expect(set({ sender: pet, senderFrame: pet.mainFrame }, "en")).rejects.toThrow("main window");
    await expect(set({ sender: main, senderFrame: {} }, "en")).rejects.toThrow("main window");
    expect(refreshMenus).not.toHaveBeenCalled();
    await set({ sender: main, senderFrame: main.mainFrame }, "en");
    expect(refreshMenus).toHaveBeenCalledOnce();
    for (const item of [main, pet, hud]) expect(item.send).toHaveBeenCalledWith("desktop:ui-locale", "en");
    expect(closed.send).not.toHaveBeenCalled();
    expect(handlers.get("desktop:ui-locale:get")!({})).toBe("en");
    expect(JSON.parse(await readFile(path.join(dir, "ui-preferences.json"), "utf8")).locale).toBe("en");
  });
});
