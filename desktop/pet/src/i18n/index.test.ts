import { afterEach, describe, expect, it, vi } from "vitest";

async function fixture(initial: "en" | "zh-CN" = "zh-CN") {
  vi.resetModules();
  let notify: (value: "en" | "zh-CN") => void = () => {};
  const api = {
    getUiLocale: vi.fn().mockResolvedValue(initial),
    onUiLocale: vi.fn((listener: typeof notify) => { notify = listener; return () => {}; }),
    setUiLocale: vi.fn(async (value: "en" | "zh-CN") => { notify(value); }),
  };
  vi.stubGlobal("window", { desktopPet: api, addEventListener: vi.fn() });
  vi.stubGlobal("document", { documentElement: { lang: "zh-CN" } });
  const locale = await import("./index");
  await locale.initializeLocale();
  return { locale, api, notify: (value: "en" | "zh-CN") => notify(value) };
}

afterEach(() => { vi.unstubAllGlobals(); });

describe("desktop UI language", () => {
  it("loads the persisted preference and follows broadcasts without replacing the page", async () => {
    const { locale, notify } = await fixture("en");
    expect(locale.t("设置")).toBe("Settings");
    expect(document.documentElement.lang).toBe("en");
    const listener = vi.fn();
    const stop = locale.subscribeLocale(listener);
    notify("zh-CN");
    expect(locale.t("设置")).toBe("设置");
    expect(document.documentElement.lang).toBe("zh-CN");
    expect(listener).toHaveBeenCalledTimes(1);
    stop();
    notify("en");
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("commits after saving and propagates failures without changing the selected language", async () => {
    const { locale, api } = await fixture();
    api.setUiLocale.mockRejectedValueOnce(new Error("disk full"));
    await expect(locale.changeLocale("en")).rejects.toThrow("disk full");
    expect(locale.getLocale()).toBe("zh-CN");
    await locale.changeLocale("en");
    expect(locale.getLocale()).toBe("en");
    expect(api.setUiLocale).toHaveBeenCalledTimes(2);
  });

  it("does not let a delayed initial read overwrite a newer broadcast", async () => {
    vi.resetModules();
    let finish!: (value: "zh-CN") => void;
    let notify!: (value: "en") => void;
    vi.stubGlobal("window", {
      desktopPet: {
        getUiLocale: () => new Promise((resolve) => { finish = resolve; }),
        onUiLocale: (listener: typeof notify) => { notify = listener; return () => {}; },
      }, addEventListener: vi.fn(),
    });
    vi.stubGlobal("document", { documentElement: { lang: "zh-CN" } });
    const locale = await import("./index");
    notify("en");
    finish("zh-CN");
    await locale.initializeLocale();
    expect(locale.getLocale()).toBe("en");
  });

  it("preserves interpolated user text and falls back for unknown keys", async () => {
    const { locale } = await fixture("en");
    const names = '<b>我的文件</b> & {{name}}';
    expect(locale.t("附件：{{names}}", { names })).toBe(`Attachments: ${names}`);
    expect(locale.t("没有翻译的开发诊断")).toBe("没有翻译的开发诊断");
  });

  it("keeps interpolation names consistent in every translation", async () => {
    const { locale } = await fixture();
    const fields = (value: string) => [...value.matchAll(/\{\{\s*([\w.]+)\s*\}\}/g)].map((match) => match[1]).sort();
    for (const [source, english] of Object.entries(locale.messages)) {
      expect(fields(english), source).toEqual(fields(source));
    }
  });

  it("updates bound DOM copy while keeping drafts and replacement content", async () => {
    const { notify } = await fixture();
    const { bindLocalizedText, bindLocalizedAttribute } = await import("./dom");
    class Element {
      textContent = "";
      value = "用户未保存草稿";
      attrs = new Map<string, string>();
      setAttribute(key: string, value: string) { this.attrs.set(key, value); }
      getAttribute(key: string) { return this.attrs.get(key) ?? null; }
    }
    const label = new Element() as unknown as HTMLElement;
    const placeholder = new Element() as unknown as HTMLInputElement;
    bindLocalizedText(label, "设置");
    bindLocalizedAttribute(placeholder, "placeholder", "想说什么？");
    notify("en");
    expect(label.textContent).toBe("Settings");
    expect(placeholder.getAttribute("placeholder")).toBe("What is on your mind?");
    expect(placeholder.value).toBe("用户未保存草稿");
    label.textContent = "用户命名的模型";
    notify("zh-CN");
    expect(label.textContent).toBe("用户命名的模型");
  });

  it("does not evaluate stale bindings on a discarded editor panel", async () => {
    const { notify } = await fixture();
    const { bindLocalizedText } = await import("./dom");
    const panel = { textContent: "", isConnected: true } as unknown as HTMLElement;
    let color: string | null = "#ffffff";
    bindLocalizedText(panel, "{{color}}", () => ({ color: color!.toUpperCase() }));
    Object.defineProperty(panel, "isConnected", { value: false });
    color = null;
    expect(() => notify("en")).not.toThrow();
  });
});
