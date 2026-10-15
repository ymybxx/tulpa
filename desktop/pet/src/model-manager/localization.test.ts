import { afterEach, expect, it, vi } from "vitest";
import type { DesktopUiLocale } from "../shared/contracts";

class TestElement {
  value = ""; textContent = ""; hidden = false; disabled = false;
  className = ""; isConnected = true;
  dataset: Record<string, string> = {};
  style: Record<string, string> = {};
  children: TestElement[] = [];
  attributes = new Map<string, string>();
  listeners = new Map<string, () => void>();
  append(...children: TestElement[]) { this.children.push(...children); }
  replaceChildren(...children: TestElement[]) { this.children = children; }
  setAttribute(name: string, value: string) { this.attributes.set(name, value); }
  getAttribute(name: string) { return this.attributes.get(name) ?? null; }
  addEventListener(event: string, listener: () => void) { this.listeners.set(event, listener); }
  all(key: string): TestElement[] {
    return [...(key in this.dataset ? [this] : []), ...this.children.flatMap(child => child.all(key))];
  }
}

afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

it("switches an open facial editor in place without replacing unsaved character wording or triggering a preview", async () => {
  vi.resetModules();
  let broadcast: (locale: DesktopUiLocale) => void = () => {};
  vi.stubGlobal("window", {
    desktopPet: {
      getUiLocale: async () => "zh-CN",
      onUiLocale: (callback: typeof broadcast) => { broadcast = callback; return () => {}; },
    },
    addEventListener() {},
  });
  vi.stubGlobal("document", { documentElement: { lang: "zh-CN" }, createElement: () => new TestElement() });
  const { initializeLocale } = await import("../i18n");
  await initializeLocale();
  const { facialDecisionEditor } = await import("./facial-decision-editor");
  const look = vi.fn();
  const editor = facialDecisionEditor({ mode: "score", levels: ["没有", "淡", "明显"],
    anchors: [0, 0.5, 1], minimum: 0, maximum: 1, columns: { names: ["用户参数"], values: value => [value] }, look });
  const root = editor.element as unknown as TestElement;
  const inputs = root.all("decisionLevel");
  inputs[1]!.value = "我未保存的说法";
  inputs[1]!.listeners.get("input")!();
  broadcast("en");
  expect(root.all("decisionModeScore")[0]!.textContent).toBe("Levels from low to high");
  expect(inputs[0]!.getAttribute("placeholder")).toBe("Description at minimum");
  expect(root.all("decisionLevel")).toEqual(inputs);
  expect(editor.read()).toEqual({ mode: "score", levels: ["没有", "我未保存的说法", "明显"] });
  expect(look).not.toHaveBeenCalled();
  broadcast("zh-CN");
  expect(root.all("decisionModeScore")[0]!.textContent).toBe("从弱到强分几档");
  expect(inputs[1]!.value).toBe("我未保存的说法");
});
