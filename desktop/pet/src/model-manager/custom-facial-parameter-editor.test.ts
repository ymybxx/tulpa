import { afterEach, describe, expect, it, vi } from "vitest";
import { customFacialParameterEditor } from "./custom-facial-parameter-editor";
import type { FacialParameterSuggestion } from "../shared/facial-test";

class Element {
  value = ""; textContent = ""; type = ""; hidden = false; disabled = false;
  checked = false; maxLength = 0; className = ""; rows = 0;
  min = ""; max = ""; step = ""; placeholder = ""; isConnected = true;
  dataset: Record<string, string> = {};
  children: Array<Element | string> = [];
  open = false; title = ""; style: Record<string, string> = {};
  listeners = new Map<string, () => void>();
  append(...children: Array<Element | string>) { this.children.push(...children); }
  replaceChildren(...children: Array<Element | string>) { this.children = children; }
  setAttribute() { /* Accessible status has no effect on this behavior test. */ }
  addEventListener(event: string, listener: () => void) { this.listeners.set(event, listener); }
  fire(event: string) { this.listeners.get(event)?.(); }
  find(key: string): Element | undefined {
    if (Object.hasOwn(this.dataset, key)) return this;
    for (const child of this.children) { if (typeof child === "string") continue; const found = child.find(key); if (found) return found; }
  }
  findAll(key: string): Element[] {
    return [...(Object.hasOwn(this.dataset, key) ? [this] : []), ...this.children.flatMap(child => typeof child === "string" ? [] : child.findAll(key))];
  }
}

const native = [{ id: "ParamCustom", name: "原名", minimum: -1, defaultValue: 0, maximum: 1 }];
afterEach(() => vi.unstubAllGlobals());

describe("custom facial parameter editor", () => {
  it("confirms how the part is split before the AI runs, then fills a reviewable draft without saving", async () => {
    vi.stubGlobal("document", { createElement: () => new Element() });
    const save = vi.fn(async () => undefined);
    const suggest = vi.fn(async (): Promise<FacialParameterSuggestion> => ({ parameterId: "ParamCustom",
      confidence: "clear", mode: "score", label: "眉尾", description: "眉尾如何变化？",
      levels: ["明显下压", "略下压", "自然", "上挑"], note: "已比较画面" }));
    const root = customFacialParameterEditor({ native, blocked: [], usedIds: [], preview: vi.fn(),
      suggest, save, close: vi.fn() }) as unknown as Element;
    // Step 1: the parameter, with the choice of AI or by hand; no form yet.
    expect(root.find("customParameterId")).toBeDefined();
    expect(root.find("customParameterSave")).toBeUndefined();
    root.find("customParameterSuggest")!.fire("click");
    // Step 2: how to split, confirmed before anything runs.
    expect(suggest).not.toHaveBeenCalled();
    const count = root.find("aiLevelCount")!; count.value = "4"; count.fire("change");
    root.find("customParameterSuggestStart")!.fire("click");
    await vi.waitFor(() => expect(suggest).toHaveBeenCalledWith("ParamCustom", "score", 4));
    await vi.waitFor(() => expect(root.find("customParameterLabel")?.value).toBe("眉尾"));
    expect(root.find("customParameterDescription")!.value).toBe("眉尾如何变化？");
    expect(root.findAll("decisionLevel").map(input => input.value)).toEqual(["明显下压", "略下压", "自然", "上挑"]);
    expect(save).not.toHaveBeenCalled();
    root.find("customParameterSave")!.fire("click");
    await vi.waitFor(() => expect(save).toHaveBeenCalledWith(expect.objectContaining({
      id: "ParamCustom", mode: "score", label: "眉尾", levels: ["明显下压", "略下压", "自然", "上挑"], enabled: true,
    })));
  });

  it("asks for yes/no wording when that is what was confirmed, and can be filled by hand instead", async () => {
    vi.stubGlobal("document", { createElement: () => new Element() });
    const suggest = vi.fn(async (): Promise<FacialParameterSuggestion> => ({ parameterId: "ParamCustom", confidence: "clear",
      mode: "binary", label: "挑眉", description: "是否挑眉？", falseDescription: "不挑眉", trueDescription: "挑眉", note: "" }));
    const root = customFacialParameterEditor({ native, blocked: [], usedIds: [], preview: vi.fn(),
      suggest, save: vi.fn(), close: vi.fn() }) as unknown as Element;
    root.find("customParameterSuggest")!.fire("click");
    root.find("aiModeBinary")!.fire("click");
    expect(root.find("aiLevelCount")).toBeUndefined();
    root.find("customParameterSuggestStart")!.fire("click");
    await vi.waitFor(() => expect(suggest).toHaveBeenCalledWith("ParamCustom", "binary", 2));
    await vi.waitFor(() => expect(root.find("decisionFalse")?.value).toBe("不挑眉"));
    expect(root.find("decisionTrue")!.value).toBe("挑眉");

    const manual = customFacialParameterEditor({ native, blocked: [], usedIds: [], preview: vi.fn(),
      suggest, save: vi.fn(), close: vi.fn() }) as unknown as Element;
    manual.find("customParameterManual")!.fire("click");
    expect(manual.find("customParameterSave")).toBeDefined();
    expect(manual.find("customParameterLabel")!.value).toBe("原名");
  });
});
