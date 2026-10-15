import { afterEach, describe, expect, it, vi } from "vitest";
import { facialDecisionEditor } from "./facial-decision-editor";

class Element {
  value = ""; textContent = ""; type = ""; hidden = false; disabled = false;
  maxLength = 0; className = "";
  dataset: Record<string, string> = {};
  children: Array<Element | string> = [];
  open = false; title = ""; style: Record<string, string> = {};
  listeners = new Map<string, () => void>();
  append(...children: Array<Element | string>) { this.children.push(...children); }
  replaceChildren(...children: Array<Element | string>) { this.children = children; }
  setAttribute() { /* Pressed state has no effect on these behavior tests. */ }
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

afterEach(() => vi.unstubAllGlobals());
describe("facial decision editor", () => {
  it("edits the steps, shows where each lands, and switches to yes/no", () => {
    vi.stubGlobal("document", { createElement: () => new Element() });
    const editor = facialDecisionEditor({ mode: "score", levels: ["没有", "淡", "明显"],
      anchors: [0, .5, 1], minimum: 0, maximum: 1, columns: { names: ["A", "B"], values: value => [value, value * 2] } });
    const root = editor.element as unknown as Element;
    expect(editor.read()).toEqual({ mode: "score", levels: ["没有", "淡", "明显"] });
    root.find("decisionAdd")!.fire("click");
    // A table: a heading of the parameters, then each step's row with where every parameter lands.
    const steps = root.findAll("decisionLevel");
    expect(steps.map(input => input.value)).toEqual(["没有", "淡", "", "明显"]);
    const rows = (root.find("decisionScore")!.children as Element[]).slice(0, 5);
    expect(rows.map(row => (row.children.slice(1, 3) as Element[]).map(cell => cell.textContent))).toEqual([
      ["A", "B"], ["0", "0"], ["0.333", "0.667"], ["0.667", "1.333"], ["1", "2"]]);
    steps[2]!.value = "中等"; steps[2]!.fire("input");
    expect(editor.read()).toEqual({ mode: "score", levels: ["没有", "淡", "中等", "明显"] });
    root.find("decisionModeBinary")!.fire("click");
    expect(root.find("decisionScore")!.hidden).toBe(true);
    expect(root.find("decisionBinary")!.hidden).toBe(false);
    expect(editor.read()).toEqual({ mode: "binary", falseDescription: "没有", trueDescription: "明显" });
  });
});
