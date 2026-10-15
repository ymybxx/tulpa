import { describe, expect, it, vi } from "vitest";
import { appMenuTemplate } from "./app-menu";

describe("application menu language", () => {
  it("uses English labels with the existing native roles and keyboard shortcut", () => {
    const open = vi.fn();
    const menu = appMenuTemplate("tulpa", open, "en");
    expect(menu.map((item) => item.label)).toEqual(["tulpa", "Edit", "Window"]);
    expect(menu[0]?.submenu).toContainEqual(expect.objectContaining({ label: "Settings…", accelerator: "CommandOrControl+,", click: open }));
    expect(menu[1]?.submenu).toContainEqual({ role: "copy", label: "Copy" });
    expect(appMenuTemplate("tulpa", open)[1]?.label).toBe("编辑");
  });
});
