import { describe, expect, it } from "vitest";

import { calculateVirtualGridWindow } from "./virtual-grid";

describe("calculateVirtualGridWindow", () => {
  it("keeps a 597-item texture library bounded to sixty rendered cards", () => {
    const value = calculateVirtualGridWindow({
      itemCount: 597,
      columns: 5,
      firstVisibleRow: 52,
      visibleRows: 4,
      overscanRows: 2,
      maxRenderedItems: 60,
    });
    expect(value.end - value.start).toBeLessThanOrEqual(60);
    expect(value.start).toBe(250);
    expect(value.rowCount).toBe(120);
  });
});
