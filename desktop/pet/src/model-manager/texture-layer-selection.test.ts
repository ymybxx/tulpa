import { describe, expect, it } from "vitest";
import type { TextureEditUnit } from "../shared/contracts";
import { TextureLayerSelection } from "./texture-layer-selection";

const unit = (id: string, drawableIds: string[]) => ({ id, drawableIds }) as TextureEditUnit;

describe("exact texture layer selection", () => {
  it("selects only the front editable mesh and keeps overlapping alternatives in draw order", () => {
    const selector = new TextureLayerSelection([unit("body", ["Body"]), unit("sleeve", ["Sleeve"])]);
    const pick = selector.pick(["NonEditable", "Sleeve", "Body"]);
    expect(pick.front.map(unit => unit.id)).toEqual(["sleeve"]);
    expect(pick.matched.map(unit => unit.id)).toEqual(["sleeve", "body"]);
    expect(selector.pick([])).toEqual({ matched: [], front: [] });
  });
  it("preserves shared UV units and highlights every mesh affected by an edit", () => {
    const selector = new TextureLayerSelection([
      unit("shared-sock", ["LeftSock", "RightSock"]), unit("trim", ["LeftSock"]), unit("body", ["Body"]),
    ]);
    const pick = selector.pick(["LeftSock", "RightSock", "Body", "LeftSock"]);
    expect(pick.front.map(unit => unit.id)).toEqual(["shared-sock", "trim"]);
    expect(pick.matched.map(unit => unit.id)).toEqual(["shared-sock", "trim", "body"]);
    expect(selector.layers(["shared-sock", "trim", "missing"])).toEqual(["LeftSock", "RightSock"]);
  });
});
