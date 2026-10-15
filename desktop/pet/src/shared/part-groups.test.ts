import { describe, expect, it } from "vitest";

import type { ModelEditCapabilityManifest, TextureEditUnit } from "./contracts";
import {
  automaticPartGroups, moveUnitsToGroup, normalizePartGroups, ungroupedUnits, unitsMatchingName, unitsOfGroup,
} from "./part-groups";

const unit = (id: string, drawableIds: string[], searchText = ""): TextureEditUnit => ({ id, drawableIds, partIds: [], searchText } as unknown as TextureEditUnit);
const capability = {
  parts: [
    { id: "hair-back", label: "后发", group: "hair-back", drawableIds: ["a", "b"] },
    { id: "other-part3", label: "中2", group: "unknown", drawableIds: ["c"] },
  ],
  textureUnits: [unit("u1", ["a"], "ArtMesh1 Part2 后发"), unit("u2", ["b"]), unit("u3", ["c"], "ArtMesh9 Part3 水印")],
} as unknown as ModelEditCapabilityManifest;

describe("part groups", () => {
  it("groups recognized parts and leaves the rest for recognition", () => {
    const groups = automaticPartGroups(capability);
    expect(groups).toEqual([{ id: "hair-back", name: "后发", drawableIds: ["a", "b"] }]);
    expect(ungroupedUnits(groups, capability).map((item) => item.id)).toEqual(["u3"]);
    expect(unitsOfGroup(groups[0]!, capability).map((item) => item.id)).toEqual(["u1", "u2"]);
  });

  it("keeps a drawable in one group and cleans names", () => {
    expect(normalizePartGroups([
      { id: "x", name: " 头发 ", drawableIds: ["a", "a"] },
      { id: "y", name: "也有a", drawableIds: ["a", "b"] },
      { id: "z", name: " ", drawableIds: ["c"] },
    ])).toEqual([{ id: "x", name: "头发", drawableIds: ["a"] }, { id: "y", name: "也有a", drawableIds: ["b"] }]);
  });

  it("moves pieces between groups and finds them by name", () => {
    const groups = [{ id: "g1", name: "后发", drawableIds: ["a", "b"] }, { id: "g2", name: "水印", drawableIds: [] }];
    const moved = moveUnitsToGroup(groups, "g2", [capability.textureUnits[0]!]);
    expect(moved).toEqual([{ id: "g1", name: "后发", drawableIds: ["b"] }, { id: "g2", name: "水印", drawableIds: ["a"] }]);
    expect(unitsMatchingName(capability, "水印").map((item) => item.id)).toEqual(["u3"]);
    expect(unitsMatchingName(capability, "artmesh").map((item) => item.id)).toEqual(["u1", "u3"]);
  });

  it("leaves never-drawn clipping masks out of every list", () => {
    const withMask = {
      ...capability,
      textureUnits: [...capability.textureUnits, { ...unit("mask", ["b", "d"], "ArtMesh292 裙子"), maskOnly: true }],
    } as ModelEditCapabilityManifest;
    const groups = automaticPartGroups(withMask);
    expect(unitsOfGroup(groups[0]!, withMask).map((item) => item.id)).toEqual(["u1", "u2"]);
    expect(ungroupedUnits(groups, withMask).map((item) => item.id)).toEqual(["u3"]);
    expect(unitsMatchingName(withMask, "裙子")).toEqual([]);
  });
});
