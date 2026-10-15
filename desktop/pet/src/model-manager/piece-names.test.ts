import { describe, expect, it } from "vitest";

import type { TextureEditUnit } from "../shared/contracts";
import { pieceSections } from "./piece-names";

const unit = (id: string, size: number, searchText: string): TextureEditUnit =>
  ({ id, searchText, modelBounds: { x: 0, y: 0, width: size, height: size } } as unknown as TextureEditUnit);

describe("piece sections", () => {
  it("sorts strands into the author's sub-parts and names them by side and length", () => {
    const sections = pieceSections([
      unit("tail-l", 0.2, "ArtMesh1 Part1 左 Part2 双马尾短 Part3 后发 Part4 部件后"),
      unit("low-l-long", 0.5, "ArtMesh2 Part5 左 Part6 长 Part7 后发下 Part3 后发 Part4 部件后"),
      unit("low-l-long-2", 0.4, "ArtMesh3 Part5 左 Part6 长 Part7 后发下 Part3 后发 Part4 部件后"),
      unit("low-r-short", 0.3, "ArtMesh4 Part8 右 Part9 短 Part7 后发下 Part3 后发 Part4 部件后"),
      unit("nape", 0.6, "ArtMesh5 Part10 后脑勺 Part3 后发 Part4 部件后"),
      unit("nape-2", 0.1, "ArtMesh6 Part10 后脑勺 Part3 后发 Part4 部件后"),
      unit("loose", 0.05, "ArtMesh7 Part3 后发 Part4 部件后"),
    ], "后发");
    expect(sections.map((section) => [section.name, section.pieces.map((piece) => `${piece.unit.id}=${piece.name}`)])).toEqual([
      ["后发下", ["low-l-long=长 · 左 1", "low-l-long-2=长 · 左 2", "low-r-short=短 · 右"]],
      ["后脑勺", ["nape=后脑勺 1", "nape-2=后脑勺 2"]],
      ["双马尾短", ["tail-l=左"]],
      ["后发", ["loose=后发"]],
    ]);
  });

  it("uses the part's own name when everything sits in one place", () => {
    const sections = pieceSections([unit("a", 0.3, "ArtMesh1 Part1 裙子"), unit("b", 0.2, "ArtMesh2 Part1 裙子")], "裙子");
    expect(sections).toHaveLength(1);
    expect(sections[0]!.pieces.map((piece) => piece.name)).toEqual(["裙子 1", "裙子 2"]);
  });
});

describe("pieces not on screen", () => {
  it("go last in their own section and keep their sub-part name", () => {
    const sections = pieceSections([
      unit("shown", 0.3, "ArtMesh1 Part1 后发上 Part2 后发"),
      unit("long-l", 0.5, "ArtMesh2 Part3 左 Part4 长 Part5 后发下 Part2 后发"),
      unit("long-l-2", 0.4, "ArtMesh3 Part3 左 Part4 长 Part5 后发下 Part2 后发"),
    ], "后发", (item) => item.id === "shown");
    expect(sections.map((section) => [section.name, section.hidden ?? false, section.pieces.map((piece) => piece.name)])).toEqual([
      ["后发上", false, ["后发"]],
      ["现在看不到的", true, ["后发下 · 长 · 左 1", "后发下 · 长 · 左 2"]],
    ]);
  });
});
