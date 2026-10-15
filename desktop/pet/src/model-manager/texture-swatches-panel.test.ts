import { describe, expect, it } from "vitest";

import type { ModelEditCapabilityManifest, PetManifest, TextureMaterial, TextureMaterialLibrary } from "../shared/contracts";
import { partSwatches, wholeOutfits } from "./texture-swatches-panel";

const generatedAt = "2026-09-25T00:00:00.000Z";
const material = (id: string, unit: string, createdAt: string, prompt = ""): TextureMaterial => ({
  id, textureUnitId: unit, label: unit, atlasIndex: 0, atlasFile: "a.png", bounds: { x: 0, y: 0, width: 10, height: 10 },
  name: prompt ? "AI 素材" : `${unit} 素材`, prompt, createdAt, imageUrl: `pet-model://m/${id}.png`, active: false,
});
const capability = {
  generatedAt,
  parts: [
    { id: "hair", label: "头发", group: "hair-back", drawableIds: ["d1", "d2"] },
    { id: "coat", label: "外套", group: "clothing", drawableIds: ["d3"] },
  ],
  textureUnits: [
    { id: "u1", drawableIds: ["d1"], partIds: [] },
    { id: "u2", drawableIds: ["d2"], partIds: [] },
    { id: "u3", drawableIds: ["d3"], partIds: [] },
  ],
} as unknown as ModelEditCapabilityManifest;

describe("partSwatches", () => {
  it("offers the original and each saved AI result as one version per part", () => {
    const library = {
      materials: [
        material("o1", "u1", generatedAt), material("o2", "u2", generatedAt), material("o3", "u3", generatedAt),
        material("p1", "u1", "2026-09-25T10:00:00.000Z", "樱粉色"), material("p2", "u2", "2026-09-25T10:00:00.000Z", "樱粉色"),
      { ...material("q1", "u1", "2026-09-26T10:00:00.000Z", "任务：编辑输入图中的 A 区域…"), original: false },
      ],
      activeAssignments: { u1: "p1", u2: "p2", u3: "o3" },
    } as unknown as TextureMaterialLibrary;
    const parts = partSwatches({} as PetManifest, capability, library);
    // 外套 has only its original, so it isn't offered.
    expect(parts.map((part) => part.name)).toEqual(["头发"]);
    expect(parts[0]!.swatches.filter((swatch) => !swatch.label.startsWith("AI 版本")).map((swatch) => [swatch.label, swatch.ai, [...swatch.materials.keys()]])).toEqual([
      ["原版", false, ["u1", "u2"]],
      ["樱粉色", true, ["u1", "u2"]],
    ]);
    // A version saved with the compiled prompt is named by its time instead.
    expect(parts[0]!.swatches.length).toBe(3);
    expect(parts[0]!.swatches.some((swatch) => swatch.label.startsWith("AI 版本"))).toBe(true);
  });

  it("lists every part for recoloring and names saved colors by their color", () => {
    const library = {
      materials: [
        material("o1", "u1", generatedAt), material("o2", "u2", generatedAt), material("o3", "u3", generatedAt),
        { ...material("c3", "u3", "2026-09-27T10:00:00.000Z", "#F4A7C0"), origin: "recolor" as const },
      ],
      activeAssignments: { u1: "o1", u2: "o2", u3: "c3" },
    } as unknown as TextureMaterialLibrary;
    expect(partSwatches({} as PetManifest, capability, library).map((part) => part.name)).toEqual(["外套"]);
    const all = partSwatches({} as PetManifest, capability, library, true);
    expect(all.map((part) => [part.name, part.unitIds])).toEqual([["头发", ["u1", "u2"]], ["外套", ["u3"]]]);
    expect(all[1]!.swatches.map((swatch) => [swatch.label, swatch.ai])).toEqual([["原版", false], ["樱粉", false]]);
  });

  it("puts on a whole AI result at once: its version where it painted, the original for that part's other pieces", () => {
    const library = {
      materials: [
        material("o1", "u1", generatedAt), material("o2", "u2", generatedAt), material("o3", "u3", generatedAt),
        // 赛博朋克 painted one hair piece and the coat; 和风 painted both hair pieces.
        material("c1", "u1", "2026-10-07T07:41:00.000Z", "赛博朋克风：外套改成黑色机能风"), material("c3", "u3", "2026-10-07T07:41:00.000Z", "赛博朋克风：外套改成黑色机能风"),
        material("w1", "u1", "2026-10-07T07:48:00.000Z", "用户要求：和风巫女，红白配色"), material("w2", "u2", "2026-10-07T07:48:00.000Z", "用户要求：和风巫女，红白配色"),
      ],
      activeAssignments: { u1: "w1", u2: "w2", u3: "o3" },
    } as unknown as TextureMaterialLibrary;
    const outfits = wholeOutfits(partSwatches({} as PetManifest, capability, library, true));
    expect(outfits.map((outfit) => [outfit.label, outfit.assignments])).toEqual([
      ["和风巫女", [{ unitId: "u1", materialId: "w1" }, { unitId: "u2", materialId: "w2" }]],
      ["赛博朋克风", [{ unitId: "u1", materialId: "c1" }, { unitId: "u2", materialId: "o2" }, { unitId: "u3", materialId: "c3" }]],
    ]);
  });
});
