import { describe, expect, it } from "vitest";

import type { ModelEditCapabilityManifest } from "../shared/contracts";
import { packageFixture } from "./test-fixture";
import {
  buildBehaviorRows,
  initialTextureFilterValue,
  textureUnitsForPart,
} from "./view-model";

describe("buildBehaviorRows", () => {
  it("shows state and emotion candidates with the unified shape", () => {
    const manifest = packageFixture().manifest;
    manifest.behaviors.states.click = [{
      id: "click-1",
      motionAssetIds: ["wave"],
      expressionAssetIds: ["smile"],
    }];

    const states = buildBehaviorRows(manifest, "state");
    expect(states.map((row) => row.key)).toEqual([
      "standby", "acknowledging", "listening", "thinking", "click",
    ]);
    expect(states.find((row) => row.key === "click")?.candidates[0]).toMatchObject({
      motionAssetIds: ["wave"],
      expressionAssetIds: ["smile"],
    });
    const emotions = buildBehaviorRows(manifest, "emotion");
    expect(emotions.map((row) => row.key)).toEqual([
      "neutral", "angry", "happy", "shy", "sad",
    ]);
    expect(emotions.find((row) => row.key === "happy")?.label).toBe("开心");
    expect(emotions.find((row) => row.key === "sad")?.label).toBe("伤心");
  });

  it("returns detached candidates so rendering cannot mutate the draft", () => {
    const manifest = packageFixture().manifest;
    manifest.behaviors.emotions.happy = [{ id: "happy", motionAssetIds: ["wave"], expressionAssetIds: [] }];
    const rows = buildBehaviorRows(manifest, "emotion");
    rows.find((row) => row.key === "happy")!.candidates[0]!.motionAssetIds.length = 0;
    expect(manifest.behaviors.emotions.happy[0]!.motionAssetIds).toEqual(["wave"]);
  });
});

describe("textureUnitsForPart", () => {
  it("uses the same semantic part mapping for AI editing and wardrobe filtering", () => {
    const capability = {
      textureUnits: [
        { id: "left-sleeve", semanticPartIds: ["arm-left", "clothing"] },
        { id: "hair", semanticPartIds: ["hair-front"] },
      ],
    } as ModelEditCapabilityManifest;

    expect(textureUnitsForPart(capability, "clothing").map((unit) => unit.id))
      .toEqual(["left-sleeve"]);
    expect(textureUnitsForPart(capability, "")).toHaveLength(2);
  });
});

describe("initialTextureFilterValue", () => {
  it("shows all textures from the general AI editor entry", () => {
    expect(initialTextureFilterValue(null)).toBe("");
  });

  it("keeps a specific part selected when editing from that part", () => {
    expect(initialTextureFilterValue("clothing")).toBe("clothing");
  });
});
