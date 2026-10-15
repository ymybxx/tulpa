import { describe, expect, it } from "vitest";

import type { ModelPackage } from "../shared/contracts";
import { textureUnitMotionNames } from "./character-wizard";

describe("action-only texture annotations", () => {
  it("resolves current motion names from stable asset IDs", () => {
    const model = {
      manifest: {
        motionAssets: {
          "motion-id-04": { id: "motion-id-04", name: "抬手", file: "mtn_04.motion3.json" },
        },
      },
    } as unknown as ModelPackage;

    expect(textureUnitMotionNames(["motion-id-04"], model)).toEqual(["抬手"]);
    model.manifest.motionAssets["motion-id-04"]!.name = "挥手问候";
    expect(textureUnitMotionNames(["motion-id-04"], model)).toEqual(["挥手问候"]);
  });
});
