import { describe, expect, it } from "vitest";

import type { TextureAtlasDelta } from "../shared/contracts";
import {
  assertTextureStreamTransitionHasState,
  shouldApplyTextureSnapshot,
} from "./texture-stream-order";

describe("shouldApplyTextureSnapshot", () => {
  it("rejects stale and duplicate revisions within one stream", () => {
    expect(shouldApplyTextureSnapshot("model:mao", 8, {
      streamId: "model:mao",
      revision: 8,
    })).toBe(false);
    expect(shouldApplyTextureSnapshot("model:mao", 8, {
      streamId: "model:mao",
      revision: 7,
    })).toBe(false);
    expect(shouldApplyTextureSnapshot("model:mao", 8, {
      streamId: "model:mao",
      revision: 9,
    })).toBe(true);
  });

  it("accepts an explicit switch to another snapshot stream", () => {
    expect(shouldApplyTextureSnapshot("job:old", 99, {
      streamId: "model:mao",
      revision: 2,
    })).toBe(true);
  });
});

describe("assertTextureStreamTransitionHasState", () => {
  it("rejects a cross-stream empty snapshot that cannot clear an old overlay", () => {
    expect(() => assertTextureStreamTransitionHasState("job:preview", {
      modelId: "mao",
      streamId: "model:mao",
      revision: 0,
      atlases: [],
      patchSets: [],
    })).toThrow("必须包含明确的图集状态");
  });

  it("allows an explicit empty patch set to restore the base atlas", () => {
    expect(() => assertTextureStreamTransitionHasState(
      "job:preview",
      explicitBaseSnapshot(),
    )).not.toThrow();
  });

  it("allows an empty revision advance inside the current stream", () => {
    expect(() => assertTextureStreamTransitionHasState("model:mao", {
      modelId: "mao",
      streamId: "model:mao",
      revision: 3,
      atlases: [],
      patchSets: [],
    })).not.toThrow();
  });
});

function explicitBaseSnapshot(): TextureAtlasDelta {
  return {
    modelId: "mao",
    streamId: "model:mao",
    revision: 2,
    atlases: [],
    patchSets: [{
      atlasIndex: 0,
      atlasFile: "textures/atlas.png",
      sourceWidth: 4,
      sourceHeight: 4,
      baseUrl: "pet-model://mao/textures/atlas.png?textureBase=1",
      patches: [],
    }],
  };
}
