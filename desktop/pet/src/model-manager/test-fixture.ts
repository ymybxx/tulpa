import type { ModelPackage, PetManifest } from "../shared/contracts";

export function packageFixture(): ModelPackage {
  const manifest = {
    schemaVersion: 3,
    id: "girl",
    name: "Girl",
    renderer: "live2d-cubism",
    entry: "girl.model3.json",
    display: { scale: 1, x: 0.5, y: 1 },
    motionAssets: {
      wave: { id: "wave", name: "挥手", file: "wave.motion3.json", group: "Wave", index: 0, contentHash: "a" },
    },
    expressionAssets: {
      smile: { id: "smile", name: "微笑", file: "smile.exp3.json", expression: "smile", contentHash: "b" },
    },
    behaviors: {
      states: { standby: [], acknowledging: [], listening: [], thinking: [], click: [] },
      emotions: { neutral: [], angry: [], happy: [], shy: [], sad: [] },
    },
    lipsync: {
      mode: "open-form",
      source: "automatic",
      needsConfirmation: false,
      mouthOpenParameter: "ParamMouthOpenY",
      mouthFormParameter: null,
      jawParameter: null,
      puckerWidenParameter: null,
      funnelParameter: null,
      vowelParameters: { a: null, e: null, i: null, o: null, u: null },
      maxOpen: 0.7,
    },
  } satisfies PetManifest;
  return {
    id: "girl",
    name: "Girl",
    url: "pet-model://girl/girl.model3.json",
    previewUrl: null,
    manifest,
    capabilities: { motionGroups: ["Wave"], expressions: ["smile"], hitAreas: [] },
  };
}
