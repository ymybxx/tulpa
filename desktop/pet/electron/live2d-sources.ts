/** Where the default look and the Live2D runtime come from; shared by the app and packaging. */

export const CUBISM_SDK_VERSION = "5-r.5";
export const CUBISM_SDK_ARCHIVE_URL =
  `https://cubism.live2d.com/sdk-web/bin/CubismSdkForWeb-${CUBISM_SDK_VERSION}.zip`;
export const CUBISM_SDK_ARCHIVE_SHA256 =
  "67064a7fb1812cf502f5c4a03bfe12cc638c75a621bb4acf06bb28763df06ba0";
export const CORE_ARCHIVE_ENTRY =
  `CubismSdkForWeb-${CUBISM_SDK_VERSION}/Core/live2dcubismcore.min.js`;
export const SAMPLE_COMMIT = "ed1e0b714826d92469b9e51cacc3346f4e393f03";
export const SAMPLE_ROOT = `https://raw.githubusercontent.com/Live2D/CubismWebSamples/${SAMPLE_COMMIT}/Samples/Resources/Mao`;
/** Where the app bundle keeps its copy, relative to the bundled models directory. */
export const BUNDLED_LIVE2D_DIR = "live2d";
export const BUNDLED_CORE = "live2dcubismcore.min.js";
export const BUNDLED_MODEL_DIR = "Mao";

export const MODEL_FILES = [
  "Mao.2048/texture_00.png",
  "Mao.cdi3.json",
  "Mao.moc3",
  "Mao.model3.json",
  "Mao.physics3.json",
  "Mao.pose3.json",
  ...Array.from({ length: 8 }, (_, index) => `expressions/exp_0${index + 1}.exp3.json`),
  "motions/mtn_01.motion3.json",
  "motions/mtn_02.motion3.json",
  "motions/mtn_03.motion3.json",
  "motions/mtn_04.motion3.json",
  "motions/sample_01.motion3.json",
  "motions/special_01.motion3.json",
  "motions/special_02.motion3.json",
  "motions/special_03.motion3.json",
];

export function isCompatibleR5CoreSource(content: string): boolean {
  return content.length >= 100_000 &&
    content.includes("Live2D Cubism Core") &&
    content.includes("MocVersion_53=6") &&
    content.includes("ColorBlendType_Overlay") &&
    content.includes("csmGetLatestMocVersion");
}
