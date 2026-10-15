/**
 * Minimal compile-time surface for the proprietary Cubism Core loaded at runtime.
 * The implementation is downloaded from Live2D and is intentionally not vendored.
 */
declare namespace Live2DCubismCore {
  type csmLogFunction = (message: string) => void;
  type csmParameterType = number;

  class Version {
    static csmGetVersion(): number;
    static csmGetLatestMocVersion(): number;
    static csmGetMocVersion(value: ArrayBuffer): number;
  }

  class Logging {
    static csmSetLogFunction(value: csmLogFunction): void;
    static csmGetLogFunction(): csmLogFunction | null;
  }

  class Memory {
    static initializeAmountOfMemory(size: number): void;
  }

  class Moc {
    [key: string]: any;
    static fromArrayBuffer(value: ArrayBuffer): Moc | null;
    hasMocConsistency(value: ArrayBuffer): number;
    _release(): void;
  }

  class Model {
    [key: string]: any;
    static fromMoc(value: Moc): Model | null;
  }

  class Utils {
    static [key: string]: any;
  }

  const ColorBlendType_Normal: number;
  const ColorBlendType_AddGlow: number;
  const ColorBlendType_Add: number;
  const ColorBlendType_Darken: number;
  const ColorBlendType_Multiply: number;
  const ColorBlendType_ColorBurn: number;
  const ColorBlendType_LinearBurn: number;
  const ColorBlendType_Lighten: number;
  const ColorBlendType_Screen: number;
  const ColorBlendType_ColorDodge: number;
  const ColorBlendType_Overlay: number;
  const ColorBlendType_SoftLight: number;
  const ColorBlendType_HardLight: number;
  const ColorBlendType_LinearLight: number;
  const ColorBlendType_Hue: number;
  const ColorBlendType_Color: number;
  const ColorBlendType_AddCompatible: number;
  const ColorBlendType_MultiplyCompatible: number;
}
