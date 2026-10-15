import type {
  ModelRenderBundle,
  PetEmotion,
  PetState,
  TextureAtlasDelta,
  NativeModelParameter,
} from "../shared/contracts";
import type { FacialOverlayBindings } from "../shared/facial-bindings";

export interface LipShape {
  volume: number;
  a: number;
  e: number;
  i: number;
  o: number;
  u: number;
  /** The voice's pitch in Hz while it is voiced, else 0; absent where it is not measured. */
  pitch?: number;
}

export interface PetModelBounds {
  x: number;
  y: number;
  width: number;
  height: number;
  /** The body's left and right edges halfway down, where the pointer usually lands. */
  middle?: { y: number; left: number; right: number };
}

export interface PetRenderer {
  nativeParameters?(): NativeModelParameter[];
  setSpeechFacialEnabled?(enabled: boolean): void;
  beginFacialOverlay?(bindings: FacialOverlayBindings): void;
  setFacialDemoTargets?(targets: Record<string, number>): void;
  endFacialOverlay?(): void;
  loadModel(value: ModelRenderBundle): Promise<void>;
  updateTextureAtlases(delta: TextureAtlasDelta): Promise<void>;
  setState(value: PetState): void;
  setEmotion(value: PetEmotion): void;
  playClick(): void;
  setLipShape(value: LipShape): void;
  setGaze(x: number, y: number, source?: "pointer" | "ambient"): void;
  getModelBounds(): PetModelBounds | null;
  setModelOffset?(x: number, y: number): void;
  setModelZoom?(zoom: number): void;
  setTextureMaximumDimension?(maximum: number): void;
  isPointInDragArea(x: number, y: number): boolean;
  cancelGeneration(): void;
  resize(): void;
  setPaused(value: boolean): void;
  destroy(): void;
}

export const CLOSED_MOUTH: LipShape = {
  volume: 0,
  a: 0,
  e: 0,
  i: 0,
  o: 0,
  u: 0,
};
