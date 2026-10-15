import type { ModelRenderBundle } from "./contracts";

/** Keys are enabled standard or explicitly configured model-native parameter IDs. */
export type Face = Record<string, number>;
export type DemoEvent = { type: string; id?: string; [key: string]: unknown };
export interface FacialDemoBridge {
  init(): Promise<{ bundle: ModelRenderBundle; output: string; blushParameter: string; physicsOutputParameterIds: string[]; cases: Array<{ id: string; text: string; context: string }> }>;
  command(value: Record<string, unknown>): void;
  onEvent(callback: (event: DemoEvent) => void): () => void;
  save(name: string, bytes: Uint8Array): Promise<string>;
}
declare global { interface Window { facialDemo: FacialDemoBridge; } }
