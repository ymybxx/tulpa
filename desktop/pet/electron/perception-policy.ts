import type { DesktopPerceptionConfiguration } from "../src/shared/contracts";
import type { PerceptionModuleSelection } from "./perception-manager";

/** The explicit policy sent to the local API; effective screen capture is derived. */
export interface PerceptionPolicy {
  operationEnabled: boolean;
  screenEnabled: boolean;
  audioEnabled: boolean;
  audioSupported?: boolean;
  /** Sent only when off: the API then keeps no audiovisual summaries. */
  summariesEnabled?: false;
}

export const DISABLED_PERCEPTION_POLICY: Readonly<PerceptionPolicy> = Object.freeze({
  operationEnabled: false,
  screenEnabled: false,
  audioEnabled: false,
});

export function perceptionPolicy(
  configuration: DesktopPerceptionConfiguration,
  audioSupported = true,
): PerceptionPolicy {
  const summaries = configuration.innerLifeEnabled !== false;
  return {
    operationEnabled: configuration.operationPerceptionEnabled,
    screenEnabled: configuration.screenPerceptionEnabled,
    // Computer sound is only ever used for summaries, so it is not captured without them.
    audioEnabled: configuration.audioPerceptionEnabled && audioSupported && summaries,
    ...(!audioSupported ? { audioSupported: false } : {}),
    ...(!summaries ? { summariesEnabled: false as const } : {}),
  };
}

export function perceptionFramesEnabled(policy: PerceptionPolicy): boolean {
  return policy.screenEnabled;
}

export function perceptionHistoryEnabled(policy: PerceptionPolicy): boolean {
  return policy.summariesEnabled !== false && (policy.screenEnabled || policy.audioEnabled);
}

export function perceptionModules(
  policy: PerceptionPolicy,
  excludeFishMemoryFromScreenCapture = false,
): PerceptionModuleSelection {
  return {
    workspace: false,
    screen: perceptionFramesEnabled(policy),
    accessibility: false,
    camera: false,
    enhancedVision: false,
    computerHistory: policy.operationEnabled,
    systemAudio: policy.audioEnabled,
    legacyScreenAnalysis: false,
    excludeHostApplicationFromCapture: excludeFishMemoryFromScreenCapture,
  };
}
