type Awaitable = void | Promise<void>;

export interface PerceptionDeactivationOperations {
  /** Stops the native capture owner. This operation is always invoked first. */
  stopPerception: (detail: string) => Awaitable;
  disableComputerHistory: () => Awaitable;
  disableEnvironmentFrames: () => Awaitable;
}

/**
 * Starts every perception shutdown operation in the same turn. Stopping the
 * native capture owner first prevents local recorder cleanup from delaying the
 * release of ScreenCaptureKit and system-audio resources.
 */
export function deactivatePerception(
  operations: PerceptionDeactivationOperations,
  detail: string,
): Promise<void> {
  const perceptionStop = invoke(() => operations.stopPerception(detail));
  const computerHistoryStop = invoke(operations.disableComputerHistory);
  const environmentFrameStop = invoke(operations.disableEnvironmentFrames);

  return Promise.all([
    perceptionStop,
    computerHistoryStop,
    environmentFrameStop,
  ]).then(() => undefined);
}

function invoke(operation: () => Awaitable): Promise<void> {
  try {
    return Promise.resolve(operation());
  } catch (error) {
    return Promise.reject(error);
  }
}
