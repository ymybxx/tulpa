/**
 * Orders model-bundle arrivals before they enter the renderer.
 *
 * The initial bundle is obtained through an async snapshot query while later
 * selections arrive as pushed commands. A pushed command is authoritative as
 * soon as it arrives, even if the older snapshot query resolves afterwards.
 */
export class ModelBundleArrivalGate {
  private epoch = 0;

  capture(): number {
    return this.epoch;
  }

  advance(): number {
    this.epoch += 1;
    return this.epoch;
  }

  isCurrent(epoch: number): boolean {
    return epoch === this.epoch;
  }
}

export type ModelBundleArrivalResult<T> =
  | { current: true; value: T }
  | { current: false };

/**
 * Runs one asynchronous stage only while its arrival is current, and checks
 * again after completion before the caller is allowed to publish side effects.
 */
export async function runCurrentModelBundleArrival<T>(
  gate: ModelBundleArrivalGate,
  epoch: number,
  operation: () => Promise<T>,
): Promise<ModelBundleArrivalResult<T>> {
  if (!gate.isCurrent(epoch)) return { current: false };
  const value = await operation();
  if (!gate.isCurrent(epoch)) return { current: false };
  return { current: true, value };
}
