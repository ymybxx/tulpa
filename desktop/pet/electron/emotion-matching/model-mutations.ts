/** Short edit transactions only; never held while sampling or waiting for a provider. */
export class ModelMutationQueue {
  private tails = new Map<string, Promise<void>>();
  run<T>(modelId: string, operation: () => Promise<T>): Promise<T> {
    const previous = this.tails.get(modelId) ?? Promise.resolve();
    const result = previous.catch(() => undefined).then(operation);
    const tail = result.then(() => undefined, () => undefined);
    this.tails.set(modelId, tail);
    void tail.then(() => { if (this.tails.get(modelId) === tail) this.tails.delete(modelId); });
    return result;
  }
}
