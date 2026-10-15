/**
 * Resolves independent resource preparations as one ownership transaction.
 * If any preparation fails, every resource that did finish is released before
 * the original failure is rethrown.
 */
export async function allOrRelease<T>(
  preparations: readonly Promise<T>[],
  release: (value: T) => void,
): Promise<T[]> {
  const settled = await Promise.allSettled(preparations);
  const failure = settled.find(
    (result): result is PromiseRejectedResult => result.status === "rejected",
  );
  if (!failure) {
    return settled.map((result) => (result as PromiseFulfilledResult<T>).value);
  }
  for (const result of settled) {
    if (result.status === "fulfilled") release(result.value);
  }
  throw failure.reason;
}

/**
 * Waits for every sibling operation before surfacing the first failure.
 *
 * This is required when successful siblings attach resources to a shared
 * owner: the owner can only be released after no sibling can mutate it again.
 */
export async function allSettledOrThrow<T>(
  operations: readonly Promise<T>[],
): Promise<T[]> {
  const settled = await Promise.allSettled(operations);
  const failure = settled.find(
    (result): result is PromiseRejectedResult => result.status === "rejected",
  );
  if (failure) throw failure.reason;
  return settled.map((result) => (result as PromiseFulfilledResult<T>).value);
}
