export async function runModelLoadTransaction<T>(input: {
  prepare(): Promise<T>;
  validate(candidate: T): void | Promise<void>;
  commit(candidate: T): void | Promise<void>;
  release(candidate: T): void;
  isCurrent(): boolean;
}): Promise<T | null> {
  const candidate = await input.prepare();
  if (!input.isCurrent()) {
    input.release(candidate);
    return null;
  }
  try {
    await input.validate(candidate);
    if (!input.isCurrent()) {
      input.release(candidate);
      return null;
    }
    await input.commit(candidate);
    return candidate;
  } catch (error) {
    input.release(candidate);
    throw error;
  }
}
