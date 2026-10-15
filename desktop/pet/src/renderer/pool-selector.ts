export class PoolSelector {
  private readonly previous = new Map<string, string>();

  constructor(private readonly random: () => number = Math.random) {}

  pick(key: string, ids: readonly string[]): string | null {
    if (!ids.length) return null;
    if (ids.length === 1) {
      this.previous.set(key, ids[0]!);
      return ids[0]!;
    }
    const previous = this.previous.get(key);
    const candidates = previous && ids.includes(previous)
      ? ids.filter((id) => id !== previous)
      : [...ids];
    const index = Math.min(
      candidates.length - 1,
      Math.floor(Math.max(0, this.random()) * candidates.length),
    );
    const selected = candidates[index]!;
    this.previous.set(key, selected);
    return selected;
  }

  clear(key?: string): void {
    if (key === undefined) this.previous.clear();
    else this.previous.delete(key);
  }
}
