/**
 * Asleep (待机) TA hears only the wake word, and looks no different from awake
 * TA. Worth saying how to wake TA when the person may miss that TA dozed off:
 * when TA first settles after starting, and when a voice conversation ends.
 * TA speaking up by itself and settling back is not one; nobody was talking.
 */
export class SleepWatch {
  private talking = false;
  private settled = false;

  /** The state TA moved to; true when TA just fell asleep unnoticed. */
  next(state: string | null): boolean {
    if (state === "acknowledging" || state === "listening") {
      this.talking = true;
      return false;
    }
    if (state !== "standby") return false;
    const say = this.talking || !this.settled;
    this.talking = false;
    this.settled = true;
    return say;
  }
}

/** The pet.state event itself, or the one a reconnect's replay carries. */
export function petStateEvent(event: unknown): { type: "pet.state"; payload: { state?: unknown } } | undefined {
  const value = event as { type?: unknown; payload?: { state?: unknown; events?: unknown } } | null;
  if (value?.type === "pet.state") return value as { type: "pet.state"; payload: { state?: unknown } };
  if (value?.type === "pet.snapshot" && Array.isArray(value.payload?.events)) {
    return [...value.payload.events].reverse().map(petStateEvent).find(Boolean);
  }
  return undefined;
}
