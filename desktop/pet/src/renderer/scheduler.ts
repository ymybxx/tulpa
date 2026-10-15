import {
  PET_EMOTIONS,
  PET_STATES,
  type PetControlEvent,
  type PetEmotion,
  type PetSnapshotEvent,
  type PetState,
} from "../shared/contracts";
import { CLOSED_MOUTH, type PetRenderer } from "./pet-renderer";
import type { SpeechFacialRuntime } from "./speech-facial";

const TURN_START_STATES = new Set<PetState>([
  "acknowledging",
  "listening",
  "thinking",
]);

export class PetScheduler {
  private sequence = 0;
  private generation: string | null = null;
  private readonly cancelled = new Set<string>();
  private state: PetState = "unbound";
  private emotion: PetEmotion = "neutral";

  constructor(private readonly renderer: PetRenderer, private readonly facial?: SpeechFacialRuntime) {}

  apply(event: PetControlEvent): void {
    if (event.type === "pet.snapshot") {
      this.applySnapshot(event as PetSnapshotEvent);
      return;
    }
    if (!Number.isSafeInteger(event.sequence) || event.sequence <= this.sequence) return;
    this.sequence = event.sequence;

    if (event.type === "pet.generation.cancel") {
      if (event.generation) this.markCancelled(event.generation);
      if (!event.generation || event.generation === this.generation) {
        this.facial?.release();
        this.renderer.cancelGeneration();
        this.renderer.setLipShape(CLOSED_MOUTH);
        this.generation = null;
      }
      return;
    }

    if (event.type === "pet.state") {
      const state = enumValue(PET_STATES, event.payload.state);
      if (!state || !this.acceptStateGeneration(event.generation, state)) return;
      this.state = state;
      if (state === "listening" || state === "standby" || state === "unbound") this.facial?.release();
      if (state === "listening") this.renderer.cancelGeneration();
      this.renderer.setState(state);
      if (state === "listening") this.renderer.setGaze(0, 0);
      else if (state === "thinking") this.renderer.setGaze(-0.28, 0.12);
      if (state === "listening" || state === "thinking" || state === "standby") {
        this.renderer.setLipShape(CLOSED_MOUTH);
      }
      if (state === "listening" || state === "standby" || state === "unbound") {
        this.emotion = "neutral";
        this.renderer.setEmotion("neutral");
      }
      return;
    }

    if (event.type === "pet.facial") {
      if (!this.acceptCurrentGeneration(event.generation, false) || !event.generation) return;
      this.facial?.apply(event.payload);
    } else if (event.type === "pet.expression") {
      if (!this.acceptCurrentGeneration(event.generation, true)) return;
      const emotion = enumValue(PET_EMOTIONS, event.payload.emotion) ?? "neutral";
      this.emotion = emotion;
      this.renderer.setEmotion(emotion);
    } else if (event.type === "pet.audio.start") {
      this.acceptCurrentGeneration(event.generation, true);
    } else if (
      event.type === "pet.audio.pause" ||
      event.type === "pet.audio.stop"
    ) {
      if (!this.acceptCurrentGeneration(event.generation, false)) return;
      if (event.type === "pet.audio.stop") this.facial?.release();
      this.renderer.setLipShape(CLOSED_MOUTH);
    }
  }

  acceptsAudio(generation: string): boolean {
    return generation === this.generation && !this.cancelled.has(generation);
  }

  interact(): void {
    this.renderer.playClick();
  }

  currentGeneration(): string | null {
    return this.generation;
  }

  replay(): void {
    this.renderer.setState(this.state);
    this.renderer.setEmotion(this.emotion);
  }

  private applySnapshot(snapshot: PetSnapshotEvent): void {
    if (!Number.isSafeInteger(snapshot.sequence)) return;
    // A reconnect may target a freshly restarted API whose sequence begins at
    // zero again. WebSocket ordering guarantees this snapshot precedes later
    // events on that connection, so it is the new ordering authority.
    const events = Array.isArray(snapshot.payload.events)
      ? [...snapshot.payload.events].sort((left, right) => left.sequence - right.sequence)
      : [];
    this.sequence = 0;
    for (const event of events) this.apply(event);
    this.sequence = Math.max(this.sequence, snapshot.sequence);
  }

  private acceptStateGeneration(generation: string | null, state: PetState): boolean {
    if (!generation) {
      if (state === "standby" || state === "unbound" || state === "listening") {
        if (this.generation) this.markCancelled(this.generation);
        this.generation = null;
      }
      return true;
    }
    if (this.cancelled.has(generation)) return false;
    if (!this.generation) {
      this.generation = generation;
      return true;
    }
    if (generation === this.generation) return true;
    if (!TURN_START_STATES.has(state)) return false;
    this.facial?.release();
    this.markCancelled(this.generation);
    this.renderer.cancelGeneration();
    this.renderer.setLipShape(CLOSED_MOUTH);
    this.generation = generation;
    return true;
  }

  private acceptCurrentGeneration(
    generation: string | null,
    allowStart: boolean,
  ): boolean {
    if (!generation) return this.generation === null;
    if (this.cancelled.has(generation)) return false;
    if (!this.generation && allowStart) this.generation = generation;
    return generation === this.generation;
  }

  private markCancelled(generation: string): void {
    this.cancelled.add(generation);
    if (this.cancelled.size <= 64) return;
    const oldest = this.cancelled.values().next().value;
    if (oldest) this.cancelled.delete(oldest);
  }
}

function enumValue<T extends string>(values: readonly T[], value: unknown): T | null {
  return typeof value === "string" && values.includes(value as T) ? (value as T) : null;
}
