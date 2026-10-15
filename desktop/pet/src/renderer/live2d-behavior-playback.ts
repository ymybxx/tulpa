import type { BehaviorCandidate } from "../shared/contracts";

export interface BehaviorRuntimeStartOptions {
  loopMotions: boolean;
  /** Keep looping while the runtime measures one full state-candidate cycle. */
  finishAfterOneCycle?: boolean;
  expressionOnlyDurationMs?: number;
  onFinished?: () => void;
}

export interface StateBehaviorSelection {
  candidate: BehaviorCandidate | null;
  rotate: boolean;
}

type TransientSource = "emotion" | "click" | "acknowledgement";

interface DeferredEmotion {
  candidate: BehaviorCandidate;
  label: string;
}

export interface Live2DBehaviorRuntime {
  startBehavior(candidate: BehaviorCandidate, options: BehaviorRuntimeStartOptions): void;
  stopBehavior(): void;
  /** State pose baseline, separate from a transient's suspended state pose. */
  beginStatePose?(): void;
  restoreStatePose?(): void;
  /** Snapshot/restore CubismPose selectors around temporary behavior layers. */
  beginTransientPose?(): void;
  restoreTransientPose?(): void;
  /** Optional face-only speech overlay support. */
  isFaceOnlyExpressionAsset?(assetId: string): boolean;
  startPersistentExpressions?(assetIds: readonly string[]): void;
  stopPersistentExpressions?(): void;
}

/**
 * Owns the state/transient lifecycle above the Cubism layer. The runtime owns
 * the actual parallel managers; this class guarantees that a transient group
 * restores the exact previously selected state candidate instead of drawing a
 * new random candidate.
 */
export class Live2DBehaviorPlayback {
  private stateCandidate: BehaviorCandidate | null = null;
  private selectNextState: (() => StateBehaviorSelection) | null = null;
  private speaking = false;
  private transientToken = 0;
  private transientActive = false;
  private transientSource: TransientSource = "emotion";
  // A click or wake acknowledgement temporarily owns the visual behavior. Keep
  // only the latest emotion, and retain the prior face context without replaying
  // its already-started body motion when the gesture ends.
  private deferredEmotion: DeferredEmotion | null = null;
  private emotionFaceExpressionAssetIds: string[] = [];
  private speechExpressionActive = false;
  private transientFaceExpressionAssetIds: string[] = [];
  private pendingSpeechExpressionAssetIds: string[] = [];
  private stateRunToken = 0;

  constructor(
    private readonly getRuntime: () => Live2DBehaviorRuntime | null,
    private readonly warn: (message: string, error?: unknown) => void = (message, error) =>
      console.warn(message, error ?? ""),
  ) {}

  setState(
    candidate: BehaviorCandidate | null,
    selectNext?: () => StateBehaviorSelection,
  ): boolean {
    ++this.stateRunToken;
    this.stateCandidate = candidate ? structuredClone(candidate) : null;
    this.selectNextState = selectNext ?? null;
    if (this.speaking || this.transientActive) return true;
    return this.startState();
  }

  setSpeaking(value: boolean): void {
    if (this.speaking === value) return;
    this.speaking = value;
    if (value) {
      ++this.stateRunToken;
      this.stopSpeechExpressions();
      this.pendingSpeechExpressionAssetIds = this.transientActive
        ? [...this.transientFaceExpressionAssetIds]
        : [];
      if (!this.transientActive && this.pendingSpeechExpressionAssetIds.length === 0) {
        this.stopBehavior();
      }
      return;
    }
    this.stopSpeechExpressions();
    this.pendingSpeechExpressionAssetIds = [];
    if (!this.transientActive) {
      this.getRuntime()?.restoreTransientPose?.();
      this.transientFaceExpressionAssetIds = [];
      this.startState();
    }
  }

  playTransient(candidate: BehaviorCandidate, label: string): boolean {
    const runtime = this.getRuntime();
    if (!runtime) return false;
    this.emotionFaceExpressionAssetIds = this.faceExpressionAssetIds(runtime, candidate);
    if (this.hasInteraction()) {
      this.deferredEmotion = { candidate: structuredClone(candidate), label };
      return true;
    }
    return this.startTransient(candidate, label, "emotion");
  }

  playClick(candidate: BehaviorCandidate): boolean {
    return this.startTransient(candidate, "state:click", "click");
  }

  playAcknowledgement(candidate: BehaviorCandidate): boolean {
    this.clearEmotionContext();
    return this.startTransient(candidate, "state:acknowledging", "acknowledgement");
  }

  cancelAcknowledgement(): void {
    if (this.transientActive && this.transientSource === "acknowledgement") {
      this.cancelTransient();
    }
  }

  /** Voice turnover must not truncate a pointer gesture or wake acknowledgement. */
  cancelEmotion(): void {
    this.clearEmotionContext();
    if (this.hasInteraction()) return;
    this.cancelTransient();
  }

  private startTransient(candidate: BehaviorCandidate, label: string, source: TransientSource): boolean {
    const runtime = this.getRuntime();
    if (!runtime) return false;
    runtime.beginTransientPose?.();
    ++this.stateRunToken;
    const token = ++this.transientToken;
    const storedCandidate = structuredClone(candidate);
    this.transientSource = source;
    this.transientFaceExpressionAssetIds = source === "emotion"
      ? this.faceExpressionAssetIds(runtime, storedCandidate)
      : [];
    this.pendingSpeechExpressionAssetIds = [];
    let transientCandidate = structuredClone(storedCandidate);
    try {
      if (source !== "emotion") {
        // Interaction expressions have the same bounded lifetime as their motion,
        // including while speaking. Do not promote them into the speech overlay.
        this.stopSpeechExpressions();
      } else if (this.speaking) {
        this.stopSpeechExpressions();
        this.startSpeechExpressions(this.transientFaceExpressionAssetIds);
        if (this.transientFaceExpressionAssetIds.length > 0) {
          transientCandidate = {
            ...transientCandidate,
            expressionAssetIds: transientCandidate.expressionAssetIds.filter(
              (id) => !this.transientFaceExpressionAssetIds.includes(id),
            ),
          };
        }
      }
      if (!transientCandidate.motionAssetIds.length && !transientCandidate.expressionAssetIds.length) {
        this.stopBehavior();
        this.transientActive = false;
        return true;
      }
      runtime.startBehavior(transientCandidate, {
        loopMotions: false,
        expressionOnlyDurationMs: 2_000,
        onFinished: () => this.finishTransient(token),
      });
      this.transientActive = true;
      return true;
    } catch (error) {
      runtime.restoreTransientPose?.();
      this.stopSpeechExpressions();
      this.warn(`启动 Live2D 行为候选失败：${label}`, error);
      this.transientActive = false;
      this.transientFaceExpressionAssetIds = [];
      this.pendingSpeechExpressionAssetIds = [];
      if (source !== "emotion") this.resumeAfterInteraction();
      else if (!this.speaking) this.startState();
      else this.stopBehavior();
      return false;
    }
  }

  hasTransient(): boolean {
    return this.transientActive;
  }

  hasSpeakingEmotion(): boolean {
    return this.transientActive || this.speechExpressionActive ||
      this.pendingSpeechExpressionAssetIds.length > 0;
  }

  cancelTransient(): void {
    this.clearEmotionContext();
    ++this.transientToken;
    if (
      !this.transientActive &&
      !this.speechExpressionActive &&
      this.pendingSpeechExpressionAssetIds.length === 0
    ) return;
    ++this.stateRunToken;
    this.transientActive = false;
    this.transientFaceExpressionAssetIds = [];
    this.pendingSpeechExpressionAssetIds = [];
    this.stopSpeechExpressions();
    this.getRuntime()?.restoreTransientPose?.();
    if (this.speaking) this.stopBehavior();
    else this.startState();
  }

  reset(): void {
    this.clearEmotionContext();
    ++this.transientToken;
    ++this.stateRunToken;
    this.transientActive = false;
    this.transientFaceExpressionAssetIds = [];
    this.pendingSpeechExpressionAssetIds = [];
    this.stopSpeechExpressions();
    this.getRuntime()?.restoreTransientPose?.();
    if (this.speaking) this.stopBehavior();
    else this.startState();
  }

  cancel(): void {
    this.clearEmotionContext();
    ++this.transientToken;
    ++this.stateRunToken;
    this.transientActive = false;
    this.transientFaceExpressionAssetIds = [];
    this.pendingSpeechExpressionAssetIds = [];
    this.stateCandidate = null;
    this.selectNextState = null;
    this.stopSpeechExpressions();
    this.getRuntime()?.restoreTransientPose?.();
    this.stopBehavior();
  }

  destroy(): void {
    this.cancel();
  }

  private finishTransient(token: number): void {
    if (!this.transientActive || token !== this.transientToken) return;
    const wasInteraction = this.transientSource !== "emotion";
    const faceExpressionAssetIds = this.pendingSpeechExpressionAssetIds.length > 0
      ? [...this.pendingSpeechExpressionAssetIds]
      : [...this.transientFaceExpressionAssetIds];
    this.transientActive = false;
    this.transientFaceExpressionAssetIds = [];
    this.pendingSpeechExpressionAssetIds = [];
    this.getRuntime()?.restoreTransientPose?.();
    if (wasInteraction) {
      this.resumeAfterInteraction();
      return;
    }
    if (this.speaking) {
      if (!this.speechExpressionActive) {
        try {
          this.startSpeechExpressions(faceExpressionAssetIds);
        } catch (error) {
          this.stopSpeechExpressions();
          this.warn("启动说话脸部 Expression 持续层失败", error);
        }
      }
      this.stopBehavior();
    }
    else {
      this.stopSpeechExpressions();
      this.startState();
    }
  }

  private hasInteraction(): boolean {
    return this.transientActive && this.transientSource !== "emotion";
  }

  private clearEmotionContext(): void {
    this.deferredEmotion = null;
    this.emotionFaceExpressionAssetIds = [];
  }

  private resumeAfterInteraction(): void {
    const deferred = this.deferredEmotion;
    this.deferredEmotion = null;
    if (deferred) {
      this.playTransient(deferred.candidate, deferred.label);
      return;
    }
    if (!this.speaking) {
      this.startState();
      return;
    }
    this.stopBehavior();
    try {
      this.startSpeechExpressions(this.emotionFaceExpressionAssetIds);
    } catch (error) {
      this.stopSpeechExpressions();
      this.warn("交互动作结束后恢复说话表情失败", error);
    }
  }

  private startState(): boolean {
    const runtime = this.getRuntime();
    if (!runtime) return false;
    if (!this.stateCandidate) {
      this.stopBehavior();
      return true;
    }
    const token = ++this.stateRunToken;
    const rotates = this.selectNextState !== null;
    try {
      // Core reset omits CubismPose selectors. The next state must start from
      // the state baseline, even when it only changes a facial expression or
      // was deferred until a transient finished.
      runtime.restoreStatePose?.();
      runtime.beginStatePose?.();
      runtime.startBehavior(structuredClone(this.stateCandidate), {
        // When the runtime owns the candidate boundary, let the Motion finish
        // naturally instead of letting Cubism wrap it through its own V2
        // endpoint-correction loop. A single-candidate state still loops.
        loopMotions: !rotates,
        finishAfterOneCycle: rotates,
        expressionOnlyDurationMs: 2_000,
        onFinished: rotates ? () => this.finishStateCycle(token) : undefined,
      });
      return true;
    } catch (error) {
      this.stopBehavior();
      this.warn("启动 Live2D 状态候选失败，已恢复模型默认姿态", error);
      return false;
    }
  }

  private stopBehavior(): void {
    const runtime = this.getRuntime();
    // Unwind inner pose first: a face-only speech overlay can replace a body
    // transient before it finishes, so its suspended state must not reappear
    // after the state baseline has already been restored.
    runtime?.restoreTransientPose?.();
    runtime?.restoreStatePose?.();
    runtime?.stopBehavior();
  }

  private finishStateCycle(token: number): void {
    if (
      token !== this.stateRunToken ||
      this.speaking ||
      this.transientActive ||
      !this.selectNextState
    ) return;
    const selection = this.selectNextState();
    this.stateCandidate = selection.candidate
      ? structuredClone(selection.candidate)
      : null;
    if (!selection.rotate) this.selectNextState = null;
    this.startState();
  }

  private faceExpressionAssetIds(
    runtime: Live2DBehaviorRuntime,
    candidate: BehaviorCandidate,
  ): string[] {
    const classifier = runtime.isFaceOnlyExpressionAsset;
    if (!classifier || !runtime.startPersistentExpressions) return [];
    return candidate.expressionAssetIds.filter(
      (id) => runtime.isFaceOnlyExpressionAsset!(id),
    );
  }

  private startSpeechExpressions(assetIds: readonly string[]): void {
    const runtime = this.getRuntime();
    if (!runtime?.startPersistentExpressions || assetIds.length === 0) return;
    runtime.startPersistentExpressions(assetIds);
    this.speechExpressionActive = true;
  }

  private stopSpeechExpressions(): void {
    if (!this.speechExpressionActive) return;
    this.getRuntime()?.stopPersistentExpressions?.();
    this.speechExpressionActive = false;
  }
}
