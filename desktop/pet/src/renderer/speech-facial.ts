import { buildBindings, conflictParts, overlayBindings, planDecisions, toNativeTargets, type DecisionPlan, type ParameterBinding } from "../facial-demo/parameter-bindings";
import { facialConflictPlan, resolveFacialConflicts, type FacialConflictPlan } from "../shared/facial-conflicts";
import type { FacialOverlayBindings } from "../shared/facial-bindings";
import type { SpeechFacialApi, SpeechFacialProfile } from "../shared/speech-facial";
import type { PetRenderer } from "./pet-renderer";

/** Model bindings shared with the test panel; no TTS playback or network await here. */
export class SpeechFacialRuntime {
  private epoch = 0;
  private profile: SpeechFacialProfile | null = null;
  private bindings: ParameterBinding[] = [];
  private plan: DecisionPlan | null = null;
  private overlay: FacialOverlayBindings | null = null;
  private conflicts: FacialConflictPlan = { exclusive: [], holds: {} };
  private overlayActive = false;
  private selectedModel: string | null = null;

  constructor(private readonly renderer: PetRenderer, private readonly api: SpeechFacialApi) {}

  async configure(modelId: string): Promise<void> {
    const epoch = ++this.epoch;
    this.selectedModel = modelId;
    this.profile = null;
    this.release();
    this.renderer.setSpeechFacialEnabled?.(false);
    if (!this.renderer.nativeParameters || !this.renderer.beginFacialOverlay) return;
    const info = await this.api.prepare(modelId);
    if (epoch !== this.epoch) return;
    const bindings = buildBindings(this.renderer.nativeParameters(), info.blushParameter,
      info.physicsOutputParameterIds, info.manualBindings, info.parameterSettings,
      info.descriptions, info.customParameters, info.criteria, info.vtsSettings);
    this.bindings = bindings.active;
    this.overlay = overlayBindings(bindings.all);
    const plan = planDecisions(bindings.active);
    this.plan = plan;
    this.conflicts = facialConflictPlan(conflictParts(bindings.active), info.parameterEffects);
    const profile: SpeechFacialProfile = {
      revision: crypto.randomUUID(), modelId, modelName: info.modelName,
      parameters: plan.parameters,
    };
    this.profile = profile;
    const result = await this.api.publish(profile);
    if (epoch !== this.epoch) return;
    this.renderer.setSpeechFacialEnabled?.(result.enabled && bindings.active.length > 0);
  }

  async refresh(modelId = this.selectedModel): Promise<void> {
    if (modelId && modelId === this.selectedModel) await this.configure(modelId);
  }

  clear(): void {
    ++this.epoch;
    this.profile = null;
    this.selectedModel = null;
    this.release();
    this.renderer.setSpeechFacialEnabled?.(false);
    void this.api.publish(null).catch(error => console.warn("清理桌宠表情配置失败", error));
  }

  apply(payload: Record<string, unknown>): void {
    if (!this.profile || payload.profileId !== this.profile.revision) return;
    const targets = payload.targets;
    if (targets === null) { this.release(); return; }
    if (!targets || typeof targets !== "object" || Array.isArray(targets)) return;
    const answers = targets as Record<string, number>;
    if (Object.keys(answers).length !== this.plan?.parameters.length) return;
    const expanded = this.plan.expand(answers);
    const values = expanded && resolveFacialConflicts(expanded, conflictParts(this.bindings), this.conflicts);
    if (!values || this.bindings.some(binding =>
      typeof values[binding.id] !== "number" || !Number.isFinite(values[binding.id])
      || values[binding.id]! < binding.min || values[binding.id]! > binding.max)) return;
    this.renderer.setSpeechFacialEnabled?.(true);
    if (!this.overlayActive) {
      this.renderer.beginFacialOverlay?.(this.overlay!);
      this.overlayActive = true;
    }
    // Each sentence retargets the SAME controller, preserving value/velocity/
    // acceleration and the author's running motion, blink, breath and physics.
    this.renderer.setFacialDemoTargets?.(toNativeTargets(values, this.bindings));
  }

  release(): void {
    if (this.overlayActive) this.renderer.endFacialOverlay?.();
    this.overlayActive = false;
  }
}
