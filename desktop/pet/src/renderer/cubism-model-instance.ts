import { CubismModelSettingJson } from "@cubism/cubismmodelsettingjson";
import { CubismDefaultParameterId } from "@cubism/cubismdefaultparameterid";
import {
  BreathParameterData,
  CubismBreath,
} from "@cubism/effect/cubismbreath";
import { CubismEyeBlink } from "@cubism/effect/cubismeyeblink";
import { CubismPose } from "@cubism/effect/cubismpose";
import { CubismPhysics } from "@cubism/physics/cubismphysics";
import {
  CubismLook,
  LookParameterData,
} from "@cubism/effect/cubismlook";
import type { CubismIdHandle } from "@cubism/id/cubismid";
import { CubismFramework } from "@cubism/live2dcubismframework";
import { CubismMatrix44 } from "@cubism/math/cubismmatrix44";
import { CubismUserModel } from "@cubism/model/cubismusermodel";
import type { ACubismMotion } from "@cubism/motion/acubismmotion";
import { CubismBreathUpdater } from "@cubism/motion/cubismbreathupdater";
import { CubismExpressionUpdater } from "@cubism/motion/cubismexpressionupdater";
import { CubismEyeBlinkUpdater } from "@cubism/motion/cubismeyeblinkupdater";
import { LeadingLookUpdater } from "./leading-look";
import type { CubismMotion } from "@cubism/motion/cubismmotion";
import { CubismPhysicsUpdater } from "@cubism/motion/cubismphysicsupdater";
import { CubismPoseUpdater } from "@cubism/motion/cubismposeupdater";
import { CubismUpdateScheduler } from "@cubism/motion/cubismupdatescheduler";
import { CubismWebGLOffscreenManager } from "@cubism/rendering/cubismoffscreenmanager";
import type { CubismRenderer_WebGL } from "@cubism/rendering/cubismrenderer_webgl";
import { CubismShaderManager_WebGL } from "@cubism/rendering/cubismshader_webgl";

import type {
  BehaviorCandidate,
  ModelPackage,
  ModelRenderBundle,
  MotionAsset,
  NativeModelParameter,
  NativeModelLayer,
  PetManifest,
  TextureAtlasDelta,
  TextureAtlasUpdate,
} from "../shared/contracts";
import { validateTextureAtlasDelta } from "../shared/texture-patch-contract";
import type { LipShape, PetModelBounds } from "./pet-renderer";
import type {
  BehaviorRuntimeStartOptions,
  Live2DBehaviorRuntime,
} from "./live2d-behavior-playback";
import { configuredVowelParameters, type Vowel } from "./viseme-parameters";
import { CubismTextureStore, type TextureRecipe } from "./cubism-texture-store";
import { finishCubismOffscreenFrame } from "./cubism-frame-lifecycle";
import { immutableTextureBaseUrl } from "./texture-base-url";
import { allSettledOrThrow } from "./all-or-release";
import {
  captureCubismPoseSelection,
  CubismBehaviorGroup,
  type CubismPoseSelectionSnapshot,
  restoreCubismDefaultParameters,
  restoreCubismPoseSelection,
} from "./cubism-behavior-group";
import { isFaceOnlyExpression } from "./face-only-behavior";
import { CubismBehaviorExpressionUpdater } from "./cubism-behavior-expression-updater";
import { CubismNaturalIdleUpdater } from "./cubism-natural-idle-updater";
import { CubismExpressiveIdle } from "./cubism-expressive-idle";
import { expressionParameterIds, mergeParameterIds, motionParameterIds, UNKNOWN_IDLE_OWNERSHIP } from "./idle-parameter-ownership";
import { CubismDefaultAppearance } from "./cubism-default-appearance";
import { CubismLayerVisibility, type LayerHighlightMode } from "./cubism-layer-visibility";
import { meshContainsPoint } from "./model-layer-hit-test";

/** Half the height, in CSS pixels, of the band beside the body where the chat entry goes. */
const MIDDLE_BAND = 24;
import { cubismMaskBufferCount } from "./cubism-mask-buffers";
import { FacialParameterController, type FacialDemoSnapshot } from "./facial-parameter-controller";
import type { FacialMotionBindings, FacialOverlayBindings } from "../shared/facial-bindings";
import { CubismFacialOverlay, SPEECH_MOTION } from "./cubism-facial-overlay";
import facialDemoParameters from "../shared/facial-demo-parameters.json";
import {
  expressionBlocksNaturalIdle,
  naturalIdleAffectedParameters,
  readNaturalIdlePhysicsParameters,
} from "./natural-idle-conflicts";
import { mouthOpening, mouthParts } from "./lip-opening";

const SHADER_PATH = "./cubism-shaders/";
const VOWELS: readonly Vowel[] = ["a", "e", "i", "o", "u"];
const MIN_BEHAVIOR_TRANSITION_SECONDS = 0.25;
const MAX_BEHAVIOR_TRANSITION_SECONDS = 1.5;
export const DRAWABLE_HIT_BOUNDS_REFRESH_SECONDS = 0.1;
const FACIAL_DEMO_AUTOMATIC_IDS = new Set(facialDemoParameters.filter(parameter => parameter.owner !== "jev").map(parameter => parameter.id));

/** Output-only rig channels must never be overwritten by a facial target. */
export function facialDemoPhysicsOutputOnlyIds(buffer: ArrayBuffer): ReadonlySet<string> {
  const settings = (JSON.parse(new TextDecoder().decode(buffer)) as {
    PhysicsSettings?: Array<{
      Input?: Array<{ Source?: { Target?: string; Id?: string } }>;
      Output?: Array<{ Destination?: { Target?: string; Id?: string } }>;
    }>;
  }).PhysicsSettings ?? [];
  const inputs = new Set<string>();
  const outputs = new Set<string>();
  for (const setting of settings) {
    for (const { Source: source } of setting.Input ?? []) {
      if (source?.Target === "Parameter" && source.Id) inputs.add(source.Id);
    }
    for (const { Destination: destination } of setting.Output ?? []) {
      if (destination?.Target === "Parameter" && destination.Id) outputs.add(destination.Id);
    }
  }
  return new Set([...outputs].filter(id => !inputs.has(id)));
}

/**
 * Keeps pointer hit geometry current without rescanning every drawable on every
 * rendered frame. Interactive reads still refresh once for the newest Cubism
 * model revision, so several pointer events between two frames share one scan.
 */
export class DrawableHitBoundsRefreshCadence {
  private elapsedSeconds = DRAWABLE_HIT_BOUNDS_REFRESH_SECONDS;
  private modelRevision = 0;
  private boundsRevision = -1;

  afterModelUpdate(deltaSeconds: number): boolean {
    this.modelRevision += 1;
    this.elapsedSeconds += deltaSeconds;
    return this.elapsedSeconds + Number.EPSILON >= DRAWABLE_HIT_BOUNDS_REFRESH_SECONDS;
  }

  interactionNeedsRefresh(): boolean {
    return this.boundsRevision !== this.modelRevision;
  }

  markRefreshed(): void {
    this.boundsRevision = this.modelRevision;
    this.elapsedSeconds = 0;
  }
}

interface DisplayMetrics {
  cssWidth: number;
  cssHeight: number;
  pixelWidth: number;
  pixelHeight: number;
}

interface ActiveBehaviorGroup {
  layers: CubismBehaviorGroup;
  blocksNaturalIdle: boolean;
  motionParameters: ReadonlySet<string>;
  expressionParameters: ReadonlySet<string>;
  loopMotions: boolean;
  finishAfterOneCycle: boolean;
  elapsedSeconds: number;
  cycleDurationSeconds: number | null;
  onFinished: (() => void) | null;
  timeout: ReturnType<typeof setTimeout> | null;
  finishQueued: boolean;
}

interface ParameterTransition {
  from: Float32Array;
  elapsedSeconds: number;
  durationSeconds: number;
}

export interface CubismModelInstanceOptions {
  /** Only the isolated analysis renderer opts in; ordinary playback is unchanged. */
  sampling?: boolean;
  maxTextureDimension?: number;
  onTextureCreated?: (resource: { width: number; height: number }) => void;
  onTextureDeleted?: () => void;
}

/** A fully loaded official Cubism model and all resources tied to its GL context. */
export class CubismModelInstance extends CubismUserModel implements Live2DBehaviorRuntime {
  private readonly setting: CubismModelSettingJson;
  private textureStore!: CubismTextureStore;
  private readonly scheduler = new CubismUpdateScheduler();
  private readonly motions = new Map<string, CubismMotion>();
  private readonly expressions = new Map<string, ACubismMotion>();
  private readonly faceOnlyExpressionAssetIds = new Set<string>();
  private readonly projection = new CubismMatrix44();
  private readonly drawMatrix = new CubismMatrix44();
  private layerVisibility: CubismLayerVisibility | null = null;
  private readonly viewport = [0, 0, 1, 1];
  private drawableHitBounds = new Float32Array(0);
  private readonly parameterIds = new Map<string, CubismIdHandle>();
  private readonly hitBoundsRefresh = new DrawableHitBoundsRefreshCadence();
  private readonly eyeBlinkIds: CubismIdHandle[] = [];
  private readonly lipSyncIds: CubismIdHandle[] = [];
  private readonly directVowelIds: Partial<Record<Vowel, CubismIdHandle>> = {};
  private activeBehavior: ActiveBehaviorGroup | null = null;
  private naturalIdleUpdater: CubismNaturalIdleUpdater | null = null;
  private expressiveIdle: CubismExpressiveIdle | null = null;
  private readonly motionOwnership = new Map<string, ReadonlySet<string>>();
  private persistentExpressionParameters: ReadonlySet<string> = new Set();
  private standaloneExpressionParameters: ReadonlySet<string> = new Set();
  private naturalIdleAffectedParameters = naturalIdleAffectedParameters([]);
  private defaultAppearance: CubismDefaultAppearance | null = null;
  private persistentExpressionBehavior: CubismBehaviorGroup | null = null;
  private readonly behaviorExpressionUpdater = new CubismBehaviorExpressionUpdater(
    () => this.activeBehavior?.layers ?? null,
    () => this.persistentExpressionBehavior,
  );
  private parameterTransition: ParameterTransition | null = null;
  private statePoseSelection: CubismPoseSelectionSnapshot | null = null;
  private transientPoseSelection: CubismPoseSelectionSnapshot | null = null;
  private modelBounds: PetModelBounds | null = null;
  private viewOffsetX = 0;
  private viewOffsetY = 0;
  private viewZoom = 1;
  private zoomPanX = 0;
  private zoomPanY = 0;
  private displayMetrics: DisplayMetrics | null = null;
  private lipTarget: LipShape = { volume: 0, a: 0, e: 0, i: 0, o: 0, u: 0 };
  private currentOpen = 0;
  private currentForm = 0;
  private currentVowels: Record<Vowel, number> = { a: 0, e: 0, i: 0, o: 0, u: 0 };
  private mouthOpenId: CubismIdHandle | null = null;
  private mouthFormId: CubismIdHandle | null = null;
  /** A VBridger rig's jaw, pucker/widen and funnel, driven by the vowel heard. */
  private mouthPartIds: { jaw: CubismIdHandle | null; puckerWiden: CubismIdHandle | null; funnel: CubismIdHandle | null } = { jaw: null, puckerWiden: null, funnel: null };
  private currentParts = { jaw: 0, puckerWiden: 0, funnel: 0 };
  private maxMouthOpen = 0.7;
  private lipSyncMode: PetManifest["lipsync"]["mode"] = "unconfigured";
  private look: LeadingLookUpdater | null = null;
  /** TA's own speaking pitch, learnt from one sentence and kept for the next. */
  private readonly spokenPitch: { key: number | null } = { key: null };
  private lipSyncValues = new Map<string, { index: number; value: number; neutral: number }>();
  private motionUpdated = false;
  private speaking = false;
  private talkEnergy = 0;
  private talkPhase = 0;
  private released = false;
  private facialDemoPhysicsOutputs: ReadonlySet<string> = new Set();
  private facialOverlay: { layer: CubismFacialOverlay; animateBase: boolean; staticBase: number[] } | null = null;
  private facialDemo: {
    controller: FacialParameterController;
    indices: ReadonlyMap<string, number>;
    eyeIndices: readonly number[];
    eyeClosedValues?: readonly number[];
    blink: CubismEyeBlink;
    breath: { index: number; base: number; maximum: number; elapsedSeconds: number } | null;
  } | null = null;
  private sampleState: {
    mode: "asset" | "runtime";
    physics?: ArrayBuffer;
    pose?: ArrayBuffer;
    partOpacities: number[];
    poseSelection?: CubismPoseSelectionSnapshot;
    pendingFinish?: ActiveBehaviorGroup;
  } | null = null;

  static async load(
    gl: WebGL2RenderingContext,
    bundle: ModelRenderBundle,
    metrics: DisplayMetrics,
    options: CubismModelInstanceOptions = {},
  ): Promise<CubismModelInstance> {
    const value = bundle.model;
    const modelResponse = await fetch(value.url);
    if (!modelResponse.ok) {
      throw new Error(`读取 Live2D model3.json 失败 (${modelResponse.status})`);
    }
    const modelJson = await modelResponse.arrayBuffer();
    const setting = new CubismModelSettingJson(modelJson, modelJson.byteLength);
    const instance = new CubismModelInstance(gl, value, bundle.appearance, setting, options);
    try {
      await instance.initialize(metrics);
      return instance;
    } catch (error) {
      instance.release();
      throw error;
    }
  }

  private constructor(
    private readonly gl: WebGL2RenderingContext,
    readonly packageValue: ModelPackage,
    private readonly initialAppearance: TextureAtlasDelta,
    setting: CubismModelSettingJson,
    private readonly options: CubismModelInstanceOptions,
  ) {
    super();
    this.setting = setting;
  }

  private async initialize(metrics: DisplayMetrics): Promise<void> {
    const modelBuffer = await this.fetchBuffer(this.setting.getModelFileName());
    this.loadModel(modelBuffer, true);
    if (!this._model) throw new Error("Cubism Core 无法创建该 moc3 模型");
    this.layerVisibility = new CubismLayerVisibility(this._model);
    if (this.options.sampling) {
      this.sampleState = {
        mode: "asset",
        partOpacities: Array.from({ length: this._model.getPartCount() }, (_, i) => this._model.getPartOpacityByIndex(i)),
      };
    }
    this.drawableHitBounds = new Float32Array(this._model.getDrawableCount() * 4);

    const layout = new Map<string, number>();
    this.setting.getLayoutMap(layout);
    this._modelMatrix.setupFromLayout(layout);

    this.collectEffectIds();
    this.defaultAppearance = new CubismDefaultAppearance(this._model);
    this.scheduler.addUpdatableList(this.defaultAppearance);
    this.scheduler.addUpdatableList(this.behaviorExpressionUpdater);
    await allSettledOrThrow([
      this.loadExpressions(this.options.sampling
        ? new Set(this.packageValue.manifest.defaultAppearance?.expressionAssetIds ?? [])
        : undefined),
      this.loadOptionalResources(),
      this.options.sampling ? Promise.resolve() : this.loadMotions(),
    ]);
    this.configureDefaultAppearance(this.packageValue.manifest.defaultAppearance);
    if (!this.options.sampling) this.configureEffects();
    else this.scheduler.sortUpdatableList();
    this.configureLipSync(this.packageValue.manifest.lipsync);

    this.createRenderer(metrics.pixelWidth, metrics.pixelHeight, this.maskBufferCount());
    const renderer = this.getRenderer();
    renderer.startUp(this.gl);
    renderer.loadShaders(SHADER_PATH);
    renderer.setIsPremultipliedAlpha(true);
    this.textureStore = this.createTextureStore(renderer);
    await this.textureStore.loadAll(this.textureUrls());
    await this.applyTextureSnapshot(this.initialAppearance, () => true, () => undefined);
    await waitForShaders(this.gl);
    this.setInitialized(true);
    this.setUpdating(false);
    this.resize(metrics);
    this.update(1 / 60);
    if (this.sampleState) this.sampleState.poseSelection = captureCubismPoseSelection(this._model);
  }

  /** Reuses the expensive Moc/GL/texture resources, but none of the previous pose state. */
  async prepareSample(candidate: BehaviorCandidate, mode: "asset" | "runtime"): Promise<void> {
    const state = this.sampleState;
    if (!state || this.released) throw new Error("该模型不是采样实例");
    this.stopPersistentExpressions();
    this.stopBehavior();
    this._motionManager.stopAllMotions();
    this._expressionManager.stopAllMotions();
    for (const motion of this.motions.values()) motion.release();
    for (const expression of this.expressions.values()) expression.release();
    this.motions.clear();
    this.expressions.clear();
    this.faceOnlyExpressionAssetIds.clear();
    this.scheduler.release();
    if (this.defaultAppearance) this.scheduler.addUpdatableList(this.defaultAppearance);
    this.scheduler.addUpdatableList(this.behaviorExpressionUpdater);
    state.mode = mode;
    state.pendingFinish = undefined;
    this.statePoseSelection = null;
    this.transientPoseSelection = null;
    restoreCubismDefaultParameters(this._model);
    if (state.poseSelection) restoreCubismPoseSelection(this._model, state.poseSelection);
    state.partOpacities.forEach((value, i) => this._model.setPartOpacityByIndex(i, value));
    this.setOpacity(1);
    this.currentOpen = this.currentForm = this.talkEnergy = this.talkPhase = 0;
    this.currentParts = { jaw: 0, puckerWiden: 0, funnel: 0 };
    this.currentVowels = { a: 0, e: 0, i: 0, o: 0, u: 0 };
    this.setLipShape({ volume: 0, a: 0, e: 0, i: 0, o: 0, u: 0 });
    this.setSpeaking(false);
    if (state.physics) {
      CubismPhysics.delete(this._physics);
      this.loadPhysics(state.physics, state.physics.byteLength);
      this._physics.stabilization(this._model);
      this.scheduler.addUpdatableList(new CubismPhysicsUpdater(this._physics));
    }
    if (state.pose) {
      CubismPose.delete(this._pose);
      this.loadPose(state.pose, state.pose.byteLength);
      this._pose.reset(this._model);
      this.scheduler.addUpdatableList(new CubismPoseUpdater(this._pose));
    }
    this.scheduler.sortUpdatableList();
    this._model.saveParameters();
    await this.loadMotions(new Set(candidate.motionAssetIds));
    await this.loadExpressions(new Set([
      ...candidate.expressionAssetIds,
      ...(this.packageValue.manifest.defaultAppearance?.expressionAssetIds ?? []),
    ]));
    this.configureDefaultAppearance(this.packageValue.manifest.defaultAppearance);
    this.defaultAppearance?.apply(this._model);
    if (this._physics && this.defaultAppearance?.parameters().size) {
      this._physics.stabilization(this._model);
    }
    this.defaultAppearance?.apply(this._model);
    // Drawing the baseline must not reuse the last candidate's drawable vertices.
    this._model.update();
  }

  /** Called by the manual clock after any screenshot at the current tick. */
  flushSampleCompletion(): void {
    const state = this.sampleState;
    if (!state) return;
    const pending = state.pendingFinish;
    state.pendingFinish = undefined;
    if (pending && state.mode === "runtime") this.finishBehavior(pending);
  }

  gpuDiagnostics(): {
    atlasCount: number;
    atlasDimensions: Array<{ width: number; height: number }>;
    offscreenContainerCount: number;
  } {
    return {
      atlasCount: this.textureStore?.count() ?? 0,
      atlasDimensions: this.textureStore?.dimensions() ?? [],
      offscreenContainerCount: CubismWebGLOffscreenManager.getInstance().getContainerSize(this.gl),
    };
  }

  setExpression(name: string | null): void {
    this.stopPersistentExpressions();
    this.stopBehavior();
    this._expressionManager.stopAllMotions();
    this.standaloneExpressionParameters = new Set();
    if (!name) { this.refreshIdleOwnership(); return; }
    const expression = this.expressions.get(name);
    if (expression) {
      this._expressionManager.startMotion(expression, false);
      this.standaloneExpressionParameters = expressionParameterIds(expression);
    }
    this.refreshIdleOwnership();
  }

  startBehavior(
    candidate: BehaviorCandidate,
    options: BehaviorRuntimeStartOptions,
  ): void {
    if (this.released) throw new Error("Live2D 模型已经释放");
    const motionLayers = candidate.motionAssetIds.map((id) => {
      const asset = this.packageValue.manifest.motionAssets[id];
      const motion = asset ? this.motions.get(motionKey(asset.group, asset.index)) : null;
      if (!asset || !motion) throw new Error(`行为候选引用的 Motion 不存在：${id}`);
      return motion;
    });
    const expressionLayers = candidate.expressionAssetIds.map((id) => {
      const asset = this.packageValue.manifest.expressionAssets[id];
      const expression = asset ? this.expressions.get(asset.expression) : null;
      if (!asset || !expression) throw new Error(`行为候选引用的 Expression 不存在：${id}`);
      return expression;
    });
    if (!motionLayers.length && !expressionLayers.length) {
      throw new Error("行为候选没有可播放素材");
    }

    const previous = this.activeBehavior;
    // Capture the current Motion pose including an unfinished handoff, before
    // breath/gaze/physics so the new frame does not apply those effects twice.
    const previousParameters = previous && this._model
      ? this.captureTransitionSourceValues()
      : null;
    const previousFadeOut = previous?.layers.fadeOutDurationSeconds() ?? 0;
    // Resolve every layer before replacing the currently visible group. Once
    // committed, each layer gets its own manager so Cubism never fades one
    // selected asset out merely because another asset starts on the same frame.
    // Group replacement separately blends the visible parameters using the
    // authored fade-out/fade-in duration, preserving Cubism's smooth handoff.
    this.stopBehavior();
    this._motionManager.stopAllMotions();
    this._expressionManager.stopAllMotions();
    this.standaloneExpressionParameters = new Set();
    let group: ActiveBehaviorGroup | null = null;
    try {
      const layers = new CubismBehaviorGroup(motionLayers, expressionLayers, options.loopMotions);
      if (previousParameters) {
        this.parameterTransition = {
          from: previousParameters,
          elapsedSeconds: 0,
          durationSeconds: behaviorTransitionDuration(
            previousFadeOut,
            layers.fadeInDurationSeconds(),
          ),
        };
      }
      const createdGroup: ActiveBehaviorGroup = {
        layers,
        blocksNaturalIdle: motionLayers.length > 0
          || expressionLayers.some((expression) => expressionBlocksNaturalIdle(expression, this.naturalIdleAffectedParameters)),
        motionParameters: mergeParameterIds(candidate.motionAssetIds.map(id => this.motionOwnership.get(id) ?? new Set([UNKNOWN_IDLE_OWNERSHIP]))),
        expressionParameters: mergeParameterIds(expressionLayers.map(expressionParameterIds)),
        loopMotions: options.loopMotions,
        finishAfterOneCycle: options.finishAfterOneCycle ?? false,
        elapsedSeconds: 0,
        cycleDurationSeconds: options.finishAfterOneCycle
          ? layers.motionCycleDurationSeconds() ?? (options.expressionOnlyDurationMs ?? 2_000) / 1_000
          : null,
        onFinished: options.onFinished ?? null,
        timeout: null,
        finishQueued: false,
      };
      group = createdGroup;
      this.activeBehavior = createdGroup;
      this.refreshIdleOwnership();
      if (!options.loopMotions && !this.sampleState) {
        const timeout = motionLayers.length
          ? motionGroupTimeoutMs(motionLayers)
          : options.expressionOnlyDurationMs ?? 2_000;
        createdGroup.timeout = setTimeout(() => this.queueBehaviorFinish(createdGroup), timeout);
      }
    } catch (error) {
      this.parameterTransition = null;
      if (group && this.activeBehavior !== group) releaseBehaviorGroup(group);
      this.stopBehavior();
      throw error;
    }
  }

  stopBehavior(): void {
    this.parameterTransition = null;
    const group = this.activeBehavior;
    this.activeBehavior = null;
    if (group) {
      releaseBehaviorGroup(group);
      if (this._model) restoreCubismDefaultParameters(this._model);
    }
    this.refreshIdleOwnership();
  }

  beginStatePose(): void {
    if (!this._model || this.statePoseSelection) return;
    this.statePoseSelection = captureCubismPoseSelection(this._model);
  }

  restoreStatePose(): void {
    const snapshot = this.statePoseSelection;
    this.statePoseSelection = null;
    if (!this._model || !snapshot) return;
    restoreCubismPoseSelection(this._model, snapshot);
  }

  beginTransientPose(): void {
    if (!this._model || this.transientPoseSelection) return;
    this.transientPoseSelection = captureCubismPoseSelection(this._model);
  }

  restoreTransientPose(): void {
    const snapshot = this.transientPoseSelection;
    this.transientPoseSelection = null;
    if (!this._model || !snapshot) return;
    restoreCubismPoseSelection(this._model, snapshot);
  }

  isFaceOnlyExpressionAsset(assetId: string): boolean {
    return this.faceOnlyExpressionAssetIds.has(assetId);
  }

  startPersistentExpressions(assetIds: readonly string[]): void {
    if (this.released) throw new Error("Live2D 模型已经释放");
    const expressions = assetIds.map((id) => {
      if (!this.faceOnlyExpressionAssetIds.has(id)) {
        throw new Error(`非脸部 Expression 不允许进入说话持续层：${id}`);
      }
      const asset = this.packageValue.manifest.expressionAssets[id];
      const expression = asset ? this.expressions.get(asset.expression) : null;
      if (!asset || !expression) throw new Error(`持续 Expression 不存在：${id}`);
      return expression;
    });
    this.stopPersistentExpressions();
    if (expressions.length > 0) {
      const group = new CubismBehaviorGroup([], expressions, false);
      try {
        group.primeExpressions(this._model);
        this.persistentExpressionBehavior = group;
        this.persistentExpressionParameters = mergeParameterIds(expressions.map(expressionParameterIds));
        this.refreshIdleOwnership();
      } catch (error) {
        group.release();
        throw error;
      }
    }
  }

  stopPersistentExpressions(): void {
    this.persistentExpressionBehavior?.release();
    this.persistentExpressionBehavior = null;
    this.persistentExpressionParameters = new Set();
    this.refreshIdleOwnership();
  }

  setLipShape(shape: LipShape): void {
    this.lipTarget = shape;
  }

  setSpeaking(speaking: boolean): void {
    this.speaking = speaking;
  }

  configureNaturalIdle(value: PetManifest["naturalIdle"]): void {
    this.naturalIdleUpdater?.configure(value);
    this.expressiveIdle?.configure(value);
  }

  configureDefaultAppearance(value: PetManifest["defaultAppearance"]): void {
    const expressions = (value?.expressionAssetIds ?? []).map((id) => {
      const asset = this.packageValue.manifest.expressionAssets[id];
      const expression = asset ? this.expressions.get(asset.expression) : null;
      if (!expression) throw new Error(`默认外观素材不存在：${id}`);
      return expression;
    });
    this.defaultAppearance?.configure(expressions, value?.parameterValues);
    this.layerVisibility?.configure(value?.hiddenDrawableIds);
    this.refreshIdleOwnership();
  }

  nativeLayers(): NativeModelLayer[] {
    const raw = this._model.getModel();
    const names = new Map(this.packageValue.capabilities.parts?.map(part => [part.id, part.name]));
    const ids: readonly string[] = raw.drawables.ids;
    return ids.map((id, index) => {
      const parent = this._model.getDrawableParentPartIndex(index);
      const partId = parent >= 0 ? raw.parts.ids[parent] ?? null : null;
      const partName = partId ? names.get(partId) ?? partId : "未分组";
      return { id, name: partName === partId || partId === null ? id : `${partName} · ${id}`,
        partId, partName, visible: this._model.getDrawableDynamicFlagIsVisible(index) && (this.layerVisibility?.authoredOpacity(index) ?? 0) > 0.02 };
    });
  }

  highlightLayers(ids: readonly string[], mode?: LayerHighlightMode): void { this.layerVisibility?.highlight(ids, mode); }

  /** Drawables the character shows right now, with the user's hidden toggles applied. */
  shownLayerIds(): Set<string> {
    const shown = new Set<string>();
    for (let index = 0; index < this._model.getDrawableCount(); index += 1) {
      // A highlight shows hidden pieces for a moment; that isn't what the character wears.
      const visible = this.layerVisibility
        ? this.layerVisibility.isShown(index)
        : this._model.getDrawableDynamicFlagIsVisible(index) && this._model.getDrawableOpacity(index) > 0.02;
      if (visible) shown.add(this._model.getDrawableId(index).getString());
    }
    return shown;
  }

  /** Where everything currently drawn is on screen, in CSS pixels. */
  visibleScreenBounds(metrics: DisplayMetrics): { x: number; y: number; width: number; height: number } | null {
    return this.layerScreenBounds([...this.shownLayerIds()], metrics);
  }

  /** Where these drawables currently are on screen, in CSS pixels. */
  layerScreenBounds(ids: readonly string[], metrics: DisplayMetrics): { x: number; y: number; width: number; height: number } | null {
    const wanted = new Set(ids);
    let left = Infinity, right = -Infinity, bottom = Infinity, top = -Infinity;
    for (let index = 0; index < this._model.getDrawableCount(); index += 1) {
      if (!wanted.has(this._model.getDrawableId(index).getString())) continue;
      const vertices = this._model.getDrawableVertices(index);
      for (let offset = 0; offset + 1 < vertices.length; offset += 2) {
        const x = vertices[offset]!, y = vertices[offset + 1]!;
        if (x < left) left = x;
        if (x > right) right = x;
        if (y < bottom) bottom = y;
        if (y > top) top = y;
      }
    }
    if (!Number.isFinite(left)) return null;
    const leftClip = this.projection.transformX(this._modelMatrix.transformX(left));
    const rightClip = this.projection.transformX(this._modelMatrix.transformX(right));
    const bottomClip = this.projection.transformY(this._modelMatrix.transformY(bottom));
    const topClip = this.projection.transformY(this._modelMatrix.transformY(top));
    const x1 = ((Math.min(leftClip, rightClip) + 1) * 0.5) * metrics.cssWidth;
    const x2 = ((Math.max(leftClip, rightClip) + 1) * 0.5) * metrics.cssWidth;
    const y1 = ((1 - Math.max(topClip, bottomClip)) * 0.5) * metrics.cssHeight;
    const y2 = ((1 - Math.min(topClip, bottomClip)) * 0.5) * metrics.cssHeight;
    return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
  }

  pickLayers(x: number, y: number, metrics: DisplayMetrics): string[] {
    const mx = this.screenToModelX(x, metrics), my = this.screenToModelY(y, metrics);
    const hits: number[] = [];
    for (let i = 0; i < this._model.getDrawableCount(); i++) {
      if (!this._model.getDrawableDynamicFlagIsVisible(i) || this._model.getDrawableOpacity(i) <= 0.02) continue;
      if (meshContainsPoint(this._model.getDrawableVertices(i), this._model.getDrawableVertexIndices(i), mx, my)) hits.push(i);
    }
    const orders = this._model.getRenderOrders();
    return hits.sort((a, b) => orders[b]! - orders[a]!).map(index => this._model.getDrawableId(index).getString());
  }

  nativeParameters(): NativeModelParameter[] {
    const names = new Map(this.packageValue.capabilities.parameters?.map(p => [p.id, p.name]));
    const ids: readonly string[] = this._model.getModel().parameters.ids;
    return ids.map((id, index) => ({
      id,
      name: names.get(id) ?? id,
      minimum: this._model.getParameterMinimumValue(index),
      maximum: this._model.getParameterMaximumValue(index),
      defaultValue: this._model.getParameterDefaultValue(index),
    }));
  }

  /**
   * The authored pose with one parameter moved alone, physics settled: the
   * same every time, so two probes differ only by what that parameter draws.
   */
  probeParameter(id: string, value: number): void {
    if (!this._model || this.released) return;
    const index = this._model.getParameterIndex(this.id(id));
    const apply = () => { if (index >= 0) this._model.setParameterValueByIndex(index, value); };
    restoreCubismDefaultParameters(this._model);
    this._pose?.reset(this._model);
    this.defaultAppearance?.apply(this._model);
    apply();
    this._physics?.stabilization(this._model);
    this.defaultAppearance?.apply(this._model);
    apply();
    this._model.update();
  }

  /** Model-manager inspection only: evaluate an authored pose without live drivers. */
  updateEditingPreview(): void {
    if (!this._model || this.released) return;
    restoreCubismDefaultParameters(this._model);
    this._pose?.reset(this._model);
    this.defaultAppearance?.apply(this._model);
    // Some user-facing parameters feed physics outputs rather than meshes.
    // Evaluate their settled pose, then reapply explicitly fixed output values.
    this._physics?.stabilization(this._model);
    this.defaultAppearance?.apply(this._model);
    this._model.update();
    this.updateDrawableHitBounds();
  }

  /** Explicit demo entry; normal pet playback never creates this controller. */
  beginFacialDemo(motion?: FacialMotionBindings): void {
    this.endFacialOverlay(true);
    if (this.facialDemo || !this._model || this.released) return;
    this.stopPersistentExpressions();
    this.stopBehavior();
    this._motionManager.stopAllMotions();
    this._expressionManager.stopAllMotions();
    this.standaloneExpressionParameters = new Set();
    this.statePoseSelection = null;
    this.transientPoseSelection = null;
    this.setSpeaking(false);
    this.setLipShape({ volume: 0, a: 0, e: 0, i: 0, o: 0, u: 0 });
    this.updateEditingPreview();
    this._model.saveParameters();
    const parameters = this.nativeParameters();
    const indices = new Map(parameters.map((parameter, index) => [parameter.id, index]));
    const baseline = Object.fromEntries(parameters.map((parameter, index) => [parameter.id, this._model.getParameterValueByIndex(index)]));
    const blink = CubismEyeBlink.create();
    const requestedEyes = motion ? motion.eyes.map(eye => this.id(eye.id)) : this.eyeBlinkIds;
    const eyes = requestedEyes.filter(id => indices.has(id.getString())
      && !FACIAL_DEMO_AUTOMATIC_IDS.has(id.getString())
      && !this.facialDemoPhysicsOutputs?.has(id.getString()));
    blink.setParameterIds(eyes);
    const breathIndex = indices.get("ParamBreath");
    const breathMaximum = breathIndex === undefined ? 0 : Math.min(1, parameters[breathIndex]!.maximum);
    const breathBase = breathIndex === undefined ? 0 : clamp(baseline.ParamBreath!, Math.max(0, parameters[breathIndex]!.minimum), breathMaximum);
    this.facialDemo = {
      controller: new FacialParameterController(parameters, baseline,
        motion ? (motion.mouth && !this.facialDemoPhysicsOutputs?.has(motion.mouth.id) && !FACIAL_DEMO_AUTOMATIC_IDS.has(motion.mouth.id) ? motion.mouth : null)
          : this.facialDemoPhysicsOutputs?.has("ParamMouthOpenY") ? null : undefined, motion?.transitions),
      indices,
      eyeIndices: eyes.map(id => indices.get(id.getString())!),
      eyeClosedValues: motion ? eyes.map(id => motion.eyes.find(eye => eye.id === id.getString())!.closed) : undefined,
      blink,
      breath: breathIndex !== undefined && breathMaximum > breathBase && !this.facialDemoPhysicsOutputs?.has("ParamBreath")
        ? { index: breathIndex, base: breathBase, maximum: breathMaximum, elapsedSeconds: 0 } : null,
    };
  }

  setFacialDemoTargets(targets: Record<string, number>): void {
    const controller = this.facialOverlay?.layer.controller ?? this.facialDemo?.controller;
    controller?.setTargets(Object.fromEntries(Object.entries(targets).filter(([id]) =>
      !FACIAL_DEMO_AUTOMATIC_IDS.has(id) && !this.facialDemoPhysicsOutputs?.has(id))));
  }

  setFacialDemoSpeechLevel(level: number): void {
    (this.facialOverlay?.layer.controller ?? this.facialDemo?.controller)?.setSpeechLevel(level);
  }

  facialDemoSnapshot(): FacialDemoSnapshot {
    if (this.facialOverlay) return {
      active: true,
      values: Object.fromEntries(this.nativeParameters().map((parameter, index) => [parameter.id, this._model.getParameterValueByIndex(index)])),
      targets: this.facialOverlay.layer.controller.snapshot().targets,
    };
    const demo = this.facialDemo;
    if (!demo) return { active: false, values: {}, targets: {} };
    return {
      active: true,
      values: Object.fromEntries([...demo.indices].map(([id, index]) => [id, this._model.getParameterValueByIndex(index)])),
      targets: demo.controller.snapshot().targets,
    };
  }

  endFacialDemo(): void {
    if (!this.facialDemo) return;
    this.facialDemo = null;
    this.updateEditingPreview();
    this._model.saveParameters();
  }

  /** Speech is additive to the live pose; existing motions, blink and idle clocks keep running. */
  beginFacialOverlay(bindings: FacialOverlayBindings, animateBase: boolean): void {
    if (!this._model || this.released) return;
    if (this.facialOverlay && JSON.stringify(this.facialOverlay.layer.bindings) === JSON.stringify(bindings)) {
      this.facialOverlay.layer.resume(); return;
    }
    this.endFacialOverlay(true);
    const allowed = (id: string) => !FACIAL_DEMO_AUTOMATIC_IDS.has(id) && !this.facialDemoPhysicsOutputs?.has(id);
    const safe = { ...bindings, neutralValues: Object.fromEntries(Object.entries(bindings.neutralValues).filter(([id]) => allowed(id))) };
    const parameters = this.nativeParameters();
    const layer = new CubismFacialOverlay(parameters, safe, { ...SPEECH_MOTION, dynamics: { ...SPEECH_MOTION.dynamics, memory: this.spokenPitch } });
    this.facialOverlay = { layer, animateBase, staticBase: parameters.map((_, index) => this._model.getParameterValueByIndex(index)) };
    this.scheduler.addUpdatableList(layer);
  }

  hasFacialOverlay(): boolean { return !!this.facialOverlay; }

  endFacialOverlay(immediate = false): void {
    const overlay = this.facialOverlay;
    if (!overlay) return;
    if (!immediate) { overlay.layer.release(); return; }
    this.scheduler.removeUpdatableList(overlay.layer);
    this.facialOverlay = null;
  }

  private finishFacialOverlayFrame(withLipSync = false): void {
    const overlay = this.facialOverlay;
    if (!overlay) return;
    overlay.layer.finishFrame(this._model, withLipSync ? this.lipSyncValues : undefined);
    if (overlay.layer.finished) this.endFacialOverlay(true);
  }

  private updateStaticFacialOverlay(delta: number): void {
    const overlay = this.facialOverlay!;
    overlay.staticBase.forEach((value, index) => this._model.setParameterValueByIndex(index, value));
    overlay.layer.onLateUpdate(this._model, delta);
    this._physics?.evaluate(this._model, delta);
    this._pose?.updateParameters(this._model, delta);
    // 定住看 stills idle motion, blinks and sway, not the voice: speech still moves the mouth.
    this.updateLip(delta);
    for (const channel of this.lipSyncValues.values()) {
      channel.value = this._model.getParameterValueByIndex(channel.index);
    }
    this.defaultAppearance?.apply(this._model);
    this.finishFacialOverlayFrame(true);
    this._model.update();
    if (this.hitBoundsRefresh.afterModelUpdate(delta)) this.updateDrawableHitBounds();
  }

  private updateFacialDemo(delta: number): void {
    const demo = this.facialDemo!;
    this._model.loadParameters();
    this.defaultAppearance?.apply(this._model);
    const owned = new Map<number, number>();
    for (const [id, value] of Object.entries(demo.controller.update(delta))) {
      const index = demo.indices.get(id)!;
      this._model.setParameterValueByIndex(index, value);
      owned.set(index, value);
    }
    const breath = demo.breath;
    if (breath && !this.defaultAppearance?.parameters().has("ParamBreath")) {
      breath.elapsedSeconds = (breath.elapsedSeconds + delta) % 3.2;
      // A gentle absolute cycle starts at the editing pose without adding any
      // head/body idle or accumulating on the saved parameter baseline.
      const value = breath.base + (breath.maximum - breath.base) * (1 - Math.cos(breath.elapsedSeconds * 2 * Math.PI / 3.2)) / 2;
      this._model.setParameterValueByIndex(breath.index, value);
      owned.set(breath.index, value);
    }
    // Cubism's authored blink clock supplies a multiplier over the expression's
    // eyelid opening, rather than replacing a half-open/closed eye with 1.
    const eyeOpen = demo.eyeIndices.map(index => this._model.getParameterValueByIndex(index));
    let blinkFactor = 1;
    if (demo.eyeClosedValues) {
      // Sample the SDK blink clock before native ranges can clamp its 0..1 factor.
      demo.blink.updateParameters({ setParameterValueById: (_id: CubismIdHandle, value: number) => { blinkFactor = value; } } as Parameters<CubismEyeBlink["updateParameters"]>[0], delta);
    } else demo.blink.updateParameters(this._model, delta);
    demo.eyeIndices.forEach((index, position) => {
      const closed = demo.eyeClosedValues?.[position] ?? 0;
      const factor = demo.eyeClosedValues ? blinkFactor : this._model.getParameterValueByIndex(index);
      const value = closed + (eyeOpen[position]! - closed) * factor;
      this._model.setParameterValueByIndex(index, value);
      owned.set(index, value);
    });
    // Real models route head, brow, eyelid and mouth inputs through physics.
    // Let the authored rig compute its output IDs; never substitute guessed IDs.
    this._physics?.evaluate(this._model, delta);
    this._pose?.updateParameters(this._model, delta);
    this.defaultAppearance?.apply(this._model);
    for (const [index, value] of owned) this._model.setParameterValueByIndex(index, value);
    this._model.update();
    if (this.hitBoundsRefresh.afterModelUpdate(delta)) this.updateDrawableHitBounds();
  }

  private refreshIdleOwnership(): void {
    this.expressiveIdle?.setOwnership(
      this.activeBehavior?.motionParameters ?? new Set(),
      mergeParameterIds([this.activeBehavior?.expressionParameters ?? new Set(), this.persistentExpressionParameters, this.standaloneExpressionParameters]),
      this.defaultAppearance?.parameters() ?? new Set(),
    );
  }

  configureLipSync(value: PetManifest["lipsync"]): void {
    this.lipSyncMode = value.mode;
    this.mouthOpenId = this.id(value.mouthOpenParameter);
    this.mouthFormId = this.id(value.mouthFormParameter);
    this.mouthPartIds = {
      jaw: this.id(value.jawParameter),
      puckerWiden: this.id(value.puckerWidenParameter),
      funnel: this.id(value.funnelParameter),
    };
    this.maxMouthOpen = Math.max(0, Math.min(1, value.maxOpen));
    const direct = configuredVowelParameters(value.vowelParameters);
    for (const vowel of Object.keys(this.directVowelIds) as Vowel[]) {
      delete this.directVowelIds[vowel];
    }
    for (const vowel of Object.keys(direct) as Vowel[]) {
      const id = this.id(direct[vowel] as string | null | undefined);
      if (id) this.directVowelIds[vowel] = id;
    }
    const lipIds = this.lipSyncMode === "viseme" ? Object.values(this.directVowelIds)
      : this.lipSyncMode === "open-form" ? [this.mouthOpenId, this.mouthFormId, ...Object.values(this.mouthPartIds)] : [];
    // updateLip writes zero-centred phonemes, even when a rig's default mouth
    // form is nonzero. Compose against the driver's neutral, not the rig's.
    this.lipSyncValues = new Map(lipIds.flatMap(id => id ? [[id.getString(), {
      index: this._model.getParameterIndex(id), value: 0, neutral: 0,
    }] as const] : []));
  }

  setGaze(x: number, y: number, source: "pointer" | "ambient" = "pointer"): void {
    this.expressiveIdle?.setGaze(x, y, source);
    if (this.look) this.look.setTarget(x, y);
    else this.setDragging(clamp(x, -1, 1), clamp(y, -1, 1));
  }

  resize(metrics: DisplayMetrics): void {
    const { pixelWidth, pixelHeight } = metrics;
    this.viewport[2] = pixelWidth;
    this.viewport[3] = pixelHeight;
    this.setRenderTargetSize(pixelWidth, pixelHeight);

    this.projection.loadIdentity();
    if (this._model.getCanvasWidth() > 1 && pixelWidth < pixelHeight) {
      this._modelMatrix.setWidth(2);
      this.projection.scale(1, pixelWidth / pixelHeight);
    } else {
      this.projection.scale(pixelHeight / pixelWidth, 1);
    }
    const display = this.packageValue.manifest.display;
    this.projection.scaleRelative(display.scale, display.scale);
    this.projection.translateRelative((display.x - 0.5) * 2, 0);
    this.projection.scale(
      this.projection.getScaleX() * this.viewZoom,
      this.projection.getScaleY() * this.viewZoom,
    );
    this.projection.translateX(this.projection.getTranslateX()
      + 2 * this.viewOffsetX / metrics.cssWidth + this.zoomPanX);
    this.projection.translateY(this.projection.getTranslateY()
      - 2 * this.viewOffsetY / metrics.cssHeight + this.zoomPanY);
    this.displayMetrics = metrics;
    this.updateVisibleBounds();
  }

  setViewOffset(x: number, y: number): void {
    if (x === this.viewOffsetX && y === this.viewOffsetY) return;
    const metrics = this.displayMetrics;
    if (metrics) {
      this.projection.translateX(this.projection.getTranslateX()
        + 2 * (x - this.viewOffsetX) / metrics.cssWidth);
      this.projection.translateY(this.projection.getTranslateY()
        - 2 * (y - this.viewOffsetY) / metrics.cssHeight);
    }
    this.viewOffsetX = x;
    this.viewOffsetY = y;
    this.updateVisibleBounds();
  }

  setViewZoom(zoom: number): void {
    if (!Number.isFinite(zoom) || zoom <= 0 || zoom === this.viewZoom) return;
    if (zoom === 1 && this.displayMetrics) {
      // Back to no zoom: no pan either, so nothing is left over once the
      // panel has shrunk (it pushed TA's head out of a small panel).
      this.viewZoom = 1;
      this.zoomPanX = 0;
      this.zoomPanY = 0;
      this.resize(this.displayMetrics);
      return;
    }
    const ratio = zoom / this.viewZoom;
    const metrics = this.displayMetrics;
    const centerX = metrics && this.modelBounds
      ? 2 * (this.modelBounds.x + this.modelBounds.width / 2) / metrics.cssWidth - 1
      : 0;
    const centerY = metrics && this.modelBounds
      ? 1 - 2 * (this.modelBounds.y + this.modelBounds.height / 2) / metrics.cssHeight
      : 0;
    const previousX = this.projection.getTranslateX();
    const previousY = this.projection.getTranslateY();
    const shiftX = (1 - ratio) * (centerX - previousX);
    const shiftY = (1 - ratio) * (centerY - previousY);
    this.projection.scale(
      this.projection.getScaleX() * ratio,
      this.projection.getScaleY() * ratio,
    );
    this.projection.translateX(previousX + shiftX);
    this.projection.translateY(previousY + shiftY);
    // Kept as a share of the panel, so it scales when the panel is resized.
    this.zoomPanX += shiftX;
    this.zoomPanY += shiftY;
    this.viewZoom = zoom;
    this.updateVisibleBounds();
  }

  getBounds(): PetModelBounds | null {
    return this.modelBounds;
  }

  hitTestScreen(x: number, y: number, metrics: DisplayMetrics): string[] {
    const modelX = this.screenToModelX(x, metrics);
    const modelY = this.screenToModelY(y, metrics);
    const hits: string[] = [];
    for (let index = 0; index < this.setting.getHitAreasCount(); index += 1) {
      if (this.isHit(this.setting.getHitAreaId(index), modelX, modelY)) {
        hits.push(this.setting.getHitAreaName(index));
      }
    }
    return hits;
  }

  containsDraggablePoint(x: number, y: number, metrics: DisplayMetrics): boolean {
    if (this.hitBoundsRefresh.interactionNeedsRefresh()) {
      this.updateDrawableHitBounds();
    }
    const modelX = this.screenToModelX(x, metrics);
    const modelY = this.screenToModelY(y, metrics);
    // A little grace around thin parts, but only around the parts themselves:
    // the empty corners of a slanted prop's box must let clicks through.
    const grace = this._model.getCanvasHeight() * 0.006;
    const points = [[modelX, modelY], [modelX - grace, modelY], [modelX + grace, modelY], [modelX, modelY - grace], [modelX, modelY + grace]] as const;
    for (let index = 0; index < this.drawableHitBounds.length; index += 4) {
      const left = this.drawableHitBounds[index]!;
      if (!Number.isFinite(left)) continue;
      if (
        modelX < left - grace ||
        modelX > this.drawableHitBounds[index + 1]! + grace ||
        modelY < this.drawableHitBounds[index + 2]! - grace ||
        modelY > this.drawableHitBounds[index + 3]! + grace
      ) continue;
      const vertices = this._model.getDrawableVertices(index / 4);
      const triangles = this._model.getDrawableVertexIndices(index / 4);
      if (points.some(([px, py]) => meshContainsPoint(vertices, triangles, px, py))) return true;
    }
    return false;
  }

  async replaceTextures(updates: TextureAtlasUpdate[]): Promise<void> {
    for (const update of updates) {
      if (update.atlasIndex < 0 || update.atlasIndex >= this.textureStore.count()) {
        throw new Error(`纹理图集索引无效：${update.atlasIndex}`);
      }
    }
    await this.textureStore.replace(updates, () => this.draw());
  }

  async replaceTextureSnapshot(
    delta: TextureAtlasDelta,
    isCurrent: () => boolean,
  ): Promise<void> {
    await this.applyTextureSnapshot(delta, isCurrent, () => this.draw());
  }

  private async applyTextureSnapshot(
    delta: TextureAtlasDelta,
    isCurrent: () => boolean,
    validate: () => void,
  ): Promise<void> {
    validateTextureAtlasDelta(delta);
    if (delta.modelId !== this.packageValue.id) {
      throw new Error(`纹理 snapshot 模型不匹配：${delta.modelId}`);
    }
    for (const update of [...delta.atlases, ...delta.patchSets]) {
      if (update.atlasIndex < 0 || update.atlasIndex >= this.textureStore.count()) {
        throw new Error(`纹理图集索引无效：${update.atlasIndex}`);
      }
    }
    if (delta.atlases.length > 0) {
      await this.textureStore.replace(delta.atlases, validate);
    }
    if (delta.patchSets.length > 0) {
      await this.textureStore.replacePatchSets(delta.patchSets, validate, isCurrent);
    }
  }

  async setTextureMaximumDimension(value: number): Promise<void> {
    await this.textureStore.setMaximumDimension(value, () => this.draw());
  }

  textureMaximumDimension(): number {
    return this.textureStore.maxDimension();
  }

  async restoreWebGL(metrics: DisplayMetrics): Promise<void> {
    const recipes = this.textureStore.recipes();
    this.textureStore.release();
    this.deleteRenderer();
    CubismShaderManager_WebGL.deleteInstance();
    CubismWebGLOffscreenManager.getInstance().removeContext(this.gl);
    this.createRenderer(metrics.pixelWidth, metrics.pixelHeight, this.maskBufferCount());
    const renderer = this.getRenderer();
    renderer.startUp(this.gl);
    renderer.loadShaders(SHADER_PATH);
    renderer.setIsPremultipliedAlpha(true);
    this.textureStore = this.createTextureStore(renderer);
    await this.textureStore.loadAll(recipes.map(recipeUrl));
    const patchSets = recipes.flatMap((recipe) =>
      recipe.kind === "patch-set" ? [recipe.value] : []
    );
    if (patchSets.length > 0) {
      await this.textureStore.replacePatchSets(patchSets, () => undefined, () => true);
    }
    await waitForShaders(this.gl);
    this.resize(metrics);
  }

  private maskBufferCount(): number {
    return cubismMaskBufferCount(this._model.getDrawableMasks(), this._model.getOffscreenMasks());
  }

  update(deltaSeconds: number): void {
    if (!this._model || this.released) return;
    if (this.facialDemo) {
      this.updateFacialDemo(Number.isFinite(deltaSeconds) ? clamp(deltaSeconds, 0, 0.1) : 0);
      return;
    }
    const delta = clamp(deltaSeconds, 1 / 240, 0.1);
    this.facialOverlay?.layer.controller.hearVoice(this.lipTarget.volume, this.lipTarget.pitch);
    if (this.facialOverlay && !this.facialOverlay.animateBase) {
      this.updateStaticFacialOverlay(delta); return;
    }
    const active = this.activeBehavior;
    if (
      active &&
      active.finishAfterOneCycle &&
      active.cycleDurationSeconds !== null &&
      !active.layers.hasMotions() &&
      active.elapsedSeconds + delta >= active.cycleDurationSeconds
    ) {
      // Expression-only candidates have no Motion completion callback, so use
      // their fixed duration before the next frame instead of waiting for the
      // fallback timer to fire.
      this.finishBehavior(active);
    }
    this._model.loadParameters();
    const current = this.activeBehavior;
    if (current) current.elapsedSeconds += delta;
    this.motionUpdated = false;
    if (current) this.motionUpdated = current.layers.updateMotions(this._model, delta);
    // Persist the new group's own baseline before blending the displayed pose.
    // Otherwise unkeyed parameters feed the previous frame's blend back into
    // the next frame's target and can retain the old body's pose indefinitely.
    this._model.saveParameters();
    this.applyParameterTransition(delta);
    this.scheduler.onLateUpdate(this._model, delta);
    const withLipSync = !this.sampleState || this.sampleState.mode === "runtime";
    if (withLipSync) {
      this.updateLip(delta);
      // Appearance presets may write these parameters too. Retain the actual
      // phoneme values so their absolute preset values cannot change its sign.
      if (this.facialOverlay) {
        for (const channel of this.lipSyncValues.values()) {
          channel.value = this._model.getParameterValueByIndex(channel.index);
        }
      }
    }
    this.defaultAppearance?.apply(this._model);
    this.finishFacialOverlayFrame(withLipSync);
    this._model.update();
    if (!this.sampleState && this.hitBoundsRefresh.afterModelUpdate(delta)) {
      this.updateDrawableHitBounds();
    }
    if (current && current === this.activeBehavior) {
      if (!current.loopMotions && current.layers.motionsFinished()) {
        this.queueBehaviorFinish(current);
      }
      if (this.sampleState && !current.loopMotions && !current.layers.hasMotions() && current.elapsedSeconds >= 2) {
        this.queueBehaviorFinish(current);
      }
    }
  }

  draw(): void {
    if (!this._model || this.released || this.gl.isContextLost()) return;
    this.unbindRendererTextures();
    CubismWebGLOffscreenManager.getInstance().beginFrameProcess(this.gl);
    try {
      this.drawMatrix.setMatrix(this.projection.getArray());
      this.drawMatrix.multiplyByMatrix(this._modelMatrix);
      const renderer = this.getRenderer();
      renderer.setMvpMatrix(this.drawMatrix);
      renderer.setRenderState(null as unknown as WebGLFramebuffer, this.viewport);
      renderer.drawModel(SHADER_PATH);
    } finally {
      // A cancelled/failed candidate must never leave an offscreen target
      // active for the next model. Otherwise WebGL can sample the same texture
      // it is currently rendering into and the whole preview needs recovery.
      finishCubismOffscreenFrame(this.gl, () => this.unbindRendererTextures());
    }
  }

  private createTextureStore(renderer: CubismRenderer_WebGL): CubismTextureStore {
    return new CubismTextureStore(this.gl, renderer, {
      maxTextureDimension: this.options.maxTextureDimension,
      onCreated: this.options.onTextureCreated,
      onDeleted: this.options.onTextureDeleted,
    });
  }

  private unbindRendererTextures(): void {
    // Official R5 uses texture units 0–2 for model/offscreen composition. Its
    // copy pass leaves RenderTarget textures bound, so the next frame can try
    // to render into a texture that is still an active sampler. Explicitly
    // unbinding prevents Chrome's framebuffer feedback-loop INVALID_OPERATION.
    for (let unit = 0; unit <= 2; unit += 1) {
      this.gl.activeTexture(this.gl.TEXTURE0 + unit);
      this.gl.bindTexture(this.gl.TEXTURE_2D, null);
    }
    this.gl.activeTexture(this.gl.TEXTURE0);
  }

  override release(): void {
    if (this.released) return;
    this.released = true;
    this.statePoseSelection = null;
    this.transientPoseSelection = null;
    this.stopPersistentExpressions();
    this.stopBehavior();
    this.textureStore?.release();
    this.expressiveIdle?.release();
    this.layerVisibility?.release();
    this.scheduler.release();
    for (const motion of this.motions.values()) motion.release();
    for (const expression of this.expressions.values()) expression.release();
    this.motions.clear();
    this.expressions.clear();
    this.motionOwnership.clear();
    this.faceOnlyExpressionAssetIds.clear();
    this.setting.release();
    super.release();
  }

  private async loadExpressions(selected?: Set<string>): Promise<void> {
    await allSettledOrThrow(Object.values(this.packageValue.manifest.expressionAssets).filter((asset) => !selected || selected.has(asset.id)).map(async (asset) => {
      const buffer = await this.fetchPackageBuffer(asset.file);
      const expression = this.loadExpression(buffer, buffer.byteLength, asset.expression);
      if (expression) {
        this.expressions.set(asset.expression, expression);
        if (isFaceOnlyExpression(expression, this.packageValue.capabilities.parameters ?? [])) {
          this.faceOnlyExpressionAssetIds.add(asset.id);
        }
      }
    }));
    if (this.expressions.size > 0 && !this.sampleState) {
      this.scheduler.addUpdatableList(new CubismExpressionUpdater(this._expressionManager));
    }
  }

  private async loadOptionalResources(): Promise<void> {
    const jobs: Promise<void>[] = [];
    const physics = this.setting.getPhysicsFileName();
    if (physics) jobs.push(this.fetchBuffer(physics).then((buffer) => {
      if (this.sampleState) this.sampleState.physics = buffer;
      this.loadPhysics(buffer, buffer.byteLength);
      this.facialDemoPhysicsOutputs = facialDemoPhysicsOutputOnlyIds(buffer);
      if (!this.sampleState) this.naturalIdleAffectedParameters = readNaturalIdlePhysicsParameters(buffer);
      if (this._physics) this.scheduler.addUpdatableList(new CubismPhysicsUpdater(this._physics));
    }));
    const pose = this.setting.getPoseFileName();
    if (pose) jobs.push(this.fetchBuffer(pose).then((buffer) => {
      if (this.sampleState) this.sampleState.pose = buffer;
      this.loadPose(buffer, buffer.byteLength);
      if (this._pose) this.scheduler.addUpdatableList(new CubismPoseUpdater(this._pose));
    }));
    const userData = this.setting.getUserDataFile();
    if (userData) jobs.push(this.fetchBuffer(userData).then((buffer) => {
      this.loadUserData(buffer, buffer.byteLength);
    }));
    await allSettledOrThrow(jobs);
  }

  private async loadMotions(selected?: Set<string>): Promise<void> {
    const byGroup = new Map<string, MotionAsset[]>();
    for (const asset of Object.values(this.packageValue.manifest.motionAssets)) {
      if (selected && !selected.has(asset.id)) continue;
      const group = byGroup.get(asset.group) ?? [];
      group.push(asset);
      byGroup.set(asset.group, group);
    }
    await allSettledOrThrow([...byGroup.entries()].flatMap(([group, assets]) => {
      return assets.map(async (asset) => {
        const buffer = await this.fetchPackageBuffer(asset.file);
        const motion = this.loadMotion(
          buffer,
          buffer.byteLength,
          `${group}_${asset.index}`,
          undefined,
          undefined,
          this.setting,
          group,
          asset.index,
          true,
        );
        if (!motion) return;
        this.motionOwnership.set(asset.id, motionParameterIds(buffer, this.eyeBlinkIds.map(id=>id.getString()), this.lipSyncIds.map(id=>id.getString())));
        motion.setEffectIds(this.eyeBlinkIds, this.lipSyncIds);
        this.motions.set(motionKey(group, asset.index), motion);
      });
    }));
  }

  private collectEffectIds(): void {
    for (let i = 0; i < this.setting.getEyeBlinkParameterCount(); i += 1) {
      this.eyeBlinkIds.push(this.setting.getEyeBlinkParameterId(i));
    }
    for (let i = 0; i < this.setting.getLipSyncParameterCount(); i += 1) {
      this.lipSyncIds.push(this.setting.getLipSyncParameterId(i));
    }
  }

  private configureEffects(): void {
    if (this.eyeBlinkIds.length > 0) {
      this._eyeBlink = CubismEyeBlink.create(this.setting);
      this.scheduler.addUpdatableList(
        new CubismEyeBlinkUpdater(() => this.motionUpdated, this._eyeBlink),
      );
    }
    this._breath = CubismBreath.create();
    this._breath.setParameters([
      new BreathParameterData(this.requiredId(CubismDefaultParameterId.ParamAngleX), 0, 12, 6.5, 0.35),
      new BreathParameterData(this.requiredId(CubismDefaultParameterId.ParamAngleY), 0, 5, 4.1, 0.3),
      new BreathParameterData(this.requiredId(CubismDefaultParameterId.ParamAngleZ), 0, 7, 5.3, 0.25),
      new BreathParameterData(this.requiredId(CubismDefaultParameterId.ParamBodyAngleX), 0, 3, 15.5, 0.35),
      new BreathParameterData(this.requiredId(CubismDefaultParameterId.ParamBreath), 0.5, 0.5, 3.2, 1),
    ]);
    this.scheduler.addUpdatableList(new CubismBreathUpdater(this._breath));

    const look = CubismLook.create();
    look.setParameters([
      new LookParameterData(this.requiredId(CubismDefaultParameterId.ParamAngleX), 24, 0, 0),
      new LookParameterData(this.requiredId(CubismDefaultParameterId.ParamAngleY), 0, 20, 0),
      new LookParameterData(this.requiredId(CubismDefaultParameterId.ParamAngleZ), 0, 0, -18),
      new LookParameterData(this.requiredId(CubismDefaultParameterId.ParamBodyAngleX), 6, 0, 0),
    ]);
    const eyes = CubismLook.create();
    eyes.setParameters([
      new LookParameterData(this.requiredId(CubismDefaultParameterId.ParamEyeBallX), 1, 0, 0),
      new LookParameterData(this.requiredId(CubismDefaultParameterId.ParamEyeBallY), 0, 1, 0),
    ]);
    this.look = new LeadingLookUpdater(look, eyes, this._dragManager);
    this.scheduler.addUpdatableList(this.look);
    // Analysis samples must measure authored assets, including in runtime mode.
    if (!this.sampleState) {
      this.naturalIdleUpdater = new CubismNaturalIdleUpdater(
        this._model,
        () => !this.expressiveIdle?.isActive() && !this.speaking && !this.activeBehavior?.blocksNaturalIdle
          && !this.defaultAppearance?.affects(this.naturalIdleAffectedParameters),
      );
      this.naturalIdleUpdater.configure(this.packageValue.manifest.naturalIdle);
      this.scheduler.addUpdatableList(this.naturalIdleUpdater);
      this.expressiveIdle = new CubismExpressiveIdle(this._model, this._physics ?? null);
      this.expressiveIdle.configure(this.packageValue.manifest.naturalIdle);
      this.refreshIdleOwnership();
      for (const stage of this.expressiveIdle.stages) this.scheduler.addUpdatableList(stage);
    }
    this.scheduler.sortUpdatableList();
  }

  private queueBehaviorFinish(group: ActiveBehaviorGroup): void {
    if (group.finishQueued || this.activeBehavior !== group) return;
    if (this.sampleState) {
      this.sampleState.pendingFinish = group;
      return;
    }
    queueMicrotask(() => {
      this.finishBehavior(group);
    });
  }

  private finishBehavior(group: ActiveBehaviorGroup): void {
    if (group.finishQueued || this.activeBehavior !== group) return;
    group.finishQueued = true;
    const onFinished = group.onFinished;
    onFinished?.();
    if (this.activeBehavior === group) this.stopBehavior();
  }

  private captureParameterValues(): Float32Array {
    if (!this._model) return new Float32Array(0);
    const values = new Float32Array(this._model.getParameterCount());
    for (let index = 0; index < values.length; index += 1) {
      values[index] = this._model.getParameterValueByIndex(index);
    }
    return values;
  }

  private captureTransitionSourceValues(): Float32Array {
    if (!this._model) return new Float32Array(0);
    // Reconstruct the last displayed Motion pose from its clean saved baseline.
    // A replacement during a handoff must continue from the current blend,
    // while still excluding late effects such as breath, gaze, and lip-sync.
    this._model.loadParameters();
    this.applyParameterTransition(0);
    return this.captureParameterValues();
  }

  private applyParameterTransition(deltaSeconds: number): void {
    const transition = this.parameterTransition;
    if (!transition || !this._model) return;
    transition.elapsedSeconds = Math.min(
      transition.durationSeconds,
      transition.elapsedSeconds + deltaSeconds,
    );
    const progress = transition.durationSeconds <= 0
      ? 1
      : transition.elapsedSeconds / transition.durationSeconds;
    const eased = 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, Math.min(1, progress)));
    const count = Math.min(this._model.getParameterCount(), transition.from.length);
    for (let index = 0; index < count; index += 1) {
      const target = this._model.getParameterValueByIndex(index);
      const from = transition.from[index]!;
      this._model.setParameterValueByIndex(index, from + (target - from) * eased, 1);
    }
    if (progress >= 1) this.parameterTransition = null;
  }

  private updateLip(delta: number): void {
    const value = this.lipTarget;
    const desiredOpen = mouthOpening(value) * this.maxMouthOpen;
    // Mouth form widens the mouth: a wide open "a", spread "e"/"i", rounded "o"/"u".
    const desiredForm = clamp(
      value.a * 0.35 + (value.e + value.i) * 0.55 - (value.o + value.u) * 0.3,
      -1,
      1,
    );
    const elapsed = delta * 1000;
    const openFactor = 1 - Math.exp(-elapsed / (desiredOpen > this.currentOpen ? 65 : 115));
    const formFactor = 1 - Math.exp(-elapsed / 140);
    this.currentOpen += (desiredOpen - this.currentOpen) * openFactor;
    this.currentForm += (desiredForm - this.currentForm) * formFactor;
    const parts = mouthParts(value);
    this.currentParts.jaw += (parts.jaw * this.maxMouthOpen - this.currentParts.jaw) * openFactor;
    this.currentParts.puckerWiden += (parts.puckerWiden - this.currentParts.puckerWiden) * formFactor;
    this.currentParts.funnel += (parts.funnel - this.currentParts.funnel) * formFactor;
    const vowelTotal = this.lipTarget.a + this.lipTarget.e + this.lipTarget.i + this.lipTarget.o + this.lipTarget.u;
    for (const vowel of VOWELS) {
      const fallback = vowel === "a" && vowelTotal < 0.001 ? this.lipTarget.volume : 0;
      this.currentVowels[vowel] += (Math.max(this.lipTarget[vowel], fallback) - this.currentVowels[vowel]) * openFactor;
    }

    if (this.lipSyncMode === "viseme") {
      for (const vowel of VOWELS) {
        const id = this.directVowelIds[vowel];
        if (id) this._model.setParameterValueById(id, this.currentVowels[vowel] * this.maxMouthOpen);
      }
    } else if (this.lipSyncMode === "open-form" && this.mouthOpenId) {
      this._model.setParameterValueById(this.mouthOpenId, this.currentOpen);
      if (this.mouthFormId) this._model.setParameterValueById(this.mouthFormId, this.currentForm);
      for (const part of ["jaw", "puckerWiden", "funnel"] as const) {
        const id = this.mouthPartIds[part];
        if (id) this._model.setParameterValueById(id, this.currentParts[part]);
      }
    }

    const desiredEnergy = this.speaking ? Math.min(1, Math.max(this.lipTarget.volume, this.currentOpen)) : 0;
    this.talkEnergy += (desiredEnergy - this.talkEnergy) * (1 - Math.exp(-elapsed / 180));
    this.talkPhase += elapsed * (0.0032 + this.talkEnergy * 0.0012);
    if (this.speaking && !this.expressiveIdle?.isActive()) {
      this.add("ParamAngleX", Math.sin(this.talkPhase * 0.63 + 0.4) * (0.55 + this.talkEnergy * 0.8));
      this.add("ParamAngleY", Math.sin(this.talkPhase * 0.48 + 1.1) * (0.35 + this.talkEnergy * 0.55));
      this.add("ParamAngleZ", Math.sin(this.talkPhase) * (0.45 + this.talkEnergy * 1.1));
      this.add("ParamBodyAngleX", Math.sin(this.talkPhase * 0.72 + 0.8) * (0.2 + this.talkEnergy * 0.55));
    }
  }

  private updateDrawableHitBounds(): void {
    const count = this._model.getDrawableCount();
    for (let drawable = 0; drawable < count; drawable += 1) {
      const offset = drawable * 4;
      if (
        !this._model.getDrawableDynamicFlagIsVisible(drawable) ||
        this._model.getDrawableOpacity(drawable) <= 0.02
      ) {
        this.drawableHitBounds[offset] = Number.NaN;
        continue;
      }
      const vertices = this._model.getDrawableVertices(drawable);
      if (vertices.length < 2) {
        this.drawableHitBounds[offset] = Number.NaN;
        continue;
      }
      let left = vertices[0]!;
      let right = left;
      let bottom = vertices[1]!;
      let top = bottom;
      for (let index = 2; index < vertices.length; index += 2) {
        const vx = vertices[index]!;
        const vy = vertices[index + 1]!;
        if (vx < left) left = vx;
        if (vx > right) right = vx;
        if (vy < bottom) bottom = vy;
        if (vy > top) top = vy;
      }
      this.drawableHitBounds[offset] = left;
      this.drawableHitBounds[offset + 1] = right;
      this.drawableHitBounds[offset + 2] = bottom;
      this.drawableHitBounds[offset + 3] = top;
    }
    this.updateVisibleBounds();
    this.hitBoundsRefresh.markRefreshed();
  }

  private updateVisibleBounds(): void {
    const metrics = this.displayMetrics;
    if (!metrics) return;
    let left = Number.POSITIVE_INFINITY;
    let right = Number.NEGATIVE_INFINITY;
    let bottom = Number.POSITIVE_INFINITY;
    let top = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < this.drawableHitBounds.length; index += 4) {
      const drawableLeft = this.drawableHitBounds[index]!;
      if (!Number.isFinite(drawableLeft)) continue;
      if (drawableLeft < left) left = drawableLeft;
      if (this.drawableHitBounds[index + 1]! > right) right = this.drawableHitBounds[index + 1]!;
      if (this.drawableHitBounds[index + 2]! < bottom) bottom = this.drawableHitBounds[index + 2]!;
      if (this.drawableHitBounds[index + 3]! > top) top = this.drawableHitBounds[index + 3]!;
    }
    if (!Number.isFinite(left)) {
      this.modelBounds = null;
      return;
    }
    const leftClip = this.projection.transformX(this._modelMatrix.transformX(left));
    const rightClip = this.projection.transformX(this._modelMatrix.transformX(right));
    const bottomClip = this.projection.transformY(this._modelMatrix.transformY(bottom));
    const topClip = this.projection.transformY(this._modelMatrix.transformY(top));
    const x1 = ((Math.min(leftClip, rightClip) + 1) * 0.5) * metrics.cssWidth;
    const x2 = ((Math.max(leftClip, rightClip) + 1) * 0.5) * metrics.cssWidth;
    const y1 = ((1 - Math.max(topClip, bottomClip)) * 0.5) * metrics.cssHeight;
    const y2 = ((1 - Math.min(topClip, bottomClip)) * 0.5) * metrics.cssHeight;
    this.modelBounds = { x: x1, y: y1, width: x2 - x1, height: y2 - y1, middle: this.middleReach(bottom, top, metrics) };
  }

  /**
   * How wide the body is halfway down: where the meshes actually cross that
   * line, so a wide hat, hair or a slanted prop's box doesn't push it out.
   */
  private middleReach(bottom: number, top: number, metrics: DisplayMetrics): PetModelBounds["middle"] {
    const middle = (bottom + top) / 2;
    const screenY = ((1 - this.projection.transformY(this._modelMatrix.transformY(middle))) * 0.5) * metrics.cssHeight;
    // Check a band as tall as the chat entry, so a hand or prop just above or below doesn't sit under it.
    const lines = [-MIDDLE_BAND, -MIDDLE_BAND / 2, 0, MIDDLE_BAND / 2, MIDDLE_BAND].map(offset => offset ? this.screenToModelY(screenY + offset, metrics) : middle);
    let left = Number.POSITIVE_INFINITY;
    let right = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < this.drawableHitBounds.length; index += 4) {
      if (!Number.isFinite(this.drawableHitBounds[index]!)) continue;
      const crossed = lines.filter(y => this.drawableHitBounds[index + 2]! <= y && this.drawableHitBounds[index + 3]! >= y);
      if (!crossed.length) continue;
      const drawable = index / 4;
      const vertices = this._model.getDrawableVertices(drawable);
      const triangles = this._model.getDrawableVertexIndices(drawable);
      for (const y of crossed) {
        for (let corner = 0; corner + 2 < triangles.length; corner += 3) {
          for (let edge = 0; edge < 3; edge += 1) {
            const a = triangles[corner + edge]! * 2, b = triangles[corner + (edge + 1) % 3]! * 2;
            const ay = vertices[a + 1]!, by = vertices[b + 1]!;
            if ((ay > y) === (by > y)) continue;
            const x = vertices[a]! + (y - ay) / (by - ay) * (vertices[b]! - vertices[a]!);
            if (x < left) left = x;
            if (x > right) right = x;
          }
        }
      }
    }
    if (!Number.isFinite(left)) return undefined;
    const toX = (x: number) => ((this.projection.transformX(this._modelMatrix.transformX(x)) + 1) * 0.5) * metrics.cssWidth;
    const [x1, x2] = [toX(left), toX(right)].sort((a, b) => a - b);
    return { y: screenY, left: x1!, right: x2! };
  }

  private screenToModelX(x: number, metrics: DisplayMetrics): number {
    const clipX = (x / metrics.cssWidth) * 2 - 1;
    const viewX = this.projection.invertTransformX(clipX);
    return this._modelMatrix.invertTransformX(viewX);
  }

  private screenToModelY(y: number, metrics: DisplayMetrics): number {
    const clipY = 1 - (y / metrics.cssHeight) * 2;
    const viewY = this.projection.invertTransformY(clipY);
    return this._modelMatrix.invertTransformY(viewY);
  }

  private add(id: string, value: number): void {
    this._model.addParameterValueById(this.id(id), value);
  }

  private id(value: string): CubismIdHandle;
  private id(value: null | undefined): null;
  private id(value: string | null | undefined): CubismIdHandle | null;
  private id(value: string | null | undefined): CubismIdHandle | null {
    if (!value) return null;
    const cached = this.parameterIds.get(value);
    if (cached) return cached;
    const id = CubismFramework.getIdManager().getId(value);
    this.parameterIds.set(value, id);
    return id;
  }

  private requiredId(value: string | undefined): CubismIdHandle {
    if (!value) throw new Error("Cubism 标准参数 ID 缺失");
    return this.id(value);
  }

  private textureUrls(): string[] {
    const urls: string[] = [];
    for (let index = 0; index < this.setting.getTextureCount(); index += 1) {
      urls.push(immutableTextureBaseUrl(this.resolve(this.setting.getTextureFileName(index))));
    }
    return urls;
  }

  private async fetchBuffer(file: string): Promise<ArrayBuffer> {
    const url = this.resolve(file);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`读取 Live2D 资源失败 (${response.status}): ${file}`);
    return response.arrayBuffer();
  }

  private async fetchPackageBuffer(file: string): Promise<ArrayBuffer> {
    const depth = Math.max(0, this.packageValue.manifest.entry.split("/").length - 1);
    const root = new URL("../".repeat(depth), this.packageValue.url);
    const url = new URL(file, root).toString();
    const response = await fetch(url);
    if (!response.ok) throw new Error(`读取 Live2D 素材失败 (${response.status}): ${file}`);
    return response.arrayBuffer();
  }

  private resolve(file: string): string {
    return new URL(file, this.packageValue.url).toString();
  }
}

function recipeUrl(recipe: TextureRecipe): string {
  return recipe.kind === "patch-set" ? recipe.value.baseUrl : recipe.value.url;
}

function motionKey(group: string, index: number): string {
  return `${group}\u0000${index}`;
}

function motionGroupTimeoutMs(motions: CubismMotion[]): number {
  const durations = motions.map((motion) => motion.getLoopDuration());
  if (durations.some((value) => !Number.isFinite(value) || value <= 0)) return 8_000;
  return Math.max(...durations) * 1_000 + 750;
}

function behaviorTransitionDuration(
  outgoingFadeOutSeconds: number,
  incomingFadeInSeconds: number,
): number {
  const authored = Math.max(outgoingFadeOutSeconds, incomingFadeInSeconds);
  return Math.min(
    MAX_BEHAVIOR_TRANSITION_SECONDS,
    Math.max(MIN_BEHAVIOR_TRANSITION_SECONDS, authored),
  );
}

function releaseBehaviorGroup(group: ActiveBehaviorGroup): void {
  if (group.timeout !== null) clearTimeout(group.timeout);
  group.timeout = null;
  group.layers.release();
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

async function waitForShaders(gl: WebGL2RenderingContext): Promise<void> {
  const shader = CubismShaderManager_WebGL.getInstance().getShader(gl) as unknown as {
    _isShaderLoaded?: boolean;
  };
  const deadline = performance.now() + 10_000;
  while (!shader._isShaderLoaded) {
    if (performance.now() >= deadline) throw new Error("Cubism WebGL Shader 加载超时");
    await new Promise<void>((resolve) => window.setTimeout(resolve, 16));
  }
}

export type { DisplayMetrics as CubismDisplayMetrics };
