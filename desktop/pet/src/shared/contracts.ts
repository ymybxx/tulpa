import type { SystemSettingsNames } from "./system-settings-names";
import type { SettingsSection } from "../settings-next/model";
import type { EmotionAdoption, EmotionMatchingApi } from "./emotion-matching";

export const PET_STATES = [
  "unbound",
  "standby",
  "acknowledging",
  "listening",
  "thinking",
  "speaking",
] as const;

export const PET_EMOTIONS = [
  "neutral",
  "angry",
  "happy",
  "shy",
  "sad",
] as const;

export const PET_BEHAVIOR_STATES = [
  "standby",
  "acknowledging",
  "listening",
  "thinking",
  "click",
] as const;

export type PetState = (typeof PET_STATES)[number];
export type PetEmotion = (typeof PET_EMOTIONS)[number];
export type PetBehaviorState = (typeof PET_BEHAVIOR_STATES)[number];

export interface MotionRef {
  group: string;
  index?: number;
}

export interface MotionAsset extends MotionRef {
  id: string;
  name: string;
  file: string;
  index: number;
  contentHash: string;
}

export interface ExpressionAsset {
  id: string;
  name: string;
  file: string;
  expression: string;
  contentHash: string;
}

export interface BehaviorCandidate {
  id: string;
  /** Motion layers start together and update in this stable order. */
  motionAssetIds: string[];
  /** Expression layers update after motions in this stable order. */
  expressionAssetIds: string[];
}

export interface PetManifest {
  schemaVersion: 3;
  id: string;
  name: string;
  renderer: "live2d-cubism";
  entry: string;
  display: {
    scale: number;
    x: number;
    y: number;
  };
  /** Per-model procedural idle. Missing expressive settings preserve the legacy driver. */
  naturalIdle?: NaturalIdleSettings;
  /** Ordered model-provided switches held as the saved appearance in every state. */
  defaultAppearance?: DefaultAppearanceSettings;
  /** Named looks the user saved from 装扮, applied as a whole default appearance. */
  outfits?: ModelOutfit[];
  /** The user's part grouping; absent until edited, when the automatic grouping applies. */
  partGroups?: ModelPartGroup[];
  motionAssets: Record<string, MotionAsset>;
  expressionAssets: Record<string, ExpressionAsset>;
  behaviors: {
    states: Record<PetBehaviorState, BehaviorCandidate[]>;
    emotions: Record<PetEmotion, BehaviorCandidate[]>;
  };
  lipsync: {
    mode: LipSyncAdapterMode;
    source: LipSyncMappingSource;
    needsConfirmation: boolean;
    mouthOpenParameter: string | null;
    mouthFormParameter: string | null;
    jawParameter: string | null;
    puckerWidenParameter: string | null;
    funnelParameter: string | null;
    vowelParameters: VowelParameterMap;
    maxOpen: number;
  };
}

export interface NaturalIdleSettings {
  enabled: boolean;
  intensity: number;
  speed: number;
  expressive?: ExpressiveIdleSettings;
}

export type IdlePreset = "gentle" | "natural" | "active";
export type IdleChannel = "headX" | "headY" | "headZ" | "bodyX" | "bodyY" | "bodyZ" |
  "eyeX" | "eyeY" | "eyeLOpen" | "eyeROpen";
export interface ExpressiveIdleSettings {
  preset: IdlePreset;
  bodyIntensity: number;
  eyeActivity: number;
  pauseProbability: number;
  largeMotionProbability: number;
  reverseInterval: number;
  followDelay: number;
  yawRange: number;
  pitchRange: number;
  rollRange: number;
  blink: boolean;
  /** Absent uses automatic binding; null explicitly disables a channel. */
  bindings?: Partial<Record<IdleChannel, string | null>>;
  bodyDirections?: Partial<Record<"x" | "y" | "z", -1 | 1>>;
}

export interface ModelPartGroup {
  id: string;
  name: string;
  /** ArtMesh IDs; a drawable belongs to at most one group. */
  drawableIds: string[];
}

export interface PartRecognitionUpdate {
  modelId: string;
  running: boolean;
  done: number;
  total: number;
  /** Assignments made since the previous update, by texture unit. */
  assignments: Array<{ unitId: string; groupName: string }>;
  message: string;
}

export interface PartRecognitionApi {
  /** Groups the given ungrouped pieces with the picture-reading model; results arrive as updates. */
  start(modelId: string, groupNames: string[], unitIds: string[]): Promise<void>;
  cancel(modelId: string): Promise<void>;
  onUpdate(callback: (update: PartRecognitionUpdate) => void): () => void;
}

export interface ModelOutfit {
  id: string;
  name: string;
  appearance: DefaultAppearanceSettings;
}

export interface ModelToggleCreation {
  model: ModelPackage;
  /** The new expression asset, ready to be switched on in the draft. */
  assetId: string;
}

export interface DefaultAppearanceSettings {
  expressionAssetIds: string[];
  /** Explicit native parameter overrides, applied after selected expressions. */
  parameterValues?: Record<string, number>;
  /** Suppressed at draw time; enabling a layer restores its authored visibility. */
  hiddenDrawableIds?: string[];
  layerGroups?: ModelLayerGroup[];
}

export interface ModelLayerGroup { id: string; name: string; drawableIds: string[] }
export interface NativeModelLayer {
  id: string;
  name: string;
  partId: string | null;
  partName: string;
  /** Visibility from the model, before the user's saved suppression. */
  visible: boolean;
}

/** Read from the loaded Cubism model, not inferred from parameter names. */
export interface NativeModelParameter {
  id: string;
  name: string;
  minimum: number;
  maximum: number;
  defaultValue: number;
}

export interface ModelCapabilities {
  motionGroups: string[];
  expressions: string[];
  hitAreas: string[];
  /** Parameter IDs declared by the model3.json LipSync group, in priority order. */
  lipSyncParameters?: string[];
  /** Parameter metadata exported by the model's DisplayInfo (.cdi3.json). */
  parameters?: ModelParameterDescriptor[];
  parts?: Array<{ id: string; name: string }>;
  /** Moc generation reported by the installed Cubism 5.3 Core. */
  mocVersion?: number;
  usesMotionSync?: boolean;
  usesAdvancedBlendMode?: boolean;
  usesOffscreenDrawing?: boolean;
  warnings?: string[];
}

export const MODEL_EDIT_PART_GROUPS = [
  "iris",
  "eye",
  "brow",
  "face",
  "mouth",
  "hair-front",
  "hair-side",
  "hair-back",
  "clothing",
  "body",
  "arm",
  "leg",
  "shoe",
  "accessory",
  "effect",
  "unknown",
] as const;

export type ModelEditPartGroup = (typeof MODEL_EDIT_PART_GROUPS)[number];
export type ModelEditMode = "appearance" | "contour";

export interface ModelEditDirectionalSlack {
  /** Neutral Cubism canvas pixels between visible alpha and the mesh boundary. */
  pixels: number;
  /** Slack relative to the visible part width or height. */
  ratio: number;
}

export interface ModelEditPartCapability {
  id: string;
  label: string;
  group: ModelEditPartGroup;
  drawableIds: string[];
  partIds: string[];
  visiblePixels: number;
  operations: {
    /** RGB may change and alpha may only decrease inside the original silhouette. */
    appearance: { supported: boolean; alphaPolicy: "shrink-only" };
    /** Reported for inspection, but disabled until motion-sweep validation passes. */
    contour: {
      supported: boolean;
      reason: string;
      staticSlack: {
        top: ModelEditDirectionalSlack;
        right: ModelEditDirectionalSlack;
        bottom: ModelEditDirectionalSlack;
        left: ModelEditDirectionalSlack;
      };
    };
  };
}

export interface TextureEditUnit {
  /** Stable for the current atlas pixels and exact UV coverage. */
  id: string;
  atlasIndex: number;
  /** Current active atlas snapshot used as the source of the next AI edit. */
  sourceAtlasFile?: string;
  /** Stable base-atlas path used for composing and restoring material variants. */
  atlasFile: string;
  maskHash: string;
  bounds: { x: number; y: number; width: number; height: number };
  maskPath: string;
  /** Base model pixels, used only by the material picker when choosing the original version. */
  originalPreviewPath?: string;
  /** Current active appearance pixels, used by AI editing and live selection. */
  previewPath: string;
  /** Exact neutral-pose projection of every ArtMesh that references this UV unit. */
  modelMaskPath: string;
  modelMaskBounds: {
    x: number;
    y: number;
    width: number;
    height: number;
    canvasWidth: number;
    canvasHeight: number;
  };
  drawableIds: string[];
  partIds: string[];
  semanticPartIds: string[];
  /** Drawable, Part IDs and Part names, for finding pieces by name. */
  searchText?: string;
  /**
   * Stable motion asset IDs that reveal an otherwise hidden drawable using
   * this texture. Display names are intentionally resolved from the current
   * PetManifest so renaming a motion never leaves this annotation stale.
   */
  revealedByMotionIds: string[];
  label: string;
  /**
   * A clipping mask that is never drawn: it only cuts the shape of the pieces
   * layered on it, so repainting it changes nothing anyone can see.
   */
  maskOnly?: boolean;
  visiblePixels: number;
  modelBounds: { x: number; y: number; width: number; height: number };
}

export interface ModelEditCapabilityManifest {
  schemaVersion: 12;
  modelId: string;
  modelName: string;
  templateId: string;
  sourceHash: string;
  generatedAt: string;
  validation: {
    geometry: "neutral-static";
    motionSweep: "not-run";
    alphaPolicy: "shrink-only";
  };
  parts: ModelEditPartCapability[];
  textureUnits: TextureEditUnit[];
}

export interface ModelPackage {
  id: string;
  name: string;
  url: string;
  previewUrl: string | null;
  manifest: PetManifest;
  capabilities: ModelCapabilities;
}

export interface ModelRenderBundle {
  model: ModelPackage;
  appearance: TextureAtlasDelta;
}

export interface TextureMaterial {
  id: string;
  textureUnitId: string;
  label: string;
  atlasIndex: number;
  atlasFile: string;
  bounds: { x: number; y: number; width: number; height: number };
  name: string;
  prompt: string;
  createdAt: string;
  imageUrl: string;
  active: boolean;
  /** The model's own pixels, kept forever; every other material is a saved variant. */
  original?: boolean;
  /** Made by 换色 without AI; `prompt` then holds the color as #RRGGBB. */
  origin?: "recolor";
}

export interface TextureMaterialLibrary {
  schemaVersion: 4;
  modelId: string;
  /** Monotonic appearance revision used to reject stale renderer updates. */
  revision: number;
  materials: TextureMaterial[];
  activeAssignments: Record<string, string>;
  updatedAt: string;
}

export interface TextureAtlasUpdate {
  atlasIndex: number;
  atlasFile: string;
  /** Cache-busted URL of the already-composed runtime atlas. */
  url: string;
}

export interface TexturePatchUpdate {
  textureUnitId: string;
  bounds: { x: number; y: number; width: number; height: number };
  imageUrl: string;
  maskUrl: string;
}

export interface TextureAtlasPatchSet {
  atlasIndex: number;
  atlasFile: string;
  sourceWidth: number;
  sourceHeight: number;
  /** Immutable base-atlas URL. It must never resolve through an appearance runtime file. */
  baseUrl: string;
  /** Complete active non-base material set for this atlas. */
  patches: TexturePatchUpdate[];
}

export interface TextureAtlasDelta {
  modelId: string;
  /** Revisions are ordered only within one stream. */
  streamId: string;
  revision: number;
  patchSets: TextureAtlasPatchSet[];
  atlases: TextureAtlasUpdate[];
}

export interface TextureMaterialChangeResult {
  library: TextureMaterialLibrary;
  delta: TextureAtlasDelta;
}

export interface CharacterTexturePreviewChange {
  job: CharacterGenerationJob;
  delta: TextureAtlasDelta;
}

/** The coherent read model used by every model-appearance screen. */
export interface ModelAppearanceSnapshot {
  modelId: string;
  model: ModelPackage;
  capability: ModelEditCapabilityManifest;
  materials: TextureMaterialLibrary;
}

export interface ModelAlignmentGuide {
  url: string;
  appearanceKey: string;
}

export interface ModelImportIssue {
  code: string;
  message: string;
  file: string | null;
}

export interface ModelImportReport {
  status: "valid" | "warning" | "invalid";
  checks: ModelImportIssue[];
  warnings: ModelImportIssue[];
  errors: ModelImportIssue[];
}

export interface ModelImportResult {
  model: ModelPackage | null;
  report: ModelImportReport;
}

export interface ModelDeleteResult {
  deletedId: string;
  selectedModel: ModelPackage;
}

export interface PetControlEvent {
  version: number;
  sequence: number;
  type: string;
  generation: string | null;
  timestampMs: number;
  payload: Record<string, unknown>;
}

export interface PetSnapshotEvent extends PetControlEvent {
  type: "pet.snapshot";
  payload: {
    events: PetControlEvent[];
    publisherConnected: boolean;
  };
}

export interface PetAudioFrame {
  generation: string;
  samplePosition: number;
  pcm: ArrayBuffer;
}

export type DesktopMainView = "chat" | "character" | "looks" | "models" | "settings" | "plugins" | "onboarding";

export interface DesktopMainWindowState {
  view: DesktopMainView;
  chatUrl: string | null;
}

/** Whether a model reads images, from a small probe image sent when unknown. */
export type ModelImageInput = "supported" | "unsupported" | "unknown";
export interface ModelImageInputStatus { chat: ModelImageInput; reasoning: ModelImageInput }

/**
 * TA's panel inside the stage window (CSS pixels from its top-left), with
 * the model's offset inside the panel.
 */
export interface PetPanelPlacement {
  x: number;
  y: number;
  width: number;
  height: number;
  offset: { x: number; y: number };
}

export type DesktopUiLocale = "zh-CN" | "en";

export interface DesktopPetApi {
  /** Provided by the isolated preload so permission guidance matches the OS. */
  readonly platform?: string;
  /** Interface language only; does not alter character prompts or conversation content. */
  getUiLocale(): Promise<DesktopUiLocale>;
  setUiLocale(locale: DesktopUiLocale): Promise<void>;
  onUiLocale(callback: (locale: DesktopUiLocale) => void): () => void;
  chat: import("./chat").ChatApi;
  facialTest: import("./facial-test").FacialTestApi;
  speechFacial: import("./speech-facial").SpeechFacialApi;
  reportChatGeometry(geometry: import("./pet-chat").PetChatGeometry): void;
  pluginCall(action: string, input?: Record<string, unknown>): Promise<any>;
  pickPlugin(development: boolean): Promise<any>;
  emotionMatching: EmotionMatchingApi;
  partRecognition: PartRecognitionApi;
  listModels(): Promise<ModelPackage[]>;
  getModelRenderBundle(id: string): Promise<ModelRenderBundle>;
  getSelectedModelRenderBundle(): Promise<ModelRenderBundle | null>;
  /** "any": one picker that takes a model folder or its .zip. */
  importModel(kind: "zip" | "folder" | "any"): Promise<ModelImportResult | null>;
  deleteModelAsset(
    id: string,
    kind: "motion" | "expression",
    assetId: string,
  ): Promise<ModelPackage>;
  /** Writes the given parameter values as a new model toggle (an .exp3.json). */
  createModelToggle(id: string, name: string, parameterValues: Record<string, number>): Promise<ModelToggleCreation>;
  saveModelPreview(id: string, png: ArrayBuffer): Promise<ModelPackage>;
  /** Looks whose card picture is missing or no longer matches how they look. */
  outdatedModelPreviews(): Promise<string[]>;
  deleteModel(id: string): Promise<ModelDeleteResult>;
  selectModel(id: string): Promise<ModelPackage>;
  saveManifest(id: string, manifest: PetManifest, adoptions?: EmotionAdoption[]): Promise<ModelPackage>;
  getSelectedModel(): Promise<ModelPackage | null>;
  getModelAppearance(id: string, refresh?: boolean): Promise<ModelAppearanceSnapshot>;
  getModelAlignmentGuide(id: string): Promise<ModelAlignmentGuide>;
  selectTextureMaterial(
    id: string,
    textureUnitId: string,
    materialId: string,
  ): Promise<TextureMaterialChangeResult>;
  /** Several pieces at once, as one change: a whole outfit goes on in one step. */
  selectTextureMaterials(
    id: string,
    assignments: Array<{ textureUnitId: string; materialId: string }>,
  ): Promise<TextureMaterialChangeResult>;
  deleteTextureMaterial(id: string, materialId: string): Promise<TextureMaterialChangeResult>;
  /**
   * The preview with these pieces recolored to `hex` (#RRGGBB); null goes back
   * to what the model wears. Resolves null when a newer request replaced it.
   */
  previewTextureRecolor(id: string, textureUnitIds: string[], hex: string | null): Promise<TextureAtlasDelta | null>;
  /** Saves the recolor as a 换色 version; `apply: false` keeps the current look. */
  saveTextureRecolor(id: string, textureUnitIds: string[], hex: string, apply?: boolean): Promise<TextureMaterialChangeResult>;
  listMasterTemplates(): Promise<MasterTemplateSummary[]>;
  importMasterTemplate(): Promise<MasterTemplateSummary | null>;
  pickCharacterReference(): Promise<CharacterReferenceSelection | null>;
  listCharacterJobs(): Promise<CharacterGenerationJob[]>;
  planCharacterTextureBatch(
    templateId: string,
    textureUnitIds: string[],
    imageModel: TextureEditImageModel,
  ): Promise<TextureEditBatchPlanResult>;
  optimizeCharacterTexturePrompt(
    input: TexturePromptOptimizationRequest,
  ): Promise<TexturePromptOptimizationResult>;
  createCharacterJob(input: CreateCharacterGenerationRequest): Promise<CharacterGenerationJob>;
  selectCharacterCandidate(jobId: string, candidateId: string): Promise<CharacterGenerationJob>;
  pauseCharacterJob(jobId: string): Promise<CharacterGenerationJob>;
  resumeCharacterJob(jobId: string): Promise<CharacterGenerationJob>;
  cancelCharacterJob(jobId: string): Promise<CharacterGenerationJob>;
  retryCharacterJob(jobId: string): Promise<CharacterGenerationJob>;
  regenerateCharacterCandidates(jobId: string): Promise<CharacterGenerationJob>;
  approveCharacterBudget(jobId: string, limitUsd: number): Promise<CharacterGenerationJob>;
  completeCharacterModel(jobId: string, kinds: CharacterCompletionKind[]): Promise<CharacterGenerationJob>;
  regenerateCharacterPart(jobId: string, slotId: string): Promise<CharacterGenerationJob>;
  acceptCharacterTexture(
    jobId: string,
    textureUnitId: string,
    accepted: boolean,
  ): Promise<CharacterTexturePreviewChange>;
  setCharacterTexturePreview(jobId: string, enabled: boolean): Promise<TextureAtlasDelta>;
  /** `apply: false` adds the accepted versions to 换色 without wearing them. */
  saveAcceptedCharacterTexturesAndCancel(jobId: string, apply?: boolean): Promise<CharacterGenerationJob>;
  confirmCharacterModel(jobId: string): Promise<CharacterGenerationJob>;
  onCharacterProgress(callback: (progress: CharacterGenerationProgress) => void): () => void;
  setMousePassthrough(value: boolean): void;
  /** Sets the system cursor over the pet; the page's CSS cursor alone is not shown. */
  setPetCursor(cursor: "default" | "grabbing" | "pointer"): void;
  requestPointerRefresh(): boolean;
  getInteractionLocked(): Promise<boolean>;
  setInteractionUnlockHover(value: boolean): void;
  unlockInteraction(): void;
  getPetModelScale(): Promise<number>;
  getPetModelOffset(): Promise<{ x: number; y: number }>;
  /** TA's panel inside the stage window that covers the display (null before it exists). */
  getPetPanel(): Promise<PetPanelPlacement | null>;
  onPetPanel(callback: (panel: PetPanelPlacement) => void): () => void;
  getPetTextureMaximum(): Promise<2048 | 4096 | 8192>;
  /**
   * TA scaled by the page itself mid-gesture: its scale, its panel (stage
   * CSS pixels) and the model's offset. The app follows along (HUD, saving)
   * without sending the panel back.
   */
  previewPetModelScale(value: number, panel: PetPanelPlacement): void;
  setPetModelScale(value: number): Promise<number>;
  startWindowDrag(
    screenX: number, screenY: number,
    modelBounds: { x: number; y: number; width: number; height: number } | null,
  ): void;
  moveWindowDrag(screenX: number, screenY: number): void;
  endWindowDrag(): void;
  showContextMenu(): void;
  openChat(): void;
  /** Opens 设置, at `section` when given. */
  openSettings(section?: SettingsSection): void;
  /** The section 设置 was last asked to open at, handed over once. */
  takeSettingsSection(): Promise<SettingsSection | null>;
  onSettingsSection(callback: () => void): () => void;
  showMainView(view: DesktopMainView): void;
  /** 说话时显示气泡, a desktop display preference kept outside the configuration. */
  getSpeechBubbles(): Promise<boolean>;
  setSpeechBubbles(value: boolean): Promise<void>;
  /** The character the pet is currently playing, once the memory service is running. */
  getCurrentCharacter(): Promise<{ name: string } | null>;
  characters: CharactersApi;
  /** Whether speech models are still downloading or being prepared. */
  getModelReadiness(): Promise<ModelReadiness>;
  /** Whether 对话大脑 and 推理大脑 read images; null while TA is still starting. */
  getModelImageInput(): Promise<ModelImageInputStatus | null>;
  onModelReadiness(callback: (value: ModelReadiness) => void): () => void;
  retryModelPreparation(): Promise<void>;
  getMainWindowState(): Promise<DesktopMainWindowState>;
  onMainWindowState(callback: (state: DesktopMainWindowState) => void): () => void;
  getMainWindowVisible(): Promise<boolean>;
  onMainWindowVisibility(callback: (visible: boolean) => void): () => void;
  onPetWindowVisibility(callback: (visible: boolean) => void): () => void;
  getConfiguration(): Promise<DesktopConfigurationView>;
  saveConfiguration(value: DesktopConfigurationInput): Promise<DesktopConfigurationView>;
  listAudioDevices(): Promise<AudioDevice[]>;
  getOwnerVoiceStatus(): Promise<OwnerVoiceStatus>;
  getOwnerVoiceEnrollmentState(): Promise<OwnerVoiceEnrollmentState>;
  /** `append` keeps the voiceprints recorded before; otherwise they are replaced. */
  enrollOwnerVoice(append?: boolean): Promise<OwnerVoiceStatus>;
  /** Removes one voiceprint (by its time), or all of them. */
  deleteOwnerVoice(createdAt?: string): Promise<OwnerVoiceStatus>;
  /** The speech a voiceprint came from, as a playable data URL. */
  getOwnerVoiceClip(clip: string): Promise<string>;
  cancelOwnerVoiceEnrollment(): Promise<void>;
  onOwnerVoiceEnrollmentState(callback: (state: OwnerVoiceEnrollmentState) => void): () => void;
  setVoiceEnabled(value: boolean): Promise<void>;
  /** Pauses the voice companion so a page can use the microphone; returnMicrophone resumes it. */
  borrowMicrophone(): Promise<void>;
  returnMicrophone(): Promise<void>;
  setMicrophoneEnabled(value: boolean): Promise<void>;
  setPlaybackEnabled(value: boolean): Promise<void>;
  getRuntimeStatus(): Promise<DesktopRuntimeStatus>;
  listLlmLogs(filter?: import("./llm-logs").LlmLogFilter): Promise<import("./llm-logs").LlmLogPage>;
  /** Tokens by purpose: today, the last seven days, the last hour. */
  getUsage(): Promise<import("./usage").UsageSummary | null>;
  getLlmLog(id: string): Promise<import("./llm-logs").LlmLogDetail>;
  getLlmLogTypes(): Promise<{ types: string[] }>;
  getCodexNotificationStatus(): Promise<{ enabled: boolean; error: string | null; pending?: number }>;
  getGatewayStatus(): Promise<GatewayStatus>;
  resetGatewayKey(): Promise<GatewayStatus>;
  /** Sends one message through the gateway itself, as another program would. */
  testGateway(): Promise<{ ok: boolean; message: string }>;
  getPerceptionStatus(): Promise<DesktopPerceptionStatus>;
  openPerceptionPermissionSettings(
    kind: DesktopPerceptionPermissionKind,
  ): Promise<void>;
  onControl(callback: (event: PetControlEvent) => void): () => void;
  onAudio(callback: (frame: PetAudioFrame) => void): () => void;
  onConnection(callback: (connected: boolean) => void): () => void;
  onCommand(callback: (command: DesktopPetCommand) => void): () => void;
  onRuntimeStatus(callback: (status: DesktopRuntimeStatus) => void): () => void;
  onPerceptionStatus(callback: (status: DesktopPerceptionStatus) => void): () => void;
  onPerceptionConfiguration(callback: (configuration: DesktopPerceptionConfiguration) => void): () => void;
  onboarding: OnboardingApi;
}

/** One local model's preparation state, written by the runtime processes. */
export interface ModelStatusItem {
  component: string;
  label: string;
  state: "preparing" | "downloading" | "ready" | "failed";
  done?: number;
  total?: number;
  error?: string;
}

/** What the pet and settings show while speech models are prepared. */
export interface CharacterSummary {
  id: string;
  name: string;
  role: string;
  current: boolean;
  lastActiveAt: string | null;
}

export interface CharacterProfileInput {
  name: string;
  role: string;
  setting: string;
  background: string;
  wakeWords?: string[];
}

export interface CharacterDetails extends CharacterProfileInput {
  id: string;
  wakeWords: string[];
  /** False while the character still answers to the app-wide wake word. */
  wakeWordsCustomized: boolean;
  /** The full character text the models read. */
  effectiveSoul: string;
  createdAt: string | null;
}

export interface CharacterVoiceOption { model: string; label: string; voices: Array<{ id: string; label: string }> }

/** One voice service a character can speak with; `ready` is false for 通义 without its key. */
export interface CharacterVoiceService { ready: boolean; options: CharacterVoiceOption[] }

export interface CharacterVoice {
  /** Which service this character speaks with now: "dashscope" (通义) or "local" (本机). */
  provider: string;
  model: string;
  voice: string;
  kind: string;
  /** The voices of the service in use. */
  options: CharacterVoiceOption[];
  cloud: CharacterVoiceService;
  local: CharacterVoiceService;
}

export interface MemoryItem {
  id: number;
  text: string;
  lastSupportedAt: string;
  factCount: number;
}

export interface CharacterMemory {
  /** Long-term impressions (the static profile). */
  impressions: MemoryItem[];
  /** What was talked about lately (the dynamic snapshot). */
  recent: MemoryItem[];
  entities: Array<{ id: string; aliases: string[]; factCount: number }>;
  createdAt: string | null;
}

export interface EntityFact {
  id: string;
  text: string;
  mentionedAt: string;
}

export type ForgetTarget = { itemId: number } | { factId: string };
export interface ForgetPlan {
  factCount: number;
  itemCount: number;
}

export interface CharactersApi {
  list(): Promise<CharacterSummary[]>;
  /** Makes this character the one on the desktop. */
  bind(id: string): Promise<void>;
  create(input: CharacterProfileInput): Promise<string>;
  profile(id: string): Promise<CharacterDetails>;
  saveProfile(id: string, input: CharacterProfileInput): Promise<void>;
  voice(id: string): Promise<CharacterVoice>;
  saveVoice(id: string, voice: { model: string; voice: string; kind: string }): Promise<void>;
  /** A short WAV sample of the voice. */
  previewVoice(voice: { model: string; voice: string }, sentence: string): Promise<Uint8Array>;
  /** Voices cloned from recordings (cloud speech only). */
  customVoices(): Promise<Array<{ id: string; model: string; ready: boolean }>>;
  /** Clones a voice from a 16-bit PCM WAV recording; returns the voice id. */
  cloneVoice(recording: Uint8Array, targetModel: string): Promise<string>;
  memory(id: string): Promise<CharacterMemory>;
  entityFacts(id: string, entityId: string): Promise<EntityFact[]>;
  forgetPreview(id: string, target: ForgetTarget): Promise<ForgetPlan>;
  forget(id: string, target: ForgetTarget): Promise<ForgetPlan>;
}

export interface ModelReadiness {
  state: "ready" | "preparing" | "failed";
  text: string;
  detail: string;
  percent?: number | null;
  items: ModelStatusItem[];
}

export interface CharacterProfile {
  /** Display name, also used as the wake word. */
  name: string;
  /** 身份与关系 */
  role: string;
  /** 性格与表达 */
  setting: string;
}

/** A character created in onboarding, waiting for the memory API to start. */
export interface PendingCharacter extends CharacterProfile {
  /** The 形象 this character wears. */
  modelId: string | null;
}

export interface AiServiceCheckRequest {
  baseUrl: string;
  model: string;
  apiKey: string;
}

export type AiServiceCheckResult = { ok: true } | { ok: false; message: string };

export interface OnboardingPreparation {
  phase: "preparing" | "ready" | "failed";
  /** 0–100 while preparing; null when the installer gives no count. */
  percent: number | null;
  detail: string;
}

/** Answers kept across a relaunch mid-onboarding (e.g. macOS "quit and reopen"). */
export interface OnboardingDraft {
  step: "look" | "soul" | "brain" | "companion" | "permissions";
  modelId: string | null;
  name: string;
  relation: string;
  relations: string[];
  traits: string[];
  traitOptions: string[];
  customSetting: string;
  provider: "qwen" | "doubao" | "custom";
  baseUrl: string;
  model: string;
  apiKey: string;
  level: "text" | "voice" | "full";
  jevApiKey: string;
  /** 百炼业务空间 ID for the decision model when the AI service is 千问. */
  bailianWorkspaceId?: string;
}

export type SystemPermissionKind = "microphone" | "screen" | "systemAudio";
export type SystemPermissionState = "granted" | "denied" | "not-determined" | "not-required";

export interface OnboardingApi {
  /** Downloads runtime assets and installs the default model in the background. */
  prepare(): Promise<OnboardingPreparation>;
  onPreparation(callback: (value: OnboardingPreparation) => void): () => void;
  checkAiService(request: AiServiceCheckRequest): Promise<AiServiceCheckResult>;
  /** Confirms an optional Jev key, which makes 主动陪伴 judge better. */
  checkJev(apiKey: string): Promise<AiServiceCheckResult>;
  /** Confirms 百炼's decision model works for this 业务空间 ID, with `apiKey` or the saved Bailian key. */
  checkBailianDecision(workspaceId: string, apiKey?: string): Promise<AiServiceCheckResult>;
  /** Queues a new character; it is created and made current once the API runs. */
  saveCharacter(character: PendingCharacter): Promise<void>;
  permissionStatus(): Promise<Record<SystemPermissionKind, SystemPermissionState>>;
  loadDraft(): Promise<OnboardingDraft | null>;
  saveDraft(draft: OnboardingDraft): Promise<void>;
  /** The name this app has in System Settings' privacy lists ("tulpa", or "Electron" in development). */
  settingsAppName(): Promise<string>;
  /** Shows the system prompt when possible, otherwise opens System Settings. */
  requestPermission(kind: SystemPermissionKind): Promise<SystemPermissionState>;
  /** Captures the screen once so macOS asks its follow-up question now; null when unavailable. */
  warmUpCapture(): Promise<{ screen: boolean; systemAudio: string } | null>;
}

export type DesktopPetCommand =
  | { type: "facial-bindings-changed"; modelId: string }
  | { type: "open-settings" }
  | { type: "model-selected"; bundle: ModelRenderBundle }
  | { type: "interaction-lock-changed"; locked: boolean }
  | { type: "pet-model-scale-changed"; scale: number }
  | { type: "pet-model-offset-changed"; offset: { x: number; y: number } }
  | { type: "pet-texture-maximum-changed"; maximum: 2048 | 4096 | 8192 }
  | { type: "pet-pointer-position"; x: number; y: number }
  | { type: "texture-materials-updated"; change: TextureMaterialChangeResult }
  | {
      type: "texture-compaction-warning";
      modelId: string;
      revision: number;
      message: string;
    };

export type DesktopTtsProvider = "auto" | "dashscope" | "local";
export type DesktopAudioBackend = "sounddevice" | "livekit";
export type DesktopPerceptionPermissionKind = "accessibility" | "screen";

export interface DesktopPerceptionConfiguration {
  operationPerceptionEnabled: boolean;
  screenPerceptionEnabled: boolean;
  audioPerceptionEnabled: boolean;
  excludeFishMemoryFromScreenCapture: boolean;
  /**
   * 内心: audiovisual summaries and background thoughts, on one switch. Off
   * keeps the senses working in the moment without digesting them later.
   * Missing means on.
   */
  innerLifeEnabled?: boolean;
}

export type DesktopPerceptionPhase =
  | "stopped"
  | "starting"
  | "ready"
  | "restarting"
  | "degraded";

export type DesktopPerceptionCapabilityState =
  | "disabled"
  | "running"
  | "unavailable"
  | "failed";

export type DesktopPerceptionPermission =
  | "not_required"
  | "granted"
  | "denied"
  | "not_determined"
  | "restricted"
  | "unknown";

export interface DesktopPerceptionCapabilityReport {
  supported: boolean;
  requested: boolean;
  state: DesktopPerceptionCapabilityState;
  permission: DesktopPerceptionPermission;
  reason?: string;
  computerHistoryInput?: DesktopComputerHistoryInputReport;
}

export type DesktopComputerHistoryEventTapState =
  | "disabled"
  | "permission_required"
  | "running"
  | "suspended"
  | "failed";

export interface DesktopComputerHistoryInputChannelReport {
  state: DesktopPerceptionCapabilityState;
  receivedCount: number;
  lastEventAt?: number;
}

export interface DesktopComputerHistoryInputReport {
  accessibilityPermission: DesktopPerceptionPermission;
  listenEventPermission: DesktopPerceptionPermission;
  eventTapState: DesktopComputerHistoryEventTapState;
  mouse: DesktopComputerHistoryInputChannelReport;
  keyboard: DesktopComputerHistoryInputChannelReport;
  tapDisabledCount: number;
}

export interface DesktopPerceptionCapabilities {
  workspace: DesktopPerceptionCapabilityReport;
  screen: DesktopPerceptionCapabilityReport;
  accessibility: DesktopPerceptionCapabilityReport;
  camera: DesktopPerceptionCapabilityReport;
  computerHistory: DesktopPerceptionCapabilityReport;
  systemAudio?: DesktopPerceptionCapabilityReport;
}

export interface DesktopPerceptionStatus {
  audioSupported?: boolean;
  /** How this macOS names the permission lists in System Settings. */
  systemSettings?: SystemSettingsNames;
  phase: DesktopPerceptionPhase;
  detail: string;
  crashCount: number;
  helperPid: number | null;
  modules: {
    workspace: boolean;
    screen: boolean;
    accessibility: boolean;
    camera: boolean;
    enhancedVision?: boolean;
    computerHistory?: boolean;
    systemAudio?: boolean;
  };
  capabilities: DesktopPerceptionCapabilities | null;
  permissions?: {
    accessibility: DesktopPerceptionPermission;
    screen: DesktopPerceptionPermission;
  };
}

export interface DesktopTextModel {
  id: string;
  name: string;
  baseUrl: string;
  model: string;
}

export interface DesktopVoiceModel {
  id: string;
  name: string;
  provider: DesktopTtsProvider;
}

export interface DesktopModelCatalog {
  text: DesktopTextModel[];
  voice: DesktopVoiceModel[];
}

export interface DesktopModelAssignments {
  chat: string;
  reasoning: string;
  speech: string | null;
}

export interface DesktopConfiguration {
  version: 2;
  configured: boolean;
  /** Local request/response logs, including images; rolling days from the call's start. */
  llmLogRetentionDays?: number;
  voiceEnabled: boolean;
  microphoneEnabled?: boolean;
  playbackEnabled?: boolean;
  speechFacialWaitForReady?: boolean;
  /** 主动陪伴. Judged by the decision model when one is connected, otherwise by the reasoning model. */
  proactiveCompanionEnabled: boolean;
  /** The decision model (百炼 or Jev) and what it judges. */
  experimentalCompanion?: {
    /** Unset: Bailian when it can work, else Jev when its key is set; see decisionProvider. */
    decisionProvider?: "bailian" | "jev";
    /** 百炼业务空间 ID, part of the decision model's address. */
    bailianWorkspaceId?: string;
    /** Jev's model. */
    model: string;
    wakeSeconds: number;
    thoughtsEnabled?: boolean;
    thoughtSeconds?: number;
    /** 聊天时挑怎么回. Unset: on exactly when a decision model is connected. */
    chatPickEnabled?: boolean;
    /** 说话时的表情和动作 (needs a decision model). */
    speechFacialEnabled?: boolean;
  };
  codexNotificationsEnabled?: boolean;
  /** 联网查资料: TA may look things up before answering, where the reasoning model can search. */
  webSearchEnabled?: boolean;
  /** Lets other programs send TA messages over HTTP, with the gateway key. */
  gateway?: GatewayConfiguration;
  launchAtLogin: boolean;
  models: DesktopModelCatalog;
  modelAssignments: DesktopModelAssignments;
  imageEditing: {
    openaiCompatibleBaseUrl: string | null;
  };
  homeAssistant: {
    enabled: boolean;
    url: string;
  };
  perception: DesktopPerceptionConfiguration;
  rendering: {
    maxTextureDimension: 2048 | 4096 | 8192;
  };
  audio: {
    wakeWord: string;
    inputDevice: string | null;
    outputDevice: string | null;
    ownerRecognitionEnabled: boolean;
  };
  advanced: {
    audioBackend: DesktopAudioBackend;
    asrModel: "sensevoice";
    asrThreads: number;
    vadThreshold: number;
    vadMinSilenceMs: number;
    vadMinSpeechMs: number;
    smartTurnEnabled: boolean;
    smartTurnThreshold: number;
    aecDelayMode: "auto" | "fixed";
    aecDelayMs: number;
    diagnosticsEnabled: boolean;
    voiceDraftEnabled: boolean;
  };
  agent: {
    enabled: boolean;
    url: string;
    runConcurrency: number;
    runPollIntervalSeconds: number;
    httpTimeoutSeconds: number;
    sseReadTimeoutSeconds: number;
    replyContextTtlSeconds: number;
  };
}

export interface DesktopConfigurationView extends DesktopConfiguration {
  secrets: {
    jevApiKey?: boolean;
    modelApiKeys: Record<string, boolean>;
    homeAssistantToken: boolean;
    openaiCompatibleImageApiKey: boolean;
    falApiKey: boolean;
    agentApiKey: boolean;
  };
}

export interface DesktopConfigurationInput extends DesktopConfiguration {
  secrets?: {
    jevApiKey?: string | null;
    modelApiKeys?: Record<string, string | null>;
    homeAssistantToken?: string;
    openaiCompatibleImageApiKey?: string;
    falApiKey?: string;
    agentApiKey?: string | null;
  };
}

export interface AudioDevice {
  id: string;
  name: string;
  inputs: number;
  outputs: number;
}

export interface OwnerVoiceStatus {
  enrolled: boolean;
  modelReady: boolean;
  modelId: string;
  createdAt: string | null;
  /** How many voiceprints are kept; each recording in new conditions adds one. */
  enrollmentCount?: number;
  /** Each voiceprint: recorded on purpose, or added from sentences TA missed. */
  enrollments?: Array<{ createdAt: string; source: "recorded" | "added"; clip: string | null }>;
  detail: string;
  audioBackend?: DesktopAudioBackend | null;
}

export interface OwnerVoiceEnrollmentState {
  phase: "idle" | "preparing" | "recording" | "processing" | "restoring" | "failed";
  status: OwnerVoiceStatus | null;
  detail: string;
  durationSeconds?: number;
  /** While the voiceprint model downloads: bytes so far and in all. */
  download?: { received: number; total: number };
}

export type DesktopRuntimePhase =
  | "stopped"
  | "starting-api"
  | "starting-companion"
  | "ready"
  | "restarting"
  | "failed";

export interface DesktopRuntimeStatus {
  phase: DesktopRuntimePhase;
  detail: string;
  apiUrl: string | null;
  voiceEnabled: boolean;
  microphoneEnabled?: boolean;
  playbackEnabled?: boolean;
  /** The companion's audio keeps failing while it retries; shown instead of "starting". */
  voiceIssue?: { kind: "microphone" | "audio"; text: string } | null;
}
import type {
  CharacterCompletionKind,
  CharacterGenerationJob,
  CharacterGenerationProgress,
  CharacterReferenceSelection,
  CreateCharacterGenerationRequest,
  MasterTemplateSummary,
  TextureEditBatchPlanResult,
  TextureEditImageModel,
  TexturePromptOptimizationRequest,
  TexturePromptOptimizationResult,
} from "./character-generation";
import type {
  LipSyncAdapterMode,
  LipSyncMappingSource,
  ModelParameterDescriptor,
  VowelParameterMap,
} from "./lipsync-adapter";

export interface GatewayConfiguration {
  enabled: boolean;
  port: number;
}

export interface GatewayStatus {
  enabled: boolean;
  listening: boolean;
  port: number;
  /** Where other programs send to: this Mac first, then its network addresses. */
  addresses: string[];
  key: string;
  /** Why it is not listening, in plain words. */
  error: string | null;
}
