import { textModel } from "./model-configuration";
import { PluginBridge } from "./plugins/bridge";
import { configureLlmLogs, listLlmLogs, llmLogDetail, llmLogTypes, llmUsage } from "./llm-logs";
import { access, readdir, readFile, rm, stat } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { configureProcessJobs, type NativeProcessJobs } from "./platform/process-tree";
import { platformPerceptionExecutable } from "./platform/runtime-paths";

import {
  app,
  BrowserWindow,
  desktopCapturer,
  dialog,
  ipcMain,
  Menu,
  nativeImage,
  net,
  powerMonitor,
  protocol,
  screen,
  session,
  shell,
  systemPreferences,
  Tray,
} from "electron";
import type { MenuItemConstructorOptions } from "electron";

import type {
  CharacterProfileInput,
  DesktopConfigurationInput,
  DesktopConfigurationView,
  DesktopMainView,
  DesktopMainWindowState,
  DesktopPerceptionConfiguration,
  DesktopPerceptionPermission,
  DesktopPerceptionStatus,
  DesktopRuntimeStatus,
  ForgetTarget,
  ModelPackage,
  ModelRenderBundle,
  OnboardingPreparation,
  ModelImageInput,
  ModelImageInputStatus,
  ModelReadiness,
  PetManifest,
  SystemPermissionKind,
  SystemPermissionState,
  TextureAtlasDelta,
  TextureEditUnit,
} from "../src/shared/contracts";
import { applySenseAction, parseSenseAction, sensesSnapshot, type SenseAction, type SenseProblem, type SensesSnapshot } from "../src/shared/pet-senses";
import { petStateEvent, SleepWatch } from "../src/shared/sleep-watch";
import { chatAttachmentUpload } from "./chat-attachment-upload";
import { DownloadClock, ModelReadinessPoller, clearModelStatus, readModelStatus, summarizeReadiness } from "./model-readiness";
import type {
  CharacterGenerationJob,
  CreateCharacterGenerationRequest,
  TextureEditImageModel,
  TexturePromptOptimizationRequest,
  TextureReferenceSelection,
  TextureReferenceSnapshot,
} from "../src/shared/character-generation";
import {
  textureEditPlanRegions,
  textureReferenceLabels,
} from "../src/shared/character-generation";
import {
  normalizePetModelScale, petWindowScale,
} from "../src/shared/pet-view";
import { validRect, type Rect } from "../src/shared/pet-chat";
import { AppConfigStore, DEFAULT_GATEWAY_PORT } from "./app-config";
import { registerUiLocaleIpc, UiPreferencesStore } from "./ui-preferences";
import { checkAiService, checkDecisionService, checkJevService } from "./ai-service-check";
import { BAILIAN_DECISION_MODEL, BAILIAN_WORKSPACE_PATTERN, bailianDecisionEndpoint, decisionConnection, JEV_DEFAULT_MODEL } from "../src/shared/decision-model";
import { presetPreviewFile } from "./preset-previews";
import { OnboardingDraftStore } from "./onboarding-draft";
import { CharacterAppearanceStore, PendingCharacterStore, announceBirth, validatePendingCharacter } from "./character-profile";
import { CharacterApi } from "./character-api";
import { ChatHistoryApi } from "./chat-api";
import { PART_RECOGNITION_CHANNEL, PartRecognizer } from "./part-recognition";
import { mergeEmotionSuggestions, suggestionsForEmptyTargets } from "../src/shared/emotion-fill";
import { currentSystemPermission, requestSystemPermission, settingsAppName, systemPermissionKind, type SystemPermissionDeps } from "./system-permissions";
import { SystemAudioPermission } from "./system-audio-permission";
import { probePermission, warmUpCapture } from "./permission-probe";
import { assertOwnerVoiceSender } from "./owner-voice-ipc";
import { OWNER_VOICE_CLIP } from "./owner-voice-manager";
import { GatewayServer } from "./gateway-server";
import { CharacterArtifactStore } from "./character-generation/artifact-store";
import { ComputerHistoryRecorder } from "./computer-history-recorder";
import { deactivatePerception } from "./perception-deactivation";
import { EnvironmentFrameCoordinator } from "./environment-frame-coordinator";
import { AdaptiveEnvironmentFrameSampling } from "./environment-frame-sampling";
import { EnvironmentHistoryRecorder } from "./environment-history-recorder";
import { perceptionModules, perceptionPolicy } from "./perception-policy";
import { darwinMajor, supportsSystemAudio } from "./system-audio-support";
import { systemSettingsNames } from "../src/shared/system-settings-names";
import { createConfiguredImageProvider } from "./character-generation/image-provider-factory";
import { CharacterGenerationJobStore } from "./character-generation/job-store";
import {
  composeAnnotatedModelReferencePng,
  composeRegionSheetPngs,
  composePieceSheetPngs,
  composePieceLocatorPng,
  pairPieceTile,
  composeModelLocatorPng,
} from "./character-generation/model-locator";
import { TexturePromptOptimizer } from "./character-generation/prompt-optimizer";
import { createEmotionMatching } from "./emotion-matching/integration";
import type { FacialTestService } from "./facial-test-service";
import { createFacialTest } from "./facial-test-integration";
import { ModelMutationQueue } from "./emotion-matching/model-mutations";
import type { EmotionMatchingService } from "./emotion-matching/service";
import type { EmotionAdoption, EmotionMatchingJob } from "../src/shared/emotion-matching";
import { GeneratedModelBuilder } from "./character-generation/model-builder";
import { CharacterGenerationService } from "./character-generation/service";
import {
  authorizeDesktopRequest,
  type DesktopRequestAuthorization,
} from "./desktop-request-auth";
import { BUNDLED_LIVE2D_DIR, Live2dAssetInstaller } from "./live2d-assets";
import { LatestAtomicFileWriter } from "./latest-atomic-file-writer";
import {
} from "../src/shared/interaction-lock";
import {
  configureMainProcess,
  shouldThrottleRenderer,
} from "./main-options";
import { resolveMainWindowState } from "./main-window-state";
import { PetWindowLifecycle } from "./pet-window-lifecycle";
import {
  modelDialogOptions,
  referenceImageDialogOptions,
} from "./model-dialogs";
import { ModelStore } from "./model-store";
import { ModelEditCapabilityStore } from "./model-edit-capabilities";
import {
  TextureMaterialStore,
  type TextureMaterialStoreChangeResult,
} from "./texture-material-store";
import { requestedTextureDimension, TextureLodCache } from "./texture-lod-cache";
import { textureUpdateRecipients } from "./texture-update-delivery";
import { buildTexturePreviewSnapshot } from "./character-generation/texture-preview-snapshot";
import { writeRecoloredPatches } from "./texture-recolor";
import { decodePng } from "./png-reader";
import { recolorName } from "../src/shared/recolor-palette";
import { snapshotTexturePreviewAssets } from "./character-generation/texture-preview-assets";
import { AtlasCompactionCoordinator } from "./atlas-compaction-coordinator";
import { UtilityProcessAtlasCompactionRunner } from "./atlas-compaction-utility-runner";
import { pruneBoundedFileCache } from "./bounded-file-cache";
import { PetSocket } from "./pet-socket";
import { PetPanelFocus } from "./pet-panel-focus";
import { loadPetApplicationActivation } from "./pet-application-activation";
import { PetPointerMonitor } from "./pet-pointer-monitor";
import { PetChatWindow, type QuickAppAction } from "./pet-chat-window";
import { trayMenu, type TrayAction, type TrayItem } from "./tray-menu";
import { appMenuTemplate } from "./app-menu";
import { PetChatService } from "./pet-chat-service";
import { perceptionPermissionSettingsUrl } from "./perception-permission-settings";
import { PerceptionManager } from "./perception-manager";
import {
  RuntimeManager,
  bundledModelsDirectory,
  type CompanionSpeechActivityEvent,
  modelStatusDirectory,
} from "./runtime-manager";
import {
  runtimeConfigurationAction,
  type RuntimeConfigurationAction,
} from "./runtime-config-impact";
import { PET_WINDOW_BASE, petScaleShownBy, petWindowSize, placePet, resizePetBounds } from "./window-drag";
import { PetStage } from "./pet-stage";
import type { SettingsSection } from "../src/settings-next/model";
import { failureReason, serviceFailure, settingsSectionFor } from "../src/shared/service-failures";
import { activeBackground, parseUsageRows, summarizeUsage, tokenText, type UsageSummary } from "../src/shared/usage";

protocol.registerSchemesAsPrivileged([
  {
    scheme: "pet-model",
    privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
  },
  {
    scheme: "pet-assets",
    privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
  },
  {
    scheme: "pet-generation",
    privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
  },
]);
configureMainProcess(app.commandLine, process.platform);
app.commandLine.appendSwitch("autoplay-policy", "no-user-gesture-required");

interface WindowState {
  x?: number;
  y?: number;
  width: number;
  height: number;
}

interface PetWindowState extends WindowState {
  version?: 2;
  modelScale?: number;
  modelOffset?: { x: number; y: number };
}

const DEFAULT_PET_BOUNDS: WindowState = { ...PET_WINDOW_BASE };
const DEFAULT_MAIN_BOUNDS: WindowState = { width: 1180, height: 820 };
// Development only: run against a separate profile, e.g. to replay first-run onboarding.
if (!app.isPackaged && process.env.FISH_MEMORY_USER_DATA_DIR) {
  app.setPath("userData", path.resolve(process.env.FISH_MEMORY_USER_DATA_DIR));
}
const singleInstance = app.requestSingleInstanceLock();
if (!singleInstance) app.quit();

let petWindow: BrowserWindow | null = null;
/** TA's panel inside petWindow, which covers the whole display. */
let petStage: PetStage | null = null;
let petPanelFocus: PetPanelFocus | null = null;
let petNativeInput: ReturnType<typeof loadPetApplicationActivation> | null = null;
let petPointerMonitor: PetPointerMonitor | null = null;
let petChatWindow: PetChatWindow | null = null;
/** 「聊两句」 on the desktop and the main window's chat send through this one service. */
let petChatService: PetChatService | null = null;
/** The pet's latest state event, for a chat page that opens after it was sent. */
let lastPetState: unknown = null;
/** TA dozing off looks like TA idling; this notices when to say how to wake TA. */
const sleepWatch = new SleepWatch();
let petAsleep = false;
let wakeWordCache: { fishId: string; word: string } | null = null;
let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let socket: PetSocket | null = null;
let modelStore: ModelStore;
let configStore: AppConfigStore;
let pluginBridge: PluginBridge;
let assets: Live2dAssetInstaller;
let runtime: RuntimeManager;
let pendingCharacters: PendingCharacterStore;
let characterAwaitingBirth: string | null = null;
let characterAppearances: CharacterAppearanceStore;
let characterApi: CharacterApi;
let backendTarget: () => Promise<{ apiUrl: string; apiKey: string } | null> = async () => null;
let chatApi: ChatHistoryApi;
let onboardingDrafts: OnboardingDraftStore;
let onboardingPreparation: Promise<OnboardingPreparation> | null = null;
let perception: PerceptionManager;
let computerHistoryRecorder: ComputerHistoryRecorder | null = null;
let environmentFrameCoordinator: EnvironmentFrameCoordinator | null = null;
let environmentHistoryRecorder: EnvironmentHistoryRecorder | null = null;
let environmentFrameSampling: AdaptiveEnvironmentFrameSampling | null = null;
let petBoundsPath: string;
let mainBoundsPath: string;
let legacyChatBoundsPath: string;
let locked = false;
let quitting = false;
let configured = false;
let perceptionConfiguration: DesktopPerceptionConfiguration | null = null;
/** The senses as last published, so the menu-bar menu shows them without waiting. */
let latestSenses: SensesSnapshot | null = null;
let perceptionPausedBySystem = false;
let perceptionSyncGeneration = 0;
let initializationComplete = false;
let requestedMainView: DesktopMainView = "chat";
const saveBoundsTimers: Partial<Record<"pet" | "main", NodeJS.Timeout>> = {};
let petStateWriter: LatestAtomicFileWriter;
let mainStateWriter: LatestAtomicFileWriter;
let uiPreferences: UiPreferencesStore;
let petDragSession: {
  /** Panel position the model would have with zero offset. */
  start: { x: number; y: number };
  startPointer: { x: number; y: number };
  /** Model rectangle in panel coordinates without offset. */
  model: Rect | null;
} | null = null;
let petModelScale = 1;
let petModelOffset = { x: 0, y: 0 };
let petModelBounds: Rect | null = null;
let modelIdForOffset: string | null = null;
let petScalePreviewActive = false;
let petScaleCommitTimer: NodeJS.Timeout | null = null;
let availableModels: ModelPackage[] = [];
let selectedModelId: string | null = null;
let connectedApiUrl: string | null = null;
let desktopRequestAuthorization: DesktopRequestAuthorization | null = null;
let characterArtifacts: CharacterArtifactStore;
let characterService: CharacterGenerationService;
let emotionMatching: EmotionMatchingService;
let facialTest: FacialTestService | null = null;
const modelMutations = new ModelMutationQueue();
let modelEditCapabilities: ModelEditCapabilityStore;
let textureMaterialStore: TextureMaterialStore;
let textureLodCache: TextureLodCache;
const texturePreviewRevisions = new Map<string, number>();
const petWindowLifecycle = new PetWindowLifecycle({
  current: () => petWindow,
  create: createPetWindow,
  quitting: () => quitting,
  visibilityChanged: () => tray?.setContextMenu(desktopMenu()),
});

app.on("browser-window-created", (_event, window) => {
  if (process.platform === "win32") {
    window.setContentProtection(perceptionConfiguration?.excludeFishMemoryFromScreenCapture ?? true);
  }
});

app.on("second-instance", () => {
  if (initializationComplete && configured) {
    runSafely(petWindowLifecycle.show(), "显示桌宠");
  }
  runSafely(openMainWindow(configured ? "chat" : "onboarding"), "打开主窗口");
});

void app.whenReady().then(async () => {
  if (process.platform === "win32") {
    app.setAppUserModelId("app.tulpa.desktop");
    const native = loadPetApplicationActivation();
    if (!native.initializeHostJob?.()) throw new Error("无法初始化 Windows 后台进程管理");
    configureProcessJobs(native as unknown as NativeProcessJobs);
  }
  const userData = app.getPath("userData");
  uiPreferences = new UiPreferencesStore(userData);
  await uiPreferences.load();
  refreshApplicationMenu();
  assets = new Live2dAssetInstaller(userData, path.join(bundledModelsDirectory({
    packaged: app.isPackaged, resourcesPath: process.resourcesPath, projectRoot: projectRoot(),
  }), BUNDLED_LIVE2D_DIR));
  modelStore = new ModelStore(userData, assets.corePath);
  configStore = new AppConfigStore(userData, (configuration) =>
    runtime.validateOwnerVoiceConfiguration(configuration),
  );
  pluginBridge = new PluginBridge(path.join(userData, "plugin-config"), async () => {
    const status = runtime.currentStatus();
    if (!status.apiUrl) throw new Error("请先在设置中启动本地服务");
    const secrets = await configStore.runtimeSecrets();
    return { apiUrl: status.apiUrl, apiKey: secrets.internalApiKey };
  });
  configureLlmLogs(async () => {
    const status = runtime.currentStatus();
    if (!status.apiUrl) throw new Error("本地服务尚未就绪");
    return { apiUrl: status.apiUrl, apiKey: (await configStore.runtimeSecrets()).internalApiKey };
  });
  const compactionRunner = new UtilityProcessAtlasCompactionRunner(
    path.join(__dirname, "atlas-compaction-worker.js"),
  );
  textureMaterialStore = new TextureMaterialStore(
    modelStore,
    new AtlasCompactionCoordinator({
      compact: async (request) => {
        await compactionRunner.compact(request);
        await Promise.all(request.atlases.map((atlas) => rm(
          path.join(request.allowedRoots[0]!, ".appearance", "runtime", atlas.atlasFile),
          { force: true },
        )));
        const root = path.join(request.allowedRoots[0]!, ".appearance", "compiled");
        await pruneBoundedFileCache({
          root,
          maximumBytes: 1024 * 1024 * 1024,
          protectedKeys: new Set(request.atlases.map((atlas) => atlas.contentKey)),
          preserveNewest: 32,
        });
      },
    }, {
      onWarning: (request, error) => sendTextureCompactionWarning(
        request.modelId,
        request.revision,
        error,
      ),
    }),
  );
  textureLodCache = new TextureLodCache(path.join(userData, "texture-lod-cache"));
  modelEditCapabilities = new ModelEditCapabilityStore(
    modelStore,
    assets.corePath,
    (modelId, baseFile) => textureMaterialStore.resolveCompiledAtlas(modelId, baseFile),
    (modelId, sourceHash) => textureMaterialStore.activeAppearanceKey(modelId, sourceHash),
  );
  characterArtifacts = new CharacterArtifactStore(
    path.join(userData, "character-generation", "artifacts"),
  );
  const characterJobs = new CharacterGenerationJobStore(
    path.join(userData, "character-generation", "jobs"),
  );
  characterService = new CharacterGenerationService({
    jobs: characterJobs,
    artifacts: characterArtifacts,
    resolveCurrentTextureAtlas: async (modelId, atlasFile) => {
      const baseFile = modelStore.resolveModelFile(modelId, atlasFile);
      return await textureMaterialStore.resolveCompiledAtlas(modelId, baseFile) ?? baseFile;
    },
    templates: {
      load: (id) => {
        if (!id.startsWith("model-edit-")) {
          throw new Error("AI 创建角色功能已移除，请从现有模型进入纹理编辑");
        }
        return modelEditCapabilities.load(id);
      },
    },
    builder: new GeneratedModelBuilder(
      path.join(userData, "character-generation", "outputs"),
      modelStore,
      (modelId, atlasFile) => textureMaterialStore.resolveCompiledAtlas(
        modelId,
        modelStore.resolveModelFile(modelId, atlasFile),
      ),
    ),
    textureMaterials: {
      save: async (input) => modelMutations.run(input.modelId, async () => {
        const capability = await modelEditCapabilities.current(input.modelId);
        await textureMaterialStore.list(input.modelId, capability);
        const change = await textureMaterialStore.saveAndApply({
          ...input,
          sourceHash: capability.sourceHash,
        });
        sendTextureMaterialsUpdated(change);
      }),
    },
    texturePreview: {
      snapshot: async (input) => {
        const revision = (texturePreviewRevisions.get(input.jobId) ?? 0) + 1;
        texturePreviewRevisions.set(input.jobId, revision);
        const official = (await textureMaterialStore.appearanceSnapshot(
          input.modelId,
          input.affectedAtlases.map((atlas) => atlas.atlasFile),
        )).delta;
        const acceptedEdits = input.enabled
          ? await Promise.all(input.edits.map(async (edit) => {
              // Renderer recipes survive WebGL context restoration. Freeze
              // both resources under revision-unique paths so replaying an old
              // recipe can never read bytes overwritten by a later job result.
              const frozen = await snapshotTexturePreviewAssets(characterArtifacts, {
                jobId: input.jobId,
                textureUnitId: edit.textureUnitId,
                revision,
                imagePath: edit.editedPatchPath,
                maskPath: edit.maskPath,
              });
              return {
                textureUnitId: edit.textureUnitId,
                atlasIndex: edit.atlasIndex,
                atlasFile: edit.atlasFile,
                bounds: edit.bounds,
                imageUrl: characterArtifactUrl(input.jobId, frozen.imagePath, revision),
                maskUrl: characterArtifactUrl(input.jobId, frozen.maskPath, revision),
              };
            }))
          : [];
        return buildTexturePreviewSnapshot({
          official,
          jobId: input.jobId,
          revision,
          enabled: input.enabled,
          affectedAtlases: input.affectedAtlases,
          acceptedEdits,
        });
      },
    },
    providerFactory: async () => {
      const [configuration, secrets] = await Promise.all([
        configStore.load(),
        configStore.runtimeSecrets(),
      ]);
      return createConfiguredImageProvider(configuration, secrets);
    },
    deleteGeneratedModel: async (id) => {
      await modelStore.delete(id);
      await refreshTrayModels();
    },
  });
  characterService.on("progress", (progress) => {
    mainWindow?.webContents.send("character:progress", progress);
  });
  petBoundsPath = path.join(userData, "pet-window-state.json");
  mainBoundsPath = path.join(userData, "main-window-state.json");
  petStateWriter = new LatestAtomicFileWriter(petBoundsPath);
  mainStateWriter = new LatestAtomicFileWriter(mainBoundsPath);
  legacyChatBoundsPath = path.join(userData, "chat-window-state.json");
  const petView = await loadPetViewState();
  petModelScale = petView.scale;
  petModelOffset = petView.offset;
  pendingCharacters = new PendingCharacterStore(userData);
  characterAppearances = new CharacterAppearanceStore(userData);
  const apiTarget = async () => {
    const status = runtime?.currentStatus();
    if (!status?.apiUrl) return null;
    return { apiUrl: status.apiUrl, apiKey: (await configStore.runtimeSecrets()).internalApiKey };
  };
  backendTarget = apiTarget;
  characterApi = new CharacterApi(apiTarget);
  chatApi = new ChatHistoryApi(apiTarget);
  onboardingDrafts = new OnboardingDraftStore(userData);
  await clearModelStatus(modelStatusDirectory(userData));
  startModelReadinessWatch(userData);
  runtime = new RuntimeManager({
    configStore,
    userDataPath: userData,
    projectRoot: projectRoot(),
    resourcesPath: process.resourcesPath,
    packaged: app.isPackaged,
    authenticateBrowser,
    pendingCharacter: pendingCharacters,
    onCharacterCreated: async (characterId, character) => {
      await characterAppearances.bind(characterId, character.modelId);
      // TA says hello once it is fully awake (see handleRuntimeStatus).
      characterAwaitingBirth = characterId;
    },
  });
  runtime.on("status", handleRuntimeStatus);
  runtime.ownerVoice.on("state", (state) => {
    mainWindow?.webContents.send("desktop:owner-voice:state", state);
  });
  runtime.on("log", (entry) => console.log(`[${entry.role}] ${entry.line}`));
  perception = new PerceptionManager({
    helperPath: perceptionHelperExecutable(),
    helperArguments: process.platform === "win32" ? ["--host-pid", String(process.pid)] : [],
  });
  environmentFrameSampling = new AdaptiveEnvironmentFrameSampling({
    apply: (intervalMs, forceConfigure) => {
      perception.setEnvironmentFrameInterval(intervalMs, forceConfigure);
    },
  });
  runtime.on(
    "companion-speech-activity",
    (event: CompanionSpeechActivityEvent) => {
      environmentFrameSampling?.handleSpeechActivity(event);
    },
  );
  computerHistoryRecorder = new ComputerHistoryRecorder({
    deliveryMode: "context",
    requestTimeoutMs: 10_000,
    rootPath: path.join(userData, "data", "computer-history", "segments"),
    connection: async () => {
      const status = runtime.currentStatus();
      if (!status.apiUrl || !["starting-companion", "ready"].includes(status.phase)) {
        return null;
      }
      const secrets = await configStore.runtimeSecrets();
      return { apiUrl: status.apiUrl, apiKey: secrets.internalApiKey };
    },
    onErrorCode: (code) => console.warn(`Computer history recorder deferred (${code})`),
  });
  environmentHistoryRecorder = new EnvironmentHistoryRecorder({
    onErrorCode: (code) => console.warn(`Environment history deferred (${code})`),
  });
  environmentFrameCoordinator = new EnvironmentFrameCoordinator({
    claim: (frameId, signal) => perception.claimEnvironmentFrame(
      frameId,
      undefined,
      signal,
    ),
    connection: async () => {
      const status = runtime.currentStatus();
      if (!status.apiUrl || !["starting-companion", "ready"].includes(status.phase)) {
        return null;
      }
      const secrets = await configStore.runtimeSecrets();
      return { apiUrl: status.apiUrl, apiKey: secrets.internalApiKey };
    },
    onErrorCode: (code) => console.warn(`Environment frame deferred (${code})`),
    onTarget: (target) => environmentHistoryRecorder?.setTarget(target),
    onTargetPending: () => environmentHistoryRecorder?.suspendTarget(),
    onFrame: (frame) => environmentHistoryRecorder?.handleFrame(frame),
  });
  perception.on("status", (status) => {
    runtime.recordPerceptionStatus(status);
    sendPerceptionStatus(status);
    if (
      status.helperPid === null
      || !["ready", "degraded"].includes(status.phase)
      || status.capabilities?.screen.state !== "running"
    ) {
      void environmentFrameCoordinator?.clearFrame();
    }
  });
  // Typed private capture events feed the bounded semantic-history pipeline.
  // Raw media never enters generic observations, messages, or event logs.
  perception.on("system-audio-window", (value) => {
    environmentHistoryRecorder?.handleAudioWindow(value);
  });
  perception.on("interaction-event", (value) => {
    void computerHistoryRecorder?.record(value);
  });
  perception.on("interaction-events-suppressed", (value) => {
    void computerHistoryRecorder?.recordSuppressed(value.count, value.timestamp);
  });
  perception.on("environment-frame-available", (value) => {
    void environmentFrameCoordinator?.handleFrame(value);
  });
  perception.on("log", (entry) => console.log(`[${entry.role}] ${entry.line}`));
  perception.on("protocol-error", (error) => {
    console.warn("Perception helper protocol error:", error);
  });
  perception.on("helper-error", (error) => {
    console.warn("Perception helper reported an error:", error);
  });
  powerMonitor.on("lock-screen", () => pausePerceptionForSystem("锁屏"));
  powerMonitor.on("suspend", () => pausePerceptionForSystem("系统睡眠"));
  powerMonitor.on("unlock-screen", () => resumePerceptionAfterSystem("解锁"));
  powerMonitor.on("resume", () => resumePerceptionAfterSystem("系统唤醒"));

  await modelStore.initialize();
  facialTest = createFacialTest({
    userData, projectRoot: projectRoot(), resourcesPath: process.resourcesPath, packaged: app.isPackaged,
    models: modelStore, window: () => mainWindow, corePath: assets.corePath,
    petWindow: () => petWindow,
    speechConnection: async () => {
      const status = runtime.currentStatus();
      if (!status.apiUrl) throw new Error("本地服务尚未就绪");
      return { apiUrl: status.apiUrl, apiKey: (await configStore.runtimeSecrets()).internalApiKey };
    },
  });
  registerProtocols();
  registerIpc();
  emotionMatching = createEmotionMatching({
    userData, models: modelStore, config: configStore,
    bundle: async (id) => modelRenderBundle(await modelStore.load(id)),
    window: () => mainWindow,
    onUpdate: (job) => runSafely(autoFillImportedModel(job), "AI 填写表情动作"),
  });
  registerDesktopRequestAuthorization();
  createTray();
  for (const job of await characterService.listResumable()) {
    if (!job.paused) void runCharacterJob(job);
  }

  const configuration = await configStore.load();
  configured = configuration.configured;
  gateway = new GatewayServer({
    keyFile: path.join(userData, "gateway-key"),
    forward: forwardGatewayEvent,
  });
  runSafely(gateway.configure(gatewayConfiguration(configuration)), "打开外部消息入口");
  await syncPerceptionConfiguration(configuration.perception);
  if (configuration.launchAtLogin) {
    app.setLoginItemSettings({ openAtLogin: true });
  }
  if (!configuration.configured) {
    await openMainWindow("onboarding");
    return;
  }
  if (!(await assets.ready())) {
    await assets.install((detail) => {
      console.log(`[assets] ${detail}`);
    });
  }
  await installDefaultModelIfNeeded();
  await petWindowLifecycle.show();
  void runtime.start().catch((error) => showRuntimeFailure(error));
}).catch((error) => reportUnexpectedError("初始化 App", error)).finally(() => {
  initializationComplete = true;
});

app.on("window-all-closed", () => {
  // The tray remains the owner of the background desktop companion.
});
app.on("activate", () => {
  if (!initializationComplete) return;
  if (configured) runSafely(petWindowLifecycle.show(), "显示桌宠");
  else runSafely(openMainWindow(configured ? "chat" : "onboarding"), "打开主窗口");
});
app.on("before-quit", (event) => {
  if (quitting) return;
  stopModelReadinessWatch();
  if (!runtime || !perception) {
    quitting = true;
    return;
  }
  event.preventDefault();
  quitting = true;
  environmentHistoryRecorder?.stop();
  environmentFrameSampling?.dispose();
  perceptionSyncGeneration += 1;
  socket?.stop();
  void gateway?.close();
  for (const timer of Object.values(saveBoundsTimers)) if (timer) clearTimeout(timer);
  // Capture bounds before destroying renderers, then close their API
  // WebSockets before asking Uvicorn to stop. Keeping BrowserWindows alive
  // here made Uvicorn exhaust its five-second grace period and left the
  // separately managed pg0 process orphaned after every app quit.
  const savePetState = saveWindowState("pet").then(() => petStateWriter?.flush());
  const saveMainState = saveWindowState("main").then(() => mainStateWriter?.flush());
  const closePetChat = petChatWindow?.dispose();
  petChatWindow = null;
  if (petWindow && !petWindow.isDestroyed()) petWindow.destroy();
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.destroy();
  void Promise.all([
    facialTest?.dispose(),
    emotionMatching?.shutdown(),
    deactivatePerception({
      stopPerception: (detail) => perception.stop(detail),
      disableComputerHistory: () => computerHistoryRecorder?.stop(),
      disableEnvironmentFrames: () => environmentFrameCoordinator?.setEnabled(false),
    }, "感知服务已停止"),
    runtime.stop().catch((error) => console.error("Unable to stop Fish Memory runtime:", error)),
    savePetState,
    saveMainState,
    closePetChat,
    textureMaterialStore.flushCompaction(5_000).then((flushed) => {
      if (!flushed) console.warn("Live2D 图集后台压实未能在退出期限内完成");
    }),
  ])
    .finally(() => app.quit());
});

function registerProtocols(): void {
  protocol.handle("pet-model", async (request) => {
    const url = new URL(request.url);
    const relative = decodePath(url.pathname);
    if (/^\.edit-capabilities\/texture-units\/[^/]+-(?:original-)?preview\.png$/.test(relative)) {
      await modelEditCapabilities.ensureTexturePreview(url.hostname, relative);
    }
    const base = modelStore.resolveModelFile(url.hostname, relative);
    const file = url.searchParams.get("textureBase") === "1"
      ? base
      : await textureMaterialStore.resolveCompiledAtlas(url.hostname, base) ?? base;
    const maximum = requestedTextureDimension(url.searchParams.get("textureMaxDimension"));
    const resolved = maximum ? await textureLodCache.resolve(file, maximum) : file;
    return localFile(resolved);
  });
  protocol.handle("pet-assets", async (request) => {
    const url = new URL(request.url);
    const relative = [url.hostname, decodePath(url.pathname)].filter(Boolean).join("/");
    return localFile(assets.resolve(relative));
  });
  protocol.handle("pet-generation", async (request) => {
    const url = new URL(request.url);
    const file = characterArtifacts.resolve(url.hostname, decodePath(url.pathname));
    return localFile(file);
  });
}

/**
 * A local file for the app's own pages. They load from file://, so Chromium
 * treats reading these schemes as cross-origin and needs the CORS header.
 */
async function localFile(file: string): Promise<Response> {
  const response = await net.fetch(pathToFileURL(file).toString());
  const headers = new Headers(response.headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function registerIpc(): void {
  registerUiLocaleIpc({
    handle: (channel, handler) => ipcMain.handle(channel, handler),
    store: uiPreferences,
    mainContents: () => mainWindow?.webContents,
    allContents: () => BrowserWindow.getAllWindows().map((window) => window.webContents),
    refreshMenus: () => {
      refreshApplicationMenu();
      tray?.setContextMenu(desktopMenu());
      petChatWindow?.refreshLocale();
      runSafely(publishSenses(""), "更新界面语言");
      runSafely(refreshModelReadiness(), "更新模型准备状态");
      sendPerceptionStatus();
    },
  });
  ipcMain.handle("desktop:llm-logs:list", (event, filter) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return listLlmLogs(filter);
  });
  ipcMain.handle("desktop:usage", async (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return currentUsage(0);
  });
  ipcMain.handle("desktop:llm-logs:types", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return llmLogTypes();
  });
  ipcMain.handle("desktop:llm-logs:detail", (event, id: string) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return llmLogDetail(id);
  });
  ipcMain.handle("desktop:plugins:call", (event, action: string, input: Record<string, unknown>) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return pluginBridge.call(action, input);
  });
  ipcMain.handle("desktop:plugins:pick", async (event, development: boolean) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    const result = await dialog.showOpenDialog(mainWindow!, {
      title: development
        ? uiText("选择包含 fish-plugin.json 的插件目录", "Choose a plugin folder containing fish-plugin.json")
        : uiText("导入插件包", "Import plugin package"),
      properties: development ? ["openDirectory"] : ["openFile"],
      filters: development ? undefined : [{ name: uiText("Fish 插件", "Fish plugin"), extensions: ["fish-plugin"] }],
    });
    if (result.canceled) return null;
    return pluginBridge.call("load", { path: result.filePaths[0], development });
  });
  ipcMain.handle("pet:model:list", () => modelStore.list());
  ipcMain.handle("pet:model:selected", () => modelStore.selected());
  ipcMain.handle("pet:model:render-bundle", async (_event, id: string) =>
    modelRenderBundle(await modelStore.load(id)));
  ipcMain.handle("pet:model:selected-render-bundle", async () => {
    const model = await modelStore.selected();
    return model ? modelRenderBundle(model) : null;
  });
  ipcMain.handle("pet:model:appearance:get", (_event, id: string, refresh = false) =>
    loadModelAppearance(id, refresh));
  ipcMain.handle("pet:model:alignment-guide:get", async (_event, id: string) => {
    const guide = await modelEditCapabilities.ensureAlignmentGuide(id);
    return {
      appearanceKey: guide.appearanceKey,
      url: `pet-model://${id}/.edit-capabilities/alignment-guide.png?appearance=${guide.appearanceKey}`,
    };
  });
  ipcMain.handle("pet:model:texture-materials:select", async (
    event,
    id: string,
    textureUnitId: string,
    materialId: string,
  ) => {
    const change = await modelMutations.run(id, () => textureMaterialStore.select(id, textureUnitId, materialId));
    sendTextureMaterialsUpdated(change, event.sender.id);
    return change;
  });
  ipcMain.handle("pet:model:texture-materials:select-many", async (
    event,
    id: string,
    assignments: Array<{ textureUnitId: string; materialId: string }>,
  ) => {
    const change = await modelMutations.run(id, () => textureMaterialStore.selectMany(id, assignments));
    sendTextureMaterialsUpdated(change, event.sender.id);
    return change;
  });
  ipcMain.handle("pet:model:texture-recolor:preview", async (
    _event,
    id: string,
    textureUnitIds: string[],
    hex: string | null,
  ) => recolorPreview(id, textureUnitIds, hex));
  ipcMain.handle("pet:model:texture-recolor:save", async (
    event,
    id: string,
    textureUnitIds: string[],
    hex: string,
    apply?: boolean,
  ) => {
    const units = await recolorUnits(id, textureUnitIds);
    const directory = characterArtifacts.jobDirectory(recolorJobId(id));
    const patches = await writeRecoloredPatches({
      units,
      hex,
      directory: path.join(directory, "save"),
      name: "save",
      currentImage: (unit) => resolveTextureMaterialPreview(id, unit.id),
      maskFile: (unit) => modelStore.resolveModelFile(id, unit.maskPath),
    });
    if (!patches) throw new Error("没有可换色的纹理");
    const change = await modelMutations.run(id, async () => {
      const capability = await modelEditCapabilities.current(id);
      await textureMaterialStore.list(id, capability);
      return textureMaterialStore.saveAndApply({
        modelId: id,
        name: `换色 ${recolorName(hex)}`,
        prompt: hex.toUpperCase(),
        sourceHash: capability.sourceHash,
        // The library keeps the model's own UV masks, never copies.
        edits: patches.map((patch, index) => ({ ...patch, maskPath: modelStore.resolveModelFile(id, units[index]!.maskPath) })),
        apply: apply !== false,
        origin: "recolor",
      });
    });
    sendTextureMaterialsUpdated(change, event.sender.id);
    // The saved versions live in the model now; previews are done with.
    await rm(directory, { recursive: true, force: true });
    return change;
  });
  ipcMain.handle("pet:model:texture-materials:delete", async (event, id: string, materialId: string) => {
    const change = await modelMutations.run(id, () => textureMaterialStore.delete(id, materialId));
    sendTextureMaterialsUpdated(change, event.sender.id);
    return change;
  });
  ipcMain.handle("pet:model:select", async (_event, id: string) => {
    const model = await modelStore.select(id);
    selectedModelId = model.id;
    runSafely(bindAppearanceToCurrentCharacter(model.id), "记录角色形象");
    await sendSelectedModel(model);
    await refreshTrayModels();
    return model;
  });
  ipcMain.handle(
    "pet:model:save-manifest",
    async (_event, id: string, manifest: PetManifest, adoptions: EmotionAdoption[] = []) => {
      const model = await modelMutations.run(id, async () => {
        if (adoptions.length) await emotionMatching.validate(id, manifest, adoptions);
        return modelStore.saveManifest(id, manifest);
      });
      if (adoptions.length) await emotionMatching.adopted(id, adoptions, model.manifest)
        .catch(() => console.warn("模型配置已保存，但分析记录的采纳标记未能更新"));
      if (id === selectedModelId) await sendSelectedModel(model);
      await refreshTrayModels();
      return model;
    },
  );
  ipcMain.handle("pet:model:import", async (_event, kind: "zip" | "folder" | "any") => {
    const result = await showOpenDialog(modelDialogOptions(kind, uiPreferences.uiLocale));
    const selected = result.filePaths[0];
    if (result.canceled || !selected) return null;
    const folder = kind === "any" ? (await stat(selected)).isDirectory() : kind === "folder";
    const imported = folder
      ? await modelStore.importFolder(selected)
      : await modelStore.importZip(selected);
    if (imported.model) {
      selectedModelId = imported.model.id;
      await sendSelectedModel(imported.model);
      runSafely(bindAppearanceToCurrentCharacter(imported.model.id), "记录角色形象");
      // A new 形象 gets its states and emotions filled once, right away.
      modelsAwaitingAutoFill.add(imported.model.id);
      const modelId = imported.model.id;
      void emotionMatching.start(modelId).catch((error) => {
        // No picture-reading model yet: the 表情动作 tab still offers to fill them later.
        modelsAwaitingAutoFill.delete(modelId);
        console.warn("导入后没有自动填写表情动作:", error);
      });
    }
    await refreshTrayModels();
    return imported;
  });
  ipcMain.handle(
    "pet:model:asset:delete",
    async (_event, id: string, kind: "motion" | "expression", assetId: string) => {
      const model = await modelMutations.run(id, () => modelStore.deleteAsset(id, kind, assetId));
      emotionMatching?.sourceChanged(id);
      if (id === selectedModelId) await sendSelectedModel(model);
      await refreshTrayModels();
      return model;
    },
  );
  const partRecognizer = new PartRecognizer({
    capability: (modelId) => modelEditCapabilities.current(modelId),
    guide: async (modelId) => {
      await modelEditCapabilities.ensureAlignmentGuide(modelId);
      return readFile(modelStore.resolveModelFile(modelId, ".edit-capabilities/alignment-guide.png"));
    },
    readModelFile: (modelId, relative) => readFile(modelStore.resolveModelFile(modelId, relative)),
    unitPreview: async (modelId, unitId) => readFile(await resolveTextureMaterialPreview(modelId, unitId)),
    pngDataUrl: (bytes, maxLongEdge) => pngVisionDataUrl(bytes, maxLongEdge),
    route: async () => {
      const [configuration, secrets] = await Promise.all([configStore.load(), configStore.runtimeSecrets()]);
      const selected = textModel(configuration, "reasoning");
      const apiKey = secrets.modelApiKeys?.[selected.id];
      if (!apiKey) throw new Error("请先在设置里填好推理模型的 API Key");
      return { baseUrl: selected.baseUrl, model: selected.model, apiKey };
    },
    send: (update) => {
      if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(`${PART_RECOGNITION_CHANNEL}:update`, update);
    },
  });
  ipcMain.handle(`${PART_RECOGNITION_CHANNEL}:start`, (_event, modelId: string, groupNames: string[], unitIds: string[]) => {
    if (!Array.isArray(groupNames) || !Array.isArray(unitIds)) throw new Error("识图分组请求无效");
    partRecognizer.start(modelId, groupNames.filter((name) => typeof name === "string"), unitIds.filter((id) => typeof id === "string"));
  });
  ipcMain.handle(`${PART_RECOGNITION_CHANNEL}:cancel`, (_event, modelId: string) => partRecognizer.cancel(modelId));
  ipcMain.handle(
    "pet:model:toggle:create",
    async (_event, id: string, name: string, parameterValues: Record<string, number>) => {
      const created = await modelMutations.run(id, () => modelStore.createToggle(id, name, parameterValues));
      emotionMatching?.sourceChanged(id);
      if (id === selectedModelId) await sendSelectedModel(created.model);
      return created;
    },
  );
  ipcMain.handle(
    "pet:model:preview:save",
    async (_event, id: string, png: ArrayBuffer) => modelStore.savePreview(id, png, await lookKey(id)),
  );
  ipcMain.handle("pet:model:preview:outdated", async () => {
    const outdated: string[] = [];
    for (const model of await modelStore.list()) {
      try {
        if (!model.previewUrl || await modelStore.previewKey(model.id) !== await lookKey(model.id)) outdated.push(model.id);
      } catch (error) {
        console.warn(`Unable to check the card picture of ${model.id}:`, error);
      }
    }
    return outdated;
  });
  ipcMain.handle("pet:model:delete", async (_event, id: string) => {
    const wasSelected = id === selectedModelId;
    const deleted = await modelMutations.run(id, () => modelStore.delete(id));
    emotionMatching?.sourceChanged(id);
    selectedModelId = deleted.selectedModel.id;
    if (wasSelected) await sendSelectedModel(deleted.selectedModel);
    await refreshTrayModels();
    return deleted;
  });
  ipcMain.handle("character:job:list", async () =>
    (await characterService.listJobs()).filter((job) => job.input.operation === "texture-edit"),
  );
  ipcMain.handle("character:reference:pick", async () => {
    const result = await showOpenDialog(referenceImageDialogOptions(uiPreferences.uiLocale));
    const selected = result.filePaths[0];
    if (result.canceled || !selected) return null;
    return {
      path: selected,
      name: path.basename(selected),
      previewUrl: visionImageDataUrl(selected, 768),
    };
  });
  ipcMain.handle(
    "character:texture-batch:plan",
    (_event, templateId: string, textureUnitIds: string[], imageModel: TextureEditImageModel) =>
      characterService.planTextureBatch(templateId, textureUnitIds, imageModel),
  );
  ipcMain.handle(
    "character:texture-prompt:optimize",
    async (_event, input: TexturePromptOptimizationRequest) => {
      const planned = await characterService.planTextureBatch(
        input.templateId,
        input.textureUnitIds,
        input.imageModel,
      );
      if (!planned.fits) throw new Error(planned.reason);
      const planChanged = planned.hash !== input.planHash;
      const modelId = input.templateId.startsWith("model-edit-")
        ? input.templateId.slice("model-edit-".length)
        : "";
      const capability = modelId ? await modelEditCapabilities.current(modelId) : null;
      if (!capability || capability.templateId !== input.templateId) {
        throw new Error("模型纹理能力清单已失效，请重新预检");
      }
      await modelEditCapabilities.ensureAlignmentGuide(modelId);
      const resolvedReferences = await resolveTextureReferences(
        modelId,
        capability,
        input.references,
      );
      const unitById = new Map(capability.textureUnits.map((unit) => [unit.id, unit]));
      const referenceBytes = await readFile(
        modelStore.resolveModelFile(modelId, ".edit-capabilities/alignment-guide.png"),
      );
      // Every region of every canvas: one understanding covers the whole edit.
      const promptRegions = textureEditPlanRegions(planned.canvases);
      const regionAssets = await Promise.all(promptRegions.map(async (region, index) => {
        const unit = unitById.get(region.textureUnitId);
        if (!unit) throw new Error("选择中包含已失效的纹理");
        const modelMaskBytes = await readFile(modelStore.resolveModelFile(modelId, unit.modelMaskPath));
        return {
          region,
          unit,
          index,
          modelMaskBytes,
          previewBytes: await readFile(
            await resolveTextureMaterialPreview(modelId, unit.id),
          ),
        };
      }));
      const referenceImageDataUrl = pngVisionDataUrl(composeAnnotatedModelReferencePng(
        referenceBytes,
        regionAssets.map(({ region, unit, index, modelMaskBytes }) => ({
          key: region.key,
          maskBytes: modelMaskBytes,
          regionIndex: index,
          placement: unit.modelMaskBounds,
        })),
      ), 960);
      const regionSheetDataUrls = composeRegionSheetPngs(regionAssets.map(({ region, index, previewBytes }) => ({
        key: region.key,
        regionIndex: index,
        image: decodePng(previewBytes),
      }))).map((sheet) => pngVisionDataUrl(sheet, 1536));
      // For the per-piece check: each piece beside where it sits on the model.
      const guide = decodePng(referenceBytes);
      const pieceSheetDataUrls = composePieceSheetPngs(regionAssets.map(({ region, unit, index, previewBytes, modelMaskBytes }) => ({
        key: region.key,
        regionIndex: index,
        image: pairPieceTile(decodePng(previewBytes), composePieceLocatorPng(guide, modelMaskBytes, unit.modelMaskBounds)),
      }))).map((sheet) => pngVisionDataUrl(sheet, 1800));
      // The sheet and the annotated body carry the pictures; regions only need their names.
      const regions = regionAssets.map(({ region, unit }) => ({
        key: region.key,
        label: unit.label,
        canvasBounds: {
          x: region.x,
          y: region.y,
          width: region.workWidth,
          height: region.workHeight,
        },
        modelBounds: unit.modelBounds,
      }));
      const [configuration, secrets] = await Promise.all([
        configStore.load(),
        configStore.runtimeSecrets(),
      ]);
      const selectedModel = textModel(configuration, "reasoning");
      const modelApiKey = secrets.modelApiKeys?.[selectedModel.id];
      if (!modelApiKey) throw new Error("请先在设置中填写推理模型的 API Key");
      const referenceLabels = textureReferenceLabels(resolvedReferences);
      const optimization = await new TexturePromptOptimizer({
        baseUrl: selectedModel.baseUrl,
        model: selectedModel.model,
        apiKey: modelApiKey,
      }).optimize({ ...input, modelName: capability.modelName }, {
        canvas: { width: planned.plan.canvasWidth, height: planned.plan.canvasHeight },
        referenceImageDataUrl,
        regionSheetDataUrls,
        pieceSheetDataUrls,
        regions,
        references: resolvedReferences.map((reference, index) => ({
          selectionId: reference.id,
          id: reference.source === "model-preview" ? "P0" : referenceLabels[index]!,
          name: reference.name,
          source: reference.source,
          description: reference.description,
          imageDataUrl: visionImageDataUrl(reference.path, 1024),
        })),
      });
      return {
        ...optimization,
        plan: planned.plan,
        planned,
        planChanged,
        warnings: planChanged
          ? ["模型纹理清单已刷新，联合画布已自动同步；当前选择和提示词均已保留。", ...optimization.warnings]
          : optimization.warnings,
      };
    },
  );
  ipcMain.handle(
    "character:job:create",
    async (_event, input: CreateCharacterGenerationRequest) => {
      if (input.operation !== "texture-edit") {
        throw new Error("AI 创建角色和旧部位修改功能已移除，请使用纹理编辑");
      }
      const referenceImage = input.referenceImage ?? (
        input.sourceModelId
          ? modelStore.resolveModelFile(input.sourceModelId, ".preview.png")
          : null
      );
      let textureReferences: TextureReferenceSnapshot[] = [];
      if (input.operation === "texture-edit" && input.textureReferences?.length) {
        const modelId = input.templateId.startsWith("model-edit-")
          ? input.templateId.slice("model-edit-".length)
          : "";
        const capability = modelId ? await modelEditCapabilities.current(modelId) : null;
        if (!capability || capability.templateId !== input.templateId) {
          throw new Error("模型纹理能力清单已失效，请重新预检");
        }
        textureReferences = await resolveTextureReferences(
          modelId,
          capability,
          input.textureReferences,
        );
      }
      const job = await characterService.createJob({
        ...input,
        referenceImage,
        textureReferences,
      });
      void runCharacterJob(job);
      return job;
    },
  );
  ipcMain.handle("character:job:pause", (_event, jobId: string) =>
    characterService.pause(jobId));
  ipcMain.handle("character:job:cancel", async (_event, jobId: string) => {
    const job = await characterService.cancel(jobId);
    texturePreviewRevisions.delete(jobId);
    return job;
  });
  ipcMain.handle("character:job:resume", async (_event, jobId: string) => {
    const job = await characterService.resume(jobId);
    void runCharacterJob(job);
    return job;
  });
  ipcMain.handle("character:job:retry", async (_event, jobId: string) => {
    const job = await characterService.retry(jobId);
    void runCharacterJob(job);
    return job;
  });
  ipcMain.handle(
    "character:job:approve-budget",
    async (_event, jobId: string, limitUsd: number) => {
      const job = await characterService.approveBudget(jobId, limitUsd);
      void runCharacterJob(job);
      return job;
    },
  );
  ipcMain.handle(
    "character:job:texture-accept",
    (_event, jobId: string, textureUnitId: string, accepted: boolean) =>
      characterService.acceptTexture(jobId, textureUnitId, accepted),
  );
  ipcMain.handle(
    "character:job:texture-preview",
    (_event, jobId: string, enabled: boolean) =>
      characterService.texturePreview(jobId, enabled),
  );
  ipcMain.handle(
    "character:job:texture-save-and-cancel",
    async (_event, jobId: string, apply?: boolean) => {
      const job = await characterService.saveAcceptedTexturesAndCancel(jobId, apply !== false);
      texturePreviewRevisions.delete(jobId);
      return job;
    },
  );
  ipcMain.handle("character:job:confirm", async (_event, jobId: string) => {
    const job = await characterService.confirmSave(jobId);
    texturePreviewRevisions.delete(jobId);
    await refreshTrayModels();
    return job;
  });
  ipcMain.handle("desktop:config:get", () => configStore.load());
  ipcMain.handle(
    "desktop:config:save",
    async (event, value: DesktopConfigurationInput) => {
      assertOwnerVoiceSender(event, mainWindow?.webContents);
      return runtime.withConfigurationChange(async () => {
        const previous = await configStore.load();
        const previousSecrets = await configStore.runtimeSecrets();
        const saved = await configStore.save(value);
        const nextSecrets = await configStore.runtimeSecrets();
        configured = saved.configured;
        if (previous.rendering.maxTextureDimension !== saved.rendering.maxTextureDimension) {
          petWindow?.webContents.send("pet:command", {
            type: "pet-texture-maximum-changed",
            maximum: saved.rendering.maxTextureDimension,
          });
        }
        // Apply perception changes immediately after they are persisted. In
        // particular, disabling capture must not wait behind asset/model work:
        // a native sampling tick that slips through here can leave a macOS
        // capture permission sheet visible even after the helper has exited.
        if (previous.configured !== saved.configured ||
          JSON.stringify(previous.perception) !== JSON.stringify(saved.perception)) {
          await syncPerceptionConfiguration(saved.perception);
        }
        const action = runtimeConfigurationAction(previous, saved, value, {
          previous: previousSecrets, next: nextSecrets,
        });
        if (saved.launchAtLogin !== previous.launchAtLogin) {
          app.setLoginItemSettings({ openAtLogin: saved.launchAtLogin });
        }
        await gateway?.configure(gatewayConfiguration(saved));
        if (!(await assets.ready())) {
          sendRuntimeStatus({
            phase: "starting-api",
            detail: "正在准备 Live2D 运行资源…",
            apiUrl: runtime.currentStatus().apiUrl,
            voiceEnabled: saved.voiceEnabled,
          });
          await assets.install((detail) => {
            sendRuntimeStatus({
              phase: "starting-api",
              detail,
              apiUrl: runtime.currentStatus().apiUrl,
              voiceEnabled: saved.voiceEnabled,
            });
          });
        }
        await installDefaultModelIfNeeded();
        if (!petWindow || petWindow.isDestroyed()) await petWindowLifecycle.show();
        if (!previous.configured && saved.configured) {
          await onboardingDrafts.clear();
          mainWindow?.close();
        }
        await pushApiConfiguration(action, saved);
        applyRuntimeConfiguration(action);
        sendMainWindowState();
        await publishSenses();
        return configStore.load();
      });
    },
  );
  ipcMain.handle("desktop:owner-voice:status", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return runtime.ownerVoice.status();
  });
  ipcMain.handle("desktop:owner-voice:state", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return runtime.ownerVoice.currentState();
  });
  ipcMain.handle("desktop:owner-voice:enroll", (event, append: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return runtime.enrollOwnerVoice(append === true);
  });
  ipcMain.handle("desktop:owner-voice:clip", async (event, clip: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    // Only a file name the service wrote, and only from its own folder.
    if (typeof clip !== "string" || !OWNER_VOICE_CLIP.test(clip)) throw new Error("没有这段录音");
    const file = path.join(app.getPath("userData"), "data", "owner-voice", "clips", clip);
    return `data:audio/wav;base64,${(await readFile(file)).toString("base64")}`;
  });
  ipcMain.handle("desktop:owner-voice:delete", (event, createdAt: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    // Only a recording time the status listed; never a path or other argument.
    const one = typeof createdAt === "string" && /^[0-9T:.+\-]{10,64}$/.test(createdAt) ? createdAt : undefined;
    return runtime.deleteOwnerVoice(one);
  });
  ipcMain.handle("desktop:owner-voice:cancel", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return runtime.ownerVoice.cancel();
  });
  ipcMain.handle("desktop:audio-devices", () => runtime.listAudioDevices());
  ipcMain.handle("desktop:voice:set-enabled", (_event, value: boolean) =>
    runtime.setVoiceEnabled(Boolean(value)),
  );
  ipcMain.handle("desktop:microphone:borrow", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return runtime.borrowMicrophone();
  });
  ipcMain.handle("desktop:microphone:return", () => runtime.returnMicrophone());
  ipcMain.handle("desktop:microphone:set-enabled", (_event, value: boolean) => runtime.setMicrophoneEnabled(Boolean(value)));
  ipcMain.handle("desktop:playback:set-enabled", (_event, value: boolean) => runtime.setPlaybackEnabled(Boolean(value)));
  ipcMain.handle("desktop:runtime:status", () => runtime.currentStatus());
  ipcMain.handle("desktop:codex-notifications:status", () => runtime.codexNotificationStatus());
  ipcMain.handle("desktop:gateway:status", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    if (!gateway) throw new Error("外部消息入口还没准备好");
    return gateway.status();
  });
  ipcMain.handle("desktop:gateway:reset-key", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    if (!gateway) throw new Error("外部消息入口还没准备好");
    return gateway.resetKey();
  });
  ipcMain.handle("desktop:gateway:test", async (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return testGateway();
  });
  // Status reads are deliberately side-effect free. Opening or focusing the
  // settings window must never start a capture helper or request permission.
  ipcMain.handle("desktop:perception:status", () =>
    withNativePerceptionPermissions(perception.currentStatus()),
  );
  ipcMain.handle(
    "desktop:perception:permission-settings",
    async (_event, kind: unknown) => {
      const url = perceptionPermissionSettingsUrl(kind);
      if (!url) throw new Error("不支持的环境感知权限类型");
      await shell.openExternal(url);
    },
  );
  ipcMain.handle("desktop:onboarding:prepare", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return prepareOnboarding();
  });
  ipcMain.handle("desktop:onboarding:check-ai-service", (event, request: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return checkAiService(request);
  });
  ipcMain.handle("desktop:onboarding:check-jev", async (event, apiKey: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    const configuration = await configStore.load();
    return checkJevService({ apiKey, model: configuration.experimentalCompanion?.model });
  });
  ipcMain.handle("desktop:onboarding:check-bailian-decision", async (event, workspaceId: unknown, apiKey: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    const workspace = typeof workspaceId === "string" ? workspaceId.trim() : "";
    if (!BAILIAN_WORKSPACE_PATTERN.test(workspace)) return { ok: false, message: "业务空间 ID 格式不对，应形如 ws-xxxx 或 llm-xxxx" };
    const [configuration, secrets] = await Promise.all([configStore.load(), configStore.runtimeSecrets()]);
    const connection = decisionConnection({ ...configuration, experimentalCompanion: {
      ...configuration.experimentalCompanion ?? { model: JEV_DEFAULT_MODEL, wakeSeconds: 5 },
      decisionProvider: "bailian", bailianWorkspaceId: workspace } }, secrets);
    // Onboarding checks with the key just typed, before anything is saved.
    const typed = typeof apiKey === "string" ? apiKey.trim() : "";
    const checked = typed ? { endpoint: bailianDecisionEndpoint(workspace), model: BAILIAN_DECISION_MODEL, apiKey: typed } : connection;
    if (!checked) return { ok: false, message: "还没有百炼的 Key：先在 AI 服务 里添加一个百炼模型" };
    return checkDecisionService(checked, { timeout: "连接百炼超时，请检查网络", unreachable: "连不上百炼，请检查网络和业务空间 ID",
      denied: "业务空间 ID 不对，或 AI 服务 里的百炼 Key 不属于这个业务空间" });
  });
  ipcMain.handle("desktop:onboarding:save-character", async (event, value: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    await pendingCharacters.save(validatePendingCharacter(value));
  });
  ipcMain.handle("desktop:onboarding:permission-status", async (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return {
      microphone: await currentSystemPermission("microphone", systemPermissionDeps()),
      screen: await currentSystemPermission("screen", systemPermissionDeps()),
      systemAudio: await currentSystemPermission("systemAudio", systemPermissionDeps()),
    };
  });
  ipcMain.handle("desktop:onboarding:settings-app-name", () => settingsAppName(process.execPath));
  ipcMain.handle("desktop:onboarding:load-draft", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return onboardingDrafts.load();
  });
  ipcMain.handle("desktop:onboarding:save-draft", (event, draft: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return onboardingDrafts.save(draft);
  });
  // Screen only: system audio has its own permission step (SystemAudioPermission).
  ipcMain.handle("desktop:onboarding:warm-up-capture", async (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    if (process.platform !== "darwin") return null;
    return warmUpCapture(perceptionHelperExecutable(), { screen: true, audio: false });
  });
  ipcMain.handle("desktop:onboarding:request-permission", (event, kind: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return requestSystemPermission(systemPermissionKind(kind), systemPermissionDeps());
  });
  ipcMain.handle("desktop:main:state", () => currentMainWindowState());
  ipcMain.handle("desktop:main:visible", () => mainWindow?.isVisible() ?? false);
  ipcMain.handle("desktop:speech-bubbles:get", (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return petChatWindow?.bubblesEnabled ?? true;
  });
  ipcMain.handle("desktop:speech-bubbles:set", async (event, value: unknown) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    if (typeof value !== "boolean") throw new Error("气泡开关无效");
    await petChatWindow?.setBubblesEnabled(value);
  });
  ipcMain.handle("desktop:model-readiness:get", () => modelReadiness);
  // Whether 对话大脑 and 推理大脑 read images; null while TA is still starting.
  ipcMain.handle("desktop:model-image-input:get", async (): Promise<ModelImageInputStatus | null> => {
    const target = await backendTarget();
    if (!target) return null;
    try {
      const response = await fetch(`${target.apiUrl}/internal/model-image-input`, {
        headers: { Authorization: `Bearer ${target.apiKey}` },
        signal: AbortSignal.timeout(70_000),
      });
      if (!response.ok) return null;
      const body = await response.json() as Record<string, unknown>;
      const read = (value: unknown): ModelImageInput =>
        value === "supported" || value === "unsupported" ? value : "unknown";
      return { chat: read(body.chat), reasoning: read(body.reasoning) };
    } catch {
      return null;
    }
  });
  ipcMain.handle("desktop:model-readiness:retry", async (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    await retryModelPreparation();
  });
  ipcMain.handle("desktop:character:current", async (event) => {
    assertOwnerVoiceSender(event, mainWindow?.webContents);
    return currentCharacter();
  });
  const characterHandle = <A extends unknown[], R>(channel: string, handler: (...args: A) => Promise<R>) => {
    ipcMain.handle(channel, async (event, ...args) => {
      assertOwnerVoiceSender(event, mainWindow?.webContents);
      return handler(...(args as A));
    });
  };
  characterHandle("desktop:chat:page", async (before?: unknown) => {
    if (before !== undefined && !(Number.isSafeInteger(before) && (before as number) >= 0)) throw new Error("无效的翻页位置");
    return chatApi.page(before as number | undefined);
  });
  characterHandle("desktop:chat:search", async (query: unknown) => {
    if (typeof query !== "string" || query.length > 200) throw new Error("搜索内容无效");
    return chatApi.search(query);
  });
  characterHandle("desktop:chat:send", async (id: unknown, content: unknown, attachmentIds: unknown) => {
    if (typeof id !== "string" || !id || id.length > 64 || typeof content !== "string" || !content.trim() || content.length > 32_000) {
      return { ok: false, error: "无效的发送请求" };
    }
    const ids = Array.isArray(attachmentIds) && attachmentIds.every((item) => typeof item === "string") ? attachmentIds as string[] : [];
    return chatService().submit(id, content, ids);
  });
  characterHandle("desktop:chat:attach", async (upload: unknown) => {
    const valid = chatAttachmentUpload(upload);
    return valid ? chatService().attach(valid) : { ok: false, error: "附件无效" };
  });
  characterHandle("desktop:chat:attachment-image", async (id: unknown) =>
    typeof id === "string" ? chatService().attachmentImage(id) : null);
  characterHandle("desktop:chat:interrupt", async (messageId: unknown) => {
    if (typeof messageId !== "string" || !messageId || messageId.length > 64) throw new Error("无效的消息");
    await chatApi.interrupt(messageId);
  });
  characterHandle("desktop:chat:new-conversation", () => startNewConversation());
  characterHandle("desktop:chat:listen", () => chatApi.listen());
  characterHandle("desktop:chat:stop-listening", () => chatApi.stopListening());
  characterHandle("desktop:chat:pet-state", async () => lastPetState);
  characterHandle("desktop:chat:senses", () => currentSenses());
  characterHandle("desktop:chat:sense-action", async (value: unknown) => {
    const action = parseSenseAction(value);
    if (!action || action.type === "panel") throw new Error("无效的操作");
    await handleSenseAction(action);
  });
  characterHandle("desktop:characters:list", () => characterApi.list());
  characterHandle("desktop:characters:bind", async (id: string) => {
    await characterApi.bind(id);
    // The character keeps its own 形象; switching characters switches outfits.
    const modelId = await characterAppearances.modelFor(id);
    if (modelId && modelId !== selectedModelId) await selectModelFromMenu(modelId).catch(() => undefined);
    else if (selectedModelId) await characterAppearances.bind(id, selectedModelId);
    await runtime.restartCompanion();
  });
  characterHandle("desktop:characters:create", async (input: CharacterProfileInput) => {
    const id = await characterApi.create(input);
    if (selectedModelId) await characterAppearances.bind(id, selectedModelId);
    return id;
  });
  characterHandle("desktop:characters:profile", (id: string) => characterApi.profile(id));
  characterHandle("desktop:characters:save-profile", async (id: string, input: CharacterProfileInput) => {
    await characterApi.saveProfile(id, input);
    if (input.wakeWords) {
      await characterApi.saveWakeWords(id, input.wakeWords);
      wakeWordCache = null;
      // The voice process reads wake words when it (re)binds the character.
      runSafely(runtime.restartCompanion(), "应用新的唤醒词");
    }
  });
  characterHandle("desktop:characters:voice", (id: string) => characterApi.voice(id));
  characterHandle("desktop:characters:save-voice", (id: string, voice: { model: string; voice: string; kind: string }) => characterApi.saveVoice(id, voice));
  characterHandle("desktop:characters:preview-voice", (voice: { model: string; voice: string }, sentence: string) => characterApi.previewVoice(voice, sentence));
  characterHandle("desktop:characters:custom-voices", () => characterApi.customVoices());
  characterHandle("desktop:characters:clone-voice", (recording: Uint8Array, targetModel: string) => characterApi.cloneVoice(recording, targetModel));
  characterHandle("desktop:characters:memory", (id: string) => characterApi.memory(id));
  characterHandle("desktop:characters:entity-facts", (id: string, entityId: string) => characterApi.entityFacts(id, entityId));
  characterHandle("desktop:characters:forget-preview", (id: string, target: ForgetTarget) => characterApi.forgetPreview(id, target));
  characterHandle("desktop:characters:forget", (id: string, target: ForgetTarget) => characterApi.forget(id, target));
  ipcMain.on("desktop:main:show", (_event, view: unknown) => {
    if (!isMainView(view)) return;
    runSafely(openMainWindow(view), "切换主窗口页面");
  });
  ipcMain.on("desktop:chat:open", () => runSafely(openMainWindow("chat"), "打开聊天"));
  ipcMain.on("desktop:settings:open", (_event, section: unknown) =>
    runSafely(openSettings(section), "打开设置"),
  );
  ipcMain.handle("desktop:settings:section", () => {
    const section = requestedSettingsSection;
    requestedSettingsSection = null;
    return section;
  });
  ipcMain.on("pet:cursor", (event, value: unknown) => {
    if (!petWindow || event.sender !== petWindow.webContents) return;
    if (value !== "default" && value !== "grabbing" && value !== "pointer") return;
    petNativeInput?.setCursor(value);
  });
  ipcMain.on("pet:mouse-passthrough", (event, value: unknown) => {
    const window = petWindow;
    if (!window || event.sender !== window.webContents || typeof value !== "boolean") return;
    const passthrough = locked || value;
    window.setIgnoreMouseEvents(passthrough, { forward: true });
    // Hovering TA changes nothing about which app is active: taking it over
    // took focus from the app in use and could pull the user to another
    // desktop. A two-finger scroll on TA scales it whichever app is in front.
  });
  ipcMain.on("pet:chat:geometry", (event, value: unknown) => {
    if (petWindow && event.sender === petWindow.webContents) {
      petChatWindow?.reportGeometry(value);
      const bounds = (value as { bounds?: unknown } | null)?.bounds;
      if (bounds === null || validRect(bounds)) {
        // Re-place only when the model's extent in its panel changed (new
        // look, new scale). Placing again because the panel just moved fed
        // on itself: a model a little wider than its panel got the same
        // nudge every time and slid across the screen.
        const changed = JSON.stringify(bounds) !== JSON.stringify(petModelBounds);
        petModelBounds = bounds;
        // Mid-scale the page places TA itself.
        if (bounds && changed && !petDragSession && !petScalePreviewActive) {
          const window = petStage!.getBounds();
          const display = screen.getDisplayNearestPoint({
            x: window.x + window.width / 2,
            y: window.y + window.height / 2,
          }).workArea;
          const moved = placePetWindow(
            petStage!, petModelTarget(window), modelWithoutOffset(bounds), display,
          );
          if (moved) scheduleBoundsSave("pet");
        }
      }
    }
  });
  ipcMain.on("pet:pointer-refresh", (event) => {
    if (petWindow && event.sender === petWindow.webContents) petPointerMonitor?.refresh();
  });
  ipcMain.handle("pet:interaction-lock:get", () => locked);
  ipcMain.on("pet:interaction-unlock-hover", (event, value: unknown) => {
    if (
      !locked ||
      !petWindow ||
      event.sender !== petWindow.webContents ||
      typeof value !== "boolean"
    ) return;
    if (value) petWindow.setIgnoreMouseEvents(false);
    else petWindow.setIgnoreMouseEvents(true, { forward: true });
  });
  ipcMain.on("pet:interaction-unlock", (event) => {
    if (petWindow && event.sender === petWindow.webContents) {
      setInteractionLocked(false);
    }
  });
  ipcMain.handle("pet:model-scale:get", () => petModelScale);
  ipcMain.handle("pet:model-offset:get", () => ({ ...petModelOffset }));
  ipcMain.handle("pet:panel:get", () => petStage?.placement() ?? null);
  // The stage covers a display: follow its size when displays change.
  for (const event of ["display-metrics-changed", "display-added", "display-removed"] as const) {
    screen.on(event as "display-added", () => {
      const stage = petStage;
      stage?.refit((display) => {
        if (placePetWindow(stage, petModelTarget(stage.getBounds()),
          petModelBounds ? modelWithoutOffset(petModelBounds) : null, display)) {
          scheduleBoundsSave("pet");
        }
      });
    });
  }
  ipcMain.handle("pet:texture-maximum:get", async () =>
    (await configStore.load()).rendering.maxTextureDimension,
  );
  ipcMain.on(
    "pet:model-scale:preview",
    (event, value: unknown, panel: unknown) => {
      const offset = (panel as { offset?: { x?: unknown; y?: unknown } } | null)?.offset;
      if (
        !petWindow || !petStage || event.sender !== petWindow.webContents ||
        typeof value !== "number" || !Number.isFinite(value) || !validRect(panel) ||
        typeof offset?.x !== "number" || typeof offset.y !== "number" ||
        !Number.isFinite(offset.x) || !Number.isFinite(offset.y)
      ) return;
      const placement = panel;
      petScalePreviewActive = true;
      // The page scaled and placed TA itself (every frame, no round trip);
      // follow it here for the HUD and the saved state.
      const stage = petStage.getStageBounds();
      petModelScale = normalizePetModelScale(value);
      petModelOffset = { x: offset.x, y: offset.y };
      petStage.setBounds({
        x: stage.x + placement.x, y: stage.y + placement.y, width: placement.width, height: placement.height,
      }, true);
      sendPetModelScaleChanged();
      if (petScaleCommitTimer) clearTimeout(petScaleCommitTimer);
      petScaleCommitTimer = setTimeout(() => {
        petScaleCommitTimer = null;
        petScalePreviewActive = false;
        runSafely(savePetWindowState(), "保存桌宠缩放比例");
      }, 240);
    },
  );
  ipcMain.handle("pet:model-scale:set", async (_event, value: unknown) => {
    if (typeof value !== "number") throw new Error("桌宠缩放比例无效");
    if (petScaleCommitTimer) clearTimeout(petScaleCommitTimer);
    petScaleCommitTimer = null;
    petScalePreviewActive = true;
    applyPetModelScale(value);
    petScalePreviewActive = false;
    sendPetModelScaleChanged();
    await savePetWindowState();
    return petModelScale;
  });
  ipcMain.on("pet:drag:start", (event, screenX: unknown, screenY: unknown, bounds: unknown) => {
    if (
      !petWindow ||
      locked ||
      event.sender !== petWindow.webContents ||
      typeof screenX !== "number" ||
      typeof screenY !== "number" ||
      !Number.isFinite(screenX) ||
      !Number.isFinite(screenY)
    ) return;
    petWindow.setIgnoreMouseEvents(false);
    const model = validRect(bounds) ? bounds : petModelBounds;
    petDragSession = {
      start: petModelTarget(petStage!.getBounds()),
      startPointer: { x: screenX, y: screenY },
      model: model ? modelWithoutOffset(model) : null,
    };
  });
  ipcMain.on("pet:drag:move", (event, screenX: unknown, screenY: unknown) => {
    if (
      !petWindow ||
      !petDragSession ||
      event.sender !== petWindow.webContents ||
      typeof screenX !== "number" ||
      typeof screenY !== "number" ||
      !Number.isFinite(screenX) ||
      !Number.isFinite(screenY)
    ) return;
    try {
      const target = {
        x: petDragSession.start.x + screenX - petDragSession.startPointer.x,
        y: petDragSession.start.y + screenY - petDragSession.startPointer.y,
      };
      if (!Number.isFinite(target.x) || !Number.isFinite(target.y)) return;
      const display = screen.getDisplayNearestPoint({ x: screenX, y: screenY }).workArea;
      if (petStage) placePetWindow(petStage, target, petDragSession.model, display);
    } catch (error) {
      // A malformed or out-of-range pointer event must never escape an IPC
      // listener: an uncaught main-process exception also kills the managed
      // voice runtime and makes the whole app appear unresponsive.
      petDragSession = null;
      console.warn("忽略无效的桌宠拖动坐标", error);
    }
  });
  ipcMain.on("pet:drag:end", (event) => {
    if (petWindow && event.sender === petWindow.webContents) {
      const moved = petDragSession !== null;
      petDragSession = null;
      if (moved) runSafely(saveWindowState("pet"), "保存桌宠全局位置");
    }
  });
  ipcMain.on("pet:context-menu", () => {
    if (petChatWindow && !locked) {
      runSafely(petChatWindow.openQuick(screen.getCursorScreenPoint()), "打开快捷面板");
      return;
    }
    const focus = petPanelFocus;
    focus?.setMenuOpen(true);
    desktopMenu().popup({
      window: petWindow ?? undefined,
      callback: () => focus?.setMenuOpen(false),
    });
  });
}

async function loadModelAppearance(id: string, refresh = false) {
  const capability = refresh
    ? await modelEditCapabilities.scan(id)
    : await modelEditCapabilities.current(id);
  // Capability topology seeds the first normal material for every UV unit.
  // From this point on all initial and generated materials share one schema.
  const materials = await textureMaterialStore.list(id, capability);
  // Loading the package last captures the final appearance revision in its URL.
  const model = await modelStore.load(id);
  return { modelId: id, model, capability, materials };
}

function registerDesktopRequestAuthorization(): void {
  session.defaultSession.webRequest.onBeforeSendHeaders((details, callback) => {
    callback({
      requestHeaders: authorizeDesktopRequest(
        details.url,
        {
          webContentsId: details.webContentsId,
          attachedWebContentsId: details.webContents?.id,
          resourceType: details.resourceType,
          frameOrigin: details.frame?.origin,
        },
        mainWindow?.webContents.id,
        details.requestHeaders,
        desktopRequestAuthorization,
      ),
    });
  });
}

async function createPetWindow(): Promise<void> {
  if (petWindow && !petWindow.isDestroyed()) {
    petWindow.showInactive();
    return;
  }
  const saved = await loadWindowState(petBoundsPath, DEFAULT_PET_BOUNDS, true, { width: 1, height: 1 });
  // The panel size always follows the scale; the saved panel only places it.
  let bounds: WindowState = resizePetBounds({
    x: saved.x ?? 0,
    y: saved.y ?? 0,
    width: saved.width,
    height: saved.height,
  }, petWindowSize(petWindowScale(petModelScale)));
  const center = {
    x: (bounds.x ?? 0) + bounds.width / 2,
    y: (bounds.y ?? 0) + bounds.height / 2,
  };
  const placement = placePet(
    petModelTarget({ x: bounds.x ?? 0, y: bounds.y ?? 0 }), bounds, null,
    screen.getDisplayNearestPoint(center).workArea,
  );
  bounds = { ...bounds, ...placement.window };
  petModelOffset = placement.offset;
  if (quitting) return;
  const stageArea = (point: { x: number; y: number }) => screen.getDisplayNearestPoint(point).workArea;
  petWindow = new BrowserWindow({
    // The whole display: TA's panel is a box inside it (see PetStage).
    ...stageArea(center),
    // The panel accompanies other apps' native fullscreen Spaces; input
    // activation while interacting is scoped by PetPanelFocus below.
    type: process.platform === "darwin" ? "panel" : undefined,
    acceptFirstMouse: true,
    transparent: true,
    frame: false,
    hasShadow: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    show: false,
    resizable: false,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    backgroundColor: "#00000000",
    webPreferences: rendererPreferences("pet"),
  });
  // One step above plain "floating": other apps' pinned windows share that
  // level and would otherwise cover TA while its HUD stays on top.
  petWindow.setAlwaysOnTop(true, "floating", 1);
  petWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  petWindow.setIgnoreMouseEvents(false);
  const stage = new PetStage(petWindow, {
    x: bounds.x ?? 0, y: bounds.y ?? 0, width: bounds.width, height: bounds.height,
  }, stageArea, () => petModelOffset);
  petStage = stage;
  if (process.platform === "darwin" || process.platform === "win32") {
    const nativeInput = loadPetApplicationActivation();
    petNativeInput = nativeInput;
    if (process.platform === "win32" && !nativeInput.setNonActivating?.(petWindow.getNativeWindowHandle())) {
      throw new Error("无法配置桌宠的 Windows 窗口交互");
    }
    if (process.platform === "darwin") petPanelFocus = new PetPanelFocus(petWindow, nativeInput);
    // Pointer positions are measured against the panel, not the whole stage.
    const pointerTarget = {
      isDestroyed: () => stage.isDestroyed(),
      isVisible: () => stage.isVisible(),
      getBounds: () => stage.getBounds(),
      webContents: petWindow.webContents,
    };
    petPointerMonitor = new PetPointerMonitor(pointerTarget, nativeInput, () => screen.getCursorScreenPoint(), point => petChatWindow?.pointer(point));
  }
  petWindow.on("blur", () => petPanelFocus?.forgetFocus());
  stage.on("move", () => {
    if (!petScalePreviewActive) scheduleBoundsSave("pet");
  });
  stage.on("resize", () => {
    if (!petScalePreviewActive) scheduleBoundsSave("pet");
  });
  petWindow.on("show", () => {
    petPointerMonitor?.start();
    sendPetWindowVisibility(true);
  });
  petWindow.on("hide", () => {
    petPointerMonitor?.stop();
    petPanelFocus?.forgetFocus();
    sendPetWindowVisibility(false);
  });
  petWindow.on("closed", () => {
    petDragSession = null;
    petModelBounds = null;
    petPointerMonitor?.stop();
    petPointerMonitor = null;
    petPanelFocus = null;
    void petChatWindow?.dispose();
    petChatWindow = null;
    petWindow = null;
    petStage = null;
  });
  petWindowLifecycle.bind(petWindow);
  petChatWindow = new PetChatWindow({
    pet: stage,
    preferencePath: path.join(app.getPath("userData"), "pet-chat-ui.json"),
    load: loadRenderer,
    openHistory: () => runSafely(openMainWindow("chat"), "打开聊天记录"),
    menuChanged: () => tray?.setContextMenu(desktopMenu()),
    locale: () => uiPreferences.uiLocale,
    nativePointerActive: () => petPointerMonitor?.active ?? false,
    senseAction: (action) => runSafely(handleSenseAction(action), "调整感知"),
    sensesShown: () => runSafely(refreshSensePermissions(), "读取感知权限"),
    quickInfo: async () => ({
      speaker: runtime.currentStatus().playbackEnabled ?? runtime.currentStatus().voiceEnabled ?? true,
      character: (await currentCharacter().catch(() => null))?.name ?? null,
      petState: petStateName(),
      looks: availableModels.map((model) => ({ id: model.id, name: model.name })),
      lookId: selectedModelId,
    }),
    quickAction: (action) => handleQuickAction(action),
    newConversation: () => startNewConversation(),
    service: chatService(),
  });
  petChatWindow.setLocked(locked);
  await petChatWindow.initialize();
  await refreshSensePermissions(true);
  petChatWindow.syncRuntime();
  await loadRenderer(petWindow, "index.html");
  stage.publish();
  petWindow.showInactive();
  petPointerMonitor?.start();
}

/** 「新对话」 from either chat: both windows learn it has started. */
async function startNewConversation(): Promise<{ ok: boolean; error?: string }> {
  const result = await chatService().startNewConversation();
  if (!result.ok) return result;
  petChatWindow?.newConversation();
  mainWindow?.webContents.send("desktop:chat:new-conversation", result.sessionId);
  return { ok: true };
}

function chatService(): PetChatService {
  petChatService ??= new PetChatService({
    status: () => runtime.currentStatus(),
    apiKey: async () => (await configStore.runtimeSecrets()).internalApiKey,
    onText: (text, status, failure) => petChatWindow?.showText(text, status, true, failure ?? null),
  });
  return petChatService;
}

let requestedSettingsSection: SettingsSection | null = null;
const SETTINGS_SECTION_IDS = new Set<string>(["ai", "jev", "companion", "usage", "extensions", "permissions", "general", "advanced"]);

/** 设置, at the section that fixes what the user was just told about. */
async function openSettings(section?: unknown): Promise<void> {
  if (typeof section === "string" && SETTINGS_SECTION_IDS.has(section)) requestedSettingsSection = section as SettingsSection;
  await openMainWindow("settings");
  mainWindow?.webContents.send("desktop:settings:section");
}

async function openMainWindow(view: DesktopMainView): Promise<void> {
  requestedMainView = view;
  // Hovering the pet lent it tulpa's activation and remembers the app to give
  // it back to once the pointer leaves. The person asked for tulpa: keep it,
  // before creating the window lets the pointer leave first.
  petPanelFocus?.forgetFocus();
  petChatWindow?.forgetFocus();
  if (!mainWindow) await createMainWindow();
  sendMainWindowState();
  mainWindow?.show();
  // The person asked for this window: bring tulpa forward, not only the window,
  // or the first click in it only activates the app.
  if (process.platform === "darwin") {
    // Activation lands a moment later and hands focus back to the window that
    // last had it, often the pet's panel; give it to this window instead.
    const refocus = (): void => { if (mainWindow && !mainWindow.isDestroyed()) mainWindow.focus(); };
    app.once("did-become-active", refocus);
    setTimeout(() => app.removeListener("did-become-active", refocus), 1_000);
    // app.focus alone was refused after the pet's hover gave the foreground
    // back to the previous app; the pet's own activation still gets through.
    petNativeInput?.acquire();
    app.focus({ steal: true });
  }
  mainWindow?.focus();
}

async function createMainWindow(): Promise<void> {
  if (mainWindow) return;
  const bounds = await loadMainWindowState();
  mainWindow = new BrowserWindow({
    ...bounds,
    minWidth: 820,
    minHeight: 620,
    show: false,
    title: "tulpa",
    backgroundColor: "#111816",
    webPreferences: rendererPreferences("main"),
  });
  // The management window is opened occasionally. Destroying it on close
  // releases its Chromium renderer; openMainWindow recreates it on demand.
  mainWindow.on("close", () => {
    void facialTest?.stop();
    void runtime.returnMicrophone();
    if (!quitting) void runtime.ownerVoice.cancel();
  });
  mainWindow.on("show", () => sendMainWindowVisibility(true));
  mainWindow.on("hide", () => {
    void facialTest?.stop();
    void runtime.returnMicrophone();
    sendMainWindowVisibility(false);
    void runtime.ownerVoice.cancel();
  });
  mainWindow.webContents.on("render-process-gone", () => { void facialTest?.stop(); void runtime.returnMicrophone(); void runtime.ownerVoice.cancel(); });
  mainWindow.webContents.on("will-navigate", () => { void facialTest?.stop(); void runtime.returnMicrophone(); void runtime.ownerVoice.cancel(); });
  mainWindow.on("move", () => scheduleBoundsSave("main"));
  mainWindow.on("resize", () => scheduleBoundsSave("main"));
  mainWindow.on("closed", () => { mainWindow = null; });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isExternalUrl(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-frame-navigate", (details) => {
    if (details.isMainFrame || isLocalChatUrl(details.url)) return;
    details.preventDefault();
    if (isExternalUrl(details.url)) void shell.openExternal(details.url);
  });
  await loadRenderer(mainWindow, "main.html");
}

function rendererPreferences(window: "pet" | "main"): Electron.WebPreferences {
  return {
    preload: path.join(__dirname, "preload.js"),
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: false,
    backgroundThrottling: shouldThrottleRenderer(window),
  };
}

async function loadRenderer(window: BrowserWindow, file: string): Promise<void> {
  const developmentUrl = process.env.VITE_DEV_SERVER_URL;
  if (developmentUrl) {
    await window.loadURL(new URL(file, `${developmentUrl}/`).toString());
  } else {
    await window.loadFile(path.join(__dirname, "../../dist-renderer", file));
  }
}

async function authenticateBrowser(apiUrl: string, apiKey: string): Promise<void> {
  const response = await session.defaultSession.fetch(`${apiUrl}/v1/auth/unlock`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ apiKey }),
  });
  if (!response.ok) throw new Error(`App 本地认证失败 (${response.status})`);
  desktopRequestAuthorization = {
    origin: new URL(apiUrl).origin,
    bearer: `Bearer ${apiKey}`,
  };
}

/** A new character's first words, once the pet is listening for them. */
function announceBirthSoon(apiUrl: string, characterId: string): void {
  characterAwaitingBirth = null;
  setTimeout(() => {
    void configStore.runtimeSecrets()
      .then((secrets) => announceBirth(apiUrl, secrets.internalApiKey, characterId))
      .catch((error) => console.warn("新角色的第一句话没说出来:", error));
  }, 3_000);
}

function handleRuntimeStatus(status: DesktopRuntimeStatus): void {
  petChatWindow?.syncRuntime();
  runSafely(publishSenses(), "更新感知状态");
  sendRuntimeStatus(status);
  sendMainWindowState();
  tray?.setContextMenu(desktopMenu());
  if (status.phase === "ready" && status.apiUrl) {
    connectToFishMemory(status.apiUrl);
    if (characterAwaitingBirth) announceBirthSoon(status.apiUrl, characterAwaitingBirth);
  } else if (
    status.phase === "restarting" ||
    status.phase === "stopped" ||
    status.phase === "failed"
  ) {
    desktopRequestAuthorization = null;
    socket?.stop();
    socket = null;
    connectedApiUrl = null;
    if (status.phase === "failed" && !quitting) {
      runSafely(openMainWindow("settings"), "打开诊断设置");
    }
  }
  if (!quitting) {
    perceptionSyncGeneration += 1;
    runSafely(
      configStore.load().then((value) => syncPerceptionConfiguration(value.perception)),
      "同步本机感知运行状态",
    );
  }
}

/** Hands the running API what changed; restarts are left to applyRuntimeConfiguration. */
let gateway: GatewayServer | null = null;

function gatewayConfiguration(configuration: { gateway?: { enabled: boolean; port: number } }) {
  return configuration.gateway ?? { enabled: false, port: DEFAULT_GATEWAY_PORT };
}

/** Hands an event another program sent in to the local service. */
async function forwardGatewayEvent(event: unknown): Promise<{ status: number; body: unknown }> {
  const status = runtime.currentStatus();
  if (!status.apiUrl) throw new Error("本地服务尚未就绪");
  const response = await fetch(`${status.apiUrl}/internal/gateway/events`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${(await configStore.runtimeSecrets()).internalApiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(event),
    signal: AbortSignal.timeout(15_000),
  });
  return { status: response.status, body: await response.json().catch(() => null) };
}

/** One message through the gateway's own address and key, as another program sends it. */
async function testGateway(): Promise<{ ok: boolean; message: string }> {
  if (!gateway) return { ok: false, message: "外部消息入口还没准备好" };
  const current = await gateway.status();
  if (!current.listening) return { ok: false, message: current.error ?? "先打开开关" };
  try {
    const response = await fetch(current.addresses[0]!, {
      method: "POST",
      headers: { Authorization: `Bearer ${current.key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ source: "tulpa 设置", text: "这是一条测试消息：外部消息入口已经通了。" }),
      signal: AbortSignal.timeout(20_000),
    });
    if (response.status === 202) return { ok: true, message: "发过去了，TA 收到了" };
    const body = await response.json().catch(() => null) as { error?: string } | null;
    return { ok: false, message: body?.error ?? `没发出去（${response.status}）` };
  } catch {
    return { ok: false, message: "没发出去，等 TA 上线再试" };
  }
}

async function pushApiConfiguration(action: RuntimeConfigurationAction, saved: DesktopConfigurationView): Promise<void> {
  if (action === "configure-api" || action === "configure-api-and-companion" ||
    ((action === "none" || action === "restart-companion") && runtime.hasPendingApiConfiguration)) {
    await runtime.configureApi(saved);
  }
}

function applyRuntimeConfiguration(action: RuntimeConfigurationAction): void {
  const current = runtime.currentStatus();
  let operation: Promise<void> | null = null;
  if (action === "start") operation = runtime.start();
  else if (action === "stop") operation = runtime.stop();
  else if (action === "restart-runtime") operation = runtime.restart();
  else if (action === "restart-companion" || action === "configure-api-and-companion") {
    operation = runtime.restartCompanion();
  }
  else if (configured && (current.phase === "stopped" || current.phase === "failed")) {
    operation = runtime.start();
  }
  if (operation) void operation.catch((error) => showRuntimeFailure(error));
}

function systemPermissionDeps(): SystemPermissionDeps {
  return {
    platform: process.platform,
    mediaAccessStatus: (kind) => kind === "systemAudio" ? "not-determined" : systemPreferences.getMediaAccessStatus(kind),
    askForMicrophone: () => process.platform === "darwin"
      ? systemPreferences.askForMediaAccess("microphone")
      : Promise.resolve(systemPreferences.getMediaAccessStatus("microphone") === "granted"),
    primeScreenCapture: async () => {
      await desktopCapturer.getSources({ types: ["screen"], thumbnailSize: { width: 1, height: 1 } });
    },
    openExternal: (url) => shell.openExternal(url),
    probe: (kind) => probePermission(perceptionHelperExecutable(), kind),
    systemAudio: systemAudioPermission,
  };
}

/** Recording system audio (电脑音频感知): judged from the helper's own reports and short captures. */
const systemAudioPermission = new SystemAudioPermission({
  supported: () => supportsSystemAudio(),
  screenGranted: () => process.platform === "win32" || systemPreferences.getMediaAccessStatus("screen") === "granted",
  capability: () => {
    const status = perception?.currentStatus();
    return status?.helperPid ? status.capabilities?.systemAudio : undefined;
  },
  probe: async () => {
    const result = await warmUpCapture(perceptionHelperExecutable(), { screen: false, audio: true });
    return result ? result.systemAudio === "running" : null;
  },
});

/**
 * Onboarding shows the default character while the user answers questions,
 * so runtime assets download in the background instead of after setup.
 */
function prepareOnboarding(): Promise<OnboardingPreparation> {
  onboardingPreparation ??= (async (): Promise<OnboardingPreparation> => {
    const report = (value: OnboardingPreparation) => {
      mainWindow?.webContents.send("desktop:onboarding:preparation", value);
      return value;
    };
    try {
      const configuration = await configStore.load();
      if (!(await assets.ready())) {
        report({ phase: "preparing", percent: 0, detail: "正在准备运行环境…" });
        await assets.install((detail) => {
          report({ phase: "preparing", percent: installPercent(detail), detail });
        });
      }
      await installDefaultModelIfNeeded();
      await attachPresetPreviews();
      return report({ phase: "ready", percent: 100, detail: "运行环境已就绪" });
    } catch (error) {
      onboardingPreparation = null;
      return report({
        phase: "failed",
        percent: null,
        detail: error instanceof Error ? error.message : String(error),
      });
    }
  })();
  return onboardingPreparation;
}

function installPercent(detail: string): number | null {
  const match = detail.match(/\((\d+)\/(\d+)\)/u);
  if (!match) return null;
  const done = Number(match[1]);
  const total = Number(match[2]);
  return total > 0 ? Math.round((done / total) * 100) : null;
}

function sendRuntimeStatus(status: DesktopRuntimeStatus): void {
  for (const window of [petWindow, mainWindow]) {
    window?.webContents.send("desktop:runtime:status", status);
  }
}

function sendPerceptionStatus(status = perception.currentStatus()): void {
  const visibleStatus = withNativePerceptionPermissions(status);
  for (const window of [petWindow, mainWindow]) {
    window?.webContents.send("desktop:perception:status", visibleStatus);
  }
  tray?.setContextMenu(desktopMenu());
}

function withNativePerceptionPermissions(
  status: DesktopPerceptionStatus,
): DesktopPerceptionStatus {
  const helperAccessibility = status.capabilities?.computerHistory
    .computerHistoryInput?.accessibilityPermission
    ?? status.capabilities?.accessibility.permission;
  const nativeAccessibility: DesktopPerceptionPermission = (
    process.platform === "win32" || systemPreferences.isTrustedAccessibilityClient(false)
      ? "granted"
      : "denied"
  );
  const rawScreen = process.platform === "win32" ? "granted" : systemPreferences.getMediaAccessStatus("screen");
  const nativeScreen: DesktopPerceptionPermission = rawScreen === "not-determined"
    ? "not_determined"
    : rawScreen;
  return {
    ...status,
    audioSupported: supportsSystemAudio(),
    systemSettings: systemSettingsNames(darwinMajor(), uiPreferences.uiLocale, process.platform),
    permissions: {
      accessibility: helperAccessibility
        && !["unknown", "not_required"].includes(helperAccessibility)
        ? helperAccessibility
        : nativeAccessibility,
      screen: status.capabilities?.screen.permission
        && !["unknown", "not_required"].includes(
          status.capabilities.screen.permission,
        )
        ? status.capabilities.screen.permission
        : nativeScreen,
    },
  };
}

async function syncPerceptionConfiguration(
  configuration: DesktopConfigurationInput["perception"],
): Promise<void> {
  const exclusionChanged = perceptionConfiguration !== null
    && perceptionConfiguration.excludeFishMemoryFromScreenCapture
      !== configuration.excludeFishMemoryFromScreenCapture;
  if (JSON.stringify(perceptionConfiguration) !== JSON.stringify(configuration)) {
    perceptionConfiguration = { ...configuration };
    mainWindow?.webContents.send("desktop:perception:configuration", perceptionConfiguration);
    tray?.setContextMenu(desktopMenu());
  }
  if (process.platform === "win32") {
    for (const window of BrowserWindow.getAllWindows()) {
      window.setContentProtection(configuration.excludeFishMemoryFromScreenCapture);
    }
  }
  const syncGeneration = ++perceptionSyncGeneration;
  if (exclusionChanged) {
    environmentHistoryRecorder?.clearCapturedWindow();
    await environmentFrameCoordinator?.clearFrame();
    if (syncGeneration !== perceptionSyncGeneration) return;
    environmentHistoryRecorder?.clearCapturedWindow();
  }
  const policy = perceptionPolicy(configuration, supportsSystemAudio());
  environmentFrameSampling?.setPolicy(policy);
  const modules = perceptionModules(policy, configuration.excludeFishMemoryFromScreenCapture);
  const anyEnabled = policy.operationEnabled || policy.screenEnabled || policy.audioEnabled;
  if (!configured || !anyEnabled) {
    environmentHistoryRecorder?.setEnabled(false);
    await deactivatePerception({
      stopPerception: (detail) => perception.stop(detail),
      disableComputerHistory: () => computerHistoryRecorder?.setEnabled(false),
      disableEnvironmentFrames: () => environmentFrameCoordinator?.setPolicy({
        ...policy, operationEnabled: false, screenEnabled: false, audioEnabled: false,
      }),
    }, "感知服务未启用");
    if (syncGeneration !== perceptionSyncGeneration) return;
    sendPerceptionStatus();
    return;
  }
  const runtimeStatus = runtime.currentStatus();
  if (
    !runtimeStatus.apiUrl
    || !["starting-companion", "ready"].includes(runtimeStatus.phase)
  ) {
    environmentHistoryRecorder?.setEnabled(false);
    await deactivatePerception({
      stopPerception: (detail) => perception.stop(detail),
      disableComputerHistory: () => computerHistoryRecorder?.setEnabled(false),
      disableEnvironmentFrames: () => environmentFrameCoordinator?.setEnabled(false),
    }, "感知服务正在等待本地 API");
    if (syncGeneration !== perceptionSyncGeneration) return;
    sendPerceptionStatus();
    return;
  }
  if (perceptionPausedBySystem) {
    environmentHistoryRecorder?.setEnabled(false);
    await deactivatePerception({
      stopPerception: (detail) => perception.stop(detail),
      disableComputerHistory: () => computerHistoryRecorder?.setEnabled(false),
      disableEnvironmentFrames: () => environmentFrameCoordinator?.setEnabled(false),
    }, "感知服务已因锁屏或睡眠暂停");
    if (syncGeneration !== perceptionSyncGeneration) return;
    sendPerceptionStatus();
    return;
  }
  environmentHistoryRecorder?.setPolicy(policy);
  const recorderReady = computerHistoryRecorder?.setEnabled(
    policy.operationEnabled,
  );
  const environmentFrameReady = environmentFrameCoordinator?.setPolicy(policy);
  await Promise.all([recorderReady, environmentFrameReady]);
  if (syncGeneration !== perceptionSyncGeneration) return;
  perception.start(modules);
  sendPerceptionStatus();
}

function pausePerceptionForSystem(reason: string): void {
  perceptionSyncGeneration += 1;
  perceptionPausedBySystem = true;
  environmentHistoryRecorder?.setEnabled(false);
  void environmentFrameCoordinator?.clearFrame();
  const stopped = perception.stop(`感知服务已因${reason}暂停`);
  sendPerceptionStatus();
  runSafely(
    stopped.then(() => configStore.load())
      .then((value) => syncPerceptionConfiguration(value.perception)),
    `${reason}时暂停主动观察`,
  );
}

function resumePerceptionAfterSystem(reason: string): void {
  if (quitting) return;
  perceptionSyncGeneration += 1;
  perceptionPausedBySystem = false;
  runSafely(
    configStore.load().then((value) => syncPerceptionConfiguration(value.perception)),
    `${reason}后恢复本机感知`,
  );
}

function currentMainWindowState(): DesktopMainWindowState {
  return resolveMainWindowState(requestedMainView, configured, runtime.currentStatus());
}

function sendMainWindowState(): void {
  mainWindow?.webContents.send("desktop:main:state", currentMainWindowState());
}

function sendMainWindowVisibility(visible: boolean): void {
  mainWindow?.webContents.send("desktop:main:visibility", visible);
}

function sendPetWindowVisibility(visible: boolean): void {
  petWindow?.webContents.send("pet:visibility", visible);
}

function connectToFishMemory(apiUrl: string): void {
  if (connectedApiUrl === apiUrl && socket) return;
  socket?.stop();
  connectedApiUrl = apiUrl;
  runSafely(
    configStore.runtimeSecrets().then((secrets) => {
      socket = new PetSocket({ apiUrl, apiKey: secrets.internalApiKey });
      socket.on("control", (value) => {
        if (typeof value === "object" && value !== null
          && (value as { type?: unknown }).type === "history.boundary") {
          const payload = (value as {
            payload?: { occurredAt?: unknown; fishId?: unknown };
          }).payload;
          if (typeof payload?.occurredAt === "number"
            && payload.fishId === environmentFrameCoordinator?.currentFishId) {
            perception.cutEnvironmentWindow(payload.occurredAt);
          }
          return;
        }
        if (typeof value === "object" && value !== null
          && (value as { type?: unknown }).type === "pet.assistant.changed") {
          runSafely(environmentFrameCoordinator?.refreshBinding() ?? Promise.resolve(),
            "同步当前宠物感知绑定");
          return;
        }
        if (isComputerHistoryRefreshRequest(value)) {
          const fishId = computerHistoryRefreshFishId(value);
          if (fishId) void computerHistoryRecorder?.requestRefresh(fishId);
          return;
        }
        if ((value as { type?: unknown } | null)?.type === "pet.owner_voice.unrecognized") {
          ownerVoiceUnrecognized(value);
          return;
        }
        if ((value as { type?: unknown } | null)?.type === "pet.service.problems") {
          serviceProblemsChanged(value);
          return;
        }
        // A reconnect replays the latest problems inside the snapshot.
        const replayed = (value as { type?: unknown; payload?: { events?: unknown } } | null);
        if (replayed?.type === "pet.snapshot" && Array.isArray(replayed.payload?.events)) {
          const problems = replayed.payload.events.find((event) => (event as { type?: unknown })?.type === "pet.service.problems");
          if (problems) serviceProblemsChanged(problems);
        }
        petWindow?.webContents.send("pet:control", value);
        petChatWindow?.control(value);
        const type = (value as { type?: unknown } | null)?.type;
        // A reconnect replays the state inside the snapshot.
        const stateEvent = petStateEvent(value);
        if (stateEvent) {
          lastPetState = stateEvent;
          petStateChanged(typeof stateEvent.payload?.state === "string" ? stateEvent.payload.state : null);
          if (type === "pet.snapshot") mainWindow?.webContents.send("desktop:chat:pet-event", stateEvent);
        }
        if (type === "pet.state" || type === "pet.playback.progress" || type === "pet.snapshot" || type === "pet.voice.hearing") {
          mainWindow?.webContents.send("desktop:chat:pet-event", value);
        }
      });
      socket.on("audio", (value) => petWindow?.webContents.send("pet:audio", value));
      socket.on("connection", (value) => {
        petWindow?.webContents.send("pet:connection", value);
        if (value) {
          runSafely(environmentFrameCoordinator?.refreshBinding() ?? Promise.resolve(),
            "恢复当前宠物感知绑定");
        } else petChatWindow?.disconnected();
      });
      socket.start();
    }),
    "连接桌宠事件通道",
  );
}

function computerHistoryRefreshFishId(value: unknown): string | null {
  if (typeof value !== "object" || value === null) return null;
  const payload = (value as { payload?: unknown }).payload;
  if (typeof payload !== "object" || payload === null) return null;
  const fishId = (payload as { fishId?: unknown }).fishId;
  return typeof fishId === "string"
    && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu.test(fishId)
    ? fishId
    : null;
}

function isComputerHistoryRefreshRequest(value: unknown): boolean {
  return typeof value === "object"
    && value !== null
    && (value as { type?: unknown }).type === "computer_history.refresh_requested";
}

async function installDefaultModelIfNeeded(): Promise<void> {
  if (await modelStore.selected()) return;
  const configured = process.env.FISH_MEMORY_DEFAULT_LIVE2D_MODEL_DIR;
  const source = configured ? path.resolve(configured) : assets.modelRoot;
  try {
    await access(source);
    // The shipped look gets a Chinese name; its folder is only called "Mao".
    const imported = await modelStore.importFolder(source, configured ? undefined : "小巫师");
    if (!imported.model) {
      throw new Error(imported.report.errors[0]?.message ?? "默认 Live2D 模型导入失败");
    }
    if (!configured) await nameBuiltInExpressions(imported.model);
  } catch (error) {
    console.error(`Unable to install the default Live2D model from ${source}:`, error);
  }
  await attachPresetPreviews();
  await refreshTrayModels();
}

/** What each of 小巫师's expression files shows, read from their parameters. */
const BUILT_IN_EXPRESSION_NAMES: Record<string, string> = {
  exp_01: "平常", exp_02: "眯眼笑", exp_03: "闭眼", exp_04: "星星眼",
  exp_05: "难过", exp_06: "脸红", exp_07: "吃惊", exp_08: "生气",
};

/** The files are only numbered, so the switches get names a person can read. */
async function nameBuiltInExpressions(model: ModelPackage): Promise<void> {
  const manifest = structuredClone(model.manifest);
  let renamed = false;
  for (const asset of Object.values(manifest.expressionAssets)) {
    const name = BUILT_IN_EXPRESSION_NAMES[asset.name];
    if (name) { asset.name = name; renamed = true; }
  }
  if (renamed) await modelStore.saveManifest(model.id, manifest);
}

/**
 * What a look shows at rest: its files, saved default appearance and applied
 * texture materials. Its card picture is redrawn once this changes.
 */
async function lookKey(id: string): Promise<string> {
  const [stamp, snapshot] = await Promise.all([modelStore.lookStamp(id), textureMaterialStore.appearanceSnapshot(id)]);
  return `${stamp}:${snapshot.delta.revision}`;
}

/** Gives built-in 形象 their shipped card thumbnail instead of rendering one on first run. */
async function attachPresetPreviews(): Promise<void> {
  for (const model of await modelStore.list()) {
    if (model.previewUrl) continue;
    const file = presetPreviewFile(model.url, app.getAppPath(), app.isPackaged);
    if (!file) continue;
    try {
      await modelStore.savePreview(model.id, await readFile(file), await lookKey(model.id));
    } catch (error) {
      console.error(`Unable to attach the preset preview for ${model.id}:`, error);
    }
  }
}

function createTray(): void {
  // Use a real PNG instead of SVG: packaged Electron can produce an empty
  // nativeImage for SVG data URLs on some macOS versions. Template images
  // automatically switch between dark and light menu bars.
  const trayIcon = nativeImage.createFromDataURL(
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAkCAYAAADhAJiYAAAAAXNSR0IArs4c6QAAA3lJREFUWIXFl19ojXEYxz/vztmG2VpbWJGSC0XSLoQIu8DQEqHcKEUo/4qSWrZWEjfucEHTLlbUXA8l4khalC0zcrHtRmZhxDZne918f3q83nN23p13PHU657y/5/d9vr/n7+9NEK94QFK//ZC1nADiJGNJJIFy/R4MITjlhAAWARuAtUAFkAIeAc9Erj9XYvmIBxQBF4BRGbwCLAUKpFMJXDP6cTviDzIAl0RkAFgdondCZMPWYpd5wHcgDezVsyKRrQXaRXZcOs1A9VQQSShxa2XsE1CmZ85zs4E64LFItQA1JtnzFk9ErMyVsX6TMwT0lgEvczEQBJ9Id1zGC1RR2/W8FFgO3AaqlEvj2leokPYB3dqbd6W5EJQDx4AOQ84HGoFeoAl4AzwHjpjQJOOsLAe0C+gxJHqBVlXYYWAVUA8cNTqvgZ0BnFjInDVG7ihRp4XozwHmA1uAe2ZPfRykXILuF+hP4GSIXiFwCLgLvANWmLVTKnUf2BfAjSTuJAuBL8qXA2bN5UQV8MB44rgh6TAOau0zsCCAn7O4UzQJ7Kb+u6p0gFeN996rD1ljTr9Neg0B/MjyVEBrBGKBPFVTWh58mOFgnoatDzyZyGAyy1qx1nuBTtNXEqZBeobkD4XK9RjftIZO9aGkcEcmQ2gEuAjMAO4DtzTRx7SeVqhciEb1Pyhn1DIagW/ZyJChU7tOehq4DAwDGzUwbwA7gN2aSTXq0g6rAlgHrNeVoxs4B6xUt24QXipqx54FDCkM1+WNNk1pP8KnWvvSwhlW1VZmMhzMdvd/sU4+qpM7T4Y1w2xSbLyXULjKgCUZ7GfNoaA7S1R1TQqBr3FRpvWPmnGOQEr6JSG4GftQkJCrpG4laNJ0Wj/QS1DZu8tWB7A5w8F84RTKS68C9n5LWIPygA/AeVVYqZ65cCXMtxfYF4ZdrLWZwHRV6kAmL4URcqFqBLbqtjckrxGhMpzeW+1PCa8xIk6olIaQTwAvTEjaM+xNmNYwoWRLakyv+JorYIiMab97/fkrb6IQyrbZj+j2nPQnPXUDSenmVt63wnwI2dN6uj8HPRAZPx9C45pPvvKkGtij0i7KJV/iFHeQFpEZMTnSB3SpS9dKb8re5Z245rjN3BhH9G27ete/IOPElXFzlmnfanRzBo1D6oBNegVyb7g9Gj+DMdn4P/ILlqXym1VdztAAAAAASUVORK5CYII=",
  ).resize({ width: 18, height: 18, quality: "best" });
  trayIcon.setTemplateImage(true);
  tray = new Tray(trayIcon);
  if (trayIcon.isEmpty()) tray.setTitle("✦");
  tray.setToolTip("tulpa");
  tray.setContextMenu(desktopMenu());
  tray.on("click", () => {
    tray?.popUpContextMenu(desktopMenu());
  });
  runSafely(refreshTrayModels(), "刷新模型列表");
}

/** "listening", "speaking"… from the latest pet.state event. */
function petStateName(): string | null {
  const state = (lastPetState as { payload?: { state?: unknown } } | null)?.payload?.state;
  return typeof state === "string" ? state : null;
}

async function handleQuickAction(action: QuickAppAction): Promise<void> {
  switch (action.type) {
    case "look": await selectModelFromMenu(action.id); return;
    case "hide":
      // Say how to get TA back before going: a hidden pet looks like a lost one.
      petChatWindow?.showText(uiText("我先藏起来啦。想找我，就再打开一次 tulpa，或者点菜单栏里的 tulpa 图标～", "I'll hide for now. To bring me back, open tulpa again or click the tulpa icon in the menu bar."), "completed");
      await new Promise((resolve) => setTimeout(resolve, 2_500));
      await petWindowLifecycle.toggle();
      return;
    case "main": await openMainWindow(requestedMainView === "onboarding" && configured ? "chat" : requestedMainView); return;
    case "quit": app.quit(); return;
    case "tile":
      if (action.key === "speaker") {
        await runtime.setPlaybackEnabled(!(runtime.currentStatus().playbackEnabled ?? true));
        tray?.setContextMenu(desktopMenu());
      } else if (action.key === "passthrough") setInteractionLocked(true);
      else await openMainWindow(action.key);
  }
}

function uiText(zh: string, en: string): string {
  return uiPreferences?.uiLocale === "en" ? en : zh;
}

function refreshApplicationMenu(): void {
  Menu.setApplicationMenu(Menu.buildFromTemplate(appMenuTemplate(app.getName(), () => {
    runSafely(openMainWindow("settings"), uiText("打开设置", "Open settings"));
  }, uiPreferences.uiLocale)));
}

function desktopMenu(): Menu {
  const status = runtime?.currentStatus();
  const voiceEnabled = status?.voiceEnabled ?? true;
  const items = trayMenu({
    locked,
    petVisible: Boolean(petWindow && !petWindow.isDestroyed() && petWindow.isVisible()),
    playback: status?.playbackEnabled ?? voiceEnabled,
    microphone: status?.microphoneEnabled ?? voiceEnabled,
    audioSupported: supportsSystemAudio(),
    senses: configured ? latestSenses : null,
    bubbles: petChatWindow ? petChatWindow.bubblesEnabled : null,
    looks: availableModels.map((model) => ({ id: model.id, name: model.name })),
    lookId: selectedModelId,
  }, uiPreferences.uiLocale);
  const build = (item: TrayItem): MenuItemConstructorOptions => {
    if ("separator" in item) return { type: "separator" };
    const { label, action, checked, radio, enabled, submenu } = item;
    return {
      label,
      ...(checked !== undefined ? { type: radio ? "radio" as const : "checkbox" as const, checked } : {}),
      ...(enabled !== undefined ? { enabled } : {}),
      ...(submenu ? { submenu: submenu.map(build) } : {}),
      ...(action ? { click: () => runSafely(handleTrayAction(action), label) } : {}),
    };
  };
  return Menu.buildFromTemplate(items.map(build));
}

async function handleTrayAction(action: TrayAction): Promise<void> {
  const status = runtime.currentStatus();
  switch (action.type) {
    case "chat": await openMainWindow("chat"); return;
    // Back to wherever the user last was in the main window, not a fixed page.
    case "main": await openMainWindow(requestedMainView === "onboarding" && configured ? "chat" : requestedMainView); return;
    case "pet": await (configured ? petWindowLifecycle.toggle() : openMainWindow("settings")); return;
    case "speaker": await runtime.setPlaybackEnabled(!(status.playbackEnabled ?? status.voiceEnabled ?? true)); break;
    case "microphone": await runtime.setMicrophoneEnabled(!(status.microphoneEnabled ?? status.voiceEnabled ?? true)); break;
    case "sense": await handleSenseAction({ type: "toggle", key: action.key }); return;
    case "companion":
      await handleSenseAction(latestSenses?.companionPaused ? { type: "resume" } : { type: "pause", closeSenses: false });
      return;
    case "bubbles": if (petChatWindow) await petChatWindow.setBubblesEnabled(!petChatWindow.bubblesEnabled); return;
    case "look": await selectModelFromMenu(action.id); return;
    case "passthrough": setInteractionLocked(!locked); return;
    case "quit": app.quit(); return;
  }
  tray?.setContextMenu(desktopMenu());
}

// ---------- speech model readiness ----------

let modelReadiness: ModelReadiness = { state: "ready", text: "", detail: "", items: [] };
let modelReadinessSignature = "";
let refreshModelReadiness: () => Promise<void> = async () => {};
let stopModelReadinessWatch: () => void = () => {};
let modelPreparationRetry: Promise<void> | null = null;

/** Both the settings page and pet status use the same recovery attempt. */
function retryModelPreparation(): Promise<void> {
  if (modelPreparationRetry) return modelPreparationRetry;
  modelPreparationRetry = (async () => {
    modelReadinessSignature = "";
    try {
      await runtime.retryModelPreparation();
    } finally {
      await refreshModelReadiness();
    }
  })().finally(() => { modelPreparationRetry = null; });
  return modelPreparationRetry;
}

/** Runtime processes report model downloads to files; show progress where the user looks. */
function startModelReadinessWatch(userData: string): void {
  stopModelReadinessWatch();
  const directory = modelStatusDirectory(userData);
  const clock = new DownloadClock();
  const poller = new ModelReadinessPoller(() => readModelStatus(directory), (items) => {
    const status = runtime?.currentStatus();
    const starting = !configured || !status ? null
      : status.phase === "starting-api" ? uiText("TA 正在醒来…", "Your companion is waking up…")
      : status.phase === "starting-companion" ? uiText("正在启动语音…", "Starting voice…")
      : status.phase === "restarting" ? uiText("正在重新启动…", "Restarting…")
      : null;
    const issue = configured && status?.phase === "starting-companion" ? status.voiceIssue : null;
    const summary = summarizeReadiness(items, starting, clock, Date.now(), uiPreferences.uiLocale, configured ? status : undefined);
    // A companion stuck retrying its audio says why, instead of "starting" forever.
    const next = issue && summary.state !== "failed"
      ? {
        ...summary,
        state: "failed" as const,
        text: uiText(issue.text, issue.kind === "microphone" ? "Microphone access is blocked, so your companion cannot hear you" : "Voice is unavailable; retrying"),
        detail: issue.kind === "microphone" ? uiText("点一下，去系统设置里允许麦克风", "Click to allow microphone access in System Settings") : uiText("点一下重新启动语音", "Click to restart voice"),
      }
      : summary;
    const signature = JSON.stringify(next);
    if (signature === modelReadinessSignature) return;
    modelReadinessSignature = signature;
    modelReadiness = next;
    petChatWindow?.setReadiness(next.state === "ready" ? null : next);
    mainWindow?.webContents.send("desktop:model-readiness", next);
  });
  refreshModelReadiness = () => poller.refresh();
  const timer = setInterval(() => runSafely(poller.refresh(), "读取模型准备状态"), 1_500);
  timer.unref();
  stopModelReadinessWatch = () => { clearInterval(timer); poller.stop(); };
}

/** The bound assistant's display name, read from the running memory service. */
async function currentCharacter(): Promise<{ name: string } | null> {
  const status = runtime.currentStatus();
  if (!status.apiUrl) return null;
  const headers = { Authorization: `Bearer ${(await configStore.runtimeSecrets()).internalApiKey}` };
  const binding = await fetch(`${status.apiUrl}/v1/fishes/assistant`, { headers, signal: AbortSignal.timeout(5_000) });
  if (!binding.ok) return null;
  const { fishId } = await binding.json() as { fishId?: unknown };
  if (typeof fishId !== "string") return null;
  const soul = await fetch(`${status.apiUrl}/v1/fishes/${encodeURIComponent(fishId)}/soul`, { headers, signal: AbortSignal.timeout(5_000) });
  if (!soul.ok) return null;
  const { name } = await soul.json() as { name?: unknown };
  return typeof name === "string" && name.trim() ? { name: name.trim() } : null;
}

// ---------- senses capsule (屏幕感知 / 电脑音频感知 / 麦克风收音 | 内心) ----------

let sensePermissions: Record<SystemPermissionKind, SystemPermissionState> = { microphone: "granted", screen: "granted", systemAudio: "granted" };
let sensePermissionsReadAt = 0;
let senseWaiting: SystemPermissionKind | null = null;
let senseWaitTimer: NodeJS.Timeout | null = null;

/** Seconds of the owner's speech TA just missed, kept in the companion's memory. */
let ownerVoiceMissedSeconds = 0;
/** Key or quota problems the local service reported, until each service works again. */
let serviceProblems: NonNullable<ReturnType<typeof serviceFailure>>[] = [];
let usageCache: { at: number; summary: UsageSummary } | null = null;

/** Tokens by purpose from the local service; reused for half a minute. */
async function currentUsage(maxAgeMs = 30_000): Promise<UsageSummary | null> {
  if (usageCache && Date.now() - usageCache.at < maxAgeMs) return usageCache.summary;
  const dayStart = new Date();
  dayStart.setHours(0, 0, 0, 0);
  try {
    const summary = summarizeUsage(parseUsageRows(await llmUsage(dayStart.toISOString())));
    usageCache = { at: Date.now(), summary };
    return summary;
  } catch {
    return usageCache?.summary ?? null;
  }
}
const OWNER_VOICE_ADD_SECONDS = 3;

async function currentSenses(): Promise<SensesSnapshot | null> {
  if (!configured) return null;
  const configuration = await configStore.load();
  const senses = sensesSnapshot({
    perception: configuration.perception,
    microphoneEnabled: configuration.microphoneEnabled ?? configuration.voiceEnabled,
    proactiveCompanionEnabled: configuration.proactiveCompanionEnabled,
    audioSupported: supportsSystemAudio(),
    permissions: sensePermissions,
    waiting: senseWaiting,
  });
  // Asleep matters only while the wake word can be heard.
  const hearing = senses.senses.some((sense) => sense.key === "microphone" && sense.status === "on");
  const wakeWord = petAsleep && hearing ? await currentWakeWord() : "";
  return {
    ...senses,
    ...(wakeWord ? { sleeping: { wakeWord } } : {}),
    ownerVoiceMissed: configuration.audio.ownerRecognitionEnabled
      && ownerVoiceMissedSeconds >= OWNER_VOICE_ADD_SECONDS,
    ...(serviceProblems.length ? { problems: serviceProblems.map((failure): SenseProblem => ({
      text: failureReason(failure, uiPreferences.uiLocale), section: settingsSectionFor(failure),
    })) } : {}),
    ...await usageLine(),
  };
}

/** The word that wakes the current character, else its name. */
async function currentWakeWord(): Promise<string> {
  try {
    const fishId = environmentFrameCoordinator?.currentFishId
      ?? (await characterApi.list()).find((character) => character.current)?.id;
    if (!fishId) return "";
    if (wakeWordCache?.fishId !== fishId) {
      const profile = await characterApi.profile(fishId);
      wakeWordCache = { fishId, word: profile.wakeWords[0]?.trim() || profile.name };
    }
    return wakeWordCache.word;
  } catch {
    return "";
  }
}

/** TA fell asleep or woke: the capsule shows it, and TA says once how to wake it. */
function petStateChanged(state: string | null): void {
  const asleep = state === "standby";
  if (!asleep) petChatWindow?.sleepHint(null);
  const say = sleepWatch.next(state);
  if (asleep === petAsleep && !say) return;
  petAsleep = asleep;
  runSafely((async () => {
    await publishSenses();
    const word = latestSenses?.sleeping?.wakeWord;
    if (say && word && petAsleep) {
      petChatWindow?.sleepHint(uiText(`我先眯会儿，喊“${word}”叫我`, `I'll nap for now. Say “${word}” to wake me`));
    }
  })(), "更新睡着提示");
}

async function usageLine(): Promise<Pick<SensesSnapshot, "usage">> {
  const summary = await currentUsage();
  return summary ? { usage: { today: tokenText(summary.today.tokens, uiPreferences.uiLocale), background: activeBackground(summary) } } : {};
}

function serviceProblemsChanged(value: unknown): void {
  const codes = (value as { payload?: { problems?: unknown } }).payload?.problems;
  serviceProblems = (Array.isArray(codes) ? codes : []).flatMap((code) => {
    const failure = serviceFailure(code);
    return failure ? [failure] : [];
  });
  runSafely(publishSenses(), "更新服务问题");
}

function ownerVoiceUnrecognized(value: unknown): void {
  const payload = (value as { payload?: Record<string, unknown> }).payload ?? {};
  ownerVoiceMissedSeconds = typeof payload.seconds === "number" ? payload.seconds : 0;
  const note = typeof payload.error === "string" ? payload.error
    : typeof payload.enrollmentCount === "number" ? uiText(`加好了，现在认得 ${payload.enrollmentCount} 段你的声音`, `Added. I now recognize ${payload.enrollmentCount} samples of your voice`) : undefined;
  // The result of adding shows where the prompt was, not only in the panel.
  if (note) petChatWindow?.showText(note, "completed");
  runSafely(publishSenses(note), "更新主人声纹提示");
}

async function publishSenses(note?: string): Promise<void> {
  const snapshot = await currentSenses();
  latestSenses = snapshot;
  tray?.setContextMenu(desktopMenu());
  petChatWindow?.setSenses(snapshot, snapshot ? note : "");
  mainWindow?.webContents.send("desktop:chat:senses", snapshot);
}

/** Permissions change outside the app, so they are re-read whenever the capsule appears. */
async function refreshSensePermissions(force = false): Promise<void> {
  if (!force && Date.now() - sensePermissionsReadAt < 4_000) return;
  sensePermissionsReadAt = Date.now();
  const deps = systemPermissionDeps();
  const [microphone, screen, systemAudio] = await Promise.all([
    currentSystemPermission("microphone", deps),
    currentSystemPermission("screen", deps),
    currentSystemPermission("systemAudio", deps),
  ]);
  sensePermissions = { microphone, screen, systemAudio };
  const resolved = senseWaiting !== null && (sensePermissions[senseWaiting] === "granted");
  if (resolved) stopSenseWait();
  await publishSenses(resolved ? "" : undefined);
}

function stopSenseWait(): void {
  senseWaiting = null;
  if (senseWaitTimer) clearInterval(senseWaitTimer);
  senseWaitTimer = null;
}

async function handleSenseAction(action: Exclude<SenseAction, { type: "panel" }>): Promise<void> {
  if (action.type === "retry-models") {
    if (runtime.currentStatus().voiceIssue?.kind === "microphone") {
      // The companion retries on its own once macOS allows the microphone.
      await handleSenseAction({ type: "fix", kind: "microphone" });
      return;
    }
    await retryModelPreparation();
    return;
  }
  if (action.type === "settings") {
    await openSettings(action.section);
    return;
  }
  if (action.type === "owner-voice-add") {
    const note = uiText("正在把刚才那几句加入声纹…", "Adding those sentences to your voiceprint…");
    petChatWindow?.showText(note, "completed");
    await publishSenses(note);
    await chatApi.addMissedOwnerVoice();
    return;
  }
  if (action.type === "fix") {
    const state = await requestSystemPermission(action.kind, systemPermissionDeps());
    sensePermissions = { ...sensePermissions, [action.kind]: state };
    if (state === "granted" || state === "not-required") {
      stopSenseWait();
      await publishSenses("");
      return;
    }
    senseWaiting = action.kind;
    if (!senseWaitTimer) {
      const startedAt = Date.now();
      senseWaitTimer = setInterval(() => {
        if (Date.now() - startedAt > 3 * 60_000) {
          stopSenseWait();
          runSafely(publishSenses(""), "更新感知状态");
          return;
        }
        runSafely(refreshSensePermissions(true), "读取感知权限");
      }, 1_500);
    }
    const name = settingsAppName(process.execPath);
    await publishSenses(action.kind === "screen"
      ? uiText(`在系统设置里打开「${name}」的开关。系统问要不要退出并重新打开时，选「以后」就行，不用重启。`, `Enable “${name}” in System Settings. If asked to quit and reopen, choose “Later”; no restart is needed.`)
      : uiText(`在系统设置的麦克风列表里打开「${name}」的开关，回来就会亮起。`, `Enable “${name}” in the Microphone list in System Settings, then return here.`));
    return;
  }
  await applySenseChange(action);
}

async function applySenseChange(action: Extract<SenseAction, { type: "toggle" | "pause" | "resume" }>): Promise<void> {
  let microphone: { before: boolean; after: boolean } | null = null;
  await runtime.withConfigurationChange(async () => {
    const previous = await configStore.load();
    const before = previous.microphoneEnabled ?? previous.voiceEnabled;
    const next = applySenseAction({
      perception: previous.perception,
      microphoneEnabled: before,
      proactiveCompanionEnabled: previous.proactiveCompanionEnabled,
    }, action, supportsSystemAudio());
    microphone = { before, after: next.microphoneEnabled };
    const { secrets: _flags, ...persisted } = previous;
    const input: DesktopConfigurationInput = {
      ...persisted,
      proactiveCompanionEnabled: next.proactiveCompanionEnabled,
      perception: next.perception,
    };
    const saved = await configStore.save(input);
    await syncPerceptionConfiguration(saved.perception);
    const change = runtimeConfigurationAction(previous, saved, input);
    // 暂停/恢复 and 内心 live in the API; without this they only reached the file.
    await pushApiConfiguration(change, saved);
    applyRuntimeConfiguration(change);
  });
  const change = microphone as { before: boolean; after: boolean } | null;
  // The microphone switch is applied live by the runtime, without a restart.
  if (change && change.before !== change.after) await runtime.setMicrophoneEnabled(change.after);
  tray?.setContextMenu(desktopMenu());
  await publishSenses();
}

function setInteractionLocked(value: boolean): void {
  const entering = value && !locked;
  locked = value;
  petChatWindow?.setLocked(value);
  petDragSession = null;
  if (locked) petPanelFocus?.setHovered(false);
  petWindow?.webContents.send("pet:command", {
    type: "interaction-lock-changed",
    locked,
  });
  petWindow?.setIgnoreMouseEvents(locked, { forward: true });
  tray?.setToolTip(
    locked
      ? "tulpa（鼠标穿透中，点这里选「关闭鼠标穿透」）"
      : "tulpa",
  );
  tray?.setContextMenu(desktopMenu());
  // Clicks now pass through TA, so say once how to get them back.
  if (entering) petChatWindow?.showText(uiText("我先不挡你的鼠标啦。想找我时，把鼠标移到我身上，点那个小锁就好～", "Your clicks will pass through me now. To interact again, hover over me and click the little lock."), "completed");
}

const modelsAwaitingAutoFill = new Set<string>();

/** Fills only still-empty states and emotions of a newly imported 形象 and saves them. */
async function autoFillImportedModel(job: EmotionMatchingJob): Promise<void> {
  if (!modelsAwaitingAutoFill.has(job.modelId)) return;
  if (job.status === "failed" || job.status === "cancelled" || job.status === "stale") modelsAwaitingAutoFill.delete(job.modelId);
  if (job.status !== "completed" && job.status !== "partial") return;
  modelsAwaitingAutoFill.delete(job.modelId);
  const model = await modelStore.load(job.modelId);
  const suggestions = suggestionsForEmptyTargets(model.manifest, job.suggestions);
  if (!suggestions.length) return;
  const next = mergeEmotionSuggestions(model.manifest, suggestions);
  const adoptions = suggestions.map((suggestion) => ({ jobId: job.id, suggestionId: suggestion.id, candidateId: suggestion.candidate.id }));
  const saved = await modelMutations.run(model.id, async () => {
    await emotionMatching.validate(model.id, next, adoptions);
    return modelStore.saveManifest(model.id, next);
  });
  await emotionMatching.adopted(model.id, adoptions, saved.manifest).catch(() => undefined);
  if (model.id === selectedModelId) await sendSelectedModel(saved);
}

/** The 形象 a character wears follows the last one chosen while it was current. */
async function bindAppearanceToCurrentCharacter(modelId: string): Promise<void> {
  // Right after onboarding the character does not exist yet; it is created with
  // this look once TA is up, so there is nothing to record here.
  if (await pendingCharacters.pending()) return;
  if (!runtime?.currentStatus().apiUrl) {
    console.warn("TA 还没醒，这次换的形象没有记到角色上");
    return;
  }
  const current = (await characterApi.list()).find((character) => character.current);
  if (current) await characterAppearances.bind(current.id, modelId);
}

async function selectModelFromMenu(id: string): Promise<void> {
  const model = await modelStore.select(id);
  selectedModelId = model.id;
  await petWindowLifecycle.show();
  await sendSelectedModel(model);
  await refreshTrayModels();
}

async function sendSelectedModel(model: ModelPackage): Promise<void> {
  const bundle = await modelRenderBundle(model);
  if (modelIdForOffset !== null && modelIdForOffset !== model.id) {
    petModelOffset = { x: 0, y: 0 };
    petModelBounds = null;
    sendPetModelOffsetChanged();
    await savePetWindowState();
  }
  modelIdForOffset = model.id;
  petWindow?.webContents.send("pet:command", { type: "model-selected", bundle });
}

async function modelRenderBundle(model: ModelPackage): Promise<ModelRenderBundle> {
  return {
    model,
    appearance: (await textureMaterialStore.appearanceSnapshot(model.id)).delta,
  };
}

function sendTextureMaterialsUpdated(
  change: TextureMaterialStoreChangeResult,
  senderId: number | null = null,
): void {
  emotionMatching?.sourceChanged(change.delta.modelId);
  const command = { type: "texture-materials-updated" as const, change };
  const mainContents = mainWindow?.webContents;
  const petContents = petWindow?.webContents;
  const recipients = new Set(textureUpdateRecipients({
    senderId,
    mainId: mainContents?.id,
    petId: petContents?.id,
    modelIsSelectedPet: change.delta.modelId === selectedModelId,
  }));
  for (const contents of [mainContents, petContents]) {
    if (contents && !contents.isDestroyed() && recipients.has(contents.id)) {
      contents.send("pet:command", command);
    }
  }
}

function sendTextureCompactionWarning(modelId: string, revision: number, error: Error): void {
  const message = `后台图集缓存生成失败，不影响当前素材：${error.message}`;
  console.warn(message, { modelId, revision });
  mainWindow?.webContents.send("pet:command", {
    type: "texture-compaction-warning",
    modelId,
    revision,
    message,
  });
}

const recolorPreviewRevisions = new Map<string, number>();

/** Recolor previews use the generation-artifact scheme under a per-model id. */
function recolorJobId(modelId: string): string {
  return `recolor-${createHash("sha256").update(modelId).digest("hex").slice(0, 16)}`;
}

async function recolorUnits(modelId: string, textureUnitIds: readonly string[]): Promise<TextureEditUnit[]> {
  const capability = await modelEditCapabilities.current(modelId);
  const byId = new Map(capability.textureUnits.map((unit) => [unit.id, unit]));
  const units = [...new Set(textureUnitIds)].flatMap((unitId) => {
    const unit = byId.get(unitId);
    return unit && !unit.maskOnly ? [unit] : [];
  });
  if (!units.length) throw new Error("没有可换色的纹理");
  return units;
}

/**
 * The preview with these pieces recolored to `hex` on top of what the model
 * wears; `hex` null takes the preview back to what it wears.
 */
async function recolorPreview(modelId: string, textureUnitIds: readonly string[], hex: string | null): Promise<TextureAtlasDelta | null> {
  const jobId = recolorJobId(modelId);
  const revision = (recolorPreviewRevisions.get(modelId) ?? 0) + 1;
  recolorPreviewRevisions.set(modelId, revision);
  const units = await recolorUnits(modelId, textureUnitIds);
  const affectedAtlases = [...new Map(units.map((unit) => [unit.atlasIndex, { atlasIndex: unit.atlasIndex, atlasFile: unit.atlasFile }])).values()];
  const official = (await textureMaterialStore.appearanceSnapshot(
    modelId,
    affectedAtlases.map((atlas) => atlas.atlasFile),
  )).delta;
  const directory = characterArtifacts.jobDirectory(jobId);
  const superseded = () => recolorPreviewRevisions.get(modelId) !== revision;
  const patches = hex
    ? await writeRecoloredPatches({
      units,
      hex,
      directory,
      name: String(revision),
      currentImage: (unit) => resolveTextureMaterialPreview(modelId, unit.id),
      maskFile: (unit) => modelStore.resolveModelFile(modelId, unit.maskPath),
      superseded,
    })
    : [];
  // A newer color was picked meanwhile; its preview is the one to show.
  if (!patches || superseded()) return null;
  // Keep this preview and the one before it (the renderer may still be loading it).
  const stale = (await readdir(directory).catch(() => [] as string[])).filter((file) => {
    const match = /^recolor-[0-9a-f]+-(\d+)\.png$/.exec(file);
    return match && Number(match[1]) < revision - 1;
  });
  await Promise.all(stale.map((file) => rm(path.join(directory, file), { force: true })));
  return buildTexturePreviewSnapshot({
    official,
    jobId,
    revision,
    enabled: Boolean(hex),
    affectedAtlases,
    acceptedEdits: patches.map((patch) => ({
      textureUnitId: patch.textureUnitId,
      atlasIndex: patch.atlasIndex,
      atlasFile: patch.atlasFile,
      bounds: patch.bounds,
      imageUrl: characterArtifactUrl(jobId, patch.editedPatchPath, revision),
      maskUrl: characterArtifactUrl(jobId, patch.maskPath, revision),
    })),
  });
}

function characterArtifactUrl(jobId: string, file: string, revision: number): string {
  const root = characterArtifacts.jobDirectory(jobId);
  const relative = path.relative(root, file);
  if (
    !relative ||
    relative.startsWith("..") ||
    path.isAbsolute(relative) ||
    path.basename(relative) !== relative
  ) throw new Error("纹理预览资源不属于当前任务");
  return `pet-generation://${jobId}/${encodeURIComponent(relative)}` +
    `?previewRevision=${revision}`;
}

function showOpenDialog(options: Electron.OpenDialogOptions) {
  const owner = mainWindow ?? petWindow;
  return owner
    ? dialog.showOpenDialog(owner, options)
    : dialog.showOpenDialog(options);
}

async function resolveTextureReferences(
  modelId: string,
  capability: Awaited<ReturnType<ModelEditCapabilityStore["current"]>>,
  selections: readonly TextureReferenceSelection[],
): Promise<TextureReferenceSnapshot[]> {
  if (selections.length > 6) throw new Error("一次最多添加 6 张参考图");
  const ids = new Set<string>();
  const unitById = new Map(capability.textureUnits.map((unit) => [unit.id, unit]));
  const resolved: TextureReferenceSnapshot[] = [];
  let hasModelPreview = false;
  for (const selection of selections) {
    const id = selection.id.trim();
    if (!id || ids.has(id)) throw new Error("参考图 ID 无效或重复");
    ids.add(id);
    let imagePath: string;
    let referenceName = selection.name.trim().slice(0, 80) || `参考图 ${resolved.length + 1}`;
    if (selection.source === "model-preview") {
      if (hasModelPreview) throw new Error("当前人物正面图只能添加一次");
      hasModelPreview = true;
      await modelEditCapabilities.ensureAlignmentGuide(modelId);
      imagePath = modelStore.resolveModelFile(modelId, ".edit-capabilities/alignment-guide.png");
    } else if (selection.source === "texture-unit") {
      const textureUnitId = selection.textureUnitId ?? "";
      const unit = unitById.get(textureUnitId);
      if (!unit) throw new Error("参考纹理已经失效，请重新选择");
      referenceName = `${unit.label} 素材`;
      imagePath = selection.materialId
        ? await textureMaterialStore.resolveMaterialImage(modelId, textureUnitId, selection.materialId)
        : await textureMaterialStore.resolveActiveMaterialImage(modelId, textureUnitId);
      await modelEditCapabilities.ensureTexturePreview(
        modelId,
        path.relative(modelStore.resolveModelRoot(modelId), imagePath).split(path.sep).join("/"),
      );
    } else {
      if (!selection.path || !path.isAbsolute(selection.path)) {
        throw new Error("上传的参考图路径无效");
      }
      imagePath = selection.path;
    }
    const extension = path.extname(imagePath).toLowerCase();
    if (![".png", ".jpg", ".jpeg", ".webp"].includes(extension)) {
      throw new Error(`参考图 ${selection.name} 格式不受支持`);
    }
    await access(imagePath);
    const file = await stat(imagePath);
    if (!file.isFile() || file.size <= 0 || file.size > 25 * 1024 * 1024) {
      throw new Error(`参考图 ${selection.name} 为空或超过 25 MB`);
    }
    resolved.push({
      id,
      source: selection.source,
      name: referenceName,
      description: selection.source === "upload"
        ? selection.description.trim().slice(0, 300)
        : "",
      path: imagePath,
      promptId: selection.promptId,
      deliveryUse: selection.deliveryUse,
    });
  }
  return resolved;
}

async function resolveTextureMaterialPreview(modelId: string, textureUnitId: string): Promise<string> {
  const imagePath = await textureMaterialStore.resolveActiveMaterialImage(modelId, textureUnitId);
  await modelEditCapabilities.ensureTexturePreview(
    modelId,
    path.relative(modelStore.resolveModelRoot(modelId), imagePath).split(path.sep).join("/"),
  );
  return imagePath;
}

function visionImageDataUrl(file: string, maximumSize: number): string {
  const image = nativeImage.createFromPath(file);
  if (image.isEmpty()) throw new Error(`无法读取参考图：${path.basename(file)}`);
  const size = image.getSize();
  const scale = Math.min(1, maximumSize / Math.max(size.width, size.height));
  const preview = scale < 1
    ? image.resize({
      width: Math.max(1, Math.round(size.width * scale)),
      height: Math.max(1, Math.round(size.height * scale)),
      quality: "best",
    })
    : image;
  return preview.toDataURL();
}

async function runCharacterJob(job: CharacterGenerationJob): Promise<void> {
  if (job.input.operation !== "texture-edit") return;
  try {
    if (job.phase === "draft" && !job.paused && job.candidates.length === 0) {
      await characterService.generateCandidates(job.id);
    } else if (["decomposing", "completing"].includes(job.phase) && !job.paused) {
      const result = await characterService.processSelected(job.id);
      if (result.outputModelId) await refreshTrayModels();
    }
  } catch (error) {
    console.error("Character generation task failed:", error);
  }
}

async function refreshTrayModels(): Promise<void> {
  availableModels = await modelStore.list();
  selectedModelId = (await modelStore.selected())?.id ?? null;
  modelIdForOffset = selectedModelId;
  tray?.setContextMenu(desktopMenu());
}

function scheduleBoundsSave(kind: "pet" | "main"): void {
  const current = saveBoundsTimers[kind];
  if (current) clearTimeout(current);
  saveBoundsTimers[kind] = setTimeout(
    () => {
      delete saveBoundsTimers[kind];
      runSafely(saveWindowState(kind), "保存窗口位置");
    },
    250,
  );
}

async function saveWindowState(kind: "pet" | "main"): Promise<void> {
  if (kind === "pet") {
    await savePetWindowState();
    return;
  }
  const window = mainWindow;
  if (!window) return;
  await mainStateWriter.write(`${JSON.stringify(window.getBounds(), null, 2)}\n`);
}

async function savePetWindowState(): Promise<void> {
  const bounds = petStage?.getBounds() ?? await loadWindowState(
    petBoundsPath,
    DEFAULT_PET_BOUNDS,
    true,
  );
  await petStateWriter.write(`${JSON.stringify({
    version: 2,
    ...bounds,
    modelScale: petModelScale,
    modelOffset: petModelOffset,
  }, null, 2)}\n`);
}

function applyPetModelScale(value: number, anchor?: { x: number; y: number }): void {
  const next = normalizePetModelScale(value);
  const window = petStage;
  if (window && !window.isDestroyed() && Math.abs(next - petModelScale) > 1e-6) {
    const previousWindowScale = petWindowScale(petModelScale);
    const nextWindowScale = petWindowScale(next);
    if (Math.abs(nextWindowScale - previousWindowScale) > 1e-6) {
      const resized = resizePetBounds(window.getBounds(), petWindowSize(nextWindowScale), anchor);
      const ratio = nextWindowScale / previousWindowScale;
      petModelOffset = {
        x: petModelOffset.x * ratio,
        y: petModelOffset.y * ratio,
      };
      // Size, place and offset in one step: the panel and the model's place
      // in it reach the page together.
      const point = anchor ?? { x: resized.x + resized.width / 2, y: resized.y + resized.height / 2 };
      const placement = placePet(petModelTarget(resized), resized, null, screen.getDisplayNearestPoint(point).workArea);
      petModelOffset = placement.offset;
      window.setBounds({ ...resized, x: placement.window.x, y: placement.window.y });
    } else {
      const bounds = window.getBounds();
      const point = anchor ?? { x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2 };
      placePetWindow(window, petModelTarget(bounds), null, screen.getDisplayNearestPoint(point).workArea);
    }
  }
  petModelScale = next;
}

/** Panel position the model would have with zero offset. */
function petModelTarget(window: { x: number; y: number }): { x: number; y: number } {
  return { x: window.x + petModelOffset.x, y: window.y + petModelOffset.y };
}

function modelWithoutOffset(model: Rect): Rect {
  return { ...model, x: model.x - petModelOffset.x, y: model.y - petModelOffset.y };
}

/** Apply placePet to the live panel; returns whether anything changed. */
function placePetWindow(
  stage: PetStage,
  target: { x: number; y: number },
  model: Rect | null,
  display: Rect,
): boolean {
  const before = stage.getBounds();
  const placement = placePet(target, before, model, display);
  const moved = placement.window.x !== before.x || placement.window.y !== before.y;
  const offsetChanged = placement.offset.x !== petModelOffset.x || placement.offset.y !== petModelOffset.y;
  petModelOffset = placement.offset;
  // A move carries the new offset with it; only an offset on its own is sent apart.
  if (moved) stage.setPosition(placement.window.x, placement.window.y);
  else if (offsetChanged) sendPetModelOffsetChanged();
  return moved || offsetChanged;
}

function sendPetModelOffsetChanged(): void {
  petWindow?.webContents.send("pet:command", {
    type: "pet-model-offset-changed", offset: petModelOffset,
  });
}

function sendPetModelScaleChanged(): void {
  const command = { type: "pet-model-scale-changed" as const, scale: petModelScale };
  petWindow?.webContents.send("pet:command", command);
  mainWindow?.webContents.send("pet:command", command);
}

async function loadPetViewState(): Promise<{
  scale: number;
  offset: { x: number; y: number };
}> {
  try {
    const value = JSON.parse(await readFile(petBoundsPath, "utf8")) as PetWindowState;
    const stored = normalizePetModelScale(Number(value.modelScale));
    // Version 1 kept the panel at scale 1; version 2 panels may have drifted.
    const scale = value.version === 2
      ? normalizePetModelScale(petScaleShownBy(value, stored, petWindowScale(stored)), stored)
      : stored;
    const offset = value.modelOffset;
    return {
      scale,
      offset: offset && Number.isFinite(offset.x) && Number.isFinite(offset.y)
        && Math.abs(offset.x) <= 4_096 && Math.abs(offset.y) <= 4_096
        ? { x: offset.x, y: offset.y }
        : { x: 0, y: 0 },
    };
  } catch {
    return { scale: 1, offset: { x: 0, y: 0 } };
  }
}

async function loadMainWindowState(): Promise<WindowState> {
  try {
    await access(mainBoundsPath);
    return loadWindowState(mainBoundsPath, DEFAULT_MAIN_BOUNDS, false);
  } catch {
    return loadWindowState(legacyChatBoundsPath, DEFAULT_MAIN_BOUNDS, false);
  }
}

async function loadWindowState(
  file: string,
  defaults: WindowState,
  alignRight: boolean,
  /** A shrunken pet panel is legitimately small. */
  minimum = { width: 240, height: 320 },
): Promise<WindowState> {
  try {
    const value = JSON.parse(await readFile(file, "utf8")) as WindowState;
    if (!(value.width >= minimum.width) || !(value.height >= minimum.height)) {
      return centeredBounds(defaults, alignRight);
    }
    const point = { x: value.x ?? 0, y: value.y ?? 0 };
    const area = screen.getDisplayNearestPoint(point).workArea;
    if (
      point.x > area.x + area.width - 80 ||
      point.y > area.y + area.height - 80 ||
      point.x + value.width < area.x + 80 ||
      point.y + value.height < area.y + 80
    ) return centeredBounds(defaults, alignRight);
    return { x: value.x, y: value.y, width: value.width, height: value.height };
  } catch {
    return centeredBounds(defaults, alignRight);
  }
}

function centeredBounds(defaults: WindowState, alignRight: boolean): WindowState {
  const area = screen.getPrimaryDisplay().workArea;
  return {
    ...defaults,
    x: alignRight
      ? area.x + area.width - defaults.width - 24
      : area.x + Math.round((area.width - defaults.width) / 2),
    y: area.y + Math.round((area.height - defaults.height) / 2),
  };
}

function projectRoot(): string {
  return app.isPackaged
    ? process.resourcesPath
    : path.resolve(app.getAppPath(), "../..");
}

function perceptionHelperExecutable(): string {
  return platformPerceptionExecutable({
    packaged: app.isPackaged, resourcesPath: process.resourcesPath, projectRoot: projectRoot(),
  });
}

function decodePath(value: string): string {
  return value
    .slice(1)
    .split("/")
    .map((segment) => decodeURIComponent(segment))
    .join("/");
}

function pngDataUrl(value: Uint8Array): string {
  return `data:image/png;base64,${Buffer.from(value).toString("base64")}`;
}

function pngVisionDataUrl(value: Uint8Array, maxLongEdge: number): string {
  const source = nativeImage.createFromBuffer(Buffer.from(value));
  if (source.isEmpty()) throw new Error("视觉提示词参考图无法读取");
  const size = source.getSize();
  // Vision APIs reject images with a side under ~10px; tiny UV pieces exist.
  const minimumSide = 32;
  const scale = Math.min(maxLongEdge / Math.max(size.width, size.height),
    Math.max(1, minimumSide / Math.min(size.width, size.height)));
  if (scale === 1) return pngDataUrl(source.toPNG());
  const resized = source.resize({
    width: Math.max(1, Math.round(size.width * scale)),
    height: Math.max(1, Math.round(size.height * scale)),
    quality: "best",
  });
  return pngDataUrl(resized.toPNG());
}

function isMainView(value: unknown): value is DesktopMainView {
  return value === "chat" || value === "character" || value === "looks" || value === "models" || value === "settings" || value === "plugins";
}

function isExternalUrl(value: string): boolean {
  try {
    return new Set(["http:", "https:"]).has(new URL(value).protocol);
  } catch {
    return false;
  }
}

function isLocalChatUrl(value: string): boolean {
  const apiUrl = runtime.currentStatus().apiUrl;
  if (!apiUrl) return false;
  try {
    return new URL(value).origin === new URL(apiUrl).origin;
  } catch {
    return false;
  }
}

function showRuntimeFailure(error: unknown): void {
  console.error("Unable to start Fish Memory runtime:", error);
  runSafely(openMainWindow("settings"), "打开诊断设置");
}

function runSafely(operation: Promise<unknown>, label: string): void {
  void operation.catch((error) => reportUnexpectedError(label, error));
}

function reportUnexpectedError(label: string, error: unknown): void {
  if (quitting) return;
  const message = error instanceof Error ? error.message : String(error);
  console.error(`${label}失败:`, error);
  dialog.showErrorBox("tulpa", uiText(`${label}失败：${message}`, `The action could not be completed: ${message}`));
}
