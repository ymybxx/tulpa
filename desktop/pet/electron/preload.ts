import { contextBridge, ipcRenderer } from "electron";

import type {
  DesktopPetApi,
  ModelReadiness,
  ModelDeleteResult,
  ModelImportResult,
  ModelPackage,
  ModelToggleCreation,
  ModelRenderBundle,
} from "../src/shared/contracts";
import { decodeAudioFrame } from "../src/shared/audio-protocol";

const api: DesktopPetApi = {
  platform: process.platform,
  getUiLocale: () => ipcRenderer.invoke("desktop:ui-locale:get"),
  setUiLocale: (locale) => ipcRenderer.invoke("desktop:ui-locale:set", locale),
  onUiLocale: (callback) => on("desktop:ui-locale", callback),
  speechFacial: {
    prepare: modelId => ipcRenderer.invoke("pet:speech-facial:prepare", modelId),
    publish: profile => ipcRenderer.invoke("pet:speech-facial:publish", profile),
  },
  facialTest: {
    prepare: (modelId) => ipcRenderer.invoke("pet:facial-test:prepare", modelId),
    start: (request) => ipcRenderer.invoke("pet:facial-test:start", request),
    stop: (id) => ipcRenderer.invoke("pet:facial-test:stop", id),
    saveBinding: (modelId, standardId, binding, settings, description, decision) => ipcRenderer.invoke("pet:facial-test:save-binding", modelId, standardId, binding, settings, description, decision),
    saveCustomParameter: (modelId, value) => ipcRenderer.invoke("pet:facial-test:save-custom-parameter", modelId, value),
    deleteCustomParameter: (modelId, id) => ipcRenderer.invoke("pet:facial-test:delete-custom-parameter", modelId, id),
    suggestParameter: (modelId, parameterId, mode, levelCount) => ipcRenderer.invoke("pet:facial-test:suggest-parameter", modelId, parameterId, mode, levelCount),
    autoSetup: (modelId, context) => ipcRenderer.invoke("pet:facial-test:auto-setup", modelId, context),
    cancelAutoSetup: () => ipcRenderer.invoke("pet:facial-test:auto-setup:cancel"),
    onAutoSetupProgress: (callback) => on("pet:facial-test:auto-setup:progress", callback),
    onEvent: (callback) => on("pet:facial-test:event", callback),
  },
  reportChatGeometry: (geometry) => ipcRenderer.send("pet:chat:geometry", geometry),
  pluginCall: (action, input = {}) => ipcRenderer.invoke("desktop:plugins:call", action, input),
  pickPlugin: (development) => ipcRenderer.invoke("desktop:plugins:pick", development),
  requestPointerRefresh: () => {
    if (process.platform !== "darwin" && process.platform !== "win32") return false;
    ipcRenderer.send("pet:pointer-refresh");
    return true;
  },
  emotionMatching: {
    start: (id, routeId) => ipcRenderer.invoke("pet:emotion-matching:start", id, routeId),
    get: (id) => ipcRenderer.invoke("pet:emotion-matching:get", id),
    cancel: (id) => ipcRenderer.invoke("pet:emotion-matching:cancel", id),
    validate: (id, manifest, adoptions) => ipcRenderer.invoke("pet:emotion-matching:validate", id, manifest, adoptions),
    onUpdate: (callback) => on("pet:emotion-matching:update", callback),
  },
  partRecognition: {
    start: (id, groupNames, unitIds) => ipcRenderer.invoke("pet:part-recognition:start", id, groupNames, unitIds),
    cancel: (id) => ipcRenderer.invoke("pet:part-recognition:cancel", id),
    onUpdate: (callback) => on("pet:part-recognition:update", callback),
  },
  listModels: () => ipcRenderer.invoke("pet:model:list") as Promise<ModelPackage[]>,
  getModelRenderBundle: (id) =>
    ipcRenderer.invoke("pet:model:render-bundle", id) as Promise<ModelRenderBundle>,
  getSelectedModelRenderBundle: () =>
    ipcRenderer.invoke("pet:model:selected-render-bundle") as Promise<ModelRenderBundle | null>,
  importModel: (kind) =>
    ipcRenderer.invoke("pet:model:import", kind) as Promise<ModelImportResult | null>,
  deleteModelAsset: (id, kind, assetId) =>
    ipcRenderer.invoke("pet:model:asset:delete", id, kind, assetId) as Promise<ModelPackage>,
  createModelToggle: (id, name, parameterValues) =>
    ipcRenderer.invoke("pet:model:toggle:create", id, name, parameterValues) as Promise<ModelToggleCreation>,
  saveModelPreview: (id, png) =>
    ipcRenderer.invoke("pet:model:preview:save", id, png) as Promise<ModelPackage>,
  outdatedModelPreviews: () => ipcRenderer.invoke("pet:model:preview:outdated") as Promise<string[]>,
  deleteModel: (id) =>
    ipcRenderer.invoke("pet:model:delete", id) as Promise<ModelDeleteResult>,
  selectModel: (id) =>
    ipcRenderer.invoke("pet:model:select", id) as Promise<ModelPackage>,
  saveManifest: (id, manifest, adoptions) =>
    ipcRenderer.invoke("pet:model:save-manifest", id, manifest, adoptions) as Promise<ModelPackage>,
  getSelectedModel: () =>
    ipcRenderer.invoke("pet:model:selected") as Promise<ModelPackage | null>,
  getModelAppearance: (id, refresh = false) =>
    ipcRenderer.invoke("pet:model:appearance:get", id, refresh),
  getModelAlignmentGuide: (id) =>
    ipcRenderer.invoke("pet:model:alignment-guide:get", id),
  selectTextureMaterial: (id, textureUnitId, materialId) =>
    ipcRenderer.invoke("pet:model:texture-materials:select", id, textureUnitId, materialId),
  selectTextureMaterials: (id, assignments) =>
    ipcRenderer.invoke("pet:model:texture-materials:select-many", id, assignments),
  deleteTextureMaterial: (id, materialId) =>
    ipcRenderer.invoke("pet:model:texture-materials:delete", id, materialId),
  previewTextureRecolor: (id, textureUnitIds, hex) =>
    ipcRenderer.invoke("pet:model:texture-recolor:preview", id, textureUnitIds, hex),
  saveTextureRecolor: (id, textureUnitIds, hex, apply) =>
    ipcRenderer.invoke("pet:model:texture-recolor:save", id, textureUnitIds, hex, apply),
  listMasterTemplates: () => ipcRenderer.invoke("character:template:list"),
  importMasterTemplate: () => ipcRenderer.invoke("character:template:import"),
  pickCharacterReference: () => ipcRenderer.invoke("character:reference:pick"),
  listCharacterJobs: () => ipcRenderer.invoke("character:job:list"),
  planCharacterTextureBatch: (templateId, textureUnitIds, imageModel) =>
    ipcRenderer.invoke("character:texture-batch:plan", templateId, textureUnitIds, imageModel),
  optimizeCharacterTexturePrompt: (input) =>
    ipcRenderer.invoke("character:texture-prompt:optimize", input),
  createCharacterJob: (input) => ipcRenderer.invoke("character:job:create", input),
  selectCharacterCandidate: (jobId, candidateId) =>
    ipcRenderer.invoke("character:job:select-candidate", jobId, candidateId),
  pauseCharacterJob: (jobId) => ipcRenderer.invoke("character:job:pause", jobId),
  resumeCharacterJob: (jobId) => ipcRenderer.invoke("character:job:resume", jobId),
  cancelCharacterJob: (jobId) => ipcRenderer.invoke("character:job:cancel", jobId),
  retryCharacterJob: (jobId) => ipcRenderer.invoke("character:job:retry", jobId),
  regenerateCharacterCandidates: (jobId) =>
    ipcRenderer.invoke("character:job:regenerate-candidates", jobId),
  approveCharacterBudget: (jobId, limitUsd) =>
    ipcRenderer.invoke("character:job:approve-budget", jobId, limitUsd),
  completeCharacterModel: (jobId, kinds) =>
    ipcRenderer.invoke("character:job:complete", jobId, kinds),
  regenerateCharacterPart: (jobId, slotId) =>
    ipcRenderer.invoke("character:job:regenerate-part", jobId, slotId),
  acceptCharacterTexture: (jobId, textureUnitId, accepted) =>
    ipcRenderer.invoke("character:job:texture-accept", jobId, textureUnitId, accepted),
  setCharacterTexturePreview: (jobId, enabled) =>
    ipcRenderer.invoke("character:job:texture-preview", jobId, enabled),
  saveAcceptedCharacterTexturesAndCancel: (jobId, apply) =>
    ipcRenderer.invoke("character:job:texture-save-and-cancel", jobId, apply),
  confirmCharacterModel: (jobId) => ipcRenderer.invoke("character:job:confirm", jobId),
  onCharacterProgress: (callback) => on("character:progress", callback),
  setMousePassthrough: (value) => ipcRenderer.send("pet:mouse-passthrough", value),
  setPetCursor: (cursor) => ipcRenderer.send("pet:cursor", cursor),
  getInteractionLocked: () => ipcRenderer.invoke("pet:interaction-lock:get"),
  setInteractionUnlockHover: (value) =>
    ipcRenderer.send("pet:interaction-unlock-hover", value),
  unlockInteraction: () => ipcRenderer.send("pet:interaction-unlock"),
  getPetModelScale: () => ipcRenderer.invoke("pet:model-scale:get"),
  getPetModelOffset: () => ipcRenderer.invoke("pet:model-offset:get"),
  getPetPanel: () => ipcRenderer.invoke("pet:panel:get"),
  onPetPanel: (callback) => on("pet:panel", callback),
  getPetTextureMaximum: () => ipcRenderer.invoke("pet:texture-maximum:get"),
  previewPetModelScale: (value, panel) =>
    ipcRenderer.send("pet:model-scale:preview", value, panel),
  setPetModelScale: (value) => ipcRenderer.invoke("pet:model-scale:set", value),
  startWindowDrag: (screenX, screenY, modelBounds) =>
    ipcRenderer.send("pet:drag:start", screenX, screenY, modelBounds),
  moveWindowDrag: (screenX, screenY) =>
    ipcRenderer.send("pet:drag:move", screenX, screenY),
  endWindowDrag: () => ipcRenderer.send("pet:drag:end"),
  showContextMenu: () => ipcRenderer.send("pet:context-menu"),
  openChat: () => ipcRenderer.send("desktop:chat:open"),
  openSettings: (section) => ipcRenderer.send("desktop:settings:open", section),
  takeSettingsSection: () => ipcRenderer.invoke("desktop:settings:section"),
  onSettingsSection: (callback) => on("desktop:settings:section", callback),
  showMainView: (view) => ipcRenderer.send("desktop:main:show", view),
  getSpeechBubbles: () => ipcRenderer.invoke("desktop:speech-bubbles:get"),
  setSpeechBubbles: (value) => ipcRenderer.invoke("desktop:speech-bubbles:set", value),
  getCurrentCharacter: () => ipcRenderer.invoke("desktop:character:current"),
  chat: {
    page: (before) => ipcRenderer.invoke("desktop:chat:page", before),
    search: (query) => ipcRenderer.invoke("desktop:chat:search", query),
    send: (requestId, content, attachmentIds) => ipcRenderer.invoke("desktop:chat:send", requestId, content, attachmentIds ?? []),
    attach: (upload) => ipcRenderer.invoke("desktop:chat:attach", upload),
    attachmentImage: (id) => ipcRenderer.invoke("desktop:chat:attachment-image", id),
    interrupt: (messageId) => ipcRenderer.invoke("desktop:chat:interrupt", messageId),
    listen: () => ipcRenderer.invoke("desktop:chat:listen"),
    newConversation: () => ipcRenderer.invoke("desktop:chat:new-conversation"),
    onNewConversation: (callback) => on("desktop:chat:new-conversation", callback),
    stopListening: () => ipcRenderer.invoke("desktop:chat:stop-listening"),
    senses: () => ipcRenderer.invoke("desktop:chat:senses"),
    senseAction: (action) => ipcRenderer.invoke("desktop:chat:sense-action", action),
    petState: () => ipcRenderer.invoke("desktop:chat:pet-state"),
    onPetEvent: (callback) => on("desktop:chat:pet-event", callback),
    onSenses: (callback) => on("desktop:chat:senses", callback),
  },
  characters: {
    list: () => ipcRenderer.invoke("desktop:characters:list"),
    bind: (id) => ipcRenderer.invoke("desktop:characters:bind", id),
    create: (input) => ipcRenderer.invoke("desktop:characters:create", input),
    profile: (id) => ipcRenderer.invoke("desktop:characters:profile", id),
    saveProfile: (id, input) => ipcRenderer.invoke("desktop:characters:save-profile", id, input),
    voice: (id) => ipcRenderer.invoke("desktop:characters:voice", id),
    saveVoice: (id, voice) => ipcRenderer.invoke("desktop:characters:save-voice", id, voice),
    previewVoice: (voice, sentence) => ipcRenderer.invoke("desktop:characters:preview-voice", voice, sentence),
    customVoices: () => ipcRenderer.invoke("desktop:characters:custom-voices"),
    cloneVoice: (recording, targetModel) => ipcRenderer.invoke("desktop:characters:clone-voice", recording, targetModel),
    memory: (id) => ipcRenderer.invoke("desktop:characters:memory", id),
    entityFacts: (id, entityId) => ipcRenderer.invoke("desktop:characters:entity-facts", id, entityId),
    forgetPreview: (id, target) => ipcRenderer.invoke("desktop:characters:forget-preview", id, target),
    forget: (id, target) => ipcRenderer.invoke("desktop:characters:forget", id, target),
  },
  getModelReadiness: () => ipcRenderer.invoke("desktop:model-readiness:get"),
  getModelImageInput: () => ipcRenderer.invoke("desktop:model-image-input:get"),
  onModelReadiness: (callback) => {
    const listener = (_event: Electron.IpcRendererEvent, value: ModelReadiness): void => callback(value);
    ipcRenderer.on("desktop:model-readiness", listener);
    return () => ipcRenderer.removeListener("desktop:model-readiness", listener);
  },
  retryModelPreparation: () => ipcRenderer.invoke("desktop:model-readiness:retry"),
  getMainWindowState: () => ipcRenderer.invoke("desktop:main:state"),
  onMainWindowState: (callback) => on("desktop:main:state", callback),
  getMainWindowVisible: () => ipcRenderer.invoke("desktop:main:visible"),
  onMainWindowVisibility: (callback) => on("desktop:main:visibility", callback),
  onPetWindowVisibility: (callback) => on("pet:visibility", callback),
  getConfiguration: () => ipcRenderer.invoke("desktop:config:get"),
  saveConfiguration: (value) => ipcRenderer.invoke("desktop:config:save", value),
  listAudioDevices: () => ipcRenderer.invoke("desktop:audio-devices"),
  getOwnerVoiceStatus: () => ipcRenderer.invoke("desktop:owner-voice:status"),
  getOwnerVoiceEnrollmentState: () => ipcRenderer.invoke("desktop:owner-voice:state"),
  enrollOwnerVoice: (append) => ipcRenderer.invoke("desktop:owner-voice:enroll", append === true),
  getOwnerVoiceClip: (clip) => ipcRenderer.invoke("desktop:owner-voice:clip", clip),
  deleteOwnerVoice: (createdAt) => ipcRenderer.invoke("desktop:owner-voice:delete", typeof createdAt === "string" ? createdAt : undefined),
  cancelOwnerVoiceEnrollment: () => ipcRenderer.invoke("desktop:owner-voice:cancel"),
  onOwnerVoiceEnrollmentState: (callback) => on("desktop:owner-voice:state", callback),
  setVoiceEnabled: (value) => ipcRenderer.invoke("desktop:voice:set-enabled", value),
  borrowMicrophone: () => ipcRenderer.invoke("desktop:microphone:borrow"),
  returnMicrophone: () => ipcRenderer.invoke("desktop:microphone:return"),
  setMicrophoneEnabled: (value) => ipcRenderer.invoke("desktop:microphone:set-enabled", value),
  setPlaybackEnabled: (value) => ipcRenderer.invoke("desktop:playback:set-enabled", value),
  getRuntimeStatus: () => ipcRenderer.invoke("desktop:runtime:status"),
  listLlmLogs: (filter) => ipcRenderer.invoke("desktop:llm-logs:list", filter),
  getUsage: () => ipcRenderer.invoke("desktop:usage"),
  getLlmLog: (id) => ipcRenderer.invoke("desktop:llm-logs:detail", id),
  getLlmLogTypes: () => ipcRenderer.invoke("desktop:llm-logs:types"),
  getCodexNotificationStatus: () => ipcRenderer.invoke("desktop:codex-notifications:status"),
  getGatewayStatus: () => ipcRenderer.invoke("desktop:gateway:status"),
  resetGatewayKey: () => ipcRenderer.invoke("desktop:gateway:reset-key"),
  testGateway: () => ipcRenderer.invoke("desktop:gateway:test"),
  getPerceptionStatus: () => ipcRenderer.invoke("desktop:perception:status"),
  openPerceptionPermissionSettings: (kind) =>
    ipcRenderer.invoke("desktop:perception:permission-settings", kind),
  onControl: (callback) => on("pet:control", callback),
  onAudio: (callback) => {
    const listener = (_event: Electron.IpcRendererEvent, raw: Uint8Array) => {
      const frame = decodeAudioFrame(raw);
      if (frame) callback(frame);
    };
    ipcRenderer.on("pet:audio", listener);
    return () => ipcRenderer.removeListener("pet:audio", listener);
  },
  onConnection: (callback) => on("pet:connection", callback),
  onCommand: (callback) => on("pet:command", callback),
  onRuntimeStatus: (callback) => on("desktop:runtime:status", callback),
  onPerceptionStatus: (callback) => on("desktop:perception:status", callback),
  onPerceptionConfiguration: (callback) => on("desktop:perception:configuration", callback),
  onboarding: {
    prepare: () => ipcRenderer.invoke("desktop:onboarding:prepare"),
    onPreparation: (callback) => on("desktop:onboarding:preparation", callback),
    checkAiService: (request) => ipcRenderer.invoke("desktop:onboarding:check-ai-service", request),
    checkJev: (apiKey) => ipcRenderer.invoke("desktop:onboarding:check-jev", apiKey),
    checkBailianDecision: (workspaceId, apiKey) => ipcRenderer.invoke("desktop:onboarding:check-bailian-decision", workspaceId, apiKey),
    saveCharacter: (character) => ipcRenderer.invoke("desktop:onboarding:save-character", character),
    permissionStatus: () => ipcRenderer.invoke("desktop:onboarding:permission-status"),
    requestPermission: (kind) => ipcRenderer.invoke("desktop:onboarding:request-permission", kind),
    warmUpCapture: () => ipcRenderer.invoke("desktop:onboarding:warm-up-capture"),
    settingsAppName: () => ipcRenderer.invoke("desktop:onboarding:settings-app-name"),
    loadDraft: () => ipcRenderer.invoke("desktop:onboarding:load-draft"),
    saveDraft: (draft) => ipcRenderer.invoke("desktop:onboarding:save-draft", draft),
  },
};

contextBridge.exposeInMainWorld("desktopPet", api);

function on<T>(channel: string, callback: (value: T) => void): () => void {
  const listener = (_event: Electron.IpcRendererEvent, value: T) => callback(value);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}
