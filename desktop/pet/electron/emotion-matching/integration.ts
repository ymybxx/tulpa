import path from "node:path";
import { readFile } from "node:fs/promises";
import { ipcMain, Notification, type BrowserWindow } from "electron";
import type { AppConfigStore } from "../app-config";
import type { ModelStore } from "../model-store";
import type { ModelRenderBundle, PetManifest } from "../../src/shared/contracts";
import { EMOTION_MATCHING_CHANNEL, type EmotionAdoption, type EmotionMatchingJob } from "../../src/shared/emotion-matching";
import { textModel } from "../model-configuration";
import standardParameters from "../../src/shared/facial-demo-parameters.json";
import { facialTestModelInfo } from "../facial-test-model";
import { suggestFacialParameter } from "../facial-parameter-suggestion";
import { displayGroups, physicsLinks, runAutoSetup } from "../facial-auto-setup";
import { AUTOMATIC_FACIAL_IDS } from "../../src/shared/facial-bindings";
import type { FacialAutoSetupContext } from "../../src/shared/facial-test";
import { EmotionAssetReader } from "./asset-analysis";
import { EmotionSamplingHost } from "./sampling-host";
import { EmotionMatchingStore } from "./store";
import { EmotionMatchingService } from "./service";

export function createEmotionMatching(options: {
  userData: string;
  models: ModelStore;
  config: AppConfigStore;
  bundle(id: string): Promise<ModelRenderBundle>;
  window(): BrowserWindow | null;
  /** Called for every job update, after the renderer has been told. */
  onUpdate?(job: EmotionMatchingJob): void;
}): EmotionMatchingService {
  const reader = new EmotionAssetReader((id, file) => options.models.resolveModelFile(id, file));
  const notified = new Set<string>();
  const route = async (id?: string) => {
    const [configuration, secrets] = await Promise.all([options.config.load(), options.config.runtimeSecrets()]);
    const selected = id ? configuration.models.text.find((m) => m.id === id) : textModel(configuration, "reasoning");
    if (!selected) throw new Error("请选择已配置且支持图片的分析模型");
    const apiKey = secrets.modelApiKeys?.[selected.id];
    if (!apiKey) throw new Error("请先配置分析模型的 API Key");
    return { ...selected, apiKey };
  };
  const service = new EmotionMatchingService(new EmotionMatchingStore(path.join(options.userData, "emotion-matching")), {
    source: async (id) => reader.source(await options.bundle(id)),
    route,
    sampler: () => new EmotionSamplingHost(),
    updated: (job) => {
      const window = options.window();
      if (window && !window.isDestroyed()) window.webContents.send(`${EMOTION_MATCHING_CHANNEL}:update`, job);
      options.onUpdate?.(job);
      if (["completed", "partial", "failed", "stale"].includes(job.status) && !notified.has(job.id)) {
        if (notified.size > 100) notified.clear();
        notified.add(job.id);
        if (Notification.isSupported()) new Notification({ title: `${job.modelName} · 状态与情绪匹配`, body: job.message, silent: true }).show();
      }
    },
  });
  const trusted = (event: Electron.IpcMainInvokeEvent) => {
    const window = options.window();
    if (!window || event.sender !== window.webContents || event.senderFrame !== event.sender.mainFrame) throw new Error("无效的模型管理请求");
  };
  ipcMain.handle(`${EMOTION_MATCHING_CHANNEL}:start`, (event, modelId: string, routeId?: string) => { trusted(event); return service.start(modelId, routeId); });
  ipcMain.handle(`${EMOTION_MATCHING_CHANNEL}:get`, (event, modelId: string) => { trusted(event); return service.get(modelId); });
  ipcMain.handle(`${EMOTION_MATCHING_CHANNEL}:cancel`, (event, jobId: string) => { trusted(event); return service.cancel(jobId); });
  ipcMain.handle(`${EMOTION_MATCHING_CHANNEL}:validate`, (event, modelId: string, manifest: PetManifest, adoptions: EmotionAdoption[]) => {
    trusted(event); return service.validate(modelId, manifest, adoptions);
  });
  let suggestionActive = false;
  ipcMain.handle("pet:facial-test:suggest-parameter", async (event, modelId: string, parameterId: string,
    mode: "score" | "binary", levelCount: number) => {
    trusted(event);
    if (suggestionActive) throw new Error("已有参数正在分析，请稍候");
    if (typeof modelId !== "string" || !modelId || modelId.length > 200
      || typeof parameterId !== "string" || !parameterId || parameterId.length > 100
      || standardParameters.some(parameter => parameter.id === parameterId)
      || mode !== "score" && mode !== "binary"
      || !Number.isInteger(levelCount) || levelCount < 2 || levelCount > 10) throw new Error("参数识别请求无效");
    suggestionActive = true;
    const controller = new AbortController();
    const window = options.window()!;
    const abort = () => controller.abort();
    window.once("closed", abort);
    const sampler = new EmotionSamplingHost();
    try {
      const model = await options.models.load(modelId);
      const info = await facialTestModelInfo(model, file => readFile(options.models.resolveModelFile(modelId, file), "utf8"));
      if (info.physicsOutputParameterIds.includes(parameterId)) throw new Error("物理输出参数不能交给决策模型");
      const analysisRoute = await route();
      controller.signal.throwIfAborted();
      await sampler.load(await options.bundle(modelId), controller.signal);
      const sample = await sampler.sampleParameter(parameterId, mode, levelCount, controller.signal);
      return await suggestFacialParameter(sample, mode, levelCount, analysisRoute, controller.signal);
    } finally {
      window.removeListener("closed", abort);
      await sampler.dispose();
      suggestionActive = false;
    }
  });
  let autoSetup: AbortController | null = null;
  ipcMain.handle("pet:facial-test:auto-setup", async (event, modelId: string, context: FacialAutoSetupContext) => {
    trusted(event);
    if (autoSetup || suggestionActive) throw new Error("已有参数正在分析，请稍候");
    const strings = (value: unknown) => Array.isArray(value) && value.length <= 1000 && value.every(item => typeof item === "string" && item.length <= 200);
    if (typeof modelId !== "string" || !modelId || modelId.length > 200 || !context || typeof context !== "object"
      || !strings(context.used) || !strings(context.open)
      || !Number.isInteger(context.customSlots) || context.customSlots < 0 || context.customSlots > 16) throw new Error("一键配置请求无效");
    const controller = autoSetup = new AbortController();
    const window = options.window()!;
    const abort = () => controller.abort();
    window.once("closed", abort);
    const sampler = new EmotionSamplingHost();
    try {
      const model = await options.models.load(modelId);
      const readAsset = (file: string) => readFile(options.models.resolveModelFile(modelId, file), "utf8");
      const info = await facialTestModelInfo(model, readAsset);
      const analysisRoute = await route();
      const bundle = await options.bundle(modelId);
      await sampler.load(bundle, controller.signal);
      return await runAutoSetup({
        context, sampler, route: analysisRoute, signal: controller.signal,
        groups: await displayGroups(model.manifest.entry, readAsset),
        links: await physicsLinks(model.manifest.entry, readAsset),
        blocked: new Set([...info.physicsOutputParameterIds, ...AUTOMATIC_FACIAL_IDS]),
        progress: progress => { if (!window.isDestroyed()) window.webContents.send("pet:facial-test:auto-setup:progress", progress); },
      });
    } catch (error) {
      if (controller.signal.aborted) throw new Error("已取消");
      throw error;
    } finally {
      window.removeListener("closed", abort);
      await sampler.dispose();
      if (autoSetup === controller) autoSetup = null;
    }
  });
  ipcMain.handle("pet:facial-test:auto-setup:cancel", (event) => { trusted(event); autoSetup?.abort(); });
  return service;
}
