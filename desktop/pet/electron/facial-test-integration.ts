import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import catalog from "../src/shared/facial-demo-parameters.json";
import { probeParameterEffects } from "./cubism-core-inspector";
import type { ParameterEffects } from "../src/shared/facial-conflicts";
import type { ModelPackage } from "../src/shared/contracts";
import { ipcMain, type BrowserWindow } from "electron";
import type { ModelStore } from "./model-store";
import { FacialTestService } from "./facial-test-service";
import { facialTestModelInfo } from "./facial-test-model";
import { FacialBindingStore } from "./facial-binding-store";
import type { SpeechFacialProfile } from "../src/shared/speech-facial";
import { AUTOMATIC_FACIAL_IDS, type CustomFacialParameter, type FacialDecisionRule, type FacialParameterSettings, type ManualParameterBinding } from "../src/shared/facial-bindings";

export function createFacialTest(options: {
  userData: string;
  projectRoot: string;
  resourcesPath: string;
  packaged: boolean;
  models: ModelStore;
  bindingDirectory?: string;
  /** Cubism Core, for reading what each parameter does to the drawables. */
  corePath?: string;
  window(): BrowserWindow | null;
  petWindow?(): BrowserWindow | null;
  speechConnection?(): Promise<{ apiUrl: string; apiKey: string }>;
}): FacialTestService {
  const bindings = new FacialBindingStore(options.bindingDirectory ?? options.userData);
  const service = new FacialTestService({
    ...options,
    resolveModel: async id => {
      const model = await options.models.load(id);
      return { modelId: model.id, modelName: model.name };
    },
    emit: event => {
      const window = options.window();
      if (window && !window.isDestroyed()) window.webContents.send("pet:facial-test:event", event);
    },
  });
  const trusted = (event: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent) => {
    const window = options.window();
    if (!window || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== event.sender.mainFrame) throw new Error("无效的模型测试请求");
  };
  // Per model file: parameter ID → effect, filled in as parts are added.
  const effectCache = new Map<string, ParameterEffects>();
  const parameterEffects = async (model: ModelPackage, ids: string[]): Promise<ParameterEffects> => {
    if (!options.corePath) return {};
    try {
      const entry = model.manifest.entry;
      const descriptor = JSON.parse(await readFile(options.models.resolveModelFile(model.id, entry), "utf8"));
      if (typeof descriptor?.FileReferences?.Moc !== "string") return {};
      const mocFile = options.models.resolveModelFile(model.id, path.posix.join(path.posix.dirname(entry), descriptor.FileReferences.Moc));
      const key = `${mocFile}\0${(await stat(mocFile)).mtimeMs}`;
      const known = effectCache.get(key) ?? {};
      const missing = ids.filter(id => !(id in known));
      if (missing.length) {
        const found = await probeParameterEffects(options.corePath, mocFile, missing);
        // Parameters the model lacks are remembered as having no effect.
        for (const id of missing) known[id] = found[id] ?? { shown: [], hidden: [], moved: [] };
        effectCache.set(key, known);
      }
      return Object.fromEntries(ids.map(id => [id, known[id]!]));
    } catch (error) {
      // Without effects every part simply combines, as before.
      console.warn("读取参数影响失败", error);
      return {};
    }
  };
  const prepare = async (id: string) => {
    if (typeof id !== "string" || !id || id.length > 200) throw new Error("请选择一个模型");
    const model = await options.models.load(id);
    const info = await facialTestModelInfo(model, file => readFile(options.models.resolveModelFile(id, file), "utf8"));
    const configuration = await bindings.getConfiguration(id);
    const drivable = [...new Set([
      ...catalog.filter(entry => entry.owner === "jev").map(entry => entry.id),
      ...Object.values(configuration.manualBindings).flat().map(binding => binding.parameterId),
      ...(configuration.customParameters ?? []).map(value => value.id),
      ...info.blushParameter ? [info.blushParameter] : [],
    ])];
    return { ...info, ...configuration, parameterEffects: await parameterEffects(model, drivable) };
  };
  ipcMain.handle("pet:facial-test:prepare", async (event, id: string) => {
    trusted(event);
    return prepare(id);
  });
  const trustedPet = (event: Electron.IpcMainInvokeEvent) => {
    const window = options.petWindow?.();
    if (!window || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== event.sender.mainFrame) throw new Error("无效的桌宠表情请求");
  };
  ipcMain.handle("pet:speech-facial:prepare", async (event, id: string) => {
    trustedPet(event);
    if ((await options.models.selected())?.id !== id) throw new Error("模型已切换");
    return prepare(id);
  });
  // Serialize profile changes so an old HTTP completion cannot replace a newer
  // model/configuration. This work is outside all speech/audio critical paths.
  let publication = Promise.resolve<unknown>(undefined);
  ipcMain.handle("pet:speech-facial:publish", (event, profile: SpeechFacialProfile | null) => {
    trustedPet(event);
    const next = publication.catch(() => undefined).then(async () => {
      if (profile && (typeof profile !== "object" || typeof profile.modelId !== "string"
        || !Array.isArray(profile.parameters) || profile.parameters.length > 49
        || (await options.models.selected())?.id !== profile.modelId)) throw new Error("无效的模型表情配置");
      if (!options.speechConnection) throw new Error("语音服务尚未就绪");
      const connection = await options.speechConnection();
      const response = await fetch(new URL("/v1/pet/facial-profile", connection.apiUrl), {
        method: "PUT", headers: { "Content-Type": "application/json", Authorization: `Bearer ${connection.apiKey}` },
        body: JSON.stringify(profile), signal: AbortSignal.timeout(3_000),
      });
      if (!response.ok) throw new Error(`同步桌宠表情配置失败：${response.status}`);
      return response.json();
    });
    publication = next;
    return next;
  });
  ipcMain.handle("pet:facial-test:start", (event, request) => { trusted(event); return service.start(request); });
  ipcMain.handle("pet:facial-test:stop", (event, id: string) => { trusted(event); return service.stop(id); });
  const notifyBindingChanged = (id: string) => {
    const pet = options.petWindow?.();
    if (pet && !pet.isDestroyed()) pet.webContents.send("pet:command", { type: "facial-bindings-changed", modelId: id });
  };
  ipcMain.handle("pet:facial-test:save-binding", async (event, id: string, standardId: string, binding: ManualParameterBinding | ManualParameterBinding[] | null, settings?: FacialParameterSettings, description?: string, decision?: FacialDecisionRule) => {
    trusted(event);
    if (typeof id !== "string" || !id || id.length > 200) throw new Error("请选择一个模型");
    const model = await options.models.load(id);
    const info = await facialTestModelInfo(model, file => readFile(options.models.resolveModelFile(id, file), "utf8"));
    const outputs = binding === null ? [] : Array.isArray(binding) ? binding : [binding];
    if (outputs.some(output => AUTOMATIC_FACIAL_IDS.has(output?.parameterId) || info.physicsOutputParameterIds.includes(output?.parameterId))) {
      throw new Error("此参数由呼吸或物理模拟控制，不能绑定");
    }
    await bindings.save(id, standardId, binding, settings, description, decision);
    notifyBindingChanged(id);
    return prepare(id);
  });
  ipcMain.handle("pet:facial-test:save-custom-parameter", async (event, id: string, value: CustomFacialParameter) => {
    trusted(event);
    const model = await options.models.load(id);
    const info = await facialTestModelInfo(model, file => readFile(options.models.resolveModelFile(id, file), "utf8"));
    if (AUTOMATIC_FACIAL_IDS.has(value?.id) || info.physicsOutputParameterIds.includes(value?.id)) {
      throw new Error("此参数由呼吸或物理模拟控制，不能绑定");
    }
    await bindings.saveCustom(id, value);
    notifyBindingChanged(id);
    return prepare(id);
  });
  ipcMain.handle("pet:facial-test:delete-custom-parameter", async (event, id: string, parameterId: string) => {
    trusted(event);
    await bindings.saveCustom(id, null, parameterId);
    notifyBindingChanged(id);
    return prepare(id);
  });
  return service;
}
