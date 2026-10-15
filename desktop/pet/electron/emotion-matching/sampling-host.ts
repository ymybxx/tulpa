import { randomUUID } from "node:crypto";
import path from "node:path";
import { BrowserWindow, ipcMain } from "electron";
import type { ModelRenderBundle } from "../../src/shared/contracts";
import type { EmotionAsset, EmotionSample, EmotionSampleMode, EmotionSamplerRequest, EmotionSamplerResponse } from "../../src/shared/emotion-matching";
import type { FacialParameterSample, FacialParameterSurvey } from "../../src/shared/facial-test";

export interface EmotionSampler {
  load(bundle: ModelRenderBundle, signal: AbortSignal): Promise<void>;
  sample(asset: EmotionAsset, mode: EmotionSampleMode, signal: AbortSignal): Promise<EmotionSample>;
  dispose(): Promise<void>;
}

/** One disposable Chromium renderer per active task, independent of the management UI. */
export class EmotionSamplingHost implements EmotionSampler {
  private window: BrowserWindow | null = null;

  async load(bundle: ModelRenderBundle, signal: AbortSignal): Promise<void> {
    signal.throwIfAborted();
    const window = new BrowserWindow({
      width: 512, height: 768, show: false, skipTaskbar: true, focusable: false,
      webPreferences: {
        preload: path.join(__dirname, "../emotion-sampler-preload.js"),
        contextIsolation: true, nodeIntegration: false, sandbox: true,
        backgroundThrottling: false,
      },
    });
    this.window = window;
    window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    const abort = () => { if (!window.isDestroyed()) window.destroy(); };
    signal.addEventListener("abort", abort, { once: true });
    window.once("closed", () => signal.removeEventListener("abort", abort));
    const developmentUrl = process.env.VITE_DEV_SERVER_URL;
    if (developmentUrl) await window.loadURL(new URL("emotion-sampler.html", `${developmentUrl}/`).href);
    else await window.loadFile(path.join(__dirname, "../../../dist-renderer/emotion-sampler.html"));
    window.webContents.on("will-navigate", (event) => event.preventDefault());
    await this.request({ id: randomUUID(), operation: "load", bundle }, signal);
  }

  async sample(asset: EmotionAsset, mode: EmotionSampleMode, signal: AbortSignal): Promise<EmotionSample> {
    const response = await this.request({ id: randomUUID(), operation: "sample", asset, mode }, signal);
    if (!response.sample || response.sample.asset.id !== asset.id || response.sample.mode !== mode ||
      !Array.isArray(response.sample.frames) || response.sample.frames.length > 17 ||
      response.sample.frames.some((frame) => typeof frame.image !== "string" || !frame.image.startsWith("data:image/jpeg;base64,") || frame.image.length > 2_000_000)) {
      throw new Error("采样返回了无效图像");
    }
    return response.sample;
  }

  async sampleParameter(parameterId: string, mode: "score" | "binary", levelCount: number, signal: AbortSignal): Promise<FacialParameterSample> {
    const response = await this.request({ id: randomUUID(), operation: "sample-parameter", parameterId, mode, levelCount }, signal);
    const sample = response.parameterSample;
    if (!sample || sample.parameter?.id !== parameterId || !Array.isArray(sample.frames)
      || sample.frames.length < 2 || sample.frames.length > 7
      || sample.frames.some(frame => !Number.isFinite(frame.value)
        || !["default", "target"].includes(frame.kind)
        || typeof frame.image !== "string" || !frame.image.startsWith("data:image/jpeg;base64,")
        || frame.image.length > 2_000_000)) throw new Error("参数采样返回无效画面");
    return sample;
  }

  /** null surveys every parameter of the loaded model. */
  async surveyParameters(parameterIds: readonly string[] | null, signal: AbortSignal): Promise<FacialParameterSurvey[]> {
    const response = await this.request({ id: randomUUID(), operation: "survey-parameters", parameterIds: parameterIds ? [...parameterIds] : null }, signal);
    const survey = response.survey;
    if (!Array.isArray(survey) || (parameterIds && survey.length !== parameterIds.length)
      || survey.some((item, index) => (parameterIds ? item?.id !== parameterIds[index] : typeof item?.id !== "string")
        || item.parameter?.id !== item.id || !Number.isFinite(item.changed))) throw new Error("参数扫描返回无效结果");
    return survey;
  }

  async dispose(): Promise<void> {
    const window = this.window;
    if (!window || window.isDestroyed()) return;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 1000);
      try { await this.request({ id: randomUUID(), operation: "dispose" }, controller.signal); }
      finally { clearTimeout(timer); }
    } catch { /* Destroying the dedicated process is also the cancellation fallback. */ }
    finally {
      if (!window.isDestroyed()) window.destroy();
      this.window = null;
    }
  }

  private request(request: EmotionSamplerRequest, signal: AbortSignal): Promise<EmotionSamplerResponse> {
    signal.throwIfAborted();
    const window = this.window;
    if (!window || window.isDestroyed()) return Promise.reject(new Error("采样窗口已关闭"));
    const contents = window.webContents;
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = (error?: Error, response?: EmotionSamplerResponse) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        ipcMain.removeListener("emotion-sampler:response", receive);
        signal.removeEventListener("abort", abort);
        window.removeListener("closed", closed);
        contents.removeListener("render-process-gone", crashed);
        error ? reject(error) : resolve(response!);
      };
      const receive = (event: Electron.IpcMainEvent, response: EmotionSamplerResponse) => {
        if (event.sender !== contents || !response || response.id !== request.id) return;
        finish(response.error ? new Error(response.error) : undefined, response);
      };
      const abort = () => finish(new Error("分析已取消"));
      const closed = () => finish(new Error("采样窗口已关闭"));
      const crashed = () => finish(new Error("采样进程意外退出"));
      const timer = setTimeout(() => finish(new Error("模型采样超时")), 90_000);
      ipcMain.on("emotion-sampler:response", receive);
      signal.addEventListener("abort", abort, { once: true });
      window.once("closed", closed);
      contents.once("render-process-gone", crashed);
      contents.send("emotion-sampler:request", request);
    });
  }
}
