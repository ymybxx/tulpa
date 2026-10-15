import { CubismModelInstance } from "../renderer/cubism-model-instance";
import { CubismRuntimeHost } from "../renderer/cubism-runtime-host";
import { Live2DBehaviorPlayback } from "../renderer/live2d-behavior-playback";
import { CubismWebGLOffscreenManager } from "@cubism/rendering/cubismoffscreenmanager";
import type { ModelRenderBundle } from "../shared/contracts";
import type { EmotionAsset, EmotionFrame, EmotionSample, EmotionSampleMode } from "../shared/emotion-matching";
import type { FacialParameterSample, FacialParameterSurvey } from "../shared/facial-test";
import { changedRegion, sideBySide } from "./parameter-region";

/** Manual simulation has no RAF, visibility dependency, or always-running render loop. */
export class BehaviorSampleSession {
  private model: CubismModelInstance | null = null;
  private readonly gl: WebGL2RenderingContext;
  private disposed = false;

  constructor(private readonly canvas: HTMLCanvasElement) {
    canvas.width = 512;
    canvas.height = 768;
    this.gl = CubismRuntimeHost.createContext(canvas, true);
  }

  async load(bundle: ModelRenderBundle): Promise<void> {
    this.model?.release();
    const input = structuredClone(bundle);
    input.model.manifest.display = { scale: 0.95, x: 0.5, y: 0.9 };
    this.model = await CubismModelInstance.load(this.gl, input, {
      cssWidth: this.canvas.width, cssHeight: this.canvas.height,
      pixelWidth: this.canvas.width, pixelHeight: this.canvas.height,
    }, { sampling: true, maxTextureDimension: 2048 });
  }

  async sample(asset: EmotionAsset, mode: EmotionSampleMode): Promise<EmotionSample> {
    const model = this.model;
    if (!model || this.disposed) throw new Error("采样模型未加载");
    const started = performance.now();
    await model.prepareSample(asset.candidate, mode === "asset" ? "asset" : "runtime");
    const frames: EmotionFrame[] = [this.capture(`${asset.id}:baseline`, 0)];
    const playback = mode === "asset" ? null : new Live2DBehaviorPlayback(() => model, (message, error) => {
      throw error instanceof Error ? error : new Error(message);
    });
    if (mode === "speaking") {
      model.setSpeaking(true);
      playback!.setSpeaking(true);
    }
    if (playback) {
      if (mode === "state") playback.setState(asset.candidate);
      else if (!playback.playTransient(asset.candidate, "emotion-sample")) throw new Error("组合无法播放");
    } else model.startBehavior(asset.candidate, { loopMotions: false });
    const faceExpressionIds = asset.candidate.expressionAssetIds.filter((id) => model.isFaceOnlyExpressionAsset(id));
    const times = [...new Set(asset.timestampsMs.map((time) => Math.max(1, Math.round(time / (1000 / 60)))))]
      .sort((a, b) => a - b);
    const last = times.at(-1) ?? 1;
    if (last > 60 * 120 || times.length > 16) throw new Error("采样计划超出单素材限制");
    let target = 0;
    try {
      for (let tick = 1; tick <= last; tick++) {
        if (this.disposed || this.gl.isContextLost()) throw new Error("采样已取消或图形上下文丢失");
        if (mode === "speaking") {
          const volume = 0.25 + 0.2 * Math.sin(tick / 5);
          model.setLipShape({ volume, a: volume, e: 0, i: 0, o: 0, u: 0 });
        }
        model.update(1 / 60);
        if (tick === times[target]) {
          frames.push(this.capture(`${asset.id}:${target}`, tick * 1000 / 60));
          target++;
        }
        model.flushSampleCompletion();
        // Yield between small CPU chunks; never monopolize a core or queue all images.
        if (tick % 8 === 0) await new Promise((resolve) => setTimeout(resolve, 8));
      }
      return { asset, mode, frames, faceExpressionIds, elapsedMs: performance.now() - started };
    } finally {
      playback?.destroy();
      model.stopBehavior();
      model.stopPersistentExpressions();
    }
  }

  async sampleParameter(parameterId: string, mode: "score" | "binary", levelCount: number): Promise<FacialParameterSample> {
    const model = this.model;
    if (!model || this.disposed) throw new Error("采样模型未加载");
    const parameter = model.nativeParameters().find(item => item.id === parameterId);
    if (!parameter || !Number.isFinite(parameter.minimum) || !Number.isFinite(parameter.maximum)
      || parameter.minimum >= parameter.maximum || !Number.isFinite(parameter.defaultValue)
      || mode !== "score" && mode !== "binary" || !Number.isInteger(levelCount) || levelCount < 2 || levelCount > 10) {
      throw new Error("参数采样请求无效");
    }
    // Small, user-defined facial controls are hard to see in the regular
    // behavior sampler's full-body 512px view. Increase only this on-demand pass.
    this.canvas.width = 1024;
    this.canvas.height = 1536;
    model.resize({ cssWidth: 1024, cssHeight: 1536, pixelWidth: 1024, pixelHeight: 1536 });
    // Capture the exact native value of rubric levels so every screenshot maps
    // to a level the analysis model must describe. Long rubrics keep at most six
    // evenly spread levels (always including both ends) to bound request size.
    const count = mode === "binary" ? 2 : levelCount;
    const anchor = (index: number) => index === 0 ? parameter.minimum : index === count - 1 ? parameter.maximum
      : parameter.minimum + (parameter.maximum - parameter.minimum) * index / (count - 1);
    const picked = count <= 6 ? Array.from({ length: count }, (_, index) => index)
      : Array.from({ length: 6 }, (_, index) => Math.round(index * (count - 1) / 5));
    const targets = [...new Set([parameter.defaultValue, ...picked.map(anchor)])];
    const shots: Array<{ value: number; shot: HTMLCanvasElement }> = [];
    const empty = { id: "parameter-sample", motionAssetIds: [], expressionAssetIds: [] };
    for (const value of targets) {
      if (this.disposed || this.gl.isContextLost()) throw new Error("参数采样已取消");
      model.endFacialDemo();
      await model.prepareSample(empty, "asset");
      model.beginFacialDemo({ mouth: null, eyes: [] });
      model.setFacialDemoTargets({ [parameterId]: value });
      for (let tick = 0; tick < 75; tick++) {
        if (this.disposed || this.gl.isContextLost()) throw new Error("参数采样已取消");
        model.update(1 / 60);
        if (tick % 15 === 14) await new Promise(resolve => setTimeout(resolve, 0));
      }
      shots.push({ value, shot: this.snapshot() });
    }
    model.endFacialDemo();
    // Enlarge where this parameter actually changes the picture, wherever that is.
    const detail = changedRegion(shots.map(item => item.shot), this.canvas.width, this.canvas.height);
    const frames: FacialParameterSample["frames"] = shots.map(({ value, shot }) => ({
      kind: value === parameter.defaultValue ? "default" : "target", value, image: sideBySide(shot, detail) }));
    return { parameter, frames };
  }

  /**
   * Moves each parameter alone to both ends at a small size and reports how
   * much of the character changes and where. Fast enough for every parameter
   * of a model; no images leave this page.
   */
  async surveyParameters(requested: readonly string[] | null): Promise<FacialParameterSurvey[]> {
    const model = this.model;
    if (!model || this.disposed) throw new Error("采样模型未加载");
    await model.prepareSample({ id: "parameter-survey", motionAssetIds: [], expressionAssetIds: [] }, "asset");
    const width = 256, height = 384;
    this.canvas.width = width; this.canvas.height = height;
    model.resize({ cssWidth: width, cssHeight: height, pixelWidth: width, pixelHeight: height });
    const pixels = new Uint8Array(width * height * 4);
    const render = (id: string | null, value = 0): Uint8Array => {
      if (id) model.probeParameter(id, value); else model.updateEditingPreview();
      this.drawFrame();
      this.gl.readPixels(0, 0, width, height, this.gl.RGBA, this.gl.UNSIGNED_BYTE, pixels);
      return pixels.slice();
    };
    const base = render(null);
    // The character's own bounds, from pixels that differ from the clear color.
    const background = [base[0]!, base[1]!, base[2]!];
    let left = width, top = height, right = -1, bottom = -1, area = 0;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const at = (y * width + x) * 4;
      if (Math.abs(base[at]! - background[0]!) + Math.abs(base[at + 1]! - background[1]!) + Math.abs(base[at + 2]! - background[2]!) <= 24) continue;
      area++; left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y);
    }
    if (right < 0) throw new Error("模型画面为空");
    const parameters = new Map(model.nativeParameters().map(parameter => [parameter.id, parameter]));
    const parameterIds = requested ?? [...parameters.keys()];
    const results: FacialParameterSurvey[] = [];
    for (const [index, id] of parameterIds.entries()) {
      if (this.disposed || this.gl.isContextLost()) throw new Error("参数采样已取消");
      const parameter = parameters.get(id);
      if (!parameter) throw new Error(`模型没有参数 ${id}`);
      if (!(parameter.maximum > parameter.minimum)) { results.push({ id, parameter, changed: 0, box: null }); continue; }
      let changed = 0, x0 = width, y0 = height, x1 = -1, y1 = -1;
      for (const value of [parameter.minimum, parameter.maximum]) {
        if (value === parameter.defaultValue) continue;
        const frame = render(id, value);
        let count = 0;
        for (let at = 0; at < frame.length; at += 4) {
          if (Math.abs(frame[at]! - base[at]!) + Math.abs(frame[at + 1]! - base[at + 1]!) + Math.abs(frame[at + 2]! - base[at + 2]!) <= 36) continue;
          count++;
          const p = at / 4, x = p % width, y = height - 1 - Math.floor(p / width);
          x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        }
        changed = Math.max(changed, count / area);
      }
      // Readback is bottom-up; the bounds above are flipped to top-down like the character's.
      const flipTop = height - 1 - bottom, flipBottom = height - 1 - top;
      // Probes are deterministic, so any changed pixel is real; a brow in a full-body view is ~0.1%.
      results.push({ id, parameter, changed, box: x1 < 0 || changed < 0.0001 ? null : {
        x: (x0 - left) / (right - left + 1), y: (y0 - flipTop) / (flipBottom - flipTop + 1),
        width: (x1 - x0 + 1) / (right - left + 1), height: (y1 - y0 + 1) / (flipBottom - flipTop + 1) } });
      if (index % 12 === 11) await new Promise(resolve => setTimeout(resolve, 0));
    }
    model.updateEditingPreview();
    return results;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    try { this.model?.release(); } finally {
      this.model = null;
      CubismWebGLOffscreenManager.getInstance().removeContext(this.gl);
      this.gl.getExtension("WEBGL_lose_context")?.loseContext();
    }
  }

  private capture(id: string, timeMs: number): EmotionFrame {
    this.drawFrame();
    return { id, timeMs, image: this.canvas.toDataURL("image/jpeg", 0.88) };
  }

  private snapshot(): HTMLCanvasElement {
    this.drawFrame();
    const shot = document.createElement("canvas");
    shot.width = this.canvas.width; shot.height = this.canvas.height;
    const context = shot.getContext("2d");
    if (!context) throw new Error("无法生成参数对照画面");
    context.drawImage(this.canvas, 0, 0);
    return shot;
  }

  private drawFrame(): void {
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.gl.clearColor(0.9, 0.92, 0.91, 1);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.STENCIL_BUFFER_BIT);
    this.model!.draw();
    if (this.gl.isContextLost() || this.gl.getError() !== this.gl.NO_ERROR) throw new Error("采样绘制失败");
  }
}
