import { randomUUID } from "node:crypto";
import type { PetManifest } from "../../src/shared/contracts";
import { normalizeDefaultAppearance } from "../../src/shared/default-appearance";
import { targetCandidates, targetId, targetUsesStatePlayback } from "../../src/shared/behavior-target";
import { candidateSignature, emotionJobActive, EMOTION_ANALYSIS_VERSION,
  type EmotionAdoption, type EmotionAssessment, type EmotionAsset, type EmotionMatchingJob,
  type EmotionSample, type EmotionSuggestion } from "../../src/shared/emotion-matching";
import { digest, type EmotionSource } from "./asset-analysis";
import { EmotionAnalysisClient, EmotionAnalysisError, type EmotionModelRoute } from "./client";
import type { EmotionSampler } from "./sampling-host";
import { EmotionMatchingStore } from "./store";
import { abortable } from "./abortable";
import { stateValidationAsset } from "./validation-plan";

interface Run { job: EmotionMatchingJob; controller: AbortController; route: EmotionModelRoute; previous: EmotionMatchingJob | null }
export interface EmotionMatchingDependencies {
  source(modelId: string): Promise<EmotionSource>;
  route(id?: string): Promise<EmotionModelRoute>;
  sampler(): EmotionSampler;
  client?(route: EmotionModelRoute): Pick<EmotionAnalysisClient, "analyze">;
  updated(job: EmotionMatchingJob): void;
}

/** UI-independent, bounded queue. No work, timers, renderer or network exist until start(). */
export class EmotionMatchingService {
  private jobs = new Map<string, EmotionMatchingJob>();
  private runs = new Map<string, Run>();
  private queue: Run[] = [];
  private draining: Promise<void> | null = null;
  private closed = false;

  constructor(private readonly store: EmotionMatchingStore, private readonly deps: EmotionMatchingDependencies) {}

  async get(modelId: string): Promise<EmotionMatchingJob | null> {
    assertModelId(modelId);
    const stored = this.jobs.get(modelId) ?? await this.store.job(modelId);
    // A start/update may have arrived while the disk snapshot was being read.
    const job = this.jobs.get(modelId) ?? stored;
    if (job) this.jobs.set(modelId, job);
    return job ? structuredClone(job) : null;
  }

  async start(modelId: string, analysisModelId?: string): Promise<EmotionMatchingJob> {
    assertModelId(modelId);
    if (this.closed) throw new Error("应用正在退出");
    const active = this.jobs.get(modelId);
    if (active && emotionJobActive(active)) return structuredClone(active);
    if (this.runs.size >= 4) throw new Error("已有四个模型等待分析，请稍后重试");
    const route = await this.deps.route(analysisModelId);
    const previous = await this.get(modelId);
    // Recheck after async configuration lookup so rapid double-clicks cannot duplicate requests.
    const existing = this.jobs.get(modelId);
    if (existing && emotionJobActive(existing)) return structuredClone(existing);
    if (this.closed || this.runs.size >= 4) throw new Error("分析队列暂不可用");
    const now = new Date().toISOString();
    const job: EmotionMatchingJob = {
      id: randomUUID(), modelId, modelName: modelId, analysisModelId: route.id, analysisModelName: route.name,
      sourceFingerprint: "", analysisModelFingerprint: digest([route.baseUrl, route.model, EMOTION_ANALYSIS_VERSION]), revision: 0, status: "queued", phase: "preparing", totalAssets: 0,
      analysisVersion: EMOTION_ANALYSIS_VERSION,
      processedAssets: 0, cachedAssets: 0, failures: [], suggestions: [],
      message: this.runs.size ? "已排队，等待前一个模型分析完成" : "正在准备后台分析", createdAt: now, updatedAt: now,
    };
    const run = { job, route, controller: new AbortController(), previous };
    this.jobs.set(modelId, job);
    this.runs.set(job.id, run);
    this.queue.push(run);
    try { await this.publish(job); }
    catch (error) {
      this.runs.delete(job.id);
      this.queue = this.queue.filter((item) => item !== run);
      job.status = "failed";
      job.message = "无法保存分析任务，请检查本地存储";
      throw error;
    }
    this.drain();
    return structuredClone(job);
  }

  async cancel(jobId: string): Promise<void> {
    const run = this.runs.get(jobId);
    if (!run || !emotionJobActive(run.job)) return;
    run.controller.abort();
    run.job.status = "cancelled";
    run.job.message = "已取消；已验证的建议仍可查看和采纳";
    await this.publish(run.job);
  }

  sourceChanged(modelId: string): void {
    for (const run of this.runs.values()) {
      if (run.job.modelId !== modelId || !emotionJobActive(run.job)) continue;
      run.job.status = "stale";
      run.job.message = "模型素材或外观已改变，请重新分析；已有分析缓存会保留";
      run.controller.abort();
      void this.publish(run.job).catch((error) => console.warn("保存分析状态失败", error instanceof Error ? error.message : "write failed"));
    }
  }

  async shutdown(): Promise<void> {
    this.closed = true;
    for (const run of this.runs.values()) {
      const active = emotionJobActive(run.job);
      if (active) {
        run.job.status = "interrupted";
        run.job.message = "应用退出，未完成素材可继续分析";
      }
      run.controller.abort();
      if (active) await this.publish(run.job);
    }
    await this.draining;
  }

  async validate(modelId: string, manifest: PetManifest, adoptions: EmotionAdoption[]): Promise<void> {
    if (!adoptions.length) return;
    if (adoptions.length > 50 || manifest.id !== modelId) throw new Error("采纳配置无效");
    const source = await this.deps.source(modelId);
    const job = await this.get(modelId);
    const hashes = new Map(source.assets.map((asset) => [asset.id, asset.contentHash]));
    for (const adoption of adoptions) {
      const suggestion = job?.id === adoption.jobId ? job.suggestions.find((s) => s.id === adoption.suggestionId) : undefined;
      if (!suggestion) throw new Error("这份建议已更新，请重新查看建议");
      const candidate = targetCandidates(manifest, suggestion.target).find((c) => c.id === adoption.candidateId);
      if (!candidate) {
        const elsewhere = [...Object.values(manifest.behaviors.states), ...Object.values(manifest.behaviors.emotions)]
          .some((values) => values.some((c) => c.id === adoption.candidateId));
        if (elsewhere) throw new Error("建议的状态或情绪目标已修改，请重新验证或通过手工候选编辑并预览");
        continue; // The user deliberately removed this adopted draft candidate.
      }
      if (source.fingerprint !== suggestion.sourceFingerprint ||
        Object.entries(suggestion.assetHashes).some(([id, hash]) => hashes.get(id) !== hash)) throw new Error("模型素材或外观已改变，请重新分析后采纳");
      if (digest(normalizeDefaultAppearance(manifest.defaultAppearance, manifest.expressionAssets)) !==
          digest(normalizeDefaultAppearance(source.bundle.model.manifest.defaultAppearance, source.bundle.model.manifest.expressionAssets))) {
        throw new Error("默认外观已修改，请保存外观后重新分析，或通过手工候选编辑并联合预览后保存");
      }
      if (candidateSignature(candidate) !== candidateSignature(suggestion.candidate) || digest(manifest.lipsync) !== digest(suggestion.lipsync)) {
        throw new Error("候选组合或口型已修改，请重新验证建议，或通过手工候选编辑并联合预览后保存");
      }
    }
  }

  async adopted(modelId: string, adoptions: EmotionAdoption[], manifest: PetManifest): Promise<void> {
    const job = this.jobs.get(modelId);
    if (!job) return;
    for (const ref of adoptions) {
      if (ref.jobId !== job.id) continue;
      const suggestion = job.suggestions.find((s) => s.id === ref.suggestionId);
      if (suggestion && targetCandidates(manifest, suggestion.target).some((c) => c.id === ref.candidateId)) suggestion.adopted = true;
    }
    await this.publish(job);
  }

  private drain(): void {
    if (this.draining) return;
    this.draining = (async () => {
      while (this.queue.length) {
        const run = this.queue.shift()!;
        try {
          if (!run.controller.signal.aborted && !this.closed) await this.execute(run);
        } catch {
          // A failed result write must not strand other queued models or crash the app.
          run.job.status = "failed";
          run.job.message = "无法保存分析结果，请检查本地存储后重试";
          if (this.jobs.get(run.job.modelId)?.id === run.job.id) this.deps.updated(structuredClone(run.job));
        } finally { this.runs.delete(run.job.id); }
      }
    })().finally(() => { this.draining = null; if (this.queue.length) this.drain(); });
  }

  private async execute(run: Run): Promise<void> {
    const { job, route, controller } = run;
    const signal = controller.signal;
    const sampler = this.deps.sampler();
    const client = this.deps.client?.(route) ?? new EmotionAnalysisClient(route);
    const assessments = new Map<string, EmotionAssessment>();
    const progress = async (phase: EmotionMatchingJob["phase"], text: string) => {
      signal.throwIfAborted();
      job.phase = phase; job.message = text;
      await this.publish(job);
    };
    let source: EmotionSource;
    let loaded = false;
    const sampleAsset = async (asset: EmotionAsset, mode: EmotionSample["mode"], currentSignal: AbortSignal) => {
      currentSignal.throwIfAborted();
      if (!loaded) {
        await progress("preparing", "正在加载采样模型");
        try { await sampler.load(source.bundle, currentSignal); loaded = true; }
        catch (error) { throw new EmotionAnalysisError(`采样模型无法加载：${message(error)}`, true); }
      }
      await progress(mode === "asset" ? "preparing" : "validating", `正在采样：${asset.name}${mode === "speaking" ? "（说话状态）" : mode === "state" ? "（持续状态）" : mode === "runtime" ? "（触发播放）" : ""}`);
      try { return await sampler.sample(asset, mode, currentSignal); }
      catch (error) {
        if (/上下文|绘制|进程|窗口|采样超时/u.test(message(error))) {
          await sampler.dispose();
          loaded = false;
        }
        throw error;
      }
    };
    const cacheKey = (asset: EmotionAsset) => digest({ version: EMOTION_ANALYSIS_VERSION, source: source.fingerprint,
      content: asset.contentHash, name: asset.name, parameters: asset.parameters, route: [route.baseUrl, route.model] });
    const checkSource = async () => {
      signal.throwIfAborted();
      const current = await this.deps.source(job.modelId);
      const assets = new Map(current.assets.map((a) => [a.id, a.contentHash]));
      if (current.fingerprint !== source.fingerprint || source.assets.some((a) => assets.get(a.id) !== a.contentHash)) {
        job.status = "stale";
        throw new Error("目标模型的素材或外观已改变，请重新分析");
      }
    };
    try {
      job.status = "running";
      await progress("preparing", "正在读取模型素材");
      source = await this.deps.source(job.modelId);
      signal.throwIfAborted();
      job.modelName = source.bundle.model.name;
      job.sourceFingerprint = source.fingerprint;
      job.totalAssets = source.assets.length;
      if (!job.totalAssets) throw new Error("模型没有可分析的动作或表情素材");
      await progress("preparing", `已读取 ${job.totalAssets} 个素材，正在准备分析`);
      await checkSource();
      let batch: EmotionSample[] = [];
      let bytes = 0;
      let frames = 0;
      const analyzeBatch = async (samples: EmotionSample[]): Promise<EmotionAssessment[]> => {
        try { return await abortable(client.analyze(samples, signal), signal); }
        catch (error) {
          if (error instanceof EmotionAnalysisError && error.split && samples.length > 1) {
            const half = Math.ceil(samples.length / 2);
            return [...await analyzeBatch(samples.slice(0, half)), ...await analyzeBatch(samples.slice(half))];
          }
          throw error;
        }
      };
      const validateCandidate = async (asset: EmotionAsset, expected: EmotionAssessment): Promise<void> => {
        const signature = candidateSignature(asset.candidate);
        const wanted = expected.matches.filter((match) => {
          const existing = job.suggestions.filter((s) => targetId(s.target) === targetId(match.target));
          return existing.length < 3 && !existing.some((s) => candidateSignature(s.candidate) === signature);
        });
        if (!wanted.length || expected.needsMoreFrames) return;
        const reusable = run.previous?.analysisModelFingerprint === job.analysisModelFingerprint ? run.previous.suggestions.filter((s) =>
          s.sourceFingerprint === source.fingerprint && candidateSignature(s.candidate) === signature &&
          digest(s.lipsync) === digest(source.bundle.model.manifest.lipsync) &&
          Object.entries(s.assetHashes).every(([id, hash]) => source.assets.some((a) => a.id === id && a.contentHash === hash))) ?? [] : [];
        const pending = wanted.filter((match) => {
          const previous = reusable.find((s) => targetId(s.target) === targetId(match.target));
          if (!previous) return true;
          job.suggestions.push(structuredClone(previous));
          return false;
        });
        if (pending.length !== wanted.length) await this.publish(job);
        if (!pending.length) return;

        // All targets using the same playback lifecycle share one validation request.
        const groups = ["runtime", "state"] as const;
        for (const mode of groups) {
          const matches = pending.filter((m) => targetUsesStatePlayback(m.target) === (mode === "state"));
          if (!matches.length) continue;
          const sample = await sampleAsset(mode === "state" ? stateValidationAsset(asset) : asset, mode, signal);
          await progress("validating", `正在验证${mode === "state" ? "持续状态" : "实际播放"}：${asset.name}，等待模型返回`);
          const checked = (await analyzeBatch([sample])).find((result) => result.assetId === asset.id);
          if (!checked || checked.needsMoreFrames || checked.classification !== "behavior") continue;
          const accepted = matches.filter((match) => checked.matches.some((actual) => targetId(actual.target) === targetId(match.target)));
          let speaking: EmotionSample | undefined;
          let speechChecked: EmotionAssessment | undefined;
          // State animations stop when speech starts; only emotion face layers persist.
          if (accepted.some((match) => match.target.kind === "emotion")) {
            const speechAsset = { ...asset, timestampsMs: [300, Math.min(119_000, Math.max(asset.durationMs + 400, 2400))] };
            speaking = await sampleAsset(speechAsset, "speaking", signal);
            await progress("validating", `正在验证说话效果：${asset.name}，等待模型返回`);
            speechChecked = (await analyzeBatch([speaking])).find((result) => result.assetId === asset.id);
          }
          await checkSource();
          for (const match of accepted) {
            const actual = checked.matches.find((m) => targetId(m.target) === targetId(match.target))!;
            const id = randomUUID();
            const retained = match.target.kind === "emotion" && speaking && speaking.faceExpressionIds.length > 0 &&
              !speechChecked?.needsMoreFrames && speechChecked?.matches.some((m) => targetId(m.target) === targetId(match.target));
            const suggestion: EmotionSuggestion = {
              id, target: structuredClone(match.target), candidate: { ...structuredClone(asset.candidate), id: `ai-${id}` },
              strength: actual.strength === "strong" && match.strength === "strong" ? "strong" : "possible",
              reason: checked.reason, sourceFingerprint: source.fingerprint,
              assetHashes: Object.fromEntries([...asset.candidate.motionAssetIds, ...asset.candidate.expressionAssetIds]
                .map((id) => [id, source.assets.find((a) => a.id === id)!.contentHash])),
              lipsync: structuredClone(source.bundle.model.manifest.lipsync), speaking: retained ? "retained" : "transient", adopted: false,
            };
            job.suggestions.push(suggestion);
          }
          await this.publish(job);
        }
      };
      const flush = async () => {
        if (!batch.length) return;
        const pending = batch;
        batch = []; bytes = 0; frames = 0;
        await progress("analyzing", `正在识别素材 ${job.processedAssets + 1}–${job.processedAssets + pending.length}/${job.totalAssets}，等待模型返回`);
        let results: EmotionAssessment[];
        try {
          results = await analyzeBatch(pending);
          await checkSource();
        } catch (error) {
          if (signal.aborted || isStale(job) || (error instanceof EmotionAnalysisError && error.fatal)) throw error;
          pending.forEach((s) => job.failures.push({ assetId: s.asset.id, message: message(error) }));
          job.processedAssets += pending.length;
          await this.publish(job);
          return;
        }
        for (const sample of pending) {
          try {
            let result = results.find((r) => r.assetId === sample.asset.id);
            if (!result) throw new EmotionAnalysisError("模型未返回此素材的有效结果");
            if (result.needsMoreFrames) {
              const refined = { ...sample.asset, timestampsMs: Array.from({ length: 10 }, (_, i) => Math.max(17, sample.asset.durationMs * (i + 1) / 11)) };
              const extra = await sampleAsset(refined, "asset", signal);
              await progress("analyzing", `正在补充识别：${sample.asset.name}，等待模型返回`);
              result = (await analyzeBatch([extra])).find((r) => r.assetId === sample.asset.id) ?? result;
            }
            assessments.set(sample.asset.id, result);
            await this.store.cache(cacheKey(sample.asset), result);
            await validateCandidate(sample.asset, result);
          } catch (error) {
            if (signal.aborted || isStale(job) || (error instanceof EmotionAnalysisError && error.fatal)) throw error;
            job.failures.push({ assetId: sample.asset.id, message: message(error) });
          }
          job.processedAssets++;
          await this.publish(job);
        }
      };
      for (const asset of source.assets) {
        signal.throwIfAborted();
        if (asset.error) {
          job.failures.push({ assetId: asset.id, message: asset.error }); job.processedAssets++; continue;
        }
        const cached = await this.store.cached(cacheKey(asset));
        if (cached) {
          await flush();
          assessments.set(asset.id, cached);
          job.cachedAssets++; job.processedAssets++;
          try { await validateCandidate(asset, cached); }
          catch (error) {
            if (signal.aborted || isStale(job) || (error instanceof EmotionAnalysisError && error.fatal)) throw error;
            job.failures.push({ assetId: asset.id, message: `缓存素材的播放验证失败：${message(error)}` });
          }
          await this.publish(job);
          continue;
        }
        try {
          const sample = await sampleAsset(asset, "asset", signal);
          const sampleBytes = sample.frames.reduce((n, f) => n + f.image.length, 0);
          if (batch.length && (batch.length >= 4 || frames + sample.frames.length > 12 || bytes + sampleBytes > 7 * 1024 * 1024)) await flush();
          batch.push(sample); bytes += sampleBytes; frames += sample.frames.length;
          if (batch.length === 1 && job.processedAssets === 0) await flush(); // Early capability check and first suggestion.
        } catch (error) {
          if (signal.aborted || isStale(job) || (error instanceof EmotionAnalysisError && error.fatal)) throw error;
          job.failures.push({ assetId: asset.id, message: message(error) }); job.processedAssets++;
        }
      }
      await flush();
      // At most three plausible combinations per target; never enumerate the Cartesian product.
      const targets = new Map([...assessments.values()].flatMap((a) => a.matches.map((m) => [targetId(m.target), m.target] as const)));
      for (const [key, target] of targets) {
        const matching = source.assets.filter((a) => assessments.get(a.id)?.matches.some((m) => targetId(m.target) === key));
        const motions = matching.filter((a) => a.kind === "motion").slice(0, 3);
        const expression = matching.find((a) => a.kind === "expression");
        if (!expression) continue;
        for (const motion of motions) {
          if (job.suggestions.filter((s) => targetId(s.target) === key).length >= 3) break;
          const candidate = { id: `combo-${motion.id}-${expression.id}`, motionAssetIds: [motion.id], expressionAssetIds: [expression.id] };
          const asset = { ...motion, id: candidate.id, name: `${motion.name} + ${expression.name}`, candidate };
          try { await validateCandidate(asset, { assetId: asset.id, classification: "behavior", matches: [{ target, strength: "possible" }], evidence: [], reason: "", needsMoreFrames: false }); }
          catch (error) {
            if (signal.aborted || isStale(job) || (error instanceof EmotionAnalysisError && error.fatal)) throw error;
          }
        }
      }
      await checkSource();
      job.status = job.failures.length ? "partial" : "completed";
      job.message = job.failures.length ? `已生成 ${job.suggestions.length} 条建议，${job.failures.length} 项未完成，可重试` : `分析完成，${job.suggestions.length} 条配置建议待查看`;
    } catch (error) {
      if (!["cancelled", "interrupted", "stale"].includes(job.status)) job.status = "failed";
      if (!signal.aborted) job.message = message(error);
    } finally {
      await sampler.dispose();
      job.phase = "finished";
      await this.publish(job);
    }
  }

  private async publish(job: EmotionMatchingJob): Promise<void> {
    if (this.jobs.get(job.modelId)?.id !== job.id) return;
    job.revision++;
    job.updatedAt = new Date().toISOString();
    await this.store.save(job);
    if (this.jobs.get(job.modelId)?.id === job.id) this.deps.updated(structuredClone(job));
  }
}
function message(error: unknown): string { return error instanceof Error ? error.message : "分析失败"; }

function isStale(job: EmotionMatchingJob): boolean { return job.status === "stale"; }
function assertModelId(id: string): void {
  if (typeof id !== "string" || !/^[a-z0-9][a-z0-9-]{0,79}$/.test(id)) throw new Error("模型 ID 无效");
}
