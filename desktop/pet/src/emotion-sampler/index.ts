import { BehaviorSampleSession } from "./session";
import type { EmotionSamplerRequest, EmotionSamplerResponse } from "../shared/emotion-matching";

declare global {
  interface Window {
    emotionSampler: {
      onRequest(callback: (request: EmotionSamplerRequest) => void): void;
      reply(response: EmotionSamplerResponse): void;
    };
  }
}

let session: BehaviorSampleSession | null = null;
let tail = Promise.resolve();
window.emotionSampler.onRequest((request) => {
  tail = tail.catch(() => undefined).then(async () => {
    try {
      if (request.operation === "load") {
        session?.dispose();
        session = new BehaviorSampleSession(document.querySelector("canvas")!);
        await session.load(request.bundle);
      } else if (request.operation === "sample") {
        if (!session) throw new Error("采样页未初始化");
        const sample = await session.sample(request.asset, request.mode);
        window.emotionSampler.reply({ id: request.id, sample });
        return;
      } else if (request.operation === "survey-parameters") {
        if (!session) throw new Error("采样页未初始化");
        const survey = await session.surveyParameters(request.parameterIds);
        window.emotionSampler.reply({ id: request.id, survey });
        return;
      } else if (request.operation === "sample-parameter") {
        if (!session) throw new Error("采样页未初始化");
        const parameterSample = await session.sampleParameter(request.parameterId, request.mode, request.levelCount);
        window.emotionSampler.reply({ id: request.id, parameterSample });
        return;
      } else {
        session?.dispose();
        session = null;
      }
      window.emotionSampler.reply({ id: request.id });
    } catch (error) {
      window.emotionSampler.reply({ id: request.id, error: error instanceof Error ? error.message : "采样失败" });
    }
  });
});
window.addEventListener("beforeunload", () => session?.dispose());
