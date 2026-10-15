import { contextBridge, ipcRenderer } from "electron";
import type { EmotionSamplerRequest, EmotionSamplerResponse } from "../src/shared/emotion-matching";

contextBridge.exposeInMainWorld("emotionSampler", {
  onRequest: (callback: (request: EmotionSamplerRequest) => void) => {
    ipcRenderer.on("emotion-sampler:request", (_event, request: EmotionSamplerRequest) => callback(request));
  },
  reply: (response: EmotionSamplerResponse) => ipcRenderer.send("emotion-sampler:response", response),
});
