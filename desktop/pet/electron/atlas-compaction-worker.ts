import { executeAtlasCompactionRequest } from "./atlas-compaction-job";
import {
  isAtlasCompactionWorkerRequest,
  type AtlasCompactionWorkerResponse,
} from "./atlas-compaction-protocol";

const port = process.parentPort;
if (!port) throw new Error("图集压实 worker 缺少父进程通信端口");

port.once("message", (event) => {
  void respond(event.data);
});

async function respond(value: unknown): Promise<void> {
  let response: AtlasCompactionWorkerResponse;
  if (!isAtlasCompactionWorkerRequest(value)) {
    response = {
      type: "atlas-compaction-error",
      requestId: looseString(value, "requestId"),
      modelId: looseModelId(value),
      revision: looseRevision(value),
      message: "图集压实 worker 收到无效请求",
    };
  } else {
    try {
      response = {
        type: "atlas-compaction-result",
        requestId: value.requestId,
        ...await executeAtlasCompactionRequest(value.request),
      };
    } catch (error) {
      response = {
        type: "atlas-compaction-error",
        requestId: value.requestId,
        modelId: looseString(value.request, "modelId"),
        revision: looseRevisionFromRequest(value.request),
        message: sanitizeError(error),
      };
    }
  }
  port.postMessage(response);
}

function looseString(value: unknown, key: string): string {
  if (typeof value !== "object" || value === null) return "invalid";
  const candidate = (value as Record<string, unknown>)[key];
  return typeof candidate === "string" && candidate ? candidate.slice(0, 256) : "invalid";
}

function looseModelId(value: unknown): string {
  if (typeof value !== "object" || value === null) return "invalid";
  const request = (value as Record<string, unknown>).request;
  return looseString(request, "modelId");
}

function looseRevision(value: unknown): number {
  if (typeof value !== "object" || value === null) return 0;
  const request = (value as Record<string, unknown>).request;
  if (typeof request !== "object" || request === null) return 0;
  return looseRevisionFromRequest(request);
}

function looseRevisionFromRequest(request: unknown): number {
  if (typeof request !== "object" || request === null) return 0;
  const revision = (request as Record<string, unknown>).revision;
  return Number.isSafeInteger(revision) && Number(revision) >= 0 ? Number(revision) : 0;
}

function sanitizeError(value: unknown): string {
  const message = value instanceof Error ? value.message : String(value);
  return message.replace(/[\r\n]+/g, " ").slice(0, 2_000) || "图集压实失败";
}
