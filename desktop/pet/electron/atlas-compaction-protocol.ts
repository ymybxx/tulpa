import type { AtlasCompactionRequest } from "./atlas-compaction-coordinator";

export interface AtlasCompactionWorkerRequest {
  type: "atlas-compaction-request";
  requestId: string;
  request: AtlasCompactionRequest;
}

export interface AtlasCompactionWorkerResult {
  type: "atlas-compaction-result";
  requestId: string;
  modelId: string;
  revision: number;
  contentKeys: string[];
  built: number;
  reused: number;
}

export interface AtlasCompactionWorkerError {
  type: "atlas-compaction-error";
  requestId: string;
  modelId: string;
  revision: number;
  message: string;
}

export type AtlasCompactionWorkerResponse =
  | AtlasCompactionWorkerResult
  | AtlasCompactionWorkerError;

export function isAtlasCompactionWorkerRequest(
  value: unknown,
): value is AtlasCompactionWorkerRequest {
  if (!isRecord(value) || value.type !== "atlas-compaction-request") return false;
  return typeof value.requestId === "string" && value.requestId.length > 0 &&
    isRecord(value.request);
}

export function isAtlasCompactionWorkerResponse(
  value: unknown,
): value is AtlasCompactionWorkerResponse {
  if (!isRecord(value)) return false;
  if (value.type !== "atlas-compaction-result" && value.type !== "atlas-compaction-error") {
    return false;
  }
  if (
    typeof value.requestId !== "string" ||
    typeof value.modelId !== "string" ||
    !Number.isSafeInteger(value.revision)
  ) return false;
  if (value.type === "atlas-compaction-error") return typeof value.message === "string";
  return Array.isArray(value.contentKeys) &&
    value.contentKeys.every((item) => typeof item === "string") &&
    Number.isSafeInteger(value.built) && Number.isSafeInteger(value.reused);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
