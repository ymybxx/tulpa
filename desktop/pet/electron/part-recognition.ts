import { PNG } from "pngjs";

import type { ModelEditCapabilityManifest, TextureEditUnit } from "../src/shared/contracts";
import { composeAnnotatedModelReferencePng } from "./character-generation/model-locator";
import { chatCompletionsUrl } from "./character-generation/prompt-optimizer";
import { loggedFetch } from "./llm-logs";

export const PART_RECOGNITION_CHANNEL = "pet:part-recognition";
const BATCH_SIZE = 6;
const TIMEOUT_MS = 90_000;
const MAX_NEW_NAME = 8;

export interface PartRecognitionAssignment {
  unitId: string;
  groupName: string;
}

export interface PartRecognitionUpdate {
  modelId: string;
  running: boolean;
  done: number;
  total: number;
  /** Only the assignments made since the previous update. */
  assignments: PartRecognitionAssignment[];
  message: string;
}

export interface PartRecognitionDependencies {
  capability(modelId: string): Promise<ModelEditCapabilityManifest>;
  /** The neutral full-model render the masks are placed on. */
  guide(modelId: string): Promise<Uint8Array>;
  readModelFile(modelId: string, relative: string): Promise<Uint8Array>;
  /** The unit's current pixels. */
  unitPreview(modelId: string, unitId: string): Promise<Uint8Array>;
  pngDataUrl(bytes: Uint8Array, maxLongEdge: number): string;
  route(): Promise<{ baseUrl: string; model: string; apiKey: string }>;
  send(update: PartRecognitionUpdate): void;
  fetcher?: typeof fetch;
}

/**
 * Asks the picture-reading model which group each ungrouped piece belongs to.
 * Batches of pieces are marked A, B, C… on one full-model picture and also
 * shown on their own; the model picks an existing group name or a short new one.
 */
export class PartRecognizer {
  private readonly running = new Map<string, AbortController>();

  constructor(private readonly dependencies: PartRecognitionDependencies) {}

  isRunning(modelId: string): boolean {
    return this.running.has(modelId);
  }

  start(modelId: string, groupNames: readonly string[], unitIds: readonly string[]): void {
    if (this.running.has(modelId)) throw new Error("这个形象正在识别部位");
    const names = [...new Set(groupNames.map((name) => name.trim()).filter(Boolean))].slice(0, 64);
    const ids = [...new Set(unitIds)].slice(0, 600);
    if (!ids.length) throw new Error("没有需要分组的块");
    const controller = new AbortController();
    this.running.set(modelId, controller);
    void this.run(modelId, names, ids, controller.signal).finally(() => this.running.delete(modelId));
  }

  cancel(modelId: string): void {
    this.running.get(modelId)?.abort();
  }

  private async run(modelId: string, groupNames: string[], unitIds: string[], signal: AbortSignal): Promise<void> {
    const total = unitIds.length;
    let done = 0;
    let assigned = 0;
    const send = (running: boolean, assignments: PartRecognitionAssignment[], message: string) =>
      this.dependencies.send({ modelId, running, done, total, assignments, message });
    send(true, [], "");
    try {
      const capability = await this.dependencies.capability(modelId);
      const byId = new Map(capability.textureUnits.map((unit) => [unit.id, unit]));
      const units = unitIds.flatMap((id) => byId.get(id) ?? []);
      const route = await this.dependencies.route();
      const guide = await this.dependencies.guide(modelId);
      const names = [...groupNames];
      for (let start = 0; start < units.length; start += BATCH_SIZE) {
        if (signal.aborted) break;
        const batch = units.slice(start, start + BATCH_SIZE);
        const assignments = await this.recognizeBatch(modelId, batch, guide, names, route, signal);
        for (const assignment of assignments) if (!names.includes(assignment.groupName)) names.push(assignment.groupName);
        done += batch.length;
        assigned += assignments.length;
        send(true, assignments, "");
      }
      const stopped = signal.aborted;
      send(false, [], stopped
        ? `已停下：分好了 ${assigned} 块，都能改`
        : assigned < total
          ? `分好了 ${assigned} 块，还有 ${total - assigned} 块识图模型也没认出来，可以展开「没分组」手动加`
          : `分好了 ${assigned} 块，都能改`);
    } catch (error) {
      const message = signal.aborted ? `已停下：分好了 ${assigned} 块` : `识图分组失败：${error instanceof Error ? error.message : String(error)}`;
      send(false, [], message);
    }
  }

  private async recognizeBatch(
    modelId: string,
    units: readonly TextureEditUnit[],
    guide: Uint8Array,
    groupNames: readonly string[],
    route: { baseUrl: string; model: string; apiKey: string },
    signal: AbortSignal,
  ): Promise<PartRecognitionAssignment[]> {
    const keys = units.map((_, index) => String.fromCharCode(65 + index));
    const masks = await Promise.all(units.map((unit) => this.dependencies.readModelFile(modelId, unit.modelMaskPath)));
    const overview = composeAnnotatedModelReferencePng(guide, units.map((unit, index) => ({
      key: keys[index]!,
      maskBytes: masks[index]!,
      regionIndex: index,
      placement: unit.modelMaskBounds,
    })));
    const previews = await Promise.all(units.map((unit) => this.dependencies.unitPreview(modelId, unit.id)));
    const content: Array<Record<string, unknown>> = [
      { type: "text", text: "整张图是角色的正面，彩色标出了要分组的块和它们的字母。每块单独的图都垫了灰底。" },
      { type: "image_url", image_url: { url: this.dependencies.pngDataUrl(overview, 1024), detail: "high" } },
    ];
    units.forEach((unit, index) => {
      const names = (unit.searchText ?? "").split(" ").filter((word) => word && !/^ArtMesh\d*$/i.test(word)).slice(0, 8).join("、");
      const hidden = unit.revealedByMotionIds.length > 0 || /按键|按鍵|表情|動作|动作|切替/.test(unit.searchText ?? "");
      content.push({ type: "text", text: `块 ${keys[index]} 单独的样子${names ? `（图层名：${names}）` : ""}${hidden ? "；这块平时不显示，做动作或切表情时才出现，整张图里可能看不到，按它单独的样子判断" : ""}：` });
      content.push({ type: "image_url", image_url: { url: this.dependencies.pngDataUrl(onGray(previews[index]!), 384), detail: "low" } });
    });
    const system = [
      "你在给 Live2D 角色的图层块分部位组，用来换色和隐藏。",
      `已有分组：${groupNames.length ? groupNames.join("、") : "（还没有）"}。`,
      "每个字母块选一个最合适的分组：优先用已有分组的原名；都不合适时起一个新的中文短名（2 到 6 个字，比如 上衣、裙子、袜子、发饰、特效、水印）。",
      "按块实际画的是什么来判断，图层名只作参考。整张图里看不到的块，就看它单独的样子判断它是身上哪一部分。",
      "实在看不出来的块 group 填空字符串，不要猜。",
      '只输出 JSON：{"items":[{"key":"A","group":"后发"}]}，每个字母一项。',
    ].join("\n");
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal.addEventListener("abort", abort, { once: true });
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      const response = await loggedFetch("desktop.parts", this.dependencies.fetcher ?? fetch)(chatCompletionsUrl(route.baseUrl), {
        method: "POST",
        headers: { authorization: `Bearer ${route.apiKey}`, "content-type": "application/json" },
        body: JSON.stringify({
          model: route.model,
          stream: false,
          temperature: 0.1,
          max_tokens: 800,
          response_format: { type: "json_object" },
          messages: [{ role: "system", content: system }, { role: "user", content }],
        }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`识图模型返回 ${response.status}`);
      const payload = await response.json() as { choices?: Array<{ message?: { content?: unknown } }> };
      return parseAssignments(payload.choices?.[0]?.message?.content, keys, units);
    } finally {
      clearTimeout(timeout);
      signal.removeEventListener("abort", abort);
    }
  }
}

/**
 * Transparent pixels would be read as white or black by the model, hiding
 * white clothes or dark hair; a mid-gray backdrop keeps both visible.
 */
export function onGray(bytes: Uint8Array): Buffer {
  const image = PNG.sync.read(Buffer.from(bytes));
  for (let offset = 0; offset < image.data.length; offset += 4) {
    const alpha = image.data[offset + 3]! / 255;
    for (let channel = 0; channel < 3; channel += 1) {
      image.data[offset + channel] = Math.round(image.data[offset + channel]! * alpha + 128 * (1 - alpha));
    }
    image.data[offset + 3] = 255;
  }
  return PNG.sync.write(image);
}

export function parseAssignments(
  content: unknown,
  keys: readonly string[],
  units: readonly Pick<TextureEditUnit, "id">[],
): PartRecognitionAssignment[] {
  if (typeof content !== "string") throw new Error("识图模型没有返回结果");
  const text = content.trim().replace(/^```(?:json)?\s*/i, "").replace(/```$/, "");
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("识图模型返回的不是 JSON");
  }
  const items = parsed && typeof parsed === "object" && Array.isArray((parsed as { items?: unknown }).items)
    ? (parsed as { items: unknown[] }).items
    : [];
  const seen = new Set<string>();
  return items.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const { key, group } = item as Record<string, unknown>;
    if (typeof key !== "string" || typeof group !== "string") return [];
    const index = keys.indexOf(key.trim().toUpperCase());
    const name = [...group.trim()].slice(0, MAX_NEW_NAME).join("");
    if (index < 0 || !name || seen.has(keys[index]!)) return [];
    seen.add(keys[index]!);
    return [{ unitId: units[index]!.id, groupName: name }];
  });
}
