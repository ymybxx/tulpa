import path from "node:path";
import catalog from "../src/shared/facial-demo-parameters.json";
import { AUTOMATIC_FACIAL_IDS, FACIAL_MOTION_STYLES, initialManualBinding, validateCustomParameters, validateNativeBinding, type FacialMotionStyle, type ManualParameterBinding } from "../src/shared/facial-bindings";
import type { NativeModelParameter } from "../src/shared/contracts";
import type { FacialAutoSetupContext, FacialAutoSetupProgress, FacialAutoSetupProposal, FacialParameterSample, FacialParameterSurvey } from "../src/shared/facial-test";
import type { EmotionModelRoute } from "./emotion-matching/client";
import { abortable } from "./emotion-matching/abortable";
import { loggedFetch } from "./llm-logs";
import { completionAnswer, extractJson, messageText, suggestFacialParameter } from "./facial-parameter-suggestion";

/**
 * 说话表情 一键配置: find which of a model's own parameters Jev should drive.
 *
 * 1. Survey (local): move every parameter alone to both ends and see what
 *    changes on screen. Physics, breathing and anything already used is left out.
 * 2. Sort (one text request): by name, group and where it changes, each
 *    parameter is a standard part under another name, an expression or
 *    gesture worth its own part, or not for Jev (outfit switches, rig internals).
 * 3. Look (one image request each): confirm a standard part and which end is
 *    which; write steps for a new part.
 *
 * Nothing is saved: the person picks what to keep from the proposal.
 */

type Standard = (typeof catalog)[number];
const STANDARD = new Map(catalog.map(entry => [entry.id, entry]));

export interface AutoSetupCandidate {
  parameter: NativeModelParameter;
  group: string;
  box: NonNullable<FacialParameterSurvey["box"]>;
  changed: number;
  /** Standard parts this parameter moves through the author's physics. */
  drives?: string[];
}

/** For each physics input, the parameters it moves and how strongly (input weight, 0..100). */
export type PhysicsLinks = Map<string, Array<{ output: string; weight: number }>>;

export async function physicsLinks(entry: string, readAsset: (relative: string) => Promise<string>): Promise<PhysicsLinks> {
  const links: PhysicsLinks = new Map();
  try {
    const file = JSON.parse(await readAsset(entry))?.FileReferences?.Physics;
    if (typeof file !== "string" || !file) return links;
    const physics = JSON.parse(await readAsset(path.posix.join(path.posix.dirname(entry), file)));
    for (const setting of physics.PhysicsSettings ?? []) {
      const outputs = (setting.Output ?? []).flatMap((item: { Destination?: { Target?: string; Id?: string } }) =>
        item.Destination?.Target === "Parameter" && typeof item.Destination.Id === "string" ? [item.Destination.Id] : []);
      for (const input of setting.Input ?? []) {
        const id = input?.Source?.Target === "Parameter" ? input.Source.Id : null;
        if (typeof id !== "string") continue;
        const list = links.get(id) ?? [];
        for (const output of outputs) list.push({ output, weight: Number(input.Weight) || 0 });
        links.set(id, list);
      }
    }
  } catch { /* Without physics there is simply nothing to follow. */ }
  return links;
}

/**
 * Standard parts the model smooths through physics: the author moves one
 * input (「■左眉 上下」) and physics carries the standard parameter along. When
 * one free input clearly leads an open part, that input is the part's control.
 * Each input serves one part, the first open one in standard order.
 */
export function physicsDrivenParts(candidates: readonly AutoSetupCandidate[], open: readonly string[], links: PhysicsLinks): Map<string, string> {
  const assigned = new Map<string, string>();
  const free = new Set(candidates.map(candidate => candidate.parameter.id));
  for (const standardId of open) {
    let best: { input: string; weight: number } | null = null;
    for (const [input, outputs] of links) {
      if (!free.has(input) || [...assigned.values()].includes(input)) continue;
      const weight = Math.max(0, ...outputs.filter(item => item.output === standardId).map(item => item.weight));
      if (weight >= 50 && (!best || weight > best.weight)) best = { input, weight };
    }
    if (best) assigned.set(standardId, best.input);
  }
  return assigned;
}

export interface AutoSetupSampler {
  surveyParameters(parameterIds: readonly string[] | null, signal: AbortSignal): Promise<FacialParameterSurvey[]>;
  sampleParameter(parameterId: string, mode: "score" | "binary", levelCount: number, signal: AbortSignal): Promise<FacialParameterSample>;
}

/** Smaller changes are invisible at pet size; a brow moving in a full-body view is about 0.001. */
const MIN_CHANGE = 0.0003;
const DETAIL_CONCURRENCY = 3;

/** Parameters worth sorting: visible, free, and not something the model or Jev already drives. */
export function autoSetupCandidates(
  native: readonly NativeModelParameter[],
  survey: readonly FacialParameterSurvey[],
  groups: ReadonlyMap<string, string>,
  context: FacialAutoSetupContext,
  blocked: ReadonlySet<string>,
): { candidates: AutoSetupCandidate[]; invisible: NativeModelParameter[] } {
  const used = new Set(context.used);
  const byId = new Map(survey.map(item => [item.id, item]));
  const candidates: AutoSetupCandidate[] = [];
  const invisible: NativeModelParameter[] = [];
  for (const parameter of native) {
    // Standard IDs are matched by name already (or deleted on purpose).
    if (STANDARD.has(parameter.id) || AUTOMATIC_FACIAL_IDS.has(parameter.id) || blocked.has(parameter.id) || used.has(parameter.id)) continue;
    if (!(parameter.maximum > parameter.minimum)) continue;
    const found = byId.get(parameter.id);
    if (!found?.box || found.changed < MIN_CHANGE) { invisible.push(parameter); continue; }
    candidates.push({ parameter, group: groups.get(parameter.id) ?? "", box: found.box, changed: found.changed });
  }
  return { candidates, invisible };
}

/** Group names from the model's display info (cdi3), keyed by parameter ID. */
export async function displayGroups(entry: string, readAsset: (relative: string) => Promise<string>): Promise<Map<string, string>> {
  const groups = new Map<string, string>();
  try {
    const descriptor = JSON.parse(await readAsset(entry));
    const file = descriptor?.FileReferences?.DisplayInfo;
    if (typeof file !== "string" || !file) return groups;
    const info = JSON.parse(await readAsset(path.posix.join(path.posix.dirname(entry), file)));
    const names = new Map<string, string>((info.ParameterGroups ?? []).map((group: { Id?: string; Name?: string }) => [group.Id, group.Name ?? ""]));
    for (const parameter of info.Parameters ?? []) {
      if (typeof parameter?.Id === "string" && typeof parameter.GroupId === "string") groups.set(parameter.Id, names.get(parameter.GroupId) || parameter.GroupId);
    }
  } catch { /* Group names only help sorting; a model without them still works. */ }
  return groups;
}

/** Coordinates only: whether 0.15 is a brow or a hat depends on how much of the body the model shows. */
function where(box: AutoSetupCandidate["box"]): string {
  const range = (from: number, size: number) => `${from.toFixed(2)}–${Math.min(1, from + size).toFixed(2)}`;
  return `x ${range(box.x, box.width)}，y ${range(box.y, box.height)}`;
}

const fmt = (value: number) => String(Number(value.toFixed(3)));

export const SORT_PROMPT = `你在帮桌宠配置「说话表情」。桌宠每说一句话，Jev 只读这句台词，为每个「部位」选一档，模型就做出对应的表情和动作。现在要从一个 Live2D 模型自己的参数里，找出值得交给 Jev 的部分。

每个候选参数会给出：ID、作者起的名字、所在分组、取值范围，以及单独拨动它时画面在哪里变化（相对模型画出来的整个角色，x、y 从 0 到 1，y=0 是最上面，y=1 是最下面；全身模型的脸大约在最上面的五分之一）和变化面积。

逐个判断 use：
- "standard"：它其实就是「还没对应的标准部位」中的一个，只是作者起了别的名字，比如「右眉 上下」就是右眉上下，「L眼-微笑」就是左眼笑眼形状。给出 standardId。名字说的必须是同一个部位、同一种动作；只是位置挨着不算，比如嘴的参数不能对到眼睛或眉毛。一个标准部位只对应一个参数，挑最直接的那个；两只手臂一起动的参数不对应单侧手臂。
- "custom"：值得加成新部位的表情或动作，说话时会随情绪、语气出现或变化，比如生气、脸红、黑脸、星星眼、爱心眼、哭、流泪、问号、鼓腮、吐舌、嘟嘴、比嘘、托腮、挥手、身体前倾。
  kind："expression"（脸上的表情、漫画符号）或 "motion"（肢体动作）。
  mode："binary"（开关式，出现或不出现）或 "score"（程度连续变化）。
  motion：说话时怎么动。"hold" 保持到这句话说完，大多数表情用它；"swing" 来回摆动，如挥手；"pulse" 做一下就回来，如点头、耸肩。
  最多 {slots} 个，只挑最有表现力、最常用得上的。
- "skip"：不交给 Jev。reason 用几个字说明，如「发型或饰品开关，属于装扮」「作者控制动画用的内部参数」「和其他参数重复」「分隔标记」「变化不明显」。

原则：
- 换装开关（发型、饰品、服装有没有）不是表情，一律 skip。
- 同一个效果有好几个参数时（如「角度 X」出现多次、分成 in/out 两个），最多选一个。
- 拿不准就 skip。

只输出一个 JSON 对象：{"items":[{"id":"…","use":"standard","standardId":"…"},{"id":"…","use":"custom","kind":"expression","mode":"binary","motion":"hold"},{"id":"…","use":"skip","reason":"…"}]}。每个候选参数恰好出现一次，不要 Markdown，不要解释。`;

export function sortRequest(candidates: readonly AutoSetupCandidate[], open: readonly string[], slots: number): { system: string; user: string } {
  const standards = open.flatMap(id => {
    const entry = STANDARD.get(id);
    return entry ? [`${id} · ${entry.label}：${entry.description.split(/[。；]/u)[0]}`] : [];
  });
  const lines = candidates.map(({ parameter, group, box, changed, drives }) =>
    `${parameter.id} · ${parameter.name && parameter.name !== parameter.id ? parameter.name : "（无名字）"} · 分组：${group || "（无）"} · 范围 ${fmt(parameter.minimum)}～${fmt(parameter.maximum)}（默认 ${fmt(parameter.defaultValue)}）· 变化：${where(box)}，约占全身 ${(changed * 100).toFixed(2)}%${drives?.length ? ` · 通过物理带动：${drives.map(id => STANDARD.get(id)?.label ?? id).join("、")}` : ""}`);
  return {
    system: SORT_PROMPT.replace("{slots}", String(slots)),
    user: [
      "还没对应的标准部位：", ...(standards.length ? standards : ["（没有）"]), "",
      `候选参数（${lines.length} 个）：`, ...lines,
    ].join("\n"),
  };
}

export type SortedItem =
  | { id: string; use: "standard"; standardId: string }
  | { id: string; use: "custom"; kind: "expression" | "motion"; mode: "score" | "binary"; motion: FacialMotionStyle }
  | { id: string; use: "skip"; reason: string };

/** Keeps the model's answer to what it may decide: known IDs, open standard parts once each, custom within the slots. */
export function readSorted(value: unknown, candidates: readonly AutoSetupCandidate[], open: readonly string[], slots: number): SortedItem[] {
  const items = (value as { items?: unknown })?.items;
  if (!Array.isArray(items)) throw new Error("缺少 items 数组");
  const known = new Set(candidates.map(candidate => candidate.parameter.id));
  const openSet = new Set(open);
  const answers = new Map<string, Record<string, unknown>>();
  for (const item of items) {
    if (item && typeof item === "object" && typeof item.id === "string" && known.has(item.id) && !answers.has(item.id)) answers.set(item.id, item);
  }
  const takenStandard = new Set<string>();
  let customs = 0;
  return candidates.map(({ parameter: { id } }): SortedItem => {
    const answer = answers.get(id);
    if (!answer) return { id, use: "skip", reason: "AI 没有给出判断" };
    if (answer.use === "standard" && typeof answer.standardId === "string") {
      if (!openSet.has(answer.standardId)) return { id, use: "skip", reason: "对应的标准部位已经有参数了" };
      if (takenStandard.has(answer.standardId)) return { id, use: "skip", reason: "和其他参数重复" };
      takenStandard.add(answer.standardId);
      return { id, use: "standard", standardId: answer.standardId };
    }
    if (answer.use === "custom") {
      if (customs >= slots) return { id, use: "skip", reason: "自定义部位的名额已满" };
      customs++;
      return { id, use: "custom", kind: answer.kind === "motion" ? "motion" : "expression",
        mode: answer.mode === "binary" ? "binary" : "score",
        motion: FACIAL_MOTION_STYLES.includes(answer.motion as FacialMotionStyle) ? answer.motion as FacialMotionStyle : "hold" };
    }
    const reason = typeof answer.reason === "string" && answer.reason.trim() ? answer.reason.trim().slice(0, 40) : "不适合交给决策模型";
    return { id, use: "skip", reason };
  });
}

export const CHECK_PROMPT = `你在核对 Live2D 模型的一个参数是不是某个标准部位。截图是这个参数取不同值时的样子（每张左边是全身，右边是红框区域的放大）。
判断：这个参数做的动作和标准部位说的是不是同一件事？如果是，标准部位「最强一端」的样子对应哪张截图，是参数最小值还是最大值？
只输出一个 JSON 对象：{"match":true,"strongAt":"minimum"|"maximum","note":"不超过 40 字的依据"} 或 {"match":false,"note":"不超过 40 字的原因"}。不要 Markdown。`;

export function checkRequest(sample: FacialParameterSample, standard: Standard): Array<Record<string, unknown>> {
  const { parameter } = sample;
  const content: Array<Record<string, unknown>> = [{ type: "text", text: [
    `参数：${parameter.id}（${parameter.name || "无名字"}），范围 ${fmt(parameter.minimum)}～${fmt(parameter.maximum)}，默认 ${fmt(parameter.defaultValue)}。`,
    `标准部位：${standard.label}。${standard.description}`,
    `最弱一端：${standard.levels[0]}`,
    `最强一端：${standard.levels.at(-1)}`,
  ].join("\n") }];
  for (const [index, frame] of sample.frames.entries()) {
    const tag = frame.value === parameter.minimum ? "最小值" : frame.value === parameter.maximum ? "最大值" : frame.kind === "default" ? "默认" : "中间";
    content.push({ type: "text", text: `截图 ${index + 1}：${tag}（${fmt(frame.value)}）` });
    content.push({ type: "image_url", image_url: { url: frame.image, detail: "high" } });
  }
  return content;
}

export function readCheck(value: unknown): { match: boolean; strongAt: "minimum" | "maximum"; note: string } {
  if (!value || typeof value !== "object") throw new Error("返回内容不是 JSON 对象");
  const data = value as Record<string, unknown>;
  const note = typeof data.note === "string" ? data.note.trim().slice(0, 80) : "";
  if (data.match !== true) return { match: false, strongAt: "maximum", note };
  if (data.strongAt !== "minimum" && data.strongAt !== "maximum") throw new Error("strongAt 只能是 minimum 或 maximum");
  return { match: true, strongAt: data.strongAt, note };
}

/** The model's own range for a standard part; flipped when its strong end is the parameter's minimum. */
export function standardBinding(standardId: string, parameter: NativeModelParameter, strongAt: "minimum" | "maximum"): ManualParameterBinding {
  const binding = initialManualBinding(standardId, parameter);
  if (strongAt === "maximum") return binding;
  const neutral = binding.neutralValue === binding.lowerValue ? binding.upperValue
    : binding.neutralValue === binding.upperValue ? binding.lowerValue : binding.neutralValue;
  return { parameterId: parameter.id, lowerValue: binding.upperValue, neutralValue: neutral, upperValue: binding.lowerValue };
}

export async function runAutoSetup(input: {
  groups: ReadonlyMap<string, string>;
  links: PhysicsLinks;
  blocked: ReadonlySet<string>;
  context: FacialAutoSetupContext;
  sampler: AutoSetupSampler;
  route: EmotionModelRoute;
  signal: AbortSignal;
  progress(progress: FacialAutoSetupProgress): void;
  fetcher?: typeof fetch;
}): Promise<FacialAutoSetupProposal> {
  const { context, sampler, route, signal, progress } = input;
  const fetcher = input.fetcher ?? fetch;
  progress({ text: "正在看每个参数动了哪里" });
  const survey = await sampler.surveyParameters(null, signal);
  const native = survey.map(item => item.parameter);
  const name = (id: string) => { const parameter = native.find(item => item.id === id); return parameter?.name && parameter.name !== id ? parameter.name : id; };
  const proposal: FacialAutoSetupProposal = { standard: [], custom: [], skipped: [], failed: [] };

  const { candidates, invisible } = autoSetupCandidates(native, survey, input.groups, context, input.blocked);
  for (const parameter of invisible) proposal.skipped.push({ parameterId: parameter.id, parameterName: name(parameter.id), reason: "画面看不出变化" });
  if (!candidates.length) return proposal;
  for (const candidate of candidates) {
    const drives = (input.links.get(candidate.parameter.id) ?? []).filter(link => STANDARD.has(link.output) && link.weight >= 50).map(link => link.output);
    if (drives.length) candidate.drives = [...new Set(drives)];
  }
  // What the physics settles needs no guessing; the picture check still confirms each one.
  const physical = physicsDrivenParts(candidates, context.open, input.links);
  const known = new Set(physical.values());
  const rest = candidates.filter(candidate => !known.has(candidate.parameter.id));
  const open = context.open.filter(id => !physical.has(id));

  const slots = Math.max(0, Math.min(16, context.customSlots));
  let sorted: SortedItem[] = [...physical].map(([standardId, id]) => ({ id, use: "standard" as const, standardId }));
  if (rest.length) {
    progress({ text: `正在分辨哪些是表情和动作（${rest.length} 个参数）` });
    const request = sortRequest(rest, open, slots);
    sorted = [...sorted, ...await chatJson(route, request.system, request.user, 8000, signal, fetcher,
      value => readSorted(value, rest, open, slots))];
  }

  const chosen = sorted.filter(item => item.use !== "skip");
  for (const item of sorted) if (item.use === "skip") proposal.skipped.push({ parameterId: item.id, parameterName: name(item.id), reason: item.reason });
  let done = 0;
  const look = async (item: Exclude<SortedItem, { use: "skip" }>) => {
    const parameter = native.find(value => value.id === item.id)!;
    try {
      if (item.use === "standard") {
        const standard = STANDARD.get(item.standardId)!;
        const sample = await sampler.sampleParameter(item.id, "score", 3, signal);
        const check = await chatJson(route, CHECK_PROMPT, checkRequest(sample, standard), 400, signal, fetcher, readCheck);
        if (!check.match) {
          proposal.skipped.push({ parameterId: item.id, parameterName: name(item.id), reason: `看图后不像「${standard.label}」${check.note ? `：${check.note}` : ""}` });
          return;
        }
        const binding = standardBinding(item.standardId, parameter, check.strongAt);
        validateNativeBinding(item.standardId, binding, native, input.blocked);
        proposal.standard.push({ standardId: item.standardId, parameterId: item.id, parameterName: name(item.id), binding, note: check.note });
      } else {
        const levelCount = item.mode === "binary" ? 2 : 3;
        const sample = await sampler.sampleParameter(item.id, item.mode, levelCount, signal);
        const suggestion = await suggestFacialParameter(sample, item.mode, levelCount, route, signal, fetcher);
        if (suggestion.confidence !== "clear") {
          proposal.skipped.push({ parameterId: item.id, parameterName: name(item.id), reason: `看图看不出明确的表现${suggestion.note ? `：${suggestion.note}` : ""}` });
          return;
        }
        // The author's own Chinese name reads best (生气、扶下巴); otherwise the AI's.
        const own = parameter.name?.replace(/[■●⚪]/gu, "").trim() ?? "";
        const label = (/\p{Script=Han}/u.test(own) ? own : suggestion.label).slice(0, 30);
        const [value] = validateCustomParameters([{ id: item.id, label, description: suggestion.description,
          levels: suggestion.mode === "score" ? suggestion.levels! : ["否", "是"], mode: suggestion.mode,
          ...suggestion.mode === "binary" ? { falseDescription: suggestion.falseDescription, trueDescription: suggestion.trueDescription } : {},
          enabled: true, multiplier: 1, motion: item.motion }]);
        proposal.custom.push({ parameterId: item.id, parameterName: name(item.id), kind: item.kind, value: value!, note: suggestion.note });
      }
    } catch (error) {
      if (signal.aborted) throw error;
      proposal.failed.push({ parameterId: item.id, parameterName: name(item.id), error: error instanceof Error ? error.message : String(error) });
    } finally {
      done++;
      progress({ text: `正在细看「${name(item.id)}」`, done, total: chosen.length });
    }
  };
  progress({ text: "正在细看挑出来的参数", done: 0, total: chosen.length });
  const queue = [...chosen];
  await Promise.all(Array.from({ length: Math.min(DETAIL_CONCURRENCY, queue.length) }, async () => {
    for (let item = queue.shift(); item; item = queue.shift()) {
      signal.throwIfAborted();
      await look(item as Exclude<SortedItem, { use: "skip" }>);
    }
  }));
  // Keep the model's own order, so parts of one area stay together.
  const order = new Map(native.map((parameter, index) => [parameter.id, index]));
  const byOrder = (a: { parameterId: string }, b: { parameterId: string }) => order.get(a.parameterId)! - order.get(b.parameterId)!;
  proposal.standard.sort(byOrder); proposal.custom.sort(byOrder); proposal.skipped.sort(byOrder); proposal.failed.sort(byOrder);
  return proposal;
}

/** One JSON answer, asked again once with the problem named when it does not read. */
async function chatJson<T>(route: EmotionModelRoute, system: string, user: string | Array<Record<string, unknown>>, maxTokens: number,
  signal: AbortSignal, fetcher: typeof fetch, read: (value: unknown) => T): Promise<T> {
  const url = new URL(route.baseUrl);
  url.pathname = `${url.pathname.replace(/\/+$/, "").replace(/\/chat\/completions$/, "")}/chat/completions`;
  url.search = ""; url.hash = "";
  const history: Array<{ role: string; content: unknown }> = [];
  let jsonMode = true;
  let issue = "";
  for (let attempt = 0; attempt < 2; attempt++) {
    signal.throwIfAborted();
    const controller = new AbortController();
    const abort = () => controller.abort();
    signal.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(abort, 180_000);
    try {
      const body = JSON.stringify({ model: route.model, stream: true, temperature: 0.2, max_tokens: maxTokens,
        ...jsonMode ? { response_format: { type: "json_object" } } : {},
        messages: [{ role: "system", content: system }, { role: "user", content: user }, ...history] });
      const response = await abortable(loggedFetch("desktop.facial-auto-setup", fetcher)(url.href, {
        method: "POST", signal: controller.signal,
        headers: { authorization: `Bearer ${route.apiKey}`, "content-type": "application/json" }, body,
      }), controller.signal);
      if (!response.ok) {
        if (jsonMode && response.status === 400) { jsonMode = false; attempt--; continue; }
        throw new Error(`分析请求失败 (${response.status})`);
      }
      const choice = await completionAnswer(response, controller.signal);
      if (choice.finishReason === "length") throw new Error("分析结果被截断");
      const answer = messageText(choice.content);
      if (answer === null) throw new Error("分析模型没有返回内容");
      try { return read(extractJson(answer)); } catch (error) {
        issue = error instanceof Error ? error.message : "格式无效";
        history.push({ role: "assistant", content: answer }, { role: "user", content: `上一份结果有问题：${issue}。请修正后只返回完整的 JSON 对象。` });
      }
    } finally {
      clearTimeout(timer); signal.removeEventListener("abort", abort);
    }
  }
  throw new Error(`分析结果格式不对：${issue}`);
}
