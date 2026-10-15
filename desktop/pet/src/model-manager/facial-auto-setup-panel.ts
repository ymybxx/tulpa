import { t } from "../i18n";
import { bindLocalizedText } from "../i18n/dom";
import type { NativeModelParameter } from "../shared/contracts";
import type { FacialAutoSetupContext, FacialAutoSetupProgress, FacialAutoSetupProposal } from "../shared/facial-test";
import type { FacialMotionStyle } from "../shared/facial-bindings";
import { STANDARD_PARAMETERS } from "../facial-demo/parameter-bindings";
import { editorButton } from "./facial-editor-layout";

const MOTION_TEXT: Record<FacialMotionStyle, string> = { hold: "保持姿势", swing: "来回摆动", pulse: "做一下就回来" };

/**
 * One 一键配置 run, owned by the 说话表情 panel rather than its card: leaving
 * the page, switching apps or opening another part's settings only hides the
 * card, and it comes back showing wherever the run has got to.
 */
export class AutoSetupJob {
  progress: FacialAutoSetupProgress | null = null;
  result: FacialAutoSetupProposal | null = null;
  error: string | null = null;
  private readonly listeners = new Set<() => void>();

  constructor(readonly modelId: string, readonly context: FacialAutoSetupContext) {
    const stopProgress = window.desktopPet.facialTest.onAutoSetupProgress(progress => { this.progress = progress; this.changed(); });
    void window.desktopPet.facialTest.autoSetup(modelId, context).then(
      proposal => { this.result = proposal; },
      error => {
        const message = error instanceof Error ? error.message.replace(/^Error invoking remote method '[^']+': (Error: )?/u, "") : String(error);
        this.error = message;
      },
    ).finally(() => { stopProgress(); this.changed(); });
  }

  get running(): boolean { return !this.result && this.error === null; }
  cancel(): void { if (this.running) void window.desktopPet.facialTest.cancelAutoSetup(); }
  subscribe(listener: () => void): () => void { this.listeners.add(listener); return () => this.listeners.delete(listener); }
  private changed(): void { for (const listener of this.listeners) listener(); }
}

/**
 * 一键配置, in the order a person needs it: what it will do (nothing is
 * changed until they pick), how far along it is, then what it found, each
 * item ticked by default and shown on the model when hovered.
 */
export function facialAutoSetupPanel(options: {
  context: FacialAutoSetupContext;
  native: readonly NativeModelParameter[];
  job(): AutoSetupJob | null;
  start(): void;
  /** Stops a running job, or forgets a finished one, and closes the card. */
  dismiss(): void;
  /** Shows one parameter at a value on the preview; null puts the model back. */
  show(parameterId: string | null, value?: number): void;
  save(proposal: FacialAutoSetupProposal): Promise<void>;
}): { element: HTMLElement; detach(): void } {
  const root = element("section", "fbe-card fas");
  root.dataset.autoSetup = "";
  let unsubscribe: (() => void) | null = null;
  let shown: "confirm" | "running" | "review" | "error" | null = null;

  const head = (title: string) => {
    const bar = element("div", "fbe-card__head");
    bar.append(bindLocalizedText(element("h4"), title));
    return bar;
  };

  const confirm = () => {
    const open = options.context.open.length, slots = options.context.customSlots;
    const list = element("ul", "fas-plan");
    list.append(
      bindLocalizedText(item(""), "{{copy}}", () => ({ copy: (open ? t("补上对应不上的部位：现在还有 {{v0}} 个标准部位没对应到模型参数，很多只是作者起了别的名字。", { v0: open }) : t("标准部位都已经对应上了，这一项会跳过。")) })),
      bindLocalizedText(item(""), "{{copy}}", () => ({ copy: (slots ? t("加上有用的表情和动作：比如生气、脸红、星星眼、托腮、挥手，最多还能加 {{v0}} 个。", { v0: slots }) : t("自定义部位已经满 16 个，不会再加新的。")) })),
      bindLocalizedText(item(""), "发型、饰品这类装扮开关，头发、衣服的物理摆动，以及呼吸、眨眼、口型，都会跳过。"),
    );
    const note = element("p", "fbe-note");
    bindLocalizedText(note, "现有的设置不会被改动。看完后会列出找到的内容，由你挑选再保存。大约需要 2～4 分钟。");
    const foot = element("div", "fbe-foot");
    foot.append(bindLocalizedText(editorButton("", "autoSetupCancel", "le-link le-link--muted", () => options.dismiss()), "取消"),
      element("span", "fbe-grow"), bindLocalizedText(editorButton("", "autoSetupStart", "accent", () => { options.start(); render(); }), "开始"));
    root.replaceChildren(head("AI 一键配置"), bindLocalizedText(element("p", "fbe-note", ""), "AI 会把这个模型的参数挨个拨一下，看它们动了哪里，然后："), list, note, foot);
  };

  let status: HTMLElement | null = null, fill: HTMLElement | null = null, bar: HTMLElement | null = null;
  const running = (job: AutoSetupJob) => {
    if (shown !== "running") {
      status = element("p", "fas-status");
      bar = element("div", "fas-bar"); fill = element("span"); bar.append(fill);
      const foot = element("div", "fbe-foot");
      foot.append(bindLocalizedText(element("p", "fbe-feedback", ""), "可以先去做别的，看完会停在这里等你。"),
        bindLocalizedText(editorButton("", "autoSetupCancel", "secondary", () => options.dismiss()), "取消"));
      root.replaceChildren(head("AI 一键配置"), status, bar, foot);
    }
    const progress = job.progress;
    const count = progress?.total ? `（${progress.done ?? 0}/${progress.total}）` : "";
    bindLocalizedText(status!, "{{copy}}", () => ({ copy: (progress ? `${progress.text}${count}…` : t("正在准备…")) }));
    bar!.classList.toggle("fas-bar--known", !!progress?.total);
    fill!.style.width = progress?.total ? `${Math.round(100 * (progress.done ?? 0) / progress.total)}%` : "";
  };

  const failed = (message: string) => {
    const foot = element("div", "fbe-foot");
    foot.append(element("span", "fbe-grow"), bindLocalizedText(editorButton("", "autoSetupClose", "secondary", () => options.dismiss()), "关闭"));
    root.replaceChildren(head("AI 一键配置"), bindLocalizedText(element("p", "fas-status", ""), "{{copy}}", () => ({ copy: (message === "已取消" ? t("已取消。") : t("没能完成：{{v0}}", { v0: message })) })), foot);
  };

  const review = (proposal: FacialAutoSetupProposal) => {
    const picked = new Set<string>([...proposal.standard, ...proposal.custom].map(entry => entry.parameterId));
    const saveButton = editorButton("", "autoSetupSave", "accent", () => void save());
    const refresh = () => { bindLocalizedText(saveButton, "保存选中的（{{v0}}）", { v0: picked.size }); saveButton.disabled = !picked.size; };
    const strong = (parameterId: string, value?: number) => {
      const parameter = options.native.find(item => item.id === parameterId);
      return value ?? parameter?.maximum ?? 0;
    };
    const row = (parameterId: string, title: string, detail: string, strongValue: number) => {
      const line = element("label", "fas-row");
      const box = document.createElement("input"); box.type = "checkbox"; box.checked = picked.has(parameterId);
      box.dataset.autoSetupPick = parameterId;
      box.addEventListener("change", () => { if (box.checked) picked.add(parameterId); else picked.delete(parameterId); refresh(); });
      const text = element("span", "fas-row__text");
      text.append(element("b", "", title), element("small", "", detail));
      line.append(box, text);
      line.addEventListener("mouseenter", () => options.show(parameterId, strongValue));
      line.addEventListener("mouseleave", () => options.show(null));
      return line;
    };
    const section = (title: string, rows: HTMLElement[]) => {
      if (!rows.length) return [];
      const wrap = element("div", "fas-section");
      wrap.append(element("h5", "", `${title}（${rows.length}）`), ...rows);
      return [wrap];
    };
    const standardRows = proposal.standard.map(entry => {
      const label = STANDARD_PARAMETERS.find(value => value.id === entry.standardId)?.label ?? entry.standardId;
      return row(entry.parameterId, label, `对应 ${entry.parameterName}（${entry.parameterId}）${entry.binding.lowerValue > entry.binding.upperValue ? "，方向相反" : ""}`,
        entry.binding.upperValue);
    });
    const customRow = (entry: FacialAutoSetupProposal["custom"][number]) => row(entry.parameterId, entry.value.label,
      `${entry.value.mode === "binary" ? "有或没有" : `分 ${entry.value.levels.length} 档`} · ${MOTION_TEXT[entry.value.motion ?? "hold"]} · ${entry.value.description}`,
      strong(entry.parameterId));
    const expressions = proposal.custom.filter(entry => entry.kind === "expression").map(customRow);
    const motions = proposal.custom.filter(entry => entry.kind === "motion").map(customRow);
    const quiet = (title: string, entries: Array<{ parameterName: string; parameterId: string; text: string }>) => {
      if (!entries.length) return [];
      const details = element("details", "fas-quiet");
      details.append(element("summary", "", `${title}（${entries.length}）`),
        ...entries.map(entry => element("div", "fas-quiet__row", `${entry.parameterName} · ${entry.text}`)));
      return [details];
    };
    const found = standardRows.length + expressions.length + motions.length;
    const summary = bindLocalizedText(element("p", "fbe-note", ""), (found ? "这些是找到的内容，默认都选上了。鼠标移到某一项上，左边会做出这个效果；不想要的取消勾选。" : "没有找到值得加的部位。跳过的参数和原因列在下面。"));
    const foot = element("div", "fbe-foot");
    const feedback = element("p", "fbe-feedback");
    foot.append(bindLocalizedText(editorButton("", "autoSetupClose", "le-link le-link--muted", () => options.dismiss()), "不用了"), feedback, saveButton);
    const save = async () => {
      saveButton.disabled = true; bindLocalizedText(feedback, "正在保存…");
      try {
        await options.save({
          standard: proposal.standard.filter(entry => picked.has(entry.parameterId)),
          custom: proposal.custom.filter(entry => picked.has(entry.parameterId)),
          skipped: [], failed: [],
        });
      } catch (error) {
        bindLocalizedText(feedback, "保存失败：{{v0}}", { v0: error instanceof Error ? error.message : String(error) });
        saveButton.disabled = false;
      }
    };
    refresh();
    root.replaceChildren(head("AI 一键配置 · 找到的内容"), summary,
      ...section("补上的标准部位", standardRows), ...section("新表情", expressions), ...section("新动作", motions),
      ...quiet("没用上的参数", proposal.skipped.map(entry => ({ ...entry, text: entry.reason }))),
      ...quiet("没看成的参数", proposal.failed.map(entry => ({ ...entry, text: entry.error }))),
      foot);
    if (!found) saveButton.hidden = true;
  };

  const render = () => {
    const job = options.job();
    const next = !job ? "confirm" : job.result ? "review" : job.error !== null ? "error" : "running";
    if (job && !unsubscribe) unsubscribe = job.subscribe(render);
    if (next === "confirm") confirm();
    else if (next === "running") running(job!);
    else if (next !== shown) next === "review" ? review(job!.result!) : failed(job!.error!);
    shown = next;
  };
  render();
  return {
    element: root,
    /** Hides the card only; the job keeps going. */
    detach: () => { options.show(null); unsubscribe?.(); unsubscribe = null; },
  };
}

function item(text: string): HTMLElement { return element("li", "", text); }

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
