import { t } from "../i18n";
import { bindLocalizedAttribute, bindLocalizedText } from "../i18n/dom";
import { emotionJobActive, type EmotionAdoption, type EmotionMatchingJob, type EmotionSuggestion } from "../shared/emotion-matching";
import { hasUnsavedChanges, type ModelDraft } from "./draft";

export { mergeEmotionSuggestions, suggestionsForEmptyTargets, unappliedEmotionSuggestions } from "../shared/emotion-fill";
import { mergeEmotionSuggestions, suggestionsForEmptyTargets } from "../shared/emotion-fill";

export class EmotionSuggestionsPanel {
  private modelId: string | null = null;
  private job: EmotionMatchingJob | null = null;
  private refs: EmotionAdoption[] = [];
  private unsubscribe: (() => void) | null = null;
  private routeId = "";
  private routes: Array<{ id: string; name: string }> = [];
  private busy = false;
  private error = "";
  private notice = "";
  private fillJobId: string | null = null;

  constructor(private readonly root: HTMLElement, private readonly deps: {
    draft(): ModelDraft | null;
    changed(): void;
    preview(suggestion: EmotionSuggestion): Promise<void>;
    save(): Promise<boolean>;
  }) {}

  activate(): void {
    if (this.unsubscribe) return;
    this.unsubscribe = window.desktopPet.emotionMatching.onUpdate((job) => this.update(job));
    void window.desktopPet.getConfiguration().then((configuration) => {
      this.routes = configuration.models.text.map(({ id, name }) => ({ id, name }));
      if (!this.routeId) this.routeId = configuration.modelAssignments.reasoning;
      this.render();
    }).catch((error) => { this.error = String(error); this.render(); });
    if (this.modelId) this.fetch(this.modelId);
  }
  deactivate(): void { this.unsubscribe?.(); this.unsubscribe = null; }

  setModel(modelId: string): void {
    if (this.modelId === modelId) { this.render(); return; }
    this.modelId = modelId;
    this.job = null; this.refs = []; this.error = ""; this.notice = ""; this.fillJobId = null;
    this.fetch(modelId);
    this.render();
  }
  refresh(): void { this.render(); }
  resetDraft(): void { this.refs = []; this.notice = ""; this.render(); }
  editedCandidate(id: string): void { this.refs = this.refs.filter((ref) => ref.candidateId !== id); }
  adoptions(): EmotionAdoption[] { return structuredClone(this.refs); }

  private fetch(id: string): void {
    void window.desktopPet.emotionMatching.get(id).then((job) => { if (job) this.update(job); })
      .catch((error) => { if (this.modelId === id) { this.error = String(error); this.render(); } });
  }
  private update(job: EmotionMatchingJob): void {
    if (job.modelId !== this.modelId) return;
    if (this.job && (job.createdAt < this.job.createdAt || (job.id === this.job.id && job.revision <= this.job.revision))) return;
    this.job = job;
    if (this.fillJobId === job.id && (job.status === "completed" || job.status === "partial")) {
      this.fillJobId = null;
      void this.fill(job);
    }
    this.render();
  }

  private async start(): Promise<void> {
    const id = this.modelId;
    if (!id || this.busy) return;
    this.busy = true; this.error = ""; this.notice = ""; this.render();
    try {
      const job = await window.desktopPet.emotionMatching.start(id, this.routeId || undefined);
      this.fillJobId = job.id;
      this.update(job);
    } catch (error) { this.error = String(error); }
    finally { this.busy = false; this.render(); }
  }

  /** Fills only the states and emotions that are still empty, then saves like any edit. */
  private async fill(job: EmotionMatchingJob): Promise<void> {
    const draft = this.deps.draft();
    if (!draft || this.busy) return;
    const suggestions = suggestionsForEmptyTargets(draft.manifest, job.suggestions);
    if (!suggestions.length) { this.notice = "没有空着的项需要填"; this.render(); return; }
    const before = JSON.stringify(draft.manifest);
    this.busy = true; this.error = ""; this.notice = ""; this.render();
    try {
      const next = mergeEmotionSuggestions(draft.manifest, suggestions);
      const refs = suggestions.map((s) => ({ jobId: job.id, suggestionId: s.id, candidateId: s.candidate.id }));
      const adoptions = [...this.refs.filter((ref) => !refs.some((other) => other.candidateId === ref.candidateId)), ...refs];
      await window.desktopPet.emotionMatching.validate(job.modelId, next, adoptions);
      if (this.deps.draft() !== draft || JSON.stringify(draft.manifest) !== before) throw new Error("编辑内容已改变，请再点一次「让 AI 填空着的」");
      draft.manifest.behaviors = next.behaviors;
      this.refs = adoptions;
      this.deps.changed();
      if (!hasUnsavedChanges(draft)) this.refs = [];
      else if (!await this.deps.save()) throw new Error("保存没完成，填好的内容还在草稿里，可以点「保存并应用」");
      const targets = new Set(suggestions.map((s) => `${s.target.kind}:${s.target.key}`)).size;
      if (this.modelId === job.modelId) this.notice = `AI 填好了 ${targets} 项，都可以改`;
    } catch (error) { if (this.modelId === job.modelId) this.error = String(error); }
    finally { this.busy = false; this.render(); }
  }

  private render(): void {
    this.root.replaceChildren();
    const job = this.job;
    const running = !!(job && emotionJobActive(job));
    const strip = element("div", "emotion-analysis-actions le-ai-strip");
    const text = bindLocalizedText(element("span", "", ""), "{{copy}}", () => ({ copy: (running ? `${job!.message}${job!.totalAssets ? ` · ${job!.processedAssets}/${job!.totalAssets}` : ""}` : t("AI 可以看这个形象的动作和表情，帮你填空着的项；已经设置的不会动，填好的都能改")) }));
    const select = document.createElement("select");
    bindLocalizedAttribute(select, "aria-label", "用来分析的模型（要支持图片）");
    for (const route of this.routes) { const option = document.createElement("option"); option.value = route.id; option.textContent = route.name; select.append(option); }
    select.value = this.routeId;
    select.disabled = this.busy || running;
    select.addEventListener("change", () => { this.routeId = select.value; });
    strip.append(text, select, running
      ? this.button("取消", async () => { await window.desktopPet.emotionMatching.cancel(job!.id); })
      : this.button("让 AI 填空着的", () => this.start()));
    this.root.append(strip);
    if (this.error) this.root.append(element("p", "emotion-analysis-error", this.error));
    if (this.notice) {
      const notice = element("p", "emotion-analysis-status", this.notice);
      notice.setAttribute("role", "status");
      this.root.append(notice);
    }
    if (job && !running && job.failures.length) {
      const details = document.createElement("details");
      details.append(bindLocalizedText(element("summary", "", ""), "{{v0}} 个素材没看完，再点一次会接着看", { v0: job.failures.length }));
      for (const failure of job.failures.slice(0, 30)) {
        const manifest = this.deps.draft()?.manifest;
        const name = manifest?.motionAssets[failure.assetId]?.name ?? manifest?.expressionAssets[failure.assetId]?.name ?? failure.assetId;
        details.append(element("p", "emotion-analysis-error", `${name}：${failure.message}`));
      }
      this.root.append(details);
    }
  }

  private button(label: string, action: () => Promise<void>, disabled = false): HTMLButtonElement {
    const button = document.createElement("button"); button.type = "button"; button.className = "secondary"; bindLocalizedText(button, label); button.disabled = this.busy || disabled;
    button.addEventListener("click", () => { void action().catch((error) => { this.error = String(error); this.render(); }); });
    return button;
  }
}
function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
}
