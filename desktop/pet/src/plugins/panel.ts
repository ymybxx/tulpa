import { t, getLocale, subscribeLocale } from "../i18n";
import { localizeStatic, bindLocalizedAttribute, bindLocalizedText } from "../i18n/dom";
import "./styles.css";
import { ManualInputsPanel } from "./manual-inputs";

type Json = Record<string, any>;

export class PluginPanel {
  private visible = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private selected: string | null = null;
  private items: Json[] = [];
  private consumers: Json[] = [];
  private saving = false;
  private fields = new Map<string, { input: HTMLInputElement | HTMLSelectElement; schema: Json }>();
  private savedSecrets = new Set<string>();
  private reviewReports: Json[] = [];
  private reviewCursor: string | null = null;
  private reviewRevision = "";
  private manualInputs: ManualInputsPanel;

  constructor(private root: HTMLElement) {
    root.innerHTML = `<div class="plugin-shell">
      <header><div><span>EXTENSIONS · EXPERIMENTAL</span><h1 data-i18n="插件">插件</h1><p data-i18n="插件提供事件，宠物用现有的性格、记忆和声音回应。">插件提供事件，宠物用现有的性格、记忆和声音回应。</p></div>
        <div class="plugin-actions"><button data-action="builtin" data-i18n="安装直播搭档">安装直播搭档</button><button data-action="import" data-i18n="导入插件包">导入插件包</button><button data-action="directory" data-i18n="加载开发目录">加载开发目录</button></div></header>
      <p class="plugin-notice" role="status"></p>
      <div class="plugin-layout"><aside><div class="plugin-list"></div><p class="plugin-empty" data-i18n="还没有安装插件">还没有安装插件</p></aside>
      <main class="plugin-detail" hidden><h2 class="plugin-name"></h2><p class="plugin-description"></p>
      <p data-i18n="插件事件自动进入宠物当前聊天会话，随聊天会话切换。">插件事件自动进入宠物当前聊天会话，随聊天会话切换。</p>
      <p class="plugin-playback-hint"></p><p class="plugin-activity" hidden></p><form class="plugin-config"></form>
      <div class="plugin-actions"><button data-action="start" data-i18n="启动">启动</button><button data-action="stop" data-i18n="停止">停止</button><label><input type="checkbox" class="plugin-clear-secrets"><span data-i18n="清除已保存的密钥">清除已保存的密钥</span></label></div>
      <section class="plugin-manual-inputs" hidden></section>
      <section class="plugin-reviews" aria-label="运行建议" data-i18n-aria-label="运行建议">
        <div class="plugin-review-heading"><h3 data-i18n="运行建议">运行建议</h3><button data-action="review" data-i18n="分析当前片段">分析当前片段</button></div>
        <p class="plugin-review-status" role="status"></p>
        <details class="plugin-review-goals"><summary data-i18n="评估目标与参考">评估目标与参考</summary><div class="plugin-review-goal-content"></div></details>
        <div class="plugin-review-history"></div>
        <button data-action="more-reviews" hidden data-i18n="更早的建议">更早的建议</button>
      </section>
      <details class="plugin-developer"><summary data-i18n="开发与事件测试">开发与事件测试</summary>
        <label><input type="checkbox" class="plugin-dev-mode"><span data-i18n="允许本机开发工具连接当前宠物">允许本机开发工具连接当前宠物</span></label>
        <p><span data-i18n="在插件目录运行">在插件目录运行 </span><code>fish-plugin dev .</code><span data-i18n="，保存代码后仅重启插件。">，保存代码后仅重启插件。</span></p>
        <label><span data-i18n="测试事件">测试事件</span><select class="plugin-event-type"></select></label>
        <textarea class="plugin-event-data" rows="5" spellcheck="false" aria-label="事件 JSON 数据" data-i18n-aria-label="事件 JSON 数据"></textarea>
        <div class="plugin-actions"><button data-action="event" data-i18n="发送事件">发送事件</button><button data-action="repeat" data-i18n="重发同一事件">重发同一事件</button><button data-action="context" data-i18n="查看实时状态">查看实时状态</button></div>
        <p class="plugin-event-status" role="status"></p><pre class="plugin-diagnostics"></pre>
      </details><details><summary data-i18n="运行日志">运行日志</summary><pre class="plugin-logs"></pre></details>
      </main></div></div>`;
    localizeStatic(root);
    const stopLocale = subscribeLocale(() => {
      // Refresh status and review labels without replacing editable configuration fields.
      if (this.visible) {
        void this.refresh();
        void this.loadReviews().catch((error) => this.notify(String(error), true));
      }
    });
    window.addEventListener("beforeunload", stopLocale, { once: true });
    this.manualInputs = new ManualInputsPanel(this.el("plugin-manual-inputs"), (action, input) => window.desktopPet.pluginCall(action, input));
    root.addEventListener("click", (event) => {
      const action = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-action]")?.dataset.action;
      if (action) void this.perform(action);
    });
    this.el<HTMLInputElement>("plugin-dev-mode").onchange = () => void this.perform("development");
    this.el<HTMLSelectElement>("plugin-event-type").onchange = () => this.eventExample();
    this.el<HTMLFormElement>("plugin-config").onsubmit = (e) => e.preventDefault();
  }

  setVisible(visible: boolean): void {
    if (this.visible === visible) return;
    this.visible = visible;
    if (this.timer) clearTimeout(this.timer);
    if (visible) void this.refresh();
  }

  private el<T extends HTMLElement = HTMLElement>(name: string): T {
    return this.root.querySelector(`.${name}`) as T;
  }

  private notify(message: string, error = false): void {
    const element = this.el("plugin-notice");
    bindLocalizedText(element, message);
    element.classList.toggle("error", error);
  }

  private async refresh(): Promise<void> {
    if (this.timer) clearTimeout(this.timer);
    try {
      const data = await window.desktopPet.pluginCall("list");
      this.items = data.plugins;
      this.consumers = data.consumers;
      this.el<HTMLInputElement>("plugin-dev-mode").checked = data.development;
      const list = this.el("plugin-list");
      list.replaceChildren();
      this.el("plugin-empty").hidden = this.items.length > 0;
      for (const item of this.items) {
        const button = document.createElement("button");
        button.className = this.selected === item.id ? "selected" : "";
        const status: Json = { running: t("运行中"), stopped: t("已停止"), starting: t("连接中"), failed: t("启动失败") };
        button.textContent = `${item.manifest.name} · ${status[item.status] ?? item.status}`;
        button.onclick = () => void this.select(item.id);
        list.append(button);
      }
      if (!this.selected && this.items[0]) await this.select(this.items[0].id);
      const item = this.items.find((i) => i.id === this.selected);
      if (item) {
        const name = this.el("plugin-name");
        const title = `${item.manifest.name}  ${item.manifest.version}`;
        if (name.textContent !== title) name.textContent = title;
        const description = this.el("plugin-description");
        if (description.textContent !== item.manifest.description) description.textContent = item.manifest.description;
        const readInputResults = this.manualInputs.needsResults;
        this.manualInputs.update(item);
        if (readInputResults) this.manualInputs.diagnostics(item.id, await window.desktopPet.pluginCall("diagnostics", { id: item.id }));
        this.el("plugin-logs").textContent = item.logs.join("\n");
        const native = this.consumers.some((c) => c.fish_id === item.fish_id && c.session_id === item.session_id);
        const state = item.state;
        const activity = state?.connected && !state.stopped ? state.activity : null;
        this.el("plugin-activity").hidden = !activity;
        this.el("plugin-activity").textContent = activity
          ? t("当前活动：{{p0}} · {{p1}}。{{p2}}", { p0: activity.name, p1: activity.role, p2: activity.objective }) : "";
        const last = state?.last_response;
        let hint = t("播放已连接，沿用宠物当前音色和音频链路。");
        if (!native) hint = t("事件会记录到宠物当前聊天会话；语音播放尚未连接。");
        else if (last?.status === "failed" && last.reason !== "cancelled") hint = last.reason === "invalid_response" ? t("最近一次发言失败：模型没有返回有效的发言格式。可展开实时状态查看结果。") : t("最近一次发言失败，请查看运行日志。");
        else if (state?.reason === "user_busy") hint = t("正在等待你说完，再寻找发言机会。");
        else if (state?.reason === "pet_busy") hint = t("宠物正在生成或播放发言。");
        else if (last?.status === "silent") hint = t("刚才选择暂时安静，插件会继续寻找发言机会。");
        this.el("plugin-playback-hint").textContent = hint;
        this.reviewState(item);
        const revision = `${item.id}:${item.review?.last_review_at ?? ""}`;
        if (revision !== this.reviewRevision) {
          this.reviewRevision = revision;
          await this.loadReviews();
        }
      }
    } catch (error) { this.notify(String(error), true); }
    finally { if (this.visible) this.timer = setTimeout(() => void this.refresh(), 2000); }
  }

  private async select(id: string): Promise<void> {
    this.selected = id;
    this.reviewReports = [];
    this.reviewCursor = null;
    this.reviewRevision = "";
    const item = this.items.find((i) => i.id === id)!;
    this.manualInputs.update(item);
    this.el("plugin-detail").hidden = false;
    this.el("plugin-name").textContent = `${item.manifest.name}  ${item.manifest.version}`;
    this.el("plugin-description").textContent = item.manifest.description;
    const saved = await window.desktopPet.pluginCall("config", { id });
    this.savedSecrets = new Set(saved.savedSecrets);
    const form = this.el("plugin-config");
    form.replaceChildren(); this.fields.clear();
    for (const [key, schema] of Object.entries<Json>(item.description.config_schema.properties ?? {})) {
      const label = document.createElement("label");
      const title = document.createElement("span"); title.textContent = schema.title ?? key;
      const input = schema.enum ? document.createElement("select") : document.createElement("input");
      if (input instanceof HTMLSelectElement) for (const value of schema.enum) input.add(new Option(String(value), String(value)));
      else input.type = schema.writeOnly || schema.format === "password" ? "password" : schema.type === "boolean" ? "checkbox" : ["integer", "number"].includes(schema.type) ? "number" : "text";
      input.autocomplete = "off";
      const value = saved.values[key] ?? schema.default ?? "";
      if (input instanceof HTMLInputElement && input.type === "checkbox") input.checked = Boolean(value);
      else input.value = String(value);
      if (input instanceof HTMLInputElement && this.savedSecrets.has(key)) bindLocalizedAttribute(input, "placeholder", "已保存；留空保留");
      label.append(title, input);
      if (schema.description) { const hint = document.createElement("small"); hint.textContent = schema.description; label.append(hint); }
      form.append(label); this.fields.set(key, { input, schema });
    }
    this.el<HTMLSelectElement>("plugin-event-type").replaceChildren(...item.description.events.map((event: Json) => new Option(`${event.type} · speak=${event.speak}`, event.type)));
    this.eventExample();
    this.reviewState(item);
    await this.loadReviews();
  }

  private reviewState(item: Json): void {
    const review = item.review;
    const labels: Json = {
      observing: t("正在积累新的运行片段"), queued: t("已排队，等待后台分析"),
      analysing: t("正在分析本次运行片段"), waiting_user: t("等待当前对话结束后再分析"),
      insufficient: t("终态事件还不够，继续观察"), failed: t("上次分析未完成，插件继续运行"),
      stopped: t("插件已停止，建议历史仍可查看"),
    };
    this.el("plugin-review-status").textContent = review
      ? t("{{p0}}。自动分析间隔约 {{p1}} 分钟，至少 {{p2}} 个终态事件。建议仅供查看。", { p0: labels[review.phase] ?? review.phase, p1: Math.round(review.interval_seconds / 60), p2: review.minimum_events })
      : item.description.review ? t("启动插件后自动分析运行片段，建议会保存在这里。") : t("此插件尚未声明运行评估目标。");
    const button = this.root.querySelector<HTMLButtonElement>('[data-action="review"]')!;
    button.disabled = !review || !item.description.review || ["queued", "analysing"].includes(review.phase);
    const goals = this.el("plugin-review-goal-content");
    goals.replaceChildren();
    const profile = item.description.review;
    this.el<HTMLDetailsElement>("plugin-review-goals").hidden = !profile;
    if (profile) {
      goals.append(textNode("p", profile.goal));
      for (const criterion of profile.criteria ?? []) goals.append(textNode("p", `${criterion.name}：${criterion.description}`));
      for (const ref of profile.references ?? []) {
        if (typeof ref.url !== "string" || !ref.url.startsWith("https://")) continue;
        const link = document.createElement("a");
        link.href = ref.url; link.target = "_blank"; link.rel = "noopener noreferrer";
        link.textContent = ref.title;
        goals.append(link, textNode("small", ref.note ?? ""));
      }
    }
  }

  private async loadReviews(more = false): Promise<void> {
    const id = this.selected;
    if (!id) return;
    const data = await window.desktopPet.pluginCall("reviews", { id, ...(more ? { cursor: this.reviewCursor } : {}) });
    if (this.selected !== id) return;
    const initiallyEmpty = this.reviewReports.length === 0;
    const reports = new Map([...this.reviewReports, ...data.items].map((r: Json) => [r.id, r]));
    this.reviewReports = [...reports.values()].sort((a, b) => b.created_at.localeCompare(a.created_at));
    if (more || initiallyEmpty) this.reviewCursor = data.next_cursor;
    const container = this.el("plugin-review-history");
    const opened = new Set([...container.querySelectorAll<HTMLDetailsElement>("details[open]")].map((d) => d.dataset.reportId));
    container.replaceChildren();
    if (!this.reviewReports.length) container.append(textNode("p", t("还没有建议记录。启动后的真实事件、已播内容和打断回执将用于分析。")));
    for (const report of this.reviewReports) {
      const card = document.createElement("details"); card.className = "plugin-review-card";
      card.dataset.reportId = report.id;
      card.open = opened.has(report.id) || (initiallyEmpty && report === this.reviewReports[0]);
      const status = report.status === "failed" ? t("分析失败") : report.status === "insufficient" ? t("证据不足") : t("运行建议");
      card.append(textNode("summary", `${formatTime(report.created_at)} · ${status} · v${report.plugin_version}`));
      card.append(textNode("small", t("片段：{{p0}} — {{p1}} · {{p2}}", { p0: formatTime(report.started_at), p1: formatTime(report.ended_at), p2: report.trigger === "automatic" ? t("自动分析") : t("手动分析") })));
      card.append(textNode("small", t("宠物 {{p0}} · 会话 {{p1}} · 运行 {{p2}}", { p0: report.fish_id.slice(0, 8), p1: report.session_id.slice(0, 8), p2: report.run_id.slice(0, 8) })));
      if (!report.context_version) card.append(textNode("small", t("旧版评估未收录普通聊天回复，关于未回应用户的结论可能不完整。")));
      if (report.error) card.append(textNode("p", report.error));
      if (report.omitted_earlier_events > 0) card.append(textNode("small", t("本次分析最近的连续片段，较早的 {{p0}} 个事件未纳入结论；原始记录仍保留。", { p0: report.omitted_earlier_events })));
      if (report.analysis_metrics?.duration_ms != null) {
        const timing = report.analysis_metrics;
        card.append(textNode("small", t("分析耗时 {{p0}} 秒 · 输入 {{p1}} 字符 · 已返回 {{p2}} 字符{{p3}}", { p0: (timing.duration_ms / 1000).toFixed(1), p1: timing.input_chars, p2: timing.output_chars, p3: timing.first_content_ms != null ? t(" · 首段 {{p0}} 秒", { p0: (timing.first_content_ms / 1000).toFixed(1) }) : t(" · 尚未返回内容") })));
      }
      const metrics = report.metrics;
      if (metrics) {
        const stats = textNode("p", t("采样 {{p0}}/{{p1}} 个事件 · 实际发声 {{p2}} 段 · 中断 {{p3}} 次 · 完全重复 {{p4}} 段", { p0: metrics.sampled_events, p1: metrics.total_events, p2: metrics.spoken_responses, p3: metrics.status_counts.interrupted ?? 0, p4: metrics.exact_duplicate_responses }));
        stats.className = "plugin-review-metrics"; card.append(stats);
        if (typeof metrics.after_playback_to_request_median_seconds === "number") card.append(textNode("small", t("播完到下次请求间隔中位数：{{p0}} 秒（不含下一段生成与 TTS 时间）", { p0: metrics.after_playback_to_request_median_seconds })));
      }
      const assessment = report.assessment;
      if (assessment) {
        card.append(textNode("p", assessment.summary));
        for (const strength of assessment.strengths) card.append(textNode("p", t("可保留：{{p0}}", { p0: strength })));
        const evidence = new Map((report.evidence ?? []).map((e: Json) => [e.id, e]));
        if (assessment.turns?.length) {
          const turns = document.createElement("details"); turns.className = "plugin-review-evidence";
          turns.append(textNode("summary", t("逐段内容检查（{{p0}} 段）", { p0: assessment.turns.length })));
          const connections: Json = { specific: t("针对具体内容"), generic: t("泛泛而谈"), unrelated: t("与素材无关"), uncertain: t("证据不足"), not_applicable: t("自主话题") };
          const grounding: Json = { supported: t("有证据支持"), contradicted: t("与证据冲突"), uncertain: t("依据不足"), not_applicable: t("无需事实核对") };
          for (const turn of assessment.turns) {
            const item = evidence.get(turn.evidence_id) as Json | undefined;
            turns.append(textNode("blockquote", `${turn.evidence_id} · ${item?.text ?? t("无已播文字")}`));
            if (turn.material) turns.append(textNode("p", t("可用素材：{{p0}}", { p0: turn.material })));
            if (turn.grounding) turns.append(textNode("p", t("理解核对：{{p0}} · {{p1}}", { p0: grounding[turn.grounding] ?? turn.grounding, p1: turn.grounding_reason ?? "" })));
            else turns.append(textNode("small", t("旧版报告未单独核对内容理解。")));
            turns.append(textNode("p", t("实际新增内容：{{p0}}", { p0: turn.contribution || t("未体现独立内容") })));
            turns.append(textNode("small", `${connections[turn.connection] ?? turn.connection}${turn.depends_on_reply ? t(" · 内容依赖别人接话") : ""}${turn.repeats?.length ? t(" · 重复 {{p0}}", { p0: turn.repeats.join("、") }) : ""}`));
          }
          card.append(turns);
        }
        for (const finding of assessment.findings) {
          const section = document.createElement("div"); section.className = "plugin-review-finding";
          section.append(textNode("h4", `${finding.severity === "major" ? t("主要") : t("次要")} · ${finding.title}`));
          const targets: Json = { plugin_logic: t("插件逻辑"), event_description: t("事件描述"), context: t("上下文"), audio: t("音频链路"), model: t("模型能力") };
          const confidence: Json = { high: t("高"), medium: t("中"), low: t("低") };
          section.append(textNode("small", t("{{p0}} · {{p1}} · 置信度{{p2}}", { p0: finding.criterion, p1: targets[finding.target] ?? finding.target, p2: confidence[finding.confidence] ?? finding.confidence })));
          section.append(textNode("p", finding.observation), textNode("p", t("建议：{{p0}}", { p0: finding.suggestion })));
          const quotes = document.createElement("details"); quotes.className = "plugin-review-evidence";
          quotes.append(textNode("summary", t("查看证据（{{p0}} 条）", { p0: finding.evidence_ids.length })));
          for (const key of finding.evidence_ids) {
            const item = evidence.get(key) as Json | undefined;
            if (!item) continue;
            if (item.source === "plugin" && typeof item.data?.text === "string") quotes.append(textNode("blockquote", t("事件输入{{p0}}：{{p1}}", { p0: item.data.viewer_name ? ` · ${item.data.viewer_name}` : "", p1: item.data.text })));
            const speaker = item.source === "user" ? t("用户") : item.source === "assistant" ? t("普通聊天回复") : item.type;
            quotes.append(textNode("blockquote", `${key} · ${formatTime(item.at)} · ${speaker}\n${item.text || t("无已播文本（{{p0}}）", { p0: item.status ?? t("未知") })}`));
            if (item.delivery === "visible_text") quotes.append(textNode("small", t("可见文字，未确认语音播放")));
            if (item.source === "plugin" && "screen_frame" in item) quotes.append(textNode("small", item.screen_frame?.observedAt
              ? t("这次发言附带了 {{p0}} 的画面", { p0: formatTime(item.screen_frame.observedAt) })
              : t("这次发言没有截图附件；文字感知单独提供")));
            if (item.event_id) quotes.append(textNode("small", t("事件 {{p0}}", { p0: item.event_id })));
            if (item.context_audit) {
              const audit = item.context_audit;
              quotes.append(textNode("small", t("请求时的活动：{{p0}} · 历史 {{p1}} 条", { p0: audit.activity?.name ?? t("未声明"), p1: audit.history_messages })));
              if (audit.latest_user_input) quotes.append(textNode("blockquote", t("请求已携带的最近用户原话：{{p0}}", { p0: audit.latest_user_input })));
            }
          }
          section.append(quotes);
          card.append(section);
        }
        for (const limitation of assessment.limitations) card.append(textNode("p", t("证据范围：{{p0}}", { p0: limitation })));
      }
      container.append(card);
    }
    this.root.querySelector<HTMLButtonElement>('[data-action="more-reviews"]')!.hidden = !this.reviewCursor;
  }

  private eventExample(): void {
    const item = this.items.find((i) => i.id === this.selected);
    const type = this.el<HTMLSelectElement>("plugin-event-type").value;
    const schema = item?.description.events.find((e: Json) => e.type === type)?.schema;
    this.el<HTMLTextAreaElement>("plugin-event-data").value = JSON.stringify(example(schema ?? {}), null, 2);
  }

  private lastEvent: Json | null = null;
  private updateItem(item: Json): void {
    const index = this.items.findIndex((value) => value.id === item.id);
    if (index < 0) this.items.push(item);
    else this.items[index] = item;
  }

  private async perform(action: string): Promise<void> {
    if (this.saving) return;
    this.saving = true;
    this.notify("正在处理…");
    try {
      const id = this.selected;
      if (action === "builtin") {
        const installed = await window.desktopPet.pluginCall("builtin");
        this.updateItem(installed);
        await this.select(installed.id);
      }
      else if (action === "import" || action === "directory") await window.desktopPet.pickPlugin(action === "directory");
      else if (action === "start") {
        const config: Json = {};
        for (const [key, { input, schema }] of this.fields) {
          if (this.savedSecrets.has(key) && input.value === "" && !this.el<HTMLInputElement>("plugin-clear-secrets").checked) continue;
          config[key] = schema.type === "boolean" ? (input as HTMLInputElement).checked : ["integer", "number"].includes(schema.type) ? Number(input.value) : input.value;
        }
        const started = await window.desktopPet.pluginCall("start", { id, config });
        this.updateItem(started);
        if (id) await this.select(id);
      } else if (action === "stop") await window.desktopPet.pluginCall("stop", { id });
      else if (action === "review") await window.desktopPet.pluginCall("review", { id });
      else if (action === "more-reviews") await this.loadReviews(true);
      else if (action === "development") await window.desktopPet.pluginCall("development", { enabled: this.el<HTMLInputElement>("plugin-dev-mode").checked });
      else if (action === "context") this.el("plugin-diagnostics").textContent = JSON.stringify(await window.desktopPet.pluginCall("diagnostics", { id }), null, 2);
      else if (action === "event" || action === "repeat") {
        if (action === "event") this.lastEvent = { event_id: crypto.randomUUID(), type: this.el<HTMLSelectElement>("plugin-event-type").value, data: JSON.parse(this.el<HTMLTextAreaElement>("plugin-event-data").value) };
        if (!this.lastEvent) throw new Error(t("请先发送一条事件"));
        const result = await window.desktopPet.pluginCall("event", { id, event: this.lastEvent });
        const reasons: Json = { no_playback_consumer: t("尚未连接播放端"), pet_busy: t("宠物正在说话"), user_busy: t("用户正在交流"), silent_event: t("仅记录事件"), disconnected: t("插件已断开") };
        this.el("plugin-event-status").textContent = result.duplicate ? t("重复事件：返回已有记录，不会再次处理。") : result.reason === "experimental_event" ? t("已记录，按普通事件处理；是否回应由实验模式判断。") : result.disposition === "responding" ? t("已接收，正在处理响应；生成结束后还会等待实际播放完成。") : t("已记录，本次不发声：{{p0}}。", { p0: reasons[result.reason] ?? result.reason ?? t("仅存储") });
        this.el("plugin-diagnostics").textContent = JSON.stringify(result, null, 2);
      }
      this.notify("已完成");
      await this.refresh();
    } catch (error) { this.notify(error instanceof Error ? error.message : String(error), true); }
    finally { this.saving = false; }
  }
}

function textNode(tag: string, value: unknown): HTMLElement {
  const node = document.createElement(tag);
  node.textContent = String(value ?? "");
  return node;
}

function formatTime(value: string): string {
  return new Date(value).toLocaleString(getLocale(), { hour12: false });
}

function example(schema: Json): unknown {
  if (schema.default !== undefined) return schema.default;
  if (schema.examples?.length) return schema.examples[0];
  if (schema.enum?.length) return schema.enum[0];
  if (schema.type === "object") return Object.fromEntries(Object.entries<Json>(schema.properties ?? {}).map(([k, v]) => [k, example(v)]));
  if (schema.type === "array") return [];
  if (schema.type === "boolean") return false;
  if (schema.type === "integer" || schema.type === "number") return schema.minimum ?? 0;
  return schema.minLength ? "测试" : "";
}
