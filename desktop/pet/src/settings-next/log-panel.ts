import { getLocale, subscribeLocale, t } from "../i18n";
import "./log-styles.css";
import {
  LOG_STATUS_NAMES,
  LOG_TYPE_NAMES,
  logTypeName,
  type LlmLogDetail,
  type LlmLogSummary,
} from "../shared/llm-logs";
import { readableRequest, readableResponse } from "./log-json";

export class LlmLogPanel {
  private visible = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private version = 0;
  private detailVersion = 0;
  private cursor: number | null = null;
  private selected: string | null = null;
  private detail: LlmLogDetail | null = null;
  private tab = "request";
  private stopLocale: (() => void) | null = null;
  private chromeText: Array<{ node: Text; source: string }> = [];
  private rows = new Map<HTMLElement, LlmLogSummary>();
  private notice: { source: string; values?: Record<string, string | number> } = { source: "" };

  constructor(private root: HTMLElement) {
    root.innerHTML = `<header class="llm-log-heading"><div><h2>LLM 调用日志</h2><p>查看每次实际发送的请求与收到的返回。</p></div><button type="button" data-log-action="refresh">刷新</button></header>
      <div class="llm-log-filters"><label>调用类型<select class="log-type"><option value="">全部类型</option></select></label>
      <label>结果<select class="log-status"><option value="">全部结果</option><option value="completed">已完成</option><option value="running">进行中</option><option value="failed">失败</option><option value="cancelled">已取消</option></select></label>
      <label class="log-auto-label"><input type="checkbox" class="log-auto">自动刷新</label></div>
      <p class="log-notice" role="status"></p><div class="llm-log-layout"><aside class="log-list-panel"><div class="log-list"></div><button class="log-more" type="button" hidden>加载更早记录</button></aside>
      <div class="log-detail-panel"><div class="log-detail-placeholder">选择一条调用，查看完整请求与返回。</div><div class="log-detail" hidden><div class="log-detail-heading"></div>
      <div class="log-tabs"><button type="button" data-log-tab="request">请求</button><button type="button" data-log-tab="response">返回</button><button type="button" data-log-tab="raw">原始数据</button><button type="button" class="log-copy">复制完整内容</button></div>
      <div class="log-detail-content"></div></div></div></div>`;
    // Capture only this fixed shell, before any request or user content is added.
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode as Text;
      if (node.data.trim()) this.chromeText.push({ node, source: node.data });
    }
    this.localize();
    this.el<HTMLSelectElement>("log-type").onchange = () => this.changeFilter();
    this.el<HTMLSelectElement>("log-status").onchange = () =>
      this.changeFilter();
    this.el<HTMLInputElement>("log-auto").onchange = () => this.schedule();
    this.el<HTMLButtonElement>("log-more").onclick = () =>
      void this.refresh(true);
    root.querySelector<HTMLButtonElement>(
      "[data-log-action=refresh]",
    )!.onclick = () => void this.refresh();
    root.querySelectorAll<HTMLButtonElement>("[data-log-tab]").forEach(
      (button) =>
        (button.onclick = () => {
          this.tab = button.dataset.logTab!;
          this.renderDetail();
        }),
    );
    this.el<HTMLButtonElement>("log-copy").onclick = () => void this.copy();
  }
  private localize(): void {
    for (const { node, source } of this.chromeText) node.data = t(source);
    this.el("log-notice").textContent = t(this.notice.source, this.notice.values);
    for (const option of this.el<HTMLSelectElement>("log-type").options) {
      option.text = option.value ? t(logTypeName(option.value)) : t("全部类型");
    }
    for (const [row, item] of this.rows) {
      row.querySelector("strong")!.textContent = t(logTypeName(item.call_type));
      row.querySelector(".log-state")!.textContent = t(LOG_STATUS_NAMES[item.status]);
      row.querySelector("small")!.textContent = this.rowMeta(item);
    }
    if (this.detail) {
      const content = this.el("log-detail-content");
      const expanded = [...content.querySelectorAll("details")].map((node) => node.open);
      this.renderDetail();
      // Opening a JSON section can add more sections lazily; re-query each time.
      expanded.forEach((open, index) => {
        const section = content.querySelectorAll("details")[index];
        if (section && section.open !== open) {
          section.open = open;
          section.dispatchEvent(new Event("toggle"));
        }
      });
    }
  }
  private setNotice(source: string, values?: Record<string, string | number>): void {
    this.notice = { source, values };
    this.el("log-notice").textContent = t(source, values);
  }
  private rowMeta(item: LlmLogSummary): string {
    return `${new Date(item.started_at).toLocaleString(getLocale(), { hour12: false })} · ${item.duration_ms === null ? t("进行中") : t("{{seconds}} 秒", { seconds: (item.duration_ms / 1000).toFixed(2) })}`;
  }
  private el<T extends HTMLElement = HTMLElement>(name: string): T {
    return this.root.querySelector(`.${name}`) as T;
  }
  private changeFilter(): void {
    this.selected = null;
    this.detail = null;
    ++this.detailVersion;
    this.el("log-detail").hidden = true;
    this.el("log-detail-placeholder").hidden = false;
    void this.refresh();
  }
  setVisible(value: boolean): void {
    if (this.visible === value) return;
    this.visible = value;
    if (this.timer) clearTimeout(this.timer);
    this.stopLocale?.();
    this.stopLocale = value ? subscribeLocale(() => this.localize()) : null;
    if (value) { this.localize(); void this.refresh(); }
    else { ++this.version; ++this.detailVersion; }
  }
  private schedule(): void {
    if (this.timer) clearTimeout(this.timer);
    if (
      this.visible &&
      !document.hidden &&
      this.el<HTMLInputElement>("log-auto").checked
    )
      this.timer = setTimeout(() => void this.refresh(), 5000);
  }
  private typesShown = "";
  private async refreshTypes(): Promise<void> {
    try {
      const { types } = await window.desktopPet.getLlmLogTypes();
      const select = this.el<HTMLSelectElement>("log-type");
      const previous = select.value;
      // Only what was actually called on this computer, in the names' order.
      const present = new Set(previous ? [...types, previous] : types);
      const available = [
        ...Object.keys(LOG_TYPE_NAMES).filter((type) => present.has(type)),
        ...[...present].filter((type) => !(type in LOG_TYPE_NAMES)),
      ];
      // Rebuilt only when it changed, so an open list is never swapped under the pointer.
      const signature = available.join("\n");
      if (signature === this.typesShown) return;
      this.typesShown = signature;
      select.replaceChildren(
        new Option(t("全部类型"), ""),
        ...available.map((type) => new Option(t(logTypeName(type)), type)),
      );
      select.value = previous;
    } catch {
      /* The list shows a single actionable connection error. */
    }
  }
  private async refresh(more = false): Promise<void> {
    if (!this.visible) return;
    // New kinds of calls appear, and the first read may come before TA is up.
    if (!more) void this.refreshTypes();
    const version = ++this.version;
    const button = this.el<HTMLButtonElement>("log-more");
    button.disabled = true;
    try {
      const data = await window.desktopPet.listLlmLogs({
        call_type: this.el<HTMLSelectElement>("log-type").value || undefined,
        status: this.el<HTMLSelectElement>("log-status").value || undefined,
        before: more && this.cursor ? this.cursor : undefined,
      });
      if (version !== this.version) return;
      const list = this.el("log-list");
      if (!more) { list.replaceChildren(); this.rows.clear(); }
      for (const item of data.items) list.append(this.row(item));
      this.cursor = data.next_cursor;
      button.hidden = !this.cursor;
      if (data.dropped_records) this.setNotice("有 {{count}} 条记录未能写入，列表可能不完整。", { count: data.dropped_records });
      else this.setNotice(list.children.length ? "" : "暂无符合条件的调用。新日志从本次更新后开始记录。");
      if (!this.selected && data.items[0]) await this.select(data.items[0].id);
      else if (this.detail?.status === "running")
        await this.select(this.detail.id);
    } catch (error) {
      if (version === this.version)
        this.setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      if (version === this.version) {
        button.disabled = false;
        this.schedule();
      }
    }
  }
  private row(item: LlmLogSummary): HTMLElement {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "log-row";
    button.dataset.id = item.id;
    button.classList.toggle("selected", item.id === this.selected);
    const title = document.createElement("strong");
    title.textContent = t(logTypeName(item.call_type));
    const status = document.createElement("span");
    status.className = `log-state ${item.status}`;
    status.textContent = t(LOG_STATUS_NAMES[item.status]);
    const meta = document.createElement("small");
    meta.textContent = this.rowMeta(item);
    const model = document.createElement("small");
    model.textContent = item.model;
    button.append(title, status, meta, model);
    this.rows.set(button, item);
    button.onclick = () => void this.select(item.id);
    return button;
  }
  private async select(id: string): Promise<void> {
    const version = ++this.detailVersion;
    this.selected = id;
    this.root
      .querySelectorAll<HTMLElement>(".log-row")
      .forEach((row) =>
        row.classList.toggle("selected", row.dataset.id === id),
      );
    try {
      const detail = await window.desktopPet.getLlmLog(id);
      if (version !== this.detailVersion) return;
      this.detail = detail;
      this.el("log-detail-placeholder").hidden = true;
      this.el("log-detail").hidden = false;
      this.renderDetail();
    } catch (error) {
      if (version === this.detailVersion)
        this.setNotice(error instanceof Error ? error.message : String(error));
    }
  }
  private renderDetail(): void {
    const detail = this.detail;
    if (!detail) return;
    const heading = this.el("log-detail-heading");
    heading.replaceChildren();
    const title = document.createElement("h3");
    title.textContent = `${t(logTypeName(detail.call_type))} · ${t(LOG_STATUS_NAMES[detail.status])}`;
    const meta = document.createElement("p");
    meta.textContent = `${detail.model} · ${new Date(detail.started_at).toLocaleString(getLocale(), { hour12: false })}${detail.duration_ms === null ? "" : ` · ${t("{{seconds}} 秒", { seconds: (detail.duration_ms / 1000).toFixed(2) })}`}`;
    heading.append(title, meta);
    if (detail.error) {
      const error = document.createElement("p");
      error.className = "log-error";
      error.textContent = detail.error;
      heading.append(error);
    }
    if ((detail.response as { truncated?: boolean } | null)?.truncated) {
      const notice = document.createElement("p");
      notice.className = "log-error";
      notice.textContent =
        t("返回超过 16 MB，日志仅保留前 16 MB；实际调用未受截断影响。");
      heading.append(notice);
    }
    this.root
      .querySelectorAll<HTMLButtonElement>("[data-log-tab]")
      .forEach((button) => {
        button.classList.toggle("active", button.dataset.logTab === this.tab);
        button.setAttribute(
          "aria-selected",
          String(button.dataset.logTab === this.tab),
        );
      });
    const content = this.el("log-detail-content");
    content.replaceChildren();
    if (this.tab === "request") content.append(readableRequest(detail.request));
    else if (this.tab === "response")
      content.append(readableResponse(detail.response));
    else {
      for (const [label, value] of [
        ["实际请求 JSON", detail.request],
        ["完整返回记录", detail.response],
        ["调用信息", { ...detail, request: undefined, response: undefined }],
      ] as const) {
        const section = document.createElement("details");
        const title = document.createElement("summary");
        title.textContent = t(label);
        section.append(title);
        section.addEventListener("toggle", () => {
          if (section.open && !section.querySelector("pre")) {
            const pre = document.createElement("pre");
            pre.className = "log-raw";
            pre.textContent = JSON.stringify(value, null, 2);
            section.append(pre);
          }
        });
        content.append(section);
      }
    }
  }
  private async copy(): Promise<void> {
    if (!this.detail) return;
    const value =
      this.tab === "request"
        ? this.detail.request
        : this.tab === "response"
          ? this.detail.response
          : this.detail;
    try {
      await navigator.clipboard.writeText(
        typeof value === "string" ? value : JSON.stringify(value, null, 2),
      );
      this.setNotice("已复制完整内容");
    } catch {
      this.setNotice("复制失败，可在详情中选中文字复制。");
    }
  }
}
