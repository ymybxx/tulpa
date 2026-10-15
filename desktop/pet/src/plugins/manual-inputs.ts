import { t, getLocale } from "../i18n";
import { bindLocalizedText } from "../i18n/dom";
type Json = Record<string, any>;
type Call = (action: string, input: Json) => Promise<Json>;
type Field = { key: string; schema: Json; input: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement };
type Row = { node: HTMLElement; status: HTMLElement; reply: HTMLElement; eventId?: string; pending: boolean };
type Form = { node: HTMLFormElement; fields: Field[]; button: HTMLButtonElement; status: HTMLElement; history: HTMLElement; rows: Row[]; sending: boolean };

const TERMINAL = new Set(["completed", "interrupted", "silent", "failed", "cancelled", "stored_only"]);
const REASONS: Record<string, string> = {
  pet_busy: "宠物正忙，已记录；这条不会排队补播。",
  user_busy: "你正在说话，已记录；这条不会排队补播。",
  no_playback_consumer: "已记录，但当前会话未连接语音伴侣。",
  disconnected: "插件已断开，处理结果未确认。",
  duplicate_message: "已过滤：这是重复投递的同一条消息。",
  duplicate_text: "已过滤：同一观众在两分钟内重复发送相同内容。",
  rate_limited: "已过滤：发送过快，请稍后再试。",
  invalid_input: "输入不完整或格式不正确，请检查后重试。",
  not_ready: "插件还没准备好，请稍后再试。",
  input_failed: "插件处理失败，请查看运行日志。",
  input_timeout: "输入结果尚未确认，请查看宠物反应和运行日志。",
};

function inputReceiptSource(result: Json): string {
  if (result.disposition === "responding") return "已接收，宠物正在处理。";
  if (result.disposition === "queued") return result.reason === "no_playback_consumer"
    ? "已排队，等待当前会话连接语音伴侣。"
    : "已排队，当前发言结束后按到达顺序处理。";
  const reason = REASONS[result.reason];
  if (reason) return reason;
  if (result.disposition === "stored_only") return "已记录，本次未发声。";
  if (result.disposition === "filtered") return "已被插件过滤。";
  if (result.disposition === "rejected") return "插件未接受这条输入。";
  return "处理结果未确认，请查看运行日志。";
}

export function inputReceiptText(result: Json): string {
  return t(inputReceiptSource(result));
}

/** Forms belong to plugin declarations; the App does not interpret platform payloads. */
export class ManualInputsPanel {
  private forms = new Map<string, Form>();
  private id: string | null = null;
  private running = false;
  private definitions: Json[] = [];
  private displayKey = "";

  constructor(private root: HTMLElement, private call: Call) {}

  update(item: Json): void {
    this.id = item.id;
    this.running = item.status === "running" && item.state?.connected && !item.state?.stopped;
    this.definitions = item.description.inputs ?? [];
    const key = `${item.id}:${JSON.stringify(this.definitions)}`;
    if (key !== this.displayKey) {
      this.displayKey = key;
      this.root.replaceChildren(...this.definitions.map((definition) => this.form(item.id, definition).node));
    }
    this.root.hidden = this.definitions.length === 0;
    for (const definition of this.definitions) {
      const form = this.form(item.id, definition);
      form.button.disabled = !this.running || form.sending;
      if (!form.rows.length) bindLocalizedText(form.status, this.running ? "可随时发送；⌘/Ctrl + Enter 快捷发送。" : "启动插件后即可发送，输入内容会保留。");
    }
    if (item.state?.last_response) this.response(item.id, item.state.last_response);
  }

  get needsResults(): boolean {
    return this.id !== null && this.definitions.some((d) => this.form(this.id!, d).rows.some((r) => r.pending));
  }

  diagnostics(id: string, data: Json): void {
    for (const event of data.events ?? []) {
      if (event.method === "response") this.response(id, event.params);
    }
  }

  private response(id: string, result: Json): void {
    const labels: Record<string, string> = {
      generated: "已生成，等待实际播放。", speaking: "正在播放。",
      completed: "已播完。", interrupted: "已被打断。", silent: "本次未发声。",
      failed: "本次响应失败，请查看运行日志。", cancelled: "本次响应已取消。",
    };
    for (const [key, form] of this.forms) {
      if (!key.startsWith(`${id}:`)) continue;
      for (const row of form.rows) {
        if (!row.eventId || row.eventId !== result.event_id) continue;
        const label = labels[result.status];
        if (label) bindLocalizedText(row.status, label);
        if (["completed", "interrupted"].includes(result.status) && result.text) {
          bindLocalizedText(row.reply, "宠物：{{text}}", { text: result.text });
          row.reply.hidden = false;
        }
        if (TERMINAL.has(result.status)) row.pending = false;
      }
    }
  }

  private form(id: string, definition: Json): Form {
    const key = `${id}:${definition.type}`;
    const existing = this.forms.get(key);
    if (existing) return existing;
    const node = document.createElement("form");
    node.className = "plugin-manual-form";
    node.setAttribute("aria-label", definition.title);
    node.append(text("h3", definition.title), text("p", definition.description));
    const fields: Field[] = [];
    const schema = definition.schema;
    for (const [key, value] of Object.entries<Json>(schema.properties ?? {})) {
      const label = document.createElement("label");
      const input = value.enum ? document.createElement("select")
        : value.format === "textarea" ? document.createElement("textarea") : document.createElement("input");
      input.name = key;
      input.setAttribute("aria-label", value.title ?? key);
      input.required = (schema.required ?? []).includes(key);
      if (input instanceof HTMLSelectElement) {
        for (const option of value.enum) input.add(new Option(String(option), String(option)));
      } else {
        input.placeholder = value.placeholder ?? "";
        if (value.maxLength != null) input.maxLength = value.maxLength;
        if (input instanceof HTMLInputElement) {
          input.type = ["number", "integer"].includes(value.type) ? "number" : "text";
          if (value.minimum != null) input.min = String(value.minimum);
          if (value.maximum != null) input.max = String(value.maximum);
          if (value.examples?.length) {
            const choices = document.createElement("datalist");
            choices.id = `plugin-input-${crypto.randomUUID()}`;
            input.setAttribute("list", choices.id);
            choices.replaceChildren(...value.examples.map((v: string) => new Option(String(v), String(v))));
            label.append(choices);
          }
        } else {
          input.rows = 3;
          input.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.isComposing) {
              event.preventDefault();
              node.requestSubmit();
            }
          });
        }
      }
      input.value = String(value.default ?? "");
      input.addEventListener("input", () => input.setCustomValidity(""));
      label.append(text("span", value.title ?? key), input);
      if (value.description) label.append(text("small", value.description));
      node.append(label);
      fields.push({ key, schema: value, input });
    }
    const button = document.createElement("button");
    button.type = "submit";
    if (definition.submit_label) button.textContent = definition.submit_label;
    else bindLocalizedText(button, "发送");
    const status = text("p", "");
    status.className = "plugin-manual-status";
    status.setAttribute("role", "status");
    const history = document.createElement("div");
    history.className = "plugin-manual-history";
    node.append(button, status, history);
    const form: Form = { node, fields, button, status, history, rows: [], sending: false };
    node.onsubmit = (event) => { event.preventDefault(); void this.send(id, definition, form); };
    this.forms.set(key, form);
    return form;
  }

  private async send(id: string, definition: Json, form: Form): Promise<void> {
    if (id !== this.id || !this.running || form.sending) return;
    const data: Json = {};
    for (const field of form.fields) {
      const value = field.input.value.trim();
      if ((field.input.required || field.schema.minLength > 0) && !value) {
        field.input.setCustomValidity(t("请填写此项"));
        field.input.reportValidity();
        return;
      }
      data[field.key] = ["number", "integer"].includes(field.schema.type) ? Number(value) : value;
    }
    if (!form.node.reportValidity()) return;
    form.sending = true;
    form.button.disabled = true;
    const node = document.createElement("article");
    const names = form.fields.filter((f) => !(f.input instanceof HTMLTextAreaElement)).map((f) => data[f.key]);
    node.append(text("small", `${new Date().toLocaleTimeString(getLocale())} · ${names.join(" · ")}`));
    const content = form.fields.filter((f) => f.input instanceof HTMLTextAreaElement).map((f) => data[f.key]).join("\n");
    if (content) node.append(text("p", content));
    const status = text("small", "");
    bindLocalizedText(status, "正在送入插件…");
    const reply = text("p", "");
    reply.className = "plugin-manual-reply";
    reply.hidden = true;
    node.append(status, reply);
    const row: Row = { node, status, reply, pending: false };
    form.rows.unshift(row);
    form.history.prepend(node);
    while (form.rows.length > 12) form.rows.pop()!.node.remove();
    try {
      const result = await this.call("input", { id, input: { event_id: crypto.randomUUID(), type: definition.type, data } });
      row.eventId = result.event_id;
      row.pending = ["responding", "queued"].includes(result.disposition);
      bindLocalizedText(status, inputReceiptSource(result));
      bindLocalizedText(form.status, inputReceiptSource(result));
      if (["responding", "queued", "stored_only"].includes(result.disposition)) {
        for (const field of form.fields) {
          if (field.input instanceof HTMLTextAreaElement && field.input.value.trim() === data[field.key]) field.input.value = "";
        }
      }
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : String(error);
      form.status.textContent = status.textContent;
    } finally {
      form.sending = false;
      form.button.disabled = !this.running || this.id !== id;
    }
  }
}

function text(tag: string, value: unknown): HTMLElement {
  const node = document.createElement(tag);
  node.textContent = String(value ?? "");
  return node;
}
