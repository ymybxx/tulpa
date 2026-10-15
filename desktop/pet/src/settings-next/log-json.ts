import { t } from "../i18n";
import { parsedText, responseContent } from "../shared/llm-logs";

function node<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text?: string,
  className?: string,
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

/** Text is always rendered as text nodes, including JSON returned by a model. */
export function readableJson(value: unknown, depth = 0): HTMLElement {
  const parsed = parsedText(value);
  if (parsed !== value) {
    const wrapper = node("div", undefined, "log-json-decoded");
    wrapper.append(
      node("span", t("JSON 文本"), "log-json-hint"),
      readableJson(parsed, depth),
    );
    return wrapper;
  }
  if (typeof value === "string") {
    if (value.startsWith("data:image/") && value.length > 500) {
      const details = node("details", undefined, "log-image");
      details.append(
        node("summary", t("图片数据 · {{size}} KB", { size: Math.ceil(value.length / 1024) })),
      );
      details.addEventListener("toggle", () => {
        if (details.open && !details.querySelector("img")) {
          const image = node("img");
          image.src = value;
          image.alt = t("本次请求中的图片");
          details.append(image);
        }
      });
      return details;
    }
    return node("pre", value || t("（空字符串）"), "log-text");
  }
  if (value === null || typeof value !== "object")
    return node("span", String(value), "log-literal");
  const entries = Object.entries(value);
  const container = node("div", undefined, "log-json-object");
  if (!entries.length)
    return node("span", Array.isArray(value) ? "[]" : "{}", "log-literal");
  for (const [key, item] of entries) {
    const row = node("div", undefined, "log-json-field");
    const structured = item !== null && typeof item === "object";
    if (structured) {
      const details = node("details");
      details.open = depth < 2;
      details.append(
        node(
          "summary",
          `${key}  ${Array.isArray(item) ? t("[{{count}} 项]", { count: item.length }) : t("{ {{count}} 个字段 }", { count: Object.keys(item).length })}`,
        ),
      );
      let rendered = false;
      const render = () => {
        if (details.open && !rendered) {
          rendered = true;
          details.append(readableJson(item, depth + 1));
        }
      };
      details.addEventListener("toggle", render);
      render();
      row.append(details);
    } else {
      row.append(
        node("span", key, "log-json-key"),
        readableJson(item, depth + 1),
      );
    }
    container.append(row);
  }
  return container;
}

export function readableRequest(value: unknown): HTMLElement {
  if (!value || typeof value !== "object" || Array.isArray(value))
    return readableJson(value);
  const request = value as Record<string, unknown>;
  const messages = request.messages ?? request.input;
  if (!Array.isArray(messages)) return readableJson(value);
  const root = node("div", undefined, "log-request");
  root.append(node("h3", t("发给模型的消息 · {{count}} 条", { count: messages.length })));
  messages.forEach((message, index) => {
    const data =
      message && typeof message === "object"
        ? (message as Record<string, unknown>)
        : {};
    const card = node("article", undefined, "log-message");
    const roles: Record<string, string> = {
      system: t("系统"),
      developer: t("开发者"),
      user: t("用户 / 事件输入"),
      assistant: t("助手"),
      tool: t("工具"),
    };
    const role = String(data.role ?? "input");
    card.append(
      node(
        "h4",
        `${String(index + 1).padStart(2, "0")}  ${roles[role] ?? role} · ${role}`,
      ),
    );
    card.append(readableJson(data.content ?? message));
    const extra = Object.fromEntries(
      Object.entries(data).filter(
        ([key]) => key !== "role" && key !== "content",
      ),
    );
    if (Object.keys(extra).length) {
      const fields = node("details");
      fields.append(node("summary", t("其他消息字段")), readableJson(extra));
      card.append(fields);
    }
    root.append(card);
  });
  const parameters = Object.fromEntries(
    Object.entries(request).filter(
      ([key]) => key !== "messages" && key !== "input",
    ),
  );
  const details = node("details", undefined, "log-parameters");
  details.append(node("summary", t("调用参数")), readableJson(parameters));
  root.append(details);
  return root;
}

export function readableResponse(value: unknown): HTMLElement {
  if (value === null) return readableJson(t("尚未收到返回"));
  const root = node("div");
  root.append(readableJson(responseContent(value)));
  if (!value || typeof value !== "object") return root;
  const response = value as Record<string, unknown>;
  if (response.reasoning) {
    const reasoning = node("details");
    reasoning.append(
      node("summary", t("模型推理内容")),
      readableJson(response.reasoning),
    );
    root.append(reasoning);
  }
  if (response.decode_error)
    root.append(
      node(
        "p",
        t("返回数据解压失败，可在原始数据中查看已收到的内容。"),
        "log-error",
      ),
    );
  const details = node("details");
  details.append(node("summary", t("完整返回字段")));
  let rendered = false;
  details.addEventListener("toggle", () => {
    if (!details.open || rendered) return;
    rendered = true;
    const fields = Object.fromEntries(
      Object.entries(response).filter(([key]) => key !== "raw"),
    );
    details.append(readableJson(fields));
  });
  root.append(details);
  return root;
}
