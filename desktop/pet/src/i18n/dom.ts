import { subscribeLocale, t } from "./index";

type Values = Record<string, string | number> | (() => Record<string, string | number>);
type Binding = { source: string; values?: Values; previous: string };
const resolve = (values: Values | undefined) => typeof values === "function" ? values() : values;
const bindings = new WeakMap<HTMLElement, Map<string, Binding>>();
const nodes = new Set<WeakRef<HTMLElement>>();

function bind(node: HTMLElement, attr: string, source: string, values?: Values): void {
  let entries = bindings.get(node);
  if (!entries) { entries = new Map(); bindings.set(node, entries); nodes.add(new WeakRef(node)); }
  const value = t(source, resolve(values));
  entries.set(attr, { source, values, previous: value });
  if (attr === "textContent") node.textContent = value;
  else node.setAttribute(attr, value);
}

export function bindLocalizedText<T extends HTMLElement>(node: T, source: string, values?: Values): T {
  bind(node, "textContent", source, values);
  return node;
}

export function bindLocalizedAttribute<T extends HTMLElement>(node: T, attr: string, source: string, values?: Values): T {
  bind(node, attr, source, values);
  return node;
}

subscribeLocale(() => {
  for (const reference of nodes) {
    const node = reference.deref();
    if (!node) { nodes.delete(reference); continue; }
    if (node.isConnected === false) continue;
    for (const [attr, entry] of bindings.get(node) ?? []) {
      const current = attr === "textContent" ? node.textContent : node.getAttribute(attr);
      // An imperative view may have replaced a placeholder with live/user data.
      // Only update copy we still own.
      if (current !== entry.previous) continue;
      const value = t(entry.source, resolve(entry.values));
      if (attr === "textContent") node.textContent = value;
      else node.setAttribute(attr, value);
      entry.previous = value;
    }
  }
});

/** Explicitly marked static HTML only; never walks user text or changes input values. */
export function localizeStatic(root: ParentNode): () => void {
  for (const node of root.querySelectorAll<HTMLElement>("[data-i18n]")) {
    bindLocalizedText(node, node.dataset.i18n!);
  }
  for (const attr of ["title", "placeholder", "aria-label"] as const) {
    for (const node of root.querySelectorAll<HTMLElement>(`[data-i18n-${attr}]`)) {
      bindLocalizedAttribute(node, attr, node.getAttribute(`data-i18n-${attr}`)!);
    }
  }
  // Bindings use weak references; repeated mounts do not retain removed trees.
  return () => {};
}
