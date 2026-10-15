export interface VirtualGridOptions {
  minColumnWidth: number;
  rowHeight: number;
  gap: number;
  overscanRows?: number;
  maxRenderedItems?: number;
}

export function calculateVirtualGridWindow(input: {
  itemCount: number;
  columns: number;
  firstVisibleRow: number;
  visibleRows: number;
  overscanRows: number;
  maxRenderedItems: number;
}): { start: number; end: number; startRow: number; rowCount: number } {
  const rows = Math.ceil(input.itemCount / input.columns);
  const startRow = Math.max(0, input.firstVisibleRow - input.overscanRows);
  const requestedRows = input.visibleRows + input.overscanRows * 2;
  const renderedRows = Math.max(
    1,
    Math.min(requestedRows, Math.ceil(input.maxRenderedItems / input.columns)),
  );
  const endRow = Math.min(rows, startRow + renderedRows);
  return {
    start: startRow * input.columns,
    end: Math.min(input.itemCount, endRow * input.columns),
    startRow,
    rowCount: rows,
  };
}

/** Bounded grid window that keeps image/DOM cost independent of item count. */
export class VirtualGrid<T> {
  private readonly spacer = document.createElement("div");
  private readonly content = document.createElement("div");
  private readonly observer: ResizeObserver;
  private items: readonly T[] = [];
  private renderItem: ((value: T) => HTMLElement) | null = null;
  private frame = 0;

  constructor(
    private readonly viewport: HTMLElement,
    private readonly options: VirtualGridOptions,
  ) {
    viewport.classList.add("bounded-virtual-grid");
    this.spacer.className = "bounded-virtual-grid-spacer";
    this.content.className = "bounded-virtual-grid-content";
    this.spacer.append(this.content);
    viewport.replaceChildren(this.spacer);
    viewport.addEventListener("scroll", this.schedule, { passive: true });
    this.observer = new ResizeObserver(this.schedule);
    this.observer.observe(viewport);
  }

  set(items: readonly T[], renderItem: (value: T) => HTMLElement): void {
    this.items = items;
    this.renderItem = renderItem;
    this.viewport.scrollTop = 0;
    this.render();
  }

  destroy(): void {
    cancelAnimationFrame(this.frame);
    this.viewport.removeEventListener("scroll", this.schedule);
    this.observer.disconnect();
    this.items = [];
    this.renderItem = null;
    this.viewport.replaceChildren();
    this.viewport.classList.remove("bounded-virtual-grid");
  }

  private readonly schedule = (): void => {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.render();
    });
  };

  private render(): void {
    if (!this.renderItem) return;
    const { minColumnWidth, rowHeight, gap } = this.options;
    const width = Math.max(1, this.viewport.clientWidth);
    const columns = Math.max(1, Math.floor((width + gap) / (minColumnWidth + gap)));
    const overscan = this.options.overscanRows ?? 2;
    const firstVisible = Math.floor(this.viewport.scrollTop / rowHeight);
    const visibleRows = Math.ceil(Math.max(rowHeight, this.viewport.clientHeight) / rowHeight);
    const maximumItems = this.options.maxRenderedItems ?? 60;
    const window = calculateVirtualGridWindow({
      itemCount: this.items.length,
      columns,
      firstVisibleRow: firstVisible,
      visibleRows,
      overscanRows: overscan,
      maxRenderedItems: maximumItems,
    });
    this.spacer.style.height = `${Math.max(0, window.rowCount * rowHeight - gap)}px`;
    this.content.style.transform = `translateY(${window.startRow * rowHeight}px)`;
    this.content.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
    this.content.style.gap = `${gap}px`;
    this.content.replaceChildren(...this.items.slice(window.start, window.end).map(this.renderItem));
  }
}
