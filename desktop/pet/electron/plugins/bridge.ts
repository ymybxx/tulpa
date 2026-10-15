import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { safeStorage } from "electron";

type Value = Record<string, unknown>;
interface Connection { apiUrl: string; apiKey: string; }

/** App-only management. Live events connect directly from Python to the API. */
export class PluginBridge {
  constructor(private root: string, private connect: () => Promise<Connection>) {}

  private async request(route: string, body?: Value, method?: string): Promise<any> {
    const connection = await this.connect();
    const response = await fetch(connection.apiUrl + route, {
      method: method ?? (body ? "POST" : "GET"),
      headers: { Authorization: `Bearer ${connection.apiKey}`, "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(240_000),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : `插件操作失败（${response.status}）`);
    return value;
  }

  private configPath(id: unknown): string {
    if (typeof id !== "string" || !/^[0-9a-f-]{36}$/i.test(id)) throw new Error("插件实例 ID 不合法");
    return path.join(this.root, `${id}.bin`);
  }

  private async configuration(id: unknown): Promise<Value> {
    try {
      return JSON.parse(safeStorage.decryptString(await readFile(this.configPath(id))));
    } catch (error: any) {
      if (error.code === "ENOENT") return {};
      throw new Error("无法读取插件配置，请重新填写");
    }
  }

  async call(action: string, input: Value = {}): Promise<any> {
    if (action === "list") return this.request("/internal/extensions/manager");
    if (action === "config") {
      const stored = await this.configuration(input.id);
      const listing = await this.request("/internal/extensions/manager");
      const item = listing.plugins.find((p: any) => p.id === input.id);
      if (!item) throw new Error("插件不存在");
      const secretFields = Object.entries(item.description.config_schema.properties ?? {})
        .filter(([, schema]: any) => schema.writeOnly || schema.format === "password").map(([key]) => key);
      return { values: Object.fromEntries(Object.entries(stored).filter(([key]) => !secretFields.includes(key))),
        savedSecrets: secretFields.filter((key) => Boolean(stored[key])) };
    }
    if (action === "start") {
      const config = { ...await this.configuration(input.id), ...(input.config as Value ?? {}) };
      if (!safeStorage.isEncryptionAvailable() || (process.platform === "linux" && safeStorage.getSelectedStorageBackend() === "basic_text")) {
        throw new Error("系统安全存储暂不可用，不能保存插件凭据");
      }
      const file = this.configPath(input.id);
      await mkdir(this.root, { recursive: true });
      await writeFile(`${file}.tmp`, safeStorage.encryptString(JSON.stringify(config)), { mode: 0o600 });
      await rename(`${file}.tmp`, file);
      return this.request("/internal/extensions/manager/start", { ...input, config });
    }
    if (["load", "builtin", "stop", "development", "reviews", "review", "input"].includes(action)) {
      return this.request(`/internal/extensions/manager/${action}`, input);
    }
    if (action === "diagnostics" || action === "event") {
      this.configPath(input.id);
      return this.request(`/internal/extensions/instances/${input.id}${action === "event" ? "/events" : ""}`,
        action === "event" ? input.event as Value : undefined);
    }
    throw new Error("不支持的插件操作");
  }
}
