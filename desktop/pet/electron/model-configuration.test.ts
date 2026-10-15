import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, expect, it } from "vitest";

import { AppConfigStore } from "./app-config";
import { buildRuntimeEnvironment } from "./runtime-manager";
import { runtimeConfigurationAction } from "./runtime-config-impact";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))); });

async function setup() {
  const root = await mkdtemp(path.join(os.tmpdir(), "fish-model-config-"));
  roots.push(root);
  const store = new AppConfigStore(root);
  const { secrets: _, ...initial } = await store.load();
  return { store, root, initial };
}

it("defaults both text modules to the only configured model", async () => {
  const { store, initial } = await setup();
  const saved = await store.save({
    ...initial, configured: true,
    models: { ...initial.models, text: [{ id: "my-text", name: "唯一模型", baseUrl: "https://one.invalid/v1", model: "one" }] },
    secrets: { modelApiKeys: { "my-text": "one-key" } },
  });
  expect(saved.modelAssignments.chat).toBe("my-text");
  expect(saved.modelAssignments.reasoning).toBe("my-text");
  expect(saved.configured).toBe(true);
});

it("routes chat, reasoning and TTS independently and keeps keys out of public config", async () => {
  const { store, root, initial } = await setup();
  const saved = await store.save({
    ...initial, configured: true,
    models: {
      text: [
        { id: "chat", name: "对话", baseUrl: "https://chat.invalid/v1", model: "character" },
        { id: "reason", name: "后台", baseUrl: "https://reason.invalid/v1", model: "flash" },
      ],
      voice: [{ id: "voice-a", name: "声音甲", provider: "dashscope" }, { id: "voice-b", name: "声音乙", provider: "dashscope" }],
    },
    modelAssignments: { chat: "chat", reasoning: "reason", speech: "voice-b" },
    secrets: { modelApiKeys: { chat: "chat-key", reason: "reason-key", "voice-a": "voice-a-key", "voice-b": "voice-b-key" } },
  });
  const freshStore = new AppConfigStore(root);
  const loaded = await freshStore.load();
  expect(loaded.models).toEqual(saved.models);
  const runtime = await buildRuntimeEnvironment(loaded, await freshStore.runtimeSecrets(), root, null);
  expect(JSON.parse(runtime.FISH_MEMORY_DESKTOP_MODEL_ROUTES!)).toEqual({
    chat: { baseUrl: "https://chat.invalid/v1", model: "character", apiKey: "chat-key" },
    reasoning: { baseUrl: "https://reason.invalid/v1", model: "flash", apiKey: "reason-key" },
  });
  expect(runtime.FISH_MEMORY_LLM_MODEL).toBe("character");
  expect(runtime.FISH_MEMORY_TTS_API_KEY).toBe("voice-b-key");
  expect(runtime.FISH_MEMORY_TTS_PROVIDER).toBe("dashscope");
  const publicConfig = await readFile(path.join(root, "config.json"), "utf8");
  for (const key of ["chat-key", "reason-key", "voice-a-key", "voice-b-key"]) {
    expect(publicConfig).not.toContain(key);
    expect(JSON.stringify(loaded)).not.toContain(key);
  }
  const { secrets: _flags, ...current } = loaded;
  const next = await store.save({
    ...current,
    models: { ...current.models, voice: current.models.voice.slice(1) },
    secrets: { modelApiKeys: { reason: "new-reason-key" } },
  });
  const keys = (await store.runtimeSecrets()).modelApiKeys!;
  expect(keys).toEqual({ chat: "chat-key", reason: "new-reason-key", "voice-b": "voice-b-key" });
  expect(next.modelAssignments).toEqual(saved.modelAssignments);
});

it("rejects dangling selections when multiple models exist and leaves saved state intact", async () => {
  const { store, initial } = await setup();
  const text = initial.models.text[0]!;
  await expect(store.save({
    ...initial,
    models: { ...initial.models, text: [text, { ...text, id: "second" }] },
    modelAssignments: { ...initial.modelAssignments, reasoning: "missing" },
    secrets: { modelApiKeys: { "text-default": "not-saved" } },
  })).rejects.toThrow("所选模块模型不存在");
  expect((await store.runtimeSecrets()).modelApiKeys?.["text-default"]).toBeUndefined();
});

it("only hot-applies selected endpoint or selected key changes", async () => {
  const { store, initial } = await setup();
  const previous = await store.save({ ...initial, configured: true, secrets: { modelApiKeys: { "text-default": "key" } } });
  const { secrets: _flags, ...base } = previous;
  const unused = { ...base.models.text[0]!, id: "unused", name: "未选用" };
  const input = { ...base, models: { ...base.models, text: [...base.models.text, unused] } };
  const next = await store.save(input);
  expect(runtimeConfigurationAction(previous, next, input)).toBe("none");
  expect(runtimeConfigurationAction(previous, next, { ...input, secrets: { modelApiKeys: { unused: "new" } } })).toBe("none");
  expect(runtimeConfigurationAction(previous, next, { ...input, secrets: { modelApiKeys: { "text-default": "new" } } })).toBe("configure-api");
  const selected = { ...next, modelAssignments: { ...next.modelAssignments, reasoning: "unused" } };
  expect(runtimeConfigurationAction(next, selected, input)).toBe("configure-api");
});

it("reloads when the key the 百炼 decision model borrows changes, even for no brain", async () => {
  const { store, initial } = await setup();
  const previous = await store.save({ ...initial, configured: true, secrets: { modelApiKeys: { "text-default": "key" } } });
  const { secrets: _flags, ...base } = previous;
  const doubao = { id: "doubao", name: "豆包", baseUrl: "https://ark.cn-beijing.volces.com/api/v3", model: "doubao" };
  const input = { ...base, models: { ...base.models, text: [doubao, ...base.models.text] },
    modelAssignments: { ...base.modelAssignments, chat: "doubao", reasoning: "doubao" },
    experimentalCompanion: { ...base.experimentalCompanion!, bailianWorkspaceId: "ws-1234" } };
  const next = await store.save({ ...input, secrets: { modelApiKeys: { doubao: "d" } } });
  expect(runtimeConfigurationAction(next, next, { ...input, secrets: { modelApiKeys: { "text-default": "new" } } })).toBe("configure-api");
  const jev = { ...input, experimentalCompanion: { ...input.experimentalCompanion, decisionProvider: "jev" as const } };
  const onJev = await store.save(jev);
  expect(runtimeConfigurationAction(onJev, onJev, { ...jev, secrets: { modelApiKeys: { "text-default": "new" } } })).not.toBe("configure-api");
});
