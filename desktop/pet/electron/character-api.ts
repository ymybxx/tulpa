import WebSocket from "ws";

import type {
  CharacterDetails,
  CharacterMemory,
  CharacterProfileInput,
  CharacterSummary,
  CharacterVoice,
  CharacterVoiceService,
  EntityFact,
  ForgetTarget,
  ForgetPlan,
} from "../src/shared/contracts";

export interface ApiTarget {
  apiUrl: string;
  apiKey: string;
}

type Json = Record<string, unknown>;

/** Character (Fish) operations of the local memory service, for the 角色 page. */
export class CharacterApi {
  constructor(private readonly target: () => Promise<ApiTarget | null>) {}

  async list(): Promise<CharacterSummary[]> {
    const payload = await this.request<{ items?: Json[] }>("GET", "/v1/fishes");
    return (payload.items ?? [])
      .filter((item) => item.evaluation !== true)
      .map((item) => ({
        id: String(item.fishId),
        name: typeof item.name === "string" ? item.name : "",
        role: typeof item.role === "string" ? item.role : "",
        current: item.isAssistant === true,
        lastActiveAt: typeof item.lastActiveAt === "string" ? item.lastActiveAt : null,
      }));
  }

  async bind(id: string): Promise<void> {
    await this.request("PUT", `/v1/fishes/${encode(id)}/assistant`);
  }

  async create(input: CharacterProfileInput): Promise<string> {
    const payload = await this.request<{ fishId?: string }>("POST", "/v1/fishes", soulBody(input));
    if (!payload.fishId) throw new Error("没有拿到新角色的编号");
    if (input.wakeWords?.length) await this.saveWakeWords(payload.fishId, input.wakeWords);
    return payload.fishId;
  }

  async profile(id: string): Promise<CharacterDetails> {
    const [soul, wake, management] = await Promise.all([
      this.request<Json>("GET", `/v1/fishes/${encode(id)}/soul`),
      this.request<Json>("GET", `/v1/fishes/${encode(id)}/wake-words`),
      this.request<Json>("GET", `/internal/fishes/${encode(id)}/management`),
    ]);
    return {
      id,
      name: text(soul.name),
      role: text(soul.role),
      setting: text(soul.setting),
      background: text(soul.background),
      wakeWords: Array.isArray(wake.words) ? wake.words.map(String) : [],
      wakeWordsCustomized: wake.customized === true,
      effectiveSoul: text(management.effectiveSoul),
      createdAt: typeof management.createdAt === "string" ? management.createdAt : null,
    };
  }

  async saveProfile(id: string, input: CharacterProfileInput): Promise<void> {
    await this.request("PATCH", `/v1/fishes/${encode(id)}/soul`, soulBody(input));
  }

  async saveWakeWords(id: string, words: string[]): Promise<void> {
    await this.request("PUT", `/v1/fishes/${encode(id)}/wake-words`, { words });
  }

  async voice(id: string): Promise<CharacterVoice> {
    const [selection, cloudCatalog, localCatalog] = await Promise.all([
      this.request<Json>("GET", `/v1/fishes/${encode(id)}/voice`),
      this.request<Json>("GET", "/v1/tts/models?provider=dashscope"),
      this.request<Json>("GET", "/v1/tts/models?provider=local"),
    ]);
    const config = (selection.config ?? {}) as Json;
    const service = (catalog: Json): CharacterVoiceService => ({
      ready: catalog.ready !== false,
      options: (Array.isArray(catalog.models) ? catalog.models as Json[] : []).map((model) => ({
        model: text(model.id),
        label: text(model.label),
        voices: (Array.isArray(model.voices) ? model.voices as Json[] : []).map((voice) => ({
          id: text(voice.id),
          label: text(voice.label),
        })),
      })),
    });
    const cloud = service(cloudCatalog);
    const local = service(localCatalog);
    const provider = text(selection.provider);
    return {
      provider,
      model: text(config.model),
      voice: text(config.voice),
      kind: text(config.kind),
      options: provider === "local" ? local.options : cloud.options,
      cloud,
      local,
    };
  }

  async saveVoice(id: string, voice: { model: string; voice: string; kind: string }): Promise<void> {
    await this.request("PATCH", `/v1/fishes/${encode(id)}/voice`, voice);
  }

  async customVoices(): Promise<Array<{ id: string; model: string; ready: boolean }>> {
    const payload = await this.request<{ items?: Json[] }>("GET", "/internal/tts/custom-voices");
    return (payload.items ?? []).map((voice) => ({
      id: text(voice.voiceId),
      model: text(voice.targetModel),
      ready: text(voice.status).toUpperCase() === "OK",
    }));
  }

  /** Clones a voice from a 16-bit PCM WAV recording; returns the new voice id. */
  async cloneVoice(recording: Uint8Array, targetModel: string): Promise<string> {
    const target = await this.requireTarget();
    const response = await fetch(`${target.apiUrl}/internal/tts/custom-voices?targetModel=${encode(targetModel)}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${target.apiKey}`, "Content-Type": "audio/wav" },
      body: Buffer.from(recording),
      signal: AbortSignal.timeout(60_000),
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    const voice = await response.json() as Json;
    const id = text(voice.voiceId);
    if (!id) throw new Error("没有拿到新音色");
    return id;
  }

  async memory(id: string): Promise<CharacterMemory> {
    const management = await this.request<Json>("GET", `/internal/fishes/${encode(id)}/management`);
    const items = (snapshot: unknown) => {
      const value = snapshot as { items?: Json[] } | null;
      return (value?.items ?? []).map((item) => ({
        id: Number(item.id),
        text: text(item.text),
        lastSupportedAt: text(item.lastSupportedAt),
        factCount: Number(item.sourceFactCount) || 0,
      }));
    };
    const entities = Array.isArray(management.entities) ? management.entities as Json[] : [];
    return {
      impressions: items(management.staticProfile),
      recent: items(management.dynamicSnapshot),
      entities: entities.map((entity) => ({
        id: text(entity.entityId),
        aliases: Array.isArray(entity.aliases) ? entity.aliases.map(String) : [],
        factCount: Number(entity.factCount) || 0,
      })),
      createdAt: typeof management.createdAt === "string" ? management.createdAt : null,
    };
  }

  async entityFacts(id: string, entityId: string): Promise<EntityFact[]> {
    const payload = await this.request<{ items?: Json[] }>(
      "GET",
      `/internal/fishes/${encode(id)}/entities/${encode(entityId)}/facts?limit=200`,
    );
    return (payload.items ?? []).map((fact) => ({
      id: text(fact.factId),
      text: text(fact.text),
      mentionedAt: text(fact.mentionedAt),
    }));
  }

  async forgetPreview(id: string, target: ForgetTarget): Promise<ForgetPlan> {
    return forgetPlan(await this.request<Json>("POST", `/v1/fishes/${encode(id)}/memory/forget-preview`, target));
  }

  async forget(id: string, target: ForgetTarget): Promise<ForgetPlan> {
    return forgetPlan(await this.request<Json>("POST", `/v1/fishes/${encode(id)}/memory/forget`, target));
  }

  /** Synthesizes a short sample with the given voice; returns a WAV file. */
  async previewVoice(voice: { model: string; voice: string }, sentence: string): Promise<Uint8Array> {
    const target = await this.requireTarget();
    const url = `${target.apiUrl.replace(/^http/, "ws")}/v1/tts/stream`;
    let sampleRate = 24_000;
    const chunks: Buffer[] = [];
    await new Promise<void>((resolve, reject) => {
      const socket = new WebSocket(url, { headers: { Authorization: `Bearer ${target.apiKey}` } });
      const timer = setTimeout(() => { socket.terminate(); reject(new Error("试听超时")); }, 30_000);
      const finish = (error?: Error) => {
        clearTimeout(timer);
        socket.close();
        if (error) reject(error);
        else resolve();
      };
      socket.on("open", () => {
        socket.send(JSON.stringify({ type: "start", model: voice.model, voice: voice.voice, audio_format: "pcm", sample_rate: sampleRate }));
        socket.send(JSON.stringify({ type: "append", text: sentence }));
        socket.send(JSON.stringify({ type: "finish" }));
      });
      socket.on("message", (data, binary) => {
        if (binary) {
          chunks.push(Buffer.from(data as Buffer));
          return;
        }
        const event = JSON.parse(String(data)) as { type?: string; message?: string; sampleRate?: unknown };
        // The provider may answer at another rate than asked; trust what it reports.
        if (event.type === "ready" && typeof event.sampleRate === "number") sampleRate = event.sampleRate;
        if (event.type === "finished") finish();
        else if (event.type === "error") finish(new Error(event.message || "语音生成失败"));
      });
      socket.on("error", () => finish(new Error("连不上语音服务")));
    });
    return wav(Buffer.concat(chunks), sampleRate);
  }

  private async requireTarget(): Promise<ApiTarget> {
    const target = await this.target();
    if (!target) throw new Error("TA 还在醒来，稍等一下再试");
    return target;
  }

  private async request<T = unknown>(method: string, path: string, body?: unknown): Promise<T> {
    const target = await this.requireTarget();
    const response = await fetch(`${target.apiUrl}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${target.apiKey}`,
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(15_000),
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    if (response.status === 204) return undefined as T;
    return await response.json() as T;
  }
}

function soulBody(input: CharacterProfileInput): Json {
  return {
    name: input.name,
    role: input.role,
    setting: input.setting,
    background: input.background,
  };
}

function forgetPlan(value: Json): ForgetPlan {
  return { factCount: Number(value.factCount) || 0, itemCount: Number(value.itemCount) || 0 };
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function encode(value: string): string {
  return encodeURIComponent(value);
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json() as { error?: { message?: string }; detail?: unknown };
    if (payload.error?.message) return payload.error.message;
    if (Array.isArray(payload.detail)) {
      const first = payload.detail[0] as { msg?: string } | undefined;
      if (first?.msg) return first.msg.replace(/^Value error, /, "");
    }
  } catch { /* fall through */ }
  return `请求失败（${response.status}）`;
}

/** 16-bit mono PCM to a WAV container the renderer can play directly. */
export function wav(pcm: Buffer, sampleRate: number): Uint8Array {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcm.length, 40);
  return new Uint8Array(Buffer.concat([header, pcm]));
}
