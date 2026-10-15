import { describe, expect, it } from "vitest";

import type { DesktopConfiguration } from "./contracts";
import { chatPickEnabled, decisionConnection, decisionReady, JEV_ENDPOINT } from "./decision-model";

const models = {
  text: [
    { id: "doubao", name: "豆包", baseUrl: "https://ark.cn-beijing.volces.com/api/v3", model: "doubao" },
    { id: "bailian", name: "百炼", baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1", model: "qwen" },
  ],
  voice: [{ id: "voice", name: "通义", provider: "dashscope" }],
} as DesktopConfiguration["models"];
const configuration = (companion: Partial<NonNullable<DesktopConfiguration["experimentalCompanion"]>> = {}) =>
  ({ models, experimentalCompanion: { model: "jev-1.13.0", wakeSeconds: 5, ...companion } });

describe("the decision model", () => {
  it("calls 百炼 at the workspace address with the saved Bailian key", () => {
    expect(decisionConnection(configuration({ bailianWorkspaceId: "ws-example123" }),
      { modelApiKeys: { doubao: "d", bailian: "b" }, jevApiKey: "j" })).toEqual({
      provider: "bailian",
      endpoint: "https://ws-example123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/systemone",
      model: "decision-model-preview",
      apiKey: "b",
    });
  });

  it("borrows the 通义 voice key when no Bailian text model has one", () => {
    expect(decisionConnection(configuration({ bailianWorkspaceId: "llm-abc123" }), { modelApiKeys: { voice: "v" } })?.apiKey).toBe("v");
  });

  it("keeps a Jev-only setup on Jev until a choice is made", () => {
    expect(decisionConnection(configuration(), { jevApiKey: "j" })).toEqual({
      provider: "jev", endpoint: JEV_ENDPOINT, model: "jev-1.13.0", apiKey: "j",
    });
  });

  it("connects nothing when the chosen model lacks what it needs", () => {
    expect(decisionConnection(configuration({ decisionProvider: "bailian" }), { modelApiKeys: { bailian: "b" } })).toBeNull();
    expect(decisionConnection(configuration({ decisionProvider: "jev", bailianWorkspaceId: "ws-1234" }),
      { modelApiKeys: { bailian: "b" } })).toBeNull();
    expect(decisionConnection(configuration({ bailianWorkspaceId: "bad id!" }), { modelApiKeys: { bailian: "b" } })).toBeNull();
  });

  it("reads readiness from the saved-key flags the settings page has", () => {
    const has = (id: string) => id === "bailian";
    expect(decisionReady(configuration({ bailianWorkspaceId: "ws-1234" }), has, false)).toBe(true);
    expect(decisionReady(configuration(), has, false)).toBe(false);
    expect(decisionReady(configuration({ decisionProvider: "jev" }), has, true)).toBe(true);
  });

  it("picks replies when connected unless the person chose otherwise", () => {
    expect(chatPickEnabled(configuration(), true)).toBe(true);
    expect(chatPickEnabled(configuration({ chatPickEnabled: false }), true)).toBe(false);
  });
});
