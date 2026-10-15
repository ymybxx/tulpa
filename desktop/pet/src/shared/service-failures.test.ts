import { describe, expect, it } from "vitest";

import { chatMessage } from "./chat";
import { parseSenseAction } from "./pet-senses";
import {
  failureReason, needsSettings, replyFailureNote, serviceFailure, settingsSectionFor,
} from "./service-failures";

describe("service failures", () => {
  it("reads only codes the local service sends", () => {
    expect(serviceFailure("jev:timeout")).toEqual({ service: "jev", kind: "timeout" });
    for (const value of ["jev", "jev:bored", "mind:timeout", "chat:quota:extra", 3, null]) {
      expect(serviceFailure(value)).toBeNull();
    }
  });

  it("says why a reply never came, naming the service as 设置 does", () => {
    const note = (code: string) => replyFailureNote(serviceFailure(code), "小鱼", { said: false });
    expect(note("jev:timeout")).toBe("决策模型这次没有及时回应，小鱼没能回答，再说一遍试试");
    expect(note("chat:connection")).toBe("连不上对话大脑，小鱼没能回答。看看网络或代理");
    expect(note("chat:key")).toBe("对话大脑的 Key 不对或已失效，小鱼没能回答");
    expect(note("chat:quota")).toBe("对话大脑的额度可能用完了，小鱼没能回答");
    expect(note("chat:rate_limit")).toBe("对话大脑说请求太频繁，小鱼没能回答，稍等一下再说");
    expect(note("reasoning:error")).toBe("推理大脑这次出错了，小鱼没能回答，再说一遍试试");
    expect(replyFailureNote(null, "小鱼", { said: false })).toBe("小鱼这次没能回答，再说一遍试试");
  });

  it("says why a reply stopped halfway or was never heard", () => {
    expect(replyFailureNote(serviceFailure("chat:connection"), "小鱼", { said: true }))
      .toBe("这句没能说完：连不上对话大脑");
    expect(replyFailureNote(null, "小鱼", { said: true })).toBe("这句没能说完");
    expect(replyFailureNote(serviceFailure("voice:rejected"), "小鱼", { said: true }))
      .toBe("这句没能念出来：说话的声音不接受这句话的内容");
  });

  it("sends key and quota problems to the section that fixes them", () => {
    const quota = serviceFailure("jev:quota")!;
    expect(needsSettings(quota)).toBe(true);
    expect(settingsSectionFor(quota)).toBe("jev");
    expect(settingsSectionFor(serviceFailure("reasoning:key")!)).toBe("ai");
    expect(needsSettings(serviceFailure("chat:timeout")!)).toBe(false);
    expect(failureReason(quota)).toBe("决策模型的额度可能用完了");
    expect(parseSenseAction({ type: "settings", section: "jev" })).toEqual({ type: "settings", section: "jev" });
    expect(parseSenseAction({ type: "settings", section: "elsewhere" })).toEqual({ type: "settings" });
  });

  it("keeps a message's failure when reading the conversation", () => {
    const base = { messageId: "m", sequence: 2, role: "assistant", content: "", status: "failed", createdAt: "2026-09-28T14:00:00Z" };
    expect(chatMessage({ ...base, failure: "jev:timeout" })?.failure).toEqual({ service: "jev", kind: "timeout" });
    expect(chatMessage({ ...base, failure: "nonsense" })?.failure).toBeUndefined();
  });
});
