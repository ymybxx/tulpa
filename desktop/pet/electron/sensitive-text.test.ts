import { describe, expect, it } from "vitest";

import { redactSensitiveText } from "./sensitive-text";

describe("redactSensitiveText", () => {
  it("redacts common credentials while preserving useful surrounding context", () => {
    const result = redactSensitiveText([
      "Editing deployment config",
      "OPENAI_API_KEY=sk-1234567890abcdefghijklmnop",
      "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
      "访问令牌：secret-value-123456",
      "https://user:password@example.test/path?token=query-secret",
      "-----BEGIN PRIVATE KEY----- abcdefghijklmnopqrstuvwxyz0123456789 -----END PRIVATE KEY-----",
    ].join("\n"), 500);

    expect(result).toContain("Editing deployment config");
    expect(result).toContain("[REDACTED]");
    expect(result).not.toContain("1234567890abcdefghijklmnop");
    expect(result).not.toContain("abcdefghijklmnopqrstuvwxyz");
    expect(result).not.toContain("secret-value-123456");
    expect(result).not.toContain("user:password");
    expect(result).not.toContain("query-secret");
    expect(result).not.toContain("BEGIN PRIVATE KEY");
  });

  it("bounds the redacted result by Unicode characters", () => {
    expect(redactSensitiveText("你好".repeat(20), 7)).toBe("你好你好你好你");
  });
});
