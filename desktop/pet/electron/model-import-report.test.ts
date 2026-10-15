import { describe, expect, it } from "vitest";

import { invalidImportReport, validImportReport } from "./model-import-report";

describe("model import reports", () => {
  it("summarizes successful checks and warnings", () => {
    const report = validImportReport(
      [{ code: "entry", message: "找到模型入口", file: "girl.model3.json" }],
      [{ code: "moc-size", message: "MOC 文件异常小", file: "girl.moc3" }],
    );

    expect(report.status).toBe("warning");
    expect(report.checks).toHaveLength(1);
    expect(report.warnings).toHaveLength(1);
    expect(report.errors).toEqual([]);
  });

  it("turns an import exception into an invalid report", () => {
    const report = invalidImportReport(new Error("模型路径越界"));

    expect(report.status).toBe("invalid");
    expect(report.errors[0]).toMatchObject({
      code: "import-failed",
      message: "模型路径越界",
      file: null,
    });
  });
});
