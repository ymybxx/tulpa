import type {
  ModelImportIssue,
  ModelImportReport,
} from "../src/shared/contracts";

export function validImportReport(
  checks: ModelImportIssue[] = [],
  warnings: ModelImportIssue[] = [],
): ModelImportReport {
  return {
    status: warnings.length ? "warning" : "valid",
    checks,
    warnings,
    errors: [],
  };
}

export function invalidImportReport(error: unknown): ModelImportReport {
  return {
    status: "invalid",
    checks: [],
    warnings: [],
    errors: [{
      code: "import-failed",
      message: error instanceof Error ? error.message : String(error),
      file: null,
    }],
  };
}
