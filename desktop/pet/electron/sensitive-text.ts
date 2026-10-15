const LABELED_SECRET = /\b(api[ _-]?key|access[ _-]?token|auth(?:orization)?|bearer|client[ _-]?secret|password|passwd|secret|token)\b\s*(?::|=|\bis\b)\s*(?:"[^"]{4,}"|'[^']{4,}'|[^\s,;]{4,})/giu;
const CHINESE_LABELED_SECRET = /(密码|密钥|令牌|访问令牌|鉴权)\s*(?:：|:|=|是)\s*(?:"[^"]{4,}"|'[^']{4,}'|[^\s，,；;]{4,})/gu;
const ENV_SECRET = /\b([A-Z][A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|TOKEN|SECRET|PASSWORD|PASSWD|PWD)[A-Z0-9_]*)\s*=\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)/gu;
const AUTHORIZATION_BEARER = /\bAuthorization\s*[:=]\s*Bearer\s+[A-Za-z0-9._~+/=-]{8,}/giu;
const BEARER_TOKEN = /\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/gu;
const JWT = /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/gu;
const PROVIDER_TOKEN = /\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{16,}|AKIA[0-9A-Z]{16})\b/gu;
const URL_CREDENTIALS = /(https?:\/\/)[^/@\s:]+:[^@/\s]+@/giu;
const SECRET_QUERY = /([?&](?:api[_-]?key|access[_-]?token|password|secret|token)=)[^&#\s]+/giu;
const PRIVATE_KEY = /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)/gu;
const LONG_OPAQUE_VALUE = /\b(?:[A-Fa-f0-9]{40,}|[A-Za-z0-9+/]{48,}={0,2})\b/gu;

/** Best-effort local redaction before derived screen text reaches durable state. */
export function redactSensitiveText(value: string, maximum: number): string {
  let redacted = value
    .replace(PRIVATE_KEY, "[REDACTED]")
    .replace(AUTHORIZATION_BEARER, "Authorization=[REDACTED]")
    .replace(BEARER_TOKEN, "Bearer [REDACTED]")
    .replace(ENV_SECRET, "$1=[REDACTED]")
    .replace(LABELED_SECRET, "$1=[REDACTED]")
    .replace(CHINESE_LABELED_SECRET, "$1=[REDACTED]")
    .replace(JWT, "[REDACTED]")
    .replace(PROVIDER_TOKEN, "[REDACTED]")
    .replace(URL_CREDENTIALS, "$1[REDACTED]@")
    .replace(SECRET_QUERY, "$1[REDACTED]")
    .replace(LONG_OPAQUE_VALUE, "[REDACTED]");
  redacted = redacted.split(/\s+/u).join(" ").trim();
  return [...redacted].slice(0, maximum).join("");
}
