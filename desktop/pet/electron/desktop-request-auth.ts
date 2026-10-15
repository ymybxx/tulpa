export interface DesktopRequestAuthorization {
  origin: string;
  bearer: string;
}

export interface DesktopRequestContext {
  webContentsId?: number;
  attachedWebContentsId?: number;
  resourceType?: string;
  frameOrigin?: string;
}

export function authorizeDesktopRequest(
  url: string,
  context: DesktopRequestContext,
  mainWebContentsId: number | undefined,
  headers: Record<string, string>,
  authorization: DesktopRequestAuthorization | null,
): Record<string, string> {
  if (!authorization || mainWebContentsId === undefined) return headers;
  if (apiOrigin(url) !== authorization.origin) return headers;

  const isWebSocket = context.resourceType === "webSocket";
  if (isWebSocket) {
    // A main-window view can embed content from another origin. Do not let an
    // external frame turn the local WebSocket into a bearer-token oracle.
    const requestOrigin = requestHeader(headers, "origin");
    if (
      requestOrigin !== authorization.origin &&
      context.frameOrigin !== authorization.origin
    ) return headers;
  }

  const ownerIds = [context.webContentsId, context.attachedWebContentsId].filter(
    (value): value is number => value !== undefined,
  );
  if (ownerIds.length > 0) {
    if (ownerIds.some((ownerId) => ownerId !== mainWebContentsId)) return headers;
  } else {
    // Electron may omit both WebContents identifiers for a browser WebSocket.
    // In that narrow case, authorize only a local API-origin handshake. Never
    // use this origin fallback for HTTP or for a request owned by another view.
    if (!isWebSocket) return headers;
  }
  return { ...headers, Authorization: authorization.bearer };
}

function requestHeader(headers: Record<string, string>, expectedName: string): string | null {
  const entry = Object.entries(headers).find(
    ([name]) => name.toLowerCase() === expectedName,
  );
  return entry?.[1] ?? null;
}

function apiOrigin(url: string): string | null {
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "ws:") parsed.protocol = "http:";
    else if (parsed.protocol === "wss:") parsed.protocol = "https:";
    else if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
    return parsed.origin;
  } catch {
    return null;
  }
}
