import { mkdtemp, readFile } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GatewayServer, localAddresses, type ForwardResult } from "./gateway-server";
import { assertPrivateFile } from "../test-support/private-file";

const servers: GatewayServer[] = [];
afterEach(async () => { await Promise.all(servers.splice(0).map((server) => server.close())); });

async function freePort(): Promise<number> {
  return new Promise((resolve) => {
    const probe = createServer();
    probe.listen(0, "127.0.0.1", () => {
      const port = (probe.address() as { port: number }).port;
      probe.close(() => resolve(port));
    });
  });
}

async function gateway(forward = vi.fn(async (): Promise<ForwardResult> => ({ status: 202, body: { accepted: true, id: "x" } }))) {
  const keyFile = path.join(await mkdtemp(path.join(tmpdir(), "gateway-")), "key");
  const server = new GatewayServer({ keyFile, forward });
  servers.push(server);
  const port = await freePort();
  await server.configure({ enabled: true, port });
  const { key } = await server.status();
  const send = (body: unknown, headers: Record<string, string> = {}, route = "/events") =>
    fetch(`http://127.0.0.1:${port}${route}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", ...headers },
      body: typeof body === "string" ? body : JSON.stringify(body),
    });
  return { server, port, key, keyFile, forward, send };
}

describe("GatewayServer", () => {
  it("forwards an event sent with the key", async () => {
    const { send, forward } = await gateway();
    const response = await send({ text: "快递到了", speak: false });
    expect(response.status).toBe(202);
    expect(await response.json()).toEqual({ accepted: true, id: "x" });
    expect(forward).toHaveBeenCalledWith({ text: "快递到了", speak: false });
  });

  it("keeps out requests without the right key, and never forwards them", async () => {
    const { send, forward } = await gateway();
    expect((await send({ text: "a" }, { Authorization: "Bearer wrong" })).status).toBe(401);
    expect((await send({ text: "a" }, { Authorization: "" })).status).toBe(401);
    expect(forward).not.toHaveBeenCalled();
  });

  it("answers only POST /events with JSON of a bounded size", async () => {
    const { send, port, key } = await gateway();
    expect((await send({ text: "a" }, {}, "/other")).status).toBe(404);
    expect((await fetch(`http://127.0.0.1:${port}/events`, { headers: { Authorization: `Bearer ${key}` } })).status).toBe(405);
    expect((await send("text=a", { "Content-Type": "application/x-www-form-urlencoded" })).status).toBe(415);
    expect((await send("{oops")).status).toBe(400);
    expect((await send({ text: "x".repeat(70_000) })).status).toBe(413);
  });

  it("explains what the local service could not take", async () => {
    const forward = vi.fn(async (): Promise<ForwardResult> => ({
      status: 400, body: { error: { code: "invalid_request", message: "Field required" } },
    }));
    const { send } = await gateway(forward);
    const response = await send({ nope: 1 });
    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "内容不对：需要 text，可选 title、source、id、speak", detail: "Field required" });
    forward.mockRejectedValueOnce(new Error("down"));
    expect((await send({ text: "a" })).status).toBe(503);
  });

  it("limits how fast key holders can send", async () => {
    const { send } = await gateway();
    const statuses = [];
    for (let index = 0; index < 61; index += 1) statuses.push((await send({ text: String(index) })).status);
    expect(statuses.slice(0, 60).every((status) => status === 202)).toBe(true);
    expect(statuses[60]).toBe(429);
  });

  it("keeps its key private and replaces it on request", async () => {
    const { server, keyFile, key, send } = await gateway();
    await assertPrivateFile(keyFile);
    expect((await readFile(keyFile, "utf8")).trim()).toBe(key);
    const next = await server.resetKey();
    expect(next.key).not.toBe(key);
    expect((await send({ text: "a" })).status).toBe(401);
  });

  it("stops listening when switched off and says when the port is taken", async () => {
    const { server, port, send } = await gateway();
    await server.configure({ enabled: false, port });
    await expect(send({ text: "a" })).rejects.toThrow();
    const other = await gateway();
    await other.server.configure({ enabled: true, port: other.port });
    const clash = new GatewayServer({ keyFile: other.keyFile, forward: vi.fn() });
    servers.push(clash);
    await clash.configure({ enabled: true, port: other.port });
    const status = await clash.status();
    expect(status.listening).toBe(false);
    expect(status.error).toContain(`端口 ${other.port}`);
    expect(status.addresses[0]).toBe(`http://127.0.0.1:${other.port}/events`);
  });

  it("lists only addresses other devices can reach", () => {
    const item = (address: string, internal = false) => ({ address, internal, family: "IPv4" as const, netmask: "", mac: "", cidr: null });
    expect(localAddresses({
      lo0: [item("127.0.0.1", true)],
      en0: [item("192.168.1.85")],
      utun4: [item("198.18.0.1")],
      en5: [item("169.254.3.2")],
      utun7: [item("100.101.1.2")],
      en8: [item("10.0.0.3"), item("172.20.1.1")],
    })).toEqual(["192.168.1.85", "100.101.1.2", "10.0.0.3", "172.20.1.1"]);
  });
});
