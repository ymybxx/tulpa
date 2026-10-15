import Foundation
import XCTest
@testable import PerceptionCore

final class EventWriterTests: XCTestCase {
    @MainActor
    func testBlockedBulkWriteDoesNotBlockMainActorAndControlJumpsQueue() async throws {
        let sink = BlockingEventSink()
        let writer = PerceptionEventWriter(
            maximumBulkEvents: 2,
            sink: { sink.write($0) }
        )
        writer.send(event: "system_audio_window", body: Self.window(index: 1))
        for _ in 0..<1_000 where !sink.firstWriteStarted { await Task.yield() }
        XCTAssertTrue(sink.firstWriteStarted)

        writer.send(event: "system_audio_window", body: Self.window(index: 2))
        writer.send(event: "system_audio_window", body: Self.window(index: 3))
        writer.send(event: "system_audio_window", body: Self.window(index: 4))
        // This executes on MainActor while the sink worker is blocked. The old
        // synchronous writer would hang here waiting for its global lock.
        writer.send(
            event: "ack",
            requestId: "ping-1",
            body: HelperAcknowledgement(command: "ping")
        )

        sink.releaseFirstWrite()
        await writer.flush()
        let envelopes = try sink.outputs.map(Self.decode)
        XCTAssertEqual(
            envelopes.compactMap { $0["event"] as? String },
            ["system_audio_window", "ack",
             "system_audio_window", "system_audio_window"]
        )
        XCTAssertEqual(envelopes[1]["requestId"] as? String, "ping-1")
        XCTAssertEqual(
            envelopes.compactMap { $0["sequence"] as? Int },
            [1, 2, 3, 4]
        )
        let windowIndexes = envelopes.compactMap { envelope -> Int? in
            guard envelope["event"] as? String == "system_audio_window",
                  let body = envelope["body"] as? [String: Any]
            else { return nil }
            return body["windowIndex"] as? Int
        }
        XCTAssertEqual(windowIndexes, [1, 3, 4])
    }

    @MainActor
    func testBulkQueueIsAlsoBoundedByEncodedBodyBytes() async throws {
        let sink = BlockingEventSink()
        let body = Self.window(index: 1)
        let encodedBytes = try JSONEncoder().encode(body).count
        let writer = PerceptionEventWriter(
            maximumBulkEvents: 10,
            maximumBulkBytes: encodedBytes * 2,
            sink: { sink.write($0) }
        )
        writer.send(event: "system_audio_window", body: body)
        for _ in 0..<1_000 where !sink.firstWriteStarted { await Task.yield() }
        XCTAssertTrue(sink.firstWriteStarted)

        writer.send(event: "system_audio_window", body: Self.window(index: 2))
        writer.send(event: "system_audio_window", body: Self.window(index: 3))
        writer.send(event: "system_audio_window", body: Self.window(index: 4))

        sink.releaseFirstWrite()
        await writer.flush()
        let envelopes = try sink.outputs.map(Self.decode)
        let windowIndexes = envelopes.compactMap { envelope -> Int? in
            guard let body = envelope["body"] as? [String: Any] else { return nil }
            return body["windowIndex"] as? Int
        }
        XCTAssertEqual(windowIndexes, [1, 3, 4])
    }

    @MainActor
    func testControlQueueCoalescesStateAndKeepsResponsesAheadOfIt() async throws {
        let sink = BlockingEventSink()
        let writer = PerceptionEventWriter(
            maximumControlEvents: 3,
            sink: { sink.write($0) }
        )
        writer.send(event: "system_audio_window", body: Self.window(index: 1))
        for _ in 0..<1_000 where !sink.firstWriteStarted { await Task.yield() }
        XCTAssertTrue(sink.firstWriteStarted)

        writer.send(event: "status", body: ["revision": 1])
        writer.send(event: "status", body: ["revision": 2])
        writer.send(event: "ack", requestId: "first", body: ["ok": true])
        writer.send(event: "ack", requestId: "second", body: ["ok": true])

        sink.releaseFirstWrite()
        await writer.flush()
        let envelopes = try sink.outputs.map(Self.decode)
        XCTAssertEqual(
            envelopes.compactMap { $0["event"] as? String },
            ["system_audio_window", "ack", "ack", "status"]
        )
        XCTAssertEqual(envelopes[1]["requestId"] as? String, "first")
        XCTAssertEqual(envelopes[2]["requestId"] as? String, "second")
        XCTAssertEqual(
            (envelopes[3]["body"] as? [String: Any])?["revision"] as? Int,
            2
        )
    }

    @MainActor
    func testNewestResponseEvictsStaleStateThenOldestResponseAtControlLimit() async throws {
        let sink = BlockingEventSink()
        let writer = PerceptionEventWriter(
            maximumControlEvents: 2,
            sink: { sink.write($0) }
        )
        writer.send(event: "system_audio_window", body: Self.window(index: 1))
        for _ in 0..<1_000 where !sink.firstWriteStarted { await Task.yield() }
        XCTAssertTrue(sink.firstWriteStarted)

        writer.send(event: "status", body: ["revision": 1])
        writer.send(event: "ack", requestId: "old", body: ["ok": true])
        writer.send(event: "ack", requestId: "new", body: ["ok": true])
        writer.send(event: "ack", requestId: "shutdown", body: ["ok": true])

        sink.releaseFirstWrite()
        await writer.flush()
        let envelopes = try sink.outputs.map(Self.decode)
        XCTAssertEqual(
            envelopes.compactMap { $0["requestId"] as? String },
            ["new", "shutdown"]
        )
        XCTAssertFalse(envelopes.contains { $0["event"] as? String == "status" })
    }

    @MainActor
    func testConcurrentFlushCallersShareOneStoredBarrier() async {
        let sink = BlockingEventSink()
        let writer = PerceptionEventWriter(sink: { sink.write($0) })
        writer.send(event: "system_audio_window", body: Self.window(index: 1))
        for _ in 0..<1_000 where !sink.firstWriteStarted { await Task.yield() }
        XCTAssertTrue(sink.firstWriteStarted)

        let flushes = (0..<100).map { _ in Task { await writer.flush() } }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertEqual(writer.pendingFlushBarrierCount, 1)

        sink.releaseFirstWrite()
        for flush in flushes { await flush.value }
        XCTAssertEqual(writer.pendingFlushBarrierCount, 0)
    }

    @MainActor
    func testRuntimeShutdownWaitsForAsynchronousWriterFlush() async {
        let sink = BlockingEventSink()
        let writer = PerceptionEventWriter(sink: { sink.write($0) })
        var exited = false
        let runtime = PerceptionRuntime(writer: writer, onShutdown: { exited = true })
        runtime.start()
        for _ in 0..<1_000 where !sink.firstWriteStarted { await Task.yield() }
        XCTAssertTrue(sink.firstWriteStarted)

        let shutdown = Task { @MainActor in await runtime.shutdown() }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertFalse(exited)
        sink.releaseFirstWrite()
        await shutdown.value
        XCTAssertTrue(exited)
    }

    private struct WindowFixture: Encodable {
        let captureSessionId: String
        let windowIndex: Int
        let payload: String
    }

    private static func window(index: Int) -> WindowFixture {
        .init(
            captureSessionId: "session",
            windowIndex: index,
            payload: String(repeating: String(index), count: 32)
        )
    }

    private static func decode(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }
}

private final class BlockingEventSink: @unchecked Sendable {
    private let lock = NSLock()
    private let release = DispatchSemaphore(value: 0)
    private var writeCount = 0
    private var stored: [Data] = []

    var firstWriteStarted: Bool {
        lock.lock()
        defer { lock.unlock() }
        return writeCount > 0
    }

    var outputs: [Data] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }

    func write(_ data: Data) {
        lock.lock()
        writeCount += 1
        let shouldBlock = writeCount == 1
        lock.unlock()
        if shouldBlock { release.wait() }
        lock.lock()
        stored.append(data)
        lock.unlock()
    }

    func releaseFirstWrite() { release.signal() }
}
