import Foundation

/// The protocol pipe can apply backpressure when Electron's main thread is
/// busy. Callers only encode and enqueue; one private worker owns every write,
/// so a blocked media event can never hold MainActor or the protocol lock.
public final class PerceptionEventWriter: @unchecked Sendable {
    typealias Sink = @Sendable (Data) throws -> Void

    private struct PendingEvent: Sendable {
        let event: String
        let requestId: String?
        let body: Data
    }

    private static let defaultMaximumControlEvents = 8
    private static let defaultMaximumBulkEvents = 16
    private static let defaultMaximumBulkBytes = 2 * 1_024 * 1_024
    private let sink: Sink
    private let maximumControlEvents: Int
    private let maximumBulkEvents: Int
    private let maximumBulkBytes: Int
    private let writeQueue = DispatchQueue(
        label: "app.fishmemory.perception.protocol-writer",
        qos: .userInitiated
    )
    private let lock = NSLock()
    private var controls: [PendingEvent] = []
    private var bulk: [PendingEvent] = []
    private var bulkBytes = 0
    private var flushBarrier: EventWriterFlushBarrier?
    private var flushTask: Task<Void, Never>?
    private var sequence: UInt64 = 0
    private var draining = false
    private var closed = false

    public convenience init(output: FileHandle = .standardOutput) {
        self.init { data in try output.write(contentsOf: data) }
    }

    init(
        maximumControlEvents: Int = defaultMaximumControlEvents,
        maximumBulkEvents: Int = defaultMaximumBulkEvents,
        maximumBulkBytes: Int = defaultMaximumBulkBytes,
        sink: @escaping Sink
    ) {
        precondition(maximumControlEvents > 0)
        precondition(maximumBulkEvents > 0)
        precondition(maximumBulkBytes > 0)
        self.maximumControlEvents = maximumControlEvents
        self.maximumBulkEvents = maximumBulkEvents
        self.maximumBulkBytes = maximumBulkBytes
        self.sink = sink
        controls.reserveCapacity(maximumControlEvents)
        bulk.reserveCapacity(maximumBulkEvents)
    }

    public func send<Body: Encodable>(
        event: String,
        requestId: String? = nil,
        body: Body
    ) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let bodyData = try? encoder.encode(body) else { return }
        enqueue(PendingEvent(
            event: event,
            requestId: requestId,
            body: bodyData
        ))
    }

    /// Wait until accepted events are written or deliberately discarded by the
    /// bounded bulk policy. The parent retains its three-second forced-exit
    /// boundary if the OS pipe itself never becomes writable.
    public func flush() async {
        guard let task = flushTaskIfNeeded() else { return }
        await task.value
    }

    private func flushTaskIfNeeded() -> Task<Void, Never>? {
        lock.lock()
        if !draining && controls.isEmpty && bulk.isEmpty {
            lock.unlock()
            return nil
        }
        let task: Task<Void, Never>
        if let current = flushTask {
            task = current
        } else {
            let barrier = EventWriterFlushBarrier()
            let current = Task.detached { await barrier.wait() }
            flushBarrier = barrier
            flushTask = current
            task = current
        }
        lock.unlock()
        return task
    }

    /// Exposed internally so the concurrency test can verify that every caller
    /// joins one drain barrier instead of allocating one stored continuation.
    var pendingFlushBarrierCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return flushBarrier == nil ? 0 : 1
    }

    private func enqueue(_ item: PendingEvent) {
        lock.lock()
        guard !closed else { lock.unlock(); return }
        if Self.isControlEvent(item.event) {
            enqueueControlLocked(item)
        } else {
            // A single media window is deliberately bounded by the protocol.
            // Refuse an unexpectedly oversized body instead of allowing a
            // blocked stdout pipe to grow the helper without limit.
            guard item.body.count <= maximumBulkBytes else {
                lock.unlock()
                return
            }
            while !bulk.isEmpty && (
                bulk.count >= maximumBulkEvents
                    || bulkBytes + item.body.count > maximumBulkBytes
            ) {
                bulkBytes -= bulk.removeFirst().body.count
            }
            bulk.append(item)
            bulkBytes += item.body.count
        }
        let shouldStart = !draining
        draining = true
        lock.unlock()
        if shouldStart { writeQueue.async { [weak self] in self?.drain() } }
    }

    private func drain() {
        while let item = takeNext() {
            do {
                sequence &+= 1
                try sink(try encode(item, sequence: sequence))
            } catch {
                closeAfterWriteFailure()
                return
            }
        }
    }

    private func takeNext() -> PendingEvent? {
        lock.lock()
        if !controls.isEmpty {
            let item = controls.removeFirst()
            lock.unlock()
            return item
        }
        if !bulk.isEmpty {
            let item = bulk.removeFirst()
            bulkBytes -= item.body.count
            lock.unlock()
            return item
        }
        draining = false
        let barrier = flushBarrier
        flushBarrier = nil
        flushTask = nil
        lock.unlock()
        barrier?.signal()
        return nil
    }

    private func closeAfterWriteFailure() {
        lock.lock()
        closed = true
        controls.removeAll()
        bulk.removeAll()
        bulkBytes = 0
        draining = false
        let barrier = flushBarrier
        flushBarrier = nil
        flushTask = nil
        lock.unlock()
        barrier?.signal()
    }

    /// Keep request responses ahead of replaceable state notifications. Under
    /// sustained backpressure, the newest state replaces the old one and a new
    /// response evicts stale state before it can evict an older response.
    private func enqueueControlLocked(_ item: PendingEvent) {
        let replaceable = Self.isReplaceableControl(item)
        if replaceable,
           let existing = controls.lastIndex(where: {
               Self.isReplaceableControl($0) && $0.event == item.event
           }) {
            controls[existing] = item
            return
        }
        if item.event == "hello",
           let existing = controls.firstIndex(where: { $0.event == "hello" }) {
            controls[existing] = item
            return
        }

        if controls.count >= maximumControlEvents {
            if let staleState = controls.firstIndex(where: Self.isReplaceableControl) {
                controls.remove(at: staleState)
            } else if replaceable {
                return
            } else if let oldestResponse = controls.firstIndex(where: { $0.event != "hello" }) {
                controls.remove(at: oldestResponse)
            } else {
                return
            }
        }

        if item.event == "hello" {
            controls.insert(item, at: 0)
        } else if replaceable {
            controls.append(item)
        } else {
            let firstState = controls.firstIndex(where: Self.isReplaceableControl)
                ?? controls.endIndex
            controls.insert(item, at: firstState)
        }
    }

    private func encode(_ item: PendingEvent, sequence: UInt64) throws -> Data {
        let body = try JSONSerialization.jsonObject(
            with: item.body,
            options: [.fragmentsAllowed]
        )
        var envelope: [String: Any] = [
            "protocolVersion": perceptionProtocolVersion,
            "sequence": sequence,
            "sentAt": Date().timeIntervalSince1970,
            "event": item.event,
            "body": body,
        ]
        if let requestId = item.requestId { envelope["requestId"] = requestId }
        var data = try JSONSerialization.data(
            withJSONObject: envelope,
            options: [.sortedKeys, .withoutEscapingSlashes]
        )
        data.append(0x0A)
        return data
    }

    private static func isControlEvent(_ event: String) -> Bool {
        switch event {
        case "system_audio_window", "observation",
             "interaction_event", "interaction_events_suppressed",
             "vision_candidate_available", "environment_frame_available":
            return false
        default:
            return true
        }
    }

    private static func isReplaceableControl(_ item: PendingEvent) -> Bool {
        item.requestId == nil && item.event != "hello"
    }
}

private final class EventWriterFlushBarrier: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var signalled = false

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if signalled {
                lock.unlock()
                continuation.resume()
            } else {
                precondition(self.continuation == nil)
                self.continuation = continuation
                lock.unlock()
            }
        }
    }

    func signal() {
        lock.lock()
        signalled = true
        let continuation = continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume()
    }
}
