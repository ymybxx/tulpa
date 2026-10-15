import Foundation

public struct BoundedFrameWriterSnapshot: Equatable, Sendable {
    public let queuedFrameCount: Int
    public let queuedCumulativeFrameCount: Int
    public let droppedFrameCount: UInt64
    public let writtenFrameCount: UInt64
    public let isAcceptingFrames: Bool
}

public enum BoundedFrameWriterPriority: Equatable, Sendable {
    /// Ordinary high-rate data such as microphone capture. Under saturation an
    /// ordinary frame may never evict a cumulative acknowledgement.
    case normal

    /// Only the newest queued value for a key is useful. Playback render
    /// acknowledgements are cumulative, so replacing an older one is lossless.
    case latestCumulative(key: String)
}

public final class BoundedFrameWriter: @unchecked Sendable {
    public typealias Write = @Sendable (Data) throws -> Void
    public typealias OnWriteError = @Sendable (Error) -> Void

    private struct State {
        var frames: [QueuedFrame] = []
        var droppedFrameCount: UInt64 = 0
        var writtenFrameCount: UInt64 = 0
        var isAcceptingFrames = true
        var shouldStop = false
    }

    private struct QueuedFrame {
        let data: Data
        let priority: BoundedFrameWriterPriority
    }

    private let capacity: Int
    private let write: Write
    private let onWriteError: OnWriteError
    private let condition = NSCondition()
    private var state = State()
    private let workerFinished = DispatchGroup()
    private lazy var worker: Thread = {
        let thread = Thread { [weak self] in
            self?.runWriterLoop()
        }
        thread.name = "fish-memory.audio-frame-writer"
        thread.qualityOfService = .userInitiated
        return thread
    }()

    public init(
        capacity: Int,
        write: @escaping Write,
        onWriteError: @escaping OnWriteError = { _ in }
    ) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.write = write
        self.onWriteError = onWriteError
    }

    deinit {
        stop(discardPending: true)
    }

    public func start() {
        condition.lock()
        let shouldStart = !worker.isExecuting && !worker.isFinished
        if shouldStart {
            workerFinished.enter()
        }
        condition.unlock()
        if shouldStart {
            worker.start()
        }
    }

    @discardableResult
    public func enqueue(
        _ frame: Data,
        priority: BoundedFrameWriterPriority = .normal
    ) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        guard state.isAcceptingFrames else { return false }

        if case let .latestCumulative(key) = priority {
            let matchingIndexes = state.frames.indices.filter { index in
                state.frames[index].priority == .latestCumulative(key: key)
            }
            for index in matchingIndexes.reversed() {
                state.frames.remove(at: index)
                state.droppedFrameCount &+= 1
            }
        }

        if state.frames.count == capacity {
            let ordinaryIndex = state.frames.firstIndex {
                $0.priority == .normal
            }
            if let ordinaryIndex {
                state.frames.remove(at: ordinaryIndex)
                state.droppedFrameCount &+= 1
            } else if case .normal = priority {
                // A capture frame must not evict the final cumulative playback
                // acknowledgement; discard this incoming capture instead.
                state.droppedFrameCount &+= 1
                return false
            } else {
                state.frames.removeFirst()
                state.droppedFrameCount &+= 1
            }
        }
        state.frames.append(QueuedFrame(data: frame, priority: priority))
        condition.signal()
        return true
    }

    public func stop(discardPending: Bool) {
        condition.lock()
        state.isAcceptingFrames = false
        state.shouldStop = true
        if discardPending {
            state.droppedFrameCount &+= UInt64(state.frames.count)
            state.frames.removeAll(keepingCapacity: false)
        }
        condition.broadcast()
        condition.unlock()
    }

    @discardableResult
    public func waitUntilStopped(timeout: DispatchTime) -> Bool {
        workerFinished.wait(timeout: timeout) == .success
    }

    public var snapshot: BoundedFrameWriterSnapshot {
        condition.lock()
        defer { condition.unlock() }
        return BoundedFrameWriterSnapshot(
            queuedFrameCount: state.frames.count,
            queuedCumulativeFrameCount: state.frames.reduce(into: 0) {
                count, frame in
                if case .latestCumulative = frame.priority {
                    count += 1
                }
            },
            droppedFrameCount: state.droppedFrameCount,
            writtenFrameCount: state.writtenFrameCount,
            isAcceptingFrames: state.isAcceptingFrames
        )
    }

    private func runWriterLoop() {
        defer { workerFinished.leave() }
        while let frame = nextFrame() {
            do {
                try write(frame)
                condition.lock()
                state.writtenFrameCount &+= 1
                condition.unlock()
            } catch {
                condition.lock()
                state.isAcceptingFrames = false
                state.shouldStop = true
                state.droppedFrameCount &+= UInt64(state.frames.count)
                state.frames.removeAll(keepingCapacity: false)
                condition.broadcast()
                condition.unlock()
                onWriteError(error)
                return
            }
        }
    }

    private func nextFrame() -> Data? {
        condition.lock()
        defer { condition.unlock() }
        while state.frames.isEmpty, !state.shouldStop {
            condition.wait()
        }
        if let cumulativeIndex = state.frames.firstIndex(where: {
            if case .latestCumulative = $0.priority { return true }
            return false
        }) {
            return state.frames.remove(at: cumulativeIndex).data
        }
        if let frame = state.frames.first {
            state.frames.removeFirst()
            return frame.data
        }
        return nil
    }
}
