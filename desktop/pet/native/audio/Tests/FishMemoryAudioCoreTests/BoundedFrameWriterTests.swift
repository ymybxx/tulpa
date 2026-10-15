import FishMemoryAudioCore
import Foundation
import XCTest

final class BoundedFrameWriterTests: XCTestCase {
    func testWritesOnWorkerAndStopsAfterDraining() {
        let written = ThreadSafeFrames()
        let writer = BoundedFrameWriter(capacity: 4) { data in
            written.append(data)
        }
        writer.start()
        XCTAssertTrue(writer.enqueue(Data([1])))
        XCTAssertTrue(writer.enqueue(Data([2])))
        writer.stop(discardPending: false)

        XCTAssertTrue(writer.waitUntilStopped(timeout: .now() + 2))
        XCTAssertEqual(written.snapshot, [Data([1]), Data([2])])
        XCTAssertEqual(writer.snapshot.writtenFrameCount, 2)
        XCTAssertFalse(writer.snapshot.isAcceptingFrames)
    }

    func testQueueIsBoundedAndDropsOldestFrame() {
        let mayWrite = DispatchSemaphore(value: 0)
        let didEnterWrite = DispatchSemaphore(value: 0)
        let written = ThreadSafeFrames()
        let writer = BoundedFrameWriter(capacity: 2) { data in
            didEnterWrite.signal()
            mayWrite.wait()
            written.append(data)
        }
        writer.start()
        XCTAssertTrue(writer.enqueue(Data([0])))
        XCTAssertEqual(didEnterWrite.wait(timeout: .now() + 2), .success)

        XCTAssertTrue(writer.enqueue(Data([1])))
        XCTAssertTrue(writer.enqueue(Data([2])))
        XCTAssertTrue(writer.enqueue(Data([3])))
        XCTAssertEqual(writer.snapshot.queuedFrameCount, 2)
        XCTAssertEqual(writer.snapshot.droppedFrameCount, 1)

        mayWrite.signal()
        mayWrite.signal()
        mayWrite.signal()
        writer.stop(discardPending: false)
        XCTAssertTrue(writer.waitUntilStopped(timeout: .now() + 2))

        XCTAssertEqual(written.snapshot, [Data([0]), Data([2]), Data([3])])
    }

    func testCumulativeFrameSurvivesCaptureSaturationAndCoalesces() {
        let mayWrite = DispatchSemaphore(value: 0)
        let didEnterWrite = DispatchSemaphore(value: 0)
        let written = ThreadSafeFrames()
        let writer = BoundedFrameWriter(capacity: 3) { data in
            didEnterWrite.signal()
            mayWrite.wait()
            written.append(data)
        }
        writer.start()
        XCTAssertTrue(writer.enqueue(Data([0])))
        XCTAssertEqual(didEnterWrite.wait(timeout: .now() + 2), .success)

        XCTAssertTrue(writer.enqueue(Data([1])))
        XCTAssertTrue(writer.enqueue(
            Data([90]),
            priority: .latestCumulative(key: "playback")
        ))
        XCTAssertTrue(writer.enqueue(Data([2])))
        XCTAssertTrue(writer.enqueue(Data([3])))
        XCTAssertTrue(writer.enqueue(
            Data([99]),
            priority: .latestCumulative(key: "playback")
        ))
        XCTAssertTrue(writer.enqueue(Data([4])))

        let saturated = writer.snapshot
        XCTAssertEqual(saturated.queuedFrameCount, 3)
        XCTAssertEqual(saturated.queuedCumulativeFrameCount, 1)

        mayWrite.signal()
        mayWrite.signal()
        mayWrite.signal()
        mayWrite.signal()
        writer.stop(discardPending: false)
        XCTAssertTrue(writer.waitUntilStopped(timeout: .now() + 2))

        XCTAssertEqual(written.snapshot.first, Data([0]))
        XCTAssertEqual(written.snapshot.dropFirst().first, Data([99]))
        XCTAssertTrue(written.snapshot.contains(Data([99])))
        XCTAssertFalse(written.snapshot.contains(Data([90])))
    }
}

private final class ThreadSafeFrames: @unchecked Sendable {
    private let lock = NSLock()
    private var frames: [Data] = []

    func append(_ data: Data) {
        lock.lock()
        frames.append(data)
        lock.unlock()
    }

    var snapshot: [Data] {
        lock.lock()
        defer { lock.unlock() }
        return frames
    }
}
