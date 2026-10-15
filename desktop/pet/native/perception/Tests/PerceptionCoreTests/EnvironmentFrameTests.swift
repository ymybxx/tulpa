import CoreGraphics
import CoreMedia
import CoreVideo
import XCTest
@testable import PerceptionCore

final class EnvironmentFrameTests: XCTestCase {
    func testModernEnvironmentCaptureDoesNotNeedALifecycleStream() {
        XCTAssertFalse(screenCaptureNeedsPersistentStream(
            legacyScreenAnalysis: false,
            screenshotManagerAvailable: true
        ))
        XCTAssertTrue(screenCaptureNeedsPersistentStream(
            legacyScreenAnalysis: true,
            screenshotManagerAvailable: true
        ))
        XCTAssertTrue(screenCaptureNeedsPersistentStream(
            legacyScreenAnalysis: false,
            screenshotManagerAvailable: false
        ))
    }

    func testEnvironmentConfigurationsDoNotContinuouslyScaleTheDisplay() {
        let lifecycle = makeScreenStreamConfiguration(
            displayWidth: 3_840,
            displayHeight: 2_160,
            legacyScreenAnalysis: false
        )
        XCTAssertEqual(lifecycle.width, 2)
        XCTAssertEqual(lifecycle.height, 2)
        XCTAssertEqual(lifecycle.sourceRect, CGRect(x: 0, y: 0, width: 1, height: 1))
        XCTAssertEqual(lifecycle.minimumFrameInterval, CMTime(value: 60, timescale: 1))
        XCTAssertEqual(lifecycle.queueDepth, 1)
        XCTAssertFalse(lifecycle.showsCursor)

        let screenshot = makeEnvironmentScreenshotConfiguration(
            displayWidth: 3_840,
            displayHeight: 2_160
        )
        XCTAssertEqual(screenshot.width, 960)
        XCTAssertEqual(screenshot.height, 540)
        XCTAssertEqual(screenshot.minimumFrameInterval, CMTime(value: 1, timescale: 1))
        XCTAssertEqual(screenshot.queueDepth, 1)
    }

    func testBrokerIsThreeSecondOneShotMemorySlot() throws {
        let clock = EnvironmentFrameTestClock(now: 100)
        let broker = EnvironmentFrameBroker(clock: { clock.now })
        let metadata = try XCTUnwrap(broker.store(
            jpeg: Data([0xff, 0xd8, 0xff, 0xff, 0xd9]),
            observedAt: 100,
            displayID: 2,
            pixelWidth: 960,
            pixelHeight: 540,
            generation: 7
        ))

        XCTAssertEqual(metadata.expiresAt, 103)
        XCTAssertEqual(metadata.byteLength, 5)
        let payload = try XCTUnwrap(broker.claim(frameId: metadata.frameId))
        XCTAssertEqual(payload.mimeType, "image/jpeg")
        XCTAssertEqual(payload.jpegBase64, "/9j//9k=")
        XCTAssertNil(broker.claim(frameId: metadata.frameId))
    }

    func testBrokerRejectsExpiredOversizeAndWrongGeneration() throws {
        let clock = EnvironmentFrameTestClock(now: 104)
        let broker = EnvironmentFrameBroker(clock: { clock.now })
        XCTAssertNil(broker.store(
            jpeg: Data([1]),
            observedAt: 100,
            displayID: 1,
            pixelWidth: 1,
            pixelHeight: 1,
            generation: 1
        ))
        clock.now = 100
        XCTAssertNil(broker.store(
            jpeg: Data(count: EnvironmentFrameBroker.maximumJPEGBytes + 1),
            observedAt: 100,
            displayID: 1,
            pixelWidth: 1,
            pixelHeight: 1,
            generation: 1
        ))
        let metadata = try XCTUnwrap(broker.store(
            jpeg: Data([1]),
            observedAt: 100,
            displayID: 1,
            pixelWidth: 1,
            pixelHeight: 1,
            generation: 4
        ))
        XCTAssertNil(broker.claim(frameId: metadata.frameId, generation: 3))
    }

    func testEncoderDefaultsBoundCadenceDimensionsAndBytes() {
        let configuration = EnvironmentFrameEncoderConfiguration()
        XCTAssertEqual(configuration.maximumLongEdge, 960)
        XCTAssertEqual(
            configuration.maximumJPEGBytes,
            EnvironmentFrameBroker.maximumJPEGBytes
        )
        XCTAssertEqual(configuration.minimumFrameInterval, 1)
    }

    func testEncoderLimitsFullScreenFramesToOnePerSecond() throws {
        let clock = EnvironmentFrameTestClock(now: 100)
        let broker = EnvironmentFrameBroker(clock: { clock.now })
        let delivered = expectation(description: "two rate-limited frames")
        delivered.expectedFulfillmentCount = 2
        let encoder = EnvironmentFrameEncoder(
            configuration: .init(minimumFrameInterval: 1),
            broker: broker,
            encode: { _ in
                EncodedEnvironmentFrame(
                    jpeg: Data([0xff, 0xd8, 0xff, 0xff, 0xd9]),
                    pixelWidth: 960,
                    pixelHeight: 540
                )
            },
            onAvailable: { _ in delivered.fulfill() }
        )
        let pixelBuffer = try XCTUnwrap(Self.pixelBuffer())
        encoder.activate(generation: 4)

        encoder.submit(
            pixelBuffer: pixelBuffer,
            generation: 4,
            observedAt: 100,
            displayID: 1
        )
        encoder.submit(
            pixelBuffer: pixelBuffer,
            generation: 4,
            observedAt: 100.2,
            displayID: 1
        )
        encoder.submit(
            pixelBuffer: pixelBuffer,
            generation: 4,
            observedAt: 101.1,
            displayID: 1
        )

        wait(for: [delivered], timeout: 1)
        XCTAssertEqual(broker.currentMetadata()?.observedAt, 101.1)
        encoder.invalidate(generation: 4)
        XCTAssertNil(broker.currentMetadata())
    }

    func testStaticPixelsAreCapturedAgainOnEachTickWithTheirActualCaptureTime() async throws {
        let state = SamplingTestState(pixels: try XCTUnwrap(Self.pixelBuffer()))
        let loop = EnvironmentFrameSamplingLoop(
            automatic: false,
            permitted: { state.permitted },
            capture: { generation in state.capture(generation: generation) },
            consume: { frame, _, current in if current() { state.deliver(frame) } },
            onInvalidate: { _ in state.clear() }
        )
        loop.activate(generation: 1)
        for _ in 0..<3 {
            loop.tick()
            await loop.waitUntilIdle()
        }
        XCTAssertEqual(state.captureCount, 3)
        XCTAssertEqual(state.observedTimes, [101, 102, 103])
        // The fake returns identical static pixels, but capture itself is called
        // each time. No previous JPEG or fabricated new frame is replayed.
        loop.invalidate()
        loop.tick()
        await loop.waitUntilIdle()
        XCTAssertEqual(state.captureCount, 3)
        XCTAssertTrue(state.observedTimes.isEmpty)
    }

    func testExplicitIntervalUpdateImmediatelyTicksEvenWhenValueIsUnchanged() async throws {
        let state = SamplingTestState(pixels: try XCTUnwrap(Self.pixelBuffer()))
        let loop = EnvironmentFrameSamplingLoop(
            automatic: false,
            permitted: { true },
            capture: { generation in state.capture(generation: generation) },
            consume: { frame, _, current in if current() { state.deliver(frame) } },
            onInvalidate: { _ in state.clear() }
        )
        loop.activate(generation: 1)

        XCTAssertTrue(loop.updateInterval(milliseconds: 1_000, tickImmediately: true))
        await loop.waitUntilIdle()
        XCTAssertEqual(state.captureCount, 1)
        XCTAssertTrue(loop.updateInterval(milliseconds: 5_000, tickImmediately: true))
        await loop.waitUntilIdle()
        XCTAssertEqual(state.captureCount, 2)
        XCTAssertTrue(loop.updateInterval(milliseconds: 5_000, tickImmediately: false))
        await Task.yield()
        XCTAssertEqual(state.captureCount, 2)
        XCTAssertFalse(loop.updateInterval(milliseconds: 249, tickImmediately: true))
        XCTAssertFalse(loop.updateInterval(milliseconds: 60_001, tickImmediately: true))
        XCTAssertEqual(state.captureCount, 2)
        loop.invalidate()
    }

    func testIntervalUpdateCancelsThePreviousTimerCadence() async throws {
        let state = SamplingTestState(pixels: try XCTUnwrap(Self.pixelBuffer()))
        let first = expectation(description: "initial immediate tick")
        let second = expectation(description: "reschedule immediate tick")
        let loop = EnvironmentFrameSamplingLoop(
            intervalMilliseconds: 250,
            permitted: { true },
            capture: { generation in
                let frame = state.capture(generation: generation)
                switch state.captureCount {
                case 1: first.fulfill()
                case 2: second.fulfill()
                default: break
                }
                return frame
            },
            consume: { frame, _, current in if current() { state.deliver(frame) } },
            onInvalidate: { _ in state.clear() }
        )
        loop.activate(generation: 1)
        await fulfillment(of: [first], timeout: 1)
        await loop.waitUntilIdle()

        XCTAssertTrue(loop.updateInterval(milliseconds: 60_000, tickImmediately: true))
        await fulfillment(of: [second], timeout: 1)
        await loop.waitUntilIdle()
        try await Task.sleep(nanoseconds: 400_000_000)
        XCTAssertEqual(state.captureCount, 2)
        loop.invalidate()
    }

    func testRescheduleRevokesOldTimerTickAlreadyCheckingPermission() async throws {
        let state = SamplingTestState(pixels: try XCTUnwrap(Self.pixelBuffer()))
        let permissionGate = SamplingTestPermissionGate()
        let entered = expectation(description: "old timer checks permission")
        permissionGate.onFirstCheck = { entered.fulfill() }
        let loop = EnvironmentFrameSamplingLoop(
            intervalMilliseconds: 250,
            permitted: { permissionGate.check() },
            capture: { generation in state.capture(generation: generation) },
            consume: { frame, _, current in if current() { state.deliver(frame) } },
            onInvalidate: { _ in state.clear() }
        )
        loop.activate(generation: 1)
        await fulfillment(of: [entered], timeout: 1)

        XCTAssertTrue(loop.updateInterval(milliseconds: 60_000, tickImmediately: false))
        permissionGate.open()
        try await Task.sleep(nanoseconds: 20_000_000)
        XCTAssertEqual(state.captureCount, 0)

        XCTAssertTrue(loop.updateInterval(milliseconds: 60_000, tickImmediately: true))
        await loop.waitUntilIdle()
        XCTAssertEqual(state.captureCount, 1)
        loop.invalidate()
    }

    func testBusyCaptureDropsTicksAndRestartWaitsForCancelledNativeCall() async throws {
        let state = SamplingTestState(pixels: try XCTUnwrap(Self.pixelBuffer()))
        let started = expectation(description: "first real capture started")
        let gate = SamplingTestGate()
        let loop = EnvironmentFrameSamplingLoop(
            automatic: false,
            permitted: { state.permitted },
            capture: { generation in
                let frame = state.capture(generation: generation)
                if generation == 1 {
                    started.fulfill()
                    // A non-cancellable ScreenshotManager callback still owns
                    // the native slot until it completes.
                    await gate.wait()
                }
                return frame
            },
            consume: { frame, _, current in if current() { state.deliver(frame) } },
            onInvalidate: { _ in state.clear() }
        )
        loop.activate(generation: 1)
        loop.tick()
        await fulfillment(of: [started], timeout: 1)
        for _ in 0..<10 { loop.tick() }
        XCTAssertEqual(state.captureCount, 1)
        loop.invalidate()
        loop.activate(generation: 2)
        loop.tick()
        XCTAssertEqual(state.captureCount, 1)
        gate.open()
        await loop.waitUntilIdle()
        XCTAssertTrue(state.observedTimes.isEmpty)
        loop.tick()
        await loop.waitUntilIdle()
        XCTAssertEqual(state.captureCount, 2)
        XCTAssertEqual(state.observedTimes, [102])
        loop.invalidate()
    }

    func testEncodingIsInsideSingleFlightAndLateFrameCannotEscapeCancellation() async throws {
        let state = SamplingTestState(pixels: try XCTUnwrap(Self.pixelBuffer()))
        let started = expectation(description: "encoding started")
        let gate = SamplingTestGate()
        let loop = EnvironmentFrameSamplingLoop(
            automatic: false,
            permitted: { state.permitted },
            capture: { generation in state.capture(generation: generation) },
            consume: { frame, _, current in
                started.fulfill()
                await gate.wait()
                if current() { state.deliver(frame) }
            },
            onInvalidate: { _ in state.clear() }
        )
        loop.activate(generation: 1)
        loop.tick()
        await fulfillment(of: [started], timeout: 1)
        for _ in 0..<10 { loop.tick() }
        XCTAssertEqual(state.captureCount, 1)
        loop.invalidate(generation: 1)
        // Lock/unlock can resume the same ScreenMonitor generation. The loop's
        // epoch must still reject the old encoder's result after reactivation.
        loop.activate(generation: 1)
        loop.tick()
        XCTAssertEqual(state.captureCount, 1)
        gate.open()
        await loop.waitUntilIdle()
        XCTAssertTrue(state.observedTimes.isEmpty)
        XCTAssertEqual(state.clearCount, 1)
        loop.invalidate()
    }

    func testPermissionLossWhileCaptureIsPendingClearsAndStopsTheLoop() async throws {
        let state = SamplingTestState(pixels: try XCTUnwrap(Self.pixelBuffer()))
        let started = expectation(description: "second capture started")
        let gate = SamplingTestGate()
        let loop = EnvironmentFrameSamplingLoop(
            automatic: false,
            permitted: { state.permitted },
            capture: { generation in
                let frame = state.capture(generation: generation)
                if state.captureCount == 2 {
                    started.fulfill()
                    await gate.wait()
                }
                return frame
            },
            consume: { frame, _, current in if current() { state.deliver(frame) } },
            onInvalidate: { _ in state.clear() }
        )
        loop.activate(generation: 1)
        loop.tick()
        await loop.waitUntilIdle()
        XCTAssertEqual(state.observedTimes, [101])
        loop.tick()
        await fulfillment(of: [started], timeout: 1)
        state.permitted = false
        gate.open()
        await loop.waitUntilIdle()
        XCTAssertTrue(state.observedTimes.isEmpty)
        XCTAssertEqual(state.clearCount, 1)
        state.permitted = true
        loop.tick()
        await loop.waitUntilIdle()
        XCTAssertEqual(state.captureCount, 2)
    }

    func testPeriodicEncoderDiscardsPixelsWhenInvalidatedDuringEncoding() async throws {
        let entered = expectation(description: "JPEG encoder running")
        let release = DispatchSemaphore(value: 0)
        let broker = EnvironmentFrameBroker(clock: { 100 })
        let delivered = expectation(description: "cancelled JPEG is private")
        delivered.isInverted = true
        let encoder = EnvironmentFrameEncoder(
            broker: broker,
            encode: { _ in
                entered.fulfill()
                _ = release.wait(timeout: .now() + 2)
                return EncodedEnvironmentFrame(jpeg: Data([1]), pixelWidth: 2, pixelHeight: 2)
            },
            onAvailable: { _ in delivered.fulfill() }
        )
        let frame = CapturedEnvironmentFrame(
            pixelBuffer: try XCTUnwrap(Self.pixelBuffer()), observedAt: 100, displayID: 1
        )
        encoder.activate(generation: 1)
        let task = Task { await encoder.submitAndWait(frame, generation: 1, isStillCurrent: { true }) }
        await fulfillment(of: [entered], timeout: 1)
        encoder.invalidate(generation: 1)
        release.signal()
        await task.value
        XCTAssertNil(broker.currentMetadata())
        await fulfillment(of: [delivered], timeout: 0.02)
    }

    func testSameGenerationRestartCannotOvertakeOldEncoderInvalidation() async throws {
        let state = SamplingTestState(pixels: try XCTUnwrap(Self.pixelBuffer()))
        let entered = expectation(description: "old encoder invalidation entered")
        let restarting = expectation(description: "same generation restart attempted")
        let release = DispatchSemaphore(value: 0)
        let broker = EnvironmentFrameBroker(clock: { 100 })
        let encoder = EnvironmentFrameEncoder(
            broker: broker,
            encode: { _ in .init(jpeg: Data([1]), pixelWidth: 2, pixelHeight: 2) },
            onAvailable: { _ in }
        )
        let loop = EnvironmentFrameSamplingLoop(
            automatic: false,
            permitted: { true },
            capture: { generation in state.capture(generation: generation) },
            consume: { frame, generation, current in
                await encoder.submitAndWait(frame, generation: generation, isStillCurrent: current)
            },
            onInvalidate: { generation in
                if state.clearCount == 0 {
                    entered.fulfill()
                    _ = release.wait(timeout: .now() + 2)
                }
                encoder.invalidate(generation: generation)
                state.clear()
            }
        )
        encoder.activate(generation: 1)
        loop.activate(generation: 1)
        let invalidating = Task.detached { loop.invalidate(generation: 1) }
        await fulfillment(of: [entered], timeout: 1)
        let reactivating = Task.detached {
            restarting.fulfill()
            loop.invalidate()
            encoder.activate(generation: 1)
            loop.activate(generation: 1)
        }
        await fulfillment(of: [restarting], timeout: 1)
        try await Task.sleep(nanoseconds: 20_000_000)
        release.signal()
        await invalidating.value
        await reactivating.value
        loop.tick()
        await loop.waitUntilIdle()
        XCTAssertEqual(broker.currentMetadata()?.observedAt, 101)
        loop.invalidate()
    }

    private static func pixelBuffer() -> CVPixelBuffer? {
        var value: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            2,
            2,
            kCVPixelFormatType_32BGRA,
            nil,
            &value
        )
        return status == kCVReturnSuccess ? value : nil
    }
}

private final class SamplingTestGate: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var opened = false

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if opened {
                lock.unlock()
                continuation.resume()
            } else {
                self.continuation = continuation
                lock.unlock()
            }
        }
    }

    func open() {
        lock.lock()
        opened = true
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume()
    }
}

private final class SamplingTestPermissionGate: @unchecked Sendable {
    private let lock = NSLock()
    private let release = DispatchSemaphore(value: 0)
    private var checks = 0
    var onFirstCheck: (@Sendable () -> Void)?

    func check() -> Bool {
        lock.lock()
        checks += 1
        let isFirst = checks == 1
        let onFirstCheck = self.onFirstCheck
        lock.unlock()
        if isFirst {
            onFirstCheck?()
            release.wait()
        }
        return true
    }

    func open() {
        release.signal()
    }
}

private final class SamplingTestState: @unchecked Sendable {
    private let pixels: CVPixelBuffer
    private let lock = NSLock()
    private var allowed = true
    private var captures = 0
    private var times: [Double] = []
    private var clears = 0
    init(pixels: CVPixelBuffer) { self.pixels = pixels }
    var permitted: Bool {
        get { lock.lock(); defer { lock.unlock() }; return allowed }
        set { lock.lock(); allowed = newValue; lock.unlock() }
    }
    var captureCount: Int { lock.lock(); defer { lock.unlock() }; return captures }
    var observedTimes: [Double] { lock.lock(); defer { lock.unlock() }; return times }
    var clearCount: Int { lock.lock(); defer { lock.unlock() }; return clears }
    func capture(generation: UInt64) -> CapturedEnvironmentFrame {
        lock.lock()
        captures += 1
        let observedAt = Double(100 + captures)
        lock.unlock()
        return CapturedEnvironmentFrame(pixelBuffer: pixels, observedAt: observedAt, displayID: 1)
    }
    func deliver(_ frame: CapturedEnvironmentFrame) {
        lock.lock(); times.append(frame.observedAt); lock.unlock()
    }
    func clear() { lock.lock(); clears += 1; times = []; lock.unlock() }
}

private final class EnvironmentFrameTestClock: @unchecked Sendable {
    var now: Double

    init(now: Double) {
        self.now = now
    }
}
