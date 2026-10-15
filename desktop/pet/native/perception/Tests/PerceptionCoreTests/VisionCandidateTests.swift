import CoreGraphics
import CoreVideo
import XCTest
@testable import PerceptionCore

final class VisionCandidateTests: XCTestCase {
    func testBrokerUsesCaptureTimeTTLAndClaimConsumesCandidate() throws {
        let clock = VisionTestClock(now: 101)
        let broker = VisionCandidateBroker(clock: { clock.now })
        let metadata = try XCTUnwrap(broker.store(
            jpeg: Data([1, 2, 3]),
            privacyContext: privacyContext(revision: 1, observedAt: 100),
            applicationContext: applicationContext(token: "WINDOW-1")
        ))

        XCTAssertEqual(metadata.expiresAt, 105)
        let payload = try XCTUnwrap(broker.claim(
            candidateId: metadata.candidateId,
            isStillAuthorized: { _, _ in true }
        ))
        XCTAssertEqual(payload.mimeType, "image/jpeg")
        XCTAssertEqual(payload.jpegBase64, "AQID")
        XCTAssertNil(broker.claim(
            candidateId: metadata.candidateId,
            isStillAuthorized: { _, _ in true }
        ))
    }

    func testExpiredEncodingIsNeverStoredAndExpiredClaimClearsSlot() throws {
        let clock = VisionTestClock(now: 106)
        let broker = VisionCandidateBroker(clock: { clock.now })
        XCTAssertNil(broker.store(
            jpeg: Data([1]),
            privacyContext: privacyContext(revision: 1, observedAt: 100),
            applicationContext: applicationContext()
        ))

        clock.now = 101
        let metadata = try XCTUnwrap(broker.store(
            jpeg: Data([1]),
            privacyContext: privacyContext(revision: 2, observedAt: 100),
            applicationContext: applicationContext()
        ))
        clock.now = 105
        XCTAssertNil(broker.claim(
            candidateId: metadata.candidateId,
            isStillAuthorized: { _, _ in true }
        ))
        XCTAssertNil(broker.currentMetadata())
    }

    func testNewCandidateSupersedesOldAndWrongIdDoesNotConsumeCurrent() throws {
        let clock = VisionTestClock(now: 101)
        let broker = VisionCandidateBroker(clock: { clock.now })
        let first = try XCTUnwrap(broker.store(
            jpeg: Data([1]),
            privacyContext: privacyContext(revision: 1, observedAt: 100),
            applicationContext: applicationContext(token: "WINDOW-1")
        ))
        let second = try XCTUnwrap(broker.store(
            jpeg: Data([2]),
            privacyContext: privacyContext(revision: 2, observedAt: 100),
            applicationContext: applicationContext(token: "WINDOW-1")
        ))

        XCTAssertNil(broker.claim(
            candidateId: first.candidateId,
            isStillAuthorized: { _, _ in true }
        ))
        XCTAssertEqual(broker.currentMetadata()?.candidateId, second.candidateId)
        XCTAssertNotNil(broker.claim(
            candidateId: second.candidateId,
            isStillAuthorized: { _, _ in true }
        ))
    }

    func testRevokedContextAndOversizePayloadAreUnavailable() throws {
        let clock = VisionTestClock(now: 101)
        let broker = VisionCandidateBroker(clock: { clock.now })
        XCTAssertNil(broker.store(
            jpeg: Data(count: VisionCandidateBroker.maximumJPEGBytes + 1),
            privacyContext: privacyContext(revision: 1, observedAt: 100),
            applicationContext: applicationContext()
        ))
        let metadata = try XCTUnwrap(broker.store(
            jpeg: Data([1]),
            privacyContext: privacyContext(revision: 2, observedAt: 100),
            applicationContext: applicationContext()
        ))
        XCTAssertNil(broker.claim(
            candidateId: metadata.candidateId,
            isStillAuthorized: { _, _ in false }
        ))
        XCTAssertNil(broker.currentMetadata())
    }

    func testEncoderIsDisabledByDefaultAndRejectsOversizeOutput() throws {
        let clock = VisionTestClock(now: 101)
        let broker = VisionCandidateBroker(clock: { clock.now })
        let encodeCount = LockedCounter()
        let encoder = VisionCandidateEncoder(
            configuration: .init(minimumCandidateInterval: 0),
            broker: broker,
            encode: { _, _ in
                encodeCount.increment()
                return Data(count: VisionCandidateBroker.maximumJPEGBytes + 1)
            },
            isStillAuthorized: { _, _ in true },
            onAvailable: { _ in XCTFail("oversize candidate must stay silent") }
        )
        encoder.activate(generation: 1, enabled: false)
        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 1, observedAt: 100),
            applicationContext: applicationContext()
        )
        Thread.sleep(forTimeInterval: 0.02)
        XCTAssertEqual(encodeCount.value, 0)

        encoder.setEnabled(true, generation: 1)
        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 2, observedAt: 100),
            applicationContext: applicationContext()
        )
        waitUntil { encodeCount.value == 1 }
        XCTAssertNil(broker.currentMetadata())
        encoder.invalidate(generation: 1)
    }

    func testEncoderRateLimitsSparseCandidatesFromCaptureTime() throws {
        let clock = VisionTestClock(now: 101)
        let broker = VisionCandidateBroker(clock: { clock.now })
        let encodeCount = LockedCounter()
        let encoder = VisionCandidateEncoder(
            configuration: .init(minimumCandidateInterval: 30),
            broker: broker,
            encode: { _, _ in
                encodeCount.increment()
                return Data([1])
            },
            isStillAuthorized: { _, _ in true },
            onAvailable: { _ in }
        )
        encoder.activate(generation: 1, enabled: true)

        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 1, observedAt: 100),
            applicationContext: applicationContext()
        )
        waitUntil { encodeCount.value == 1 }
        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 2, observedAt: 129.9),
            applicationContext: applicationContext()
        )
        Thread.sleep(forTimeInterval: 0.02)
        XCTAssertEqual(encodeCount.value, 1)

        clock.now = 131
        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 3, observedAt: 130),
            applicationContext: applicationContext()
        )
        waitUntil { encodeCount.value == 2 }
        encoder.invalidate(generation: 1)
    }

    func testRevokedFrameDoesNotConsumeSparseCandidateInterval() throws {
        let clock = VisionTestClock(now: 101)
        let broker = VisionCandidateBroker(clock: { clock.now })
        let authorization = MutableVisionAuthorization(allowed: false)
        let encodeCount = LockedCounter()
        let encoder = VisionCandidateEncoder(
            configuration: .init(minimumCandidateInterval: 30),
            broker: broker,
            encode: { _, _ in
                encodeCount.increment()
                return Data([1])
            },
            isStillAuthorized: { _, _ in authorization.allowed },
            onAvailable: { _ in }
        )
        encoder.activate(generation: 1, enabled: true)

        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 1, observedAt: 100),
            applicationContext: applicationContext()
        )
        Thread.sleep(forTimeInterval: 0.03)
        XCTAssertEqual(encodeCount.value, 0)

        authorization.allowed = true
        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 2, observedAt: 101),
            applicationContext: applicationContext()
        )
        waitUntil { encodeCount.value == 1 }
        encoder.invalidate(generation: 1)
    }

    func testEncoderKeepsLatestROIFrameAndStopDropsInflightResult() throws {
        let clock = VisionTestClock(now: 101)
        let broker = VisionCandidateBroker(clock: { clock.now })
        let firstStarted = expectation(description: "first encode started")
        let release = DispatchSemaphore(value: 0)
        let available = VisionMetadataRecorder()
        let encoder = VisionCandidateEncoder(
            configuration: .init(minimumCandidateInterval: 0),
            broker: broker,
            encode: { pixelBuffer, region in
                XCTAssertEqual(region, Self.region)
                let width = CVPixelBufferGetWidth(pixelBuffer)
                if width == 1 {
                    firstStarted.fulfill()
                    _ = release.wait(timeout: .now() + 2)
                }
                return Data([UInt8(width)])
            },
            isStillAuthorized: { _, _ in true },
            onAvailable: { available.record($0) }
        )
        encoder.activate(generation: 1, enabled: true)
        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 1),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 1, observedAt: 100),
            applicationContext: applicationContext()
        )
        wait(for: [firstStarted], timeout: 2)
        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 2),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 2, observedAt: 100),
            applicationContext: applicationContext()
        )
        encoder.submit(
            pixelBuffer: try pixelBuffer(width: 3),
            regionOfInterest: Self.region,
            generation: 1,
            privacyContext: privacyContext(revision: 3, observedAt: 100),
            applicationContext: applicationContext()
        )
        release.signal()
        waitUntil { available.revisions.count == 2 }
        XCTAssertEqual(available.revisions, [1, 3])

        let blockedStarted = expectation(description: "blocked encode started")
        let blockedRelease = DispatchSemaphore(value: 0)
        let stoppedBroker = VisionCandidateBroker(clock: { clock.now })
        let stoppedEncoder = VisionCandidateEncoder(
            broker: stoppedBroker,
            encode: { _, _ in
                blockedStarted.fulfill()
                _ = blockedRelease.wait(timeout: .now() + 2)
                return Data([1])
            },
            isStillAuthorized: { _, _ in true },
            onAvailable: { _ in XCTFail("stopped encoder must not publish") }
        )
        stoppedEncoder.activate(generation: 4, enabled: true)
        stoppedEncoder.submit(
            pixelBuffer: try pixelBuffer(width: 4),
            regionOfInterest: Self.region,
            generation: 4,
            privacyContext: privacyContext(revision: 4, observedAt: 100),
            applicationContext: applicationContext()
        )
        wait(for: [blockedStarted], timeout: 2)
        stoppedEncoder.invalidate(generation: 4)
        blockedRelease.signal()
        Thread.sleep(forTimeInterval: 0.02)
        XCTAssertNil(stoppedBroker.currentMetadata())
        encoder.invalidate(generation: 1)
    }

    func testPixelCropIsStrictlyInsideROIAndNeverFallsBackToFullImage() {
        XCTAssertEqual(VisionCandidateEncoder.inwardPixelCrop(
            regionOfInterest: CGRect(x: 0.25, y: 0.1, width: 0.5, height: 0.6),
            imageExtent: CGRect(x: 0, y: 0, width: 200, height: 100)
        ), CGRect(x: 50, y: 10, width: 100, height: 60))
        XCTAssertNil(VisionCandidateEncoder.inwardPixelCrop(
            regionOfInterest: CGRect(x: -0.01, y: 0, width: 1, height: 1),
            imageExtent: CGRect(x: 0, y: 0, width: 200, height: 100)
        ))
    }

    private func privacyContext(
        revision: UInt64,
        observedAt: Double
    ) -> ScreenTextPrivacyContext {
        ScreenTextPrivacyContext(
            revision: revision,
            observedAt: observedAt,
            displayID: 1
        )
    }

    private func applicationContext(
        token: String = "WINDOW-1"
    ) -> ScreenTextApplicationContext {
        ScreenTextApplicationContext(
            processIdentifier: 7,
            appBundleId: "com.example.editor",
            appName: "Editor",
            windowTitle: "Document",
            windowFrame: AccessibilityWindowFrame(
                x: 0,
                y: 0,
                width: 800,
                height: 600
            ),
            contextToken: token
        )
    }

    private func pixelBuffer(width: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        XCTAssertEqual(CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            1,
            kCVPixelFormatType_32BGRA,
            nil,
            &buffer
        ), kCVReturnSuccess)
        return try XCTUnwrap(buffer)
    }

    private func waitUntil(
        _ condition: () -> Bool,
        timeout: TimeInterval = 2
    ) {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition(), Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.005))
        }
        XCTAssertTrue(condition())
    }

    private static let region = CGRect(
        x: 0.1,
        y: 0.2,
        width: 0.7,
        height: 0.6
    )
}

private final class VisionTestClock: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Double

    init(now: Double) {
        stored = now
    }

    var now: Double {
        get {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }
        set {
            lock.lock()
            stored = newValue
            lock.unlock()
        }
    }
}

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = 0
    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
    func increment() {
        lock.lock()
        stored += 1
        lock.unlock()
    }
}

private final class MutableVisionAuthorization: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Bool

    init(allowed: Bool) {
        stored = allowed
    }

    var allowed: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }
        set {
            lock.lock()
            stored = newValue
            lock.unlock()
        }
    }
}

private final class VisionMetadataRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var metadata: [VisionCandidateMetadata] = []
    var revisions: [UInt64] {
        lock.lock()
        defer { lock.unlock() }
        return metadata.map(\.revision)
    }
    func record(_ value: VisionCandidateMetadata) {
        lock.lock()
        metadata.append(value)
        lock.unlock()
    }
}
