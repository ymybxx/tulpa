import CoreGraphics
import CoreVideo
import XCTest
@testable import PerceptionCore

final class FocusedWindowCaptureTests: XCTestCase {
    private let primaryDisplay = CGRect(x: 0, y: 0, width: 1_440, height: 900)

    func testMatcherRequiresOneExactOnScreenPIDBundleTitleAndFrameMatch() {
        let context = applicationContext()
        let exact = descriptor(windowID: 10)

        XCTAssertEqual(
            FocusedWindowMatcher.uniqueWindowID(
                for: context,
                displayFrame: primaryDisplay,
                in: [exact]
            ),
            10
        )
        XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: [descriptor(windowID: 11, processIdentifier: 8)]
        ))
        XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: [descriptor(windowID: 12, bundleIdentifier: "com.other.app")]
        ))
        XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: [descriptor(windowID: 13, title: "Other")]
        ))
        XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: [descriptor(
                windowID: 14,
                frame: CGRect(x: 20, y: 20, width: 800, height: 600)
            )]
        ))
        XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: [descriptor(windowID: 15, isOnScreen: false)]
        ))
    }

    func testMatcherRejectsMissingAndAmbiguousWindowsWithoutFallback() {
        let context = applicationContext()

        XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: []
        ))
        XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: [descriptor(windowID: 1), descriptor(windowID: 2)]
        ))
    }

    func testDisplayPolicyTrustsNegativeOriginSecondaryAndRejectsMissingOrMismatch() {
        let secondary = CGRect(x: -1_920, y: 0, width: 1_920, height: 1_080)
        let displays = [
            FocusedWindowDisplayDescriptor(displayID: 1, frame: primaryDisplay),
            FocusedWindowDisplayDescriptor(displayID: 2, frame: secondary),
        ]

        XCTAssertEqual(FocusedWindowDisplayPolicy.trustedDisplayFrame(
            requestedDisplayID: 2,
            displays: displays,
            coreGraphicsFrame: secondary
        ), secondary)
        XCTAssertNil(FocusedWindowDisplayPolicy.trustedDisplayFrame(
            requestedDisplayID: 3,
            displays: displays,
            coreGraphicsFrame: secondary
        ))
        XCTAssertNil(FocusedWindowDisplayPolicy.trustedDisplayFrame(
            requestedDisplayID: 2,
            displays: displays,
            coreGraphicsFrame: CGRect(x: 0, y: 0, width: 1_920, height: 1_080)
        ))
    }

    func testTriggerDisplayCannotAuthorizeWindowOnAnotherDisplay() {
        let windowOnDisplayB = CGRect(x: 1_440, y: 0, width: 1_200, height: 800)
        let context = applicationContext(frame: windowOnDisplayB)

        XCTAssertFalse(FocusedWindowDisplayPolicy.contains(
            context.windowFrame.rect,
            in: primaryDisplay
        ))
        XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: [descriptor(windowID: 20, frame: windowOnDisplayB)]
        ))
    }

    func testCrossDisplayAndPartiallyOffscreenWindowsFailClosed() {
        let crossing = CGRect(x: 1_300, y: 50, width: 300, height: 500)
        let partiallyOffscreen = CGRect(x: -1, y: 20, width: 800, height: 600)

        for frame in [crossing, partiallyOffscreen] {
            let context = applicationContext(frame: frame)
            XCTAssertFalse(FocusedWindowDisplayPolicy.contains(
                context.windowFrame.rect,
                in: primaryDisplay
            ))
            XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
                for: context,
                displayFrame: primaryDisplay,
                in: [descriptor(windowID: 21, frame: frame)]
            ))
        }
    }

    func testFullscreenWindowExactlyOnDisplayEdgesIsAllowed() {
        let context = applicationContext(frame: primaryDisplay)

        XCTAssertTrue(FocusedWindowDisplayPolicy.contains(
            context.windowFrame.rect,
            in: primaryDisplay
        ))
        XCTAssertEqual(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: [descriptor(windowID: 22, frame: primaryDisplay)]
        ), 22)
    }

    func testNegativeCoordinateSecondaryWindowMatchesOnlyItsOwnDisplay() {
        let secondary = CGRect(x: -1_920, y: 0, width: 1_920, height: 1_080)
        let window = CGRect(x: -1_820, y: 100, width: 960, height: 540)
        let context = applicationContext(frame: window)

        XCTAssertEqual(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: secondary,
            in: [descriptor(windowID: 23, frame: window)]
        ), 23)
        XCTAssertNil(FocusedWindowMatcher.uniqueWindowID(
            for: context,
            displayFrame: primaryDisplay,
            in: [descriptor(windowID: 23, frame: window)]
        ))
    }

    func testCoordinatorKeepsOneCaptureInFlightAndOnlyLatestPendingRequest() async {
        let capture = ControlledFocusedWindowCapture()
        let frames = FocusedWindowFrameRecorder()
        let failures = FocusedWindowFailureRecorder()
        let coordinator = FocusedWindowCaptureCoordinator(
            capture: { request in await capture.capture(request) },
            onFrame: { frames.record($0) },
            onFailure: { failures.record($0) }
        )
        coordinator.activate(generation: 1)
        let first = request(revision: 1)
        let middle = request(revision: 2)
        let latest = request(revision: 3)

        XCTAssertTrue(coordinator.submit(first))
        await waitUntil { capture.startedRevisions == [1] }
        XCTAssertTrue(coordinator.submit(middle))
        XCTAssertTrue(coordinator.submit(latest))
        XCTAssertEqual(capture.startedRevisions, [1])

        capture.resolve(revision: 1, withFrame: true)
        await waitUntil { capture.startedRevisions == [1, 3] }
        XCTAssertEqual(frames.revisions, [1])
        XCTAssertEqual(capture.maximumConcurrentCaptures, 1)

        capture.resolve(revision: 3, withFrame: true)
        await waitUntil { frames.revisions == [1, 3] }
        XCTAssertEqual(capture.startedRevisions, [1, 3])
        XCTAssertEqual(failures.revisions, [])
        coordinator.stop(generation: 1)
    }

    func testStopCancelsInFlightAndSuppressesLateFrameAndFailureCallbacks() async {
        let capture = ControlledFocusedWindowCapture()
        let frames = FocusedWindowFrameRecorder()
        let failures = FocusedWindowFailureRecorder()
        let coordinator = FocusedWindowCaptureCoordinator(
            capture: { request in await capture.capture(request) },
            onFrame: { frames.record($0) },
            onFailure: { failures.record($0) }
        )
        coordinator.activate(generation: 4)
        XCTAssertTrue(coordinator.submit(request(revision: 4, generation: 4)))
        await waitUntil { capture.startedRevisions == [4] }

        coordinator.stop(generation: 4)
        await waitUntil { capture.cancelledRevisions == [4] }

        XCTAssertEqual(frames.revisions, [])
        XCTAssertEqual(failures.revisions, [])
        XCTAssertFalse(coordinator.submit(request(revision: 5, generation: 4)))
    }

    private func descriptor(
        windowID: CGWindowID,
        processIdentifier: Int32 = 7,
        bundleIdentifier: String? = "com.example.editor",
        title: String? = "Document",
        frame: CGRect = CGRect(x: 10, y: 20, width: 800, height: 600),
        isOnScreen: Bool = true
    ) -> FocusedWindowDescriptor {
        FocusedWindowDescriptor(
            windowID: windowID,
            processIdentifier: processIdentifier,
            bundleIdentifier: bundleIdentifier,
            title: title,
            frame: frame,
            isOnScreen: isOnScreen
        )
    }

    private func request(
        revision: UInt64,
        generation: UInt64 = 1
    ) -> FocusedWindowCaptureRequest {
        FocusedWindowCaptureRequest(
            generation: generation,
            privacyContext: ScreenTextPrivacyContext(
                revision: revision,
                observedAt: Double(revision),
                displayID: 1
            ),
            applicationContext: applicationContext()
        )
    }

    private func applicationContext(
        frame: CGRect = CGRect(x: 10, y: 20, width: 800, height: 600)
    ) -> ScreenTextApplicationContext {
        ScreenTextApplicationContext(
            processIdentifier: 7,
            appBundleId: "com.example.editor",
            appName: "Editor",
            windowTitle: "Document",
            windowFrame: AccessibilityWindowFrame(
                x: frame.minX,
                y: frame.minY,
                width: frame.width,
                height: frame.height
            ),
            contextToken: "00000000-0000-0000-0000-000000000040"
        )
    }

    private func waitUntil(
        _ condition: @escaping @Sendable () -> Bool,
        iterations: Int = 1_000
    ) async {
        for _ in 0 ..< iterations {
            if condition() { return }
            try? await Task.sleep(nanoseconds: 1_000_000)
        }
        XCTFail("condition was not satisfied")
    }
}

private final class ControlledFocusedWindowCapture: @unchecked Sendable {
    private struct Pending {
        let request: FocusedWindowCaptureRequest
        let continuation: CheckedContinuation<FocusedWindowSurfaceFrame?, Never>
    }

    private let lock = NSLock()
    private var pending: [UInt64: Pending] = [:]
    private var started: [UInt64] = []
    private var cancelled: [UInt64] = []
    private var activeCount = 0
    private var maximumActiveCount = 0

    var startedRevisions: [UInt64] {
        lock.lock()
        defer { lock.unlock() }
        return started
    }

    var cancelledRevisions: [UInt64] {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    var maximumConcurrentCaptures: Int {
        lock.lock()
        defer { lock.unlock() }
        return maximumActiveCount
    }

    func capture(
        _ request: FocusedWindowCaptureRequest
    ) async -> FocusedWindowSurfaceFrame? {
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                lock.lock()
                activeCount += 1
                maximumActiveCount = max(maximumActiveCount, activeCount)
                started.append(request.privacyContext.revision)
                pending[request.privacyContext.revision] = Pending(
                    request: request,
                    continuation: continuation
                )
                lock.unlock()
            }
        } onCancel: {
            cancel(revision: request.privacyContext.revision)
        }
    }

    func resolve(revision: UInt64, withFrame: Bool) {
        lock.lock()
        guard let pending = pending.removeValue(forKey: revision) else {
            lock.unlock()
            return
        }
        activeCount -= 1
        lock.unlock()
        pending.continuation.resume(
            returning: withFrame ? makeFrame(request: pending.request) : nil
        )
    }

    private func cancel(revision: UInt64) {
        lock.lock()
        guard let pending = pending.removeValue(forKey: revision) else {
            lock.unlock()
            return
        }
        activeCount -= 1
        cancelled.append(revision)
        lock.unlock()
        pending.continuation.resume(returning: nil)
    }

    private func makeFrame(
        request: FocusedWindowCaptureRequest
    ) -> FocusedWindowSurfaceFrame? {
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            16,
            16,
            kCVPixelFormatType_32BGRA,
            nil,
            &buffer
        ) == kCVReturnSuccess, let buffer else { return nil }
        return FocusedWindowSurfaceFrame(pixelBuffer: buffer, request: request)
    }
}

private final class FocusedWindowFrameRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var frames: [FocusedWindowSurfaceFrame] = []

    var revisions: [UInt64] {
        lock.lock()
        defer { lock.unlock() }
        return frames.map(\.request.privacyContext.revision)
    }

    func record(_ frame: FocusedWindowSurfaceFrame) {
        lock.lock()
        frames.append(frame)
        lock.unlock()
    }
}

private final class FocusedWindowFailureRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var requests: [FocusedWindowCaptureRequest] = []

    var revisions: [UInt64] {
        lock.lock()
        defer { lock.unlock() }
        return requests.map(\.privacyContext.revision)
    }

    func record(_ request: FocusedWindowCaptureRequest) {
        lock.lock()
        requests.append(request)
        lock.unlock()
    }
}
