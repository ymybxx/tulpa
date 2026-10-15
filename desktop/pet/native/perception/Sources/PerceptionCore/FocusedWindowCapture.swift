import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit

struct FocusedWindowDescriptor: Equatable, Sendable {
    let windowID: CGWindowID
    let processIdentifier: Int32
    let bundleIdentifier: String?
    let title: String?
    let frame: CGRect
    let isOnScreen: Bool
}

struct FocusedWindowDisplayDescriptor: Equatable, Sendable {
    let displayID: CGDirectDisplayID
    let frame: CGRect
}

/// Validates that ScreenCaptureKit and Core Graphics identify the same physical
/// display geometry before any focused-window metadata or pixels are accepted.
enum FocusedWindowDisplayPolicy {
    private static let geometryTolerance: CGFloat = 0.5

    static func trustedDisplayFrame(
        requestedDisplayID: CGDirectDisplayID,
        displays: [FocusedWindowDisplayDescriptor],
        coreGraphicsFrame: CGRect
    ) -> CGRect? {
        let matches = displays.filter { $0.displayID == requestedDisplayID }
        guard matches.count == 1 else { return nil }
        let screenCaptureFrame = matches[0].frame
        guard isFinitePositive(screenCaptureFrame),
              isFinitePositive(coreGraphicsFrame),
              approximatelyEqual(screenCaptureFrame, coreGraphicsFrame)
        else { return nil }
        return screenCaptureFrame
    }

    static func contains(_ windowFrame: CGRect, in displayFrame: CGRect) -> Bool {
        isFinitePositive(windowFrame) && isFinitePositive(displayFrame)
            && windowFrame.minX >= displayFrame.minX
            && windowFrame.minY >= displayFrame.minY
            && windowFrame.maxX <= displayFrame.maxX
            && windowFrame.maxY <= displayFrame.maxY
    }

    private static func approximatelyEqual(_ first: CGRect, _ second: CGRect) -> Bool {
        abs(first.minX - second.minX) <= geometryTolerance
            && abs(first.minY - second.minY) <= geometryTolerance
            && abs(first.width - second.width) <= geometryTolerance
            && abs(first.height - second.height) <= geometryTolerance
    }

    private static func isFinitePositive(_ frame: CGRect) -> Bool {
        frame.origin.x.isFinite && frame.origin.y.isFinite
            && frame.width.isFinite && frame.height.isFinite
            && frame.width > 0 && frame.height > 0
    }
}

/// A pure, fail-closed join between the AX focused-window identity and the
/// refreshed ScreenCaptureKit catalog. A non-unique match is never guessed.
enum FocusedWindowMatcher {
    static let frameTolerance: CGFloat = 4

    static func uniqueWindowID(
        for context: ScreenTextApplicationContext,
        displayFrame: CGRect,
        in candidates: [FocusedWindowDescriptor]
    ) -> CGWindowID? {
        let matches = candidates.filter { candidate in
            candidate.isOnScreen
                && candidate.processIdentifier == context.processIdentifier
                && normalizedBundleIdentifier(candidate.bundleIdentifier)
                    == normalizedBundleIdentifier(context.appBundleId)
                && normalizedWindowTitle(candidate.title)
                    == normalizedWindowTitle(context.windowTitle)
                && approximatelyEqual(candidate.frame, context.windowFrame.rect)
                && FocusedWindowDisplayPolicy.contains(
                    candidate.frame,
                    in: displayFrame
                )
        }
        return matches.count == 1 ? matches[0].windowID : nil
    }

    private static func approximatelyEqual(_ first: CGRect, _ second: CGRect) -> Bool {
        isFinitePositive(first) && isFinitePositive(second)
            && abs(first.minX - second.minX) <= frameTolerance
            && abs(first.minY - second.minY) <= frameTolerance
            && abs(first.width - second.width) <= frameTolerance
            && abs(first.height - second.height) <= frameTolerance
    }

    private static func isFinitePositive(_ frame: CGRect) -> Bool {
        frame.origin.x.isFinite && frame.origin.y.isFinite
            && frame.width.isFinite && frame.height.isFinite
            && frame.width > 0 && frame.height > 0
    }

    private static func normalizedBundleIdentifier(_ value: String?) -> String? {
        guard let value else { return nil }
        let normalized = value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalized.isEmpty ? nil : normalized
    }

    private static func normalizedWindowTitle(_ value: String?) -> String? {
        guard let value else { return nil }
        let normalized = value
            .precomposedStringWithCanonicalMapping
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        guard !normalized.isEmpty else { return nil }
        return String(normalized.prefix(256))
    }
}

struct FocusedWindowCaptureRequest: Equatable, Sendable {
    let generation: UInt64
    let privacyContext: ScreenTextPrivacyContext
    let applicationContext: ScreenTextApplicationContext
}

struct FocusedWindowSurfaceFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let request: FocusedWindowCaptureRequest
}

/// Latest-only scheduler around the independent-window capture operation. It
/// permits one native capture at a time and retains at most one pending request.
final class FocusedWindowCaptureCoordinator: @unchecked Sendable {
    typealias Capture = @Sendable (
        FocusedWindowCaptureRequest
    ) async -> FocusedWindowSurfaceFrame?
    typealias FrameHandler = @Sendable (FocusedWindowSurfaceFrame) -> Void
    typealias FailureHandler = @Sendable (FocusedWindowCaptureRequest) -> Void

    private let capture: Capture
    private let onFrame: FrameHandler
    private let onFailure: FailureHandler
    private let lock = NSLock()
    private var activeGeneration: UInt64?
    private var activeRunID: UInt64 = 0
    private var activeTask: Task<Void, Never>?
    private var pendingRequest: FocusedWindowCaptureRequest?

    init(
        capture: @escaping Capture,
        onFrame: @escaping FrameHandler,
        onFailure: @escaping FailureHandler = { _ in }
    ) {
        self.capture = capture
        self.onFrame = onFrame
        self.onFailure = onFailure
    }

    func activate(generation: UInt64) {
        lock.lock()
        activeRunID &+= 1
        let previous = activeTask
        activeTask = nil
        pendingRequest = nil
        activeGeneration = generation
        lock.unlock()
        previous?.cancel()
    }

    @discardableResult
    func submit(_ request: FocusedWindowCaptureRequest) -> Bool {
        lock.lock()
        guard activeGeneration == request.generation else {
            lock.unlock()
            return false
        }
        guard activeTask == nil else {
            pendingRequest = request
            lock.unlock()
            return true
        }
        startLocked(request)
        lock.unlock()
        return true
    }

    /// After this method returns, the stopped generation cannot deliver a frame.
    func stop(generation: UInt64? = nil) {
        lock.lock()
        if let generation, activeGeneration != generation {
            lock.unlock()
            return
        }
        activeRunID &+= 1
        activeGeneration = nil
        pendingRequest = nil
        let previous = activeTask
        activeTask = nil
        lock.unlock()
        previous?.cancel()
    }

    private func startLocked(_ request: FocusedWindowCaptureRequest) {
        activeRunID &+= 1
        let runID = activeRunID
        let capture = self.capture
        activeTask = Task { [weak self] in
            let frame = await capture(request)
            self?.complete(runID: runID, request: request, frame: frame)
        }
    }

    private func complete(
        runID: UInt64,
        request: FocusedWindowCaptureRequest,
        frame: FocusedWindowSurfaceFrame?
    ) {
        lock.lock()
        guard activeRunID == runID,
              activeGeneration == request.generation
        else {
            lock.unlock()
            return
        }
        activeTask = nil
        if let frame, frame.request == request {
            // Keep the callback under the lifecycle lock. stop() therefore
            // cannot return before a currently accepted callback completes.
            onFrame(frame)
        } else {
            onFailure(request)
        }
        let next = pendingRequest
        pendingRequest = nil
        if let next, next.generation == activeGeneration {
            startLocked(next)
        }
        lock.unlock()
    }
}

@available(macOS 12.3, *)
final class NativeFocusedWindowSurfaceCapturer: @unchecked Sendable {
    typealias AuthorizationValidator = @Sendable (
        ScreenTextPrivacyContext,
        ScreenTextApplicationContext
    ) -> Bool

    private let timeout: TimeInterval
    private let maximumLongEdge: Int
    private let isStillAuthorized: AuthorizationValidator

    init(
        timeout: TimeInterval = 1,
        maximumLongEdge: Int = 1_280,
        isStillAuthorized: @escaping AuthorizationValidator
    ) {
        precondition(timeout > 0 && timeout <= 1 && timeout.isFinite)
        precondition(maximumLongEdge > 0)
        self.timeout = timeout
        self.maximumLongEdge = maximumLongEdge
        self.isStillAuthorized = isStillAuthorized
    }

    func capture(
        _ request: FocusedWindowCaptureRequest
    ) async -> FocusedWindowSurfaceFrame? {
        let attempt = FocusedWindowCaptureAttempt(
            request: request,
            timeout: timeout,
            maximumLongEdge: maximumLongEdge,
            isStillAuthorized: isStillAuthorized
        )
        return await withTaskCancellationHandler {
            await attempt.capture()
        } onCancel: {
            attempt.cancel()
        }
    }
}

@available(macOS 12.3, *)
private final class FocusedWindowCaptureAttempt: NSObject,
    SCStreamOutput,
    SCStreamDelegate,
    @unchecked Sendable
{
    private typealias Continuation = CheckedContinuation<FocusedWindowSurfaceFrame?, Never>

    private let request: FocusedWindowCaptureRequest
    private let timeout: TimeInterval
    private let maximumLongEdge: Int
    private let isStillAuthorized: NativeFocusedWindowSurfaceCapturer.AuthorizationValidator
    private let opaqueBackgroundColor = CGColor(
        red: 0,
        green: 0,
        blue: 0,
        alpha: 1
    )
    private let sampleQueue = DispatchQueue(
        label: "app.fishmemory.perception.focused-window.sample",
        qos: .userInitiated
    )
    private let timeoutQueue = DispatchQueue(
        label: "app.fishmemory.perception.focused-window.timeout",
        qos: .utility
    )
    private let lock = NSLock()
    private var continuation: Continuation?
    private var timeoutWorkItem: DispatchWorkItem?
    private var stream: SCStream?
    private var completed = false

    init(
        request: FocusedWindowCaptureRequest,
        timeout: TimeInterval,
        maximumLongEdge: Int,
        isStillAuthorized: @escaping NativeFocusedWindowSurfaceCapturer.AuthorizationValidator
    ) {
        self.request = request
        self.timeout = timeout
        self.maximumLongEdge = maximumLongEdge
        self.isStillAuthorized = isStillAuthorized
        super.init()
    }

    func capture() async -> FocusedWindowSurfaceFrame? {
        await withCheckedContinuation { continuation in
            lock.lock()
            guard !completed else {
                lock.unlock()
                continuation.resume(returning: nil)
                return
            }
            self.continuation = continuation
            let timeoutWorkItem = DispatchWorkItem { [weak self] in
                self?.finish(nil)
            }
            self.timeoutWorkItem = timeoutWorkItem
            lock.unlock()

            timeoutQueue.asyncAfter(
                deadline: .now() + timeout,
                execute: timeoutWorkItem
            )
            refreshShareableWindows()
        }
    }

    func cancel() {
        finish(nil)
    }

    private func refreshShareableWindows() {
        guard isStillAuthorized(
            request.privacyContext,
            request.applicationContext
        ) else {
            finish(nil)
            return
        }
        SCShareableContent.getExcludingDesktopWindows(
            false,
            onScreenWindowsOnly: true
        ) { [weak self] content, _ in
            self?.accept(content)
        }
    }

    private func accept(_ content: SCShareableContent?) {
        guard !isCompleted(),
              let content,
              isStillAuthorized(
                  request.privacyContext,
                  request.applicationContext
              )
        else {
            finish(nil)
            return
        }
        let displayDescriptors = content.displays.map {
            FocusedWindowDisplayDescriptor(
                displayID: $0.displayID,
                frame: $0.frame
            )
        }
        guard let displayFrame = FocusedWindowDisplayPolicy.trustedDisplayFrame(
            requestedDisplayID: request.privacyContext.displayID,
            displays: displayDescriptors,
            coreGraphicsFrame: CGDisplayBounds(request.privacyContext.displayID)
        ), FocusedWindowDisplayPolicy.contains(
            request.applicationContext.windowFrame.rect,
            in: displayFrame
        ) else {
            finish(nil)
            return
        }
        // Read titles and frames only for windows already owned by the exact
        // authorized process/bundle. Unrelated app metadata never enters even
        // the short-lived matcher array.
        let matchingApplicationWindows = content.windows.filter { window in
            guard window.isOnScreen,
                  let application = window.owningApplication,
                  application.processID
                    == request.applicationContext.processIdentifier,
                  application.bundleIdentifier.caseInsensitiveCompare(
                    request.applicationContext.appBundleId
                  ) == .orderedSame
            else { return false }
            return true
        }
        let descriptors = matchingApplicationWindows.compactMap(Self.descriptor)
        guard let windowID = FocusedWindowMatcher.uniqueWindowID(
            for: request.applicationContext,
            displayFrame: displayFrame,
            in: descriptors
        ), let window = matchingApplicationWindows.first(where: {
            $0.windowID == windowID
        })
        else {
            finish(nil)
            return
        }
        startStream(window: window)
    }

    private func startStream(window: SCWindow) {
        guard !isCompleted(),
              isStillAuthorized(
                  request.privacyContext,
                  request.applicationContext
              )
        else {
            finish(nil)
            return
        }
        let configuration = SCStreamConfiguration()
        let size = Self.outputSize(
            frame: window.frame,
            maximumLongEdge: maximumLongEdge
        )
        guard size.width > 0, size.height > 0 else {
            finish(nil)
            return
        }
        configuration.width = size.width
        configuration.height = size.height
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 30)
        configuration.queueDepth = 1
        configuration.pixelFormat = kCVPixelFormatType_32BGRA
        configuration.scalesToFit = true
        configuration.showsCursor = false
        configuration.backgroundColor = opaqueBackgroundColor
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let stream = SCStream(
            filter: filter,
            configuration: configuration,
            delegate: self
        )
        do {
            try stream.addStreamOutput(
                self,
                type: .screen,
                sampleHandlerQueue: sampleQueue
            )
        } catch {
            finish(nil)
            return
        }

        lock.lock()
        guard !completed else {
            lock.unlock()
            try? stream.removeStreamOutput(self, type: .screen)
            return
        }
        self.stream = stream
        lock.unlock()
        stream.startCapture { [weak self] error in
            if error != nil { self?.finish(nil) }
        }
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .screen,
              sampleBuffer.isValid,
              let attachments = focusedWindowFrameAttachments(sampleBuffer),
              let rawStatus = attachments[.status] as? Int,
              SCFrameStatus(rawValue: rawStatus) == .complete,
              let pixelBuffer = sampleBuffer.imageBuffer,
              CVPixelBufferGetWidth(pixelBuffer) > 0,
              CVPixelBufferGetHeight(pixelBuffer) > 0,
              isStillAuthorized(
                  request.privacyContext,
                  request.applicationContext
              )
        else { return }
        finish(FocusedWindowSurfaceFrame(
            pixelBuffer: pixelBuffer,
            request: request
        ))
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        _ = stream
        _ = error
        finish(nil)
    }

    private func finish(_ frame: FocusedWindowSurfaceFrame?) {
        lock.lock()
        guard !completed else {
            lock.unlock()
            return
        }
        completed = true
        let continuation = self.continuation
        self.continuation = nil
        let timeoutWorkItem = self.timeoutWorkItem
        self.timeoutWorkItem = nil
        let stream = self.stream
        self.stream = nil
        lock.unlock()

        timeoutWorkItem?.cancel()
        if let stream {
            stream.stopCapture { [weak self] _ in
                guard let self else { return }
                try? stream.removeStreamOutput(self, type: .screen)
            }
        }
        continuation?.resume(returning: frame)
    }

    private func isCompleted() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return completed
    }

    private static func descriptor(_ window: SCWindow) -> FocusedWindowDescriptor? {
        guard let application = window.owningApplication else { return nil }
        return FocusedWindowDescriptor(
            windowID: window.windowID,
            processIdentifier: application.processID,
            bundleIdentifier: application.bundleIdentifier,
            title: window.title,
            frame: window.frame,
            isOnScreen: window.isOnScreen
        )
    }

    private static func outputSize(
        frame: CGRect,
        maximumLongEdge: Int
    ) -> (width: Int, height: Int) {
        guard frame.width.isFinite, frame.height.isFinite,
              frame.width > 0, frame.height > 0
        else { return (0, 0) }
        let longEdge = max(frame.width, frame.height)
        let scale = min(1, CGFloat(maximumLongEdge) / longEdge)
        return (
            max(1, Int((frame.width * scale).rounded())),
            max(1, Int((frame.height * scale).rounded()))
        )
    }
}

@available(macOS 12.3, *)
private func focusedWindowFrameAttachments(
    _ sampleBuffer: CMSampleBuffer
) -> [SCStreamFrameInfo: Any]? {
    guard let values = CMSampleBufferGetSampleAttachmentsArray(
        sampleBuffer,
        createIfNecessary: false
    ) as? [[SCStreamFrameInfo: Any]] else { return nil }
    return values.first
}
