import AppKit
import Darwin
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit

public enum ScreenMonitorState: String, Equatable, Sendable {
    case stopped
    case starting
    case running
    case unavailable
    case failed
}

public struct ScreenMonitorStatus: Equatable, Sendable {
    public let state: ScreenMonitorState
    public let displayID: CGDirectDisplayID?
    public let reason: String?

    public init(
        state: ScreenMonitorState,
        displayID: CGDirectDisplayID? = nil,
        reason: String? = nil
    ) {
        self.state = state
        self.displayID = displayID
        self.reason = reason
    }
}

public struct ScreenChangeObservation: Codable, Equatable, Sendable {
    public let source: String
    public let kind: String
    public let revision: UInt64
    public let observedAt: Double
    public let displayID: CGDirectDisplayID
    public let frameWidth: Int
    public let frameHeight: Int
    public let evaluation: ScreenChangeEvaluation

    public init(
        displayID: CGDirectDisplayID,
        frameWidth: Int,
        frameHeight: Int,
        evaluation: ScreenChangeEvaluation
    ) {
        source = "screen"
        kind = "screen_change"
        revision = evaluation.revision
        observedAt = Date().timeIntervalSince1970
        self.displayID = displayID
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.evaluation = evaluation
    }
}

/// ScreenCaptureKit owner for one physical display. Modern environment capture
/// uses bounded periodic screenshots; only legacy analysis or older macOS
/// requires a continuous display stream. OCR and sparse VLM receive pixels only
/// from the separately refreshed focused-window stream.
@available(macOS 12.3, *)
public final class ScreenMonitor: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    public typealias ObservationHandler = @Sendable (ScreenChangeObservation) -> Void
    public typealias TextObservationHandler = @Sendable (ScreenTextObservation) -> Void
    public typealias StatusHandler = @MainActor @Sendable (ScreenMonitorStatus) -> Void

    private let requestedDisplayID: CGDirectDisplayID?
    private let captureQueue = DispatchQueue(
        label: "app.fishmemory.perception.screen.capture",
        qos: .userInitiated
    )
    private let callbackQueue = DispatchQueue(
        label: "app.fishmemory.perception.screen.callback",
        qos: .utility
    )
    private let lifecycleLock = NSLock()
    private let onObservation: ObservationHandler
    private let onStatus: StatusHandler
    private let screenTextPrivacyGate: ScreenTextPrivacyGate
    private let textTriggerController: ScreenTextTriggerController
    private let textRecognizer: ScreenTextRecognizer
    private let visionCandidateEncoder: VisionCandidateEncoder
    private let environmentFrameEncoder: EnvironmentFrameEncoder
    private let focusedWindowCapture: FocusedWindowCaptureCoordinator
    private let hasScreenRecordingPermission: @Sendable () -> Bool
    private var excludeHostApplicationFromCapture: Bool
    private let excludedApplicationBundleIdentifiers: Set<String>
    private let parentProcessIdentifier: Int32?
    private var detector: ScreenChangeDetector
    private var pixelChangeAnalyzer: ScreenPixelChangeAnalyzer
    private var stream: SCStream?
    private var captureActive = false
    private var activeDisplayID: CGDirectDisplayID?
    private var environmentCaptureFilter: SCContentFilter?
    private var environmentCaptureConfiguration: SCStreamConfiguration?
    private var environmentCaptureSuspended = false
    private var environmentFrameSampler: EnvironmentFrameSamplingLoop!
    private var captureGeneration: UInt64 = 0
    private var exclusionRevision: UInt64 = 0
    private var exclusionUpdatePending = false
    private var exclusionUpdateTask: Task<Void, Never>?
    private var enhancedVisionEnabled: Bool
    private var legacyScreenAnalysisEnabled: Bool
    private var callbackPending = false
    private var latestObservation: ScreenChangeObservation?

    public init(
        displayID: CGDirectDisplayID? = nil,
        detectorConfiguration: ScreenChangeDetectorConfiguration = .default,
        hasScreenRecordingPermission: @escaping @Sendable () -> Bool = {
            CGPreflightScreenCaptureAccess()
        },
        excludedApplicationBundleIdentifiers: Set<String>? = nil,
        parentProcessIdentifier: Int32? = getppid(),
        excludeHostApplicationFromCapture: Bool = true,
        onObservation: @escaping ObservationHandler,
        screenTextConfiguration: ScreenTextRecognitionConfiguration = .default,
        screenTextPrivacyGate: @escaping ScreenTextPrivacyGate = ScreenTextPrivacyGates.denyAll,
        screenTextAuthorizationValidator: ScreenTextAuthorizationValidator? = nil,
        onTextObservation: @escaping TextObservationHandler = { _ in },
        enhancedVisionEnabled: Bool = false,
        legacyScreenAnalysisEnabled: Bool = true,
        visionCandidateBroker: VisionCandidateBroker = .init(),
        onVisionCandidateAvailable: @escaping @Sendable (
            VisionCandidateMetadata
        ) -> Void = { _ in },
        environmentFrameBroker: EnvironmentFrameBroker = .init(),
        onEnvironmentFrameAvailable: @escaping @Sendable (
            EnvironmentFrameMetadata
        ) -> Void = { _ in },
        onStatus: @escaping StatusHandler = { _ in }
    ) {
        requestedDisplayID = displayID
        detector = ScreenChangeDetector(configuration: detectorConfiguration)
        pixelChangeAnalyzer = ScreenPixelChangeAnalyzer()
        let triggerController = ScreenTextTriggerController(
            minimumRecognitionInterval: screenTextConfiguration.minimumRecognitionInterval
        )
        textTriggerController = triggerController
        self.hasScreenRecordingPermission = hasScreenRecordingPermission
        self.excludeHostApplicationFromCapture = excludeHostApplicationFromCapture
        self.excludedApplicationBundleIdentifiers =
            excludedApplicationBundleIdentifiers ?? Self.defaultExcludedBundleIdentifiers
        self.parentProcessIdentifier = parentProcessIdentifier
        self.onObservation = onObservation
        self.screenTextPrivacyGate = screenTextPrivacyGate
        self.enhancedVisionEnabled = enhancedVisionEnabled
        self.legacyScreenAnalysisEnabled = legacyScreenAnalysisEnabled
        let authorizationValidator = screenTextAuthorizationValidator ?? {
            candidate,
            expectedContext in
            screenTextPrivacyGate(candidate) == expectedContext
        }
        let candidateEncoder = VisionCandidateEncoder(
            broker: visionCandidateBroker,
            isStillAuthorized: authorizationValidator,
            onAvailable: onVisionCandidateAvailable
        )
        visionCandidateEncoder = candidateEncoder
        environmentFrameEncoder = EnvironmentFrameEncoder(
            broker: environmentFrameBroker,
            onAvailable: onEnvironmentFrameAvailable
        )
        let recognizer = ScreenTextRecognizer(
            configuration: screenTextConfiguration,
            isStillAuthorized: authorizationValidator,
            onAnalysis: { analysis in
                guard ScreenTextCandidatePolicy.allowsVisionCandidate(analysis) else {
                    return
                }
                candidateEncoder.submit(
                    pixelBuffer: analysis.pixelBuffer,
                    regionOfInterest: analysis.visionRegionOfInterest,
                    generation: analysis.generation,
                    privacyContext: analysis.context,
                    applicationContext: analysis.applicationContext
                )
            },
            onCompletion: { _, _, succeeded in
                triggerController.completeAttempt(succeeded: succeeded)
            },
            onObservation: onTextObservation
        )
        textRecognizer = recognizer
        let surfaceCapturer = NativeFocusedWindowSurfaceCapturer(
            isStillAuthorized: authorizationValidator
        )
        focusedWindowCapture = FocusedWindowCaptureCoordinator(
            capture: { request in
                await surfaceCapturer.capture(request)
            },
            onFrame: { frame in
                recognizer.submit(
                    pixelBuffer: frame.pixelBuffer,
                    generation: frame.request.generation,
                    context: frame.request.privacyContext,
                    applicationContext: frame.request.applicationContext,
                    visionRegionOfInterest: CGRect(x: 0, y: 0, width: 1, height: 1)
                )
            },
            onFailure: { _ in
                triggerController.completeAttempt(succeeded: false)
            }
        )
        self.onStatus = onStatus
        super.init()
        environmentFrameSampler = EnvironmentFrameSamplingLoop(
            permitted: { [weak self] in self?.canCaptureEnvironmentFrame() == true },
            capture: { [weak self] generation in
                await self?.captureEnvironmentFrame(generation: generation)
            },
            consume: { [environmentFrameEncoder] frame, generation, isCurrent in
                await environmentFrameEncoder.submitAndWait(
                    frame, generation: generation, isStillCurrent: isCurrent
                )
            },
            onInvalidate: { [environmentFrameEncoder] generation in
                environmentFrameEncoder.invalidate(generation: generation)
            }
        )
        let center = NSWorkspace.shared.notificationCenter
        for name in [NSWorkspace.screensDidSleepNotification,
                     NSWorkspace.sessionDidResignActiveNotification] {
            center.addObserver(self, selector: #selector(suspendEnvironmentCapture),
                               name: name, object: nil)
        }
        for name in [NSWorkspace.screensDidWakeNotification,
                     NSWorkspace.sessionDidBecomeActiveNotification] {
            center.addObserver(self, selector: #selector(resumeEnvironmentCapture),
                               name: name, object: nil)
        }
        DistributedNotificationCenter.default().addObserver(
            self, selector: #selector(suspendEnvironmentCapture),
            name: NSNotification.Name("com.apple.screenIsLocked"), object: nil
        )
        DistributedNotificationCenter.default().addObserver(
            self, selector: #selector(resumeEnvironmentCapture),
            name: NSNotification.Name("com.apple.screenIsUnlocked"), object: nil
        )
    }

    public static func requestPermission() -> Bool {
        CGPreflightScreenCaptureAccess() || CGRequestScreenCaptureAccess()
    }

    public func setEnhancedVisionEnabled(_ enabled: Bool) {
        let (generation, active) = withLifecycleLock {
            enhancedVisionEnabled = enabled
            return (
                captureActive ? captureGeneration : nil,
                enabled && legacyScreenAnalysisEnabled
            )
        }
        visionCandidateEncoder.setEnabled(active, generation: generation)
    }

    public func setEnvironmentFrameInterval(
        milliseconds: Int,
        tickImmediately: Bool
    ) {
        _ = environmentFrameSampler.updateInterval(
            milliseconds: milliseconds,
            tickImmediately: tickImmediately
        )
    }

    public func setLegacyScreenAnalysisEnabled(_ enabled: Bool) {
        let state: (UInt64, Bool)? = withLifecycleLock {
            guard legacyScreenAnalysisEnabled != enabled else { return nil }
            legacyScreenAnalysisEnabled = enabled
            detector.reset()
            pixelChangeAnalyzer.reset()
            textTriggerController.reset()
            latestObservation = nil
            guard captureActive else { return nil }
            return (captureGeneration, enhancedVisionEnabled)
        }
        guard let (generation, enhancedVision) = state else { return }
        if enabled {
            environmentFrameSampler.invalidate(generation: generation)
            environmentFrameEncoder.activate(generation: generation)
            focusedWindowCapture.activate(generation: generation)
            textRecognizer.activate(generation: generation)
            visionCandidateEncoder.setEnabled(
                enhancedVision,
                generation: generation
            )
        } else {
            focusedWindowCapture.stop(generation: generation)
            textRecognizer.invalidate(generation: generation)
            visionCandidateEncoder.setEnabled(false, generation: generation)
            startEnvironmentSampling(generation: generation)
        }
    }

    public func setExcludeHostApplicationFromCapture(_ enabled: Bool) {
        let generation = withLifecycleLock { () -> UInt64? in
            guard excludeHostApplicationFromCapture != enabled else { return nil }
            excludeHostApplicationFromCapture = enabled
            exclusionRevision &+= 1
            guard captureActive else { return nil }
            exclusionUpdatePending = true
            return captureGeneration
        }
        guard let generation else { return }
        environmentFrameSampler.invalidate(generation: generation)
        environmentFrameEncoder.invalidate(generation: generation)
        scheduleExclusionUpdate(generation: generation)
    }

    private func scheduleExclusionUpdate(generation: UInt64) {
        let previous = withLifecycleLock { exclusionUpdateTask }
        let task = Task { [weak self] in
            await previous?.value
            await self?.applyExclusionUpdate(generation: generation)
        }
        withLifecycleLock { exclusionUpdateTask = task }
    }

    private func applyExclusionUpdate(generation: UInt64) async {
        let target = withLifecycleLock { () -> (UInt64, CGDirectDisplayID, Bool, SCStream?)? in
            guard captureActive, captureGeneration == generation,
                  exclusionUpdatePending, let activeDisplayID else { return nil }
            return (exclusionRevision, activeDisplayID,
                    excludeHostApplicationFromCapture, stream)
        }
        guard let (revision, displayID, excludeHost, stream) = target else { return }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true
            )
            guard let display = content.displays.first(where: { $0.displayID == displayID }) else {
                throw ScreenCaptureFilterError.displayNotFound
            }
            let filter = makeCaptureFilter(
                display: display,
                applications: content.applications,
                excludeHost: excludeHost
            )
            guard withLifecycleLock({
                captureActive && captureGeneration == generation
                    && exclusionRevision == revision
            }) else { return }
            if let stream { try await stream.updateContentFilter(filter) }
            let applied = withLifecycleLock { () -> Bool in
                guard captureActive, captureGeneration == generation,
                      exclusionRevision == revision else { return false }
                environmentCaptureFilter = filter
                exclusionUpdatePending = false
                return true
            }
            if applied {
                environmentFrameEncoder.activate(generation: generation)
                startEnvironmentSampling(generation: generation)
            }
        } catch {
            let current = withLifecycleLock {
                captureActive && captureGeneration == generation
                    && exclusionRevision == revision
            }
            guard current else { return }
            await stop()
            _ = publishStatus(.init(
                state: .failed,
                reason: "screen_capture_filter_update_failed: \(error.localizedDescription)"
            ))
        }
    }

    private func makeCaptureFilter(
        display: SCDisplay,
        applications: [SCRunningApplication],
        excludeHost: Bool
    ) -> SCContentFilter {
        let excludedApplications = applications.filter { application in
            shouldExcludeScreenApplication(
                bundleIdentifier: application.bundleIdentifier,
                processIdentifier: application.processID,
                parentProcessIdentifier: parentProcessIdentifier,
                mainBundleIdentifier: Bundle.main.bundleIdentifier,
                excludedBundleIdentifiers: excludedApplicationBundleIdentifiers,
                excludeHostApplication: excludeHost
            )
        }
        return SCContentFilter(
            display: display,
            excludingApplications: excludedApplications,
            exceptingWindows: []
        )
    }

    /// Starts without prompting. Permission UI remains an Electron/product
    /// decision; an untrusted helper reports unavailable immediately.
    @discardableResult
    public func start() async -> ScreenMonitorStatus {
        if let runningDisplayID = withLifecycleLock({
            captureActive ? activeDisplayID : nil
        }) {
            return publishStatus(.init(state: .running, displayID: runningDisplayID))
        }
        guard hasScreenRecordingPermission() else {
            return publishStatus(.init(
                state: .unavailable,
                reason: "screen_recording_permission_required"
            ))
        }
        publishStatus(.init(state: .starting))
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false,
                onScreenWindowsOnly: true
            )
            let targetID = requestedDisplayID ?? CGMainDisplayID()
            guard let display = content.displays.first(where: { $0.displayID == targetID }) else {
                return publishStatus(.init(
                    state: .unavailable,
                    displayID: targetID,
                    reason: "display_not_found"
                ))
            }

            let (shouldRunLegacyAnalysis, excludeHost, filterRevision) = withLifecycleLock {
                (legacyScreenAnalysisEnabled, excludeHostApplicationFromCapture,
                 exclusionRevision)
            }
            let configuration = makeScreenStreamConfiguration(
                displayWidth: display.width,
                displayHeight: display.height,
                legacyScreenAnalysis: shouldRunLegacyAnalysis
            )

            let filter = makeCaptureFilter(
                display: display,
                applications: content.applications,
                excludeHost: excludeHost
            )
            let screenshotConfiguration = makeEnvironmentScreenshotConfiguration(
                displayWidth: display.width,
                displayHeight: display.height
            )
            let screenshotManagerAvailable: Bool
            if #available(macOS 14.0, *) {
                screenshotManagerAvailable = true
            } else {
                screenshotManagerAvailable = false
            }
            let needsPersistentScreenStream = screenCaptureNeedsPersistentStream(
                legacyScreenAnalysis: shouldRunLegacyAnalysis,
                screenshotManagerAvailable: screenshotManagerAvailable
            )
            let nextStream: SCStream?
            if needsPersistentScreenStream {
                let candidate = SCStream(
                    filter: filter,
                    configuration: configuration,
                    delegate: self
                )
                try candidate.addStreamOutput(
                    self,
                    type: .screen,
                    sampleHandlerQueue: captureQueue
                )
                nextStream = candidate
            } else {
                nextStream = nil
            }
            let generation = withLifecycleLock {
                captureGeneration &+= 1
                captureActive = true
                stream = nextStream
                activeDisplayID = display.displayID
                environmentCaptureFilter = filter
                exclusionUpdatePending = exclusionRevision != filterRevision
                environmentCaptureConfiguration = screenshotConfiguration
                detector.reset()
                pixelChangeAnalyzer.reset()
                textTriggerController.reset()
                visionCandidateEncoder.activate(
                    generation: captureGeneration,
                    enabled: enhancedVisionEnabled
                        && legacyScreenAnalysisEnabled
                )
                environmentFrameEncoder.activate(generation: captureGeneration)
                return captureGeneration
            }
            if shouldRunLegacyAnalysis {
                focusedWindowCapture.activate(generation: generation)
                textRecognizer.activate(generation: generation)
            }
            if let nextStream { try await nextStream.startCapture() }
            if !shouldRunLegacyAnalysis { startEnvironmentSampling(generation: generation) }
            if withLifecycleLock({ exclusionUpdatePending }) {
                environmentFrameSampler.invalidate(generation: generation)
                environmentFrameEncoder.invalidate(generation: generation)
                scheduleExclusionUpdate(generation: generation)
            }
            return publishStatus(.init(state: .running, displayID: display.displayID))
        } catch {
            let stoppedGeneration = withLifecycleLock {
                let generation = captureGeneration
                captureGeneration &+= 1
                captureActive = false
                stream = nil
                activeDisplayID = nil
                environmentCaptureFilter = nil
                environmentCaptureConfiguration = nil
                exclusionUpdatePending = false
                pixelChangeAnalyzer.reset()
                textTriggerController.reset()
                return generation
            }
            focusedWindowCapture.stop(generation: stoppedGeneration)
            textRecognizer.invalidate(generation: stoppedGeneration)
            visionCandidateEncoder.invalidate(generation: stoppedGeneration)
            environmentFrameEncoder.invalidate(generation: stoppedGeneration)
            environmentFrameSampler.invalidate(generation: stoppedGeneration)
            return publishStatus(.init(
                state: .failed,
                reason: "screen_capture_start_failed: \(error.localizedDescription)"
            ))
        }
    }

    public func stop() async {
        let (current, stoppedGeneration) = withLifecycleLock {
            let value = stream
            captureActive = false
            stream = nil
            activeDisplayID = nil
            environmentCaptureFilter = nil
            environmentCaptureConfiguration = nil
            exclusionUpdatePending = false
            pixelChangeAnalyzer.reset()
            let generation = captureGeneration
            captureGeneration &+= 1
            textTriggerController.reset()
            return (value, generation)
        }
        focusedWindowCapture.stop(generation: stoppedGeneration)
        textRecognizer.invalidate(generation: stoppedGeneration)
        visionCandidateEncoder.invalidate(generation: stoppedGeneration)
        environmentFrameEncoder.invalidate(generation: stoppedGeneration)
        environmentFrameSampler.invalidate(generation: stoppedGeneration)
        guard let current else {
            _ = publishStatus(.init(state: .stopped))
            return
        }
        do {
            try await current.stopCapture()
            try current.removeStreamOutput(self, type: .screen)
        } catch {
            _ = publishStatus(.init(
                state: .failed,
                reason: "screen_capture_stop_failed: \(error.localizedDescription)"
            ))
            return
        }
        _ = publishStatus(.init(state: .stopped))
    }

    public func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .screen, sampleBuffer.isValid else { return }
        guard let attachments = frameAttachments(sampleBuffer) else { return }
        guard let rawStatus = attachments[.status] as? Int,
              SCFrameStatus(rawValue: rawStatus) == .complete,
              let pixelBuffer = sampleBuffer.imageBuffer
        else { return }

        let frameContext: (CGDirectDisplayID, UInt64)? = withLifecycleLock {
            // In environment-only mode the periodic screenshot owns frame
            // delivery, including static displays where this stream is idle.
            guard self.stream === stream, legacyScreenAnalysisEnabled,
                  !exclusionUpdatePending,
                  !environmentCaptureSuspended, let displayID = activeDisplayID else {
                return nil
            }
            return (displayID, captureGeneration)
        }
        if let (displayID, generation) = frameContext {
            environmentFrameEncoder.submit(
                pixelBuffer: pixelBuffer,
                generation: generation,
                observedAt: Date().timeIntervalSince1970,
                displayID: displayID
            )
        }
        let shouldRunLegacyAnalysis = withLifecycleLock {
            self.stream === stream && legacyScreenAnalysisEnabled
                && !exclusionUpdatePending
        }
        guard shouldRunLegacyAnalysis else { return }
        let timestamp = CMTimeGetSeconds(sampleBuffer.presentationTimeStamp)
        let analysisTimestamp = timestamp.isFinite
            ? timestamp
            : ProcessInfo.processInfo.systemUptime
        let dirtyRects = screenDirtyRects(from: attachments[.dirtyRects])
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let captureContext: (
            CGDirectDisplayID,
            UInt64,
            ScreenChangeEvaluation
        )? = withLifecycleLock {
            guard self.stream === stream,
                  let displayID = activeDisplayID,
                  legacyScreenAnalysisEnabled,
                  detector.isAnalysisDue(timestamp: analysisTimestamp),
                  let changedAreaRatio = pixelChangeAnalyzer.changedAreaRatio(
                    pixelBuffer: pixelBuffer
                  ),
                  let evaluation = detector.ingestMeasured(
                    timestamp: analysisTimestamp,
                    changedAreaRatio: changedAreaRatio,
                    changedRegionCount: dirtyRects.count
                  )
            else { return nil }
            return (displayID, captureGeneration, evaluation)
        }
        guard let (displayID, generation, evaluation) = captureContext else {
            return
        }
        let observation = ScreenChangeObservation(
            displayID: displayID,
            frameWidth: width,
            frameHeight: height,
            evaluation: evaluation
        )
        enqueueLatest(observation)

        let shouldRecognizeText = withLifecycleLock {
            guard self.stream === stream, captureGeneration == generation else {
                return false
            }
            return textTriggerController.shouldRecognize(evaluation: evaluation)
        }
        guard shouldRecognizeText else { return }
        let privacyContext = ScreenTextPrivacyContext(
            revision: evaluation.revision,
            observedAt: observation.observedAt,
            displayID: displayID
        )
        guard let applicationContext = screenTextPrivacyGate(privacyContext) else {
            textTriggerController.completeAttempt(succeeded: false)
            return
        }
        let submitted = focusedWindowCapture.submit(FocusedWindowCaptureRequest(
            generation: generation,
            privacyContext: privacyContext,
            applicationContext: applicationContext
        ))
        if !submitted {
            textTriggerController.completeAttempt(succeeded: false)
        }
    }

    public func stream(_ stream: SCStream, didStopWithError error: Error) {
        let stoppedGeneration: UInt64? = withLifecycleLock {
            if self.stream === stream {
                captureActive = false
                self.stream = nil
                activeDisplayID = nil
                environmentCaptureFilter = nil
                environmentCaptureConfiguration = nil
                exclusionUpdatePending = false
                pixelChangeAnalyzer.reset()
                let generation = captureGeneration
                captureGeneration &+= 1
                textTriggerController.reset()
                return generation
            }
            return nil
        }
        guard let stoppedGeneration else { return }
        focusedWindowCapture.stop(generation: stoppedGeneration)
        textRecognizer.invalidate(generation: stoppedGeneration)
        visionCandidateEncoder.invalidate(generation: stoppedGeneration)
        environmentFrameEncoder.invalidate(generation: stoppedGeneration)
        environmentFrameSampler.invalidate(generation: stoppedGeneration)
        _ = publishStatus(.init(
            state: .failed,
            reason: "screen_capture_stopped: \(error.localizedDescription)"
        ))
    }

    private func startEnvironmentSampling(generation: UInt64) {
        guard canCaptureEnvironmentFrame() else { return }
        withLifecycleLock {
            guard captureActive, captureGeneration == generation,
                  !legacyScreenAnalysisEnabled, !environmentCaptureSuspended,
                  !exclusionUpdatePending else { return }
            environmentFrameSampler.invalidate()
            environmentFrameEncoder.activate(generation: generation)
            environmentFrameSampler.activate(generation: generation)
        }
    }

    private func canCaptureEnvironmentFrame() -> Bool {
        let displayID = withLifecycleLock {
            captureActive && !legacyScreenAnalysisEnabled && !environmentCaptureSuspended
                && !exclusionUpdatePending
                ? activeDisplayID : nil
        }
        guard let displayID, hasScreenRecordingPermission(),
              CGDisplayIsAsleep(displayID) == 0,
              let session = CGSessionCopyCurrentDictionary() as? [String: Any],
              session[kCGSessionOnConsoleKey as String] as? Bool == true,
              session["CGSSessionScreenIsLocked"] as? Bool != true
        else { return false }
        return true
    }

    private func captureEnvironmentFrame(generation: UInt64) async -> CapturedEnvironmentFrame? {
        let context = withLifecycleLock { () -> (SCContentFilter, SCStreamConfiguration, CGDirectDisplayID)? in
            guard captureGeneration == generation, captureActive,
                  !exclusionUpdatePending,
                  let filter = environmentCaptureFilter,
                  let configuration = environmentCaptureConfiguration,
                  let displayID = activeDisplayID else { return nil }
            return (filter, configuration, displayID)
        }
        guard let (filter, configuration, displayID) = context,
              canCaptureEnvironmentFrame(), !Task.isCancelled else { return nil }
        if #available(macOS 14.0, *) {
            // This API asks WindowServer for current pixels even when the
            // continuous SCStream has reported only idle frames.
            return await withCheckedContinuation { continuation in
                SCScreenshotManager.captureSampleBuffer(
                    contentFilter: filter, configuration: configuration
                ) { sampleBuffer, _ in
                    guard let sampleBuffer, sampleBuffer.isValid,
                          let pixels = sampleBuffer.imageBuffer else {
                        continuation.resume(returning: nil)
                        return
                    }
                    continuation.resume(returning: CapturedEnvironmentFrame(
                        pixelBuffer: pixels,
                        observedAt: Date().timeIntervalSince1970,
                        displayID: displayID
                    ))
                }
            }
        }
        let attempt = EnvironmentScreenshotStreamAttempt(
            filter: filter, configuration: configuration, displayID: displayID
        )
        return await withTaskCancellationHandler {
            await attempt.capture()
        } onCancel: {
            attempt.cancel()
        }
    }

    @objc private func suspendEnvironmentCapture(_ notification: Notification) {
        let generation = withLifecycleLock { () -> UInt64 in
            environmentCaptureSuspended = true
            return captureGeneration
        }
        environmentFrameSampler.invalidate(generation: generation)
        environmentFrameEncoder.invalidate(generation: generation)
    }

    @objc private func resumeEnvironmentCapture(_ notification: Notification) {
        let generation = withLifecycleLock { () -> UInt64 in
            environmentCaptureSuspended = false
            if captureActive && legacyScreenAnalysisEnabled {
                environmentFrameEncoder.activate(generation: captureGeneration)
            }
            return captureGeneration
        }
        startEnvironmentSampling(generation: generation)
    }

    deinit {
        environmentFrameSampler.invalidate()
        NSWorkspace.shared.notificationCenter.removeObserver(self)
        DistributedNotificationCenter.default().removeObserver(self)
    }

    private func enqueueLatest(_ observation: ScreenChangeObservation) {
        lifecycleLock.lock()
        latestObservation = observation
        guard !callbackPending else {
            lifecycleLock.unlock()
            return
        }
        callbackPending = true
        lifecycleLock.unlock()

        callbackQueue.async { [weak self] in
            guard let self else { return }
            while true {
                self.lifecycleLock.lock()
                guard let next = self.latestObservation else {
                    self.callbackPending = false
                    self.lifecycleLock.unlock()
                    return
                }
                self.latestObservation = nil
                self.lifecycleLock.unlock()
                self.onObservation(next)
            }
        }
    }

    @discardableResult
    private func publishStatus(_ status: ScreenMonitorStatus) -> ScreenMonitorStatus {
        Task { @MainActor [onStatus] in onStatus(status) }
        return status
    }

    private func withLifecycleLock<Value>(_ body: () -> Value) -> Value {
        lifecycleLock.lock()
        defer { lifecycleLock.unlock() }
        return body()
    }

    private static let defaultExcludedBundleIdentifiers: Set<String> = [
        "app.tulpa.desktop",
        "app.fishmemory.desktop",
    ]
}

private enum ScreenCaptureFilterError: Error {
    case displayNotFound
}

extension ScreenMonitor: ScreenMonitoring {}

func screenCaptureNeedsPersistentStream(
    legacyScreenAnalysis: Bool,
    screenshotManagerAvailable: Bool
) -> Bool {
    legacyScreenAnalysis || !screenshotManagerAvailable
}

@available(macOS 12.3, *)
func makeScreenStreamConfiguration(
    displayWidth: Int,
    displayHeight: Int,
    legacyScreenAnalysis: Bool
) -> SCStreamConfiguration {
    let configuration = SCStreamConfiguration()
    if legacyScreenAnalysis {
        configuration.width = displayWidth
        configuration.height = displayHeight
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 30)
        configuration.queueDepth = 3
    } else {
        // The real environment image comes from the periodic screenshot path.
        // Limit the old-system lifecycle stream to one display point instead
        // of asking WindowServer to downscale the entire changing display.
        configuration.width = 2
        configuration.height = 2
        configuration.sourceRect = CGRect(x: 0, y: 0, width: 1, height: 1)
        configuration.minimumFrameInterval = CMTime(value: 60, timescale: 1)
        configuration.queueDepth = 1
    }
    configuration.pixelFormat = kCVPixelFormatType_32BGRA
    configuration.showsCursor = false
    return configuration
}

@available(macOS 12.3, *)
func makeEnvironmentScreenshotConfiguration(
    displayWidth: Int,
    displayHeight: Int
) -> SCStreamConfiguration {
    let configuration = SCStreamConfiguration()
    let scale = min(1, 960.0 / Double(max(displayWidth, displayHeight)))
    configuration.width = max(1, Int(Double(displayWidth) * scale))
    configuration.height = max(1, Int(Double(displayHeight) * scale))
    configuration.pixelFormat = kCVPixelFormatType_32BGRA
    configuration.showsCursor = false
    configuration.queueDepth = 1
    // macOS 12.3/13 feed this configuration to a temporary SCStream. Avoid
    // inheriting ScreenCaptureKit's 60 fps default while waiting for frame one.
    configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
    return configuration
}

/// macOS 12.3/13 equivalent of a screenshot: start a filtered display stream,
/// take its first actual frame, and wait for stream teardown before returning.
@available(macOS 12.3, *)
private final class EnvironmentScreenshotStreamAttempt: NSObject,
    SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private let filter: SCContentFilter
    private let configuration: SCStreamConfiguration
    private let displayID: CGDirectDisplayID
    private let queue = DispatchQueue(label: "app.fishmemory.perception.environment-screenshot")
    private let lock = NSLock()
    private var continuation: CheckedContinuation<CapturedEnvironmentFrame?, Never>?
    private var stream: SCStream?
    private var timeout: DispatchWorkItem?
    private var completed = false
    private var starting = false
    private var result: CapturedEnvironmentFrame?

    init(filter: SCContentFilter, configuration: SCStreamConfiguration,
         displayID: CGDirectDisplayID) {
        self.filter = filter
        self.configuration = configuration
        self.displayID = displayID
        super.init()
    }

    func capture() async -> CapturedEnvironmentFrame? {
        await withCheckedContinuation { continuation in
            lock.lock()
            guard !completed else {
                lock.unlock()
                continuation.resume(returning: nil)
                return
            }
            self.continuation = continuation
            let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
            do {
                try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
            } catch {
                lock.unlock()
                finish(nil)
                return
            }
            self.stream = stream
            starting = true
            let timeout = DispatchWorkItem { [weak self] in self?.finish(nil) }
            self.timeout = timeout
            lock.unlock()
            queue.asyncAfter(deadline: .now() + 2, execute: timeout)
            stream.startCapture { [self] error in
                lock.lock()
                starting = false
                if error != nil { completed = true }
                let shouldFinish = completed
                lock.unlock()
                if shouldFinish { tearDown() }
            }
        }
    }

    func cancel() { finish(nil) }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of outputType: SCStreamOutputType) {
        guard outputType == .screen, sampleBuffer.isValid,
              let attachments = frameAttachments(sampleBuffer),
              let rawStatus = attachments[.status] as? Int,
              let status = SCFrameStatus(rawValue: rawStatus),
              status == .complete || status == .started,
              let pixels = sampleBuffer.imageBuffer else { return }
        finish(CapturedEnvironmentFrame(
            pixelBuffer: pixels, observedAt: Date().timeIntervalSince1970, displayID: displayID
        ))
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) { finish(nil) }

    private func finish(_ frame: CapturedEnvironmentFrame?) {
        lock.lock()
        guard !completed else { lock.unlock(); return }
        completed = true
        result = frame
        let shouldFinish = !starting
        lock.unlock()
        if shouldFinish { tearDown() }
    }

    private func tearDown() {
        lock.lock()
        let continuation = self.continuation
        self.continuation = nil
        let stream = self.stream
        self.stream = nil
        let timeout = self.timeout
        self.timeout = nil
        let result = self.result
        self.result = nil
        lock.unlock()
        timeout?.cancel()
        guard let continuation else { return }
        if let stream {
            stream.stopCapture { [self] _ in
                try? stream.removeStreamOutput(self, type: .screen)
                continuation.resume(returning: result)
            }
        } else {
            continuation.resume(returning: result)
        }
    }
}

func shouldExcludeScreenApplication(
    bundleIdentifier: String?,
    processIdentifier: Int32,
    parentProcessIdentifier: Int32?,
    mainBundleIdentifier: String?,
    excludedBundleIdentifiers: Set<String>,
    excludeHostApplication: Bool = true
) -> Bool {
    guard excludeHostApplication else { return false }
    if processIdentifier == parentProcessIdentifier { return true }
    guard let bundleIdentifier, !bundleIdentifier.isEmpty else { return false }
    if bundleIdentifier == mainBundleIdentifier
        || excludedBundleIdentifiers.contains(bundleIdentifier)
    {
        return true
    }
    return bundleIdentifier.hasPrefix("app.tulpa.")
        || bundleIdentifier.hasPrefix("app.fishmemory.")
        || bundleIdentifier.hasPrefix("com.fishmemory.")
        || bundleIdentifier.hasPrefix("com.fish-memory.")
}

@available(macOS 12.3, *)
private func frameAttachments(
    _ sampleBuffer: CMSampleBuffer
) -> [SCStreamFrameInfo: Any]? {
    guard let array = CMSampleBufferGetSampleAttachmentsArray(
        sampleBuffer,
        createIfNecessary: false
    ) as? [[SCStreamFrameInfo: Any]] else { return nil }
    return array.first
}

@available(macOS 12.3, *)
private func screenDirtyRects(from value: Any?) -> [CGRect] {
    if let rectangles = value as? [CGRect] { return rectangles }
    if let dictionaries = value as? [CFDictionary] {
        return dictionaries.compactMap(CGRect.init(dictionaryRepresentation:))
    }
    if let values = value as? [NSValue] {
        return values.map(\.rectValue)
    }
    return []
}
