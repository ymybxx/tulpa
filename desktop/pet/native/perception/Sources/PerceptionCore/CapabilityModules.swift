import ApplicationServices
import AVFoundation
import CoreGraphics
import Foundation

@MainActor
public protocol ScreenMonitoring: AnyObject {
    func start() async -> ScreenMonitorStatus
    func stop() async
    func setEnhancedVisionEnabled(_ enabled: Bool)
    func setLegacyScreenAnalysisEnabled(_ enabled: Bool)
    func setExcludeHostApplicationFromCapture(_ enabled: Bool)
    func setEnvironmentFrameInterval(milliseconds: Int, tickImmediately: Bool)
}

public struct ScreenMonitorCallbacks: Sendable {
    public let onObservation: @Sendable (ScreenChangeObservation) -> Void
    public let screenTextPrivacyGate: ScreenTextPrivacyGate
    public let screenTextAuthorizationValidator: ScreenTextAuthorizationValidator
    public let onTextObservation: @Sendable (ScreenTextObservation) -> Void
    public let enhancedVisionEnabled: Bool
    public let legacyScreenAnalysisEnabled: Bool
    public let excludeHostApplicationFromCapture: Bool
    public let visionCandidateBroker: VisionCandidateBroker
    public let onVisionCandidateAvailable: @Sendable (VisionCandidateMetadata) -> Void
    public let environmentFrameBroker: EnvironmentFrameBroker
    public let onEnvironmentFrameAvailable: @Sendable (EnvironmentFrameMetadata) -> Void
    public let onStatus: ScreenMonitor.StatusHandler

    public init(
        onObservation: @escaping @Sendable (ScreenChangeObservation) -> Void,
        screenTextPrivacyGate: @escaping ScreenTextPrivacyGate = ScreenTextPrivacyGates.denyAll,
        screenTextAuthorizationValidator: @escaping ScreenTextAuthorizationValidator = {
            candidate,
            expected in
            ScreenTextPrivacyGates.denyAll(candidate) == expected
        },
        onTextObservation: @escaping @Sendable (ScreenTextObservation) -> Void = { _ in },
        enhancedVisionEnabled: Bool = false,
        legacyScreenAnalysisEnabled: Bool = true,
        excludeHostApplicationFromCapture: Bool = false,
        visionCandidateBroker: VisionCandidateBroker = .init(),
        onVisionCandidateAvailable: @escaping @Sendable (
            VisionCandidateMetadata
        ) -> Void = { _ in },
        environmentFrameBroker: EnvironmentFrameBroker = .init(),
        onEnvironmentFrameAvailable: @escaping @Sendable (
            EnvironmentFrameMetadata
        ) -> Void = { _ in },
        onStatus: @escaping ScreenMonitor.StatusHandler
    ) {
        self.onObservation = onObservation
        self.screenTextPrivacyGate = screenTextPrivacyGate
        self.screenTextAuthorizationValidator = screenTextAuthorizationValidator
        self.onTextObservation = onTextObservation
        self.enhancedVisionEnabled = enhancedVisionEnabled
        self.legacyScreenAnalysisEnabled = legacyScreenAnalysisEnabled
        self.excludeHostApplicationFromCapture = excludeHostApplicationFromCapture
        self.visionCandidateBroker = visionCandidateBroker
        self.onVisionCandidateAvailable = onVisionCandidateAvailable
        self.environmentFrameBroker = environmentFrameBroker
        self.onEnvironmentFrameAvailable = onEnvironmentFrameAvailable
        self.onStatus = onStatus
    }
}

@MainActor
public final class LazyCapabilityModules {
    public typealias ScreenHandler = @Sendable (ScreenChangeObservation) -> Void
    public typealias ScreenTextHandler = @Sendable (ScreenTextObservation) -> Void
    public typealias VisionCandidateHandler = @Sendable (
        VisionCandidateMetadata
    ) -> Void
    public typealias EnvironmentFrameHandler = @Sendable (
        EnvironmentFrameMetadata
    ) -> Void
    public typealias AccessibilityHandler = @MainActor (AccessibilityObservation) -> Void
    public typealias CameraHandler = @Sendable (CameraObservation) -> Void
    public typealias ComputerHistoryHandler = @MainActor (ComputerHistoryEvent) -> Void
    public typealias ComputerHistorySuppressionHandler = @MainActor (
        ComputerHistorySuppressionReport
    ) -> Void
    public typealias StatusHandler = @MainActor () -> Void
    public typealias ScreenPermissionRequester = @Sendable () async -> Bool
    public typealias ScreenMonitorFactory = @MainActor (
        ScreenMonitorCallbacks
    ) -> any ScreenMonitoring
    public typealias EnhancedVisionActivation = @Sendable (
        PerceptionModuleSelection
    ) -> Bool
    public typealias ComputerHistoryMonitorFactory = @MainActor (
        @escaping ComputerHistoryHandler,
        @escaping ComputerHistorySuppressionHandler,
        @escaping StatusHandler
    ) -> any ComputerHistoryMonitoring
    public typealias ComputerHistoryTrustCheck = @MainActor () -> Bool

    private let onScreenObservation: ScreenHandler
    private let onScreenTextObservation: ScreenTextHandler
    private let onVisionCandidateAvailable: VisionCandidateHandler
    private let onEnvironmentFrameAvailable: EnvironmentFrameHandler
    private let onAccessibilityObservation: AccessibilityHandler
    private let onCameraObservation: CameraHandler
    private let onComputerHistoryEvent: ComputerHistoryHandler
    private let onComputerHistorySuppressed: ComputerHistorySuppressionHandler
    private let onStatusChanged: StatusHandler
    private let requestScreenPermission: ScreenPermissionRequester
    private let makeScreenMonitor: ScreenMonitorFactory
    private let enhancedVisionActivation: EnhancedVisionActivation
    private let makeComputerHistoryMonitor: ComputerHistoryMonitorFactory
    private let computerHistoryTrustCheck: ComputerHistoryTrustCheck
    private var screenMonitor: (any ScreenMonitoring)?
    private var startingScreenMonitor: (any ScreenMonitoring)?
    private var screenObservationGate: ScreenObservationGate?
    private var accessibilityMonitor: AccessibilityMonitor?
    private var cameraMonitor: CameraMonitor?
    private var computerHistoryMonitor: (any ComputerHistoryMonitoring)?
    private var screenStartTask: Task<Void, Never>?
    private var accessibilityPermissionTask: Task<Void, Never>?
    private var cameraPermissionTask: Task<Void, Never>?
    private var computerHistoryPermissionTask: Task<Void, Never>?
    private var screenGeneration = 0
    private var accessibilityGeneration = 0
    private var cameraGeneration = 0
    private var computerHistoryGeneration = 0
    private var selection = PerceptionModuleSelection.disabled
    private var environmentFrameIntervalMs =
        EnvironmentFrameSamplingLoop.defaultIntervalMilliseconds
    private var screenReport = LazyCapabilityModules.disabledScreenReport()
    private var accessibilityReport = LazyCapabilityModules.disabledAccessibilityReport()
    private var cameraReport = LazyCapabilityModules.disabledCameraReport()
    private var computerHistoryReport =
        LazyCapabilityModules.disabledComputerHistoryReport()
    public let screenTextPrivacyContext: ScreenTextPrivacyContextStore
    public let visionCandidateBroker: VisionCandidateBroker
    public let environmentFrameBroker: EnvironmentFrameBroker
    private var enhancedVisionActive = false
    private let systemAudio: SystemAudioModule

    func cutEnvironmentWindow(at timestamp: Double) {
        systemAudio.cutWindow(at: timestamp)
    }

    public init(
        screenTextPrivacyContext: ScreenTextPrivacyContextStore = .init(),
        visionCandidateBroker: VisionCandidateBroker = .init(),
        environmentFrameBroker: EnvironmentFrameBroker = .init(),
        onScreenObservation: @escaping ScreenHandler = { _ in },
        onScreenTextObservation: @escaping ScreenTextHandler = { _ in },
        onVisionCandidateAvailable: @escaping VisionCandidateHandler = { _ in },
        onEnvironmentFrameAvailable: @escaping EnvironmentFrameHandler = { _ in },
        onAccessibilityObservation: @escaping AccessibilityHandler = { _ in },
        onCameraObservation: @escaping CameraHandler = { _ in },
        onComputerHistoryEvent: @escaping ComputerHistoryHandler = { _ in },
        onComputerHistorySuppressed: @escaping ComputerHistorySuppressionHandler = {
            _ in
        },
        onStatusChanged: @escaping StatusHandler = {},
        requestScreenPermission: @escaping ScreenPermissionRequester = {
            await Task.detached(priority: .userInitiated) {
                ScreenMonitor.requestPermission()
            }.value
        },
        makeScreenMonitor: @escaping ScreenMonitorFactory = { callbacks in
            ScreenMonitor(
                excludeHostApplicationFromCapture:
                    callbacks.excludeHostApplicationFromCapture,
                onObservation: callbacks.onObservation,
                screenTextPrivacyGate: callbacks.screenTextPrivacyGate,
                screenTextAuthorizationValidator:
                    callbacks.screenTextAuthorizationValidator,
                onTextObservation: callbacks.onTextObservation,
                enhancedVisionEnabled: callbacks.enhancedVisionEnabled,
                legacyScreenAnalysisEnabled: callbacks.legacyScreenAnalysisEnabled,
                visionCandidateBroker: callbacks.visionCandidateBroker,
                onVisionCandidateAvailable: callbacks.onVisionCandidateAvailable,
                environmentFrameBroker: callbacks.environmentFrameBroker,
                onEnvironmentFrameAvailable: callbacks.onEnvironmentFrameAvailable,
                onStatus: callbacks.onStatus
            )
        },
        enhancedVisionActivation: @escaping EnhancedVisionActivation = {
            $0.enhancedVisionActive
        },
        computerHistoryTrustCheck: @escaping ComputerHistoryTrustCheck = {
            AXIsProcessTrusted()
        },
        onSystemAudioWindow: @escaping @Sendable (SystemAudioWindow) -> Void = { _ in },
        makeComputerHistoryMonitor: @escaping ComputerHistoryMonitorFactory = {
            onEvent,
            onSuppressed,
            onInputStatusChanged in
            ComputerHistoryMonitor(
                onSuppressed: onSuppressed,
                onInputStatusChanged: onInputStatusChanged,
                onEvent: onEvent
            )
        }
    ) {
        self.screenTextPrivacyContext = screenTextPrivacyContext
        self.visionCandidateBroker = visionCandidateBroker
        self.environmentFrameBroker = environmentFrameBroker
        self.onScreenObservation = onScreenObservation
        self.onScreenTextObservation = onScreenTextObservation
        self.onVisionCandidateAvailable = onVisionCandidateAvailable
        self.onEnvironmentFrameAvailable = onEnvironmentFrameAvailable
        self.onAccessibilityObservation = onAccessibilityObservation
        self.onCameraObservation = onCameraObservation
        self.onComputerHistoryEvent = onComputerHistoryEvent
        self.onComputerHistorySuppressed = onComputerHistorySuppressed
        self.onStatusChanged = onStatusChanged
        self.requestScreenPermission = requestScreenPermission
        self.makeScreenMonitor = makeScreenMonitor
        self.enhancedVisionActivation = enhancedVisionActivation
        self.computerHistoryTrustCheck = computerHistoryTrustCheck
        self.makeComputerHistoryMonitor = makeComputerHistoryMonitor
        systemAudio = SystemAudioModule(
            onWindow: onSystemAudioWindow,
            onStatusChanged: onStatusChanged
        )
    }

    public func configure(
        _ next: PerceptionModuleSelection,
        environmentFrameIntervalMs: Int? = nil
    ) -> PerceptionCapabilities {
        selection = next
        if let environmentFrameIntervalMs {
            self.environmentFrameIntervalMs = environmentFrameIntervalMs
            screenMonitor?.setEnvironmentFrameInterval(
                milliseconds: environmentFrameIntervalMs,
                tickImmediately: true
            )
            startingScreenMonitor?.setEnvironmentFrameInterval(
                milliseconds: environmentFrameIntervalMs,
                tickImmediately: true
            )
        }
        enhancedVisionActive = enhancedVisionActivation(next)
        screenMonitor?.setEnhancedVisionEnabled(enhancedVisionActive)
        startingScreenMonitor?.setEnhancedVisionEnabled(enhancedVisionActive)
        screenMonitor?.setLegacyScreenAnalysisEnabled(next.legacyScreenAnalysis)
        startingScreenMonitor?.setLegacyScreenAnalysisEnabled(
            next.legacyScreenAnalysis
        )
        screenMonitor?.setExcludeHostApplicationFromCapture(
            next.excludeHostApplicationFromCapture
        )
        startingScreenMonitor?.setExcludeHostApplicationFromCapture(
            next.excludeHostApplicationFromCapture
        )
        if !enhancedVisionActive { visionCandidateBroker.clear() }
        configureScreen(enabled: next.screen)
        configureSystemAudio()
        configureAccessibility(enabled: next.accessibility)
        configureCamera(enabled: next.camera)
        configureComputerHistory(enabled: next.computerHistory)
        return report()
    }

    public func stopAll() async {
        selection = .disabled
        enhancedVisionActive = false
        visionCandidateBroker.clear()
        environmentFrameBroker.clear()
        let screenStop = stopScreen()
        let audioStop = systemAudio.stop()
        stopAccessibility()
        stopCamera()
        stopComputerHistory()
        onStatusChanged()
        await screenStop?.value
        _ = await audioStop?.value
    }

    public func report() -> PerceptionCapabilities {
        PerceptionCapabilities(
            workspace: CapabilityReport(
                supported: true,
                requested: selection.workspace,
                state: selection.workspace ? .running : .disabled,
                permission: .notRequired
            ),
            screen: screenReport,
            accessibility: accessibilityReport,
            camera: cameraReport,
            computerHistory: computerHistoryMonitor?.currentReport()
                ?? computerHistoryReport,
            systemAudio: systemAudio.report
        )
    }

    public var moduleSelection: PerceptionModuleSelection { selection }

    private func configureSystemAudio() {
        systemAudio.configure(requested: selection.systemAudio,
                              screenEnabled: selection.screen)
    }

    private func configureScreen(enabled: Bool) {
        guard enabled else {
            stopScreen()
            return
        }
        guard screenMonitor == nil, screenStartTask == nil else { return }
        screenGeneration += 1
        let generation = screenGeneration
        let observationGate = ScreenObservationGate(
            changeHandler: onScreenObservation,
            textHandler: onScreenTextObservation,
            visionHandler: onVisionCandidateAvailable,
            environmentFrameHandler: onEnvironmentFrameAvailable
        )
        screenObservationGate = observationGate
        screenReport = CapabilityReport(
            supported: true,
            requested: true,
            state: .unavailable,
            permission: screenPermission(),
            reason: "screen_recording_permission_required"
        )
        onStatusChanged()
        screenStartTask = Task { [weak self] in
            guard let self else { return }
            let granted = await self.requestScreenPermission()
            guard !Task.isCancelled,
                  generation == self.screenGeneration,
                  self.selection.screen
            else { return }
            guard granted else {
                observationGate.invalidate()
                if self.screenObservationGate === observationGate {
                    self.screenObservationGate = nil
                }
                self.screenStartTask = nil
                self.screenReport = CapabilityReport(
                    supported: true,
                    requested: true,
                    state: .unavailable,
                    permission: screenPermission(),
                    reason: "screen_recording_permission_required"
                )
                self.onStatusChanged()
                return
            }
            let monitor = self.makeScreenMonitor(ScreenMonitorCallbacks(
                onObservation: { observation in
                    observationGate.deliver(change: observation)
                },
                screenTextPrivacyGate: { [screenTextPrivacyContext] candidate in
                    screenTextPrivacyContext.authorization(for: candidate)
                },
                screenTextAuthorizationValidator: {
                    [screenTextPrivacyContext] _, expected in
                    screenTextPrivacyContext.isStillAuthorized(expected)
                },
                onTextObservation: { observation in
                    observationGate.deliver(text: observation)
                },
                enhancedVisionEnabled: self.enhancedVisionActive,
                legacyScreenAnalysisEnabled: self.selection.legacyScreenAnalysis,
                excludeHostApplicationFromCapture:
                    self.selection.excludeHostApplicationFromCapture,
                visionCandidateBroker: self.visionCandidateBroker,
                onVisionCandidateAvailable: { metadata in
                    observationGate.deliver(vision: metadata)
                },
                environmentFrameBroker: self.environmentFrameBroker,
                onEnvironmentFrameAvailable: { metadata in
                    observationGate.deliver(environmentFrame: metadata)
                },
                onStatus: { [weak self] status in
                    self?.acceptScreenStatus(status, generation: generation)
                }
            ))
            self.startingScreenMonitor = monitor
            monitor.setEnvironmentFrameInterval(
                milliseconds: self.environmentFrameIntervalMs,
                tickImmediately: false
            )
            monitor.setEnhancedVisionEnabled(self.enhancedVisionActive)
            monitor.setLegacyScreenAnalysisEnabled(
                self.selection.legacyScreenAnalysis
            )
            monitor.setExcludeHostApplicationFromCapture(
                self.selection.excludeHostApplicationFromCapture
            )
            let status = await monitor.start()
            guard !Task.isCancelled,
                  generation == self.screenGeneration,
                  self.selection.screen
            else {
                observationGate.invalidate()
                let ownsStop = self.startingScreenMonitor === monitor
                if ownsStop {
                    self.startingScreenMonitor = nil
                    await monitor.stop()
                }
                return
            }
            // enhancedVision may have changed while start() was awaiting
            // ScreenCaptureKit permission/catalog setup. Reapply the current
            // derived gate instead of retaining the construction-time value.
            monitor.setEnhancedVisionEnabled(self.enhancedVisionActive)
            monitor.setLegacyScreenAnalysisEnabled(
                self.selection.legacyScreenAnalysis
            )
            monitor.setExcludeHostApplicationFromCapture(
                self.selection.excludeHostApplicationFromCapture
            )
            if self.startingScreenMonitor === monitor {
                self.startingScreenMonitor = nil
            }
            self.screenStartTask = nil
            self.screenMonitor = status.state == .running ? monitor : nil
            if status.state != .running {
                observationGate.invalidate()
                if self.screenObservationGate === observationGate {
                    self.screenObservationGate = nil
                }
            }
            self.acceptScreenStatus(status, generation: generation)
        }
    }

    private func acceptScreenStatus(_ status: ScreenMonitorStatus, generation: Int) {
        guard generation == screenGeneration, selection.screen else { return }
        let state: CapabilityState
        switch status.state {
        case .running: state = .running
        case .failed: state = .failed
        case .stopped: state = .disabled
        case .starting, .unavailable: state = .unavailable
        }
        screenReport = CapabilityReport(
            supported: true,
            requested: true,
            state: state,
            permission: screenPermission(),
            reason: status.reason
        )
        configureSystemAudio()
        onStatusChanged()
    }

    @discardableResult
    private func stopScreen() -> Task<Void, Never>? {
        screenGeneration += 1
        visionCandidateBroker.clear()
        environmentFrameBroker.clear()
        screenObservationGate?.invalidate()
        screenObservationGate = nil
        let pendingStart = screenStartTask
        pendingStart?.cancel()
        screenStartTask = nil
        let starting = startingScreenMonitor
        starting?.setEnhancedVisionEnabled(false)
        starting?.setLegacyScreenAnalysisEnabled(false)
        startingScreenMonitor = nil
        let monitor = screenMonitor
        screenMonitor = nil
        monitor?.setEnhancedVisionEnabled(false)
        monitor?.setLegacyScreenAnalysisEnabled(false)
        screenReport = Self.disabledScreenReport()
        guard starting != nil || monitor != nil || pendingStart != nil else {
            return nil
        }
        return Task {
            if let starting { await starting.stop() }
            if let monitor { await monitor.stop() }
            if let pendingStart { await pendingStart.value }
        }
    }

    private func configureAccessibility(enabled: Bool) {
        guard enabled else {
            stopAccessibility()
            return
        }
        guard accessibilityMonitor == nil, accessibilityPermissionTask == nil else { return }
        screenTextPrivacyContext.clearAccessibility()
        accessibilityGeneration += 1
        let generation = accessibilityGeneration
        if AXIsProcessTrusted() {
            startAccessibility(generation: generation)
            return
        }
        _ = AccessibilityMonitor.requestPermissionPrompt()
        accessibilityReport = CapabilityReport(
            supported: true,
            requested: true,
            state: .unavailable,
            permission: .notDetermined,
            reason: "accessibility_permission_required"
        )
        onStatusChanged()
        accessibilityPermissionTask = Task { [weak self] in
            guard let self else { return }
            for _ in 0 ..< 120 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard !Task.isCancelled else { return }
                guard generation == self.accessibilityGeneration,
                      self.selection.accessibility else { return }
                if AXIsProcessTrusted() {
                    self.accessibilityPermissionTask = nil
                    self.startAccessibility(generation: generation)
                    return
                }
            }
            self.accessibilityPermissionTask = nil
        }
    }

    private func startAccessibility(generation: Int) {
        guard generation == accessibilityGeneration, selection.accessibility else { return }
        let monitor = AccessibilityMonitor { [weak self] observation in
            guard let self else { return }
            self.visionCandidateBroker.clear()
            self.screenTextPrivacyContext.updateAccessibility(observation)
            self.onAccessibilityObservation(observation)
        }
        accessibilityMonitor = monitor
        accessibilityReport = monitor.start()
        onStatusChanged()
    }

    private func stopAccessibility() {
        accessibilityGeneration += 1
        visionCandidateBroker.clear()
        screenTextPrivacyContext.clearAccessibility()
        accessibilityPermissionTask?.cancel()
        accessibilityPermissionTask = nil
        accessibilityMonitor?.stop()
        accessibilityMonitor = nil
        accessibilityReport = Self.disabledAccessibilityReport()
    }

    private func configureCamera(enabled: Bool) {
        guard enabled else {
            stopCamera()
            return
        }
        guard cameraMonitor == nil, cameraPermissionTask == nil else { return }
        cameraGeneration += 1
        let generation = cameraGeneration
        cameraReport = CapabilityReport(
            supported: true,
            requested: true,
            state: .unavailable,
            permission: cameraPermission(),
            reason: "camera_permission_required"
        )
        onStatusChanged()
        cameraPermissionTask = Task { [weak self] in
            guard let self else { return }
            let granted = await CameraMonitor.requestPermission()
            guard generation == self.cameraGeneration, self.selection.camera else { return }
            self.cameraPermissionTask = nil
            guard granted else {
                self.cameraReport = CapabilityReport(
                    supported: true,
                    requested: true,
                    state: .unavailable,
                    permission: cameraPermission(),
                    reason: "camera_permission_required"
                )
                self.onStatusChanged()
                return
            }
            let monitor = CameraMonitor(onObservation: self.onCameraObservation)
            self.cameraMonitor = monitor
            self.cameraReport = monitor.start()
            self.onStatusChanged()
        }
    }

    private func stopCamera() {
        cameraGeneration += 1
        cameraPermissionTask?.cancel()
        cameraPermissionTask = nil
        cameraMonitor?.stop()
        cameraMonitor = nil
        cameraReport = Self.disabledCameraReport()
    }

    private func configureComputerHistory(enabled: Bool) {
        guard enabled else {
            stopComputerHistory()
            return
        }
        guard computerHistoryMonitor == nil,
              computerHistoryPermissionTask == nil
        else { return }
        computerHistoryGeneration += 1
        let generation = computerHistoryGeneration
        if computerHistoryTrustCheck() {
            startComputerHistory(generation: generation)
            return
        }
        _ = AccessibilityMonitor.requestPermissionPrompt()
        computerHistoryReport = CapabilityReport(
            supported: true,
            requested: true,
            state: .unavailable,
            permission: .notDetermined,
            reason: "accessibility_permission_required"
        )
        onStatusChanged()
        computerHistoryPermissionTask = Task { [weak self] in
            guard let self else { return }
            for _ in 0 ..< 120 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard !Task.isCancelled,
                      generation == self.computerHistoryGeneration,
                      self.selection.computerHistory
                else { return }
                if self.computerHistoryTrustCheck() {
                    self.computerHistoryPermissionTask = nil
                    self.startComputerHistory(generation: generation)
                    return
                }
            }
            self.computerHistoryPermissionTask = nil
        }
    }

    private func startComputerHistory(generation: Int) {
        guard generation == computerHistoryGeneration,
              selection.computerHistory
        else { return }
        let monitor = makeComputerHistoryMonitor(
            onComputerHistoryEvent,
            onComputerHistorySuppressed,
            onStatusChanged
        )
        computerHistoryMonitor = monitor
        computerHistoryReport = monitor.start()
        onStatusChanged()
    }

    private func stopComputerHistory() {
        computerHistoryGeneration += 1
        computerHistoryPermissionTask?.cancel()
        computerHistoryPermissionTask = nil
        computerHistoryMonitor?.stop()
        computerHistoryMonitor = nil
        computerHistoryReport = Self.disabledComputerHistoryReport()
    }

    private static func disabledScreenReport() -> CapabilityReport {
        return CapabilityReport(
            supported: true,
            requested: false,
            state: .disabled,
            permission: screenPermission()
        )
    }

    private static func disabledAccessibilityReport() -> CapabilityReport {
        CapabilityReport(
            supported: true,
            requested: false,
            state: .disabled,
            permission: AXIsProcessTrusted() ? .granted : .notDetermined
        )
    }

    private static func disabledCameraReport() -> CapabilityReport {
        CapabilityReport(
            supported: true,
            requested: false,
            state: .disabled,
            permission: cameraPermission()
        )
    }

    private static func disabledComputerHistoryReport() -> CapabilityReport {
        let accessibilityPermission: CapabilityPermission = AXIsProcessTrusted()
            ? .granted
            : .notDetermined
        return CapabilityReport(
            supported: true,
            requested: false,
            state: .disabled,
            permission: accessibilityPermission,
            computerHistoryInput: .disabled(
                accessibilityPermission: accessibilityPermission
            )
        )
    }
}

final class ScreenObservationGate: @unchecked Sendable {
    private let lock = NSLock()
    private let changeHandler: LazyCapabilityModules.ScreenHandler
    private let textHandler: LazyCapabilityModules.ScreenTextHandler
    private let visionHandler: LazyCapabilityModules.VisionCandidateHandler
    private let environmentFrameHandler: LazyCapabilityModules.EnvironmentFrameHandler
    private var active = true

    init(
        changeHandler: @escaping LazyCapabilityModules.ScreenHandler,
        textHandler: @escaping LazyCapabilityModules.ScreenTextHandler,
        visionHandler: @escaping LazyCapabilityModules.VisionCandidateHandler,
        environmentFrameHandler: @escaping LazyCapabilityModules.EnvironmentFrameHandler
    ) {
        self.changeHandler = changeHandler
        self.textHandler = textHandler
        self.visionHandler = visionHandler
        self.environmentFrameHandler = environmentFrameHandler
    }

    func deliver(change observation: ScreenChangeObservation) {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return }
        changeHandler(observation)
    }

    func deliver(text observation: ScreenTextObservation) {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return }
        textHandler(observation)
    }

    func deliver(vision metadata: VisionCandidateMetadata) {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return }
        visionHandler(metadata)
    }

    func deliver(environmentFrame metadata: EnvironmentFrameMetadata) {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return }
        environmentFrameHandler(metadata)
    }

    func invalidate() {
        lock.lock()
        active = false
        lock.unlock()
    }
}

private func screenPermission() -> CapabilityPermission {
    CGPreflightScreenCaptureAccess() ? .granted : .notDetermined
}

private func cameraPermission() -> CapabilityPermission {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized: return .granted
    case .denied: return .denied
    case .restricted: return .restricted
    case .notDetermined: return .notDetermined
    @unknown default: return .unknown
    }
}
