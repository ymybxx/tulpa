import CoreGraphics
import Foundation

/// The media clock runs for either input. Audio owns its own permission and
/// capture lifecycle; screen-only windows carry no PCM and never open a mic.
@MainActor
final class SystemAudioModule {
    typealias Factory = @MainActor (SystemAudioCallbacks) -> (any SystemAudioMonitoring)?
    typealias SegmenterFactory = @MainActor (
        @escaping @Sendable (SystemAudioWindow) -> Void
    ) -> SystemAudioWindowSegmenter
    private let factory: Factory
    private let segmenterFactory: SegmenterFactory
    private let supported: Bool
    private let onWindow: @Sendable (SystemAudioWindow) -> Void
    private let onStatusChanged: @MainActor () -> Void
    private var segmenter: SystemAudioWindowSegmenter?
    private var monitor: (any SystemAudioMonitoring)?
    private var starting: (any SystemAudioMonitoring)?
    private var startTask: Task<Void, Never>?
    private var stopTask: Task<Bool, Never>?
    private var stopping: [any SystemAudioMonitoring] = []
    private var generation = 0
    private var requested = false
    private var shouldRun = false
    private var startAllowed = false
    private var gate: SystemAudioModuleGate?
    private(set) var report: CapabilityReport

    init(
        supported: Bool = SystemAudioModule.isSupported,
        factory: @escaping Factory = { callbacks in
            makeSystemAudioMonitor(callbacks: callbacks)
        },
        segmenterFactory: @escaping SegmenterFactory = { handler in
            SystemAudioWindowSegmenter(handler: handler)
        },
        onWindow: @escaping @Sendable (SystemAudioWindow) -> Void,
        onStatusChanged: @escaping @MainActor () -> Void
    ) {
        self.supported = supported
        self.factory = factory
        self.segmenterFactory = segmenterFactory
        self.onWindow = onWindow
        self.onStatusChanged = onStatusChanged
        report = .init(supported: supported, requested: false, state: .disabled, permission: .notDetermined)
    }

    nonisolated static var isSupported: Bool {
        if #available(macOS 13.0, *) { return true }
        return false
    }

    func cutWindow(at timestamp: Double) {
        segmenter?.cut(at: timestamp)
    }

    func configure(requested: Bool, screenEnabled: Bool) {
        if self.requested != requested {
            _ = stop()
        }
        self.requested = requested
        guard requested || screenEnabled else {
            _ = stop()
            return
        }
        let unavailableReason = !requested ? "system_audio_disabled"
            : !supported ? "macos_13_required"
            : "system_audio_starting"
        ensureSegmenter(initialUnavailableReason: unavailableReason)
        shouldRun = requested && supported
        startAllowed = shouldRun
        guard shouldRun else {
            _ = stopCapture()
            segmenter?.markBackendUnavailable(reason: unavailableReason)
            report = .init(
                supported: supported, requested: requested,
                state: requested ? .unavailable : .disabled,
                permission: CGPreflightScreenCaptureAccess() ? .granted : .notDetermined,
                reason: !requested ? nil : "macos_13_required"
            )
            return
        }
        if monitor == nil, starting == nil {
            segmenter?.markBackendUnavailable(reason: "system_audio_starting")
        }
        startIfPossible()
    }

    private func ensureSegmenter(initialUnavailableReason: String) {
        guard segmenter == nil else { return }
        let next = segmenterFactory(onWindow)
        segmenter = next
        next.start(initialUnavailableReason: initialUnavailableReason)
    }

    private func startIfPossible() {
        guard shouldRun, startAllowed, monitor == nil, starting == nil,
              startTask == nil else { return }
        guard stopTask == nil, stopping.isEmpty else {
            report = .init(
                supported: true,
                requested: true,
                state: .unavailable,
                permission: .granted,
                reason: "system_audio_waiting_for_cleanup"
            )
            if stopTask == nil { _ = startCleanupTask(pendingStart: nil) }
            return
        }
        generation += 1
        let token = generation
        guard let segmenter else { return }
        let deliveryGate = SystemAudioModuleGate(
            onSamples: { segmenter.append($0) },
            onCaptureGap: { segmenter.markGap($0) }
        )
        gate = deliveryGate
        let callbacks = SystemAudioCallbacks(
            onSamples: { deliveryGate.deliver($0) },
            onCaptureGap: { deliveryGate.deliver($0) },
            onStatus: { [weak self] report in
                guard let self, self.generation == token, self.requested else { return }
                self.report = report
                if report.state == .running {
                    self.segmenter?.markBackendRunning()
                } else {
                    self.segmenter?.markBackendUnavailable(
                        reason: report.reason ?? "system_audio_capture_failed"
                    )
                    deliveryGate.invalidate()
                    self.startAllowed = false
                    if let current = self.monitor {
                        self.monitor = nil
                        self.addStopping(current)
                        _ = self.startCleanupTask(pendingStart: nil)
                    }
                }
                self.onStatusChanged()
            }
        )
        guard let next = factory(callbacks) else {
            report = .init(supported: false, requested: true, state: .unavailable,
                           permission: .notDetermined, reason: "macos_13_required")
            segmenter.markBackendUnavailable(reason: "macos_13_required")
            return
        }
        starting = next
        report = .init(supported: true, requested: true, state: .unavailable,
                       permission: .granted, reason: "system_audio_starting")
        startTask = Task { [weak self] in
            let result = await next.start()
            guard let self else {
                deliveryGate.invalidate()
                _ = await next.stop()
                return
            }
            guard !Task.isCancelled, self.generation == token, self.requested else {
                deliveryGate.invalidate()
                return
            }
            self.starting = nil
            self.report = result
            self.monitor = result.state == .running ? next : nil
            if result.state == .running {
                self.segmenter?.markBackendRunning()
            } else {
                self.segmenter?.markBackendUnavailable(
                    reason: result.reason ?? "system_audio_capture_start_failed"
                )
                deliveryGate.invalidate()
                self.startAllowed = false
                if !(await next.stop()) { self.addStopping(next) }
            }
            self.startTask = nil
            self.onStatusChanged()
        }
    }

    @discardableResult
    func stop() -> Task<Bool, Never>? {
        requested = false
        shouldRun = false
        startAllowed = false
        segmenter?.stop()
        segmenter = nil
        let cleanup = stopCapture()
        report = .init(
            supported: supported,
            requested: false,
            state: .disabled,
            permission: CGPreflightScreenCaptureAccess() ? .granted : .notDetermined
        )
        return cleanup
    }

    @discardableResult
    private func stopCapture() -> Task<Bool, Never>? {
        generation += 1
        gate?.invalidate()
        gate = nil
        // A shutdown that follows an earlier disable must join the cleanup
        // already in flight. Returning nil here would let stopAll finish and
        // terminate the helper while HAL resources were still being removed.
        if let stopTask { return stopTask }
        let pendingStart = startTask
        pendingStart?.cancel()
        startTask = nil
        let current = monitor
        let inProgress = starting
        monitor = nil
        starting = nil
        if let inProgress { addStopping(inProgress) }
        if let current { addStopping(current) }
        return startCleanupTask(pendingStart: pendingStart)
    }

    private func addStopping(_ candidate: any SystemAudioMonitoring) {
        guard !stopping.contains(where: { $0 === candidate }) else { return }
        stopping.append(candidate)
    }

    @discardableResult
    private func startCleanupTask(
        pendingStart: Task<Void, Never>?
    ) -> Task<Bool, Never>? {
        guard stopTask == nil else { return nil }
        let targets = stopping
        guard !targets.isEmpty || pendingStart != nil else { return nil }
        let task = Task { [weak self] () -> Bool in
            var failed: [any SystemAudioMonitoring] = []
            // Stop a backend whose start is suspended before awaiting the start
            // task. The cancelled start observes the generation change and can
            // no longer publish or create a replacement backend.
            for target in targets where !(await target.stop()) {
                failed.append(target)
            }
            if let pendingStart { await pendingStart.value }
            if pendingStart != nil, !failed.isEmpty {
                var stillFailed: [any SystemAudioMonitoring] = []
                for target in failed where !(await target.stop()) {
                    stillFailed.append(target)
                }
                failed = stillFailed
            }
            guard let self else { return failed.isEmpty }
            self.stopping = failed
            self.stopTask = nil
            if failed.isEmpty {
                if self.shouldRun, self.startAllowed { self.startIfPossible() }
            } else {
                self.startAllowed = false
                self.report = .init(
                    supported: self.supported,
                    requested: self.requested,
                    state: .failed,
                    permission: .notDetermined,
                    reason: "system_audio_cleanup_failed"
                )
                self.onStatusChanged()
            }
            return failed.isEmpty
        }
        stopTask = task
        return task
    }
}

enum SystemAudioBackendKind: Equatable, Sendable {
    case unsupported
    case screenCaptureKit
    case coreAudioProcessTap
}

/// Keep the deployment target at macOS 12.3 while choosing the best runtime
/// backend. Process taps are weak-linked behind their 14.2 availability guard;
/// macOS 13 keeps the existing ScreenCaptureKit implementation.
func selectSystemAudioBackend(for version: OperatingSystemVersion) -> SystemAudioBackendKind {
    if version.majorVersion > 14
        || (version.majorVersion == 14 && version.minorVersion >= 2) {
        return .coreAudioProcessTap
    }
    if version.majorVersion >= 13 { return .screenCaptureKit }
    return .unsupported
}

@MainActor
func makeSystemAudioMonitor(
    callbacks: SystemAudioCallbacks,
    operatingSystemVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion
) -> (any SystemAudioMonitoring)? {
    switch selectSystemAudioBackend(for: operatingSystemVersion) {
    case .coreAudioProcessTap:
        if #available(macOS 14.2, *) {
            return TapWithScreenCaptureFallback(
                primary: CoreAudioProcessTapMonitor(callbacks: callbacks),
                makeFallback: { SystemAudioMonitor(callbacks: callbacks) }
            )
        }
        return nil
    case .screenCaptureKit:
        if #available(macOS 13.0, *) {
            return SystemAudioMonitor(callbacks: callbacks)
        }
        return nil
    case .unsupported:
        return nil
    }
}

private final class SystemAudioModuleGate: @unchecked Sendable {
    private let lock = NSLock()
    private var active = true
    private let onSamples: @Sendable (SystemAudioPCMBlock) -> Void
    private let onCaptureGap: @Sendable (SystemAudioCaptureGap) -> Void

    init(onSamples: @escaping @Sendable (SystemAudioPCMBlock) -> Void,
         onCaptureGap: @escaping @Sendable (SystemAudioCaptureGap) -> Void) {
        self.onSamples = onSamples
        self.onCaptureGap = onCaptureGap
    }

    private var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return active
    }

    func deliver(_ block: SystemAudioPCMBlock) { if isActive { onSamples(block) } }
    func deliver(_ gap: SystemAudioCaptureGap) { if isActive { onCaptureGap(gap) } }
    func invalidate() { lock.lock(); active = false; lock.unlock() }
}
