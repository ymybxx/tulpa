import XCTest
@testable import PerceptionCore

final class CapabilityModulesTests: XCTestCase {
    @MainActor
    func testComputerHistoryIsIndependentDefaultOffAndStopsCleanly() {
        let monitor = ControlledComputerHistoryMonitor()
        var delivered: [ComputerHistoryEvent] = []
        var statusChangeCount = 0
        let modules = LazyCapabilityModules(
            onComputerHistoryEvent: { delivered.append($0) },
            onStatusChanged: { statusChangeCount += 1 },
            computerHistoryTrustCheck: { true },
            makeComputerHistoryMonitor: { onEvent, onSuppressed, onStatus in
                monitor.onEvent = onEvent
                monitor.onSuppressed = onSuppressed
                monitor.onStatusChanged = onStatus
                return monitor
            }
        )

        XCTAssertEqual(modules.report().computerHistory.state, .disabled)
        _ = modules.configure(PerceptionModuleSelection(
            screen: false,
            accessibility: false,
            camera: false,
            computerHistory: true
        ))

        XCTAssertEqual(monitor.startCallCount, 1)
        XCTAssertEqual(modules.report().computerHistory.state, .running)
        let statusCountBeforeDegrade = statusChangeCount
        monitor.inputReport = ComputerHistoryInputReport(
            accessibilityPermission: .granted,
            listenEventPermission: .denied,
            eventTapState: .permissionRequired,
            mouse: .init(state: .unavailable, receivedCount: 0, lastEventAt: nil),
            keyboard: .init(state: .unavailable, receivedCount: 0, lastEventAt: nil),
            tapDisabledCount: 0
        )
        monitor.emitInputStatusChanged()
        XCTAssertGreaterThan(statusChangeCount, statusCountBeforeDegrade)
        XCTAssertEqual(modules.report().computerHistory.state, .unavailable)
        XCTAssertEqual(
            modules.report().computerHistory.reason,
            "input_monitoring_permission_required"
        )
        monitor.emit(ComputerHistoryEvent(
            kind: .keyboardTextInput,
            timestamp: 1,
            app: .init(bundleIdentifier: nil, name: nil),
            window: .init(title: nil, url: nil),
            keyboard: .init(text: "a")
        ))
        XCTAssertEqual(delivered.count, 1)

        _ = modules.configure(.disabled)
        XCTAssertEqual(monitor.stopCallCount, 1)
        XCTAssertEqual(modules.report().computerHistory.state, .disabled)
    }

    @MainActor
    func testStaleScreenStartIsStoppedWithoutReportingRunning() async {
        let monitor = ControlledScreenMonitor()
        let recorder = ScreenObservationRecorder()
        var modules: LazyCapabilityModules?
        var reportedStates: [CapabilityState] = []
        modules = LazyCapabilityModules(
            onScreenObservation: { recorder.record($0) },
            onStatusChanged: {
                if let state = modules?.report().screen.state {
                    reportedStates.append(state)
                }
            },
            requestScreenPermission: { true },
            makeScreenMonitor: { callbacks in
                monitor.onObservation = callbacks.onObservation
                monitor.screenTextPrivacyGate = callbacks.screenTextPrivacyGate
                monitor.onTextObservation = callbacks.onTextObservation
                monitor.onStatus = callbacks.onStatus
                return monitor
            }
        )
        let enabled = PerceptionModuleSelection(
            screen: true,
            accessibility: false,
            camera: false
        )

        _ = modules?.configure(enabled)
        await waitUntil { monitor.startCallCount == 1 }
        _ = modules?.configure(.disabled)
        monitor.emitObservation(Self.screenObservation())
        monitor.completeStart(.init(state: .running, displayID: 1))
        await waitUntil { monitor.stopCallCount == 1 }

        XCTAssertEqual(monitor.stopCallCount, 1)
        XCTAssertEqual(recorder.count, 0)
        XCTAssertEqual(modules?.report().screen.state, .disabled)
        XCTAssertFalse(reportedStates.contains(.running))
    }

    @MainActor
    func testScreenTextUsesSharedPrivacyContextAndStopsDeliveringWhenDisabled() async throws {
        let monitor = ControlledScreenMonitor()
        let textRecorder = ScreenTextObservationRecorder()
        let privacyContext = ScreenTextPrivacyContextStore()
        let modules = LazyCapabilityModules(
            screenTextPrivacyContext: privacyContext,
            onScreenTextObservation: { textRecorder.record($0) },
            requestScreenPermission: { true },
            makeScreenMonitor: { callbacks in
                monitor.onObservation = callbacks.onObservation
                monitor.screenTextPrivacyGate = callbacks.screenTextPrivacyGate
                monitor.onTextObservation = callbacks.onTextObservation
                monitor.onStatus = callbacks.onStatus
                return monitor
            }
        )
        let enabled = PerceptionModuleSelection(
            screen: true,
            accessibility: false,
            camera: false
        )

        _ = modules.configure(enabled)
        await waitUntil { monitor.startCallCount == 1 }
        monitor.completeStart(.init(state: .running, displayID: 1))
        await waitUntil { modules.report().screen.state == .running }

        let observedAt = Date().timeIntervalSince1970
        privacyContext.updateWorkspace(
            FrontmostApplication(
                bundleIdentifier: "com.example.editor",
                name: "Editor",
                processIdentifier: 7
            ),
            invalidateAccessibility: true
        )
        privacyContext.updateAccessibility(AccessibilityObservation(
            revision: 1,
            kind: "accessibility_snapshot",
            observedAt: observedAt,
            processIdentifier: 7,
            bundleIdentifier: "com.example.editor",
            applicationName: "Editor",
            windowTitle: "Document",
            windowFrame: AccessibilityWindowFrame(
                x: 10,
                y: 20,
                width: 800,
                height: 600
            ),
            focusedRole: nil,
            focusedSubrole: nil,
            focusedTitle: nil,
            focusedValue: nil,
            contextTitles: [],
            secureField: false,
            nodesInspected: 0,
            truncated: false
        ))
        let candidate = ScreenTextPrivacyContext(
            revision: 2,
            observedAt: observedAt,
            displayID: 1
        )
        let authorization = try XCTUnwrap(monitor.authorize(candidate))
        monitor.emitText(ScreenTextObservation(
            revision: 2,
            observedAt: observedAt,
            displayID: 1,
            text: "Visible text",
            applicationContext: authorization
        ))
        XCTAssertEqual(textRecorder.count, 1)
        XCTAssertNotNil(modules.visionCandidateBroker.store(
            jpeg: Data([1]),
            privacyContext: candidate,
            applicationContext: authorization
        ))
        XCTAssertNotNil(modules.visionCandidateBroker.currentMetadata())

        _ = modules.configure(.disabled)
        monitor.emitText(ScreenTextObservation(
            revision: 3,
            observedAt: observedAt,
            displayID: 1,
            text: "Stale text",
            applicationContext: authorization
        ))
        XCTAssertEqual(textRecorder.count, 1)
        XCTAssertNil(privacyContext.authorization(for: candidate))
        XCTAssertNil(modules.visionCandidateBroker.currentMetadata())
    }

    @MainActor
    func testScreenStartReappliesLatestEnhancedVisionGate() async {
        let monitor = ControlledScreenMonitor()
        let modules = LazyCapabilityModules(
            requestScreenPermission: { true },
            makeScreenMonitor: { callbacks in
                monitor.setEnhancedVisionEnabled(callbacks.enhancedVisionEnabled)
                monitor.setLegacyScreenAnalysisEnabled(
                    callbacks.legacyScreenAnalysisEnabled
                )
                monitor.setExcludeHostApplicationFromCapture(
                    callbacks.excludeHostApplicationFromCapture
                )
                monitor.onStatus = callbacks.onStatus
                return monitor
            },
            // Keep this test independent from real AX permission prompts while
            // exercising the same in-flight screen-start race.
            enhancedVisionActivation: { $0.enhancedVision }
        )
        let starting = PerceptionModuleSelection(
            screen: true,
            accessibility: false,
            camera: false,
            enhancedVision: true
        )
        let changedWhileStarting = PerceptionModuleSelection(
            screen: true,
            accessibility: false,
            camera: false,
            enhancedVision: false,
            legacyScreenAnalysis: false,
            excludeHostApplicationFromCapture: true
        )

        _ = modules.configure(starting)
        await waitUntil { monitor.startCallCount == 1 }
        XCTAssertTrue(monitor.enhancedVisionEnabled)
        _ = modules.configure(changedWhileStarting)
        XCTAssertFalse(monitor.enhancedVisionEnabled)
        XCTAssertFalse(monitor.legacyScreenAnalysisEnabled)
        XCTAssertTrue(monitor.excludeHostApplicationFromCapture)

        monitor.completeStart(.init(state: .running, displayID: 1))
        await waitUntil { modules.report().screen.state == .running }

        XCTAssertFalse(monitor.enhancedVisionEnabled)
        XCTAssertFalse(monitor.legacyScreenAnalysisEnabled)
        XCTAssertTrue(monitor.excludeHostApplicationFromCapture)
        _ = modules.configure(PerceptionModuleSelection(
            screen: true,
            accessibility: false,
            camera: false,
            legacyScreenAnalysis: false,
            excludeHostApplicationFromCapture: false
        ))
        XCTAssertFalse(monitor.excludeHostApplicationFromCapture)
        XCTAssertEqual(monitor.startCallCount, 1)
        XCTAssertEqual(monitor.stopCallCount, 0)
        await modules.stopAll()
    }

    @MainActor
    func testEnvironmentFrameIntervalIsSetBeforeStartAndExplicitUpdatesRequestTick() async {
        let monitor = ControlledScreenMonitor()
        let modules = LazyCapabilityModules(
            requestScreenPermission: { true },
            makeScreenMonitor: { callbacks in
                monitor.onStatus = callbacks.onStatus
                return monitor
            }
        )
        let enabled = PerceptionModuleSelection(
            screen: true,
            accessibility: false,
            camera: false,
            legacyScreenAnalysis: false
        )

        _ = modules.configure(enabled, environmentFrameIntervalMs: 5_000)
        await waitUntil { monitor.startCallCount == 1 }
        XCTAssertEqual(monitor.environmentFrameIntervals, [5_000])
        XCTAssertEqual(monitor.environmentFrameImmediateTicks, [false])
        monitor.completeStart(.init(state: .running, displayID: 1))
        await waitUntil { modules.report().screen.state == .running }

        _ = modules.configure(enabled, environmentFrameIntervalMs: 1_000)
        _ = modules.configure(enabled)
        _ = modules.configure(enabled, environmentFrameIntervalMs: 1_000)
        XCTAssertEqual(monitor.environmentFrameIntervals, [5_000, 1_000, 1_000])
        XCTAssertEqual(monitor.environmentFrameImmediateTicks, [false, true, true])
        await modules.stopAll()
    }

    private static func screenObservation() -> ScreenChangeObservation {
        ScreenChangeObservation(
            displayID: 1,
            frameWidth: 100,
            frameHeight: 100,
            evaluation: ScreenChangeEvaluation(
                revision: 1,
                timestamp: 1,
                dirtyAreaRatio: 0.5,
                dirtyRectCount: 1,
                state: .changing,
                stateChanged: false,
                significantChange: true
            )
        )
    }

    @MainActor
    private func waitUntil(
        _ condition: () -> Bool,
        iterations: Int = 20
    ) async {
        for _ in 0 ..< iterations {
            if condition() { return }
            await Task.yield()
        }
    }
}

@MainActor
private final class ControlledComputerHistoryMonitor: ComputerHistoryMonitoring {
    var onEvent: ComputerHistoryMonitor.EventHandler?
    var onSuppressed: ComputerHistoryMonitor.SuppressionHandler?
    var onStatusChanged: ComputerHistoryMonitor.InputStatusHandler?
    private(set) var startCallCount = 0
    private(set) var stopCallCount = 0
    var inputReport = ComputerHistoryInputReport(
        accessibilityPermission: .granted,
        listenEventPermission: .granted,
        eventTapState: .running,
        mouse: .init(state: .running, receivedCount: 0, lastEventAt: nil),
        keyboard: .init(state: .running, receivedCount: 0, lastEventAt: nil),
        tapDisabledCount: 0
    )

    func start() -> CapabilityReport {
        startCallCount += 1
        return CapabilityReport(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        )
    }

    func stop() {
        stopCallCount += 1
    }

    func currentReport() -> CapabilityReport {
        let active = startCallCount > stopCallCount
        let inputRunning = inputReport.eventTapState == .running
        return CapabilityReport(
            supported: true,
            requested: active,
            state: active ? (inputRunning ? .running : .unavailable) : .disabled,
            permission: .granted,
            reason: active && !inputRunning
                ? "input_monitoring_permission_required"
                : nil,
            computerHistoryInput: inputReport
        )
    }

    func emit(_ event: ComputerHistoryEvent) {
        onEvent?(event)
    }

    func emitInputStatusChanged() {
        onStatusChanged?()
    }
}

@MainActor
private final class ControlledScreenMonitor: ScreenMonitoring {
    private var continuation: CheckedContinuation<ScreenMonitorStatus, Never>?
    var onStatus: ScreenMonitor.StatusHandler?
    var onObservation: (@Sendable (ScreenChangeObservation) -> Void)?
    var screenTextPrivacyGate: ScreenTextPrivacyGate?
    var onTextObservation: (@Sendable (ScreenTextObservation) -> Void)?
    private(set) var startCallCount = 0
    private(set) var stopCallCount = 0
    private(set) var enhancedVisionEnabled = false
    private(set) var legacyScreenAnalysisEnabled = true
    private(set) var excludeHostApplicationFromCapture = false
    private(set) var environmentFrameIntervals: [Int] = []
    private(set) var environmentFrameImmediateTicks: [Bool] = []

    func start() async -> ScreenMonitorStatus {
        startCallCount += 1
        return await withCheckedContinuation { continuation = $0 }
    }

    func stop() async {
        stopCallCount += 1
    }

    func setEnhancedVisionEnabled(_ enabled: Bool) {
        enhancedVisionEnabled = enabled
    }

    func setLegacyScreenAnalysisEnabled(_ enabled: Bool) {
        legacyScreenAnalysisEnabled = enabled
    }

    func setExcludeHostApplicationFromCapture(_ enabled: Bool) {
        excludeHostApplicationFromCapture = enabled
    }

    func setEnvironmentFrameInterval(milliseconds: Int, tickImmediately: Bool) {
        environmentFrameIntervals.append(milliseconds)
        environmentFrameImmediateTicks.append(tickImmediately)
    }

    func completeStart(_ status: ScreenMonitorStatus) {
        onStatus?(status)
        continuation?.resume(returning: status)
        continuation = nil
    }

    func emitObservation(_ observation: ScreenChangeObservation) {
        onObservation?(observation)
    }

    func authorize(
        _ candidate: ScreenTextPrivacyContext
    ) -> ScreenTextApplicationContext? {
        screenTextPrivacyGate?(candidate)
    }

    func emitText(_ observation: ScreenTextObservation) {
        onTextObservation?(observation)
    }
}

private final class ScreenObservationRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var observations: [ScreenChangeObservation] = []

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return observations.count
    }

    func record(_ observation: ScreenChangeObservation) {
        lock.lock()
        observations.append(observation)
        lock.unlock()
    }
}

private final class ScreenTextObservationRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var observations: [ScreenTextObservation] = []

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return observations.count
    }

    func record(_ observation: ScreenTextObservation) {
        lock.lock()
        observations.append(observation)
        lock.unlock()
    }
}
