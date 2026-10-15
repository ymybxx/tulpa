import AppKit
import XCTest
@testable import PerceptionCore

final class PerceptionRuntimeTests: XCTestCase {
    @MainActor
    func testWorkspaceIsDisabledUntilExplicitlyConfiguredAndStopsObserving() async throws {
        let pipe = Pipe()
        let runtime = PerceptionRuntime(
            writer: PerceptionEventWriter(output: pipe.fileHandleForWriting),
            onShutdown: {}
        )

        runtime.start()
        runtime.handle(line: Self.configureLine(workspace: true, requestId: "enable"))
        runtime.handle(line: Self.configureLine(workspace: false, requestId: "disable"))
        NSWorkspace.shared.notificationCenter.post(
            name: NSWorkspace.didActivateApplicationNotification,
            object: nil
        )
        await runtime.shutdown()
        try pipe.fileHandleForWriting.close()

        let envelopes = try Self.decodeEnvelopes(pipe.fileHandleForReading.readDataToEndOfFile())
        let hello = try XCTUnwrap(envelopes.first { $0["event"] as? String == "hello" })
        let helloWorkspace = try Self.workspaceCapability(from: hello)
        XCTAssertEqual(helloWorkspace["requested"] as? Bool, false)
        XCTAssertEqual(helloWorkspace["state"] as? String, "disabled")

        let workspaceObservations = envelopes.filter { envelope in
            guard envelope["event"] as? String == "observation",
                  let body = envelope["body"] as? [String: Any]
            else { return false }
            return body["source"] as? String == "workspace"
        }
        XCTAssertEqual(workspaceObservations.count, 1)

        let disableStatus = try XCTUnwrap(envelopes.last { envelope in
            envelope["event"] as? String == "status"
                && envelope["requestId"] as? String == "disable"
        })
        let disabledWorkspace = try Self.workspaceCapability(from: disableStatus)
        XCTAssertEqual(disabledWorkspace["requested"] as? Bool, false)
        XCTAssertEqual(disabledWorkspace["state"] as? String, "disabled")
    }

    @MainActor
    func testRuntimeAndModulesShareForegroundPrivacyContextAndAppSwitchClearsAX() async throws {
        let pipe = Pipe()
        let privacyContext = ScreenTextPrivacyContextStore()
        let modules = LazyCapabilityModules(
            screenTextPrivacyContext: privacyContext
        )
        let monitor = ControlledRuntimeWorkspaceMonitor()
        let runtime = PerceptionRuntime(
            writer: PerceptionEventWriter(output: pipe.fileHandleForWriting),
            modules: modules,
            makeWorkspaceMonitor: { handler in
                monitor.onObservation = handler
                return monitor
            },
            onShutdown: {}
        )
        runtime.start()
        runtime.handle(line: Self.configureLine(workspace: true, requestId: "enable"))

        let observedAt = Date().timeIntervalSince1970
        monitor.emit(
            kind: "foreground_app_changed",
            application: FrontmostApplication(
                bundleIdentifier: "com.example.editor",
                name: "Editor",
                processIdentifier: 7
            )
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
            revision: 1,
            observedAt: observedAt,
            displayID: 1
        )
        let authorization = try XCTUnwrap(
            privacyContext.authorization(for: candidate)
        )
        XCTAssertNotNil(modules.visionCandidateBroker.store(
            jpeg: Data([1]),
            privacyContext: candidate,
            applicationContext: authorization
        ))

        monitor.emit(
            kind: "foreground_app_changed",
            application: FrontmostApplication(
                bundleIdentifier: "com.example.other",
                name: "Other",
                processIdentifier: 8
            )
        )
        XCTAssertNil(privacyContext.authorization(for: candidate))
        XCTAssertNil(modules.visionCandidateBroker.currentMetadata())

        await runtime.shutdown()
        try pipe.fileHandleForWriting.close()
        XCTAssertTrue(monitor.stopped)
    }

    @MainActor
    func testVisionCandidateClaimReturnsImageOnceThenFixedUnavailable() async throws {
        let pipe = Pipe()
        let privacyContext = ScreenTextPrivacyContextStore()
        let broker = VisionCandidateBroker()
        let modules = LazyCapabilityModules(
            screenTextPrivacyContext: privacyContext,
            visionCandidateBroker: broker
        )
        let runtime = PerceptionRuntime(
            writer: PerceptionEventWriter(output: pipe.fileHandleForWriting),
            modules: modules,
            onShutdown: {}
        )
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
        let candidateContext = ScreenTextPrivacyContext(
            revision: 2,
            observedAt: observedAt,
            displayID: 1
        )
        let appContext = try XCTUnwrap(
            privacyContext.authorization(for: candidateContext)
        )
        let metadata = try XCTUnwrap(broker.store(
            jpeg: Data([1, 2, 3]),
            privacyContext: candidateContext,
            applicationContext: appContext
        ))

        runtime.start()
        runtime.handle(line: Self.claimLine(
            candidateId: metadata.candidateId,
            requestId: "claim-success"
        ))
        runtime.handle(line: Self.claimLine(
            candidateId: metadata.candidateId,
            requestId: "claim-consumed"
        ))
        runtime.handle(line: Self.claimLine(
            candidateId: "not-a-uuid",
            requestId: "claim-invalid"
        ))
        await runtime.shutdown()
        try pipe.fileHandleForWriting.close()

        let envelopes = try Self.decodeEnvelopes(
            pipe.fileHandleForReading.readDataToEndOfFile()
        )
        let success = try XCTUnwrap(envelopes.first {
            $0["event"] as? String == "vision_candidate"
        })
        let successBody = try XCTUnwrap(success["body"] as? [String: Any])
        XCTAssertEqual(success["requestId"] as? String, "claim-success")
        XCTAssertEqual(successBody["candidateId"] as? String, metadata.candidateId)
        XCTAssertEqual(successBody["mimeType"] as? String, "image/jpeg")
        XCTAssertEqual(successBody["jpegBase64"] as? String, "AQID")

        let unavailable = envelopes.filter {
            $0["event"] as? String == "vision_candidate_unavailable"
        }
        XCTAssertEqual(unavailable.count, 2)
        for envelope in unavailable {
            let body = try XCTUnwrap(envelope["body"] as? [String: Any])
            XCTAssertEqual(Set(body.keys), Set(["code"]))
            XCTAssertEqual(body["code"] as? String, "unavailable")
        }
    }

    @MainActor
    func testConfigureRoutesFrameIntervalAndRejectsOutOfRange() async throws {
        let pipe = Pipe()
        let monitor = ImmediateRuntimeScreenMonitor()
        let modules = LazyCapabilityModules(
            requestScreenPermission: { true },
            makeScreenMonitor: { callbacks in
                monitor.onStatus = callbacks.onStatus
                return monitor
            }
        )
        let runtime = PerceptionRuntime(
            writer: PerceptionEventWriter(output: pipe.fileHandleForWriting),
            modules: modules,
            onShutdown: {}
        )
        runtime.start()
        runtime.handle(line: """
            {"protocolVersion":1,"requestId":"initial","command":"configure","environmentFrameIntervalMs":5000,"modules":{"screen":true,"accessibility":false,"camera":false,"legacyScreenAnalysis":false}}
            """)
        for _ in 0 ..< 20 where modules.report().screen.state != .running {
            await Task.yield()
        }
        XCTAssertEqual(monitor.environmentFrameIntervals, [5_000])
        XCTAssertEqual(monitor.environmentFrameImmediateTicks, [false])

        runtime.handle(line: """
            {"protocolVersion":1,"requestId":"speech-start","command":"configure","environmentFrameIntervalMs":1000,"modules":{"screen":true,"accessibility":false,"camera":false,"legacyScreenAnalysis":false}}
            """)
        runtime.handle(line: """
            {"protocolVersion":1,"requestId":"invalid","command":"configure","environmentFrameIntervalMs":249,"modules":{"screen":false,"accessibility":false,"camera":false}}
            """)
        XCTAssertEqual(monitor.environmentFrameIntervals, [5_000, 1_000])
        XCTAssertEqual(monitor.environmentFrameImmediateTicks, [false, true])
        XCTAssertTrue(modules.moduleSelection.screen)

        await runtime.shutdown()
        try pipe.fileHandleForWriting.close()
        let envelopes = try Self.decodeEnvelopes(pipe.fileHandleForReading.readDataToEndOfFile())
        let error = try XCTUnwrap(envelopes.first {
            $0["event"] as? String == "error" && $0["requestId"] as? String == "invalid"
        })
        let body = try XCTUnwrap(error["body"] as? [String: Any])
        XCTAssertEqual(body["code"] as? String, "invalid_environment_frame_interval")
        XCTAssertFalse(envelopes.contains {
            $0["event"] as? String == "ack" && $0["requestId"] as? String == "invalid"
        })
    }

    @MainActor
    func testShutdownWaitsForScreenCaptureBeforeInvokingExit() async {
        let pipe = Pipe()
        let monitor = ImmediateRuntimeScreenMonitor()
        let modules = LazyCapabilityModules(
            requestScreenPermission: { true },
            makeScreenMonitor: { callbacks in
                monitor.onStatus = callbacks.onStatus
                return monitor
            }
        )
        var exited = false
        let runtime = PerceptionRuntime(
            writer: PerceptionEventWriter(output: pipe.fileHandleForWriting),
            modules: modules,
            onShutdown: { exited = true }
        )
        runtime.start()
        runtime.handle(line: Self.configureLine(
            workspace: false,
            requestId: "enable-screen",
            screen: true
        ))
        for _ in 0..<100 where modules.report().screen.state != .running {
            await Task.yield()
        }
        monitor.blockStop = true

        let shutdown = Task { await runtime.shutdown() }
        for _ in 0..<100 where monitor.stopContinuation == nil { await Task.yield() }
        XCTAssertFalse(exited)
        monitor.completeStop()
        await shutdown.value
        XCTAssertTrue(exited)
        try? pipe.fileHandleForWriting.close()
    }

    private static func configureLine(
        workspace: Bool,
        requestId: String,
        screen: Bool = false
    ) -> String {
        """
        {"protocolVersion":1,"requestId":"\(requestId)","command":"configure","modules":{"workspace":\(workspace),"screen":\(screen),"accessibility":false,"camera":false}}
        """
    }

    private static func claimLine(candidateId: String, requestId: String) -> String {
        """
        {"protocolVersion":1,"requestId":"\(requestId)","command":"claim_vision_candidate","candidateId":"\(candidateId)"}
        """
    }

    private static func decodeEnvelopes(_ data: Data) throws -> [[String: Any]] {
        try XCTUnwrap(String(data: data, encoding: .utf8))
            .split(separator: "\n")
            .map { line in
                try XCTUnwrap(
                    JSONSerialization.jsonObject(with: Data(line.utf8))
                        as? [String: Any]
                )
            }
    }

    private static func workspaceCapability(
        from envelope: [String: Any]
    ) throws -> [String: Any] {
        let body = try XCTUnwrap(envelope["body"] as? [String: Any])
        let capabilities = try XCTUnwrap(body["capabilities"] as? [String: Any])
        return try XCTUnwrap(capabilities["workspace"] as? [String: Any])
    }
}

@MainActor
private final class ControlledRuntimeWorkspaceMonitor: WorkspaceMonitoring {
    var onObservation: WorkspaceMonitor.ObservationHandler?
    private(set) var stopped = false

    func start() {}

    func stop() {
        stopped = true
    }

    func emit(kind: String, application: FrontmostApplication?) {
        onObservation?(kind, application)
    }
}

@MainActor
private final class ImmediateRuntimeScreenMonitor: ScreenMonitoring {
    var onStatus: ScreenMonitor.StatusHandler?
    private(set) var environmentFrameIntervals: [Int] = []
    private(set) var environmentFrameImmediateTicks: [Bool] = []
    var blockStop = false
    var stopContinuation: CheckedContinuation<Void, Never>?

    func start() async -> ScreenMonitorStatus {
        .init(state: .running, displayID: 1)
    }

    func stop() async {
        if blockStop {
            await withCheckedContinuation { stopContinuation = $0 }
        }
    }

    func completeStop() {
        blockStop = false
        stopContinuation?.resume()
        stopContinuation = nil
    }

    func setEnhancedVisionEnabled(_ enabled: Bool) {}

    func setLegacyScreenAnalysisEnabled(_ enabled: Bool) {}

    func setExcludeHostApplicationFromCapture(_ enabled: Bool) {}

    func setEnvironmentFrameInterval(milliseconds: Int, tickImmediately: Bool) {
        environmentFrameIntervals.append(milliseconds)
        environmentFrameImmediateTicks.append(tickImmediately)
    }
}
