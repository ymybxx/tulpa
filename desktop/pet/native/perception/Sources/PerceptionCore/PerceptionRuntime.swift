import AppKit
import CoreFoundation
import Foundation

@MainActor
public final class PerceptionRuntime {
    public typealias WorkspaceMonitorFactory = @MainActor (
        @escaping WorkspaceMonitor.ObservationHandler
    ) -> any WorkspaceMonitoring

    private let writer: PerceptionEventWriter
    private let providedModules: LazyCapabilityModules?
    private let screenTextPrivacyContext: ScreenTextPrivacyContextStore
    private let visionCandidateBroker: VisionCandidateBroker
    private let environmentFrameBroker: EnvironmentFrameBroker
    private let makeWorkspaceMonitor: WorkspaceMonitorFactory
    private let onShutdown: @MainActor () -> Void
    private lazy var modules: LazyCapabilityModules = makeModules()
    private var workspaceMonitor: (any WorkspaceMonitoring)?
    private var workspaceRequested = false
    private var observationRevision: UInt64 = 0
    private var started = false
    private var stopped = false

    public init(
        writer: PerceptionEventWriter = PerceptionEventWriter(),
        modules: LazyCapabilityModules? = nil,
        makeWorkspaceMonitor: @escaping WorkspaceMonitorFactory = { handler in
            WorkspaceMonitor(onObservation: handler)
        },
        onShutdown: @escaping @MainActor () -> Void = {
            CFRunLoopStop(CFRunLoopGetMain())
        }
    ) {
        self.writer = writer
        providedModules = modules
        screenTextPrivacyContext = modules?.screenTextPrivacyContext ?? .init()
        visionCandidateBroker = modules?.visionCandidateBroker ?? .init()
        environmentFrameBroker = modules?.environmentFrameBroker ?? .init()
        self.makeWorkspaceMonitor = makeWorkspaceMonitor
        self.onShutdown = onShutdown
    }

    public func start() {
        guard !started, !stopped else { return }
        started = true
        writer.send(
            event: "hello",
            body: HelperHello(
                helperVersion: "0.1.0",
                minimumMacOS: "12.3",
                capabilities: capabilities()
            )
        )
        sendStatus()
    }

    public func handle(line: String) {
        do {
            let command = try PerceptionProtocolCodec.decodeCommand(line)
            guard command.protocolVersion == perceptionProtocolVersion else {
                writer.send(
                    event: "error",
                    requestId: command.requestId,
                    body: HelperError(
                        code: "unsupported_protocol_version",
                        message: "Expected protocolVersion 1",
                        fatal: false
                    )
                )
                return
            }
            switch command.command {
            case "configure":
                guard let selection = command.modules else {
                    sendCommandError(
                        requestId: command.requestId,
                        code: "missing_modules",
                        message: "configure requires modules"
                    )
                    return
                }
                if let interval = command.environmentFrameIntervalMs,
                   !EnvironmentFrameSamplingLoop.allowedIntervalMilliseconds.contains(interval) {
                    sendCommandError(
                        requestId: command.requestId,
                        code: "invalid_environment_frame_interval",
                        message: "environmentFrameIntervalMs must be between 250 and 60000"
                    )
                    return
                }
                workspaceRequested = selection.workspace
                configureWorkspace(enabled: selection.workspace)
                _ = modules.configure(
                    selection,
                    environmentFrameIntervalMs: command.environmentFrameIntervalMs
                )
                writer.send(
                    event: "ack",
                    requestId: command.requestId,
                    body: HelperAcknowledgement(command: command.command)
                )
                sendStatus(requestId: command.requestId)
            case "status":
                sendStatus(requestId: command.requestId)
            case "ping":
                writer.send(
                    event: "ack",
                    requestId: command.requestId,
                    body: HelperAcknowledgement(command: command.command)
                )
            case "claim_vision_candidate":
                claimVisionCandidate(command)
            case "claim_environment_frame":
                claimEnvironmentFrame(command)
            case "cut_environment_window":
                if let timestamp = command.boundaryAt, timestamp.isFinite {
                    modules.cutEnvironmentWindow(at: timestamp)
                }
                writer.send(event: "ack", requestId: command.requestId,
                            body: HelperAcknowledgement(command: command.command))
            case "shutdown":
                writer.send(
                    event: "ack",
                    requestId: command.requestId,
                    body: HelperAcknowledgement(command: command.command)
                )
                Task { await shutdown() }
            default:
                sendCommandError(
                    requestId: command.requestId,
                    code: "unknown_command",
                    message: "Unknown command: \(command.command)"
                )
            }
        } catch {
            sendCommandError(
                requestId: nil,
                code: "invalid_command",
                message: "Command must be one JSON object per line"
            )
        }
    }

    public func shutdown() async {
        guard !stopped else { return }
        stopped = true
        workspaceRequested = false
        workspaceMonitor?.stop()
        workspaceMonitor = nil
        screenTextPrivacyContext.clearWorkspace()
        visionCandidateBroker.clear()
        environmentFrameBroker.clear()
        await modules.stopAll()
        // The writer is asynchronous so protocol backpressure never blocks
        // MainActor. Preserve the shutdown acknowledgement and preceding
        // control events before the helper exits; Electron still enforces its
        // existing three-second hard-stop boundary for a permanently wedged pipe.
        await writer.flush()
        onShutdown()
    }

    private func sendObservation(kind: String, application: FrontmostApplication?) {
        let observedAt = Date().timeIntervalSince1970
        visionCandidateBroker.clear()
        screenTextPrivacyContext.updateWorkspace(
            application,
            invalidateAccessibility: kind != "foreground_app_heartbeat"
        )
        guard workspaceRequested, workspaceMonitor != nil, !stopped else { return }
        observationRevision += 1
        writer.send(
            event: "observation",
            body: WorkspaceObservation(
                revision: observationRevision,
                kind: kind,
                observedAt: observedAt,
                frontmostApplication: application
            )
        )
    }

    private func configureWorkspace(enabled: Bool) {
        guard enabled else {
            workspaceMonitor?.stop()
            workspaceMonitor = nil
            screenTextPrivacyContext.clearWorkspace()
            visionCandidateBroker.clear()
            return
        }
        guard workspaceMonitor == nil, started, !stopped else { return }
        let monitor = makeWorkspaceMonitor { [weak self] kind, application in
            self?.sendObservation(kind: kind, application: application)
        }
        workspaceMonitor = monitor
        monitor.start()
    }

    private func makeModules() -> LazyCapabilityModules {
        if let providedModules { return providedModules }
        let eventWriter = writer
        return LazyCapabilityModules(
            screenTextPrivacyContext: screenTextPrivacyContext,
            visionCandidateBroker: visionCandidateBroker,
            environmentFrameBroker: environmentFrameBroker,
            onScreenObservation: { observation in
                eventWriter.send(event: "observation", body: observation)
            },
            onScreenTextObservation: { observation in
                eventWriter.send(event: "observation", body: observation)
            },
            onVisionCandidateAvailable: { metadata in
                eventWriter.send(event: "vision_candidate_available", body: metadata)
            },
            onEnvironmentFrameAvailable: { metadata in
                eventWriter.send(event: "environment_frame_available", body: metadata)
            },
            onAccessibilityObservation: { observation in
                eventWriter.send(event: "observation", body: observation)
            },
            onCameraObservation: { observation in
                eventWriter.send(event: "observation", body: observation)
            },
            onComputerHistoryEvent: { event in
                eventWriter.send(event: "interaction_event", body: event)
            },
            onComputerHistorySuppressed: { report in
                eventWriter.send(
                    event: "interaction_events_suppressed",
                    body: report
                )
            },
            onStatusChanged: { [weak self] in self?.sendStatus() },
            onSystemAudioWindow: { window in
                eventWriter.send(event: "system_audio_window", body: window)
            }
        )
    }

    private func sendStatus(requestId: String? = nil) {
        writer.send(
            event: "status",
            requestId: requestId,
            body: HelperStatus(
                phase: stopped ? "stopped" : "ready",
                modules: modules.moduleSelection,
                capabilities: capabilities()
            )
        )
    }

    private func capabilities() -> PerceptionCapabilities {
        let moduleCapabilities = modules.report()
        let workspaceRunning = workspaceMonitor != nil && !stopped
        return PerceptionCapabilities(
            workspace: CapabilityReport(
                supported: true,
                requested: workspaceRequested,
                state: workspaceRunning ? .running : .disabled,
                permission: .notRequired
            ),
            screen: moduleCapabilities.screen,
            accessibility: moduleCapabilities.accessibility,
            camera: moduleCapabilities.camera,
            computerHistory: moduleCapabilities.computerHistory,
            systemAudio: moduleCapabilities.systemAudio
        )
    }

    private func sendCommandError(requestId: String?, code: String, message: String) {
        writer.send(
            event: "error",
            requestId: requestId,
            body: HelperError(code: code, message: message)
        )
    }

    private func claimVisionCandidate(_ command: PerceptionCommand) {
        guard let candidateId = command.candidateId,
              let uuid = UUID(uuidString: candidateId),
              uuid.uuidString == candidateId,
              let payload = visionCandidateBroker.claim(
                  candidateId: candidateId,
                  isStillAuthorized: { [screenTextPrivacyContext] candidate, expected in
                      screenTextPrivacyContext.authorization(for: candidate) == expected
                  }
              )
        else {
            writer.send(
                event: "vision_candidate_unavailable",
                requestId: command.requestId,
                body: VisionCandidateUnavailable()
            )
            return
        }
        writer.send(
            event: "vision_candidate",
            requestId: command.requestId,
            body: payload
        )
    }

    private func claimEnvironmentFrame(_ command: PerceptionCommand) {
        guard let frameId = command.frameId,
              let uuid = UUID(uuidString: frameId),
              uuid.uuidString == frameId,
              let payload = environmentFrameBroker.claim(frameId: frameId)
        else {
            writer.send(
                event: "environment_frame_unavailable",
                requestId: command.requestId,
                body: EnvironmentFrameUnavailable()
            )
            return
        }
        writer.send(
            event: "environment_frame",
            requestId: command.requestId,
            body: payload
        )
    }
}
