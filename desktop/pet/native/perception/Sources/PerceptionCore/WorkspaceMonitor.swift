import AppKit
import Foundation

@MainActor
protocol WorkspaceHeartbeatToken: AnyObject {
    func cancel()
}

@MainActor
public protocol WorkspaceMonitoring: AnyObject {
    func start()
    func stop()
}

@MainActor
protocol WorkspaceHeartbeatScheduling: AnyObject {
    func schedule(
        every interval: TimeInterval,
        action: @escaping @MainActor () -> Void
    ) -> any WorkspaceHeartbeatToken
}

@MainActor
public final class WorkspaceMonitor: NSObject {
    public typealias ObservationHandler = @MainActor (String, FrontmostApplication?) -> Void

    private let workspace: NSWorkspace
    private let heartbeatInterval: TimeInterval
    private let heartbeatScheduler: any WorkspaceHeartbeatScheduling
    private let onObservation: ObservationHandler
    private var heartbeatToken: (any WorkspaceHeartbeatToken)?
    private var started = false

    public convenience init(
        workspace: NSWorkspace = .shared,
        heartbeatInterval: TimeInterval = 30,
        onObservation: @escaping ObservationHandler
    ) {
        self.init(
            workspace: workspace,
            heartbeatInterval: heartbeatInterval,
            heartbeatScheduler: RunLoopWorkspaceHeartbeatScheduler(),
            onObservation: onObservation
        )
    }

    init(
        workspace: NSWorkspace,
        heartbeatInterval: TimeInterval,
        heartbeatScheduler: any WorkspaceHeartbeatScheduling,
        onObservation: @escaping ObservationHandler
    ) {
        precondition(heartbeatInterval > 0)
        self.workspace = workspace
        self.heartbeatInterval = heartbeatInterval
        self.heartbeatScheduler = heartbeatScheduler
        self.onObservation = onObservation
        super.init()
    }

    public func start() {
        guard !started else { return }
        started = true
        let center = workspace.notificationCenter
        center.addObserver(
            self,
            selector: #selector(didActivateApplication(_:)),
            name: NSWorkspace.didActivateApplicationNotification,
            object: nil
        )
        center.addObserver(
            self,
            selector: #selector(activeSpaceDidChange(_:)),
            name: NSWorkspace.activeSpaceDidChangeNotification,
            object: nil
        )
        for name in Self.privacyBoundaryNotifications {
            center.addObserver(
                self,
                selector: #selector(privacyBoundaryReached(_:)),
                name: name,
                object: nil
            )
        }
        for name in Self.resumeNotifications {
            center.addObserver(
                self,
                selector: #selector(workspaceResumed(_:)),
                name: name,
                object: nil
            )
        }
        heartbeatToken = heartbeatScheduler.schedule(
            every: heartbeatInterval,
            action: { [weak self] in self?.emitHeartbeat() }
        )
        onObservation(
            "foreground_app_snapshot",
            Self.describe(workspace.frontmostApplication)
        )
    }

    public func stop() {
        guard started else { return }
        started = false
        heartbeatToken?.cancel()
        heartbeatToken = nil
        let center = workspace.notificationCenter
        center.removeObserver(self)
    }

    @objc private func didActivateApplication(_ notification: Notification) {
        guard started else { return }
        let application = notification.userInfo?[NSWorkspace.applicationUserInfoKey]
            as? NSRunningApplication
        onObservation("foreground_app_changed", Self.describe(application))
    }

    @objc private func activeSpaceDidChange(_ notification: Notification) {
        guard started else { return }
        _ = notification
        onObservation(
            "active_space_changed",
            Self.describe(workspace.frontmostApplication)
        )
    }

    @objc private func privacyBoundaryReached(_ notification: Notification) {
        guard started else { return }
        let kind = notification.name == NSWorkspace.screensDidSleepNotification
            ? "screens_did_sleep"
            : "session_resigned_active"
        onObservation(kind, nil)
    }

    @objc private func workspaceResumed(_ notification: Notification) {
        guard started else { return }
        let kind = notification.name == NSWorkspace.screensDidWakeNotification
            ? "screens_did_wake"
            : "session_became_active"
        onObservation(kind, Self.describe(workspace.frontmostApplication))
    }

    private func emitHeartbeat() {
        guard started else { return }
        onObservation(
            "foreground_app_heartbeat",
            Self.describe(workspace.frontmostApplication)
        )
    }

    private static func describe(_ application: NSRunningApplication?) -> FrontmostApplication? {
        guard let application else { return nil }
        return FrontmostApplication(
            bundleIdentifier: application.bundleIdentifier,
            name: application.localizedName,
            processIdentifier: application.processIdentifier
        )
    }

    private static let privacyBoundaryNotifications = [
        NSWorkspace.screensDidSleepNotification,
        NSWorkspace.sessionDidResignActiveNotification,
    ]

    private static let resumeNotifications = [
        NSWorkspace.screensDidWakeNotification,
        NSWorkspace.sessionDidBecomeActiveNotification,
    ]
}

extension WorkspaceMonitor: WorkspaceMonitoring {}

@MainActor
private final class RunLoopWorkspaceHeartbeatScheduler: WorkspaceHeartbeatScheduling {
    func schedule(
        every interval: TimeInterval,
        action: @escaping @MainActor () -> Void
    ) -> any WorkspaceHeartbeatToken {
        let token = RunLoopWorkspaceHeartbeatToken(action: action)
        token.start(every: interval)
        return token
    }
}

@MainActor
private final class RunLoopWorkspaceHeartbeatToken: NSObject, WorkspaceHeartbeatToken {
    private let action: @MainActor () -> Void
    private var timer: Timer?

    init(action: @escaping @MainActor () -> Void) {
        self.action = action
        super.init()
    }

    func start(every interval: TimeInterval) {
        guard timer == nil else { return }
        let timer = Timer(
            timeInterval: interval,
            target: self,
            selector: #selector(timerDidFire(_:)),
            userInfo: nil,
            repeats: true
        )
        self.timer = timer
        RunLoop.main.add(timer, forMode: .common)
    }

    func cancel() {
        timer?.invalidate()
        timer = nil
    }

    @objc private func timerDidFire(_ timer: Timer) {
        guard timer.isValid else { return }
        action()
    }
}
