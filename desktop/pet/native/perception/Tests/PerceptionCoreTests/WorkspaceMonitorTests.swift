import AppKit
import XCTest
@testable import PerceptionCore

final class WorkspaceMonitorTests: XCTestCase {
    @MainActor
    func testHeartbeatRepeatsCurrentApplicationAndStopsWithMonitor() {
        let scheduler = ControlledWorkspaceHeartbeatScheduler()
        var kinds: [String] = []
        let monitor = WorkspaceMonitor(
            workspace: .shared,
            heartbeatInterval: 30,
            heartbeatScheduler: scheduler,
            onObservation: { kind, _ in kinds.append(kind) }
        )

        monitor.start()
        XCTAssertEqual(scheduler.interval, 30)
        XCTAssertEqual(kinds, ["foreground_app_snapshot"])

        scheduler.fire()
        XCTAssertEqual(
            kinds,
            ["foreground_app_snapshot", "foreground_app_heartbeat"]
        )

        monitor.stop()
        XCTAssertTrue(scheduler.token.isCancelled)
        scheduler.fire()
        XCTAssertEqual(kinds.count, 2)
    }

    @MainActor
    func testSessionAndScreenPrivacyBoundariesEmitUnknownApplication() {
        let scheduler = ControlledWorkspaceHeartbeatScheduler()
        var events: [(String, FrontmostApplication?)] = []
        let workspace = NSWorkspace.shared
        let monitor = WorkspaceMonitor(
            workspace: workspace,
            heartbeatInterval: 30,
            heartbeatScheduler: scheduler,
            onObservation: { events.append(($0, $1)) }
        )
        monitor.start()
        events.removeAll()

        workspace.notificationCenter.post(
            name: NSWorkspace.sessionDidResignActiveNotification,
            object: nil
        )
        workspace.notificationCenter.post(
            name: NSWorkspace.screensDidSleepNotification,
            object: nil
        )

        XCTAssertEqual(events.map(\.0), [
            "session_resigned_active",
            "screens_did_sleep",
        ])
        XCTAssertTrue(events.allSatisfy { $0.1 == nil })
        monitor.stop()
    }
}

@MainActor
private final class ControlledWorkspaceHeartbeatScheduler: WorkspaceHeartbeatScheduling {
    let token = ControlledWorkspaceHeartbeatToken()
    private var action: (@MainActor () -> Void)?
    private(set) var interval: TimeInterval?

    func schedule(
        every interval: TimeInterval,
        action: @escaping @MainActor () -> Void
    ) -> any WorkspaceHeartbeatToken {
        self.interval = interval
        self.action = action
        return token
    }

    func fire() {
        guard !token.isCancelled else { return }
        action?()
    }
}

@MainActor
private final class ControlledWorkspaceHeartbeatToken: WorkspaceHeartbeatToken {
    private(set) var isCancelled = false

    func cancel() {
        isCancelled = true
    }
}
