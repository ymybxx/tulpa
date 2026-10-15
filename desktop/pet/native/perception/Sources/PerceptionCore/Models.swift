import Foundation

public enum CapabilityPermission: String, Codable, Sendable {
    case notRequired = "not_required"
    case granted
    case denied
    case notDetermined = "not_determined"
    case restricted
    case unknown
}

public enum CapabilityState: String, Codable, Sendable {
    case disabled
    case running
    case unavailable
    case failed
}

public enum ComputerHistoryEventTapState: String, Codable, Sendable {
    case disabled
    case permissionRequired = "permission_required"
    case running
    case suspended
    case failed
}

public struct ComputerHistoryInputChannelReport: Codable, Equatable, Sendable {
    public let state: CapabilityState
    public let receivedCount: UInt64
    public let lastEventAt: Double?

    public init(
        state: CapabilityState,
        receivedCount: UInt64,
        lastEventAt: Double?
    ) {
        self.state = state
        self.receivedCount = receivedCount
        self.lastEventAt = lastEventAt
    }
}

public struct ComputerHistoryInputReport: Codable, Equatable, Sendable {
    public let accessibilityPermission: CapabilityPermission
    public let listenEventPermission: CapabilityPermission
    public let eventTapState: ComputerHistoryEventTapState
    public let mouse: ComputerHistoryInputChannelReport
    public let keyboard: ComputerHistoryInputChannelReport
    public let tapDisabledCount: UInt64

    public init(
        accessibilityPermission: CapabilityPermission,
        listenEventPermission: CapabilityPermission,
        eventTapState: ComputerHistoryEventTapState,
        mouse: ComputerHistoryInputChannelReport,
        keyboard: ComputerHistoryInputChannelReport,
        tapDisabledCount: UInt64
    ) {
        self.accessibilityPermission = accessibilityPermission
        self.listenEventPermission = listenEventPermission
        self.eventTapState = eventTapState
        self.mouse = mouse
        self.keyboard = keyboard
        self.tapDisabledCount = tapDisabledCount
    }

    public static func disabled(
        accessibilityPermission: CapabilityPermission
    ) -> ComputerHistoryInputReport {
        let channel = ComputerHistoryInputChannelReport(
            state: .disabled,
            receivedCount: 0,
            lastEventAt: nil
        )
        return ComputerHistoryInputReport(
            accessibilityPermission: accessibilityPermission,
            listenEventPermission: .notDetermined,
            eventTapState: .disabled,
            mouse: channel,
            keyboard: channel,
            tapDisabledCount: 0
        )
    }

    func replacingAccessibilityPermission(
        _ permission: CapabilityPermission
    ) -> ComputerHistoryInputReport {
        ComputerHistoryInputReport(
            accessibilityPermission: permission,
            listenEventPermission: listenEventPermission,
            eventTapState: eventTapState,
            mouse: mouse,
            keyboard: keyboard,
            tapDisabledCount: tapDisabledCount
        )
    }
}

public struct CapabilityReport: Codable, Equatable, Sendable {
    public let supported: Bool
    public let requested: Bool
    public let state: CapabilityState
    public let permission: CapabilityPermission
    public let reason: String?
    public let computerHistoryInput: ComputerHistoryInputReport?

    public init(
        supported: Bool,
        requested: Bool,
        state: CapabilityState,
        permission: CapabilityPermission,
        reason: String? = nil,
        computerHistoryInput: ComputerHistoryInputReport? = nil
    ) {
        self.supported = supported
        self.requested = requested
        self.state = state
        self.permission = permission
        self.reason = reason
        self.computerHistoryInput = computerHistoryInput
    }
}

public struct PerceptionCapabilities: Codable, Equatable, Sendable {
    public let workspace: CapabilityReport
    public let screen: CapabilityReport
    public let accessibility: CapabilityReport
    public let camera: CapabilityReport
    public let computerHistory: CapabilityReport
    public let systemAudio: CapabilityReport?

    public init(
        workspace: CapabilityReport,
        screen: CapabilityReport,
        accessibility: CapabilityReport,
        camera: CapabilityReport,
        computerHistory: CapabilityReport,
        systemAudio: CapabilityReport? = nil
    ) {
        self.workspace = workspace
        self.screen = screen
        self.accessibility = accessibility
        self.camera = camera
        self.computerHistory = computerHistory
        self.systemAudio = systemAudio
    }
}

public struct HelperHello: Codable, Equatable, Sendable {
    public let helperVersion: String
    public let minimumMacOS: String
    public let capabilities: PerceptionCapabilities

    public init(
        helperVersion: String,
        minimumMacOS: String,
        capabilities: PerceptionCapabilities
    ) {
        self.helperVersion = helperVersion
        self.minimumMacOS = minimumMacOS
        self.capabilities = capabilities
    }
}

public struct HelperStatus: Codable, Equatable, Sendable {
    public let phase: String
    public let modules: PerceptionModuleSelection
    public let capabilities: PerceptionCapabilities

    public init(
        phase: String,
        modules: PerceptionModuleSelection,
        capabilities: PerceptionCapabilities
    ) {
        self.phase = phase
        self.modules = modules
        self.capabilities = capabilities
    }
}

public struct HelperAcknowledgement: Codable, Equatable, Sendable {
    public let command: String

    public init(command: String) {
        self.command = command
    }
}

public struct HelperError: Codable, Equatable, Sendable {
    public let code: String
    public let message: String
    public let fatal: Bool

    public init(code: String, message: String, fatal: Bool = false) {
        self.code = code
        self.message = message
        self.fatal = fatal
    }
}

public struct FrontmostApplication: Codable, Equatable, Sendable {
    public let bundleIdentifier: String?
    public let name: String?
    public let processIdentifier: Int32

    public init(bundleIdentifier: String?, name: String?, processIdentifier: Int32) {
        self.bundleIdentifier = bundleIdentifier
        self.name = name
        self.processIdentifier = processIdentifier
    }
}

public struct WorkspaceObservation: Codable, Equatable, Sendable {
    public let source: String
    public let revision: UInt64
    public let kind: String
    public let observedAt: Double
    public let frontmostApplication: FrontmostApplication?

    public init(
        revision: UInt64,
        kind: String,
        observedAt: Double,
        frontmostApplication: FrontmostApplication?
    ) {
        self.source = "workspace"
        self.revision = revision
        self.kind = kind
        self.observedAt = observedAt
        self.frontmostApplication = frontmostApplication
    }
}
