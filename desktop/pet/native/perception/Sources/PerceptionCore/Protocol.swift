import Foundation

public let perceptionProtocolVersion = 1

public struct PerceptionCommand: Codable, Equatable, Sendable {
    public let protocolVersion: Int
    public let requestId: String?
    public let command: String
    public let modules: PerceptionModuleSelection?
    public let environmentFrameIntervalMs: Int?
    public let candidateId: String?
    public let frameId: String?
    public let boundaryAt: Double?

    public init(
        protocolVersion: Int,
        requestId: String? = nil,
        command: String,
        modules: PerceptionModuleSelection? = nil,
        environmentFrameIntervalMs: Int? = nil,
        candidateId: String? = nil,
        frameId: String? = nil,
        boundaryAt: Double? = nil
    ) {
        self.protocolVersion = protocolVersion
        self.requestId = requestId
        self.command = command
        self.modules = modules
        self.environmentFrameIntervalMs = environmentFrameIntervalMs
        self.candidateId = candidateId
        self.frameId = frameId
        self.boundaryAt = boundaryAt
    }
}

public struct PerceptionModuleSelection: Codable, Equatable, Sendable {
    public let workspace: Bool
    public let screen: Bool
    public let accessibility: Bool
    public let camera: Bool
    public let enhancedVision: Bool
    public let computerHistory: Bool
    public let legacyScreenAnalysis: Bool
    public let excludeHostApplicationFromCapture: Bool
    public let systemAudio: Bool

    public init(
        workspace: Bool = false,
        screen: Bool,
        accessibility: Bool,
        camera: Bool,
        enhancedVision: Bool = false,
        computerHistory: Bool = false,
        legacyScreenAnalysis: Bool = true,
        excludeHostApplicationFromCapture: Bool = false,
        systemAudio: Bool = false
    ) {
        self.workspace = workspace
        self.screen = screen
        self.accessibility = accessibility
        self.camera = camera
        self.enhancedVision = enhancedVision
        self.computerHistory = computerHistory
        self.legacyScreenAnalysis = legacyScreenAnalysis
        self.excludeHostApplicationFromCapture = excludeHostApplicationFromCapture
        self.systemAudio = systemAudio
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        workspace = try container.decodeIfPresent(Bool.self, forKey: .workspace) ?? false
        screen = try container.decode(Bool.self, forKey: .screen)
        accessibility = try container.decode(Bool.self, forKey: .accessibility)
        camera = try container.decode(Bool.self, forKey: .camera)
        enhancedVision = try container.decodeIfPresent(
            Bool.self,
            forKey: .enhancedVision
        ) ?? false
        computerHistory = try container.decodeIfPresent(
            Bool.self,
            forKey: .computerHistory
        ) ?? false
        legacyScreenAnalysis = try container.decodeIfPresent(
            Bool.self,
            forKey: .legacyScreenAnalysis
        ) ?? true
        excludeHostApplicationFromCapture = try container.decodeIfPresent(
            Bool.self,
            forKey: .excludeHostApplicationFromCapture
        ) ?? false
        systemAudio = try container.decodeIfPresent(Bool.self, forKey: .systemAudio) ?? false
    }

    public static let disabled = PerceptionModuleSelection(
        workspace: false,
        screen: false,
        accessibility: false,
        camera: false,
        enhancedVision: false,
        computerHistory: false,
        legacyScreenAnalysis: false
    )

    public var enhancedVisionActive: Bool {
        enhancedVision
            && legacyScreenAnalysis
            && screen
            && workspace
            && accessibility
    }

    private enum CodingKeys: String, CodingKey {
        case workspace
        case screen
        case accessibility
        case camera
        case enhancedVision
        case computerHistory
        case legacyScreenAnalysis
        case excludeHostApplicationFromCapture
        case systemAudio
    }
}

public struct PerceptionEventEnvelope<Body: Encodable>: Encodable {
    public let protocolVersion: Int
    public let sequence: UInt64
    public let sentAt: Double
    public let event: String
    public let requestId: String?
    public let body: Body

    public init(
        protocolVersion: Int = perceptionProtocolVersion,
        sequence: UInt64,
        sentAt: Double,
        event: String,
        requestId: String? = nil,
        body: Body
    ) {
        self.protocolVersion = protocolVersion
        self.sequence = sequence
        self.sentAt = sentAt
        self.event = event
        self.requestId = requestId
        self.body = body
    }
}

public enum PerceptionProtocolCodec {
    public static func decodeCommand(_ line: String) throws -> PerceptionCommand {
        guard let data = line.data(using: .utf8) else {
            throw PerceptionProtocolError.invalidUTF8
        }
        return try JSONDecoder().decode(PerceptionCommand.self, from: data)
    }

    public static func encodeEvent<Body: Encodable>(
        _ envelope: PerceptionEventEnvelope<Body>
    ) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(envelope)
        data.append(0x0A)
        return data
    }
}

public enum PerceptionProtocolError: Error, Equatable {
    case invalidUTF8
}
