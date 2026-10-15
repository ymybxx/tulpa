import XCTest
@testable import PerceptionCore

final class ProtocolTests: XCTestCase {
    func testDecodesConfigureCommand() throws {
        let command = try PerceptionProtocolCodec.decodeCommand(
            #"{"protocolVersion":1,"requestId":"r1","command":"configure","modules":{"workspace":true,"screen":true,"accessibility":false,"camera":true}}"#
        )

        XCTAssertEqual(command.protocolVersion, 1)
        XCTAssertEqual(command.requestId, "r1")
        XCTAssertEqual(command.command, "configure")
        XCTAssertNil(command.environmentFrameIntervalMs)
        XCTAssertEqual(
            command.modules,
            PerceptionModuleSelection(
                workspace: true,
                screen: true,
                accessibility: false,
                camera: true
            )
        )
    }

    func testConfigureEnvironmentFrameIntervalIsOptionalAndAdditive() throws {
        let command = try PerceptionProtocolCodec.decodeCommand(
            #"{"protocolVersion":1,"command":"configure","environmentFrameIntervalMs":5000,"modules":{"screen":true,"accessibility":false,"camera":false}}"#
        )

        XCTAssertEqual(command.environmentFrameIntervalMs, 5_000)
        XCTAssertEqual(command.modules?.screen, true)
        XCTAssertEqual(command.protocolVersion, perceptionProtocolVersion)
    }

    func testConfigureWithoutWorkspaceRemainsProtocolV1Compatible() throws {
        let command = try PerceptionProtocolCodec.decodeCommand(
            #"{"protocolVersion":1,"command":"configure","modules":{"screen":true,"accessibility":false,"camera":false}}"#
        )

        XCTAssertEqual(command.protocolVersion, 1)
        XCTAssertEqual(command.modules?.workspace, false)
        XCTAssertEqual(command.modules?.enhancedVision, false)
        XCTAssertEqual(command.modules?.computerHistory, false)
        XCTAssertEqual(command.modules?.legacyScreenAnalysis, true)
        XCTAssertEqual(command.modules?.excludeHostApplicationFromCapture, false)
    }

    func testConfigureComputerHistoryIsOptionalAndAdditive() throws {
        let command = try PerceptionProtocolCodec.decodeCommand(
            #"{"protocolVersion":1,"command":"configure","modules":{"screen":false,"accessibility":false,"camera":false,"computerHistory":true}}"#
        )

        XCTAssertEqual(command.modules?.computerHistory, true)
        XCTAssertEqual(command.protocolVersion, perceptionProtocolVersion)
    }

    func testLegacyScreenAnalysisCanBeDisabledAdditively() throws {
        let command = try PerceptionProtocolCodec.decodeCommand(
            #"{"protocolVersion":1,"command":"configure","modules":{"screen":true,"accessibility":false,"camera":false,"legacyScreenAnalysis":false}}"#
        )

        XCTAssertEqual(command.modules?.legacyScreenAnalysis, false)
    }

    func testScreenCaptureExclusionIsOptionalAndIndependentFromLegacyAnalysis() throws {
        let command = try PerceptionProtocolCodec.decodeCommand(
            #"{"protocolVersion":1,"command":"configure","modules":{"screen":true,"accessibility":false,"camera":false,"legacyScreenAnalysis":false,"excludeHostApplicationFromCapture":true}}"#
        )

        XCTAssertEqual(command.modules?.legacyScreenAnalysis, false)
        XCTAssertEqual(command.modules?.excludeHostApplicationFromCapture, true)
        let encoded = try JSONEncoder().encode(command.modules)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(object["excludeHostApplicationFromCapture"] as? Bool, true)
    }

    func testConfigureEnhancedVisionIsAdditiveAndRequiresAllContextModules() throws {
        let command = try PerceptionProtocolCodec.decodeCommand(
            #"{"protocolVersion":1,"command":"configure","modules":{"workspace":true,"screen":true,"accessibility":true,"camera":false,"enhancedVision":true}}"#
        )

        XCTAssertEqual(command.modules?.enhancedVision, true)
        XCTAssertEqual(command.modules?.enhancedVisionActive, true)
        XCTAssertFalse(PerceptionModuleSelection(
            workspace: true,
            screen: true,
            accessibility: false,
            camera: true,
            enhancedVision: true
        ).enhancedVisionActive)
        XCTAssertFalse(PerceptionModuleSelection(
            workspace: false,
            screen: false,
            accessibility: false,
            camera: true,
            enhancedVision: true
        ).enhancedVisionActive)
    }

    func testDecodesExactVisionCandidateClaim() throws {
        let candidateId = "8C6424AE-5EC2-4456-9C49-69C349E05D68"
        let command = try PerceptionProtocolCodec.decodeCommand(
            #"{"protocolVersion":1,"requestId":"claim-1","command":"claim_vision_candidate","candidateId":"\#(candidateId)"}"#
        )

        XCTAssertEqual(command.command, "claim_vision_candidate")
        XCTAssertEqual(command.candidateId, candidateId)
    }

    func testDecodesExactEnvironmentFrameClaim() throws {
        let frameId = "00000000-0000-0000-0000-000000000001"
        let command = try PerceptionProtocolCodec.decodeCommand(
            """
            {"protocolVersion":1,"requestId":"request-1","command":"claim_environment_frame","frameId":"\(frameId)"}
            """
        )

        XCTAssertEqual(command.command, "claim_environment_frame")
        XCTAssertEqual(command.frameId, frameId)
        XCTAssertNil(command.candidateId)
        XCTAssertNil(command.modules)
    }

    func testEnvironmentFrameMetadataContainsNoPixels() throws {
        let metadata = EnvironmentFrameMetadata(
            frameId: "00000000-0000-0000-0000-000000000001",
            revision: 2,
            observedAt: 100,
            expiresAt: 103,
            displayID: 1,
            pixelWidth: 960,
            pixelHeight: 540,
            byteLength: 100
        )
        let data = try JSONEncoder().encode(metadata)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        XCTAssertEqual(Set(object.keys), Set([
            "frameId", "revision", "observedAt", "expiresAt", "displayID",
            "pixelWidth", "pixelHeight", "byteLength",
        ]))
        XCTAssertNil(object["jpegBase64"])
    }

    func testModuleSelectionEncodingIncludesWorkspace() throws {
        let selection = PerceptionModuleSelection(
            workspace: false,
            screen: true,
            accessibility: false,
            camera: false
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(selection))
                as? [String: Any]
        )

        XCTAssertEqual(object["workspace"] as? Bool, false)
        XCTAssertEqual(object["computerHistory"] as? Bool, false)
        XCTAssertEqual(object["legacyScreenAnalysis"] as? Bool, true)
    }

    func testEncodesOneNewlineDelimitedEnvelope() throws {
        let envelope = PerceptionEventEnvelope(
            sequence: 7,
            sentAt: 123.5,
            event: "ack",
            requestId: "r7",
            body: HelperAcknowledgement(command: "ping")
        )

        let data = try PerceptionProtocolCodec.encodeEvent(envelope)
        let line = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertTrue(line.hasSuffix("\n"))
        XCTAssertEqual(line.filter { $0 == "\n" }.count, 1)

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(object["protocolVersion"] as? Int, 1)
        XCTAssertEqual(object["event"] as? String, "ack")
        XCTAssertEqual(object["sequence"] as? Int, 7)
    }

    func testComputerHistoryCapabilityExposesIndependentInputHealthAdditively() throws {
        let input = ComputerHistoryInputReport(
            accessibilityPermission: .granted,
            listenEventPermission: .denied,
            eventTapState: .permissionRequired,
            mouse: .init(state: .unavailable, receivedCount: 3, lastEventAt: 12),
            keyboard: .init(state: .unavailable, receivedCount: 4, lastEventAt: 13),
            tapDisabledCount: 2
        )
        let report = CapabilityReport(
            supported: true,
            requested: true,
            state: .unavailable,
            permission: .granted,
            reason: "input_monitoring_permission_required",
            computerHistoryInput: input
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(report))
                as? [String: Any]
        )
        let encodedInput = try XCTUnwrap(
            object["computerHistoryInput"] as? [String: Any]
        )
        XCTAssertEqual(encodedInput["accessibilityPermission"] as? String, "granted")
        XCTAssertEqual(encodedInput["listenEventPermission"] as? String, "denied")
        XCTAssertEqual(encodedInput["eventTapState"] as? String, "permission_required")
        XCTAssertEqual(
            (encodedInput["mouse"] as? [String: Any])?["receivedCount"] as? Int,
            3
        )
        XCTAssertEqual(
            (encodedInput["keyboard"] as? [String: Any])?["receivedCount"] as? Int,
            4
        )

        let legacyShape = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(CapabilityReport(
                supported: true,
                requested: false,
                state: .disabled,
                permission: .notRequired
            ))) as? [String: Any]
        )
        XCTAssertNil(legacyShape["computerHistoryInput"])
    }

    func testWorkspaceObservationHasExplicitSourceAndRevision() throws {
        let observation = WorkspaceObservation(
            revision: 4,
            kind: "frontmost_application_changed",
            observedAt: 123,
            frontmostApplication: nil
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(observation))
                as? [String: Any]
        )
        XCTAssertEqual(object["source"] as? String, "workspace")
        XCTAssertEqual(object["revision"] as? Int, 4)
        XCTAssertEqual(object["kind"] as? String, "frontmost_application_changed")
    }

    func testVisionCandidateMetadataContainsNoImageTitleOrFrame() throws {
        let metadata = VisionCandidateMetadata(
            candidateId: "8C6424AE-5EC2-4456-9C49-69C349E05D68",
            revision: 3,
            observedAt: 100,
            expiresAt: 105,
            displayID: 1,
            appBundleId: "com.example.editor",
            appName: "Editor",
            windowKey: "1BCBEF2B-14EE-4E61-8184-AF1908135448"
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(metadata))
                as? [String: Any]
        )

        XCTAssertNil(object["jpegBase64"])
        XCTAssertNil(object["windowTitle"])
        XCTAssertNil(object["windowFrame"])
        XCTAssertEqual(object["candidateId"] as? String, metadata.candidateId)
    }
}
