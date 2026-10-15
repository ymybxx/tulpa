import CoreMedia
import XCTest
@testable import PerceptionCore

final class CameraMonitorTests: XCTestCase {
    func testDefaultCaptureIsVideoOnlyVGAAtFifteenFramesPerSecond() {
        let configuration = CameraMonitorConfiguration()

        XCTAssertEqual(configuration.width, 640)
        XCTAssertEqual(configuration.height, 480)
        XCTAssertEqual(configuration.captureFramesPerSecond, 15)
        XCTAssertEqual(configuration.analysisFramesPerSecond, 10)
    }

    func testLatestValueGateKeepsOnlyNewestPendingFrame() {
        let gate = LatestValueGate<Int>()

        XCTAssertEqual(gate.submit(1), 1)
        XCTAssertNil(gate.submit(2))
        XCTAssertNil(gate.submit(3))
        XCTAssertEqual(gate.takeNextOrFinish(), 3)
        XCTAssertNil(gate.takeNextOrFinish())
        XCTAssertEqual(gate.submit(4), 4)
    }

    func testLatestValueGateResetDoesNotAllowConcurrentDrains() {
        let gate = LatestValueGate<Int>()

        XCTAssertEqual(gate.submit(1), 1)
        gate.reset()
        XCTAssertNil(gate.submit(2))
        XCTAssertEqual(gate.takeNextOrFinish(), 2)
        XCTAssertNil(gate.takeNextOrFinish())
    }

    func testCameraFramesAreAnalyzedExactlyOnceWithoutMirroring() {
        XCTAssertEqual(
            CameraMonitor.visionImageOrientation.rawValue,
            CGImagePropertyOrientation.up.rawValue
        )
    }

    func testAnalysisThrottleMeasuresFromLastAcceptedFrame() {
        let monitor = CameraMonitor(
            configuration: CameraMonitorConfiguration(
                captureFramesPerSecond: 15,
                analysisFramesPerSecond: 10
            ),
            onObservation: { _ in }
        )

        XCTAssertTrue(monitor.shouldAccept(timestamp: .zero))
        XCTAssertFalse(
            monitor.shouldAccept(timestamp: CMTime(value: 1, timescale: 15))
        )
        XCTAssertTrue(
            monitor.shouldAccept(timestamp: CMTime(value: 2, timescale: 15))
        )
    }

    func testFaceProxiesStayBoundedAndRemainSoftSignals() {
        XCTAssertEqual(
            try XCTUnwrap(
                FaceLandmarkMetrics.fatigueProxy(
                    closedEyeProxy: 1,
                    mouthOpenness: 1
                )
            ),
            1,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(FaceLandmarkMetrics.attentionProxy(yaw: 0, pitch: 0)),
            1,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(FaceLandmarkMetrics.attentionProxy(yaw: 2, pitch: 0)),
            0,
            accuracy: 0.0001
        )
        XCTAssertNil(
            FaceLandmarkMetrics.fatigueProxy(
                closedEyeProxy: nil,
                mouthOpenness: nil
            )
        )
    }

    func testCameraObservationContainsNoFrameOrImagePayload() throws {
        let observation = CameraObservation(
            revision: 1,
            observedAt: 10,
            facePresent: false,
            faceCount: 0,
            primaryFace: nil
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(observation))
                as? [String: Any]
        )
        XCTAssertNil(object["frame"])
        XCTAssertNil(object["image"])
        XCTAssertEqual(object["source"] as? String, "camera")
        XCTAssertEqual(object["kind"] as? String, "camera_face_observation")
        XCTAssertEqual(object["facePresent"] as? Bool, false)
    }
}
