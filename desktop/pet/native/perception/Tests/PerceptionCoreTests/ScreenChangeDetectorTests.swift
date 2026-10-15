import CoreGraphics
import XCTest
@testable import PerceptionCore

final class ScreenChangeDetectorTests: XCTestCase {
    func testMeasuredPixelRatioReachesStableEvenWhenCompositorKeepsRepainting() {
        var detector = ScreenChangeDetector()

        XCTAssertEqual(detector.ingestMeasured(
            timestamp: 0,
            changedAreaRatio: 1,
            changedRegionCount: 1
        )?.state, .changing)
        XCTAssertEqual(detector.ingestMeasured(
            timestamp: 0.2,
            changedAreaRatio: 0,
            changedRegionCount: 1
        )?.state, .changing)
        XCTAssertEqual(detector.ingestMeasured(
            timestamp: 0.4,
            changedAreaRatio: 0,
            changedRegionCount: 1
        )?.state, .changing)
        let stable = detector.ingestMeasured(
            timestamp: 0.6,
            changedAreaRatio: 0,
            changedRegionCount: 1
        )

        XCTAssertEqual(stable?.state, .stable)
        XCTAssertEqual(stable?.dirtyAreaRatio, 0)
        XCTAssertEqual(stable?.dirtyRectCount, 1)
    }

    func testScreenObservationHasTopLevelContractFields() throws {
        let evaluation = ScreenChangeEvaluation(
            revision: 7,
            timestamp: 10,
            dirtyAreaRatio: 0.25,
            dirtyRectCount: 1,
            state: .changing,
            stateChanged: true,
            significantChange: true
        )
        let observation = ScreenChangeObservation(
            displayID: 1,
            frameWidth: 100,
            frameHeight: 80,
            evaluation: evaluation
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(observation))
                as? [String: Any]
        )
        XCTAssertEqual(object["source"] as? String, "screen")
        XCTAssertEqual(object["kind"] as? String, "screen_change")
        XCTAssertEqual(object["revision"] as? Int, 7)
        XCTAssertNotNil(object["observedAt"])
    }

    func testScreenCaptureExcludesFishMemoryAndItsParentApplication() {
        let excluded = Set(["app.fishmemory.desktop"])

        XCTAssertTrue(shouldExcludeScreenApplication(
            bundleIdentifier: "app.fishmemory.desktop",
            processIdentifier: 10,
            parentProcessIdentifier: 20,
            mainBundleIdentifier: nil,
            excludedBundleIdentifiers: excluded
        ))
        XCTAssertTrue(shouldExcludeScreenApplication(
            bundleIdentifier: "app.tulpa.desktop.perception",
            processIdentifier: 10,
            parentProcessIdentifier: 20,
            mainBundleIdentifier: nil,
            excludedBundleIdentifiers: excluded
        ))
        XCTAssertTrue(shouldExcludeScreenApplication(
            bundleIdentifier: "com.fish-memory.desktop.debug",
            processIdentifier: 10,
            parentProcessIdentifier: 20,
            mainBundleIdentifier: nil,
            excludedBundleIdentifiers: excluded
        ))
        XCTAssertTrue(shouldExcludeScreenApplication(
            bundleIdentifier: "com.github.Electron",
            processIdentifier: 20,
            parentProcessIdentifier: 20,
            mainBundleIdentifier: nil,
            excludedBundleIdentifiers: excluded
        ))
        XCTAssertFalse(shouldExcludeScreenApplication(
            bundleIdentifier: "com.apple.Safari",
            processIdentifier: 30,
            parentProcessIdentifier: 20,
            mainBundleIdentifier: nil,
            excludedBundleIdentifiers: excluded
        ))
        XCTAssertFalse(shouldExcludeScreenApplication(
            bundleIdentifier: "app.fishmemory.desktop",
            processIdentifier: 20,
            parentProcessIdentifier: 20,
            mainBundleIdentifier: "app.fishmemory.desktop",
            excludedBundleIdentifiers: excluded,
            excludeHostApplication: false
        ))
    }

    @available(macOS 12.3, *)
    func testScreenMonitorDoesNotPromptWhenPermissionIsMissing() async {
        let monitor = ScreenMonitor(
            hasScreenRecordingPermission: { false },
            onObservation: { _ in }
        )

        let status = await monitor.start()

        XCTAssertEqual(status.state, .unavailable)
        XCTAssertEqual(status.reason, "screen_recording_permission_required")
    }

    func testAnalyzesAtFiveHertzUsingTheNewestEligibleFrame() throws {
        var detector = ScreenChangeDetector()

        XCTAssertNotNil(detector.ingest(
            timestamp: 0,
            frameWidth: 100,
            frameHeight: 100,
            dirtyRects: []
        ))
        XCTAssertNil(detector.ingest(
            timestamp: 0.1,
            frameWidth: 100,
            frameHeight: 100,
            dirtyRects: [CGRect(x: 0, y: 0, width: 100, height: 100)]
        ))
        let next = try XCTUnwrap(detector.ingest(
            timestamp: 0.2,
            frameWidth: 100,
            frameHeight: 100,
            dirtyRects: [CGRect(x: 0, y: 0, width: 20, height: 20)]
        ))

        XCTAssertEqual(next.revision, 2)
        XCTAssertEqual(next.dirtyAreaRatio, 0.04, accuracy: 0.0001)
    }

    func testComputesClippedUnionAreaWithoutDoubleCountingOverlap() throws {
        var detector = ScreenChangeDetector()
        let evaluation = try XCTUnwrap(detector.ingest(
            timestamp: 0,
            frameWidth: 100,
            frameHeight: 100,
            dirtyRects: [
                CGRect(x: -10, y: 0, width: 30, height: 20),
                CGRect(x: 10, y: 0, width: 20, height: 20),
            ]
        ))

        XCTAssertEqual(evaluation.dirtyRectCount, 2)
        XCTAssertEqual(evaluation.dirtyAreaRatio, 0.06, accuracy: 0.0001)
    }

    func testBecomesStableAfterThreeQuietSamples() throws {
        var detector = ScreenChangeDetector()
        let first = try XCTUnwrap(ingest(&detector, index: 0, ratio: 0.01))
        let second = try XCTUnwrap(ingest(&detector, index: 1, ratio: 0.01))
        let third = try XCTUnwrap(ingest(&detector, index: 2, ratio: 0.01))

        XCTAssertEqual(first.state, .changing)
        XCTAssertEqual(second.state, .changing)
        XCTAssertEqual(third.state, .stable)
        XCTAssertTrue(third.stateChanged)
    }

    func testDetectsSevenMovingSamplesWithinTenSampleWindow() throws {
        var detector = ScreenChangeDetector()
        var latest: ScreenChangeEvaluation?
        for index in 0 ..< 10 {
            latest = ingest(&detector, index: index, ratio: index < 7 ? 0.2 : 0.05)
        }

        let evaluation = try XCTUnwrap(latest)
        XCTAssertEqual(evaluation.state, .continuousMotion)
        XCTAssertTrue(evaluation.stateChanged)
    }

    func testContinuousMotionReturnsToStableOnlyAfterQuietRun() throws {
        var detector = ScreenChangeDetector()
        for index in 0 ..< 10 {
            _ = ingest(&detector, index: index, ratio: index < 7 ? 0.2 : 0.05)
        }
        let firstQuiet = try XCTUnwrap(ingest(&detector, index: 10, ratio: 0.0))
        let secondQuiet = try XCTUnwrap(ingest(&detector, index: 11, ratio: 0.0))
        let thirdQuiet = try XCTUnwrap(ingest(&detector, index: 12, ratio: 0.0))

        XCTAssertEqual(firstQuiet.state, .continuousMotion)
        XCTAssertEqual(secondQuiet.state, .continuousMotion)
        XCTAssertEqual(thirdQuiet.state, .stable)
    }

    func testRevisionRemainsMonotonicAcrossClockReset() throws {
        var detector = ScreenChangeDetector()
        let first = try XCTUnwrap(detector.ingest(
            timestamp: 10,
            frameWidth: 100,
            frameHeight: 100,
            dirtyRects: []
        ))
        let afterClockReset = try XCTUnwrap(detector.ingest(
            timestamp: 1,
            frameWidth: 100,
            frameHeight: 100,
            dirtyRects: []
        ))

        XCTAssertEqual(first.revision, 1)
        XCTAssertEqual(afterClockReset.revision, 2)
    }

    private func ingest(
        _ detector: inout ScreenChangeDetector,
        index: Int,
        ratio: Double
    ) -> ScreenChangeEvaluation? {
        let side = sqrt(ratio) * 100
        return detector.ingest(
            timestamp: Double(index) * 0.2,
            frameWidth: 100,
            frameHeight: 100,
            dirtyRects: ratio == 0 ? [] : [CGRect(x: 0, y: 0, width: side, height: side)]
        )
    }
}
