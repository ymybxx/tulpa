import CoreGraphics
import CoreVideo
import XCTest
@testable import PerceptionCore

final class ScreenTextRecognizerTests: XCTestCase {
    func testTriggerRequiresSignificantChangeThenStableTransition() {
        var policy = ScreenTextTriggerPolicy(minimumRecognitionInterval: 3)

        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 1,
            timestamp: 0,
            state: .changing,
            stateChanged: false,
            significantChange: true
        )))
        XCTAssertTrue(policy.shouldRecognize(evaluation: evaluation(
            revision: 2,
            timestamp: 1,
            state: .stable,
            stateChanged: true,
            significantChange: false
        )))
        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 3,
            timestamp: 2,
            state: .stable,
            stateChanged: false,
            significantChange: false
        )))
    }

    func testTriggerCooldownStartsOnSuccessAndFailedAttemptCanRetry() {
        var policy = ScreenTextTriggerPolicy(minimumRecognitionInterval: 3)

        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 1,
            timestamp: 0,
            state: .changing,
            significantChange: true
        )))
        XCTAssertTrue(policy.shouldRecognize(evaluation: evaluation(
            revision: 2,
            timestamp: 1,
            state: .stable,
            stateChanged: true
        )))
        policy.completeAttempt(succeeded: true)
        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 3,
            timestamp: 1.5,
            state: .changing,
            stateChanged: true,
            significantChange: true
        )))
        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 4,
            timestamp: 2,
            state: .stable,
            stateChanged: true
        )))
        XCTAssertTrue(policy.shouldRecognize(evaluation: evaluation(
            revision: 5,
            timestamp: 4.5,
            state: .stable,
            stateChanged: false
        )))
        policy.completeAttempt(succeeded: false)
        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 6,
            timestamp: 4.7,
            state: .stable,
            stateChanged: false
        )))
        XCTAssertTrue(policy.shouldRecognize(evaluation: evaluation(
            revision: 7,
            timestamp: 7.5,
            state: .stable,
            stateChanged: false
        )))
        policy.completeAttempt(succeeded: true)
        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 8,
            timestamp: 8,
            state: .changing,
            stateChanged: true,
            significantChange: true
        )))
        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 9,
            timestamp: 8.2,
            state: .stable,
            stateChanged: true
        )))
    }

    func testStableTransitionWithoutSignificantChangeDoesNotTrigger() {
        var policy = ScreenTextTriggerPolicy(minimumRecognitionInterval: 0)

        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 1,
            timestamp: 0,
            state: .changing
        )))
        XCTAssertFalse(policy.shouldRecognize(evaluation: evaluation(
            revision: 2,
            timestamp: 1,
            state: .stable,
            stateChanged: true
        )))
    }

    func testSanitizerNormalizesDeduplicatesAndAppliesBounds() {
        let text = ScreenTextSanitizer.sanitize(
            ["  Hello   world  ", "hello world", "Second\nThird", "Fourth"],
            maximumLineCount: 3,
            maximumCharacterCount: 24
        )

        XCTAssertEqual(text, "Hello world\nSecond\nThird")
        XCTAssertLessThanOrEqual(text?.count ?? .max, 24)
        XCTAssertEqual(text?.components(separatedBy: .newlines).count, 3)
    }

    func testSecretScanRedactsLocalOCRAndProducesZeroCloudCandidates() throws {
        let observationEmitted = expectation(description: "redacted OCR emitted")
        let analysisCompleted = expectation(description: "secret analysis completed")
        let observations = TextObservationRecorder()
        let analyses = TextAnalysisRecorder()
        let recognizer = ScreenTextRecognizer(
            recognize: { _, _ in [
                "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz",
                "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abcdefghijk.lmnopqrstuv",
            ] },
            onAnalysis: { analysis in
                analyses.record(analysis)
                analysisCompleted.fulfill()
            },
            onObservation: { observation in
                observations.record(observation)
                observationEmitted.fulfill()
            }
        )
        recognizer.activate(generation: 1)

        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            generation: 1,
            context: context(revision: 1),
            applicationContext: applicationContext(),
            visionRegionOfInterest: Self.testRegion
        )
        wait(for: [observationEmitted, analysisCompleted], timeout: 2)

        XCTAssertEqual(observations.texts, [ScreenTextSecretScanner.redaction])
        XCTAssertTrue(analyses.containsSensitiveText)
        XCTAssertEqual(analyses.allowedCandidateCount, 0)
        XCTAssertFalse(observations.texts.joined().contains("sk-proj"))
        XCTAssertFalse(observations.texts.joined().contains("Bearer"))
        recognizer.invalidate(generation: 1)
    }

    func testEmptyOCRResultFailsClosedForCloudCandidate() throws {
        let analysisCompleted = expectation(description: "empty analysis completed")
        let analyses = TextAnalysisRecorder()
        let recognizer = ScreenTextRecognizer(
            recognize: { _, _ in [] },
            onAnalysis: { analysis in
                analyses.record(analysis)
                analysisCompleted.fulfill()
            },
            onObservation: { _ in XCTFail("empty OCR must not emit text") }
        )
        recognizer.activate(generation: 1)

        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            generation: 1,
            context: context(revision: 1),
            applicationContext: applicationContext(),
            visionRegionOfInterest: Self.testRegion
        )
        wait(for: [analysisCompleted], timeout: 2)

        XCTAssertFalse(analyses.recognizedText)
        XCTAssertEqual(analyses.allowedCandidateCount, 0)
        recognizer.invalidate(generation: 1)
    }

    func testSecretOutsideLocalOCRBoundsStillBlocksCloudCandidate() throws {
        let analysisCompleted = expectation(description: "bounded analysis completed")
        let analyses = TextAnalysisRecorder()
        let recognizer = ScreenTextRecognizer(
            configuration: ScreenTextRecognitionConfiguration(
                minimumRecognitionInterval: 0,
                maximumLineCount: 1,
                maximumCharacterCount: 20
            ),
            recognize: { _, _ in [
                "safe heading",
                "SERVICE_TOKEN=abcdefghijklmnopqrstuvwxyz012345",
            ] },
            onAnalysis: { analysis in
                analyses.record(analysis)
                analysisCompleted.fulfill()
            },
            onObservation: { _ in }
        )
        recognizer.activate(generation: 1)

        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            generation: 1,
            context: context(revision: 1),
            applicationContext: applicationContext(),
            visionRegionOfInterest: Self.testRegion
        )
        wait(for: [analysisCompleted], timeout: 2)

        XCTAssertTrue(analyses.containsSensitiveText)
        XCTAssertEqual(analyses.allowedCandidateCount, 0)
        recognizer.invalidate(generation: 1)
    }

    func testOCRDedupeKeyIncludesWindowContextToken() throws {
        let emitted = expectation(description: "same text emitted for two windows")
        emitted.expectedFulfillmentCount = 2
        let observations = TextObservationRecorder()
        let recognizer = ScreenTextRecognizer(
            recognize: { _, _ in ["same visible text"] },
            onObservation: { observation in
                observations.record(observation)
                emitted.fulfill()
            }
        )
        recognizer.activate(generation: 1)

        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            generation: 1,
            context: context(revision: 1),
            applicationContext: applicationContext(token: "WINDOW-1"),
            visionRegionOfInterest: Self.testRegion
        )
        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            generation: 1,
            context: context(revision: 2),
            applicationContext: applicationContext(token: "WINDOW-2"),
            visionRegionOfInterest: Self.testRegion
        )
        wait(for: [emitted], timeout: 2)

        XCTAssertEqual(observations.texts, ["same visible text", "same visible text"])
        recognizer.invalidate(generation: 1)
    }

    func testPrivacyGateDefaultsToDeny() {
        XCTAssertNil(ScreenTextPrivacyGates.denyAll(.init(
            revision: 1,
            observedAt: 2,
            displayID: 3
        )))
    }

    func testRecognizerKeepsOnlyNewestPendingFrame() throws {
        let firstStarted = expectation(description: "first recognition started")
        let twoObservations = expectation(description: "first and latest emitted")
        twoObservations.expectedFulfillmentCount = 2
        let releaseFirst = DispatchSemaphore(value: 0)
        let recorder = TextObservationRecorder()
        let recognizer = ScreenTextRecognizer(
            recognize: { pixelBuffer, region in
                XCTAssertEqual(region, Self.testRegion)
                let width = CVPixelBufferGetWidth(pixelBuffer)
                if width == 1 {
                    firstStarted.fulfill()
                    _ = releaseFirst.wait(timeout: .now() + 2)
                }
                return [String(width)]
            },
            onObservation: { observation in
                recorder.record(observation)
                twoObservations.fulfill()
            }
        )
        recognizer.activate(generation: 1)

        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 1),
            generation: 1,
            context: context(revision: 1),
            applicationContext: applicationContext(),
            visionRegionOfInterest: Self.testRegion
        )
        wait(for: [firstStarted], timeout: 2)
        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 2),
            generation: 1,
            context: context(revision: 2),
            applicationContext: applicationContext(),
            visionRegionOfInterest: Self.testRegion
        )
        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 3),
            generation: 1,
            context: context(revision: 3),
            applicationContext: applicationContext(),
            visionRegionOfInterest: Self.testRegion
        )
        releaseFirst.signal()
        wait(for: [twoObservations], timeout: 2)

        XCTAssertEqual(recorder.revisions, [1, 3])
        XCTAssertEqual(recorder.texts, ["1", "3"])
        recognizer.invalidate(generation: 1)
    }

    func testInvalidatedGenerationDropsCompletedResultAndProcessesNewestGeneration() throws {
        let staleStarted = expectation(description: "stale recognition started")
        let currentEmitted = expectation(description: "current generation emitted")
        let releaseStale = DispatchSemaphore(value: 0)
        let recorder = TextObservationRecorder()
        let recognizer = ScreenTextRecognizer(
            recognize: { pixelBuffer, region in
                XCTAssertEqual(region, Self.testRegion)
                let width = CVPixelBufferGetWidth(pixelBuffer)
                if width == 1 {
                    staleStarted.fulfill()
                    _ = releaseStale.wait(timeout: .now() + 2)
                }
                return [String(width)]
            },
            onObservation: { observation in
                recorder.record(observation)
                currentEmitted.fulfill()
            }
        )
        recognizer.activate(generation: 1)
        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 1),
            generation: 1,
            context: context(revision: 1),
            applicationContext: applicationContext(),
            visionRegionOfInterest: Self.testRegion
        )
        wait(for: [staleStarted], timeout: 2)

        recognizer.invalidate(generation: 1)
        recognizer.activate(generation: 2)
        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 2),
            generation: 2,
            context: context(revision: 2),
            applicationContext: applicationContext(),
            visionRegionOfInterest: Self.testRegion
        )
        releaseStale.signal()
        wait(for: [currentEmitted], timeout: 2)

        XCTAssertEqual(recorder.revisions, [2])
        XCTAssertEqual(recorder.texts, ["2"])
        recognizer.invalidate(generation: 2)
    }

    func testRevokedPrivacyAuthorizationDropsInFlightRecognition() throws {
        let started = expectation(description: "recognition started")
        let completed = expectation(description: "recognition completed")
        let notEmitted = expectation(description: "revoked result not emitted")
        notEmitted.isInverted = true
        let release = DispatchSemaphore(value: 0)
        let authorization = MutableAuthorizationState(allowed: true)
        let recognizer = ScreenTextRecognizer(
            recognize: { _, _ in
                started.fulfill()
                _ = release.wait(timeout: .now() + 2)
                completed.fulfill()
                return ["private text"]
            },
            isStillAuthorized: { _, _ in authorization.allowed },
            onObservation: { _ in notEmitted.fulfill() }
        )
        recognizer.activate(generation: 1)
        recognizer.submit(
            pixelBuffer: try pixelBuffer(width: 10),
            generation: 1,
            context: context(revision: 1),
            applicationContext: applicationContext(),
            visionRegionOfInterest: Self.testRegion
        )
        wait(for: [started], timeout: 2)

        authorization.allowed = false
        release.signal()
        wait(for: [completed, notEmitted], timeout: 0.2)
        recognizer.invalidate(generation: 1)
    }

    func testObservationEncodingContainsOnlyStructuredTextFields() throws {
        let observation = ScreenTextObservation(
            revision: 7,
            observedAt: 10,
            displayID: 2,
            text: "local text",
            applicationContext: applicationContext()
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(observation))
                as? [String: Any]
        )
        XCTAssertEqual(
            Set(object.keys),
            Set([
                "source", "kind", "revision", "observedAt", "displayID", "text",
                "appBundleId", "appName", "windowKey",
            ])
        )
        XCTAssertEqual(object["source"] as? String, "screen")
        XCTAssertEqual(object["kind"] as? String, "screen_text")
        XCTAssertEqual(object["revision"] as? Int, 7)
        XCTAssertEqual(object["displayID"] as? Int, 2)
        XCTAssertEqual(object["text"] as? String, "local text")
        XCTAssertEqual(object["appBundleId"] as? String, "com.example.editor")
        XCTAssertEqual(object["appName"] as? String, "Editor")
        XCTAssertEqual(object["windowKey"] as? String, "00000000-0000-0000-0000-000000000040")
    }

    private func evaluation(
        revision: UInt64,
        timestamp: TimeInterval,
        state: ScreenMotionState,
        stateChanged: Bool = false,
        significantChange: Bool = false
    ) -> ScreenChangeEvaluation {
        ScreenChangeEvaluation(
            revision: revision,
            timestamp: timestamp,
            dirtyAreaRatio: significantChange ? 0.5 : 0,
            dirtyRectCount: significantChange ? 1 : 0,
            state: state,
            stateChanged: stateChanged,
            significantChange: significantChange
        )
    }

    private func context(revision: UInt64) -> ScreenTextPrivacyContext {
        ScreenTextPrivacyContext(
            revision: revision,
            observedAt: Double(revision),
            displayID: 1
        )
    }

    private func applicationContext(
        token: String = "00000000-0000-0000-0000-000000000040"
    ) -> ScreenTextApplicationContext {
        ScreenTextApplicationContext(
            processIdentifier: 7,
            appBundleId: "com.example.editor",
            appName: "Editor",
            windowTitle: "Document",
            windowFrame: AccessibilityWindowFrame(
                x: 10,
                y: 20,
                width: 800,
                height: 600
            ),
            contextToken: token
        )
    }

    private static let testRegion = CGRect(
        x: 0.1,
        y: 0.2,
        width: 0.7,
        height: 0.6
    )

    private func pixelBuffer(width: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            1,
            kCVPixelFormatType_32BGRA,
            nil,
            &buffer
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        return try XCTUnwrap(buffer)
    }
}

private final class TextAnalysisRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var analyses: [ScreenTextAnalyzedFrame] = []

    var containsSensitiveText: Bool {
        lock.lock()
        defer { lock.unlock() }
        return analyses.last?.containsSensitiveText ?? false
    }

    var recognizedText: Bool {
        lock.lock()
        defer { lock.unlock() }
        return analyses.last?.recognizedText ?? false
    }

    var allowedCandidateCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return analyses.filter(ScreenTextCandidatePolicy.allowsVisionCandidate).count
    }

    func record(_ analysis: ScreenTextAnalyzedFrame) {
        lock.lock()
        analyses.append(analysis)
        lock.unlock()
    }
}

private final class MutableAuthorizationState: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Bool

    init(allowed: Bool) {
        value = allowed
    }

    var allowed: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
        set {
            lock.lock()
            value = newValue
            lock.unlock()
        }
    }
}

private final class TextObservationRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var observations: [ScreenTextObservation] = []

    var revisions: [UInt64] {
        lock.lock()
        defer { lock.unlock() }
        return observations.map(\.revision)
    }

    var texts: [String] {
        lock.lock()
        defer { lock.unlock() }
        return observations.map(\.text)
    }

    func record(_ observation: ScreenTextObservation) {
        lock.lock()
        observations.append(observation)
        lock.unlock()
    }
}
