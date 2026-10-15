@preconcurrency import Foundation
@testable import FishMemoryLiveKitAudioHelper
import XCTest

final class AcousticLowPowerGateTests: XCTestCase {
    func testProcessedDwellAllowsTwoIsolatedOutliersPerBlock() {
        var gate = AcousticLowPowerGate(configuration: gateConfiguration(
            processedFramesPerBlock: 10,
            processedQuietFramesPerBlock: 8,
            processedQuietBlocksRequired: 2
        ))

        var requests: [AcousticLowPowerGate.Request] = []
        for _ in 0 ..< 2 {
            for index in 0 ..< 10 {
                let levels = index < 8
                    ? AcousticFrameLevels(meanAbsolute: 20, peak: 100)
                    : AcousticFrameLevels(meanAbsolute: 30, peak: 150)
                requests.append(gate.observe(levels, bypassed: false).request)
            }
        }

        XCTAssertEqual(requests.filter { $0 == .bypassProcessing }.count, 1)
        XCTAssertEqual(requests.last, .bypassProcessing)
    }

    func testStrongFrameOrSustainedSoftActivityBreaksProcessedDwell() {
        var gate = AcousticLowPowerGate(configuration: gateConfiguration(
            processedFramesPerBlock: 10,
            processedQuietFramesPerBlock: 8,
            processedQuietBlocksRequired: 2
        ))

        for index in 0 ..< 10 {
            let levels = index == 5
                ? AcousticFrameLevels(meanAbsolute: 60, peak: 500)
                : AcousticFrameLevels(meanAbsolute: 20, peak: 100)
            XCTAssertEqual(
                gate.observe(levels, bypassed: false).request,
                .none
            )
        }
        for index in 0 ..< 10 {
            let levels = index < 7
                ? AcousticFrameLevels(meanAbsolute: 20, peak: 100)
                : AcousticFrameLevels(meanAbsolute: 32, peak: 180)
            XCTAssertEqual(
                gate.observe(levels, bypassed: false).request,
                .none
            )
        }
        emitGateFrames(
            count: 19,
            levels: AcousticFrameLevels(meanAbsolute: 20, peak: 100),
            bypassed: false,
            gate: &gate
        ) { request in
            XCTAssertEqual(request, .none)
        }
        XCTAssertEqual(
            gate.observe(
                AcousticFrameLevels(meanAbsolute: 20, peak: 100),
                bypassed: false
            ).request,
            .bypassProcessing
        )
    }

    func testMeasuredRawNoiseCalibratesAndDoesNotImmediatelyRestore() {
        var gate = AcousticLowPowerGate(configuration: .production)
        let settling: [AcousticFrameLevels] = (0 ..< 10).map { index in
            AcousticFrameLevels(
                meanAbsolute: Int32(index * 7),
                peak: Int32(index * 23)
            )
        }
        let calibration = (0 ..< 90).map { index in
            AcousticFrameLevels(
                meanAbsolute: [62, 67, 73, 82, 67][index % 5],
                peak: [210, 223, 260, 300, 223][index % 5]
            )
        }
        for levels in settling + calibration {
            XCTAssertEqual(
                gate.observe(levels, bypassed: true).request,
                .none
            )
        }
        XCTAssertEqual(
            gate.rawBaselineForTesting,
            AcousticFrameLevels(meanAbsolute: 70, peak: 243)
        )

        // Representative P50/P90/P99/max envelope values. Replaying 97.99
        // seconds after entry models the measured 100-second trace.
        let quietRaw = [
            AcousticFrameLevels(meanAbsolute: 67, peak: 223),
            AcousticFrameLevels(meanAbsolute: 82, peak: 280),
            AcousticFrameLevels(meanAbsolute: 102, peak: 339),
            AcousticFrameLevels(meanAbsolute: 67, peak: 223),
            AcousticFrameLevels(meanAbsolute: 123, peak: 419),
            AcousticFrameLevels(meanAbsolute: 67, peak: 223),
        ]
        for index in 0 ..< 9_799 {
            XCTAssertEqual(
                gate.observe(
                    quietRaw[index % quietRaw.count],
                    bypassed: true
                ).request,
                .none
            )
        }
    }

    func testRawMeanNeedsThreeFramesAndPeakRestoresImmediately() {
        var meanGate = calibratedGate(mean: 67, peak: 223)
        let meanActivity = AcousticFrameLevels(meanAbsolute: 134, peak: 300)
        XCTAssertEqual(meanGate.observe(meanActivity, bypassed: true).request, .none)
        XCTAssertEqual(meanGate.observe(meanActivity, bypassed: true).request, .none)
        XCTAssertEqual(
            meanGate.observe(meanActivity, bypassed: true).request,
            .restoreProcessing
        )

        var peakGate = calibratedGate(mean: 67, peak: 223)
        XCTAssertEqual(
            peakGate.observe(
                AcousticFrameLevels(meanAbsolute: 80, peak: 669),
                bypassed: true
            ).request,
            .restoreProcessing
        )
    }

    func testRawHardActivityEscapesDuringCalibration() {
        var gate = AcousticLowPowerGate(configuration: gateConfiguration())
        XCTAssertEqual(
            gate.observe(
                AcousticFrameLevels(meanAbsolute: 512, peak: 600),
                bypassed: true
            ).request,
            .restoreProcessing
        )
    }

    func testLongerMeasuredRawDriftCanBecomeANewCalibrationFloor() {
        var gate = AcousticLowPowerGate(configuration: .production)
        let driftedQuiet = AcousticFrameLevels(meanAbsolute: 285, peak: 675)
        emitGateFrames(
            count: 100,
            levels: driftedQuiet,
            bypassed: true,
            gate: &gate
        ) { request in
            XCTAssertEqual(request, .none)
        }
        XCTAssertEqual(gate.rawBaselineForTesting, driftedQuiet)
        emitGateFrames(
            count: 20,
            levels: driftedQuiet,
            bypassed: true,
            gate: &gate
        ) { request in
            XCTAssertEqual(request, .none)
        }
    }

    func testRawBaselineStaysFrozenForTheBypassInterval() {
        var gate = calibratedGate(mean: 80, peak: 240)
        emitGateFrames(
            count: 2,
            levels: AcousticFrameLevels(meanAbsolute: 60, peak: 200),
            bypassed: true,
            gate: &gate
        ) { request in
            XCTAssertEqual(request, .none)
        }
        XCTAssertEqual(
            gate.rawBaselineForTesting,
            AcousticFrameLevels(meanAbsolute: 80, peak: 240)
        )

        emitGateFrames(
            count: 2,
            levels: AcousticFrameLevels(meanAbsolute: 80, peak: 230),
            bypassed: true,
            gate: &gate
        ) { request in
            XCTAssertEqual(request, .none)
        }
        XCTAssertEqual(
            gate.rawBaselineForTesting,
            AcousticFrameLevels(meanAbsolute: 80, peak: 240)
        )
    }

    private func calibratedGate(mean: Int32, peak: Int32) -> AcousticLowPowerGate {
        var gate = AcousticLowPowerGate(configuration: gateConfiguration())
        emitGateFrames(
            count: 6,
            levels: AcousticFrameLevels(meanAbsolute: mean, peak: peak),
            bypassed: true,
            gate: &gate
        ) { request in
            XCTAssertEqual(request, .none)
        }
        return gate
    }
}

final class AdaptiveVoiceProcessingControllerTests: XCTestCase {
    func testQuietDwellAndMinimumProcessedLeaseGateBypass() {
        let clock = TestMonotonicClock()
        let device = BypassDevice()
        let controller = makeController(clock: clock, device: device)

        emit(value: 0, count: 4, startingAt: 0, to: controller)
        controller.drainForTesting()
        XCTAssertTrue(device.requests.isEmpty)

        clock.advance(by: 10)
        emit(value: 0, count: 4, startingAt: 4, to: controller)
        controller.drainForTesting()
        XCTAssertEqual(device.requests, [true])
        XCTAssertTrue(controller.snapshot.actualBypass)
    }

    func testCalibratedRawActivityRestoresAndUsesFixedProcessedLease() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        let transitions = TransitionRecorder()
        let controller = makeController(
            clock: clock,
            device: device,
            transitions: transitions
        )
        enterBypass(controller, clock: clock)
        emit(value: 67, count: 6, startingAt: 10, to: controller)
        emit(value: 134, count: 3, startingAt: 16, to: controller)
        controller.drainForTesting()

        XCTAssertEqual(device.requests, [true, false])
        XCTAssertFalse(controller.snapshot.actualBypass)
        XCTAssertEqual(transitions.values.last?.reason, .acousticActivity)

        clock.advance(by: 9)
        emit(value: 0, count: 4, startingAt: 20, to: controller)
        controller.drainForTesting()
        XCTAssertEqual(device.requests, [true, false])
        clock.advance(by: 1)
        emit(value: 0, count: 4, startingAt: 24, to: controller)
        controller.drainForTesting()
        XCTAssertEqual(device.requests, [true, false, true])
    }

    func testSpeechDuringEntryVerificationRollsBackFailActive() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        let controllerBox = ControllerBox()
        device.onRequest = { requested in
            guard requested, let controller = controllerBox.value else { return }
            emit(value: 500, sequence: 99, to: controller)
        }
        let controller = makeController(clock: clock, device: device)
        controllerBox.value = controller

        clock.advance(by: 10)
        emit(value: 0, count: 4, startingAt: 0, to: controller)
        controller.drainForTesting()
        XCTAssertEqual(device.requests, [true, false])
        XCTAssertFalse(controller.snapshot.actualBypass)
    }

    func testEnsureProcessedRestoresWithoutPlaybackGenerationAndExtendsLease() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        let transitions = TransitionRecorder()
        let controller = makeController(
            clock: clock,
            device: device,
            transitions: transitions
        )
        enterBypass(controller, clock: clock)
        // Speech can occupy the short raw calibration window without crossing
        // its hard escape. The KWS/VAD opcode must still restore VPIO.
        emit(value: 300, count: 6, startingAt: 10, to: controller)
        controller.drainForTesting()
        XCTAssertTrue(controller.snapshot.actualBypass)

        XCTAssertTrue(controller.ensureProcessed())
        XCTAssertEqual(device.requests, [true, false])
        XCTAssertNil(controller.snapshot.playbackGeneration)
        XCTAssertEqual(controller.snapshot.playbackHoldCount, 0)
        XCTAssertEqual(transitions.values.last?.reason, .voiceActivity)

        clock.advance(by: 9)
        XCTAssertTrue(controller.ensureProcessed())
        XCTAssertEqual(device.requests, [true, false, false])
        clock.advance(by: 9)
        emit(value: 0, count: 4, startingAt: 20, to: controller)
        controller.drainForTesting()
        XCTAssertEqual(device.requests, [true, false, false])
    }

    func testPlaybackPrewarmRestoresWithoutLeaseAndStaleEndCannotReleaseActive() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        let controller = makeController(clock: clock, device: device)
        enterBypass(controller, clock: clock)

        XCTAssertTrue(controller.prewarmPlayback(generation: 7))
        XCTAssertEqual(device.requests, [true, false])
        XCTAssertNil(controller.snapshot.playbackGeneration)

        XCTAssertTrue(controller.beginPlaybackProtection(generation: 7))
        XCTAssertEqual(controller.snapshot.playbackGeneration, 7)
        XCTAssertTrue(controller.beginPlaybackProtection(generation: 7))
        XCTAssertEqual(device.requests, [true, false, false])
        controller.endPlaybackProtection(generation: 6)
        XCTAssertEqual(controller.snapshot.playbackGeneration, 7)
        controller.endPlaybackProtection(generation: 7)
        XCTAssertNil(controller.snapshot.playbackGeneration)
    }

    func testDiagnosticForcedBypassIgnoresAcousticGate() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        let controller = makeController(clock: clock, device: device)

        XCTAssertTrue(controller.setDiagnosticBypass(true).actual)
        emit(value: 1_000, count: 8, startingAt: 0, to: controller)
        controller.drainForTesting()
        XCTAssertEqual(device.requests, [true])
        XCTAssertTrue(controller.snapshot.actualBypass)

        XCTAssertTrue(controller.ensureProcessed())
        XCTAssertEqual(device.requests, [true, false])
        XCTAssertFalse(controller.snapshot.actualBypass)
    }

    func testPlaybackFailsClosedWhenPlatformCannotRestore() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        let failures = FailureRecorder()
        let controller = makeController(
            clock: clock,
            device: device,
            failures: failures
        )
        enterBypass(controller, clock: clock)
        device.refuseRestore = true

        XCTAssertFalse(controller.beginPlaybackProtection(generation: 8))
        // The synchronous playback caller owns the single fatal notification.
        XCTAssertTrue(failures.values.isEmpty)
        XCTAssertEqual(controller.snapshot.playbackHoldCount, 0)
    }

    func testFailedBypassVerificationDisablesAdaptationAndRollsBack() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        device.refuseBypass = true
        let controller = makeController(clock: clock, device: device)

        clock.advance(by: 10)
        emit(value: 0, count: 4, startingAt: 0, to: controller)
        controller.drainForTesting()
        XCTAssertTrue(controller.snapshot.adaptationDisabled)
        XCTAssertFalse(controller.snapshot.actualBypass)
        XCTAssertEqual(device.requests, [true, false])

        emit(value: 0, count: 8, startingAt: 10, to: controller)
        controller.drainForTesting()
        XCTAssertEqual(device.requests, [true, false])
    }

    func testBypassRequiresRequestedAndActualReadbackToMatch() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        device.bypassRequestedReadback = false
        let controller = makeController(clock: clock, device: device)

        clock.advance(by: 10)
        emit(value: 0, count: 4, startingAt: 0, to: controller)
        controller.drainForTesting()

        XCTAssertEqual(device.requests, [true, false])
        XCTAssertTrue(controller.snapshot.adaptationDisabled)
        XCTAssertFalse(controller.snapshot.actualBypass)
    }

    func testRestoreRejectsActualFalseWhenRequestedReadbackStaysTrue() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        let failures = FailureRecorder()
        let controller = makeController(
            clock: clock,
            device: device,
            failures: failures
        )
        enterBypass(controller, clock: clock)
        device.restoreRequestedReadback = true

        XCTAssertFalse(controller.beginPlaybackProtection(generation: 8))
        XCTAssertTrue(failures.values.isEmpty)
        XCTAssertFalse(controller.snapshot.actualBypass)
        XCTAssertEqual(controller.snapshot.playbackHoldCount, 0)
    }

    func testStopFencesLateFrames() {
        let clock = TestMonotonicClock(now: 10)
        let device = BypassDevice()
        let controller = makeController(clock: clock, device: device)
        controller.stop()
        emit(value: 0, count: 8, startingAt: 0, to: controller)
        controller.drainForTesting()
        XCTAssertTrue(device.requests.isEmpty)
    }

    private func makeController(
        clock: TestMonotonicClock,
        device: BypassDevice,
        transitions: TransitionRecorder = TransitionRecorder(),
        failures: FailureRecorder = FailureRecorder()
    ) -> AdaptiveVoiceProcessingController {
        AdaptiveVoiceProcessingController(
            automaticLowPowerEnabled: true,
            configuration: controllerConfiguration(),
            applyBypass: device.apply,
            clock: clock.read,
            verificationSleep: { _ in },
            onTransition: transitions.append,
            onInvariantFailure: failures.append
        )
    }

    private func enterBypass(
        _ controller: AdaptiveVoiceProcessingController,
        clock: TestMonotonicClock
    ) {
        clock.advance(by: 10)
        emit(value: 0, count: 4, startingAt: 0, to: controller)
        controller.drainForTesting()
        XCTAssertTrue(controller.snapshot.actualBypass)
    }
}

private func gateConfiguration(
    processedFramesPerBlock: Int32 = 2,
    processedQuietFramesPerBlock: Int32 = 2,
    processedQuietBlocksRequired: Int32 = 2
) -> AcousticLowPowerGate.Configuration {
    AcousticLowPowerGate.Configuration(
        processedFramesPerBlock: processedFramesPerBlock,
        processedQuietFramesPerBlock: processedQuietFramesPerBlock,
        processedQuietBlocksRequired: processedQuietBlocksRequired,
        processedQuietMeanAbsoluteLimit: 24,
        processedQuietPeakLimit: 130,
        processedStrongMeanAbsoluteThreshold: 60,
        processedStrongPeakThreshold: 500,
        rawCalibrationFrames: 6,
        rawCalibrationSettlingFrames: 2,
        rawHardMeanAbsoluteThreshold: 512,
        rawHardPeakThreshold: 1_536,
        rawMeanActivityFramesRequired: 3
    )
}

private func controllerConfiguration() -> VoiceProcessingLowPowerConfiguration {
    VoiceProcessingLowPowerConfiguration(
        minimumProcessedNanoseconds: 10,
        verificationAttempts: 1,
        verificationIntervalMicroseconds: 0,
        acoustic: gateConfiguration()
    )
}

private func emitGateFrames(
    count: Int,
    levels: AcousticFrameLevels,
    bypassed: Bool,
    gate: inout AcousticLowPowerGate,
    verify: (AcousticLowPowerGate.Request) -> Void
) {
    for _ in 0 ..< count {
        verify(gate.observe(levels, bypassed: bypassed).request)
    }
}

private func emit(
    value: Int16,
    count: Int,
    startingAt sequence: UInt64,
    to controller: AdaptiveVoiceProcessingController
) {
    for offset in 0 ..< count {
        emit(
            value: value,
            sequence: sequence + UInt64(offset),
            to: controller
        )
    }
}

private func emit(
    value: Int16,
    sequence: UInt64,
    to controller: AdaptiveVoiceProcessingController
) {
    let samples = [Int16](repeating: value, count: 160)
    samples.withUnsafeBufferPointer {
        controller.observeMicrophoneFrame($0, captureSequence: sequence)
    }
}

private final class TestMonotonicClock: @unchecked Sendable {
    private let lock = NSLock()
    private var value: UInt64

    init(now: UInt64 = 0) { value = now }

    func read() -> UInt64 { lock.withLock { value } }

    func advance(by delta: UInt64) {
        lock.withLock { value += delta }
    }
}

private final class BypassDevice: @unchecked Sendable {
    private let lock = NSLock()
    private var actual = false
    private var recordedRequests: [Bool] = []
    var refuseBypass = false
    var refuseRestore = false
    var bypassRequestedReadback: Bool?
    var restoreRequestedReadback: Bool?
    var onRequest: (@Sendable (Bool) -> Void)?

    var requests: [Bool] { lock.withLock { recordedRequests } }

    func apply(_ requested: Bool) -> VoiceProcessingBypassApplyResult {
        let callback: (@Sendable (Bool) -> Void)? = lock.withLock {
            recordedRequests.append(requested)
            if requested {
                if !refuseBypass { actual = true }
            } else if !refuseRestore {
                actual = false
            }
            return onRequest
        }
        callback?(requested)
        return lock.withLock {
            VoiceProcessingBypassApplyResult(
                requested: requested
                    ? bypassRequestedReadback ?? true
                    : restoreRequestedReadback ?? false,
                actual: actual
            )
        }
    }
}

private final class TransitionRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [VoiceProcessingPowerTransition] = []

    var values: [VoiceProcessingPowerTransition] {
        lock.withLock { recorded }
    }

    func append(_ transition: VoiceProcessingPowerTransition) {
        lock.withLock { recorded.append(transition) }
    }
}

private final class FailureRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [AdaptiveVoiceProcessingError] = []

    var values: [AdaptiveVoiceProcessingError] {
        lock.withLock { recorded }
    }

    func append(_ error: AdaptiveVoiceProcessingError) {
        lock.withLock { recorded.append(error) }
    }
}

private final class ControllerBox: @unchecked Sendable {
    var value: AdaptiveVoiceProcessingController?
}
