import Darwin
import Dispatch
import Foundation

struct AcousticFrameLevels: Equatable, Sendable {
    let meanAbsolute: Int32
    let peak: Int32
}

/// Deterministic envelope gate for the fixed 10 ms microphone frames.
///
/// Apple VPIO changes the microphone noise floor when processing is bypassed,
/// so processed and raw samples deliberately use different policies. This
/// value is owned by the serialized capture callback. It allocates no memory,
/// takes no locks, and never invokes platform APIs.
struct AcousticLowPowerGate: Sendable {
    struct Configuration: Equatable, Sendable {
        let processedFramesPerBlock: Int32
        let processedQuietFramesPerBlock: Int32
        let processedQuietBlocksRequired: Int32
        let processedQuietMeanAbsoluteLimit: Int32
        let processedQuietPeakLimit: Int32
        let processedStrongMeanAbsoluteThreshold: Int32
        let processedStrongPeakThreshold: Int32
        let rawCalibrationFrames: Int32
        let rawCalibrationSettlingFrames: Int32
        let rawHardMeanAbsoluteThreshold: Int32
        let rawHardPeakThreshold: Int32
        let rawMeanActivityFramesRequired: Int32

        static let production = Configuration(
            // One 100 ms block may contain two isolated noisy frames. Sustained
            // soft speech still breaks the block and keeps VPIO processed.
            processedFramesPerBlock: 10,
            processedQuietFramesPerBlock: 8,
            processedQuietBlocksRequired: 50,
            processedQuietMeanAbsoluteLimit: 24,
            processedQuietPeakLimit: 130,
            processedStrongMeanAbsoluteThreshold: 60,
            processedStrongPeakThreshold: 500,
            // Raw input settled after about 80 ms in the measured transition.
            // Discard 100 ms, then average the following 900 ms once. The
            // resulting baseline stays frozen for this bypass interval.
            rawCalibrationFrames: 100,
            rawCalibrationSettlingFrames: 10,
            // Fail-active escape during calibration. The longer 36-second raw
            // trace reached mean=285 and peak=675 during slow environmental
            // drift, so leave enough headroom to calibrate a changed floor.
            rawHardMeanAbsoluteThreshold: 512,
            rawHardPeakThreshold: 1_536,
            rawMeanActivityFramesRequired: 3
        )

        var processedQuietMilliseconds: Int32 {
            processedFramesPerBlock * processedQuietBlocksRequired * 10
        }
    }

    enum Request: Equatable, Sendable {
        case none
        case bypassProcessing
        case restoreProcessing
    }

    struct Observation: Equatable, Sendable {
        let request: Request
        let invalidatesPendingBypass: Bool
    }

    private enum InputMode: Sendable {
        case processed
        case rawCalibration
        case rawMonitoring
    }

    private let configuration: Configuration
    private var mode: InputMode = .processed
    private var lastBypassed = false

    private var processedBlockFrames: Int32 = 0
    private var processedQuietFrames: Int32 = 0
    private var processedBlockHasStrongActivity = false
    private var processedQuietBlocks: Int32 = 0

    private var rawCalibrationFrameCount: Int32 = 0
    private var rawCalibrationMeanTotal: Int64 = 0
    private var rawCalibrationPeakTotal: Int64 = 0
    private var rawCalibrationSampleCount: Int32 = 0
    private var rawBaselineMeanAbsolute: Int32 = 1
    private var rawBaselinePeak: Int32 = 1
    private var rawMeanActivityFrames: Int32 = 0

    init(configuration: Configuration = .production) {
        precondition(configuration.processedFramesPerBlock > 0)
        precondition(
            configuration.processedQuietFramesPerBlock > 0 &&
                configuration.processedQuietFramesPerBlock <=
                configuration.processedFramesPerBlock
        )
        precondition(configuration.processedQuietBlocksRequired > 0)
        precondition(
            configuration.rawCalibrationSettlingFrames >= 0 &&
                configuration.rawCalibrationSettlingFrames <
                configuration.rawCalibrationFrames
        )
        precondition(configuration.rawMeanActivityFramesRequired > 0)
        self.configuration = configuration
    }

    mutating func observe(
        _ levels: AcousticFrameLevels,
        bypassed: Bool
    ) -> Observation {
        if bypassed != lastBypassed {
            lastBypassed = bypassed
            if bypassed {
                resetRawCalibration()
                mode = .rawCalibration
            } else {
                resetProcessedDwell()
                mode = .processed
            }
        }

        switch mode {
        case .processed:
            return observeProcessed(levels)
        case .rawCalibration:
            return observeRawCalibration(levels)
        case .rawMonitoring:
            return observeRawMonitoring(levels)
        }
    }

    var rawBaselineForTesting: AcousticFrameLevels? {
        guard mode == .rawMonitoring else { return nil }
        return AcousticFrameLevels(
            meanAbsolute: rawBaselineMeanAbsolute,
            peak: rawBaselinePeak
        )
    }

    private mutating func observeProcessed(
        _ levels: AcousticFrameLevels
    ) -> Observation {
        let isQuiet =
            levels.meanAbsolute <=
            configuration.processedQuietMeanAbsoluteLimit &&
            levels.peak <= configuration.processedQuietPeakLimit
        let isStrong =
            levels.meanAbsolute >=
            configuration.processedStrongMeanAbsoluteThreshold ||
            levels.peak >= configuration.processedStrongPeakThreshold

        processedBlockFrames += 1
        if isQuiet {
            processedQuietFrames += 1
        }
        processedBlockHasStrongActivity =
            processedBlockHasStrongActivity || isStrong

        var request: Request = .none
        if processedBlockFrames >= configuration.processedFramesPerBlock {
            let blockIsQuiet =
                processedQuietFrames >=
                configuration.processedQuietFramesPerBlock &&
                !processedBlockHasStrongActivity
            processedQuietBlocks = blockIsQuiet
                ? processedQuietBlocks + 1
                : 0
            resetProcessedBlock()
            if processedQuietBlocks >=
                configuration.processedQuietBlocksRequired {
                processedQuietBlocks = 0
                request = .bypassProcessing
            }
        }

        return Observation(
            request: request,
            invalidatesPendingBypass: !isQuiet
        )
    }

    private mutating func observeRawCalibration(
        _ levels: AcousticFrameLevels
    ) -> Observation {
        if isHardRawActivity(levels) {
            return Observation(
                request: .restoreProcessing,
                invalidatesPendingBypass: true
            )
        }

        rawCalibrationFrameCount += 1
        if rawCalibrationFrameCount >
            configuration.rawCalibrationSettlingFrames {
            rawCalibrationMeanTotal += Int64(levels.meanAbsolute)
            rawCalibrationPeakTotal += Int64(levels.peak)
            rawCalibrationSampleCount += 1
        }

        if rawCalibrationFrameCount >= configuration.rawCalibrationFrames {
            let count = max(1, rawCalibrationSampleCount)
            rawBaselineMeanAbsolute = max(
                1,
                Int32(rawCalibrationMeanTotal / Int64(count))
            )
            rawBaselinePeak = max(
                1,
                Int32(rawCalibrationPeakTotal / Int64(count))
            )
            rawMeanActivityFrames = 0
            mode = .rawMonitoring
        }

        return Observation(
            request: .none,
            invalidatesPendingBypass: false
        )
    }

    private mutating func observeRawMonitoring(
        _ levels: AcousticFrameLevels
    ) -> Observation {
        if isHardRawActivity(levels) || isRawPeakActivity(levels) {
            rawMeanActivityFrames = 0
            return Observation(
                request: .restoreProcessing,
                invalidatesPendingBypass: true
            )
        }

        if Int64(levels.meanAbsolute) >=
            Int64(rawBaselineMeanAbsolute) * 2 {
            rawMeanActivityFrames += 1
        } else {
            rawMeanActivityFrames = 0
        }

        if rawMeanActivityFrames >=
            configuration.rawMeanActivityFramesRequired {
            rawMeanActivityFrames = 0
            return Observation(
                request: .restoreProcessing,
                invalidatesPendingBypass: true
            )
        }
        return Observation(
            request: .none,
            invalidatesPendingBypass: false
        )
    }

    private func isHardRawActivity(_ levels: AcousticFrameLevels) -> Bool {
        levels.meanAbsolute >=
            configuration.rawHardMeanAbsoluteThreshold ||
            levels.peak >= configuration.rawHardPeakThreshold
    }

    private func isRawPeakActivity(_ levels: AcousticFrameLevels) -> Bool {
        Int64(levels.peak) >= Int64(rawBaselinePeak) * 3
    }

    private mutating func resetProcessedDwell() {
        resetProcessedBlock()
        processedQuietBlocks = 0
    }

    private mutating func resetProcessedBlock() {
        processedBlockFrames = 0
        processedQuietFrames = 0
        processedBlockHasStrongActivity = false
    }

    private mutating func resetRawCalibration() {
        rawCalibrationFrameCount = 0
        rawCalibrationMeanTotal = 0
        rawCalibrationPeakTotal = 0
        rawCalibrationSampleCount = 0
        rawBaselineMeanAbsolute = 1
        rawBaselinePeak = 1
        rawMeanActivityFrames = 0
    }
}

struct VoiceProcessingLowPowerConfiguration: Equatable, Sendable {
    let minimumProcessedNanoseconds: UInt64
    let verificationAttempts: Int
    let verificationIntervalMicroseconds: useconds_t
    let acoustic: AcousticLowPowerGate.Configuration

    static let production = VoiceProcessingLowPowerConfiguration(
        minimumProcessedNanoseconds: 10_000_000_000,
        verificationAttempts: 6,
        verificationIntervalMicroseconds: 5_000,
        acoustic: .production
    )
}

struct VoiceProcessingBypassApplyResult: Equatable, Sendable {
    let requested: Bool
    let actual: Bool
}

enum VoiceProcessingTransitionReason: String, Sendable {
    case quietStandby = "quiet-standby"
    case acousticActivity = "acoustic-activity"
    case voiceActivity = "voice-activity"
    case playbackPrewarm = "playback-prewarm"
    case playback = "playback"
    case diagnostic = "diagnostic"
    case failActiveRollback = "fail-active-rollback"
}

struct VoiceProcessingPowerTransition: Equatable, Sendable {
    let uptimeNanoseconds: UInt64
    let captureSequence: UInt64?
    let reason: VoiceProcessingTransitionReason
    let requestedBypass: Bool
    let sdkRequestedBypass: Bool
    let actualBypass: Bool
    let framesBeforeApplied: UInt64
    let playbackGeneration: UInt64?
    let playbackHoldCount: Int

    var succeeded: Bool {
        sdkRequestedBypass == requestedBypass &&
            actualBypass == requestedBypass
    }

    var diagnosticDescription: String {
        let capture = captureSequence.map(String.init) ?? "none"
        let generation = playbackGeneration.map(String.init) ?? "none"
        return "adaptive-vpio transition uptimeNs=\(uptimeNanoseconds) " +
            "captureSequence=\(capture) " +
            "framesBeforeApplied=\(framesBeforeApplied) " +
            "reason=\(reason.rawValue) " +
            "requestedBypassed=\(requestedBypass) " +
            "sdkRequestedBypassed=\(sdkRequestedBypass) " +
            "actualBypassed=\(actualBypass) " +
            "playbackGeneration=\(generation) " +
            "holdCount=\(playbackHoldCount) succeeded=\(succeeded)"
    }
}

enum AdaptiveVoiceProcessingError: Error, Equatable {
    case failedToRestore(reason: VoiceProcessingTransitionReason)
}

protocol VoiceProcessingPlaybackProtecting: Sendable {
    func ensureProcessed() -> Bool
    func prewarmPlayback(generation: UInt64) -> Bool
    func beginPlaybackProtection(generation: UInt64) -> Bool
    func endPlaybackProtection(generation: UInt64)
}

/// Keeps the capture graph running while bypassing only Apple's costly AEC/NS
/// processing during sustained quiet standby. Platform mutation, readback,
/// retries, logging, and playback barriers stay off the realtime callback.
final class AdaptiveVoiceProcessingController:
    VoiceProcessingPlaybackProtecting, @unchecked Sendable
{
    typealias ApplyBypass = @Sendable (Bool) -> VoiceProcessingBypassApplyResult
    typealias Clock = @Sendable () -> UInt64
    typealias VerificationSleep = @Sendable (useconds_t) -> Void
    typealias TransitionHandler = @Sendable (VoiceProcessingPowerTransition) -> Void
    typealias InvariantFailureHandler = @Sendable (AdaptiveVoiceProcessingError) -> Void

    struct Snapshot: Equatable, Sendable {
        let automaticLowPowerEnabled: Bool
        let adaptationDisabled: Bool
        let actualBypass: Bool
        let playbackGeneration: UInt64?
        let playbackHoldCount: Int
    }

    private let configuration: VoiceProcessingLowPowerConfiguration
    private let applyBypass: ApplyBypass
    private let clock: Clock
    private let verificationSleep: VerificationSleep
    private let onTransition: TransitionHandler
    private let onInvariantFailure: InvariantFailureHandler
    private let controlQueue: DispatchQueue

    // ProcessedMicrophoneSink serializes this allocation-free value.
    private var acousticGate: AcousticLowPowerGate

    // Only controlQueue accesses these fields.
    private var automaticLowPowerEnabled: Bool
    private var adaptationDisabled = false
    private var actualBypass = false
    private var earliestAutomaticBypassNanoseconds: UInt64
    private var playbackGeneration: UInt64?
    private var didReportInvariantFailure = false
    private var isStopped = false

    private var actualBypassAtomic: Int32 = 0
    private var diagnosticForcedBypassAtomic: Int32 = 0
    private var transitionPendingAtomic: Int32 = 0
    private var acousticEpochAtomic: Int32 = 0
    private var stoppedAtomic: Int32 = 0

    init(
        automaticLowPowerEnabled: Bool,
        configuration: VoiceProcessingLowPowerConfiguration = .production,
        applyBypass: @escaping ApplyBypass,
        clock: @escaping Clock = {
            DispatchTime.now().uptimeNanoseconds
        },
        verificationSleep: @escaping VerificationSleep = { delay in
            usleep(delay)
        },
        onTransition: @escaping TransitionHandler = { _ in },
        onInvariantFailure: @escaping InvariantFailureHandler = { _ in }
    ) {
        precondition(configuration.minimumProcessedNanoseconds > 0)
        precondition(configuration.verificationAttempts > 0)
        self.automaticLowPowerEnabled = automaticLowPowerEnabled
        self.configuration = configuration
        acousticGate = AcousticLowPowerGate(
            configuration: configuration.acoustic
        )
        self.applyBypass = applyBypass
        self.clock = clock
        self.verificationSleep = verificationSleep
        self.onTransition = onTransition
        self.onInvariantFailure = onInvariantFailure
        controlQueue = DispatchQueue(
            label: "fish-memory.adaptive-vpio-control",
            qos: .userInitiated
        )
        earliestAutomaticBypassNanoseconds =
            clock() + configuration.minimumProcessedNanoseconds
    }

    var snapshot: Snapshot {
        controlQueue.sync {
            Snapshot(
                automaticLowPowerEnabled: automaticLowPowerEnabled,
                adaptationDisabled: adaptationDisabled,
                actualBypass: actualBypass,
                playbackGeneration: playbackGeneration,
                playbackHoldCount: playbackGeneration == nil ? 0 : 1
            )
        }
    }

    func observeMicrophoneFrame(
        _ samples: UnsafeBufferPointer<Int16>,
        captureSequence: UInt64
    ) {
        guard !samples.isEmpty, atomicLoad(&stoppedAtomic) == 0 else { return }
        let observation = acousticGate.observe(
            Self.measure(samples),
            bypassed: atomicLoad(&actualBypassAtomic) != 0
        )
        if observation.invalidatesPendingBypass {
            _ = OSAtomicIncrement32Barrier(&acousticEpochAtomic)
        }

        switch observation.request {
        case .none:
            return
        case .bypassProcessing:
            guard claimTransition() else { return }
            let acousticEpoch = atomicLoad(&acousticEpochAtomic)
            let triggerUptime = clock()
            controlQueue.async { [weak self] in
                self?.considerEnteringBypass(
                    triggerCaptureSequence: captureSequence,
                    triggerUptimeNanoseconds: triggerUptime,
                    acousticEpoch: acousticEpoch
                )
            }
        case .restoreProcessing:
            // Diagnostic A/B bypass stays fixed against the raw acoustic gate.
            guard atomicLoad(&diagnosticForcedBypassAtomic) == 0,
                  claimTransition()
            else { return }
            let triggerUptime = clock()
            controlQueue.async { [weak self] in
                self?.restoreAfterActivity(
                    triggerCaptureSequence: captureSequence,
                    triggerUptimeNanoseconds: triggerUptime
                )
            }
        }
    }

    /// Idempotent KWS/VAD fallback. It extends the processed lease without
    /// acquiring or changing a playback generation.
    func ensureProcessed() -> Bool {
        controlQueue.sync {
            guard !isStopped else { return false }
            cancelPendingTransition()
            return ensureProcessedOnQueue(reason: .voiceActivity)
        }
    }

    /// Restore early, but do not hold a timer or playback generation. The first
    /// PCM frame checks again before it can be scheduled.
    func prewarmPlayback(generation: UInt64) -> Bool {
        controlQueue.sync {
            guard !isStopped else { return false }
            cancelPendingTransition()
            return ensureProcessedOnQueue(
                reason: .playbackPrewarm,
                generation: generation
            )
        }
    }

    func beginPlaybackProtection(generation: UInt64) -> Bool {
        controlQueue.sync {
            guard !isStopped else { return false }
            cancelPendingTransition()
            // An active generation is the lease. Engine suspension releases
            // it, so streaming PCM need not rewrite VPIO for every chunk.
            if playbackGeneration == generation { return true }
            guard ensureProcessedOnQueue(
                reason: .playback,
                generation: generation
            ) else { return false }
            playbackGeneration = generation
            return true
        }
    }

    func endPlaybackProtection(generation: UInt64) {
        controlQueue.sync {
            guard !isStopped, playbackGeneration == generation else { return }
            playbackGeneration = nil
            holdProcessingFromNow()
        }
    }

    func setDiagnosticBypass(
        _ bypassed: Bool
    ) -> VoiceProcessingBypassApplyResult {
        controlQueue.sync {
            guard !isStopped else {
                return VoiceProcessingBypassApplyResult(
                    requested: false,
                    actual: false
                )
            }
            automaticLowPowerEnabled = false
            cancelPendingTransition()
            if bypassed {
                OSAtomicOr32Barrier(1, &diagnosticForcedBypassAtomic)
            } else {
                OSAtomicAnd32Barrier(0, &diagnosticForcedBypassAtomic)
            }
            let target = playbackGeneration == nil ? bypassed : false
            let result = applyAndVerifyBlocking(target)
            updateActualBypass(result.actual)
            if bypassed, !stateMatches(result, target: true) {
                OSAtomicAnd32Barrier(0, &diagnosticForcedBypassAtomic)
            }
            emitTransition(
                target: target,
                result: result,
                reason: .diagnostic,
                captureSequence: nil,
                triggerUptimeNanoseconds: clock()
            )
            return result
        }
    }

    func drainForTesting() {
        controlQueue.sync {}
    }

    func stop() {
        OSAtomicOr32Barrier(1, &stoppedAtomic)
        controlQueue.sync {
            isStopped = true
            automaticLowPowerEnabled = false
            playbackGeneration = nil
            cancelPendingTransition()
        }
    }

    private func considerEnteringBypass(
        triggerCaptureSequence: UInt64,
        triggerUptimeNanoseconds: UInt64,
        acousticEpoch: Int32
    ) {
        let now = clock()
        guard !isStopped, automaticLowPowerEnabled, !adaptationDisabled,
              playbackGeneration == nil, !actualBypass,
              atomicLoad(&acousticEpochAtomic) == acousticEpoch,
              now >= earliestAutomaticBypassNanoseconds
        else {
            clearTransitionClaim()
            return
        }

        let result = applyAndVerifyBlocking(true)
        guard stateMatches(result, target: true) else {
            adaptationDisabled = true
            let rollback = applyAndVerifyBlocking(false)
            updateActualBypass(rollback.actual)
            clearTransitionClaim()
            emitTransition(
                target: false,
                result: rollback,
                reason: .failActiveRollback,
                captureSequence: triggerCaptureSequence,
                triggerUptimeNanoseconds: triggerUptimeNanoseconds
            )
            if !stateMatches(rollback, target: false) {
                reportInvariantFailure(reason: .failActiveRollback)
            }
            return
        }

        // Publish raw mode before releasing the claim. A subsequent raw frame
        // can enqueue restoration behind this operation.
        updateActualBypass(true)
        clearTransitionClaim()
        guard atomicLoad(&acousticEpochAtomic) == acousticEpoch else {
            rollbackEntryAfterActivity(
                triggerCaptureSequence: triggerCaptureSequence,
                triggerUptimeNanoseconds: triggerUptimeNanoseconds
            )
            return
        }
        emitTransition(
            target: true,
            result: result,
            reason: .quietStandby,
            captureSequence: triggerCaptureSequence,
            triggerUptimeNanoseconds: triggerUptimeNanoseconds
        )
    }

    private func rollbackEntryAfterActivity(
        triggerCaptureSequence: UInt64,
        triggerUptimeNanoseconds: UInt64
    ) {
        let rollback = applyAndVerifyBlocking(false)
        updateActualBypass(rollback.actual)
        holdProcessingFromNow()
        emitTransition(
            target: false,
            result: rollback,
            reason: .failActiveRollback,
            captureSequence: triggerCaptureSequence,
            triggerUptimeNanoseconds: triggerUptimeNanoseconds
        )
        if !stateMatches(rollback, target: false) {
            reportInvariantFailure(reason: .failActiveRollback)
        }
    }

    private func restoreAfterActivity(
        triggerCaptureSequence: UInt64,
        triggerUptimeNanoseconds: UInt64
    ) {
        defer { clearTransitionClaim() }
        guard actualBypass else { return }
        let result = applyAndVerifyBlocking(false)
        updateActualBypass(result.actual)
        if stateMatches(result, target: false) { holdProcessingFromNow() }
        emitTransition(
            target: false,
            result: result,
            reason: .acousticActivity,
            captureSequence: triggerCaptureSequence,
            triggerUptimeNanoseconds: triggerUptimeNanoseconds
        )
        if !stateMatches(result, target: false) {
            reportInvariantFailure(reason: .acousticActivity)
        }
    }

    private func ensureProcessedOnQueue(
        reason: VoiceProcessingTransitionReason,
        generation: UInt64? = nil
    ) -> Bool {
        let wasBypassed = actualBypass || atomicLoad(&actualBypassAtomic) != 0
        // Always write and read back the platform state. Audio options or an
        // engine rebuild can change the SDK state without touching our cache.
        let result = applyAndVerifyBlocking(false)
        updateActualBypass(result.actual)
        holdProcessingFromNow()
        if wasBypassed || !stateMatches(result, target: false) {
            emitTransition(
                target: false,
                result: result,
                reason: reason,
                captureSequence: nil,
                triggerUptimeNanoseconds: clock(),
                playbackGeneration: generation
            )
        }
        return stateMatches(result, target: false)
    }

    private func applyAndVerifyBlocking(
        _ bypassed: Bool
    ) -> VoiceProcessingBypassApplyResult {
        var result = applyBypass(bypassed)
        guard !stateMatches(result, target: bypassed) else { return result }
        for _ in 1 ..< configuration.verificationAttempts {
            verificationSleep(configuration.verificationIntervalMicroseconds)
            result = applyBypass(bypassed)
            if stateMatches(result, target: bypassed) { break }
        }
        return result
    }

    private func stateMatches(
        _ result: VoiceProcessingBypassApplyResult,
        target: Bool
    ) -> Bool {
        result.requested == target && result.actual == target
    }

    private func holdProcessingFromNow() {
        let now = clock()
        earliestAutomaticBypassNanoseconds = max(
            earliestAutomaticBypassNanoseconds,
            now + configuration.minimumProcessedNanoseconds
        )
    }

    private func cancelPendingTransition() {
        _ = OSAtomicIncrement32Barrier(&acousticEpochAtomic)
        clearTransitionClaim()
    }

    private func updateActualBypass(_ bypassed: Bool) {
        actualBypass = bypassed
        OSAtomicAnd32Barrier(0, &actualBypassAtomic)
        if bypassed { OSAtomicOr32Barrier(1, &actualBypassAtomic) }
    }

    private func emitTransition(
        target: Bool,
        result: VoiceProcessingBypassApplyResult,
        reason: VoiceProcessingTransitionReason,
        captureSequence: UInt64?,
        triggerUptimeNanoseconds: UInt64,
        playbackGeneration: UInt64? = nil
    ) {
        let now = clock()
        let framesBeforeApplied = now >= triggerUptimeNanoseconds
            ? (now - triggerUptimeNanoseconds) / 10_000_000
            : 0
        onTransition(VoiceProcessingPowerTransition(
            uptimeNanoseconds: now,
            captureSequence: captureSequence,
            reason: reason,
            requestedBypass: target,
            sdkRequestedBypass: result.requested,
            actualBypass: result.actual,
            framesBeforeApplied: framesBeforeApplied,
            playbackGeneration: playbackGeneration ?? self.playbackGeneration,
            playbackHoldCount: self.playbackGeneration == nil ? 0 : 1
        ))
    }

    private func reportInvariantFailure(
        reason: VoiceProcessingTransitionReason
    ) {
        guard !didReportInvariantFailure else { return }
        didReportInvariantFailure = true
        onInvariantFailure(.failedToRestore(reason: reason))
    }

    private func claimTransition() -> Bool {
        OSAtomicCompareAndSwap32Barrier(
            0,
            1,
            &transitionPendingAtomic
        )
    }

    private func clearTransitionClaim() {
        OSAtomicAnd32Barrier(0, &transitionPendingAtomic)
    }

    private func atomicLoad(_ value: inout Int32) -> Int32 {
        OSAtomicAdd32Barrier(0, &value)
    }

    private static func measure(
        _ samples: UnsafeBufferPointer<Int16>
    ) -> AcousticFrameLevels {
        var absoluteTotal: Int64 = 0
        var peak: Int32 = 0
        for sample in samples {
            let value = Int32(sample)
            let absolute = value >= 0 ? value : -value
            absoluteTotal += Int64(absolute)
            peak = max(peak, absolute)
        }
        return AcousticFrameLevels(
            meanAbsolute: Int32(absoluteTotal / Int64(samples.count)),
            peak: peak
        )
    }
}
