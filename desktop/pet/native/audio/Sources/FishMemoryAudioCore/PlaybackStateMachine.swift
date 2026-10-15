import Foundation

public struct ScheduledPlaybackFrame: Equatable, Sendable {
    public let generation: UInt64
    public let resetSerial: UInt64
    public let ordinal: UInt64
    public let validByteCount: UInt32
    public let pcm: Data

    public init(
        generation: UInt64,
        resetSerial: UInt64,
        ordinal: UInt64,
        validByteCount: UInt32,
        pcm: Data
    ) {
        self.generation = generation
        self.resetSerial = resetSerial
        self.ordinal = ordinal
        self.validByteCount = validByteCount
        self.pcm = pcm
    }
}

public struct PlaybackCompletion: Equatable, Sendable {
    public let generation: UInt64
    public let cumulativeValidRenderedByteCount: UInt64
    public let validByteCount: UInt32
    public let pcm: Data
}

public struct PlaybackStateSnapshot: Equatable, Sendable {
    public let generation: UInt64?
    public let pendingByteCount: Int
    public let queuedFrameCount: Int
    public let cumulativeValidRenderedByteCount: UInt64
    public let isPaused: Bool
    public let isDucked: Bool
    public let isStopped: Bool
}

/// Pure state for the streaming player. The AVAudioPlayerNode adapter owns all
/// actual scheduling; this type keeps wire-level generation and byte accounting
/// independently testable.
public final class PlaybackStateMachine: @unchecked Sendable {
    public static let duckGain: Float = 0.2

    private var generation: UInt64?
    private var resetSerial: UInt64 = 0
    private var nextOrdinal: UInt64 = 0
    private var pendingPCM = Data()
    private var pausedFrames: [ScheduledPlaybackFrame] = []
    private var cumulativeValidRenderedByteCount: UInt64 = 0
    private var isPaused = false
    private var isDucked = false
    private var isStopped = true

    public init() {}

    public var snapshot: PlaybackStateSnapshot {
        PlaybackStateSnapshot(
            generation: generation,
            pendingByteCount: pendingPCM.count,
            queuedFrameCount: pausedFrames.count,
            cumulativeValidRenderedByteCount: cumulativeValidRenderedByteCount,
            isPaused: isPaused,
            isDucked: isDucked,
            isStopped: isStopped
        )
    }

    @discardableResult
    public func reset(generation: UInt64) -> [ScheduledPlaybackFrame] {
        resetSerial &+= 1
        self.generation = generation
        nextOrdinal = 0
        pendingPCM.removeAll(keepingCapacity: true)
        pausedFrames.removeAll(keepingCapacity: true)
        cumulativeValidRenderedByteCount = 0
        isPaused = false
        isDucked = false
        isStopped = false
        return []
    }

    public func append(generation: UInt64, pcm: Data) -> [ScheduledPlaybackFrame] {
        guard self.generation == generation, !isStopped, !pcm.isEmpty else {
            return []
        }
        pendingPCM.append(pcm)
        let frameByteCount = FishMemoryAudioFrameProtocol.pcmByteCount
        let completeByteCount = pendingPCM.count / frameByteCount * frameByteCount
        var frames: [ScheduledPlaybackFrame] = []
        frames.reserveCapacity(completeByteCount / frameByteCount)
        for offset in stride(from: 0, to: completeByteCount, by: frameByteCount) {
            let start = pendingPCM.startIndex + offset
            let framePCM = pendingPCM.subdata(in: start ..< start + frameByteCount)
            frames.append(makeFrame(
                pcm: framePCM,
                validByteCount: UInt32(frameByteCount)
            ))
        }
        if completeByteCount > 0 {
            pendingPCM = Data(pendingPCM.suffix(pendingPCM.count - completeByteCount))
        }
        return deliverOrQueue(frames)
    }

    public func pause(generation: UInt64) {
        guard self.generation == generation, !isStopped else { return }
        isPaused = true
    }

    public func resume(generation: UInt64) -> [ScheduledPlaybackFrame] {
        guard self.generation == generation, !isStopped else { return [] }
        isPaused = false
        let frames = pausedFrames
        pausedFrames.removeAll(keepingCapacity: true)
        return frames
    }

    public func duck(generation: UInt64) {
        guard self.generation == generation, !isStopped else { return }
        isDucked = true
    }

    public func unduck(generation: UInt64) {
        guard self.generation == generation, !isStopped else { return }
        isDucked = false
    }

    public func stop(generation: UInt64) {
        guard self.generation == generation else { return }
        resetSerial &+= 1
        pendingPCM.removeAll(keepingCapacity: true)
        pausedFrames.removeAll(keepingCapacity: true)
        isPaused = true
        isStopped = true
    }

    public func flush(generation: UInt64) -> [ScheduledPlaybackFrame] {
        guard self.generation == generation, !isStopped,
              !pendingPCM.isEmpty
        else {
            return []
        }
        let validByteCount = UInt32(pendingPCM.count)
        var padded = pendingPCM
        padded.append(Data(
            repeating: 0,
            count: FishMemoryAudioFrameProtocol.pcmByteCount - padded.count
        ))
        pendingPCM.removeAll(keepingCapacity: true)
        return deliverOrQueue([
            makeFrame(pcm: padded, validByteCount: validByteCount),
        ])
    }

    public func complete(
        _ frame: ScheduledPlaybackFrame,
        gain: Float
    ) -> PlaybackCompletion? {
        guard generation == frame.generation,
              resetSerial == frame.resetSerial,
              !isStopped
        else {
            return nil
        }
        cumulativeValidRenderedByteCount &+= UInt64(frame.validByteCount)
        return PlaybackCompletion(
            generation: frame.generation,
            cumulativeValidRenderedByteCount: cumulativeValidRenderedByteCount,
            validByteCount: frame.validByteCount,
            pcm: Self.applyingGain(gain, to: frame.pcm)
        )
    }

    public var outputGain: Float {
        isDucked ? Self.duckGain : 1
    }

    private func makeFrame(
        pcm: Data,
        validByteCount: UInt32
    ) -> ScheduledPlaybackFrame {
        defer { nextOrdinal &+= 1 }
        return ScheduledPlaybackFrame(
            generation: generation ?? 0,
            resetSerial: resetSerial,
            ordinal: nextOrdinal,
            validByteCount: validByteCount,
            pcm: pcm
        )
    }

    private func deliverOrQueue(
        _ frames: [ScheduledPlaybackFrame]
    ) -> [ScheduledPlaybackFrame] {
        if isPaused {
            pausedFrames.append(contentsOf: frames)
            return []
        }
        return frames
    }

    public static func applyingGain(_ gain: Float, to pcm: Data) -> Data {
        guard gain != 1 else { return pcm }
        var output = Data(capacity: pcm.count)
        for offset in stride(from: 0, to: pcm.count, by: 2) {
            let bits = UInt16(pcm[offset]) | UInt16(pcm[offset + 1]) << 8
            let sample = Float(Int16(bitPattern: bits)) * gain
            let clamped = Int16(max(
                Float(Int16.min),
                min(Float(Int16.max), sample.rounded())
            ))
            var littleEndian = clamped.littleEndian
            Swift.withUnsafeBytes(of: &littleEndian) { bytes in
                output.append(contentsOf: bytes)
            }
        }
        return output
    }
}
