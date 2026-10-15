import Foundation

public final class PCM16FrameAccumulator: @unchecked Sendable {
    private let samplesPerFrame: Int
    private var pendingSamples: [Int16] = []

    public init(
        samplesPerFrame: Int = FishMemoryAudioFrameProtocol.samplesPerFrame
    ) {
        precondition(samplesPerFrame > 0)
        self.samplesPerFrame = samplesPerFrame
        pendingSamples.reserveCapacity(samplesPerFrame)
    }

    public var pendingSampleCount: Int {
        pendingSamples.count
    }

    public func append(_ input: [Int16]) -> [Data] {
        var frames: [Data] = []
        frames.reserveCapacity(
            (pendingSamples.count + input.count) / samplesPerFrame
        )
        input.withUnsafeBufferPointer { samples in
            process(samples) { frame in
                frames.append(Self.encodeLittleEndianPCM(frame))
            }
        }
        return frames
    }

    /// Delivers each complete frame synchronously from accumulator-owned or
    /// caller-owned storage. The borrowed frame pointer must not escape
    /// `frameHandler`.
    public func process(
        _ input: UnsafeBufferPointer<Int16>,
        frameHandler: (UnsafeBufferPointer<Int16>) throws -> Void
    ) rethrows {
        guard !input.isEmpty else { return }

        var inputOffset = 0
        if !pendingSamples.isEmpty {
            let copyCount = min(
                samplesPerFrame - pendingSamples.count,
                input.count
            )
            appendInput(input, offset: inputOffset, count: copyCount)
            inputOffset += copyCount
            if pendingSamples.count == samplesPerFrame {
                try pendingSamples.withUnsafeBufferPointer(frameHandler)
                pendingSamples.removeAll(keepingCapacity: true)
            }
        }

        while input.count - inputOffset >= samplesPerFrame {
            let frame = UnsafeBufferPointer(
                start: input.baseAddress?.advanced(by: inputOffset),
                count: samplesPerFrame
            )
            try frameHandler(frame)
            inputOffset += samplesPerFrame
        }

        let remainingCount = input.count - inputOffset
        if remainingCount > 0 {
            appendInput(input, offset: inputOffset, count: remainingCount)
        }
    }

    public func reset() {
        pendingSamples.removeAll(keepingCapacity: true)
    }

    private func appendInput(
        _ input: UnsafeBufferPointer<Int16>,
        offset: Int,
        count: Int
    ) {
        guard count > 0 else { return }
        pendingSamples.append(contentsOf: UnsafeBufferPointer(
            start: input.baseAddress?.advanced(by: offset),
            count: count
        ))
    }

    private static func encodeLittleEndianPCM(
        _ samples: UnsafeBufferPointer<Int16>
    ) -> Data {
        let byteCount = samples.count * MemoryLayout<Int16>.size
#if _endian(little)
        return Data(bytes: samples.baseAddress!, count: byteCount)
#else
        var data = Data(count: byteCount)
        data.withUnsafeMutableBytes { destination in
            let bytes = destination.bindMemory(to: UInt8.self)
            for (index, sample) in samples.enumerated() {
                let value = UInt16(bitPattern: sample)
                bytes[index * 2] = UInt8(truncatingIfNeeded: value)
                bytes[index * 2 + 1] = UInt8(truncatingIfNeeded: value >> 8)
            }
        }
        return data
#endif
    }
}
