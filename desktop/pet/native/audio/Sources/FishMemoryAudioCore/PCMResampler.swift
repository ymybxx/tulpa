@preconcurrency import AVFoundation
import Foundation

public struct ResampledPCM: Sendable {
    public let samples: [Int16]
    public let inputFormatChanged: Bool

    public init(samples: [Int16], inputFormatChanged: Bool) {
        self.samples = samples
        self.inputFormatChanged = inputFormatChanged
    }
}

public final class PCMResampler16k: @unchecked Sendable {
    // Conversion state is intentionally serial. The microphone sink holds its
    // state lock for the complete call, including the borrowed-sample body.
    private let outputFormat: AVAudioFormat
    private var inputFormat: AVAudioFormat?
    private var converter: AVAudioConverter?
    private var outputBuffer: AVAudioPCMBuffer?
    private var sampleScratch: [Int16] = []
    private let inputProvider = ReusableSingleUseAudioInput()

    // Internal counters make the buffer lifecycle observable in focused tests.
    // They increment only when storage grows, never on steady-state callbacks.
    internal private(set) var outputBufferAllocationCount = 0
    internal private(set) var sampleScratchGrowthCount = 0

    public init() {
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(
                FishMemoryAudioFrameProtocol.sampleRate
            ),
            channels: AVAudioChannelCount(
                FishMemoryAudioFrameProtocol.channels
            )
        ) else {
            preconditionFailure("Unable to create 16 kHz mono PCM format")
        }
        outputFormat = format
    }

    public func convert(_ input: AVAudioPCMBuffer) throws -> ResampledPCM {
        var result = ResampledPCM(samples: [], inputFormatChanged: false)
        try withConvertedSamples(input) { samples, inputFormatChanged in
            result = ResampledPCM(
                samples: Array(samples),
                inputFormatChanged: inputFormatChanged
            )
        }
        return result
    }

    /// Converts into storage owned by the resampler and lends the resulting
    /// samples only for the duration of `body`.
    ///
    /// Callers that immediately frame or copy the samples can use this path to
    /// avoid allocating a short-lived `[Int16]` on every audio callback. The
    /// borrowed pointer must not escape `body`.
    public func withConvertedSamples<Result>(
        _ input: AVAudioPCMBuffer,
        _ body: (
            UnsafeBufferPointer<Int16>,
            _ inputFormatChanged: Bool
        ) throws -> Result
    ) throws -> Result {
        guard input.frameLength > 0 else {
            return try body(
                UnsafeBufferPointer(start: nil, count: 0),
                false
            )
        }
        guard input.format.sampleRate > 0, input.format.channelCount > 0 else {
            throw PCMResamplerError.invalidInputFormat
        }

        let formatChanged = inputFormat?.isEqual(input.format) != true
        if formatChanged {
            guard let newConverter = AVAudioConverter(
                from: input.format,
                to: outputFormat
            ) else {
                throw PCMResamplerError.unsupportedConversion(
                    input: input.format.description,
                    output: outputFormat.description
                )
            }
            newConverter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
            newConverter.primeMethod = .none
            converter = newConverter
            inputFormat = input.format
        }

        guard let converter else {
            throw PCMResamplerError.converterUnavailable
        }

        let ratio = outputFormat.sampleRate / input.format.sampleRate
        let expectedFrames = Int(ceil(Double(input.frameLength) * ratio))
        let capacity = AVAudioFrameCount(max(expectedFrames + 32, 64))
        let output = try reusableOutputBuffer(minimumCapacity: capacity)
        output.frameLength = 0

        inputProvider.prepare(input)
        defer { inputProvider.clear() }
        var conversionError: NSError?
        let status = converter.convert(
            to: output,
            error: &conversionError
        ) { _, inputStatus in
            self.inputProvider.next(status: inputStatus)
        }

        if status == .error {
            throw PCMResamplerError.conversionFailed(
                conversionError?.localizedDescription ?? "unknown error"
            )
        }
        guard let channel = output.floatChannelData?[0] else {
            throw PCMResamplerError.missingOutputSamples
        }

        let sampleCount = Int(output.frameLength)
        ensureSampleScratch(count: sampleCount)
        sampleScratch.withUnsafeMutableBufferPointer { destination in
            for index in 0 ..< sampleCount {
                destination[index] = Self.floatToInt16(channel[index])
            }
        }
        return try sampleScratch.withUnsafeBufferPointer { storage in
            try body(
                UnsafeBufferPointer(
                    start: storage.baseAddress,
                    count: sampleCount
                ),
                formatChanged
            )
        }
    }

    private func reusableOutputBuffer(
        minimumCapacity: AVAudioFrameCount
    ) throws -> AVAudioPCMBuffer {
        if let outputBuffer,
           outputBuffer.frameCapacity >= minimumCapacity {
            return outputBuffer
        }
        guard let replacement = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: minimumCapacity
        ) else {
            throw PCMResamplerError.outputAllocationFailed
        }
        outputBuffer = replacement
        outputBufferAllocationCount += 1
        return replacement
    }

    private func ensureSampleScratch(count: Int) {
        guard sampleScratch.count < count else { return }
        sampleScratch.append(
            contentsOf: repeatElement(0, count: count - sampleScratch.count)
        )
        sampleScratchGrowthCount += 1
    }

    private static func floatToInt16(_ sample: Float) -> Int16 {
        if sample >= 1 { return Int16.max }
        if sample <= -1 { return Int16.min }
        return Int16((sample * Float(Int16.max)).rounded())
    }
}

private final class ReusableSingleUseAudioInput: @unchecked Sendable {
    // AVAudioConverter consumes this provider synchronously within one
    // serialized conversion. `clear()` prevents the callback buffer from
    // escaping its render callback lifetime.
    private var buffer: AVAudioPCMBuffer?
    private var wasSupplied = false

    func prepare(_ buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
        wasSupplied = false
    }

    func clear() {
        buffer = nil
        wasSupplied = false
    }

    func next(
        status: UnsafeMutablePointer<AVAudioConverterInputStatus>
    ) -> AVAudioBuffer? {
        guard !wasSupplied, let buffer else {
            status.pointee = .noDataNow
            return nil
        }
        wasSupplied = true
        status.pointee = .haveData
        return buffer
    }
}

public enum PCMResamplerError: Error, Equatable {
    case invalidInputFormat
    case unsupportedConversion(input: String, output: String)
    case converterUnavailable
    case outputAllocationFailed
    case conversionFailed(String)
    case missingOutputSamples
}
