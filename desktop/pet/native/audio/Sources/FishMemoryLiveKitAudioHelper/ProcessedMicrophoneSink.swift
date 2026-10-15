@preconcurrency import AVFoundation
import Dispatch
import FishMemoryAudioCore
import Foundation
import LiveKit

final class ProcessedMicrophoneSink: AudioRenderer, @unchecked Sendable {
    private let writer: BoundedFrameWriter
    private let onFatalError: @Sendable (Error) -> Void
    private let voiceProcessingController: AdaptiveVoiceProcessingController
    private let resampler = PCMResampler16k()
    private let accumulator = PCM16FrameAccumulator()
    private let stateLock = NSLock()
    private var sequence: UInt64 = 0
    private var hasFailed = false
    private let silence = [Int16](repeating: 0, count: FishMemoryAudioFrameProtocol.samplesPerFrame)
    private var microphoneEnabled = true

    func setMicrophoneEnabled(_ enabled: Bool) {
        stateLock.lock()
        defer { stateLock.unlock() }
        microphoneEnabled = enabled
        accumulator.reset()
    }

    init(
        writer: BoundedFrameWriter,
        voiceProcessingController: AdaptiveVoiceProcessingController,
        onFatalError: @escaping @Sendable (Error) -> Void
    ) {
        self.writer = writer
        self.voiceProcessingController = voiceProcessingController
        self.onFatalError = onFatalError
    }

    func render(pcmBuffer: AVAudioPCMBuffer) {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard !hasFailed else { return }

        do {
            try resampler.withConvertedSamples(pcmBuffer) {
                samples, inputFormatChanged in
                if inputFormatChanged {
                    accumulator.reset()
                }
                try accumulator.process(samples) { pcmSamples in
                    if microphoneEnabled {
                        try emit(pcmSamples)
                    } else {
                        try silence.withUnsafeBufferPointer { try emit($0) }
                    }
                }
            }
        } catch {
            hasFailed = true
            onFatalError(error)
        }
    }
    private func emit(_ pcmSamples: UnsafeBufferPointer<Int16>) throws {
        voiceProcessingController.observeMicrophoneFrame(pcmSamples, captureSequence: sequence)
        let packet = try FishMemoryAudioFrameProtocol.encode(
            sequence: sequence,
            uptimeNanoseconds: DispatchTime.now().uptimeNanoseconds,
            pcmSamples: pcmSamples
        )
        sequence &+= 1
        _ = writer.enqueue(packet)
    }

}
