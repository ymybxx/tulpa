import CoreVideo
import CoreImage
import Darwin
import XCTest
@testable import PerceptionCore

final class EnvironmentIdleTests: XCTestCase {
    func testSilenceCompressionPreservesContinuousResamplingAndFirstSound() throws {
        let seconds = SystemAudioWindow.durationSeconds
        let windowBytes = SystemAudioWindow.pcmBytesPerWindow
        for rate in [16_000, 32_000, 44_100, 48_000, 96_000] {
            let reference = try CoreAudioTapSampleRateConverter(inputRate: Double(rate))
            var assembler = SystemAudioWindowAssembler(startedAt: 100, startMonotonicMs: 0, initialUnavailableReason: nil)
            var expectedPCM = Data()
            var silentWindows = 0
            var comparedWindows = 0
            for second in 0..<(3 * seconds + 4) {
                var samples = [Float](repeating: 0, count: rate)
                // An impulse right at the resume boundary plus very quiet speech.
                if second == 0 || second == 3 * seconds - 2 || second == 3 * seconds - 1 {
                    samples[0] = 0.6
                    for index in 1..<rate {
                        samples[index] = Float(sin(Double(index) * 0.03)) * 0.0002
                    }
                }
                let expected = try XCTUnwrap(reference.convert(samples))
                XCTAssertTrue(assembler.append(.init(pcm: expected.pcm)))
                expectedPCM.append(expected.pcm)
                if expectedPCM.count >= windowBytes {
                    let window = assembler.finishNextWindow()
                    let bytes = window.audioSilence ? Data(repeating: 0, count: windowBytes)
                        : try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(window.audioPcmBase64)))
                    XCTAssertEqual(window.audioStatus, .complete)
                    XCTAssertEqual(bytes, expectedPCM.prefix(windowBytes), "rate=\(rate) second=\(second)")
                    expectedPCM.removeFirst(windowBytes)
                    silentWindows += window.audioSilence ? 1 : 0
                    comparedWindows += 1
                }
            }
            XCTAssertEqual(silentWindows, 1)
            XCTAssertEqual(comparedWindows, 3)
        }
    }

    func testQuietNonzeroSamplesArePreserved() {
        let quiet = Data([1, 0] + Array(repeating: UInt8(0), count: 31_998))
        let block = SystemAudioPCMBlock(pcm: quiet)
        XCTAssertFalse(block.isSilent)
        XCTAssertEqual(block.pcm, quiet)
        XCTAssertFalse(SystemAudioPCMBlock(pcm: Data([0])).isSilent)
    }

    func testSparseSilenceRetainsWindowTimeAndExactResumedPCM() throws {
        let seconds = SystemAudioWindow.durationSeconds
        let pcmBytesPerSecond = SystemAudioWindow.sampleRate * 2
        var assembler = SystemAudioWindowAssembler(startedAt: 100, startMonotonicMs: 0, initialUnavailableReason: nil)
        for _ in 0..<seconds { XCTAssertTrue(assembler.append(.init(silentSamples: 16_000))) }
        let quiet = assembler.finishNextWindow()
        XCTAssertTrue(quiet.audioSilence)
        XCTAssertEqual(quiet.audioStatus, .complete)
        XCTAssertEqual(quiet.audioCapturedSamples, SystemAudioWindow.samplesPerWindow)
        XCTAssertNil(quiet.audioPcmBase64)
        XCTAssertLessThan(try JSONEncoder().encode(quiet).count, 500)

        let beforeSpeech = seconds / 2
        let afterSpeech = seconds - beforeSpeech - 1
        for _ in 0..<beforeSpeech { XCTAssertTrue(assembler.append(.init(silentSamples: 16_000))) }
        let speech = Data([1, 0] + Array(repeating: UInt8(7), count: 31_998))
        XCTAssertTrue(assembler.append(.init(pcm: speech)))
        for _ in 0..<afterSpeech { XCTAssertTrue(assembler.append(.init(silentSamples: 16_000))) }
        let resumed = assembler.finishNextWindow()
        XCTAssertFalse(resumed.audioSilence)
        XCTAssertEqual(resumed.windowIndex, 1)
        XCTAssertEqual(resumed.startedAt, quiet.endedAt)
        let pcm = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(resumed.audioPcmBase64)))
        XCTAssertEqual(pcm, Data(repeating: 0, count: beforeSpeech * pcmBytesPerSecond)
                       + speech + Data(repeating: 0, count: afterSpeech * pcmBytesPerSecond))
    }

    func testMissingSamplesAreNeverLabeledSilence() throws {
        var assembler = SystemAudioWindowAssembler(startedAt: 100, startMonotonicMs: 0, initialUnavailableReason: nil)
        XCTAssertTrue(assembler.append(.init(silentSamples: 16_000)))
        let partial = assembler.finishNextWindow()
        XCTAssertFalse(partial.audioSilence)
        XCTAssertEqual(partial.audioStatus, .partial)
        XCTAssertNotNil(partial.audioPcmBase64)
        let missing = assembler.finishNextWindow()
        XCTAssertFalse(missing.audioSilence)
        XCTAssertEqual(missing.audioStatus, .unavailable)
    }

    func testTinyAlternatingBlocksRemainBoundedAndPreserveBytesAcrossWindows() throws {
        var assembler = SystemAudioWindowAssembler(startedAt: 100, startMonotonicMs: 0, periodMilliseconds: 50, initialUnavailableReason: nil)
        var expected = Data()
        for index in 0..<1000 {
            let pcm = Data([index.isMultiple(of: 2) ? UInt8(0) : UInt8(7), 0])
            expected.append(pcm)
            XCTAssertTrue(assembler.append(.init(pcm: pcm)))
        }
        let first = assembler.finishNextWindow()
        XCTAssertEqual(first.audioStatus, .complete)
        XCTAssertEqual(Data(base64Encoded: try XCTUnwrap(first.audioPcmBase64)), expected.prefix(1600))
        let second = assembler.finishNextWindow()
        XCTAssertEqual(second.audioStatus, .partial)
        XCTAssertEqual(Data(base64Encoded: try XCTUnwrap(second.audioPcmBase64)), expected.suffix(400) + Data(repeating: 0, count: 1200))
    }

    func testIdenticalPixelsReuseEncodingWithFreshCaptureIdentity() async throws {
        let pixels = try makeIdlePixels()
        let calls = IdleEncodeCounter()
        let broker = EnvironmentFrameBroker(clock: { 100 })
        let encoder = EnvironmentFrameEncoder(broker: broker, encode: { _ in
            calls.increment()
            return EncodedEnvironmentFrame(jpeg: Data([0xff, 0xd8, 0xff, 0xff, 0xd9]), pixelWidth: 32, pixelHeight: 32)
        }, onAvailable: { _ in })
        encoder.activate(generation: 1)
        var identities: [String] = []
        for index in 0..<5 {
            await encoder.submitAndWait(.init(pixelBuffer: pixels, observedAt: 100 + Double(index), displayID: 1), generation: 1, isStillCurrent: { true })
            let metadata = try XCTUnwrap(broker.currentMetadata())
            XCTAssertEqual(metadata.observedAt, 100 + Double(index))
            identities.append(metadata.frameId)
            XCTAssertNotNil(metadata.contentHash)
        }
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(Set(identities).count, 5)
        CVPixelBufferLockBaseAddress(pixels, [])
        CVPixelBufferGetBaseAddress(pixels)!.storeBytes(of: UInt8(1), as: UInt8.self)
        CVPixelBufferUnlockBaseAddress(pixels, [])
        await encoder.submitAndWait(.init(pixelBuffer: pixels, observedAt: 106, displayID: 1), generation: 1, isStillCurrent: { true })
        XCTAssertEqual(calls.count, 2, "one changed pixel must be re-encoded")
        await encoder.submitAndWait(.init(pixelBuffer: pixels, observedAt: 107, displayID: 2), generation: 1, isStillCurrent: { true })
        XCTAssertEqual(calls.count, 3, "display changes invalidate reuse")
        encoder.invalidate()
        encoder.activate(generation: 2)
        await encoder.submitAndWait(.init(pixelBuffer: pixels, observedAt: 108, displayID: 2), generation: 2, isStillCurrent: { true })
        XCTAssertEqual(calls.count, 4)
        encoder.invalidate()
        XCTAssertNil(broker.currentMetadata())
    }

    func testIsolatedIdleProcessingBenchmark() async throws {
        guard ProcessInfo.processInfo.environment["FISH_MEMORY_IDLE_BENCHMARK"] == "1" else { return }
        let pixels = try makeIdlePixels(width: 960, height: 624)
        let context = CIContext(options: [.cacheIntermediates: false])
        _ = EnvironmentFrameEncoder.encodeJPEG(pixelBuffer: pixels, configuration: .init(), context: context)
        let baselineBegin = clock()
        for _ in 0..<60 {
            _ = autoreleasepool { EnvironmentFrameEncoder.encodeJPEG(pixelBuffer: pixels, configuration: .init(), context: context) }
        }
        let baselineImageCPU = Double(clock() - baselineBegin) / Double(CLOCKS_PER_SEC)
        let broker = EnvironmentFrameBroker(clock: { 100 })
        let encoder = EnvironmentFrameEncoder(broker: broker, onAvailable: { _ in })
        encoder.activate(generation: 1)
        await encoder.submitAndWait(.init(pixelBuffer: pixels, observedAt: 100, displayID: 1), generation: 1, isStillCurrent: { true })
        let begin = clock()
        for index in 0..<60 {
            await encoder.submitAndWait(.init(pixelBuffer: pixels, observedAt: 101 + Double(index), displayID: 1), generation: 1, isStillCurrent: { true })
        }
        let imageCPU = Double(clock() - begin) / Double(CLOCKS_PER_SEC)
        print("IDLE_BENCHMARK image_baseline_60_cpu=\(baselineImageCPU) image_checks_60_cpu=\(imageCPU)")
        encoder.invalidate()
    }
}

private func makeIdlePixels(width: Int = 32, height: Int = 32) throws -> CVPixelBuffer {
    var pixels: CVPixelBuffer?
    XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, nil, &pixels), kCVReturnSuccess)
    let value = try XCTUnwrap(pixels)
    CVPixelBufferLockBaseAddress(value, [])
    memset(CVPixelBufferGetBaseAddress(value)!, 0, CVPixelBufferGetBytesPerRow(value) * height)
    CVPixelBufferUnlockBaseAddress(value, [])
    return value
}

private final class IdleEncodeCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func increment() { lock.lock(); value += 1; lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return value }
}
