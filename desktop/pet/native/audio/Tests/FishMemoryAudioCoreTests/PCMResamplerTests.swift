@preconcurrency import AVFoundation
@testable import FishMemoryAudioCore
import XCTest

final class PCMResamplerTests: XCTestCase {
    func testResamplesFortyEightKilohertzMonoToSixteenKilohertz() throws {
        let resampler = PCMResampler16k()
        let input = try makeSineBuffer(
            sampleRate: 48_000,
            channels: 1,
            frameCount: 480
        )

        let first = try resampler.convert(input)
        let second = try resampler.convert(input)

        XCTAssertTrue(first.inputFormatChanged)
        XCTAssertFalse(second.inputFormatChanged)
        XCTAssertEqual(first.samples.count, 160)
        XCTAssertEqual(second.samples.count, 160)
        XCTAssertGreaterThan(first.samples.map { abs(Int($0)) }.max() ?? 0, 1_000)
    }

    func testDownmixesStereo() throws {
        let resampler = PCMResampler16k()
        let input = try makeSineBuffer(
            sampleRate: 48_000,
            channels: 2,
            frameCount: 480
        )

        let result = try resampler.convert(input)

        XCTAssertEqual(result.samples.count, 160)
        XCTAssertGreaterThan(result.samples.map { abs(Int($0)) }.max() ?? 0, 1_000)
    }

    func testBorrowedSamplesAreExactlyEquivalentToOwnedConversion() throws {
        let ownedResampler = PCMResampler16k()
        let borrowedResampler = PCMResampler16k()
        let input = try makeSineBuffer(
            sampleRate: 48_000,
            channels: 2,
            frameCount: 480
        )

        let owned = try ownedResampler.convert(input)
        var borrowed: [Int16] = []
        var formatChanged = false
        try borrowedResampler.withConvertedSamples(input) { samples, changed in
            borrowed = Array(samples)
            formatChanged = changed
        }

        XCTAssertEqual(borrowed, owned.samples)
        XCTAssertEqual(formatChanged, owned.inputFormatChanged)
    }

    func testRealtimeBuffersAreReusedUntilCapacityMustGrow() throws {
        let resampler = PCMResampler16k()
        let ordinary = try makeSineBuffer(
            sampleRate: 48_000,
            channels: 1,
            frameCount: 480
        )

        for _ in 0 ..< 20 {
            try resampler.withConvertedSamples(ordinary) { samples, _ in
                XCTAssertEqual(samples.count, 160)
            }
        }
        XCTAssertEqual(resampler.outputBufferAllocationCount, 1)
        XCTAssertEqual(resampler.sampleScratchGrowthCount, 1)

        let larger = try makeSineBuffer(
            sampleRate: 48_000,
            channels: 1,
            frameCount: 960
        )
        try resampler.withConvertedSamples(larger) { samples, _ in
            XCTAssertEqual(samples.count, 320)
        }
        XCTAssertEqual(resampler.outputBufferAllocationCount, 2)
        XCTAssertEqual(resampler.sampleScratchGrowthCount, 2)

        try resampler.withConvertedSamples(ordinary) { samples, _ in
            XCTAssertEqual(samples.count, 160)
        }
        XCTAssertEqual(resampler.outputBufferAllocationCount, 2)
        XCTAssertEqual(resampler.sampleScratchGrowthCount, 2)
    }

    private func makeSineBuffer(
        sampleRate: Double,
        channels: AVAudioChannelCount,
        frameCount: AVAudioFrameCount
    ) throws -> AVAudioPCMBuffer {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: sampleRate,
            channels: channels,
            interleaved: false
        ), let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ), let channelData = buffer.int16ChannelData else {
            throw TestError.bufferAllocationFailed
        }
        buffer.frameLength = frameCount

        for channel in 0 ..< Int(channels) {
            for frame in 0 ..< Int(frameCount) {
                let phase = 2 * Double.pi * 1_000 * Double(frame) / sampleRate
                channelData[channel][frame] = Int16(sin(phase) * 12_000)
            }
        }
        return buffer
    }

    private enum TestError: Error {
        case bufferAllocationFailed
    }
}
