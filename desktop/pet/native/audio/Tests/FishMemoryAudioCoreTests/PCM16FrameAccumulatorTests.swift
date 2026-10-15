import FishMemoryAudioCore
import Foundation
import XCTest

final class PCM16FrameAccumulatorTests: XCTestCase {
    func testAccumulatesArbitraryChunksIntoTenMillisecondFrames() {
        let accumulator = PCM16FrameAccumulator()
        XCTAssertTrue(accumulator.append(Array(0 ..< 79)).isEmpty)
        XCTAssertEqual(accumulator.pendingSampleCount, 79)

        let first = accumulator.append(Array(79 ..< 200))
        XCTAssertEqual(first.count, 1)
        XCTAssertEqual(first[0].count, 320)
        XCTAssertEqual(decodePCM(first[0]), Array(0 ..< 160))
        XCTAssertEqual(accumulator.pendingSampleCount, 40)

        let remaining = accumulator.append(Array(200 ..< 480))
        XCTAssertEqual(remaining.count, 2)
        XCTAssertEqual(decodePCM(remaining[0]), Array(160 ..< 320))
        XCTAssertEqual(decodePCM(remaining[1]), Array(320 ..< 480))
        XCTAssertEqual(accumulator.pendingSampleCount, 0)
    }

    func testResetDropsOnlyPendingPartialFrame() {
        let accumulator = PCM16FrameAccumulator()
        _ = accumulator.append(Array(repeating: 7, count: 100))
        accumulator.reset()

        let frames = accumulator.append(Array(repeating: 9, count: 160))
        XCTAssertEqual(frames.count, 1)
        XCTAssertEqual(decodePCM(frames[0]), Array(repeating: 9, count: 160))
    }

    func testBorrowedFramePathMatchesOwnedDataAcrossChunkBoundaries() {
        let borrowedAccumulator = PCM16FrameAccumulator()
        let ownedAccumulator = PCM16FrameAccumulator()
        let chunks: [[Int16]] = [
            Array(-200 ..< -121),
            Array(-121 ..< 17),
            Array(17 ..< 491),
        ]
        var borrowedFrames: [[Int16]] = []
        var ownedFrames: [Data] = []

        for chunk in chunks {
            chunk.withUnsafeBufferPointer { samples in
                borrowedAccumulator.process(samples) { frame in
                    borrowedFrames.append(Array(frame))
                }
            }
            ownedFrames.append(contentsOf: ownedAccumulator.append(chunk))
        }

        XCTAssertEqual(
            borrowedFrames,
            ownedFrames.map(decodePCM)
        )
        XCTAssertEqual(
            borrowedAccumulator.pendingSampleCount,
            ownedAccumulator.pendingSampleCount
        )
        XCTAssertEqual(borrowedAccumulator.pendingSampleCount, 51)
    }

    func testLittleEndianEncodingPreservesSignedSampleBytes() {
        let accumulator = PCM16FrameAccumulator(samplesPerFrame: 4)
        let frames = accumulator.append([
            Int16.min,
            -2,
            1,
            Int16.max,
        ])

        XCTAssertEqual(frames, [Data([
            0x00, 0x80,
            0xFE, 0xFF,
            0x01, 0x00,
            0xFF, 0x7F,
        ])])
    }

    private func decodePCM(_ data: Data) -> [Int16] {
        stride(from: 0, to: data.count, by: 2).map { offset in
            let low = UInt16(data[offset])
            let high = UInt16(data[offset + 1]) << 8
            return Int16(bitPattern: low | high)
        }
    }
}
