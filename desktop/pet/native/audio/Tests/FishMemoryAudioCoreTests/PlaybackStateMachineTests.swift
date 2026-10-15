import FishMemoryAudioCore
import Foundation
import XCTest

final class PlaybackStateMachineTests: XCTestCase {
    func testResetAccumulateFlushAndCumulativeCompletion() {
        let state = PlaybackStateMachine()
        state.reset(generation: 9)

        XCTAssertTrue(state.append(
            generation: 8,
            pcm: Data(repeating: 1, count: 320)
        ).isEmpty)
        XCTAssertTrue(state.append(
            generation: 9,
            pcm: Data(repeating: 1, count: 100)
        ).isEmpty)

        let full = state.append(
            generation: 9,
            pcm: Data(repeating: 2, count: 300)
        )
        XCTAssertEqual(full.count, 1)
        XCTAssertEqual(full[0].validByteCount, 320)
        XCTAssertEqual(state.snapshot.pendingByteCount, 80)

        let tail = state.flush(generation: 9)
        XCTAssertEqual(tail.count, 1)
        XCTAssertEqual(tail[0].validByteCount, 80)
        XCTAssertEqual(tail[0].pcm.count, 320)
        XCTAssertEqual(tail[0].pcm.suffix(240), Data(repeating: 0, count: 240))

        XCTAssertEqual(
            state.complete(full[0], gain: 1)?.cumulativeValidRenderedByteCount,
            320
        )
        XCTAssertEqual(
            state.complete(tail[0], gain: 1)?.cumulativeValidRenderedByteCount,
            400
        )
    }

    func testPausePreservesFramesAndResetInvalidatesOldCompletion() {
        let state = PlaybackStateMachine()
        state.reset(generation: 4)
        state.pause(generation: 4)
        XCTAssertTrue(state.append(
            generation: 4,
            pcm: Data(repeating: 3, count: 320)
        ).isEmpty)
        XCTAssertEqual(state.snapshot.queuedFrameCount, 1)

        let resumed = state.resume(generation: 4)
        XCTAssertEqual(resumed.count, 1)
        state.reset(generation: 5)
        XCTAssertNil(state.complete(resumed[0], gain: 1))
        XCTAssertEqual(state.snapshot.cumulativeValidRenderedByteCount, 0)
    }

    func testDuckGainMatchesSpeakerPCM() {
        var pcm = Data()
        for sample: Int16 in [10_000, -10_000, 1_000, -1_000] {
            var value = sample.littleEndian
            withUnsafeBytes(of: &value) { pcm.append(contentsOf: $0) }
        }
        XCTAssertEqual(
            decodePCM(PlaybackStateMachine.applyingGain(0.2, to: pcm)),
            [2_000, -2_000, 200, -200]
        )
    }

    func testMaximumPCMChunkSplitsOnceAndRetainsOnlyTail() {
        let state = PlaybackStateMachine()
        state.reset(generation: 1)
        let byteCount = FishMemoryPlaybackFrameProtocol.maximumPCMByteCount
        let frames = state.append(
            generation: 1,
            pcm: Data(repeating: 0x12, count: byteCount)
        )
        XCTAssertEqual(frames.count, byteCount / 320)
        XCTAssertEqual(state.snapshot.pendingByteCount, byteCount % 320)
        XCTAssertEqual(frames.first?.pcm, Data(repeating: 0x12, count: 320))
        XCTAssertEqual(frames.last?.ordinal, UInt64(frames.count - 1))
    }

    private func decodePCM(_ data: Data) -> [Int16] {
        stride(from: 0, to: data.count, by: 2).map { offset in
            Int16(bitPattern: UInt16(data[offset]) | UInt16(data[offset + 1]) << 8)
        }
    }
}
