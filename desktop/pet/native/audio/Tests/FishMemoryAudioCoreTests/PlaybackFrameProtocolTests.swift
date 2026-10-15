import FishMemoryAudioCore
import Foundation
import XCTest

final class PlaybackFrameProtocolTests: XCTestCase {
    func testPCMAndControlRoundTrip() throws {
        let pcm = Data([0x34, 0x12, 0x78, 0x56])
        let pcmPacket = try FishMemoryPlaybackFrameProtocol.encodePCM(
            generation: 0x0102_0304_0506_0708,
            pcm: pcm
        )
        XCTAssertEqual(Array(pcmPacket.prefix(8)), [
            0x46, 0x4D, 0x50, 0x31, 0x0C, 0, 0, 0,
        ])
        XCTAssertEqual(
            try FishMemoryPlaybackFrameProtocol.decodeInbound(pcmPacket),
            .pcm(generation: 0x0102_0304_0506_0708, pcm: pcm)
        )

        let control = FishMemoryPlaybackFrameProtocol.encodeControl(
            operation: .flush,
            generation: 42
        )
        XCTAssertEqual(control.count, 17)
        XCTAssertEqual(
            try FishMemoryPlaybackFrameProtocol.decodeInbound(control),
            .control(operation: .flush, generation: 42)
        )

        let ensureProcessed = FishMemoryPlaybackFrameProtocol.encodeControl(
            operation: .ensureProcessed,
            generation: 43
        )
        XCTAssertEqual(ensureProcessed.count, 17)
        XCTAssertEqual(
            try FishMemoryPlaybackFrameProtocol.decodeInbound(ensureProcessed),
            .control(operation: .ensureProcessed, generation: 43)
        )
    }

    func testRenderedRoundTripHasExactFixedPayload() throws {
        let pcm = Data(repeating: 0xA5, count: 320)
        let packet = try FishMemoryPlaybackFrameProtocol.encodeRendered(
            generation: 7,
            cumulativeValidRenderedByteCount: 640,
            uptimeNanoseconds: 123,
            validByteCount: 120,
            pcm: pcm
        )
        XCTAssertEqual(packet.count, 356)
        XCTAssertEqual(Array(packet.prefix(8)), [
            0x46, 0x4D, 0x52, 0x31, 0x5C, 0x01, 0, 0,
        ])
        XCTAssertEqual(
            try FishMemoryPlaybackFrameProtocol.decodeRendered(packet),
            PlaybackRenderedFrame(
                generation: 7,
                cumulativeValidRenderedByteCount: 640,
                uptimeNanoseconds: 123,
                validByteCount: 120,
                pcm: pcm
            )
        )
    }

    func testRejectsMalformedInput() {
        XCTAssertThrowsError(try FishMemoryPlaybackFrameProtocol.encodePCM(
            generation: 1,
            pcm: Data()
        ))
        XCTAssertThrowsError(try FishMemoryPlaybackFrameProtocol.encodePCM(
            generation: 1,
            pcm: Data([1])
        ))

        var invalidControl = FishMemoryPlaybackFrameProtocol.encodeControl(
            operation: .reset,
            generation: 1
        )
        invalidControl[8] = 255
        XCTAssertThrowsError(
            try FishMemoryPlaybackFrameProtocol.decodeInbound(invalidControl)
        )
    }
}
