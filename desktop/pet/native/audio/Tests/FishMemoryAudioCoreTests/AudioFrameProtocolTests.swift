import FishMemoryAudioCore
import Foundation
import XCTest

final class AudioFrameProtocolTests: XCTestCase {
    func testEncodesExactLittleEndianWireFormat() throws {
        let pcm = Data((0 ..< FishMemoryAudioFrameProtocol.pcmByteCount).map {
            UInt8($0 % 251)
        })
        let packet = try FishMemoryAudioFrameProtocol.encode(
            sequence: 0x0102_0304_0506_0708,
            uptimeNanoseconds: 0x1112_1314_1516_1718,
            pcm: pcm
        )

        XCTAssertEqual(packet.count, 344)
        XCTAssertEqual(Array(packet[0 ..< 4]), Array("FMA1".utf8))
        XCTAssertEqual(Array(packet[4 ..< 8]), [0x50, 0x01, 0x00, 0x00])
        XCTAssertEqual(
            Array(packet[8 ..< 16]),
            [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        )
        XCTAssertEqual(
            Array(packet[16 ..< 24]),
            [0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11]
        )
        XCTAssertEqual(packet.suffix(pcm.count), pcm)
    }

    func testRoundTripsFrame() throws {
        let pcm = Data(repeating: 0xA5, count: 320)
        let encoded = try FishMemoryAudioFrameProtocol.encode(
            sequence: 42,
            uptimeNanoseconds: 987_654_321,
            pcm: pcm
        )

        XCTAssertEqual(
            try FishMemoryAudioFrameProtocol.decode(encoded),
            DecodedAudioFrame(
                sequence: 42,
                uptimeNanoseconds: 987_654_321,
                pcm: pcm
            )
        )
    }

    func testBorrowedSampleEncoderMatchesDataEncoderExactly() throws {
        let samples = (0 ..< FishMemoryAudioFrameProtocol.samplesPerFrame).map {
            index in
            Int16(truncatingIfNeeded: index &* 977 &- 32_000)
        }
        let pcm = littleEndianPCM(samples)
        let expected = try FishMemoryAudioFrameProtocol.encode(
            sequence: UInt64.max - 7,
            uptimeNanoseconds: 0xFEDC_BA98_7654_3210,
            pcm: pcm
        )
        let actual = try samples.withUnsafeBufferPointer { borrowed in
            try FishMemoryAudioFrameProtocol.encode(
                sequence: UInt64.max - 7,
                uptimeNanoseconds: 0xFEDC_BA98_7654_3210,
                pcmSamples: borrowed
            )
        }

        XCTAssertEqual(actual, expected)
        XCTAssertEqual(try FishMemoryAudioFrameProtocol.decode(actual).pcm, pcm)
    }

    func testRejectsWrongPCMSize() {
        XCTAssertThrowsError(try FishMemoryAudioFrameProtocol.encode(
            sequence: 0,
            uptimeNanoseconds: 0,
            pcm: Data(repeating: 0, count: 319)
        ))
    }

    func testBorrowedSampleEncoderRejectsWrongSampleCount() {
        let samples = Array(repeating: Int16(0), count: 159)
        XCTAssertThrowsError(try samples.withUnsafeBufferPointer { borrowed in
            try FishMemoryAudioFrameProtocol.encode(
                sequence: 0,
                uptimeNanoseconds: 0,
                pcmSamples: borrowed
            )
        }) { error in
            XCTAssertEqual(
                error as? AudioFrameProtocolError,
                .invalidPCMByteCount(expected: 320, actual: 318)
            )
        }
    }

    private func littleEndianPCM(_ samples: [Int16]) -> Data {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let value = UInt16(bitPattern: sample)
            data.append(UInt8(truncatingIfNeeded: value))
            data.append(UInt8(truncatingIfNeeded: value >> 8))
        }
        return data
    }
}
