import FishMemoryAudioCore
@testable import FishMemoryLiveKitAudioHelper
import Foundation
import XCTest

final class PlaybackFrameReaderTests: XCTestCase {
    func testReadsFragmentedPacketsAndCleanEOFInOrder() throws {
        let pipe = Pipe()
        let result = ReaderResult()
        let finished = DispatchSemaphore(value: 0)
        let reader = PlaybackFrameReader(
            input: pipe.fileHandleForReading,
            onPacket: { result.append($0) },
            onEndOfFile: { finished.signal() },
            onError: { error in
                result.fail(error)
                finished.signal()
            }
        )
        reader.start()

        let reset = FishMemoryPlaybackFrameProtocol.encodeControl(
            operation: .reset,
            generation: 17
        )
        let pcm = Data(repeating: 0, count: 320)
        let audio = try FishMemoryPlaybackFrameProtocol.encodePCM(
            generation: 17,
            pcm: pcm
        )
        try pipe.fileHandleForWriting.write(contentsOf: reset.prefix(3))
        try pipe.fileHandleForWriting.write(contentsOf: reset.dropFirst(3))
        try pipe.fileHandleForWriting.write(contentsOf: audio)
        try pipe.fileHandleForWriting.close()

        XCTAssertEqual(finished.wait(timeout: .now() + 2), .success)
        XCTAssertNil(result.errorDescription)
        XCTAssertEqual(result.packets, [
            .control(operation: .reset, generation: 17),
            .pcm(generation: 17, pcm: pcm),
        ])
        withExtendedLifetime(reader) {}
    }

    func testRejectsOversizeLengthWithoutWaitingForPayload() throws {
        let pipe = Pipe()
        let result = ReaderResult()
        let finished = DispatchSemaphore(value: 0)
        let reader = PlaybackFrameReader(
            input: pipe.fileHandleForReading,
            onPacket: { result.append($0) },
            onEndOfFile: { finished.signal() },
            onError: { error in
                result.fail(error)
                finished.signal()
            }
        )
        reader.start()

        try pipe.fileHandleForWriting.write(contentsOf: Data([
            0x46, 0x4D, 0x50, 0x31, 0xFF, 0xFF, 0xFF, 0x7F,
        ]))
        XCTAssertEqual(finished.wait(timeout: .now() + 2), .success)
        XCTAssertTrue(result.packets.isEmpty)
        XCTAssertTrue(result.errorDescription?.contains("invalid-payload-length") == true)
        try pipe.fileHandleForWriting.close()
        withExtendedLifetime(reader) {}
    }
}

private final class ReaderResult: @unchecked Sendable {
    private let lock = NSLock()
    private var storedPackets: [PlaybackInputPacket] = []
    private var storedErrorDescription: String?

    func append(_ packet: PlaybackInputPacket) {
        lock.lock()
        storedPackets.append(packet)
        lock.unlock()
    }

    func fail(_ error: Error) {
        lock.lock()
        storedErrorDescription = String(describing: error)
        lock.unlock()
    }

    var packets: [PlaybackInputPacket] {
        lock.lock()
        defer { lock.unlock() }
        return storedPackets
    }

    var errorDescription: String? {
        lock.lock()
        defer { lock.unlock() }
        return storedErrorDescription
    }
}
