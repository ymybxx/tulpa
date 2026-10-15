import FishMemoryAudioCore
import Foundation

enum PlaybackFrameReaderError: Error, CustomStringConvertible {
    case truncatedFrame(expected: Int, actual: Int)
    case invalidPayloadLength(magic: String, actual: Int)

    var description: String {
        switch self {
        case let .truncatedFrame(expected, actual):
            "truncated-frame expected=\(expected) actual=\(actual)"
        case let .invalidPayloadLength(magic, actual):
            "invalid-payload-length magic=\(magic) actual=\(actual)"
        }
    }
}

/// Reads the Python-to-helper stream on one blocking thread. stdout remains a
/// separate framed stream; diagnostics are stderr-only.
final class PlaybackFrameReader: @unchecked Sendable {
    typealias OnPacket = @Sendable (PlaybackInputPacket) -> Void
    typealias OnEndOfFile = @Sendable () -> Void
    typealias OnError = @Sendable (Error) -> Void

    private let input: FileHandle
    private let onPacket: OnPacket
    private let onEndOfFile: OnEndOfFile
    private let onError: OnError
    private let lock = NSLock()
    private var hasStarted = false
    private lazy var worker: Thread = {
        let thread = Thread { [weak self] in
            self?.readLoop()
        }
        thread.name = "fish-memory.playback-frame-reader"
        thread.qualityOfService = .userInitiated
        return thread
    }()

    init(
        input: FileHandle = .standardInput,
        onPacket: @escaping OnPacket,
        onEndOfFile: @escaping OnEndOfFile,
        onError: @escaping OnError
    ) {
        self.input = input
        self.onPacket = onPacket
        self.onEndOfFile = onEndOfFile
        self.onError = onError
    }

    func start() {
        lock.lock()
        let shouldStart = !hasStarted
        hasStarted = true
        lock.unlock()
        if shouldStart {
            worker.start()
        }
    }

    private func readLoop() {
        do {
            while let headerData = try readExact(
                FishMemoryPlaybackFrameProtocol.headerByteCount,
                allowCleanEndOfFile: true
            ) {
                let header = try FishMemoryPlaybackFrameProtocol.decodeHeader(
                    headerData
                )
                try validateInboundLength(header)
                guard let payload = try readExact(
                    header.payloadByteCount,
                    allowCleanEndOfFile: false
                ) else {
                    throw PlaybackFrameReaderError.truncatedFrame(
                        expected: header.payloadByteCount,
                        actual: 0
                    )
                }
                var packetData = headerData
                packetData.append(payload)
                onPacket(try FishMemoryPlaybackFrameProtocol.decodeInbound(
                    packetData
                ))
            }
            onEndOfFile()
        } catch {
            onError(error)
        }
    }

    private func validateInboundLength(_ header: PlaybackFrameHeader) throws {
        if header.magic == FishMemoryPlaybackFrameProtocol.pcmMagic {
            let pcmByteCount = header.payloadByteCount
                - FishMemoryPlaybackFrameProtocol.generationByteCount
            guard pcmByteCount > 0, pcmByteCount % 2 == 0,
                  pcmByteCount <= FishMemoryPlaybackFrameProtocol.maximumPCMByteCount
            else {
                throw PlaybackFrameReaderError.invalidPayloadLength(
                    magic: "FMP1",
                    actual: header.payloadByteCount
                )
            }
            return
        }
        if header.magic == FishMemoryPlaybackFrameProtocol.controlMagic,
           header.payloadByteCount
            == FishMemoryPlaybackFrameProtocol.controlPayloadByteCount
        {
            return
        }
        throw PlaybackFrameReaderError.invalidPayloadLength(
            magic: String(data: header.magic, encoding: .utf8) ?? "unknown",
            actual: header.payloadByteCount
        )
    }

    private func readExact(
        _ byteCount: Int,
        allowCleanEndOfFile: Bool
    ) throws -> Data? {
        var result = Data(capacity: byteCount)
        while result.count < byteCount {
            let chunk = try input.read(
                upToCount: byteCount - result.count
            ) ?? Data()
            if chunk.isEmpty {
                if result.isEmpty, allowCleanEndOfFile {
                    return nil
                }
                throw PlaybackFrameReaderError.truncatedFrame(
                    expected: byteCount,
                    actual: result.count
                )
            }
            result.append(chunk)
        }
        return result
    }
}
