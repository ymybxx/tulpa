import Foundation

public enum FishMemoryPlaybackFrameProtocol {
    public static let pcmMagic = Data("FMP1".utf8)
    public static let controlMagic = Data("FMC1".utf8)
    public static let renderedMagic = Data("FMR1".utf8)

    public static let headerByteCount = 8
    public static let generationByteCount = MemoryLayout<UInt64>.size
    public static let controlPayloadByteCount = 1 + generationByteCount
    public static let renderedPayloadByteCount =
        MemoryLayout<UInt64>.size * 3 + MemoryLayout<UInt32>.size
            + FishMemoryAudioFrameProtocol.pcmByteCount
    public static let maximumPCMByteCount = 1_048_576

    public static func encodePCM(
        generation: UInt64,
        pcm: Data
    ) throws -> Data {
        try validatePCM(pcm)
        let payloadByteCount = generationByteCount + pcm.count
        var packet = Data(capacity: headerByteCount + payloadByteCount)
        packet.append(pcmMagic)
        packet.appendLittleEndian(UInt32(payloadByteCount))
        packet.appendLittleEndian(generation)
        packet.append(pcm)
        return packet
    }

    public static func encodeControl(
        operation: PlaybackControlOperation,
        generation: UInt64
    ) -> Data {
        var packet = Data(capacity: headerByteCount + controlPayloadByteCount)
        packet.append(controlMagic)
        packet.appendLittleEndian(UInt32(controlPayloadByteCount))
        packet.append(operation.rawValue)
        packet.appendLittleEndian(generation)
        return packet
    }

    public static func encodeRendered(
        generation: UInt64,
        cumulativeValidRenderedByteCount: UInt64,
        uptimeNanoseconds: UInt64,
        validByteCount: UInt32,
        pcm: Data
    ) throws -> Data {
        guard pcm.count == FishMemoryAudioFrameProtocol.pcmByteCount else {
            throw PlaybackFrameProtocolError.invalidRenderedPCMByteCount(
                expected: FishMemoryAudioFrameProtocol.pcmByteCount,
                actual: pcm.count
            )
        }
        guard validByteCount <= UInt32(pcm.count), validByteCount % 2 == 0 else {
            throw PlaybackFrameProtocolError.invalidValidByteCount(
                actual: validByteCount
            )
        }

        var packet = Data(capacity: headerByteCount + renderedPayloadByteCount)
        packet.append(renderedMagic)
        packet.appendLittleEndian(UInt32(renderedPayloadByteCount))
        packet.appendLittleEndian(generation)
        packet.appendLittleEndian(cumulativeValidRenderedByteCount)
        packet.appendLittleEndian(uptimeNanoseconds)
        packet.appendLittleEndian(validByteCount)
        packet.append(pcm)
        return packet
    }

    public static func decodeInbound(_ packet: Data) throws -> PlaybackInputPacket {
        let header = try decodeHeader(packet)
        guard packet.count == headerByteCount + header.payloadByteCount else {
            throw PlaybackFrameProtocolError.invalidPacketByteCount(
                expected: headerByteCount + header.payloadByteCount,
                actual: packet.count
            )
        }

        switch header.magic {
        case pcmMagic:
            guard header.payloadByteCount > generationByteCount else {
                throw PlaybackFrameProtocolError.invalidPCMByteCount(
                    actual: max(0, header.payloadByteCount - generationByteCount)
                )
            }
            let generation = packet.readLittleEndian(
                UInt64.self,
                at: headerByteCount
            )
            let pcm = packet.subdata(
                in: headerByteCount + generationByteCount ..< packet.count
            )
            try validatePCM(pcm)
            return .pcm(generation: generation, pcm: pcm)

        case controlMagic:
            guard header.payloadByteCount == controlPayloadByteCount else {
                throw PlaybackFrameProtocolError.invalidControlPayloadByteCount(
                    expected: controlPayloadByteCount,
                    actual: header.payloadByteCount
                )
            }
            guard let operation = PlaybackControlOperation(
                rawValue: packet[headerByteCount]
            ) else {
                throw PlaybackFrameProtocolError.invalidControlOperation(
                    actual: packet[headerByteCount]
                )
            }
            let generation = packet.readLittleEndian(
                UInt64.self,
                at: headerByteCount + 1
            )
            return .control(operation: operation, generation: generation)

        default:
            throw PlaybackFrameProtocolError.invalidMagic
        }
    }

    public static func decodeRendered(
        _ packet: Data
    ) throws -> PlaybackRenderedFrame {
        let header = try decodeHeader(packet)
        guard header.magic == renderedMagic else {
            throw PlaybackFrameProtocolError.invalidMagic
        }
        guard header.payloadByteCount == renderedPayloadByteCount else {
            throw PlaybackFrameProtocolError.invalidRenderedPayloadByteCount(
                expected: renderedPayloadByteCount,
                actual: header.payloadByteCount
            )
        }
        guard packet.count == headerByteCount + renderedPayloadByteCount else {
            throw PlaybackFrameProtocolError.invalidPacketByteCount(
                expected: headerByteCount + renderedPayloadByteCount,
                actual: packet.count
            )
        }

        let generationOffset = headerByteCount
        let cumulativeOffset = generationOffset + MemoryLayout<UInt64>.size
        let uptimeOffset = cumulativeOffset + MemoryLayout<UInt64>.size
        let validOffset = uptimeOffset + MemoryLayout<UInt64>.size
        let pcmOffset = validOffset + MemoryLayout<UInt32>.size
        let validByteCount = packet.readLittleEndian(UInt32.self, at: validOffset)
        guard validByteCount <= UInt32(FishMemoryAudioFrameProtocol.pcmByteCount),
              validByteCount % 2 == 0
        else {
            throw PlaybackFrameProtocolError.invalidValidByteCount(
                actual: validByteCount
            )
        }
        return PlaybackRenderedFrame(
            generation: packet.readLittleEndian(UInt64.self, at: generationOffset),
            cumulativeValidRenderedByteCount: packet.readLittleEndian(
                UInt64.self,
                at: cumulativeOffset
            ),
            uptimeNanoseconds: packet.readLittleEndian(
                UInt64.self,
                at: uptimeOffset
            ),
            validByteCount: validByteCount,
            pcm: packet.subdata(in: pcmOffset ..< packet.count)
        )
    }

    public static func decodeHeader(_ header: Data) throws -> PlaybackFrameHeader {
        guard header.count >= headerByteCount else {
            throw PlaybackFrameProtocolError.invalidHeaderByteCount(
                expected: headerByteCount,
                actual: header.count
            )
        }
        let magic = Data(header.prefix(4))
        guard magic == pcmMagic || magic == controlMagic || magic == renderedMagic else {
            throw PlaybackFrameProtocolError.invalidMagic
        }
        return PlaybackFrameHeader(
            magic: magic,
            payloadByteCount: Int(header.readLittleEndian(UInt32.self, at: 4))
        )
    }

    private static func validatePCM(_ pcm: Data) throws {
        guard !pcm.isEmpty, pcm.count % 2 == 0,
              pcm.count <= maximumPCMByteCount
        else {
            throw PlaybackFrameProtocolError.invalidPCMByteCount(actual: pcm.count)
        }
    }
}

public enum PlaybackControlOperation: UInt8, Equatable, Sendable {
    case reset = 1
    case pause = 2
    case resume = 3
    case duck = 4
    case unduck = 5
    case stop = 6
    case flush = 7
    case ensureProcessed = 8
    case muteMicrophone = 9
    case unmuteMicrophone = 10
    case muteSpeaker = 11
    case unmuteSpeaker = 12
}

public enum PlaybackInputPacket: Equatable, Sendable {
    case pcm(generation: UInt64, pcm: Data)
    case control(operation: PlaybackControlOperation, generation: UInt64)
}

public struct PlaybackFrameHeader: Equatable, Sendable {
    public let magic: Data
    public let payloadByteCount: Int

    public init(magic: Data, payloadByteCount: Int) {
        self.magic = magic
        self.payloadByteCount = payloadByteCount
    }
}

public struct PlaybackRenderedFrame: Equatable, Sendable {
    public let generation: UInt64
    public let cumulativeValidRenderedByteCount: UInt64
    public let uptimeNanoseconds: UInt64
    public let validByteCount: UInt32
    public let pcm: Data

    public init(
        generation: UInt64,
        cumulativeValidRenderedByteCount: UInt64,
        uptimeNanoseconds: UInt64,
        validByteCount: UInt32,
        pcm: Data
    ) {
        self.generation = generation
        self.cumulativeValidRenderedByteCount = cumulativeValidRenderedByteCount
        self.uptimeNanoseconds = uptimeNanoseconds
        self.validByteCount = validByteCount
        self.pcm = pcm
    }
}

public enum PlaybackFrameProtocolError: Error, Equatable {
    case invalidMagic
    case invalidHeaderByteCount(expected: Int, actual: Int)
    case invalidPacketByteCount(expected: Int, actual: Int)
    case invalidPCMByteCount(actual: Int)
    case invalidControlPayloadByteCount(expected: Int, actual: Int)
    case invalidRenderedPayloadByteCount(expected: Int, actual: Int)
    case invalidRenderedPCMByteCount(expected: Int, actual: Int)
    case invalidControlOperation(actual: UInt8)
    case invalidValidByteCount(actual: UInt32)
}
