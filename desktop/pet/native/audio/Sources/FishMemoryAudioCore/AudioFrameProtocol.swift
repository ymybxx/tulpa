import Foundation

public enum FishMemoryAudioFrameProtocol {
    public static let magic = Data("FMA1".utf8)
    public static let sampleRate = 16_000
    public static let channels = 1
    public static let frameDurationMilliseconds = 10
    public static let samplesPerFrame = 160
    public static let pcmByteCount = samplesPerFrame * MemoryLayout<Int16>.size
    public static let payloadByteCount =
        MemoryLayout<UInt64>.size * 2 + pcmByteCount
    public static let headerByteCount = magic.count + MemoryLayout<UInt32>.size
    public static let packetByteCount = headerByteCount + payloadByteCount

    public static func encode(
        sequence: UInt64,
        uptimeNanoseconds: UInt64,
        pcm: Data
    ) throws -> Data {
        guard pcm.count == pcmByteCount else {
            throw AudioFrameProtocolError.invalidPCMByteCount(
                expected: pcmByteCount,
                actual: pcm.count
            )
        }

        var packet = Data(capacity: packetByteCount)
        packet.append(magic)
        packet.appendLittleEndian(UInt32(payloadByteCount))
        packet.appendLittleEndian(sequence)
        packet.appendLittleEndian(uptimeNanoseconds)
        packet.append(pcm)
        return packet
    }

    /// Encodes borrowed native `Int16` samples directly into the final packet.
    /// This avoids creating a separate 320-byte PCM `Data` before the packet is
    /// handed to the asynchronous writer.
    public static func encode(
        sequence: UInt64,
        uptimeNanoseconds: UInt64,
        pcmSamples: UnsafeBufferPointer<Int16>
    ) throws -> Data {
        guard pcmSamples.count == samplesPerFrame else {
            throw AudioFrameProtocolError.invalidPCMByteCount(
                expected: pcmByteCount,
                actual: pcmSamples.count * MemoryLayout<Int16>.size
            )
        }

        var packet = Data(capacity: packetByteCount)
        packet.append(magic)
        packet.appendLittleEndian(UInt32(payloadByteCount))
        packet.appendLittleEndian(sequence)
        packet.appendLittleEndian(uptimeNanoseconds)
#if _endian(little)
        packet.append(contentsOf: UnsafeRawBufferPointer(
            start: pcmSamples.baseAddress!,
            count: pcmByteCount
        ))
#else
        for sample in pcmSamples {
            packet.appendLittleEndian(UInt16(bitPattern: sample))
        }
#endif
        return packet
    }

    public static func decode(_ packet: Data) throws -> DecodedAudioFrame {
        guard packet.count == packetByteCount else {
            throw AudioFrameProtocolError.invalidPacketByteCount(
                expected: packetByteCount,
                actual: packet.count
            )
        }
        guard packet.prefix(magic.count) == magic else {
            throw AudioFrameProtocolError.invalidMagic
        }

        let payloadLength = packet.readLittleEndian(
            UInt32.self,
            at: magic.count
        )
        guard payloadLength == UInt32(payloadByteCount) else {
            throw AudioFrameProtocolError.invalidPayloadByteCount(
                expected: payloadByteCount,
                actual: Int(payloadLength)
            )
        }

        let payloadStart = headerByteCount
        let sequence = packet.readLittleEndian(UInt64.self, at: payloadStart)
        let uptime = packet.readLittleEndian(
            UInt64.self,
            at: payloadStart + MemoryLayout<UInt64>.size
        )
        let pcmStart = payloadStart + MemoryLayout<UInt64>.size * 2
        return DecodedAudioFrame(
            sequence: sequence,
            uptimeNanoseconds: uptime,
            pcm: packet.subdata(in: pcmStart ..< packet.count)
        )
    }

}

public struct DecodedAudioFrame: Equatable, Sendable {
    public let sequence: UInt64
    public let uptimeNanoseconds: UInt64
    public let pcm: Data

    public init(sequence: UInt64, uptimeNanoseconds: UInt64, pcm: Data) {
        self.sequence = sequence
        self.uptimeNanoseconds = uptimeNanoseconds
        self.pcm = pcm
    }
}

public enum AudioFrameProtocolError: Error, Equatable {
    case invalidMagic
    case invalidPacketByteCount(expected: Int, actual: Int)
    case invalidPayloadByteCount(expected: Int, actual: Int)
    case invalidPCMByteCount(expected: Int, actual: Int)
}
