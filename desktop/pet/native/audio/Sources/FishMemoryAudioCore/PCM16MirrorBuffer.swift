import Foundation

/// A bounded bridge between the authoritative voice renderer and an auxiliary
/// output clock. It never owns TTS progress or holds up microphone capture.
public final class PCM16MirrorBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float]
    private var head = 0
    private var count = 0

    public init(capacity: Int = 8_000) {
        precondition(capacity > 0)
        samples = [Float](repeating: 0, count: capacity)
    }

    public func append(_ pcm: Data) {
        lock.lock()
        defer { lock.unlock() }
        pcm.withUnsafeBytes { bytes in
            for index in stride(from: 0, to: bytes.count - bytes.count % 2, by: 2) {
                let value = UInt16(bytes[index]) | UInt16(bytes[index + 1]) << 8
                if count == samples.count {
                    head = (head + 1) % samples.count
                    count -= 1
                }
                samples[(head + count) % samples.count] = Float(Int16(bitPattern: value)) / 32768
                count += 1
            }
        }
    }

    /// Fill the whole output quantum, using silence when no confirmed audio
    /// is available. Overflow retains recent audio instead of growing delay.
    @discardableResult
    public func render(
        into output: UnsafeMutablePointer<Float>, frames: Int, channels: Int = 1
    ) -> Int {
        lock.lock()
        defer { lock.unlock() }
        let consumed = min(frames, count)
        for index in 0..<consumed {
            for channel in 0..<channels {
                output[index * channels + channel] = samples[(head + index) % samples.count]
            }
        }
        if consumed < frames {
            output.advanced(by: consumed * channels).initialize(
                repeating: 0, count: (frames - consumed) * channels
            )
        }
        head = (head + consumed) % samples.count
        count -= consumed
        return consumed
    }

    public func clear() {
        lock.lock()
        head = 0
        count = 0
        lock.unlock()
    }
}
