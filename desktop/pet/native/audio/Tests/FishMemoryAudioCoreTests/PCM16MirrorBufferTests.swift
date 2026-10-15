import FishMemoryAudioCore
import Foundation
import XCTest

final class PCM16MirrorBufferTests: XCTestCase {
    func testStereoOutputPreservesSamplesAndPadsOnlyMissingFrames() {
        let buffer = PCM16MirrorBuffer(capacity: 8)
        buffer.append(pcm([16_384, -32_768, 0]))
        var output = [Float](repeating: 99, count: 10)
        let count = output.withUnsafeMutableBufferPointer {
            buffer.render(into: $0.baseAddress!, frames: 5, channels: 2)
        }
        XCTAssertEqual(count, 3)
        XCTAssertEqual(output, [0.5, 0.5, -1, -1, 0, 0, 0, 0, 0, 0])
    }

    func testSlowAuxiliaryDeviceCannotAccumulateUnboundedDelay() {
        let buffer = PCM16MirrorBuffer(capacity: 4)
        buffer.append(pcm([1, 2, 3]))
        XCTAssertEqual(render(buffer, 2), [1, 2])
        buffer.append(pcm([4, 5, 6, 7]))
        XCTAssertEqual(render(buffer, 6), [4, 5, 6, 7, 0, 0])
    }

    func testInterruptionClearsOldTailBeforeNextGeneration() {
        let buffer = PCM16MirrorBuffer(capacity: 8)
        buffer.append(pcm([100, 200, 300]))
        XCTAssertEqual(render(buffer, 1), [100])
        buffer.clear()
        buffer.append(pcm([-500]))
        XCTAssertEqual(render(buffer, 3), [-500, 0, 0])
    }

    private func pcm(_ samples: [Int16]) -> Data {
        var result = Data()
        for sample in samples {
            var value = sample.littleEndian
            withUnsafeBytes(of: &value) { result.append(contentsOf: $0) }
        }
        return result
    }

    private func render(_ buffer: PCM16MirrorBuffer, _ frames: Int) -> [Int16] {
        var result = [Float](repeating: 99, count: frames)
        result.withUnsafeMutableBufferPointer {
            _ = buffer.render(into: $0.baseAddress!, frames: frames)
        }
        return result.map { Int16($0 * 32768) }
    }
}
