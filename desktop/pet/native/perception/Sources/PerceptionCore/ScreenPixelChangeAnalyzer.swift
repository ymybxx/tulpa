import CoreVideo
import Foundation

public struct ScreenPixelChangeAnalyzerConfiguration: Equatable, Sendable {
    public let sampleColumns: Int
    public let sampleRows: Int
    public let colorDeltaThreshold: UInt8

    public init(
        sampleColumns: Int = 96,
        sampleRows: Int = 54,
        colorDeltaThreshold: UInt8 = 16
    ) {
        precondition(sampleColumns > 0)
        precondition(sampleRows > 0)
        self.sampleColumns = sampleColumns
        self.sampleRows = sampleRows
        self.colorDeltaThreshold = colorDeltaThreshold
    }

    public static let `default` = ScreenPixelChangeAnalyzerConfiguration()
}

/// Compares a small, uniformly sampled BGRA representation in memory. The
/// compositor's dirty rectangles are only repaint hints and are deliberately
/// not used as proof that visible pixels changed.
public struct ScreenPixelChangeAnalyzer: Sendable {
    public let configuration: ScreenPixelChangeAnalyzerConfiguration
    private var previousSample: [UInt8]?

    public init(
        configuration: ScreenPixelChangeAnalyzerConfiguration = .default
    ) {
        self.configuration = configuration
    }

    public mutating func changedAreaRatio(
        pixelBuffer: CVPixelBuffer
    ) -> Double? {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer)
            == kCVPixelFormatType_32BGRA
        else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0 else { return nil }
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            == kCVReturnSuccess
        else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let columns = min(configuration.sampleColumns, width)
        let rows = min(configuration.sampleRows, height)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let base = baseAddress.assumingMemoryBound(to: UInt8.self)
        var current: [UInt8] = []
        current.reserveCapacity(columns * rows * 3)
        for row in 0 ..< rows {
            let y = min(height - 1, ((row * 2 + 1) * height) / (rows * 2))
            for column in 0 ..< columns {
                let x = min(
                    width - 1,
                    ((column * 2 + 1) * width) / (columns * 2)
                )
                let offset = y * bytesPerRow + x * 4
                current.append(base[offset])
                current.append(base[offset + 1])
                current.append(base[offset + 2])
            }
        }

        defer { previousSample = current }
        guard let previousSample,
              previousSample.count == current.count
        else { return 1 }
        let threshold = Int(configuration.colorDeltaThreshold)
        var changedSamples = 0
        var index = 0
        while index < current.count {
            let blue = abs(Int(current[index]) - Int(previousSample[index]))
            let green = abs(
                Int(current[index + 1]) - Int(previousSample[index + 1])
            )
            let red = abs(
                Int(current[index + 2]) - Int(previousSample[index + 2])
            )
            if max(blue, green, red) >= threshold {
                changedSamples += 1
            }
            index += 3
        }
        return Double(changedSamples) / Double(columns * rows)
    }

    public mutating func reset() {
        previousSample = nil
    }
}
