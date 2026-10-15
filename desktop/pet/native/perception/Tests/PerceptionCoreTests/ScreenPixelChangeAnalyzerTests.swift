import CoreVideo
import XCTest
@testable import PerceptionCore

final class ScreenPixelChangeAnalyzerTests: XCTestCase {
    func testIdenticalFramesBecomeActuallyStableDespiteCompositorRepaints() throws {
        var analyzer = ScreenPixelChangeAnalyzer(configuration: .init(
            sampleColumns: 10,
            sampleRows: 10,
            colorDeltaThreshold: 16
        ))
        let frame = try pixelBuffer(width: 100, height: 100, value: 40)

        XCTAssertEqual(analyzer.changedAreaRatio(pixelBuffer: frame), 1)
        XCTAssertEqual(analyzer.changedAreaRatio(pixelBuffer: frame), 0)
    }

    func testFullFramePixelChangeIsMeasuredAsMotion() throws {
        var analyzer = ScreenPixelChangeAnalyzer(configuration: .init(
            sampleColumns: 10,
            sampleRows: 10,
            colorDeltaThreshold: 16
        ))
        _ = analyzer.changedAreaRatio(
            pixelBuffer: try pixelBuffer(width: 100, height: 100, value: 0)
        )

        XCTAssertEqual(
            analyzer.changedAreaRatio(
                pixelBuffer: try pixelBuffer(width: 100, height: 100, value: 255)
            ),
            1
        )
    }

    func testTinyLocalChangeDoesNotLookLikeWholeScreenMotion() throws {
        var analyzer = ScreenPixelChangeAnalyzer(configuration: .init(
            sampleColumns: 10,
            sampleRows: 10,
            colorDeltaThreshold: 16
        ))
        let first = try pixelBuffer(width: 100, height: 100, value: 0)
        let second = try pixelBuffer(width: 100, height: 100, value: 0)
        try fill(second, value: 255, rectangle: (0, 0, 10, 10))
        _ = analyzer.changedAreaRatio(pixelBuffer: first)

        let ratio = try XCTUnwrap(
            analyzer.changedAreaRatio(pixelBuffer: second)
        )
        XCTAssertLessThanOrEqual(ratio, 0.02)
    }

    func testResetMakesNextFrameANewBaseline() throws {
        var analyzer = ScreenPixelChangeAnalyzer(configuration: .init(
            sampleColumns: 4,
            sampleRows: 4,
            colorDeltaThreshold: 16
        ))
        let frame = try pixelBuffer(width: 16, height: 16, value: 10)
        _ = analyzer.changedAreaRatio(pixelBuffer: frame)
        XCTAssertEqual(analyzer.changedAreaRatio(pixelBuffer: frame), 0)

        analyzer.reset()
        XCTAssertEqual(analyzer.changedAreaRatio(pixelBuffer: frame), 1)
    }

    private func pixelBuffer(
        width: Int,
        height: Int,
        value: UInt8
    ) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        XCTAssertEqual(CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            nil,
            &buffer
        ), kCVReturnSuccess)
        let result = try XCTUnwrap(buffer)
        try fill(result, value: value, rectangle: (0, 0, width, height))
        return result
    }

    private func fill(
        _ buffer: CVPixelBuffer,
        value: UInt8,
        rectangle: (x: Int, y: Int, width: Int, height: Int)
    ) throws {
        XCTAssertEqual(
            CVPixelBufferLockBaseAddress(buffer, []),
            kCVReturnSuccess
        )
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let base = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer))
            .assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        for y in rectangle.y ..< (rectangle.y + rectangle.height) {
            for x in rectangle.x ..< (rectangle.x + rectangle.width) {
                let offset = y * bytesPerRow + x * 4
                base[offset] = value
                base[offset + 1] = value
                base[offset + 2] = value
                base[offset + 3] = 255
            }
        }
    }
}
