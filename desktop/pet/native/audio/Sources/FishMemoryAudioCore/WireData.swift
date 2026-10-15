import Foundation

extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { bytes in
            append(contentsOf: bytes)
        }
    }

    func readLittleEndian<T: FixedWidthInteger>(
        _: T.Type,
        at offset: Int
    ) -> T {
        var value: T = 0
        let size = MemoryLayout<T>.size
        _ = Swift.withUnsafeMutableBytes(of: &value) { destination in
            copyBytes(
                to: destination,
                from: offset ..< offset + size
            )
        }
        return T(littleEndian: value)
    }
}
