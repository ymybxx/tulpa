import Foundation

final class StatusReporter: @unchecked Sendable {
    private let output: FileHandle
    private let lock = NSLock()

    init(output: FileHandle = .standardError) {
        self.output = output
    }

    func ready(_ message: String) {
        write(level: "ready", message: message)
    }

    func warning(_ message: String) {
        write(level: "warning", message: message)
    }

    func state(_ message: String) {
        write(level: "state", message: message)
    }

    func error(_ message: String) {
        write(level: "error", message: message)
    }

    func stopped(_ message: String) {
        write(level: "stopped", message: message)
    }

    private func write(level: String, message: String) {
        let sanitized = message
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
        let data = Data("fish-memory-audio \(level) \(sanitized)\n".utf8)
        lock.lock()
        defer { lock.unlock() }
        do {
            try output.write(contentsOf: data)
        } catch {
            // stderr is diagnostic only. Never redirect diagnostics to stdout,
            // which is reserved exclusively for binary PCM frames.
        }
    }
}
