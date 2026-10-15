import Darwin
import Dispatch
import FishMemoryAudioCore
import Foundation

final class ShutdownHandle: @unchecked Sendable {
    private let lock = NSLock()
    private weak var coordinator: ShutdownCoordinator?

    func install(_ coordinator: ShutdownCoordinator) {
        lock.lock()
        self.coordinator = coordinator
        lock.unlock()
    }

    func request(exitCode: Int32, reason: String) {
        lock.lock()
        let target = coordinator
        lock.unlock()
        DispatchQueue.main.async {
            target?.shutdown(exitCode: exitCode, reason: reason)
        }
    }
}

final class ShutdownCoordinator: @unchecked Sendable {
    private let capture: LiveKitMicrophoneCapture
    private let writer: BoundedFrameWriter
    private let reporter: StatusReporter
    private let standardOutput: FileHandle
    private let lock = NSLock()
    private var didShutdown = false
    private var signalSources: [DispatchSourceSignal] = []

    init(
        capture: LiveKitMicrophoneCapture,
        writer: BoundedFrameWriter,
        reporter: StatusReporter,
        standardOutput: FileHandle = .standardOutput
    ) {
        self.capture = capture
        self.writer = writer
        self.reporter = reporter
        self.standardOutput = standardOutput
    }

    func installSignalHandlers() {
        signalSources = [SIGTERM, SIGINT].map { signalNumber in
            Darwin.signal(signalNumber, SIG_IGN)
            let source = DispatchSource.makeSignalSource(
                signal: signalNumber,
                queue: .main
            )
            source.setEventHandler { [weak self] in
                self?.shutdown(
                    exitCode: EXIT_SUCCESS,
                    reason: "signal=\(signalNumber)"
                )
            }
            source.resume()
            return source
        }
        Darwin.signal(SIGPIPE, SIG_IGN)
    }

    func shutdown(exitCode: Int32, reason: String) -> Never {
        lock.lock()
        if didShutdown {
            lock.unlock()
            Darwin.exit(exitCode)
        }
        didShutdown = true
        lock.unlock()

        capture.stop()
        writer.stop(discardPending: true)
        try? standardOutput.close()
        let snapshot = writer.snapshot
        reporter.stopped(
            "reason=\(reason) written=\(snapshot.writtenFrameCount) " +
                "dropped=\(snapshot.droppedFrameCount)"
        )
        Darwin.exit(exitCode)
    }
}
