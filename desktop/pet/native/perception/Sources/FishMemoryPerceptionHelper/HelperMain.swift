import AVFoundation
import CoreGraphics
import Darwin
import Foundation
import PerceptionCore

@main
struct FishMemoryPerceptionHelper {
    @MainActor
    static func main() {
        if ProcessInfo.processInfo.arguments.contains(
            "--computer-history-input-probe"
        ) {
            runComputerHistoryInputProbe()
            return
        }
        // A long-running app keeps its first Screen Recording and microphone
        // answers until it relaunches; a fresh helper process reads the current grants.
        if ProcessInfo.processInfo.arguments.contains("--permission-probe") {
            let screen = CGPreflightScreenCaptureAccess()
            let microphone: String
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized: microphone = "granted"
            case .notDetermined: microphone = "not-determined"
            default: microphone = "denied"
            }
            FileHandle.standardOutput.write(Data("{\"screenCapture\":\(screen),\"microphone\":\"\(microphone)\"}\n".utf8))
            return
        }
        if ProcessInfo.processInfo.arguments.contains("--capture-warmup") {
            let arguments = ProcessInfo.processInfo.arguments
            let includeAudio = arguments.contains("--with-system-audio") || arguments.contains("--audio-only")
            let includeScreen = !arguments.contains("--audio-only")
            Task { @MainActor in
                let result = await CaptureWarmup.run(includeScreen: includeScreen, includeAudio: includeAudio)
                let data = (try? JSONEncoder().encode(result)) ?? Data("{}".utf8)
                FileHandle.standardOutput.write(data + Data("\n".utf8))
                Darwin.exit(EXIT_SUCCESS)
            }
            RunLoop.main.run()
            return
        }
        let runtime = PerceptionRuntime(onShutdown: {
            Darwin.exit(EXIT_SUCCESS)
        })
        runtime.start()

        DispatchQueue.global(qos: .utility).async {
            while let line = readLine(strippingNewline: true) {
                guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    continue
                }
                DispatchQueue.main.async {
                    runtime.handle(line: line)
                }
            }
            DispatchQueue.main.async {
                Task { @MainActor in await runtime.shutdown() }
            }
        }

        RunLoop.main.run()
    }

    @MainActor
    private static func runComputerHistoryInputProbe() {
        let duration = probeDuration()
        let monitor = ComputerHistoryMonitor(
            onSuppressed: { report in
                writeProbeRecord(type: "suppressed", body: report)
            },
            onEvent: { event in
                writeProbeRecord(type: "event", body: event)
            }
        )
        writeProbeRecord(type: "status", body: monitor.start())

        let statusTimer = Timer(timeInterval: 1, repeats: true) { _ in
            DispatchQueue.main.async {
                writeProbeRecord(type: "status", body: monitor.currentReport())
            }
        }
        RunLoop.main.add(statusTimer, forMode: .common)
        RunLoop.main.run(until: Date().addingTimeInterval(duration))
        statusTimer.invalidate()
        monitor.stop()
        writeProbeRecord(type: "status", body: monitor.currentReport())
    }

    private static func probeDuration() -> TimeInterval {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(
            of: "--computer-history-input-probe"
        ), arguments.indices.contains(index + 1),
            let requested = TimeInterval(arguments[index + 1])
        else { return 15 }
        return min(max(requested, 5), 120)
    }

    private static func writeProbeRecord<Body: Encodable>(
        type: String,
        body: Body
    ) {
        do {
            let record = ComputerHistoryProbeRecord(type: type, body: body)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(record)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data([0x0A]))
        } catch {
            FileHandle.standardError.write(Data(
                "computer-history probe encoding failed\n".utf8
            ))
        }
    }
}

private struct ComputerHistoryProbeRecord<Body: Encodable>: Encodable {
    let type: String
    let body: Body
}
