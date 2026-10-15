import Darwin
import Dispatch
import FishMemoryAudioCore
import Foundation

@main
enum FishMemoryLiveKitAudioHelper {
    static func main() {
        let reporter = StatusReporter()
        let standardOutput = FileHandle.standardOutput
        let shutdownHandle = ShutdownHandle()
        let environment = ProcessInfo.processInfo.environment
        let diagnosticBypassRequested = environment[
            "FISH_MEMORY_DIAGNOSTIC_VPIO_BYPASS"
        ] == "1"
        let automaticVpioLowPowerEnabled = environment[
            "FISH_MEMORY_AUTOMATIC_VPIO_LOW_POWER"
        ] != "0" && !diagnosticBypassRequested
        let powerTelemetryQueue = DispatchQueue(
            label: "fish-memory.vpio-power-telemetry",
            qos: .utility
        )

        let writer = BoundedFrameWriter(
            capacity: 256,
            write: { packet in
                try standardOutput.write(contentsOf: packet)
            },
            onWriteError: { error in
                reporter.error("stdout-write-failed detail=\(error)")
                shutdownHandle.request(
                    exitCode: EXIT_FAILURE,
                    reason: "stdout-write-failed"
                )
            }
        )

        let capture = LiveKitMicrophoneCapture(
            writer: writer,
            automaticVoiceProcessingLowPowerEnabled:
                automaticVpioLowPowerEnabled,
            onPowerTransition: { transition in
                powerTelemetryQueue.async {
                    reporter.state(transition.diagnosticDescription)
                }
            },
            onFatalError: { error in
                reporter.error("audio-processing-failed detail=\(error)")
                shutdownHandle.request(
                    exitCode: EXIT_FAILURE,
                    reason: "audio-processing-failed"
                )
            }
        )
        let coordinator = ShutdownCoordinator(
            capture: capture,
            writer: writer,
            reporter: reporter,
            standardOutput: standardOutput
        )
        shutdownHandle.install(coordinator)
        coordinator.installSignalHandlers()
        writer.start()

        do {
            let processingDiagnostics = try capture.start()
            let diagnosticBypassState: String
            if diagnosticBypassRequested {
                let actual = try capture.setVoiceProcessingBypassed(true)
                diagnosticBypassState = actual.map { String($0) } ?? "unavailable"
            } else {
                diagnosticBypassState = "not-requested"
            }
            reporter.ready(
                "protocol=FMA1 playbackProtocols=FMP1,FMC1,FMR1 " +
                    "sampleRate=16000 channels=1 frameMs=10 " +
                    "voiceProcessing=platform agc=false advancedDucking=false " +
                    "diagnosticVpioBypassRequested=\(diagnosticBypassRequested) " +
                    "diagnosticVpioBypassActual=\(diagnosticBypassState) " +
                    "automaticVpioLowPower=\(automaticVpioLowPowerEnabled) " +
                    processingDiagnostics
            )
            let reader = PlaybackFrameReader(
                onPacket: { packet in
                    capture.handle(packet)
                },
                onEndOfFile: {
                    shutdownHandle.request(
                        exitCode: EXIT_SUCCESS,
                        reason: "stdin-eof"
                    )
                },
                onError: { error in
                    reporter.error("stdin-frame-failed detail=\(error)")
                    shutdownHandle.request(
                        exitCode: EXIT_FAILURE,
                        reason: "stdin-frame-failed"
                    )
                }
            )
            reader.start()
            withExtendedLifetime(reader) {
                withExtendedLifetime(coordinator) {
                    dispatchMain()
                }
            }
        } catch {
            reporter.error("startup-failed detail=\(error)")
            coordinator.shutdown(
                exitCode: EXIT_FAILURE,
                reason: "startup-failed"
            )
        }

    }
}
