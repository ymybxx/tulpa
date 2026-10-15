import Foundation
import ScreenCaptureKit

/// Onboarding runs this once, right after Screen Recording is granted. The
/// first real capture is what makes macOS ask its follow-up questions (direct
/// screen access without the window picker, then recording system audio).
/// Asking them here, one after the other, keeps them from stacking up later
/// in the middle of use when the user no longer expects them.
public enum CaptureWarmup {
    public struct Result: Codable, Equatable, Sendable {
        /// False also when the screen was not asked about (audio only).
        public let screen: Bool
        /// A CapabilityState raw value, "unsupported" or "skipped".
        public let systemAudio: String
    }

    @MainActor
    public static func run(includeScreen: Bool = true, includeAudio: Bool) async -> Result {
        let screen = includeScreen ? await warmScreen() : false
        let audio = includeAudio ? await warmSystemAudio() : "skipped"
        return Result(screen: screen, systemAudio: audio)
    }

    private static func warmScreen() async -> Bool {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false,
                onScreenWindowsOnly: true
            )
            guard let display = content.displays.first else { return false }
            if #available(macOS 14.0, *) {
                let configuration = SCStreamConfiguration()
                configuration.width = 64
                configuration.height = 36
                _ = try await SCScreenshotManager.captureImage(
                    contentFilter: SCContentFilter(display: display, excludingWindows: []),
                    configuration: configuration
                )
            }
            return true
        } catch {
            return false
        }
    }

    @MainActor
    private static func warmSystemAudio() async -> String {
        let callbacks = SystemAudioCallbacks(
            onSamples: { _ in },
            onCaptureGap: { _ in },
            onStatus: { _ in }
        )
        guard let monitor = makeSystemAudioMonitor(callbacks: callbacks) else {
            return "unsupported"
        }
        let report = await monitor.start()
        // Starting the device is what raises the prompt; give it a moment to show.
        try? await Task.sleep(nanoseconds: 1_500_000_000)
        _ = await monitor.stop()
        return report.state.rawValue
    }
}
