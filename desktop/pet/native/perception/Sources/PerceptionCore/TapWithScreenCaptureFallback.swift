import CoreGraphics
import Foundation

/// Recording system audio through a Core Audio process tap needs its own
/// permission (System Settings › 录屏与系统录音 › 仅系统录音). ScreenCaptureKit
/// can record the same audio under Screen Recording alone, which System
/// Settings describes as covering "屏幕内容和音频". So when the tap is refused
/// but Screen Recording is allowed, capture falls back to ScreenCaptureKit and
/// one switch is enough.
@available(macOS 13.0, *)
@MainActor
final class TapWithScreenCaptureFallback: SystemAudioMonitoring {
    static let tapPermissionRequired = "core_audio_process_tap_permission_required"

    private let primary: any SystemAudioMonitoring
    private let makeFallback: @MainActor () -> any SystemAudioMonitoring
    private let screenRecordingAllowed: () -> Bool
    private var fallback: (any SystemAudioMonitoring)?

    init(
        primary: any SystemAudioMonitoring,
        makeFallback: @escaping @MainActor () -> any SystemAudioMonitoring,
        screenRecordingAllowed: @escaping () -> Bool = { CGPreflightScreenCaptureAccess() }
    ) {
        self.primary = primary
        self.makeFallback = makeFallback
        self.screenRecordingAllowed = screenRecordingAllowed
    }

    func start() async -> CapabilityReport {
        if let fallback { return await fallback.start() }
        let report = await primary.start()
        guard report.reason == Self.tapPermissionRequired, screenRecordingAllowed() else {
            return report
        }
        // The tap already released its resources when it reported the refusal.
        _ = await primary.stop()
        let next = makeFallback()
        fallback = next
        return await next.start()
    }

    func stop() async -> Bool {
        if let fallback { return await fallback.stop() }
        return await primary.stop()
    }
}
