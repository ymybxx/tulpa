import AppKit
import CoreAudio
import CoreMedia
import Darwin
import Foundation
import ScreenCaptureKit

@MainActor
public protocol SystemAudioMonitoring: AnyObject {
    func start() async -> CapabilityReport
    /// Returns true only after every backend-owned capture resource has been
    /// released. Callers must not construct a replacement backend on false.
    func stop() async -> Bool
}

public struct SystemAudioCallbacks: Sendable {
    public let onSamples: @Sendable (SystemAudioPCMBlock) -> Void
    public let onCaptureGap: @Sendable (SystemAudioCaptureGap) -> Void
    public let onStatus: @MainActor @Sendable (CapabilityReport) -> Void

    public init(
        onSamples: @escaping @Sendable (SystemAudioPCMBlock) -> Void,
        onCaptureGap: @escaping @Sendable (SystemAudioCaptureGap) -> Void,
        onStatus: @escaping @MainActor @Sendable (CapabilityReport) -> Void
    ) {
        self.onSamples = onSamples
        self.onCaptureGap = onCaptureGap
        self.onStatus = onStatus
    }
}

/// A separate audio-only stream leaves the display stream's visual filter
/// untouched. It neither requests microphone permission nor attaches a mic output.
@available(macOS 13.0, *)
@MainActor
final class SystemAudioMonitor: NSObject, SystemAudioMonitoring, SCStreamDelegate {
    private let callbacks: SystemAudioCallbacks
    private var stream: SCStream?
    private var samples: SystemAudioSamples?
    private var captureNeedsStop = false
    private var outputNeedsRemoval = false
    private var stopTask: Task<Bool, Never>?
    private var refreshTask: Task<Void, Never>?
    private var applicationDecisions: [pid_t: SystemAudioApplicationDecision] = [:]
    private var generation = 0

    init(callbacks: SystemAudioCallbacks) {
        self.callbacks = callbacks
    }

    func start() async -> CapabilityReport {
        if stream != nil { return report(.running) }
        guard CGPreflightScreenCaptureAccess() else {
            return report(.unavailable, reason: "screen_recording_permission_required")
        }
        generation += 1
        let token = generation
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
            guard token == generation, !Task.isCancelled else { return report(.disabled) }
            guard let display = content.displays.first(where: { $0.displayID == CGMainDisplayID() })
                    ?? content.displays.first else {
                return report(.unavailable, reason: "display_not_found")
            }
            // Keep SCK's monitored application set small. Supplying every
            // non-Fish process through `includingApplications` makes replayd
            // create an RBS monitor for the entire desktop process catalog.
            let excluded = excludedApplications(content)
            let filter = SCContentFilter(
                display: display,
                excludingApplications: excluded,
                exceptingWindows: []
            )
            let configuration = makeSystemAudioStreamConfiguration()
            let next = SCStream(filter: filter, configuration: configuration, delegate: self)
            let output = SystemAudioSamples(callbacks: callbacks)
            try next.addStreamOutput(output, type: .audio, sampleHandlerQueue: output.queue)
            stream = next
            samples = output
            captureNeedsStop = true
            outputNeedsRemoval = true
            output.activate(stream: next)
            try await next.startCapture()
            guard token == generation, stream === next, !Task.isCancelled else {
                output.invalidate()
                if stopTask == nil { _ = await stop() }
                return report(.disabled)
            }
            startFilterRefresh(stream: next, displayID: display.displayID,
                               excludedIDs: Set(excluded.map(\.processID)), token: token)
            return report(.running)
        } catch {
            guard token == generation else { return report(.disabled) }
            _ = await stop()
            return report(.failed, reason: "system_audio_capture_start_failed")
        }
    }

    func stop() async -> Bool {
        if let stopTask { return await stopTask.value }
        generation += 1
        refreshTask?.cancel()
        refreshTask = nil
        let current = stream
        let output = samples
        output?.invalidate()
        guard let current else {
            captureNeedsStop = false
            outputNeedsRemoval = false
            samples = nil
            applicationDecisions.removeAll(keepingCapacity: false)
            return true
        }
        let task = Task { @MainActor [weak self, current, output] () -> Bool in
            guard let self else { return false }
            if self.captureNeedsStop {
                do {
                    try await current.stopCapture()
                    self.captureNeedsStop = false
                } catch {
                    return false
                }
            }
            if self.outputNeedsRemoval, let output {
                do {
                    try current.removeStreamOutput(output, type: .audio)
                    self.outputNeedsRemoval = false
                } catch {
                    return false
                }
            } else {
                self.outputNeedsRemoval = false
            }
            guard self.stream === current else { return false }
            self.stream = nil
            self.samples = nil
            self.applicationDecisions.removeAll(keepingCapacity: false)
            return true
        }
        stopTask = task
        let result = await task.value
        stopTask = nil
        return result
    }

    nonisolated func stream(_ stream: SCStream, didStopWithError error: Error) {
        Task { @MainActor [weak self] in
            guard let self, self.stream === stream else { return }
            self.captureNeedsStop = false
            _ = await self.stop()
            self.callbacks.onStatus(self.report(.failed, reason: "system_audio_capture_stopped"))
        }
    }

    private func startFilterRefresh(stream: SCStream, displayID: CGDirectDisplayID,
                                    excludedIDs: Set<pid_t>, token: Int) {
        refreshTask = Task { [weak self] in
            var previous = excludedIDs
            while !Task.isCancelled {
                // The filter only changes when applications launch or exit.
                // A two-second full process-tree scan kept waking an otherwise
                // idle desktop. Ten seconds still admits newly launched media
                // apps within the current semantic window while cutting those
                // enumerations by 80 percent.
                do { try await Task.sleep(nanoseconds: 10_000_000_000) } catch { return }
                guard let self, self.generation == token, self.stream === stream else { return }
                do {
                    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
                    guard !Task.isCancelled, self.generation == token, self.stream === stream else { return }
                    let excluded = self.excludedApplications(content)
                    let identities = Set(excluded.map(\.processID))
                    guard identities != previous else { continue }
                    guard let display = content.displays.first(where: { $0.displayID == displayID }) else {
                        throw SystemAudioCaptureError.displayUnavailable
                    }
                    self.samples?.markGap(reason: "application_filter_changed")
                    try await stream.updateContentFilter(SCContentFilter(
                        display: display,
                        excludingApplications: excluded,
                        exceptingWindows: []
                    ))
                    previous = identities
                } catch {
                    guard !Task.isCancelled, self.generation == token else { return }
                    _ = await self.stop()
                    self.callbacks.onStatus(self.report(.failed, reason: "system_audio_filter_update_failed"))
                    return
                }
            }
        }
    }

    private func excludedApplications(_ content: SCShareableContent) -> [SCRunningApplication] {
        let liveProcessIDs = Set(content.applications.map(\.processID))
        applicationDecisions = applicationDecisions.filter {
            liveProcessIDs.contains($0.key)
        }
        return content.applications.filter { application in
            if let cached = applicationDecisions[application.processID],
               cached.bundleIdentifier == application.bundleIdentifier {
                return cached.excluded
            }
            let excluded = excludesKnownFishApplication(application)
            applicationDecisions[application.processID] = .init(
                bundleIdentifier: application.bundleIdentifier,
                excluded: excluded
            )
            return excluded
        }
    }

    private func excludesKnownFishApplication(_ application: SCRunningApplication) -> Bool {
        let parent = getppid()
        if application.processID == parent { return true }
        if !application.bundleIdentifier.isEmpty {
            return isFishSystemAudioBundleIdentifier(application.bundleIdentifier)
        }
        // Only bundle-less helper processes need the more expensive process
        // path/ancestry check. Normal desktop applications are decided above.
        return shouldExcludeSystemAudioApplication(
            bundleIdentifier: nil,
            processIdentifier: application.processID,
            parentProcessIdentifier: parent,
            executablePath: systemAudioProcessPath(application.processID),
            ancestors: systemAudioAncestorProcessIDs(application.processID)
        )
    }

    private func report(_ state: CapabilityState, reason: String? = nil) -> CapabilityReport {
        CapabilityReport(supported: true, requested: state != .disabled, state: state,
                         permission: CGPreflightScreenCaptureAccess() ? .granted : .notDetermined,
                         reason: reason ?? (state == .running
                             ? "screen_capture_kit_fallback" : nil))
    }
}

enum SystemAudioCaptureError: Error { case displayUnavailable }

@available(macOS 13.0, *)
func makeSystemAudioStreamConfiguration() -> SCStreamConfiguration {
    let configuration = SCStreamConfiguration()
    configuration.width = 2
    configuration.height = 2
    // An audio-only output must not ask WindowServer to downscale the entire
    // changing display into its otherwise unused video surface.
    configuration.sourceRect = CGRect(x: 0, y: 0, width: 1, height: 1)
    configuration.minimumFrameInterval = CMTime(value: 60, timescale: 1)
    configuration.queueDepth = 1
    configuration.showsCursor = false
    configuration.capturesAudio = true
    configuration.sampleRate = SystemAudioWindow.sampleRate
    configuration.channelCount = 1
    configuration.excludesCurrentProcessAudio = true
    if #available(macOS 15.0, *) { configuration.captureMicrophone = false }
    return configuration
}

private struct SystemAudioApplicationDecision {
    let bundleIdentifier: String?
    let excluded: Bool
}

func shouldExcludeSystemAudioApplication(
    bundleIdentifier: String?, processIdentifier: Int32, parentProcessIdentifier: Int32?,
    executablePath: String?, ancestors: [Int32] = []
) -> Bool {
    if processIdentifier == parentProcessIdentifier { return true }
    // Python/sounddevice and Electron renderer playback may have no Fish
    // bundle identifier. They still belong to the same desktop process tree.
    if let parentProcessIdentifier, ancestors.contains(parentProcessIdentifier) { return true }
    if let bundleIdentifier, isFishSystemAudioBundleIdentifier(bundleIdentifier) {
        return true
    }
    guard let executablePath else { return false }
    return ["fish-memory-livekit-audio-helper", "fish-memory-runtime", "fish-memory-perception-helper"]
        .contains(URL(fileURLWithPath: executablePath).lastPathComponent)
}

private func isFishSystemAudioBundleIdentifier(_ value: String) -> Bool {
    ["app.fishmemory.", "com.fishmemory.", "com.fish-memory."].contains(
        where: value.hasPrefix
    )
}

func systemAudioProcessPath(_ pid: pid_t) -> String? {
    var bytes = [CChar](repeating: 0, count: 4_096)
    let count = proc_pidpath(pid, &bytes, UInt32(bytes.count))
    return count > 0 ? String(cString: bytes) : nil
}

func systemAudioAncestorProcessIDs(_ pid: pid_t) -> [pid_t] {
    var result: [pid_t] = []
    var current = pid
    for _ in 0..<32 {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.size)
        guard proc_pidinfo(current, PROC_PIDTBSDINFO, 0, &info, size) == size else { break }
        let parent = pid_t(info.pbi_ppid)
        guard parent > 1, parent != current, !result.contains(parent) else { break }
        result.append(parent)
        current = parent
    }
    return result
}

/// ScreenCaptureKit callbacks stay binary and in-process. The module-owned
/// segmenter is the only component that creates protocol payloads.
@available(macOS 13.0, *)
private final class SystemAudioSamples: NSObject, SCStreamOutput, @unchecked Sendable {
    let queue = DispatchQueue(label: "app.fishmemory.perception.audio.capture", qos: .userInitiated)
    private let delivery: SystemAudioCaptureDelivery
    private var stream: SCStream?

    init(callbacks: SystemAudioCallbacks) {
        delivery = SystemAudioCaptureDelivery(callbacks: callbacks)
    }

    func activate(stream: SCStream) {
        queue.sync {
            self.stream = stream
        }
    }

    func invalidate() {
        // Revocation cannot wait for a blocked stdout pipe.
        delivery.invalidate()
        queue.async { [self] in
            stream = nil
        }
    }

    func markGap(reason: String) {
        queue.async { [weak self] in self?.gap(reason) }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of outputType: SCStreamOutputType) {
        guard self.stream === stream, outputType == .audio, sampleBuffer.isValid else { return }
        guard let pcm = systemAudioPCM(sampleBuffer) else { gap("invalid_audio_format"); return }
        delivery.submit(.init(pcm: pcm))
    }

    private func gap(_ reason: String) {
        guard stream != nil else { return }
        delivery.submit(.init(reason: reason))
    }
}

/// SCK supplies LPCM in an AudioBufferList. Validate its real format rather
/// than assuming the requested configuration describes arbitrary callback data.
func systemAudioPCM(_ sample: CMSampleBuffer) -> Data? {
    guard let description = sample.formatDescription,
          let pointer = CMAudioFormatDescriptionGetStreamBasicDescription(description) else { return nil }
    let format = pointer.pointee
    let count = CMSampleBufferGetNumSamples(sample)
    guard format.mFormatID == kAudioFormatLinearPCM,
          format.mSampleRate == 16_000, format.mChannelsPerFrame == 1,
          format.mFormatFlags & kAudioFormatFlagIsBigEndian == 0,
          count > 0, count <= 16_000 else { return nil }
    let isFloat = format.mFormatFlags & kAudioFormatFlagIsFloat != 0 && format.mBitsPerChannel == 32
    let isInt16 = format.mFormatFlags & kAudioFormatFlagIsSignedInteger != 0 && format.mBitsPerChannel == 16
    guard isFloat || isInt16 else { return nil }
    var list = AudioBufferList(mNumberBuffers: 1, mBuffers: AudioBuffer(mNumberChannels: 1, mDataByteSize: 0, mData: nil))
    var block: CMBlockBuffer?
    let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sample, bufferListSizeNeededOut: nil, bufferListOut: &list,
        bufferListSize: MemoryLayout<AudioBufferList>.size,
        blockBufferAllocator: kCFAllocatorDefault, blockBufferMemoryAllocator: kCFAllocatorDefault,
        flags: UInt32(kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment), blockBufferOut: &block
    )
    guard status == noErr, list.mNumberBuffers == 1,
          let bytes = list.mBuffers.mData,
          Int(list.mBuffers.mDataByteSize) == count * (isFloat ? 4 : 2) else { return nil }
    return withExtendedLifetime(block) {
        if isInt16 { return Data(bytes: bytes, count: count * 2) }
        let input = bytes.assumingMemoryBound(to: Float.self)
        var output = [Int16](repeating: 0, count: count)
        for i in 0..<count {
            guard input[i].isFinite else { return nil }
            output[i] = Int16(max(-32_768, min(32_767, (Double(input[i]) * 32_768).rounded()))).littleEndian
        }
        return output.withUnsafeBytes { Data($0) }
    }
}
