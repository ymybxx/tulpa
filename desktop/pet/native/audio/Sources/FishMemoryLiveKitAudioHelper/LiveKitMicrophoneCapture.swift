import FishMemoryAudioCore
import Foundation
import LiveKit

final class LiveKitMicrophoneCapture: @unchecked Sendable {
    private let writer: BoundedFrameWriter
    private let onFatalError: @Sendable (Error) -> Void
    private let automaticVoiceProcessingLowPowerEnabled: Bool
    private let onPowerTransition: @Sendable (VoiceProcessingPowerTransition) -> Void
    private let lock = NSLock()
    private var audioManager: AudioManager?
    private var inputObserver: ProcessedInputTapObserver?
    private var voiceProcessingController: AdaptiveVoiceProcessingController?
    private var routeMonitor: DefaultAudioDeviceMonitor?
    private var currentOutputRoute: VoiceOutputRoute?
    private var currentInputUID: String?
    private var processingOptions: AudioProcessingOptions?
    private var isStarted = false
    private var microphoneEnabled = ProcessInfo.processInfo.environment["FISH_MEMORY_MICROPHONE_ENABLED"] != "false"

    init(
        writer: BoundedFrameWriter,
        automaticVoiceProcessingLowPowerEnabled: Bool = true,
        onPowerTransition: @escaping @Sendable (
            VoiceProcessingPowerTransition
        ) -> Void = { _ in },
        onFatalError: @escaping @Sendable (Error) -> Void
    ) {
        self.writer = writer
        self.automaticVoiceProcessingLowPowerEnabled =
            automaticVoiceProcessingLowPowerEnabled
        self.onPowerTransition = onPowerTransition
        self.onFatalError = onFatalError
    }

    func start() throws -> String {
        lock.lock()
        defer { lock.unlock() }
        guard !isStarted else { return "already-started" }

        // The audio-engine ADM must be selected before anything initializes
        // WebRTC. No Room is ever created: LiveKit owns only the local
        // microphone device and Apple's Voice Processing I/O path.
        LiveKitSDK.disableLogging()
        try AudioManager.set(audioDeviceModuleType: .audioEngine)

        let audio = AudioManager.shared
        try audio.set(microphoneMuteMode: .voiceProcessing)
        audio.isMicrophoneMuted = !microphoneEnabled
        let outputRoute = try CoreAudioOutputRoute.current()
        let inputUID = try CoreAudioOutputRoute.currentInputUID()
        try audio.setPlatformVoiceProcessingAllowed(true)
        audio.isVoiceProcessingBypassed = false
        audio.isVoiceProcessingAGCEnabled = false
        audio.isAdvancedDuckingEnabled = false
        if #available(macOS 14.0, *) {
            audio.duckingLevel = .min
        }

        // Force the coupled echo-cancellation/noise-suppression path to Apple's
        // platform implementation. Unlike `.automatic`, `.platform` rejects
        // startup if VPIO is unavailable instead of silently falling back to a
        // second WebRTC software APM. AGC and the high-pass filter stay off.
        let processingOptions = AudioProcessingOptions(
            echoCancellation: true,
            autoGainControl: false,
            noiseSuppression: true,
            highpassFilter: false,
            echoCancellationMode: .platform,
            autoGainControlMode: .platform,
            noiseSuppressionMode: .platform,
            highpassFilterMode: .automatic
        )
        let bypassAdapter = PlatformVoiceProcessingBypassAdapter(
            audioManager: audio
        )
        let voiceProcessingController = AdaptiveVoiceProcessingController(
            automaticLowPowerEnabled:
                automaticVoiceProcessingLowPowerEnabled,
            applyBypass: { bypassed in
                bypassAdapter.apply(bypassed)
            },
            onTransition: onPowerTransition,
            onInvariantFailure: onFatalError
        )
        let processedSink = ProcessedMicrophoneSink(
            writer: writer,
            voiceProcessingController: voiceProcessingController,
            onFatalError: onFatalError
        )
        let mirrors = PlaybackOutputMirrors()
        let playback = LiveKitStreamingPlayback(
            writer: writer,
            voiceProcessingProtector: voiceProcessingController,
            mirrors: mirrors,
            onFatalError: onFatalError
        )
        let observer = ProcessedInputTapObserver(
            sink: processedSink,
            playback: playback,
            mirrors: mirrors,
            onFatalError: onFatalError
        )
        observer.setMicrophoneEnabled(microphoneEnabled)
        AudioEngineObserverChain.install(
            observer,
            replacing: audio.mixer,
            setObservers: audio.set(engineObservers:)
        )

        do {
            try audio.startLocalRecording(
                audioProcessingOptions: processingOptions
            )
        } catch {
            observer.stop()
            AudioEngineObserverChain.restore(
                audio.mixer,
                removing: observer,
                setObservers: audio.set(engineObservers:)
            )
            throw error
        }

        audioManager = audio
        inputObserver = observer
        self.voiceProcessingController = voiceProcessingController
        self.processingOptions = processingOptions
        currentOutputRoute = outputRoute
        currentInputUID = inputUID
        isStarted = true
        let monitor = DefaultAudioDeviceMonitor { [weak self] in
            self?.refreshDevices()
        }
        do {
            try monitor.start()
            routeMonitor = monitor
        } catch {
            isStarted = false
            voiceProcessingController.stop()
            try? audio.stopLocalRecording()
            observer.stop()
            AudioEngineObserverChain.restore(
                audio.mixer, removing: observer, setObservers: audio.set(engineObservers:)
            )
            audioManager = nil
            inputObserver = nil
            self.voiceProcessingController = nil
            self.processingOptions = nil
            currentOutputRoute = nil
            currentInputUID = nil
            throw error
        }

        let effective = audio.audioProcessingState
        let platform = audio.platformVoiceProcessingState
        let automaticQuietMilliseconds =
            VoiceProcessingLowPowerConfiguration.production
                .acoustic.processedQuietMilliseconds
        return "aec=\(effective.echoCancellation.effective) " +
            "voiceOutput=\(outputRoute.voiceOutputUID) " +
            "mirrorOutputs=\(outputRoute.mirrorOutputUIDs.joined(separator: ",")) " +
            "ns=\(effective.noiseSuppression.effective) " +
            "agc=\(effective.autoGainControl.effective) " +
            "vpioActive=\(platform.voiceProcessingEnabled.isActive) " +
            "vpioBypassed=\(platform.voiceProcessingBypassed.isActive) " +
            "automaticVpioLowPower=" +
            "\(automaticVoiceProcessingLowPowerEnabled) " +
            "automaticVpioQuietMs=" +
            "\(automaticQuietMilliseconds)"
    }

    func handle(_ packet: PlaybackInputPacket) {
        if case let .control(operation, _) = packet,
           operation == .muteMicrophone || operation == .unmuteMicrophone {
            lock.lock()
            microphoneEnabled = operation == .unmuteMicrophone
            inputObserver?.setMicrophoneEnabled(microphoneEnabled)
            audioManager?.isMicrophoneMuted = !microphoneEnabled
            lock.unlock()
            FileHandle.standardError.write(Data("fish-memory-audio microphone enabled=\(microphoneEnabled)\n".utf8))
            return
        }
        lock.lock()
        let observer = inputObserver
        lock.unlock()
        observer?.handle(packet)
    }

    /// Diagnostic A/B control for measuring Apple's VPIO processing cost.
    /// Returning nil distinguishes a stopped helper from a rejected state
    /// change; production policy does not toggle this automatically.
    func setVoiceProcessingBypassed(_ bypassed: Bool) throws -> Bool? {
        lock.lock()
        defer { lock.unlock() }
        guard isStarted, let voiceProcessingController else { return nil }
        return voiceProcessingController.setDiagnosticBypass(bypassed).actual
    }

    func stop() {
        lock.lock()
        defer { lock.unlock() }
        guard isStarted else { return }
        isStarted = false
        routeMonitor?.stop()
        routeMonitor = nil

        let observer = inputObserver
        voiceProcessingController?.stop()
        if let audioManager {
            try? audioManager.stopLocalRecording()
            observer?.stop()
            AudioEngineObserverChain.restore(
                audioManager.mixer,
                removing: observer,
                setObservers: audioManager.set(engineObservers:)
            )
        } else {
            observer?.stop()
        }
        inputObserver = nil
        voiceProcessingController = nil
        audioManager = nil
        currentOutputRoute = nil
        currentInputUID = nil
        processingOptions = nil
    }

    private func refreshDevices() {
        lock.lock()
        defer { lock.unlock() }
        guard isStarted, let audioManager, let inputObserver,
              let currentOutputRoute, let processingOptions else { return }
        do {
            let route = try CoreAudioOutputRoute.current()
            let inputUID = try CoreAudioOutputRoute.currentInputUID()
            guard route != currentOutputRoute || inputUID != currentInputUID else { return }
            if route.voiceOutputUID == currentOutputRoute.voiceOutputUID,
               inputUID == currentInputUID {
                // Adding/removing only a mirrored output leaves VPIO and its
                // microphone clock running, with no voice-recognition restart.
                inputObserver.updateMirrorRoute(route)
            } else {
                // Keep the observer/player and their unconfirmed PCM. Existing
                // graph callbacks suspend and resume playback across reconnect.
                try audioManager.stopLocalRecording()
                try audioManager.startLocalRecording(audioProcessingOptions: processingOptions)
                audioManager.isMicrophoneMuted = !microphoneEnabled
            }
            self.currentOutputRoute = route
            currentInputUID = inputUID
            FileHandle.standardError.write(Data(
                "fish-memory-audio state route-updated voiceOutput=\(route.voiceOutputUID) mirrorOutputs=\(route.mirrorOutputUIDs.joined(separator: ","))\n".utf8
            ))
        } catch {
            onFatalError(error)
        }
    }
}

private final class PlatformVoiceProcessingBypassAdapter: @unchecked Sendable {
    private let audioManager: AudioManager

    init(audioManager: AudioManager) {
        self.audioManager = audioManager
    }

    func apply(_ bypassed: Bool) -> VoiceProcessingBypassApplyResult {
        audioManager.isVoiceProcessingBypassed = bypassed
        let state = audioManager.platformVoiceProcessingState
            .voiceProcessingBypassed
        return VoiceProcessingBypassApplyResult(
            requested: state.isRequested,
            actual: state.isActive
        )
    }
}

/// Installs Fish's single graph observer while capture is active, then
/// restores LiveKit's default observer without leaving a retained `next`
/// chain behind. `AudioManager` only rewires adjacent entries and does not
/// clear a stale tail when given a one-element array.
enum AudioEngineObserverChain {
    static func install(
        _ observer: any AudioEngineObserver,
        replacing defaultObserver: any AudioEngineObserver,
        setObservers: ([any AudioEngineObserver]) -> Void
    ) {
        var installedObserver = observer
        var detachedDefaultObserver = defaultObserver
        installedObserver.next = nil
        detachedDefaultObserver.next = nil
        setObservers([installedObserver])
    }

    static func restore(
        _ defaultObserver: any AudioEngineObserver,
        removing observer: (any AudioEngineObserver)?,
        setObservers: ([any AudioEngineObserver]) -> Void
    ) {
        if var removedObserver = observer {
            removedObserver.next = nil
        }
        var restoredDefaultObserver = defaultObserver
        restoredDefaultObserver.next = nil
        setObservers([restoredDefaultObserver])
    }
}
