@preconcurrency import AVFoundation
import FishMemoryAudioCore
import Foundation
import LiveKit

/// Observes Apple's voice-processed microphone node before WebRTC transport.
///
/// LiveKit's local renderer is backed by the WebRTC APM callback. A standalone
/// capture with no Room or PeerConnection has no WebRTC AudioTransport, so that
/// callback is not driven even while AVAudioEngine and VPIO produce samples.
/// The engine observer is the SDK-supported hook for custom graph integration
/// and keeps this helper independent from a Room or a published track.
final class ProcessedInputTapObserver: AudioEngineObserver, @unchecked Sendable {
    var next: (any AudioEngineObserver)?

    private let sink: ProcessedMicrophoneSink
    private let playback: LiveKitStreamingPlayback
    private let mirrors: PlaybackOutputMirrors
    private let onFatalError: @Sendable (Error) -> Void
    private let mirrorReadiness = InputRenderReadinessGate()
    private let mirrorQueue = DispatchQueue(label: "fish-memory.output-mirror-configuration")
    // Accessed only on mirrorQueue, including shutdown, so a late first-frame
    // callback cannot reopen an output after this observer has stopped.
    private var mirrorConfigurationEnabled = true
    private let lock = NSLock()
    private weak var tappedNode: AVAudioNode?
    private var hasTap = false

    init(
        sink: ProcessedMicrophoneSink,
        playback: LiveKitStreamingPlayback,
        mirrors: PlaybackOutputMirrors,
        onFatalError: @escaping @Sendable (Error) -> Void
    ) {
        self.sink = sink
        self.playback = playback
        self.mirrors = mirrors
        self.onFatalError = onFatalError
    }

    func engineDidCreate(_ engine: AVAudioEngine) -> Int {
        playback.engineDidCreate(engine)
        return next?.engineDidCreate(engine) ?? 0
    }

    func engineWillConnectOutput(
        _ engine: AVAudioEngine,
        src: AVAudioNode,
        dst: AVAudioNode?,
        format: AVAudioFormat,
        context: [AnyHashable: Any]
    ) -> Int {
        do {
            try updateOutputRoute(engine)
        } catch {
            onFatalError(error)
            return -1
        }
        playback.engineWillConnectOutput(engine)
        return next?.engineWillConnectOutput(
            engine,
            src: src,
            dst: dst,
            format: format,
            context: context
        ) ?? 0
    }

    func engineWillConnectInput(
        _ engine: AVAudioEngine,
        src: AVAudioNode?,
        dst: AVAudioNode,
        format: AVAudioFormat,
        context: [AnyHashable: Any]
    ) -> Int {
        do {
            try updateOutputRoute(engine)
        } catch {
            onFatalError(error)
            return -1
        }
        if let src {
            installTap(on: src, format: format)
        }
        return next?.engineWillConnectInput(
            engine,
            src: src,
            dst: dst,
            format: format,
            context: context
        ) ?? 0
    }

    func engineWillStart(
        _ engine: AVAudioEngine,
        isPlayoutEnabled: Bool,
        isRecordingEnabled: Bool
    ) -> Int {
        mirrorReadiness.arm()
        playback.engineWillStart()
        return next?.engineWillStart(
            engine,
            isPlayoutEnabled: isPlayoutEnabled,
            isRecordingEnabled: isRecordingEnabled
        ) ?? 0
    }

    func engineDidStop(
        _ engine: AVAudioEngine,
        isPlayoutEnabled: Bool,
        isRecordingEnabled: Bool
    ) -> Int {
        mirrorReadiness.disarm()
        playback.engineDidStop()
        return next?.engineDidStop(
            engine,
            isPlayoutEnabled: isPlayoutEnabled,
            isRecordingEnabled: isRecordingEnabled
        ) ?? 0
    }

    func engineDidDisable(
        _ engine: AVAudioEngine,
        isPlayoutEnabled: Bool,
        isRecordingEnabled: Bool
    ) -> Int {
        mirrorReadiness.disarm()
        detach()
        playback.engineDidDisable()
        return next?.engineDidDisable(
            engine,
            isPlayoutEnabled: isPlayoutEnabled,
            isRecordingEnabled: isRecordingEnabled
        ) ?? 0
    }

    func engineWillRelease(_ engine: AVAudioEngine) -> Int {
        mirrorReadiness.disarm()
        detach()
        playback.engineWillRelease(engine)
        return next?.engineWillRelease(engine) ?? 0
    }

    func detach() {
        lock.lock()
        defer { lock.unlock() }
        guard hasTap else { return }
        tappedNode?.removeTap(onBus: 0)
        tappedNode = nil
        hasTap = false
    }

    func handle(_ packet: PlaybackInputPacket) {
        playback.handle(packet)
    }

    func setMicrophoneEnabled(_ enabled: Bool) {
        sink.setMicrophoneEnabled(enabled)
    }

    func updateMirrorRoute(_ route: VoiceOutputRoute) {
        mirrorQueue.async { [self] in
            guard mirrorConfigurationEnabled else { return }
            do {
                try mirrors.configure(route.mirrorOutputUIDs)
            } catch {
                onFatalError(error)
            }
        }
    }

    func stop() {
        mirrorReadiness.disarm()
        detach()
        playback.stop()
        mirrorQueue.sync {
            mirrorConfigurationEnabled = false
            mirrors.stop()
        }
    }

    private func updateOutputRoute(_ engine: AVAudioEngine) throws {
        let route = try CoreAudioOutputRoute.current()
        try CoreAudioOutputRoute.bindVoiceOutput(route, to: engine)
    }

    private func inputDidStart() {
        guard mirrorReadiness.claim() else { return }
        // VPIO can still be establishing its device graph when startRecording
        // returns. Opening another output unit during that work can leave its
        // input tap idle. The first processed frame is the actual readiness
        // boundary; never initialize Core Audio on the realtime callback.
        mirrorQueue.async { [self] in
            guard mirrorConfigurationEnabled else { return }
            do {
                try mirrors.configure(CoreAudioOutputRoute.current().mirrorOutputUIDs)
            } catch {
                onFatalError(error)
            }
        }
    }

    private func installTap(on node: AVAudioNode, format: AVAudioFormat) {
        lock.lock()
        defer { lock.unlock() }
        if hasTap, tappedNode === node {
            return
        }
        if hasTap {
            tappedNode?.removeTap(onBus: 0)
        }
        node.installTap(
            onBus: 0,
            bufferSize: max(1, AVAudioFrameCount(format.sampleRate / 100)),
            format: format
        ) { [weak self, sink, playback] buffer, _ in
            self?.inputDidStart()
            playback.engineDidRenderInput()
            sink.render(pcmBuffer: buffer)
        }
        tappedNode = node
        hasTap = true
    }
}
