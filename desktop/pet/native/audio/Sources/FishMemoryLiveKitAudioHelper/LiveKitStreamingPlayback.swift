@preconcurrency import AVFoundation
import Darwin
import Dispatch
import FishMemoryAudioCore
import Foundation

/// Streams Fish TTS through the same AVAudioEngine and Voice Processing I/O
/// unit that owns microphone capture. It deliberately does not use LiveKit's
/// SoundPlayer (a separate AVAudioEngine) or the room input mixer.
final class LiveKitStreamingPlayback: @unchecked Sendable {
    private let writer: BoundedFrameWriter
    private let onFatalError: @Sendable (Error) -> Void
    private let mirrors: (any RenderedAudioMirroring)?
    private let voiceProcessingProtector: (
        any VoiceProcessingPlaybackProtecting
    )?
    private let executionQueue = DispatchQueue(
        label: "fish-memory.livekit-streaming-playback",
        qos: .userInitiated
    )
    private let state = PlaybackStateMachine()
    private let inputRenderReadiness = InputRenderReadinessGate()
    private let renderedFrameOrderer = RenderedFrameOrderer()
    private let playerNode = AVAudioPlayerNode()
    private let playbackFormat: AVAudioFormat
    private weak var engine: AVAudioEngine?
    private var isAttached = false
    private var isConnected = false
    private var isEngineStarted = false
    private var hasFailed = false
    private var playbackEnabled = ProcessInfo.processInfo.environment["FISH_MEMORY_PLAYBACK_ENABLED"] != "false"
    private var outputGain: Float { playbackEnabled ? state.outputGain : 0 }
    private var scheduleEpoch: UInt64 = 0
    private var pendingFrames: [ScheduledPlaybackFrame] = []
    private var scheduledOrdinals: Set<UInt64> = []

    init(
        writer: BoundedFrameWriter,
        voiceProcessingProtector: (
            any VoiceProcessingPlaybackProtecting
        )? = nil,
        mirrors: (any RenderedAudioMirroring)? = nil,
        onFatalError: @escaping @Sendable (Error) -> Void
    ) {
        self.writer = writer
        self.voiceProcessingProtector = voiceProcessingProtector
        self.onFatalError = onFatalError
        self.mirrors = mirrors
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(FishMemoryAudioFrameProtocol.sampleRate),
            channels: AVAudioChannelCount(FishMemoryAudioFrameProtocol.channels),
            interleaved: false
        ) else {
            preconditionFailure("Unable to create fixed 16 kHz playback format")
        }
        playbackFormat = format
    }

    func handle(_ packet: PlaybackInputPacket) {
        executionQueue.sync {
            handleOnQueue(packet)
        }
    }

    func engineDidCreate(_ engine: AVAudioEngine) {
        executionQueue.sync {
            inputRenderReadiness.disarm()
            self.engine = engine
            // Match the device's render quantum before AVAudioEngine allocates
            // the player unit; negotiated larger IO buffers otherwise fail
            // with kAudioUnitErr_TooManyFramesToProcess (-10874).
            playerNode.auAudioUnit.maximumFramesToRender =
                engine.outputNode.auAudioUnit.maximumFramesToRender
            if !engine.attachedNodes.contains(playerNode) {
                engine.attach(playerNode)
            }
            isAttached = true
            isConnected = false
            isEngineStarted = false
        }
    }

    func engineWillConnectOutput(_ engine: AVAudioEngine) {
        executionQueue.sync {
            if !isAttached || self.engine !== engine {
                self.engine = engine
                playerNode.auAudioUnit.maximumFramesToRender =
                    engine.outputNode.auAudioUnit.maximumFramesToRender
                if !engine.attachedNodes.contains(playerNode) {
                    engine.attach(playerNode)
                }
                isAttached = true
            }
            guard !isConnected else { return }

            // Use the SDK's output-graph hook: its main mixer feeds this same
            // engine's VPIO output. Adding TTS upstream of that mixer preserves
            // the SDK output route and keeps the far-end reference inside VPIO
            // instead of exposing TTS as unrelated PortAudio output.
            engine.connect(
                playerNode,
                to: engine.mainMixerNode,
                format: playbackFormat
            )
            playerNode.volume = outputGain
            isConnected = true
            schedulePendingFrames()
        }
    }

    func engineWillStart() {
        executionQueue.sync {
            // This callback precedes AVAudioEngine.start(). Starting the
            // player here can be undone by the engine's reconnect work.
            isEngineStarted = true
            inputRenderReadiness.arm()
            schedulePendingFrames()
        }
    }

    func engineDidRenderInput() {
        // The tap runs every 10 ms. Only its first callback after each engine
        // start is useful as a post-start readiness signal; atomically consume
        // that signal so the realtime thread does not allocate and enqueue one
        // Dispatch work item for every subsequent audio quantum.
        guard inputRenderReadiness.claim() else { return }
        // The VPIO tap is the first reliable post-start signal. Do not block
        // its realtime thread or operate AVAudioPlayerNode from that thread.
        executionQueue.async { [weak self] in
            self?.playIfReady()
        }
    }

    func engineDidStop() {
        executionQueue.sync {
            suspendScheduledFrames()
        }
    }

    func engineDidDisable() {
        executionQueue.sync {
            suspendScheduledFrames()
            if let engine, isAttached {
                engine.disconnectNodeOutput(playerNode)
            }
            isConnected = false
        }
    }

    func engineWillRelease(_ engine: AVAudioEngine) {
        executionQueue.sync {
            suspendScheduledFrames()
            if engine.attachedNodes.contains(playerNode) {
                engine.detach(playerNode)
            }
            self.engine = nil
            isAttached = false
            isConnected = false
        }
    }

    func stop() {
        executionQueue.sync {
            mirrors?.clear()
            let generation = state.snapshot.generation
            scheduleEpoch &+= 1
            inputRenderReadiness.disarm()
            pendingFrames.removeAll(keepingCapacity: false)
            scheduledOrdinals.removeAll(keepingCapacity: false)
            renderedFrameOrderer.reset(keepingCapacity: false)
            if let generation {
                state.stop(generation: generation)
            }
            isEngineStarted = false
            playerNode.stop()
            if let engine, engine.attachedNodes.contains(playerNode) {
                engine.detach(playerNode)
            }
            engine = nil
            isAttached = false
            isConnected = false
            if let generation {
                voiceProcessingProtector?.endPlaybackProtection(
                    generation: generation
                )
            }
        }
    }

    private func handleOnQueue(_ packet: PlaybackInputPacket) {
        guard !hasFailed else { return }
        switch packet {
        case let .pcm(generation, pcm):
            schedule(state.append(generation: generation, pcm: pcm))

        case let .control(operation, generation):
            switch operation {
            case .muteSpeaker, .unmuteSpeaker:
                playbackEnabled = operation == .unmuteSpeaker
                playerNode.volume = outputGain
                if !playbackEnabled { mirrors?.clear() }
            case .muteMicrophone, .unmuteMicrophone:
                break // Capture controls are handled before the playback queue.
            case .reset:
                mirrors?.clear()
                let previousGeneration = state.snapshot.generation
                scheduleEpoch &+= 1
                pendingFrames.removeAll(keepingCapacity: true)
                scheduledOrdinals.removeAll(keepingCapacity: true)
                renderedFrameOrderer.reset(keepingCapacity: true)
                playerNode.stop()
                state.reset(generation: generation)
                playerNode.volume = outputGain
                if let previousGeneration {
                    voiceProcessingProtector?.endPlaybackProtection(
                        generation: previousGeneration
                    )
                }
                if voiceProcessingProtector?.prewarmPlayback(
                    generation: generation
                ) == false {
                    failVoiceProcessingRestore(reason: .playbackPrewarm)
                }

            case .pause:
                state.pause(generation: generation)
                if state.snapshot.generation == generation {
                    playerNode.pause()
                    mirrors?.clear()
                    voiceProcessingProtector?.endPlaybackProtection(
                        generation: generation
                    )
                }

            case .resume:
                let frames = state.resume(generation: generation)
                schedule(frames)
                playIfReady()

            case .duck:
                state.duck(generation: generation)
                if state.snapshot.generation == generation {
                    playerNode.volume = outputGain
                }

            case .unduck:
                state.unduck(generation: generation)
                if state.snapshot.generation == generation {
                    playerNode.volume = outputGain
                }

            case .stop:
                state.stop(generation: generation)
                if state.snapshot.generation == generation {
                    scheduleEpoch &+= 1
                    mirrors?.clear()
                    pendingFrames.removeAll(keepingCapacity: true)
                    scheduledOrdinals.removeAll(keepingCapacity: true)
                    renderedFrameOrderer.reset(keepingCapacity: true)
                    playerNode.stop()
                    voiceProcessingProtector?.endPlaybackProtection(
                        generation: generation
                    )
                }

            case .flush:
                schedule(state.flush(generation: generation))

            case .ensureProcessed:
                // Wake-word/VAD fallback only restores the capture processor.
                // Its generation field is intentionally ignored and it must
                // not alter playback state, buffering, or generation leases.
                if voiceProcessingProtector?.ensureProcessed() == false {
                    failVoiceProcessingRestore(reason: .voiceActivity)
                }
            }
        }
    }

    private func schedule(_ frames: [ScheduledPlaybackFrame]) {
        guard !frames.isEmpty else { return }
        pendingFrames.append(contentsOf: frames)
        schedulePendingFrames()
    }

    private func schedulePendingFrames() {
        guard isConnected, !hasFailed else { return }
        let hasUnscheduledFrames = pendingFrames.contains {
            !scheduledOrdinals.contains($0.ordinal)
        }
        let playbackGeneration = pendingFrames.first {
            !scheduledOrdinals.contains($0.ordinal)
        }?.generation
        if hasUnscheduledFrames, let playbackGeneration,
           voiceProcessingProtector?.beginPlaybackProtection(
               generation: playbackGeneration
           ) == false {
            failVoiceProcessingRestore(reason: .playback)
            return
        }
        do {
            for frame in pendingFrames where !scheduledOrdinals.contains(frame.ordinal) {
                let buffer = try makeBuffer(from: frame.pcm)
                let callbackEpoch = scheduleEpoch
                scheduledOrdinals.insert(frame.ordinal)
                playerNode.scheduleBuffer(
                    buffer,
                    completionCallbackType: .dataRendered
                ) { [weak self] _ in
                    self?.executionQueue.async { [weak self] in
                        self?.didRender(frame, callbackEpoch: callbackEpoch)
                    }
                }
            }
            playIfReady()
        } catch {
            hasFailed = true
            onFatalError(error)
        }
    }

    private func suspendScheduledFrames() {
        mirrors?.clear()
        // stop() can invoke completion handlers for buffers that never reached
        // the output. Invalidate those callbacks before stopping and retain
        // the unconfirmed frames for the next engine connection.
        scheduleEpoch &+= 1
        inputRenderReadiness.disarm()
        isEngineStarted = false
        scheduledOrdinals.removeAll(keepingCapacity: true)
        renderedFrameOrderer.reset(keepingCapacity: true)
        playerNode.stop()
        if let generation = state.snapshot.generation {
            voiceProcessingProtector?.endPlaybackProtection(
                generation: generation
            )
        }
    }

    private func playIfReady() {
        let snapshot = state.snapshot
        guard !pendingFrames.isEmpty,
              isEngineStarted, engine?.isRunning == true, isConnected,
              !snapshot.isPaused, !snapshot.isStopped,
              !playerNode.isPlaying
        else {
            return
        }
        guard let generation = pendingFrames.first?.generation else { return }
        guard voiceProcessingProtector?.beginPlaybackProtection(
            generation: generation
        ) != false else {
            failVoiceProcessingRestore(reason: .playback)
            return
        }
        playerNode.play()
    }

    private func didRender(
        _ frame: ScheduledPlaybackFrame,
        callbackEpoch: UInt64
    ) {
        // AVAudioPlayerNode also invokes scheduled completions when its engine
        // stops. Those callbacks are not proof of playback, even when their
        // requested callback type was `.dataRendered`. Preserve the frame for
        // reconnection unless it completes while the engine is actually live.
        guard !hasFailed, isEngineStarted, engine?.isRunning == true,
              callbackEpoch == scheduleEpoch,
              scheduledOrdinals.remove(frame.ordinal) != nil
        else {
            return
        }
        // AVAudioPlayerNode does not guarantee that data-rendered completion
        // handlers arrive in scheduling order. A later partial-tail callback
        // can therefore precede a full frame. Commit acknowledgements only
        // after every earlier pending frame is also confirmed, preserving the
        // wire protocol's cumulative byte and PCM ordering contract.
        for renderedFrame in renderedFrameOrderer.confirm(
            frame,
            pendingFrames: &pendingFrames
        ) {
            guard let completion = state.complete(
                renderedFrame,
                gain: outputGain
            ) else {
                continue
            }
            mirrors?.append(completion.pcm)
            do {
                let packet = try FishMemoryPlaybackFrameProtocol.encodeRendered(
                    generation: completion.generation,
                    cumulativeValidRenderedByteCount:
                        completion.cumulativeValidRenderedByteCount,
                    uptimeNanoseconds: DispatchTime.now().uptimeNanoseconds,
                    validByteCount: completion.validByteCount,
                    pcm: completion.pcm
                )
                _ = writer.enqueue(
                    packet,
                    priority: .latestCumulative(key: "playback-render")
                )
            } catch {
                hasFailed = true
                onFatalError(error)
                return
            }
        }
        if pendingFrames.isEmpty {
            voiceProcessingProtector?.endPlaybackProtection(
                generation: frame.generation
            )
        }
    }

    private func failVoiceProcessingRestore(
        reason: VoiceProcessingTransitionReason
    ) {
        guard !hasFailed else { return }
        hasFailed = true
        onFatalError(
            AdaptiveVoiceProcessingError.failedToRestore(reason: reason)
        )
    }

    private func makeBuffer(from pcm: Data) throws -> AVAudioPCMBuffer {
        let sampleCount = pcm.count / MemoryLayout<Int16>.size
        guard sampleCount == FishMemoryAudioFrameProtocol.samplesPerFrame,
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: playbackFormat,
                  frameCapacity: AVAudioFrameCount(sampleCount)
              ),
              let channel = buffer.floatChannelData?[0]
        else {
            throw StreamingPlaybackError.bufferAllocationFailed
        }
        buffer.frameLength = AVAudioFrameCount(sampleCount)
        for index in 0 ..< sampleCount {
            let offset = index * 2
            let bits = UInt16(pcm[offset]) | UInt16(pcm[offset + 1]) << 8
            channel[index] = Float(Int16(bitPattern: bits)) / 32_768
        }
        return buffer
    }
}

/// Converts potentially out-of-order AVAudioPlayerNode completion callbacks
/// back into the order in which frames were scheduled.
final class RenderedFrameOrderer {
    private var confirmedOrdinals: Set<UInt64> = []

    func confirm(
        _ frame: ScheduledPlaybackFrame,
        pendingFrames: inout [ScheduledPlaybackFrame]
    ) -> [ScheduledPlaybackFrame] {
        confirmedOrdinals.insert(frame.ordinal)
        var ready: [ScheduledPlaybackFrame] = []
        while let first = pendingFrames.first,
              confirmedOrdinals.remove(first.ordinal) != nil {
            ready.append(pendingFrames.removeFirst())
        }
        return ready
    }

    func reset(keepingCapacity: Bool) {
        confirmedOrdinals.removeAll(keepingCapacity: keepingCapacity)
    }
}

/// A realtime-safe one-shot signal armed once for every audio-engine start.
///
/// This uses the back-deployable Darwin atomic primitives because Swift's
/// `Synchronization.Atomic` requires macOS 15 while the helper supports 12.3.
final class InputRenderReadinessGate: @unchecked Sendable {
    private var pending: Int32 = 0

    func arm() {
        OSAtomicOr32Barrier(1, &pending)
    }

    func claim() -> Bool {
        // This bit does not publish any other memory, so the non-barrier CAS is
        // sufficient and keeps the steady-state realtime check as small as
        // the macOS 12-compatible primitives allow.
        OSAtomicCompareAndSwap32(1, 0, &pending)
    }

    func disarm() {
        OSAtomicAnd32Barrier(0, &pending)
    }
}

enum StreamingPlaybackError: Error {
    case bufferAllocationFailed
}
