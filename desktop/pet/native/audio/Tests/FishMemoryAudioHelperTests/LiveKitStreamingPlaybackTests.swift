@preconcurrency import AVFoundation
import AudioToolbox
import FishMemoryAudioCore
@testable import FishMemoryLiveKitAudioHelper
import Foundation
import XCTest

final class LiveKitStreamingPlaybackTests: XCTestCase {
    func testSpeakerMutePreservesEngineAndDoesNotFollowMicrophoneMute() throws {
        try requireOfflineAudioUnit()
        let engine = AVAudioEngine()
        let writer = BoundedFrameWriter(capacity: 4) { _ in }
        let protector = PlaybackProtectorSpy()
        let playback = LiveKitStreamingPlayback(
            writer: writer, voiceProcessingProtector: protector,
            onFatalError: { _ in }
        )
        playback.engineDidCreate(engine)
        defer { playback.stop() }
        let player = try XCTUnwrap(engine.attachedNodes.compactMap { $0 as? AVAudioPlayerNode }.first)
        playback.handle(.control(operation: .reset, generation: 40))
        playback.handle(.control(operation: .muteSpeaker, generation: 0))
        XCTAssertEqual(player.volume, 0)
        playback.handle(.control(operation: .reset, generation: 41))
        XCTAssertEqual(player.volume, 0, "A new reply must respect the saved speaker state")
        playback.handle(.control(operation: .unmuteSpeaker, generation: 0))
        XCTAssertEqual(player.volume, 1)
        playback.handle(.control(operation: .muteMicrophone, generation: 0))
        XCTAssertEqual(player.volume, 1)
        XCTAssertTrue(engine.attachedNodes.contains(player))
    }
    func testEnsureProcessedControlDoesNotChangePlaybackGeneration() throws {
        try requireOfflineAudioUnit()
        let writer = BoundedFrameWriter(capacity: 1) { _ in }
        let protector = PlaybackProtectorSpy()
        let playback = LiveKitStreamingPlayback(
            writer: writer,
            voiceProcessingProtector: protector,
            onFatalError: { _ in }
        )

        playback.handle(.control(
            operation: .ensureProcessed,
            generation: 9_999
        ))
        playback.handle(.control(
            operation: .ensureProcessed,
            generation: 1
        ))

        XCTAssertEqual(protector.events, ["ensure", "ensure"])
    }

    func testFailedEnsureProcessedReportsOneExactFatalReason() throws {
        try requireOfflineAudioUnit()
        let writer = BoundedFrameWriter(capacity: 1) { _ in }
        let protector = PlaybackProtectorSpy(ensureResult: false)
        let failures = PlaybackFailureRecorder()
        let playback = LiveKitStreamingPlayback(
            writer: writer,
            voiceProcessingProtector: protector,
            onFatalError: failures.append
        )

        playback.handle(.control(
            operation: .ensureProcessed,
            generation: 1
        ))
        playback.handle(.control(
            operation: .ensureProcessed,
            generation: 2
        ))

        XCTAssertEqual(
            failures.values,
            [.failedToRestore(reason: .voiceActivity)]
        )
        XCTAssertEqual(protector.events, ["ensure"])
    }

    func testPlaybackProtectionTracksCurrentGenerationThroughActualDrain() throws {
        try requireOfflineAudioUnit()
        let engine = AVAudioEngine()
        let format = try XCTUnwrap(AVAudioFormat(
            standardFormatWithSampleRate: 16_000,
            channels: 1
        ))
        try engine.enableManualRenderingMode(
            .offline,
            format: format,
            maximumFrameCount: 160
        )
        let result = PlaybackTestResult()
        let writer = BoundedFrameWriter(capacity: 64) { packet in
            result.append(try FishMemoryPlaybackFrameProtocol.decodeRendered(packet))
        }
        writer.start()
        let protector = PlaybackProtectorSpy()
        let playback = LiveKitStreamingPlayback(
            writer: writer,
            voiceProcessingProtector: protector,
            onFatalError: { result.fail($0) }
        )
        playback.engineDidCreate(engine)
        playback.engineWillConnectOutput(engine)
        playback.engineWillStart()
        try engine.start()
        defer {
            playback.stop()
            engine.stop()
            writer.stop(discardPending: false)
            _ = writer.waitUntilStopped(timeout: .now() + 2)
        }

        playback.handle(.control(operation: .reset, generation: 90))
        playback.handle(.control(operation: .stop, generation: 89))
        playback.handle(.control(operation: .flush, generation: 89))
        XCTAssertEqual(protector.events, ["prewarm:90"])

        playback.handle(.pcm(
            generation: 90,
            pcm: encodeSamples(Array(repeating: 5_000, count: 200))
        ))
        playback.handle(.control(operation: .flush, generation: 90))
        XCTAssertEqual(protector.events, ["prewarm:90", "begin:90"])

        let renderBuffer = try XCTUnwrap(AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: 160
        ))
        for _ in 0 ..< 12 {
            _ = try engine.renderOffline(160, to: renderBuffer)
        }
        XCTAssertTrue(result.waitForCumulativeBytes(400, timeout: .now() + 2))
        playback.handle(.control(operation: .duck, generation: 90))

        XCTAssertEqual(
            protector.events,
            ["prewarm:90", "begin:90", "end:90"]
        )
        XCTAssertNil(result.errorDescription)
    }

    func testOfflineSameEngineOutputRendersAndAcknowledgesPartialTail() throws {
        try requireOfflineAudioUnit()
        let engine = AVAudioEngine()
        let format = try XCTUnwrap(AVAudioFormat(
            standardFormatWithSampleRate: 16_000,
            channels: 1
        ))
        try engine.enableManualRenderingMode(
            .offline,
            format: format,
            maximumFrameCount: 160
        )
        let result = PlaybackTestResult()
        let writer = BoundedFrameWriter(capacity: 64) { packet in
            result.append(try FishMemoryPlaybackFrameProtocol.decodeRendered(packet))
        }
        writer.start()
        let mirrors = RenderedAudioMirrorSpy()
        let playback = LiveKitStreamingPlayback(
            writer: writer,
            mirrors: mirrors,
            onFatalError: { result.fail($0) }
        )
        engine.outputNode.auAudioUnit.maximumFramesToRender = 8_192
        playback.engineDidCreate(engine)
        let playerNode = try XCTUnwrap(engine.attachedNodes.compactMap {
            $0 as? AVAudioPlayerNode
        }.first)
        XCTAssertEqual(
            playerNode.auAudioUnit.maximumFramesToRender,
            engine.outputNode.auAudioUnit.maximumFramesToRender
        )
        XCTAssertEqual(playerNode.auAudioUnit.maximumFramesToRender, 8_192)
        playback.engineWillConnectOutput(engine)
        playback.engineWillStart()
        try engine.start()
        defer {
            playback.stop()
            engine.stop()
            writer.stop(discardPending: false)
            _ = writer.waitUntilStopped(timeout: .now() + 2)
        }

        playback.handle(.control(operation: .reset, generation: 31))
        let pcm = encodeSamples(Array(repeating: 8_000, count: 200))
        playback.handle(.pcm(generation: 31, pcm: pcm))
        playback.handle(.control(operation: .flush, generation: 31))
        XCTAssertTrue(mirrors.pcm.isEmpty, "Queued audio must not leak into the live output")
        XCTAssertTrue(
            playerNode.isPlaying,
            "scheduling a packet after engine start must play without another input-ready callback"
        )

        let renderBuffer = try XCTUnwrap(AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: 160
        ))
        var heardSamples: [Float] = []
        for _ in 0 ..< 12 {
            let status = try engine.renderOffline(160, to: renderBuffer)
            if status == .success, let channel = renderBuffer.floatChannelData?[0] {
                heardSamples.append(contentsOf: UnsafeBufferPointer(
                    start: channel,
                    count: Int(renderBuffer.frameLength)
                ))
            }
        }
        XCTAssertTrue(result.waitForCumulativeBytes(400, timeout: .now() + 2))
        XCTAssertNil(result.errorDescription)
        XCTAssertGreaterThan(heardSamples.map(abs).max() ?? 0, 0.2)
        let final = try XCTUnwrap(result.frames.last)
        XCTAssertEqual(final.generation, 31)
        XCTAssertEqual(final.cumulativeValidRenderedByteCount, 400)
        XCTAssertEqual(final.validByteCount, 80)
        XCTAssertEqual(final.pcm.suffix(240), Data(repeating: 0, count: 240))
        XCTAssertEqual(mirrors.pcm, pcm + Data(repeating: 0, count: 240))
        playback.handle(.control(operation: .pause, generation: 31))
        XCTAssertTrue(mirrors.pcm.isEmpty, "Pausing must discard auxiliary output tail")
    }

    func testOfflinePauseResumeDuckingAndEngineReconnect() throws {
        try requireOfflineAudioUnit()
        let engine = AVAudioEngine()
        let format = try XCTUnwrap(AVAudioFormat(
            standardFormatWithSampleRate: 16_000,
            channels: 1
        ))
        try engine.enableManualRenderingMode(
            .offline,
            format: format,
            maximumFrameCount: 160
        )
        let result = PlaybackTestResult()
        let writer = BoundedFrameWriter(capacity: 64) { packet in
            result.append(try FishMemoryPlaybackFrameProtocol.decodeRendered(packet))
        }
        writer.start()
        let playback = LiveKitStreamingPlayback(
            writer: writer,
            onFatalError: { result.fail($0) }
        )
        playback.engineDidCreate(engine)
        playback.engineWillConnectOutput(engine)
        let reconnectPlayerNode = try XCTUnwrap(engine.attachedNodes.compactMap {
            $0 as? AVAudioPlayerNode
        }.first)
        playback.engineWillStart()
        try engine.start()
        defer {
            playback.stop()
            engine.stop()
            writer.stop(discardPending: false)
            _ = writer.waitUntilStopped(timeout: .now() + 2)
        }
        let renderBuffer = try XCTUnwrap(AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: 160
        ))
        let pcm = encodeSamples(Array(repeating: 8_000, count: 160))

        playback.handle(.control(operation: .reset, generation: 44))
        playback.handle(.control(operation: .pause, generation: 44))
        playback.handle(.pcm(generation: 44, pcm: pcm))
        for _ in 0 ..< 4 {
            _ = try engine.renderOffline(160, to: renderBuffer)
        }
        XCTAssertTrue(result.frames.isEmpty)

        playback.handle(.control(operation: .resume, generation: 44))
        playback.handle(.control(operation: .duck, generation: 44))
        for _ in 0 ..< 12 {
            _ = try engine.renderOffline(160, to: renderBuffer)
        }
        XCTAssertTrue(result.waitForCumulativeBytes(320, timeout: .now() + 2))
        XCTAssertEqual(result.frames.last?.pcm, encodeSamples(
            Array(repeating: 1_600, count: 160)
        ))

        for cycle in 1 ... 20 {
            playback.handle(.pcm(generation: 44, pcm: pcm))
            engine.stop()
            playback.engineDidStop()
            playback.engineDidDisable()
            playback.engineWillConnectOutput(engine)
            playback.engineWillStart()
            XCTAssertFalse(reconnectPlayerNode.isPlaying)
            try engine.start()
            playback.engineDidRenderInput()
            // A no-op control on the same serial queue is a deterministic
            // barrier for the asynchronous input-tap signal; no timed sleep.
            playback.handle(.control(operation: .duck, generation: 44))
            XCTAssertTrue(reconnectPlayerNode.isPlaying)
            var statuses: [Int] = []
            var outputPeak: Float = 0
            for _ in 0 ..< 12 {
                let status = try engine.renderOffline(160, to: renderBuffer)
                statuses.append(status.rawValue)
                if status == .success, let channel = renderBuffer.floatChannelData?[0] {
                    for index in 0 ..< Int(renderBuffer.frameLength) {
                        outputPeak = max(outputPeak, abs(channel[index]))
                    }
                }
            }
            let expectedBytes = UInt64(cycle + 1) * 320
            guard result.waitForCumulativeBytes(expectedBytes, timeout: .now() + 2) else {
                XCTFail(
                    "cycle=\(cycle) expected=\(expectedBytes) " +
                        "actual=\(result.frames.last?.cumulativeValidRenderedByteCount ?? 0) " +
                        "statuses=\(statuses) peak=\(outputPeak) " +
                        "engineSampleTime=\(engine.manualRenderingSampleTime) " +
                        "playing=\(reconnectPlayerNode.isPlaying)"
                )
                return
            }
            XCTAssertGreaterThan(outputPeak, 0)
            XCTAssertEqual(result.frames.last?.cumulativeValidRenderedByteCount, expectedBytes)
        }
        XCTAssertNil(result.errorDescription)
    }

    func testInputRenderReadinessIsClaimedOncePerEngineStart() {
        let gate = InputRenderReadinessGate()

        XCTAssertFalse(gate.claim())
        gate.arm()

        let claimCount = LockedCounter()
        DispatchQueue.concurrentPerform(iterations: 1_000) { _ in
            if gate.claim() {
                claimCount.increment()
            }
        }
        XCTAssertEqual(claimCount.value, 1)

        gate.arm()
        XCTAssertTrue(gate.claim())
        XCTAssertFalse(gate.claim())

        gate.arm()
        gate.disarm()
        XCTAssertFalse(gate.claim())
    }

    func testPlayerNodeDetachesOnReleaseAndExplicitStop() throws {
        try requireOfflineAudioUnit()
        let engine = AVAudioEngine()
        let writer = BoundedFrameWriter(capacity: 1) { _ in }
        let playback = LiveKitStreamingPlayback(
            writer: writer,
            onFatalError: { _ in }
        )

        playback.engineDidCreate(engine)
        let playerNode = try XCTUnwrap(engine.attachedNodes.compactMap {
            $0 as? AVAudioPlayerNode
        }.first)
        playback.engineWillRelease(engine)
        XCTAssertFalse(engine.attachedNodes.contains(playerNode))

        playback.engineDidCreate(engine)
        XCTAssertTrue(engine.attachedNodes.contains(playerNode))
        playback.stop()
        XCTAssertFalse(engine.attachedNodes.contains(playerNode))
    }

    func testRenderedFrameOrdererWaitsForEarlierCallbacks() {
        let first = ScheduledPlaybackFrame(
            generation: 7,
            resetSerial: 3,
            ordinal: 0,
            validByteCount: 320,
            pcm: Data(repeating: 1, count: 320)
        )
        let tail = ScheduledPlaybackFrame(
            generation: 7,
            resetSerial: 3,
            ordinal: 1,
            validByteCount: 80,
            pcm: Data(repeating: 2, count: 320)
        )
        var pending = [first, tail]
        let orderer = RenderedFrameOrderer()

        XCTAssertEqual(
            orderer.confirm(tail, pendingFrames: &pending),
            []
        )
        XCTAssertEqual(pending, [first, tail])
        XCTAssertEqual(
            orderer.confirm(first, pendingFrames: &pending),
            [first, tail]
        )
        XCTAssertTrue(pending.isEmpty)
    }

    private func requireOfflineAudioUnit() throws {
        var description = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_GenericOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )
        guard AudioComponentFindNext(nil, &description) != nil else {
            throw XCTSkip("Offline Apple AudioUnit unavailable in this sandbox")
        }
    }

    private func encodeSamples(_ samples: [Int16]) -> Data {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            var value = sample.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        return data
    }
}

private final class RenderedAudioMirrorSpy: RenderedAudioMirroring, @unchecked Sendable {
    private let lock = NSLock()
    private var value = Data()

    var pcm: Data {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func append(_ pcm: Data) {
        lock.lock()
        value.append(pcm)
        lock.unlock()
    }

    func clear() {
        lock.lock()
        value.removeAll()
        lock.unlock()
    }
}

private final class PlaybackProtectorSpy:
    VoiceProcessingPlaybackProtecting, @unchecked Sendable
{
    private let lock = NSLock()
    private let ensureResult: Bool
    private var hold: UInt64?
    private var recordedEvents: [String] = []

    init(ensureResult: Bool = true) {
        self.ensureResult = ensureResult
    }

    var events: [String] {
        lock.withLock { recordedEvents }
    }

    func ensureProcessed() -> Bool {
        lock.withLock { recordedEvents.append("ensure") }
        return ensureResult
    }

    func prewarmPlayback(generation: UInt64) -> Bool {
        lock.withLock {
            hold = generation
            recordedEvents.append("prewarm:\(generation)")
        }
        return true
    }

    func beginPlaybackProtection(generation: UInt64) -> Bool {
        lock.withLock {
            if hold != generation ||
                recordedEvents.last != "begin:\(generation)" {
                recordedEvents.append("begin:\(generation)")
            }
            hold = generation
        }
        return true
    }

    func endPlaybackProtection(generation: UInt64) {
        lock.withLock {
            guard hold == generation else { return }
            hold = nil
            recordedEvents.append("end:\(generation)")
        }
    }
}

private final class PlaybackFailureRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [AdaptiveVoiceProcessingError] = []

    var values: [AdaptiveVoiceProcessingError] {
        lock.withLock { recorded }
    }

    func append(_ error: Error) {
        guard let error = error as? AdaptiveVoiceProcessingError else { return }
        lock.withLock { recorded.append(error) }
    }
}

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var storedValue = 0

    func increment() {
        lock.lock()
        storedValue += 1
        lock.unlock()
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return storedValue
    }
}

private final class PlaybackTestResult: @unchecked Sendable {
    private let condition = NSCondition()
    private var storedFrames: [PlaybackRenderedFrame] = []
    private var storedErrorDescription: String?

    func append(_ frame: PlaybackRenderedFrame) {
        condition.lock()
        storedFrames.append(frame)
        condition.broadcast()
        condition.unlock()
    }

    func fail(_ error: Error) {
        condition.lock()
        storedErrorDescription = String(describing: error)
        condition.broadcast()
        condition.unlock()
    }

    var frames: [PlaybackRenderedFrame] {
        condition.lock()
        defer { condition.unlock() }
        return storedFrames
    }

    var errorDescription: String? {
        condition.lock()
        defer { condition.unlock() }
        return storedErrorDescription
    }

    func waitForCumulativeBytes(_ byteCount: UInt64, timeout: DispatchTime) -> Bool {
        let now = DispatchTime.now().uptimeNanoseconds
        let remaining = timeout.uptimeNanoseconds > now
            ? timeout.uptimeNanoseconds - now : 0
        let deadline = Date().addingTimeInterval(Double(remaining) / 1_000_000_000)
        condition.lock()
        defer { condition.unlock() }
        while (storedFrames.last?.cumulativeValidRenderedByteCount ?? 0) < byteCount {
            if storedErrorDescription != nil { return false }
            if !condition.wait(until: deadline) { return false }
        }
        return true
    }
}
