import Foundation

public enum SystemAudioWindowStatus: String, Codable, Equatable, Sendable {
    case complete
    case partial
    case unavailable
}

/// One native-owned ten-second media window, with optional playback audio.
/// The same clock bounds screen-only capture; Electron pairs it with frames.
public struct SystemAudioWindow: Codable, Equatable, Sendable {
    public static let sampleRate = 16_000
    public static let durationSeconds = 10
    public static let samplesPerWindow = sampleRate * durationSeconds
    public static let pcmBytesPerWindow = samplesPerWindow * MemoryLayout<Int16>.size

    public let captureSessionId: String
    public let windowIndex: UInt64
    public let startedAt: Double
    public let endedAt: Double
    public let audioStatus: SystemAudioWindowStatus
    public let audioPcmBase64: String?
    public let audioSampleRate: Int
    public let audioCapturedSamples: Int
    public let audioGapReason: String?
    public var audioSilence: Bool = false
}

/// Backend-private transport. PCM remains binary and in-process; it is never
/// encoded as a 250 ms protocol event.
public struct SystemAudioPCMBlock: Equatable, Sendable {
    public let pcm: Data
    public let sampleCount: Int

    public init(pcm: Data) {
        sampleCount = pcm.count / MemoryLayout<Int16>.size
        // Only exact digital silence is compressed. Quiet speech is unchanged.
        self.pcm = !pcm.isEmpty && pcm.count.isMultiple(of: 2)
            && pcm.withUnsafeBytes({ $0.allSatisfy { $0 == 0 } }) ? Data() : pcm
    }

    init(silentSamples: Int) {
        precondition(silentSamples > 0)
        pcm = Data()
        sampleCount = silentSamples
    }

    var isSilent: Bool { pcm.isEmpty && sampleCount > 0 }

    func slice(_ range: Range<Int>) -> SystemAudioPCMBlock {
        if isSilent { return .init(silentSamples: range.count) }
        return .init(pcm: pcm.subdata(in: range.lowerBound * 2 ..< range.upperBound * 2))
    }
}

public struct SystemAudioCaptureGap: Equatable, Sendable {
    public let reason: String

    public init(reason: String) {
        self.reason = reason
    }
}

/// A small thread-safe revocation gate around backend callbacks. Backends call
/// it only from their serial processing queue, and invalidation never waits for
/// conversion or protocol I/O.
final class SystemAudioCaptureDelivery: @unchecked Sendable {
    private let lock = NSLock()
    private var active = true
    private let callbacks: SystemAudioCallbacks

    init(callbacks: SystemAudioCallbacks) {
        self.callbacks = callbacks
    }

    func submit(_ block: SystemAudioPCMBlock) {
        lock.lock()
        let canDeliver = active
        lock.unlock()
        if canDeliver { callbacks.onSamples(block) }
    }

    func submit(_ gap: SystemAudioCaptureGap) {
        lock.lock()
        let canDeliver = active
        lock.unlock()
        if canDeliver { callbacks.onCaptureGap(gap) }
    }

    func invalidate() {
        lock.lock()
        active = false
        lock.unlock()
    }
}

/// A bounded sequential PCM buffer. Capture continuously appends roughly
/// one-second batches; the single native window timer drains ten seconds at a
/// time. Semantic snapshots tolerate a small boundary drift, so this avoids
/// timestamp buckets, coverage maps, and cross-layer window state.
struct SystemAudioWindowAssembler {
    let captureSessionId: String
    let startedAt: Double
    let startMonotonicMs: Double
    let periodMilliseconds: Int
    let samplesPerWindow: Int
    private(set) var nextWindowIndex: UInt64 = 0
    private var emittedSamples = 0
    private var blocks: [SystemAudioPCMBlock] = []
    private var bufferedSamples = 0
    private var gapReason: String?
    private var unavailableReason: String?

    init(
        captureSessionId: String = UUID().uuidString,
        startedAt: Double,
        startMonotonicMs: Double,
        periodMilliseconds: Int = SystemAudioWindow.durationSeconds * 1_000,
        initialUnavailableReason: String? = "system_audio_starting"
    ) {
        precondition(periodMilliseconds > 0)
        precondition((SystemAudioWindow.sampleRate * periodMilliseconds).isMultiple(of: 1_000))
        self.captureSessionId = captureSessionId
        self.startedAt = startedAt
        self.startMonotonicMs = startMonotonicMs
        self.periodMilliseconds = periodMilliseconds
        samplesPerWindow = SystemAudioWindow.sampleRate * periodMilliseconds / 1_000
        unavailableReason = initialUnavailableReason
    }

    var nextDeadlineMonotonicMs: Double {
        startMonotonicMs + Double(emittedSamples + samplesPerWindow) * 1_000
            / Double(SystemAudioWindow.sampleRate)
    }

    var nextWindowStartedAt: Double {
        startedAt + Double(emittedSamples) / Double(SystemAudioWindow.sampleRate)
    }

    mutating func markBackendRunning() {
        if unavailableReason != nil {
            markGap(reason: "capture_started_late")
        }
        unavailableReason = nil
    }

    mutating func markBackendUnavailable(reason: String) {
        guard !reason.isEmpty else { return }
        unavailableReason = reason
        markGap(reason: reason)
    }

    mutating func markGap(reason: String) {
        guard !reason.isEmpty else { return }
        if gapReason == nil { gapReason = reason }
    }

    @discardableResult
    mutating func append(_ block: SystemAudioPCMBlock) -> Bool {
        guard block.sampleCount > 0,
              block.pcm.count.isMultiple(of: MemoryLayout<Int16>.size),
              block.sampleCount <= SystemAudioWindow.sampleRate * 2 else { return false }
        let accepted = min(block.sampleCount, max(0, samplesPerWindow * 2 - bufferedSamples))
        if accepted > 0 {
            let next = accepted == block.sampleCount ? block : block.slice(0..<accepted)
            if next.isSilent, let last = blocks.last, last.isSilent {
                blocks[blocks.count - 1] = .init(silentSamples: last.sampleCount + next.sampleCount)
            } else {
                blocks.append(next)
            }
            bufferedSamples += accepted
            // Bound metadata even if a backend delivers pathological one-sample
            // alternating silent/non-silent blocks. PCM remains capped at 20 s.
            if blocks.count > 128 {
                var combined = Data()
                combined.reserveCapacity(bufferedSamples * 2)
                for part in blocks {
                    combined.append(part.isSilent ? Data(repeating: 0, count: part.sampleCount * 2) : part.pcm)
                }
                blocks = [.init(pcm: combined)]
            }
        }
        if accepted < block.sampleCount { markGap(reason: "capture_overflow") }
        return accepted == block.sampleCount
    }

    mutating func finishNextWindow(sampleCount: Int? = nil) -> SystemAudioWindow {
        let samplesPerWindow = sampleCount ?? self.samplesPerWindow
        precondition(samplesPerWindow > 0 && samplesPerWindow <= self.samplesPerWindow)
        let index = nextWindowIndex
        nextWindowIndex &+= 1
        let start = nextWindowStartedAt
        emittedSamples += samplesPerWindow
        let end = nextWindowStartedAt
        let targetBytes = samplesPerWindow * MemoryLayout<Int16>.size
        let capturedSamples = min(bufferedSamples, samplesPerWindow)
        let currentGapReason = gapReason
        gapReason = nil
        guard capturedSamples > 0 else {
            return SystemAudioWindow(
                captureSessionId: captureSessionId,
                windowIndex: index,
                startedAt: start,
                endedAt: end,
                audioStatus: .unavailable,
                audioPcmBase64: nil,
                audioSampleRate: SystemAudioWindow.sampleRate,
                audioCapturedSamples: 0,
                audioGapReason: currentGapReason ?? unavailableReason ?? "no_samples"
            )
        }

        var remaining = capturedSamples
        var consumed: [SystemAudioPCMBlock] = []
        while remaining > 0, let first = blocks.first {
            let count = min(remaining, first.sampleCount)
            consumed.append(count == first.sampleCount ? first : first.slice(0..<count))
            if count == first.sampleCount { blocks.removeFirst() }
            else { blocks[0] = first.slice(count..<first.sampleCount) }
            remaining -= count
        }
        bufferedSamples -= capturedSamples
        let complete = capturedSamples == samplesPerWindow
            && currentGapReason == nil
            && unavailableReason == nil
        let silent = complete && consumed.allSatisfy(\.isSilent)
        var windowPCM = Data()
        if !silent {
            windowPCM.reserveCapacity(targetBytes)
            for block in consumed {
                windowPCM.append(block.isSilent ? Data(repeating: 0, count: block.sampleCount * 2) : block.pcm)
            }
            windowPCM.append(Data(repeating: 0, count: targetBytes - windowPCM.count))
        }
        return SystemAudioWindow(
            captureSessionId: captureSessionId,
            windowIndex: index,
            startedAt: start,
            endedAt: end,
            audioStatus: complete ? .complete : .partial,
            audioPcmBase64: silent ? nil : windowPCM.base64EncodedString(),
            audioSampleRate: SystemAudioWindow.sampleRate,
            audioCapturedSamples: capturedSamples,
            audioGapReason: complete
                ? nil : currentGapReason ?? unavailableReason ?? "insufficient_samples",
            audioSilence: silent
        )
    }
}

/// Owns the sole audio-window clock for a video-perception session. Capture
/// restarts only change availability; start/stop are reserved for the user's
/// video-perception toggle (or helper lifetime).
final class SystemAudioWindowSegmenter: @unchecked Sendable {
    typealias Clock = @Sendable () -> Double
    typealias Handler = @Sendable (SystemAudioWindow) -> Void

    private let queue: DispatchQueue
    private let periodMilliseconds: Int
    private let graceMilliseconds: Int
    private let wallClock: Clock
    private let monotonicClock: Clock
    private let handler: Handler
    private var timer: DispatchSourceTimer?
    private var assembler: SystemAudioWindowAssembler?
    private var boundaries: [Double] = []

    init(
        periodMilliseconds: Int = SystemAudioWindow.durationSeconds * 1_000,
        graceMilliseconds: Int = 1_000,
        queue: DispatchQueue = DispatchQueue(
            label: "app.fishmemory.perception.audio.windows",
            qos: .userInitiated
        ),
        wallClock: @escaping Clock = { Date().timeIntervalSince1970 },
        monotonicClock: @escaping Clock = { ProcessInfo.processInfo.systemUptime * 1_000 },
        handler: @escaping Handler
    ) {
        precondition(periodMilliseconds > 0)
        precondition(graceMilliseconds >= 0 && graceMilliseconds <= 2_000)
        self.periodMilliseconds = periodMilliseconds
        self.graceMilliseconds = graceMilliseconds
        self.queue = queue
        self.wallClock = wallClock
        self.monotonicClock = monotonicClock
        self.handler = handler
    }

    var captureSessionId: String? {
        queue.sync { assembler?.captureSessionId }
    }

    func start(initialUnavailableReason: String) {
        queue.sync {
            guard assembler == nil else { return }
            let next = SystemAudioWindowAssembler(
                startedAt: wallClock(),
                startMonotonicMs: monotonicClock(),
                periodMilliseconds: periodMilliseconds,
                initialUnavailableReason: initialUnavailableReason
            )
            assembler = next
            let timer = DispatchSource.makeTimerSource(queue: queue)
            let sampleTolerance = 0.5 / Double(SystemAudioWindow.sampleRate)
            let bucketEnd = (floor((next.startedAt + sampleTolerance) / 600) + 1) * 600
            let firstDelay = min(
                periodMilliseconds, max(1, Int(ceil((bucketEnd - next.startedAt) * 1_000)))
            )
            timer.schedule(
                deadline: .now() + .milliseconds(firstDelay + graceMilliseconds),
                repeating: .milliseconds(periodMilliseconds),
                leeway: .milliseconds(min(100, max(1, graceMilliseconds / 4)))
            )
            timer.setEventHandler { [weak self] in self?.emitDueWindows() }
            self.timer = timer
            timer.resume()
        }
    }

    func append(_ block: SystemAudioPCMBlock) {
        queue.async { [weak self] in _ = self?.assembler?.append(block) }
    }

    func markGap(_ gap: SystemAudioCaptureGap) {
        queue.async { [weak self] in
            self?.assembler?.markGap(reason: gap.reason)
        }
    }

    func markBackendRunning() {
        queue.async { [weak self] in self?.assembler?.markBackendRunning() }
    }

    func markBackendUnavailable(reason: String) {
        queue.async { [weak self] in
            self?.assembler?.markBackendUnavailable(reason: reason)
        }
    }

    func cut(at timestamp: Double) {
        guard timestamp.isFinite else { return }
        queue.async { [weak self] in
            guard let self, let assembler = self.assembler,
                  timestamp > assembler.nextWindowStartedAt,
                  timestamp <= self.wallClock() + 1 else { return }
            if !self.boundaries.contains(timestamp) {
                self.boundaries.append(timestamp)
                self.boundaries.sort()
            }
            self.emitDueWindows()
        }
    }

    func stop() {
        queue.sync {
            timer?.setEventHandler {}
            timer?.cancel()
            timer = nil
            assembler = nil
            boundaries = []
        }
    }

    private func emitDueWindows() {
        guard var assembler else { return }
        let now = monotonicClock()
        while true {
            boundaries.removeAll { $0 <= assembler.nextWindowStartedAt }
            // A boundary is rounded to the nearest PCM sample. Treat a residual
            // below half a sample as already reached, avoiding a zero-length loop.
            let sampleTolerance = 0.5 / Double(SystemAudioWindow.sampleRate)
            let bucketEnd = (
                floor((assembler.nextWindowStartedAt + sampleTolerance) / 600) + 1
            ) * 600
            let fullEnd = min(
                assembler.nextWindowStartedAt + Double(periodMilliseconds) / 1_000,
                bucketEnd
            )
            let end = min(boundaries.first ?? fullEnd, fullEnd)
            let deadline = assembler.startMonotonicMs + (end - assembler.startedAt) * 1_000
            guard now >= deadline + Double(graceMilliseconds) else {
                let delay = max(1, Int(ceil(deadline + Double(graceMilliseconds) - now)))
                timer?.schedule(
                    deadline: .now() + .milliseconds(delay),
                    repeating: .milliseconds(periodMilliseconds)
                )
                break
            }
            let samples = Int((
                (end - assembler.nextWindowStartedAt) * Double(SystemAudioWindow.sampleRate)
            ).rounded())
            if samples <= 0 {
                boundaries.removeAll { $0 <= end }
                continue
            }
            handler(assembler.finishNextWindow(
                sampleCount: min(samples, assembler.samplesPerWindow)
            ))
        }
        self.assembler = assembler
    }
}
