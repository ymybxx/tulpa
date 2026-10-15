import CoreGraphics
import CoreImage
import CoreVideo
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct EnvironmentFrameMetadata: Codable, Equatable, Sendable {
    public let frameId: String
    public let revision: UInt64
    public let observedAt: Double
    public let expiresAt: Double
    public let displayID: CGDirectDisplayID
    public let pixelWidth: Int
    public let pixelHeight: Int
    public let byteLength: Int
    public var contentHash: String? = nil
}

public struct EnvironmentFramePayload: Codable, Equatable, Sendable {
    public let frameId: String
    public let revision: UInt64
    public let observedAt: Double
    public let expiresAt: Double
    public let displayID: CGDirectDisplayID
    public let pixelWidth: Int
    public let pixelHeight: Int
    public let byteLength: Int
    public let mimeType: String
    public let jpegBase64: String
    public let contentHash: String?

    init(metadata: EnvironmentFrameMetadata, jpeg: Data) {
        frameId = metadata.frameId
        revision = metadata.revision
        observedAt = metadata.observedAt
        expiresAt = metadata.expiresAt
        displayID = metadata.displayID
        pixelWidth = metadata.pixelWidth
        pixelHeight = metadata.pixelHeight
        byteLength = metadata.byteLength
        mimeType = "image/jpeg"
        jpegBase64 = jpeg.base64EncodedString()
        contentHash = metadata.contentHash
    }
}

public struct EnvironmentFrameUnavailable: Codable, Equatable, Sendable {
    public let code: String

    public init() {
        code = "unavailable"
    }
}

/// One-shot, memory-only slot for the most recent display frame.
public final class EnvironmentFrameBroker: @unchecked Sendable {
    public typealias Clock = @Sendable () -> Double

    public static let maximumJPEGBytes = 200 * 1_024
    public static let timeToLive: TimeInterval = 3

    private struct StoredFrame {
        let metadata: EnvironmentFrameMetadata
        let jpeg: Data
        let generation: UInt64
    }

    private let clock: Clock
    private let expirationQueue = DispatchQueue(
        label: "app.fishmemory.perception.environment-frame.expiration",
        qos: .utility
    )
    private let lock = NSLock()
    private var stored: StoredFrame?
    private var expirationWorkItem: DispatchWorkItem?
    private var revision: UInt64 = 0

    public init(clock: @escaping Clock = { Date().timeIntervalSince1970 }) {
        self.clock = clock
    }

    func store(
        jpeg: Data,
        observedAt: Double,
        displayID: CGDirectDisplayID,
        pixelWidth: Int,
        pixelHeight: Int,
        generation: UInt64,
        contentHash: String? = nil
    ) -> EnvironmentFrameMetadata? {
        guard !jpeg.isEmpty,
              jpeg.count <= Self.maximumJPEGBytes,
              observedAt.isFinite,
              pixelWidth > 0,
              pixelHeight > 0
        else { return nil }
        let now = clock()
        let expiresAt = observedAt + Self.timeToLive
        guard now.isFinite, now < expiresAt else { return nil }

        lock.lock()
        revision &+= 1
        let metadata = EnvironmentFrameMetadata(
            frameId: UUID().uuidString,
            revision: revision,
            observedAt: observedAt,
            expiresAt: expiresAt,
            displayID: displayID,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight,
            byteLength: jpeg.count,
            contentHash: contentHash
        )
        let workItem = DispatchWorkItem { [weak self] in
            self?.expire(frameId: metadata.frameId)
        }
        expirationWorkItem?.cancel()
        stored = StoredFrame(metadata: metadata, jpeg: jpeg, generation: generation)
        expirationWorkItem = workItem
        lock.unlock()
        expirationQueue.asyncAfter(
            deadline: .now() + max(0, expiresAt - now),
            execute: workItem
        )
        return metadata
    }

    public func claim(frameId: String, generation: UInt64? = nil) -> EnvironmentFramePayload? {
        let now = clock()
        guard now.isFinite else {
            clear()
            return nil
        }
        lock.lock()
        defer { lock.unlock() }
        guard let stored,
              stored.metadata.frameId == frameId,
              generation == nil || stored.generation == generation,
              now < stored.metadata.expiresAt
        else {
            if stored?.metadata.frameId == frameId { clearLocked() }
            return nil
        }
        clearLocked()
        return EnvironmentFramePayload(metadata: stored.metadata, jpeg: stored.jpeg)
    }

    public func clear() {
        lock.lock()
        clearLocked()
        lock.unlock()
    }

    func currentMetadata() -> EnvironmentFrameMetadata? {
        let now = clock()
        lock.lock()
        defer { lock.unlock() }
        guard let stored, now.isFinite, now < stored.metadata.expiresAt else {
            clearLocked()
            return nil
        }
        return stored.metadata
    }

    private func expire(frameId: String) {
        lock.lock()
        if stored?.metadata.frameId == frameId { clearLocked() }
        lock.unlock()
    }

    private func clearLocked() {
        expirationWorkItem?.cancel()
        expirationWorkItem = nil
        stored = nil
    }
}

struct EnvironmentFrameEncoderConfiguration: Equatable, Sendable {
    let maximumLongEdge: Int
    let maximumJPEGBytes: Int
    let compressionQualities: [Double]
    let minimumFrameInterval: TimeInterval

    init(
        maximumLongEdge: Int = 960,
        maximumJPEGBytes: Int = EnvironmentFrameBroker.maximumJPEGBytes,
        compressionQualities: [Double] = [0.72, 0.58, 0.44, 0.32, 0.22],
        minimumFrameInterval: TimeInterval = 1
    ) {
        precondition(maximumLongEdge > 0)
        precondition(maximumJPEGBytes > 0)
        precondition(!compressionQualities.isEmpty)
        precondition(compressionQualities.allSatisfy { $0 > 0 && $0 <= 1 })
        precondition(minimumFrameInterval >= 0 && minimumFrameInterval.isFinite)
        self.maximumLongEdge = maximumLongEdge
        self.maximumJPEGBytes = maximumJPEGBytes
        self.compressionQualities = compressionQualities
        self.minimumFrameInterval = minimumFrameInterval
    }
}

struct EncodedEnvironmentFrame: Equatable, Sendable {
    let jpeg: Data
    let pixelWidth: Int
    let pixelHeight: Int
}

private struct PendingEnvironmentFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let generation: UInt64
    let observedAt: Double
    let displayID: CGDirectDisplayID
}

final class EnvironmentFrameEncoder: @unchecked Sendable {
    typealias EncodeHandler = @Sendable (CVPixelBuffer) -> EncodedEnvironmentFrame?
    typealias AvailableHandler = @Sendable (EnvironmentFrameMetadata) -> Void

    private let configuration: EnvironmentFrameEncoderConfiguration
    private let broker: EnvironmentFrameBroker
    private let encodeHandler: EncodeHandler
    private let onAvailable: AvailableHandler
    private let encodingQueue = DispatchQueue(
        label: "app.fishmemory.perception.environment-frame.encoding",
        qos: .utility
    )
    private let frameGate = LatestValueGate<PendingEnvironmentFrame>()
    private let stateLock = NSLock()
    private var activeGeneration: UInt64?
    private var lastAcceptedObservedAt: Double?
    // Accessed on encodingQueue only; one bounded cached image per lifecycle.
    private var cachedEncoding: (key: String, encoded: EncodedEnvironmentFrame, hash: String)?

    init(
        configuration: EnvironmentFrameEncoderConfiguration = .init(),
        broker: EnvironmentFrameBroker,
        encode: EncodeHandler? = nil,
        onAvailable: @escaping AvailableHandler
    ) {
        self.configuration = configuration
        self.broker = broker
        self.onAvailable = onAvailable
        if let encode {
            encodeHandler = encode
        } else {
            let imageEncoder = EnvironmentFrameImageEncoder(configuration: configuration)
            encodeHandler = { pixelBuffer in
                imageEncoder.encode(pixelBuffer)
            }
        }
    }

    func activate(generation: UInt64) {
        stateLock.lock()
        activeGeneration = generation
        lastAcceptedObservedAt = nil
        stateLock.unlock()
        frameGate.reset()
        broker.clear()
        encodingQueue.async { [weak self] in self?.cachedEncoding = nil }
    }

    func invalidate(generation: UInt64? = nil) {
        stateLock.lock()
        if let generation, activeGeneration != generation {
            stateLock.unlock()
            return
        }
        activeGeneration = nil
        lastAcceptedObservedAt = nil
        stateLock.unlock()
        frameGate.reset()
        broker.clear()
        encodingQueue.async { [weak self] in self?.cachedEncoding = nil }
    }

    func submit(
        pixelBuffer: CVPixelBuffer,
        generation: UInt64,
        observedAt: Double,
        displayID: CGDirectDisplayID
    ) {
        guard beginEncoding(generation: generation, observedAt: observedAt) else { return }
        let frame = PendingEnvironmentFrame(
            pixelBuffer: pixelBuffer,
            generation: generation,
            observedAt: observedAt,
            displayID: displayID
        )
        guard let first = frameGate.submit(frame) else { return }
        encodingQueue.async { [weak self] in self?.drain(startingWith: first) }
    }

    /// Periodic screenshots await encoding too, so a slow encoder cannot build
    /// a second screenshot queue. The authorization is rechecked after encoding.
    func submitAndWait(
        _ frame: CapturedEnvironmentFrame,
        generation: UInt64,
        isStillCurrent: @escaping @Sendable () -> Bool
    ) async {
        guard isStillCurrent(), isCurrent(generation: generation) else { return }
        await withCheckedContinuation { continuation in
            encodingQueue.async { [self] in
                autoreleasepool {
                    guard isStillCurrent(), isCurrent(generation: generation),
                          let encoded = encodeOrReuse(frame.pixelBuffer, generation: generation, displayID: frame.displayID),
                          isStillCurrent(),
                          let metadata = storeIfCurrent(encoded, frame: .init(
                            pixelBuffer: frame.pixelBuffer,
                            generation: generation,
                            observedAt: frame.observedAt,
                            displayID: frame.displayID
                          ))
                    else { return }
                    if isStillCurrent() {
                        onAvailable(metadata)
                    } else {
                        broker.clear()
                    }
                }
                continuation.resume()
            }
        }
    }

    private func drain(startingWith first: PendingEnvironmentFrame) {
        var current: PendingEnvironmentFrame? = first
        while let frame = current {
            autoreleasepool { encode(frame) }
            current = frameGate.takeNextOrFinish()
        }
    }

    private func encode(_ frame: PendingEnvironmentFrame) {
        guard isCurrent(generation: frame.generation),
              let encoded = encodeOrReuse(frame.pixelBuffer, generation: frame.generation, displayID: frame.displayID),
              !encoded.jpeg.isEmpty,
              encoded.jpeg.count <= configuration.maximumJPEGBytes,
              let metadata = storeIfCurrent(encoded, frame: frame)
        else { return }
        onAvailable(metadata)
    }

    private func storeIfCurrent(
        _ encoded: EncodedEnvironmentFrame,
        frame: PendingEnvironmentFrame
    ) -> EnvironmentFrameMetadata? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard activeGeneration == frame.generation else { return nil }
        return broker.store(
            jpeg: encoded.jpeg,
            observedAt: frame.observedAt,
            displayID: frame.displayID,
            pixelWidth: encoded.pixelWidth,
            pixelHeight: encoded.pixelHeight,
            generation: frame.generation,
            contentHash: cachedEncoding?.hash
        )
    }

    private func encodeOrReuse(
        _ pixels: CVPixelBuffer, generation: UInt64, displayID: CGDirectDisplayID
    ) -> EncodedEnvironmentFrame? {
        let digest = environmentFramePixelDigest(pixels)
        let key = digest.map { "\(generation):\(displayID):\($0)" }
        if let key, cachedEncoding?.key == key { return cachedEncoding?.encoded }
        cachedEncoding = nil
        guard let encoded = encodeHandler(pixels) else { return nil }
        if let key {
            cachedEncoding = (key, encoded, SHA256.hash(data: encoded.jpeg).map { String(format: "%02x", $0) }.joined())
        }
        return encoded
    }

    private func beginEncoding(generation: UInt64, observedAt: Double) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard activeGeneration == generation, observedAt.isFinite else { return false }
        if let lastAcceptedObservedAt,
           observedAt >= lastAcceptedObservedAt,
           observedAt - lastAcceptedObservedAt < configuration.minimumFrameInterval
        {
            return false
        }
        lastAcceptedObservedAt = observedAt
        return true
    }

    private func isCurrent(generation: UInt64) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return activeGeneration == generation
    }

    static func encodeJPEG(
        pixelBuffer: CVPixelBuffer,
        configuration: EnvironmentFrameEncoderConfiguration,
        context: CIContext = CIContext(options: [.cacheIntermediates: false])
    ) -> EncodedEnvironmentFrame? {
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        let sourceWidth = image.extent.width
        let sourceHeight = image.extent.height
        guard sourceWidth > 0, sourceHeight > 0 else { return nil }
        let longEdge = max(sourceWidth, sourceHeight)
        if longEdge > CGFloat(configuration.maximumLongEdge) {
            let scale = CGFloat(configuration.maximumLongEdge) / longEdge
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        guard let cgImage = context.createCGImage(image, from: image.extent) else { return nil }
        let outputWidth = cgImage.width
        let outputHeight = cgImage.height
        guard outputWidth > 0, outputHeight > 0 else { return nil }
        for quality in configuration.compressionQualities {
            let data = NSMutableData()
            guard let destination = CGImageDestinationCreateWithData(
                data,
                UTType.jpeg.identifier as CFString,
                1,
                nil
            ) else { return nil }
            CGImageDestinationAddImage(
                destination,
                cgImage,
                [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary
            )
            guard CGImageDestinationFinalize(destination) else { continue }
            if data.length <= configuration.maximumJPEGBytes {
                return EncodedEnvironmentFrame(
                    jpeg: data as Data,
                    pixelWidth: outputWidth,
                    pixelHeight: outputHeight
                )
            }
        }
        return nil
    }
}

/// Exact visible pixels, including color interpretation. Row padding is not
/// image content. Unsupported layouts simply take the ordinary encode path.
func environmentFramePixelDigest(_ pixels: CVPixelBuffer) -> String? {
    guard CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA,
          !CVPixelBufferIsPlanar(pixels),
          CVPixelBufferLockBaseAddress(pixels, .readOnly) == kCVReturnSuccess else { return nil }
    defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixels) else { return nil }
    let width = CVPixelBufferGetWidth(pixels), height = CVPixelBufferGetHeight(pixels)
    let stride = CVPixelBufferGetBytesPerRow(pixels)
    guard width > 0, height > 0, stride >= width * 4 else { return nil }
    var hash = SHA256()
    hash.update(data: Data("\(width):\(height):BGRA".utf8))
    if let colorSpace = CVImageBufferGetColorSpace(pixels),
       let profile = colorSpace.takeUnretainedValue().copyICCData() { hash.update(data: profile as Data) }
    for row in 0..<height {
        hash.update(bufferPointer: UnsafeRawBufferPointer(start: base.advanced(by: row * stride), count: width * 4))
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

/// One encoder instance owns one serial queue, so its Core Image context can
/// safely be reused instead of rebuilding GPU/CPU resources every second.
private final class EnvironmentFrameImageEncoder: @unchecked Sendable {
    private let configuration: EnvironmentFrameEncoderConfiguration
    private let context = CIContext(options: [.cacheIntermediates: false])

    init(configuration: EnvironmentFrameEncoderConfiguration) {
        self.configuration = configuration
    }

    func encode(_ pixelBuffer: CVPixelBuffer) -> EncodedEnvironmentFrame? {
        EnvironmentFrameEncoder.encodeJPEG(
            pixelBuffer: pixelBuffer,
            configuration: configuration,
            context: context
        )
    }
}

struct CapturedEnvironmentFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let observedAt: Double
    let displayID: CGDirectDisplayID
}

/// A periodic request for fresh pixels, not a replay of the latest JPEG.
/// One worker owns capture and encoding; missed ticks are dropped, never queued.
final class EnvironmentFrameSamplingLoop: @unchecked Sendable {
    static let defaultIntervalMilliseconds = 1_000
    static let allowedIntervalMilliseconds = 250 ... 60_000

    typealias Capture = @Sendable (UInt64) async -> CapturedEnvironmentFrame?
    typealias Consume = @Sendable (
        CapturedEnvironmentFrame, UInt64, @escaping @Sendable () -> Bool
    ) async -> Void

    private let capture: Capture
    private let consume: Consume
    private let permitted: @Sendable () -> Bool
    private let onInvalidate: @Sendable (UInt64) -> Void
    private let automatic: Bool
    private let queue = DispatchQueue(
        label: "app.fishmemory.perception.environment-sampling",
        qos: .utility
    )
    private let lock = NSLock()
    private var generation: UInt64?
    private var epoch: UInt64 = 0
    private var timerRevision: UInt64 = 0
    private var intervalMilliseconds: Int
    private var timer: DispatchSourceTimer?
    private var worker: Task<Void, Never>?
    private var workerEpoch: UInt64?

    init(
        automatic: Bool = true,
        intervalMilliseconds: Int = EnvironmentFrameSamplingLoop.defaultIntervalMilliseconds,
        permitted: @escaping @Sendable () -> Bool,
        capture: @escaping Capture,
        consume: @escaping Consume,
        onInvalidate: @escaping @Sendable (UInt64) -> Void
    ) {
        precondition(Self.allowedIntervalMilliseconds.contains(intervalMilliseconds))
        self.automatic = automatic
        self.intervalMilliseconds = intervalMilliseconds
        self.permitted = permitted
        self.capture = capture
        self.consume = consume
        self.onInvalidate = onInvalidate
    }

    func activate(generation: UInt64) {
        invalidate()
        lock.lock()
        self.generation = generation
        epoch &+= 1
        timerRevision &+= 1
        if automatic {
            let timer = makeTimer(
                intervalMilliseconds: intervalMilliseconds,
                epoch: epoch,
                timerRevision: timerRevision,
                startsImmediately: true
            )
            self.timer = timer
            timer.resume()
        }
        lock.unlock()
    }

    /// Replaces the active cadence without invalidating an in-flight capture.
    /// Explicit configure requests may ask for an immediate fresh frame even
    /// when the numeric interval is unchanged.
    @discardableResult
    func updateInterval(milliseconds: Int, tickImmediately: Bool) -> Bool {
        guard Self.allowedIntervalMilliseconds.contains(milliseconds) else { return false }
        lock.lock()
        intervalMilliseconds = milliseconds
        guard generation != nil else {
            lock.unlock()
            return true
        }
        timerRevision &+= 1
        let scheduledEpoch = epoch
        let scheduledTimerRevision = timerRevision
        let previousTimer = timer
        let nextTimer = automatic
            ? makeTimer(
                intervalMilliseconds: milliseconds,
                epoch: scheduledEpoch,
                timerRevision: scheduledTimerRevision,
                startsImmediately: false
            )
            : nil
        timer = nextTimer
        // Publish only a resumed timer so invalidate() can safely cancel it
        // from another lifecycle callback as soon as the lock is released.
        nextTimer?.resume()
        lock.unlock()
        previousTimer?.cancel()
        if tickImmediately {
            tick(
                expectedEpoch: scheduledEpoch,
                expectedTimerRevision: scheduledTimerRevision
            )
        }
        return true
    }

    func invalidate(generation expected: UInt64? = nil, epoch expectedEpoch: UInt64? = nil) {
        lock.lock()
        guard (expected == nil || generation == expected),
              (expectedEpoch == nil || epoch == expectedEpoch) else {
            lock.unlock()
            return
        }
        let previous = generation
        generation = nil
        epoch &+= 1
        timerRevision &+= 1
        let timer = self.timer
        self.timer = nil
        let worker = self.worker
        // This synchronous encoder/broker clear must finish before a new epoch
        // activates the same ScreenMonitor generation. It must not reenter us.
        if let previous { onInvalidate(previous) }
        lock.unlock()
        timer?.cancel()
        worker?.cancel()
    }

    func tick() { tick(expectedEpoch: nil, expectedTimerRevision: nil) }

    private func tick(expectedEpoch: UInt64?, expectedTimerRevision: UInt64?) {
        lock.lock()
        guard let generation, worker == nil,
              expectedEpoch == nil || epoch == expectedEpoch,
              expectedTimerRevision == nil || timerRevision == expectedTimerRevision else {
            lock.unlock()
            return
        }
        let epoch = self.epoch
        lock.unlock()
        guard permitted() else { invalidate(generation: generation, epoch: epoch); return }
        lock.lock()
        guard self.generation == generation, self.epoch == epoch, worker == nil,
              expectedTimerRevision == nil || timerRevision == expectedTimerRevision else {
            lock.unlock()
            return
        }
        workerEpoch = epoch
        worker = Task { [self] in
            if isCurrent(generation: generation, epoch: epoch),
               let frame = await capture(generation),
               !Task.isCancelled,
               isCurrent(generation: generation, epoch: epoch) {
                await consume(frame, generation) { [weak self] in
                    self?.isCurrent(generation: generation, epoch: epoch) == true
                }
            }
            finishWorker(epoch: epoch)
        }
        lock.unlock()
    }

    private func makeTimer(
        intervalMilliseconds: Int,
        epoch: UInt64,
        timerRevision: UInt64,
        startsImmediately: Bool
    ) -> DispatchSourceTimer {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(
            deadline: startsImmediately
                ? .now()
                : .now() + .milliseconds(intervalMilliseconds),
            repeating: .milliseconds(intervalMilliseconds),
            leeway: .milliseconds(min(150, max(1, intervalMilliseconds / 4)))
        )
        timer.setEventHandler { [weak self] in
            self?.tick(
                expectedEpoch: epoch,
                expectedTimerRevision: timerRevision
            )
        }
        return timer
    }

    private func isCurrent(generation: UInt64, epoch: UInt64) -> Bool {
        lock.lock()
        let current = self.generation == generation && self.epoch == epoch
        lock.unlock()
        guard current else { return false }
        guard permitted() else {
            invalidate(generation: generation, epoch: epoch)
            return false
        }
        return true
    }

    func waitUntilIdle() async {
        await currentWorker()?.value
    }

    private func currentWorker() -> Task<Void, Never>? {
        lock.lock()
        defer { lock.unlock() }
        return worker
    }

    private func finishWorker(epoch: UInt64) {
        lock.lock()
        if workerEpoch == epoch {
            worker = nil
            workerEpoch = nil
        }
        lock.unlock()
    }

    deinit {
        timer?.cancel()
        worker?.cancel()
    }
}
