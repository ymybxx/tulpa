import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct VisionCandidateMetadata: Codable, Equatable, Sendable {
    public let candidateId: String
    public let revision: UInt64
    public let observedAt: Double
    public let expiresAt: Double
    public let displayID: CGDirectDisplayID
    public let appBundleId: String
    public let appName: String?
    public let windowKey: String
}

public struct VisionCandidatePayload: Codable, Equatable, Sendable {
    public let candidateId: String
    public let revision: UInt64
    public let observedAt: Double
    public let expiresAt: Double
    public let displayID: CGDirectDisplayID
    public let appBundleId: String
    public let appName: String?
    public let windowKey: String
    public let mimeType: String
    public let jpegBase64: String

    init(metadata: VisionCandidateMetadata, jpeg: Data) {
        candidateId = metadata.candidateId
        revision = metadata.revision
        observedAt = metadata.observedAt
        expiresAt = metadata.expiresAt
        displayID = metadata.displayID
        appBundleId = metadata.appBundleId
        appName = metadata.appName
        windowKey = metadata.windowKey
        mimeType = "image/jpeg"
        jpegBase64 = jpeg.base64EncodedString()
    }
}

public struct VisionCandidateUnavailable: Codable, Equatable, Sendable {
    public let code: String

    public init() {
        code = "unavailable"
    }
}

public final class VisionCandidateBroker: @unchecked Sendable {
    public typealias AuthorizationValidator = @Sendable (
        ScreenTextPrivacyContext,
        ScreenTextApplicationContext
    ) -> Bool
    public typealias Clock = @Sendable () -> Double

    public static let maximumJPEGBytes = 200 * 1_024

    private struct StoredCandidate {
        let metadata: VisionCandidateMetadata
        let jpeg: Data
        let privacyContext: ScreenTextPrivacyContext
        let applicationContext: ScreenTextApplicationContext
    }

    public let timeToLive: TimeInterval
    private let clock: Clock
    private let expirationQueue = DispatchQueue(
        label: "app.fishmemory.perception.vision.expiration",
        qos: .utility
    )
    private let lock = NSLock()
    private var stored: StoredCandidate?
    private var expirationWorkItem: DispatchWorkItem?

    public init(
        timeToLive: TimeInterval = 5,
        clock: @escaping Clock = { Date().timeIntervalSince1970 }
    ) {
        precondition(timeToLive > 0 && timeToLive.isFinite)
        self.timeToLive = timeToLive
        self.clock = clock
    }

    func store(
        jpeg: Data,
        privacyContext: ScreenTextPrivacyContext,
        applicationContext: ScreenTextApplicationContext
    ) -> VisionCandidateMetadata? {
        guard !jpeg.isEmpty, jpeg.count <= Self.maximumJPEGBytes else { return nil }
        let now = clock()
        let expiresAt = privacyContext.observedAt + timeToLive
        guard now.isFinite, expiresAt.isFinite, now < expiresAt else { return nil }
        let metadata = VisionCandidateMetadata(
            candidateId: UUID().uuidString,
            revision: privacyContext.revision,
            observedAt: privacyContext.observedAt,
            expiresAt: expiresAt,
            displayID: privacyContext.displayID,
            appBundleId: applicationContext.appBundleId,
            appName: applicationContext.appName,
            windowKey: applicationContext.contextToken
        )
        let workItem = DispatchWorkItem { [weak self] in
            self?.expire(candidateId: metadata.candidateId)
        }

        lock.lock()
        expirationWorkItem?.cancel()
        stored = StoredCandidate(
            metadata: metadata,
            jpeg: jpeg,
            privacyContext: privacyContext,
            applicationContext: applicationContext
        )
        expirationWorkItem = workItem
        lock.unlock()
        expirationQueue.asyncAfter(
            deadline: .now() + max(0, expiresAt - now),
            execute: workItem
        )
        return metadata
    }

    public func claim(
        candidateId: String,
        isStillAuthorized: AuthorizationValidator
    ) -> VisionCandidatePayload? {
        let now = clock()
        guard now.isFinite else {
            clear()
            return nil
        }
        lock.lock()
        defer { lock.unlock() }
        guard let candidate = stored,
              candidate.metadata.candidateId == candidateId,
              now < candidate.metadata.expiresAt,
              isStillAuthorized(
                  candidate.privacyContext,
                  candidate.applicationContext
              )
        else {
            if stored?.metadata.candidateId == candidateId {
                clearLocked()
            }
            return nil
        }
        clearLocked()
        return VisionCandidatePayload(
            metadata: candidate.metadata,
            jpeg: candidate.jpeg
        )
    }

    public func clear() {
        lock.lock()
        clearLocked()
        lock.unlock()
    }

    func currentMetadata() -> VisionCandidateMetadata? {
        let now = clock()
        lock.lock()
        defer { lock.unlock() }
        guard let stored, now.isFinite, now < stored.metadata.expiresAt else {
            clearLocked()
            return nil
        }
        return stored.metadata
    }

    private func expire(candidateId: String) {
        lock.lock()
        if stored?.metadata.candidateId == candidateId {
            clearLocked()
        }
        lock.unlock()
    }

    private func clearLocked() {
        expirationWorkItem?.cancel()
        expirationWorkItem = nil
        stored = nil
    }
}

struct VisionCandidateEncoderConfiguration: Equatable, Sendable {
    let maximumLongEdge: Int
    let maximumJPEGBytes: Int
    let compressionQualities: [Double]
    let minimumCandidateInterval: TimeInterval

    init(
        maximumLongEdge: Int = 1_280,
        maximumJPEGBytes: Int = VisionCandidateBroker.maximumJPEGBytes,
        compressionQualities: [Double] = [0.82, 0.68, 0.52, 0.38, 0.25],
        minimumCandidateInterval: TimeInterval = 30
    ) {
        precondition(maximumLongEdge > 0)
        precondition(maximumJPEGBytes > 0)
        precondition(!compressionQualities.isEmpty)
        precondition(compressionQualities.allSatisfy { $0 > 0 && $0 <= 1 })
        precondition(
            minimumCandidateInterval >= 0
                && minimumCandidateInterval.isFinite
        )
        self.maximumLongEdge = maximumLongEdge
        self.maximumJPEGBytes = maximumJPEGBytes
        self.compressionQualities = compressionQualities
        self.minimumCandidateInterval = minimumCandidateInterval
    }
}

private struct VisionCandidateFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let regionOfInterest: CGRect
    let generation: UInt64
    let privacyContext: ScreenTextPrivacyContext
    let applicationContext: ScreenTextApplicationContext
}

final class VisionCandidateEncoder: @unchecked Sendable {
    typealias EncodeHandler = @Sendable (CVPixelBuffer, CGRect) -> Data?
    typealias AuthorizationValidator = VisionCandidateBroker.AuthorizationValidator
    typealias AvailableHandler = @Sendable (VisionCandidateMetadata) -> Void

    private let configuration: VisionCandidateEncoderConfiguration
    private let broker: VisionCandidateBroker
    private let encode: EncodeHandler
    private let isStillAuthorized: AuthorizationValidator
    private let onAvailable: AvailableHandler
    private let encodingQueue = DispatchQueue(
        label: "app.fishmemory.perception.vision.encoding",
        qos: .utility
    )
    private let frameGate = LatestValueGate<VisionCandidateFrame>()
    private let stateLock = NSLock()
    private var activeGeneration: UInt64?
    private var enabled = false
    private var lastAcceptedObservedAt: Double?

    init(
        configuration: VisionCandidateEncoderConfiguration = .init(),
        broker: VisionCandidateBroker,
        encode: EncodeHandler? = nil,
        isStillAuthorized: @escaping AuthorizationValidator,
        onAvailable: @escaping AvailableHandler
    ) {
        self.configuration = configuration
        self.broker = broker
        self.isStillAuthorized = isStillAuthorized
        self.onAvailable = onAvailable
        if let encode {
            self.encode = encode
        } else {
            let defaultEncode: EncodeHandler = { pixelBuffer, region in
                Self.encodeJPEG(
                    pixelBuffer: pixelBuffer,
                    regionOfInterest: region,
                    configuration: configuration
                )
            }
            self.encode = defaultEncode
        }
    }

    func activate(generation: UInt64, enabled: Bool) {
        stateLock.lock()
        activeGeneration = generation
        self.enabled = enabled
        lastAcceptedObservedAt = nil
        stateLock.unlock()
        frameGate.reset()
        if !enabled { broker.clear() }
    }

    func setEnabled(_ enabled: Bool, generation: UInt64?) {
        stateLock.lock()
        if self.enabled != enabled { lastAcceptedObservedAt = nil }
        self.enabled = enabled
        if let generation { activeGeneration = generation }
        stateLock.unlock()
        if !enabled {
            frameGate.reset()
            broker.clear()
        }
    }

    func invalidate(generation: UInt64? = nil) {
        stateLock.lock()
        if let generation, activeGeneration != generation {
            stateLock.unlock()
            return
        }
        activeGeneration = nil
        enabled = false
        lastAcceptedObservedAt = nil
        stateLock.unlock()
        frameGate.reset()
        broker.clear()
    }

    func submit(
        pixelBuffer: CVPixelBuffer,
        regionOfInterest: CGRect,
        generation: UInt64,
        privacyContext: ScreenTextPrivacyContext,
        applicationContext: ScreenTextApplicationContext
    ) {
        guard isCurrent(generation: generation) else { return }
        let frame = VisionCandidateFrame(
            pixelBuffer: pixelBuffer,
            regionOfInterest: regionOfInterest,
            generation: generation,
            privacyContext: privacyContext,
            applicationContext: applicationContext
        )
        guard let first = frameGate.submit(frame) else { return }
        encodingQueue.async { [weak self] in
            self?.drain(startingWith: first)
        }
    }

    private func drain(startingWith first: VisionCandidateFrame) {
        var current: VisionCandidateFrame? = first
        while let frame = current {
            autoreleasepool { encode(frame) }
            current = frameGate.takeNextOrFinish()
        }
    }

    private func encode(_ frame: VisionCandidateFrame) {
        // Privacy, exact-window matching, one-shot capture, and OCR secret
        // scanning all happen before submit(). Count the sparse interval only
        // when a still-authorized frame is about to begin safe JPEG encoding.
        guard isStillAuthorized(frame.privacyContext, frame.applicationContext),
              beginSafeEncoding(
                  generation: frame.generation,
                  observedAt: frame.privacyContext.observedAt
              ),
              let jpeg = encode(frame.pixelBuffer, frame.regionOfInterest),
              !jpeg.isEmpty,
              jpeg.count <= configuration.maximumJPEGBytes,
              isStillAuthorized(frame.privacyContext, frame.applicationContext)
        else { return }

        stateLock.lock()
        defer { stateLock.unlock() }
        guard enabled, activeGeneration == frame.generation,
              isStillAuthorized(frame.privacyContext, frame.applicationContext),
              let metadata = broker.store(
                  jpeg: jpeg,
                  privacyContext: frame.privacyContext,
                  applicationContext: frame.applicationContext
              )
        else { return }
        onAvailable(metadata)
    }

    private func beginSafeEncoding(
        generation: UInt64,
        observedAt: Double
    ) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard enabled, activeGeneration == generation, observedAt.isFinite else {
            return false
        }
        if let lastAcceptedObservedAt,
           observedAt >= lastAcceptedObservedAt,
           observedAt - lastAcceptedObservedAt
               < configuration.minimumCandidateInterval
        {
            return false
        }
        lastAcceptedObservedAt = observedAt
        return true
    }

    private func isCurrent(generation: UInt64) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return enabled && activeGeneration == generation
    }

    private static func encodeJPEG(
        pixelBuffer: CVPixelBuffer,
        regionOfInterest: CGRect,
        configuration: VisionCandidateEncoderConfiguration
    ) -> Data? {
        let source = CIImage(cvPixelBuffer: pixelBuffer)
        let width = source.extent.width
        let height = source.extent.height
        guard width > 0, height > 0 else { return nil }
        guard let crop = inwardPixelCrop(
            regionOfInterest: regionOfInterest,
            imageExtent: source.extent
        ) else { return nil }

        var image = source.cropped(to: crop)
        let longEdge = max(crop.width, crop.height)
        if longEdge > CGFloat(configuration.maximumLongEdge) {
            let scale = CGFloat(configuration.maximumLongEdge) / longEdge
            image = image.transformed(by: CGAffineTransform(
                scaleX: scale,
                y: scale
            ))
        }
        let context = CIContext(options: [.cacheIntermediates: false])
        guard let cgImage = context.createCGImage(image, from: image.extent) else {
            return nil
        }
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
                return data as Data
            }
        }
        return nil
    }

    static func inwardPixelCrop(
        regionOfInterest: CGRect,
        imageExtent: CGRect
    ) -> CGRect? {
        guard imageExtent.width > 0, imageExtent.height > 0,
              regionOfInterest.minX >= 0,
              regionOfInterest.minY >= 0,
              regionOfInterest.maxX <= 1,
              regionOfInterest.maxY <= 1
        else { return nil }
        let rawCrop = CGRect(
            x: imageExtent.minX + regionOfInterest.minX * imageExtent.width,
            y: imageExtent.minY + regionOfInterest.minY * imageExtent.height,
            width: regionOfInterest.width * imageExtent.width,
            height: regionOfInterest.height * imageExtent.height
        )
        let crop = CGRect(
            x: ceil(rawCrop.minX),
            y: ceil(rawCrop.minY),
            width: floor(rawCrop.maxX) - ceil(rawCrop.minX),
            height: floor(rawCrop.maxY) - ceil(rawCrop.minY)
        )
        guard crop.width > 0, crop.height > 0,
              imageExtent.contains(crop)
        else { return nil }
        return crop
    }
}
