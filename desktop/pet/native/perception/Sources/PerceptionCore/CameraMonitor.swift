import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import ImageIO
import Vision

public struct CameraMonitorConfiguration: Equatable, Sendable {
    public let width: Int32
    public let height: Int32
    public let captureFramesPerSecond: Int32
    public let analysisFramesPerSecond: Double

    public init(
        width: Int32 = 640,
        height: Int32 = 480,
        captureFramesPerSecond: Int32 = 15,
        analysisFramesPerSecond: Double = 10
    ) {
        precondition(width > 0)
        precondition(height > 0)
        precondition(captureFramesPerSecond > 0)
        precondition(analysisFramesPerSecond > 0)
        precondition(analysisFramesPerSecond <= Double(captureFramesPerSecond))
        self.width = width
        self.height = height
        self.captureFramesPerSecond = captureFramesPerSecond
        self.analysisFramesPerSecond = analysisFramesPerSecond
    }
}

public struct CameraFaceSignals: Codable, Equatable, Sendable {
    public let confidence: Double
    public let yawRadians: Double?
    public let pitchRadians: Double?
    public let rollRadians: Double?
    public let leftEyeOpenness: Double?
    public let rightEyeOpenness: Double?
    public let mouthOpenness: Double?
    public let smileProxy: Double?
    public let fatigueProxy: Double?
    public let attentionProxy: Double?

    public init(
        confidence: Double,
        yawRadians: Double?,
        pitchRadians: Double?,
        rollRadians: Double?,
        leftEyeOpenness: Double?,
        rightEyeOpenness: Double?,
        mouthOpenness: Double?,
        smileProxy: Double?,
        fatigueProxy: Double?,
        attentionProxy: Double?
    ) {
        self.confidence = confidence
        self.yawRadians = yawRadians
        self.pitchRadians = pitchRadians
        self.rollRadians = rollRadians
        self.leftEyeOpenness = leftEyeOpenness
        self.rightEyeOpenness = rightEyeOpenness
        self.mouthOpenness = mouthOpenness
        self.smileProxy = smileProxy
        self.fatigueProxy = fatigueProxy
        self.attentionProxy = attentionProxy
    }
}

public struct CameraObservation: Codable, Equatable, Sendable {
    public let source: String
    public let kind: String
    public let revision: UInt64
    public let observedAt: Double
    public let facePresent: Bool
    public let faceCount: Int
    public let primaryFace: CameraFaceSignals?

    public init(
        revision: UInt64,
        observedAt: Double,
        facePresent: Bool,
        faceCount: Int,
        primaryFace: CameraFaceSignals?
    ) {
        self.source = "camera"
        self.kind = "camera_face_observation"
        self.revision = revision
        self.observedAt = observedAt
        self.facePresent = facePresent
        self.faceCount = faceCount
        self.primaryFace = primaryFace
    }
}

public final class CameraMonitor: NSObject, @unchecked Sendable {
    public typealias ObservationHandler = @Sendable (CameraObservation) -> Void

    private let configuration: CameraMonitorConfiguration
    private let onObservation: ObservationHandler
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(
        label: "com.fish-memory.perception.camera.session",
        qos: .userInitiated
    )
    private let captureQueue = DispatchQueue(
        label: "com.fish-memory.perception.camera.capture",
        qos: .userInteractive
    )
    private let analysisQueue = DispatchQueue(
        label: "com.fish-memory.perception.camera.analysis",
        qos: .userInitiated
    )
    private let lifecycleLock = NSLock()
    private let revisionLock = NSLock()
    private let frameGate = LatestValueGate<CameraFrame>()
    private let visionHandler = VNSequenceRequestHandler()
    private let landmarksRequest = VNDetectFaceLandmarksRequest()

    private var configured = false
    private var started = false
    private var captureGeneration: UInt64 = 0
    private var revision: UInt64 = 0
    private var lastAcceptedTimestamp = CMTime.invalid

    public init(
        configuration: CameraMonitorConfiguration = .init(),
        onObservation: @escaping ObservationHandler
    ) {
        self.configuration = configuration
        self.onObservation = onObservation
        super.init()
    }

    @discardableResult
    public func start() -> CapabilityReport {
        lifecycleLock.lock()
        if started {
            lifecycleLock.unlock()
            return runningReport()
        }
        lifecycleLock.unlock()

        let permission = Self.cameraPermission()
        guard permission == .granted else {
            return CapabilityReport(
                supported: true,
                requested: true,
                state: .unavailable,
                permission: permission,
                reason: "camera_permission_required"
            )
        }

        do {
            try sessionQueue.sync {
                if !configured {
                    try configureCaptureSession()
                    configured = true
                }
            }
        } catch {
            return CapabilityReport(
                supported: true,
                requested: true,
                state: .failed,
                permission: permission,
                reason: "camera_configuration_failed"
            )
        }

        lifecycleLock.lock()
        captureGeneration += 1
        started = true
        lifecycleLock.unlock()
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
        return runningReport()
    }

    public func stop() {
        lifecycleLock.lock()
        let wasStarted = started
        started = false
        captureGeneration += 1
        lifecycleLock.unlock()
        guard wasStarted else { return }

        frameGate.reset()
        captureQueue.async { [weak self] in
            self?.lastAcceptedTimestamp = .invalid
        }
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    public static func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    continuation.resume(returning: granted)
                }
            }
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    private func configureCaptureSession() throws {
        guard let device = AVCaptureDevice.default(for: .video) else {
            throw CameraMonitorError.noCamera
        }
        let input = try AVCaptureDeviceInput(device: device)

        session.beginConfiguration()
        defer { session.commitConfiguration() }
        guard session.canSetSessionPreset(.vga640x480) else {
            throw CameraMonitorError.unsupportedResolution
        }
        session.sessionPreset = .vga640x480
        guard session.canAddInput(input) else {
            throw CameraMonitorError.cannotAddVideoInput
        }
        session.addInput(input)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_32BGRA,
        ]
        videoOutput.setSampleBufferDelegate(self, queue: captureQueue)
        guard session.canAddOutput(videoOutput) else {
            throw CameraMonitorError.cannotAddVideoOutput
        }
        session.addOutput(videoOutput)
        if let connection = videoOutput.connection(with: .video) {
            connection.automaticallyAdjustsVideoMirroring = false
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = false
            }
        }

        try configureFrameRate(device)
    }

    private func configureFrameRate(_ device: AVCaptureDevice) throws {
        let framesPerSecond = Double(configuration.captureFramesPerSecond)
        guard device.activeFormat.videoSupportedFrameRateRanges.contains(where: {
            $0.minFrameRate <= framesPerSecond && $0.maxFrameRate >= framesPerSecond
        }) else {
            throw CameraMonitorError.unsupportedFrameRate
        }
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        let duration = CMTime(
            value: 1,
            timescale: configuration.captureFramesPerSecond
        )
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
    }

    private func runningReport() -> CapabilityReport {
        CapabilityReport(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        )
    }

    private static func cameraPermission() -> CapabilityPermission {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return .granted
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        @unknown default: return .unknown
        }
    }

    func shouldAccept(timestamp: CMTime) -> Bool {
        guard timestamp.isValid else { return true }
        guard lastAcceptedTimestamp.isValid else {
            lastAcceptedTimestamp = timestamp
            return true
        }
        let elapsed = CMTimeGetSeconds(timestamp - lastAcceptedTimestamp)
        guard elapsed >= (1 / configuration.analysisFramesPerSecond) else {
            return false
        }
        lastAcceptedTimestamp = timestamp
        return true
    }

    private func beginAnalysis(_ frame: CameraFrame) {
        guard let first = frameGate.submit(frame) else { return }
        analysisQueue.async { [weak self] in
            self?.drainFrames(startingWith: first)
        }
    }

    private func drainFrames(startingWith first: CameraFrame) {
        var current: CameraFrame? = first
        while let frame = current {
            autoreleasepool {
                analyze(frame)
            }
            current = frameGate.takeNextOrFinish()
        }
    }

    private func analyze(_ frame: CameraFrame) {
        do {
            try visionHandler.perform(
                [landmarksRequest],
                on: frame.pixelBuffer,
                orientation: Self.visionImageOrientation
            )
            guard isCurrent(generation: frame.generation) else { return }
            let faces = landmarksRequest.results ?? []
            let primary = faces.max { first, second in
                first.boundingBox.width * first.boundingBox.height
                    < second.boundingBox.width * second.boundingBox.height
            }
            emit(faces: faces, primary: primary)
        } catch {
            // A failed frame is dropped. The next latest frame remains eligible;
            // no image or Vision intermediate is retained or written to disk.
        }
    }

    private func isCurrent(generation: UInt64) -> Bool {
        lifecycleLock.lock()
        defer { lifecycleLock.unlock() }
        return started && captureGeneration == generation
    }

    static let visionImageOrientation: CGImagePropertyOrientation = .up

    private func emit(faces: [VNFaceObservation], primary: VNFaceObservation?) {
        revisionLock.lock()
        revision += 1
        let nextRevision = revision
        revisionLock.unlock()
        onObservation(
            CameraObservation(
                revision: nextRevision,
                observedAt: Date().timeIntervalSince1970,
                facePresent: primary != nil,
                faceCount: faces.count,
                primaryFace: primary.map(Self.signals)
            )
        )
    }

    private static func signals(for face: VNFaceObservation) -> CameraFaceSignals {
        let leftEye = FaceLandmarkMetrics.eyeOpenness(face.landmarks?.leftEye)
        let rightEye = FaceLandmarkMetrics.eyeOpenness(face.landmarks?.rightEye)
        let mouth = FaceLandmarkMetrics.mouthOpenness(face.landmarks?.outerLips)
        let smile = FaceLandmarkMetrics.smileProxy(face.landmarks?.outerLips)
        let eyeAverage = FaceLandmarkMetrics.average(leftEye, rightEye)
        let closedEyeProxy = eyeAverage.map { 1 - $0 }
        let fatigue = FaceLandmarkMetrics.fatigueProxy(
            closedEyeProxy: closedEyeProxy,
            mouthOpenness: mouth
        )
        let yaw = face.yaw?.doubleValue
        let pitch = face.pitch?.doubleValue
        return CameraFaceSignals(
            confidence: Double(face.confidence),
            yawRadians: yaw,
            pitchRadians: pitch,
            rollRadians: face.roll?.doubleValue,
            leftEyeOpenness: leftEye,
            rightEyeOpenness: rightEye,
            mouthOpenness: mouth,
            smileProxy: smile,
            fatigueProxy: fatigue,
            attentionProxy: FaceLandmarkMetrics.attentionProxy(
                yaw: yaw,
                pitch: pitch
            )
        )
    }
}

extension CameraMonitor: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        _ = output
        _ = connection
        lifecycleLock.lock()
        let isStarted = started
        let generation = captureGeneration
        lifecycleLock.unlock()
        guard isStarted else { return }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        guard shouldAccept(timestamp: timestamp),
            let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else {
            return
        }
        beginAnalysis(CameraFrame(pixelBuffer: pixelBuffer, generation: generation))
    }
}

enum CameraMonitorError: Error {
    case noCamera
    case unsupportedResolution
    case unsupportedFrameRate
    case cannotAddVideoInput
    case cannotAddVideoOutput
}

private struct CameraFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let generation: UInt64
}

final class LatestValueGate<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var processing = false
    private var pending: Value?

    func submit(_ value: Value) -> Value? {
        lock.lock()
        defer { lock.unlock() }
        pending = value
        guard !processing else { return nil }
        processing = true
        let next = pending
        pending = nil
        return next
    }

    func takeNextOrFinish() -> Value? {
        lock.lock()
        defer { lock.unlock() }
        if let pending {
            self.pending = nil
            return pending
        }
        processing = false
        return nil
    }

    func reset() {
        lock.lock()
        pending = nil
        lock.unlock()
    }
}

enum FaceLandmarkMetrics {
    static func eyeOpenness(_ region: VNFaceLandmarkRegion2D?) -> Double? {
        guard let ratio = aspectRatio(region) else { return nil }
        return clamp(ratio / 0.45)
    }

    static func mouthOpenness(_ region: VNFaceLandmarkRegion2D?) -> Double? {
        guard let ratio = aspectRatio(region) else { return nil }
        return clamp(ratio / 0.55)
    }

    static func smileProxy(_ region: VNFaceLandmarkRegion2D?) -> Double? {
        guard let bounds = bounds(region) else { return nil }
        return clamp((bounds.width - 0.32) / 0.32)
    }

    static func fatigueProxy(
        closedEyeProxy: Double?,
        mouthOpenness: Double?
    ) -> Double? {
        guard closedEyeProxy != nil || mouthOpenness != nil else { return nil }
        let eye = closedEyeProxy ?? 0
        let yawn = mouthOpenness.map { clamp(($0 - 0.45) / 0.55) } ?? 0
        return clamp((eye * 0.7) + (yawn * 0.3))
    }

    static func attentionProxy(yaw: Double?, pitch: Double?) -> Double? {
        guard yaw != nil || pitch != nil else { return nil }
        let horizontal = abs(yaw ?? 0) / 0.8
        let vertical = abs(pitch ?? 0) / 0.7
        return clamp(1 - max(horizontal, vertical))
    }

    static func average(_ first: Double?, _ second: Double?) -> Double? {
        switch (first, second) {
        case let (first?, second?): return (first + second) / 2
        case let (first?, nil): return first
        case let (nil, second?): return second
        case (nil, nil): return nil
        }
    }

    static func aspectRatio(_ region: VNFaceLandmarkRegion2D?) -> Double? {
        guard let bounds = bounds(region), bounds.width > 0 else { return nil }
        return bounds.height / bounds.width
    }

    static func bounds(
        _ region: VNFaceLandmarkRegion2D?
    ) -> (width: Double, height: Double)? {
        guard let region, region.pointCount > 1 else { return nil }
        let points = region.normalizedPoints
        guard let first = points.first else { return nil }
        var minimumX = Double(first.x)
        var maximumX = Double(first.x)
        var minimumY = Double(first.y)
        var maximumY = Double(first.y)
        for point in points.dropFirst() {
            minimumX = min(minimumX, Double(point.x))
            maximumX = max(maximumX, Double(point.x))
            minimumY = min(minimumY, Double(point.y))
            maximumY = max(maximumY, Double(point.y))
        }
        return (maximumX - minimumX, maximumY - minimumY)
    }

    static func clamp(_ value: Double) -> Double {
        min(1, max(0, value))
    }
}
