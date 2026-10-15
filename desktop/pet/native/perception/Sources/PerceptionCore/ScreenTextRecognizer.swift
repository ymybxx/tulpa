import CoreGraphics
import CoreVideo
import Foundation
import Vision

public struct ScreenTextRecognitionConfiguration: Equatable, Sendable {
    public let minimumRecognitionInterval: TimeInterval
    public let maximumLineCount: Int
    public let maximumCharacterCount: Int

    public init(
        minimumRecognitionInterval: TimeInterval = 3,
        maximumLineCount: Int = 24,
        maximumCharacterCount: Int = 2_000
    ) {
        precondition(minimumRecognitionInterval >= 0)
        precondition(maximumLineCount > 0)
        precondition(maximumCharacterCount > 0)
        self.minimumRecognitionInterval = minimumRecognitionInterval
        self.maximumLineCount = maximumLineCount
        self.maximumCharacterCount = maximumCharacterCount
    }

    public static let `default` = ScreenTextRecognitionConfiguration()
}

public struct ScreenTextPrivacyContext: Equatable, Sendable {
    public let revision: UInt64
    public let observedAt: Double
    public let displayID: CGDirectDisplayID

    public init(
        revision: UInt64,
        observedAt: Double,
        displayID: CGDirectDisplayID
    ) {
        self.revision = revision
        self.observedAt = observedAt
        self.displayID = displayID
    }
}

public typealias ScreenTextPrivacyGate = @Sendable (
    ScreenTextPrivacyContext
) -> ScreenTextApplicationContext?

public enum ScreenTextPrivacyGates {
    /// ScreenCaptureKit does not reliably identify private-browsing or other
    /// sensitive frontmost content for a whole-display stream. OCR therefore
    /// stays fail-closed until a caller combines trusted app/window signals and
    /// explicitly allows a particular stable frame.
    public static let denyAll: ScreenTextPrivacyGate = { _ in nil }
}

public struct ScreenTextObservation: Codable, Equatable, Sendable {
    public let source: String
    public let kind: String
    public let revision: UInt64
    public let observedAt: Double
    public let displayID: CGDirectDisplayID
    public let text: String
    public let appBundleId: String
    public let appName: String?
    public let windowKey: String

    public init(
        revision: UInt64,
        observedAt: Double,
        displayID: CGDirectDisplayID,
        text: String,
        applicationContext: ScreenTextApplicationContext
    ) {
        source = "screen"
        kind = "screen_text"
        self.revision = revision
        self.observedAt = observedAt
        self.displayID = displayID
        self.text = text
        appBundleId = applicationContext.appBundleId
        appName = applicationContext.appName
        windowKey = applicationContext.contextToken
    }
}

/// Pure transition policy: OCR is eligible only after a significant change is
/// followed by a transition into stable state. Stable frames on their own and
/// continuous-motion frames never trigger recognition.
public struct ScreenTextTriggerPolicy: Sendable {
    public let minimumRecognitionInterval: TimeInterval
    private var significantChangePending = false
    private var attemptInFlight = false
    private var lastEvaluationTimestamp: TimeInterval?
    private var lastAttemptTimestamp: TimeInterval?

    public init(minimumRecognitionInterval: TimeInterval = 3) {
        precondition(minimumRecognitionInterval >= 0)
        self.minimumRecognitionInterval = minimumRecognitionInterval
    }

    public mutating func shouldRecognize(
        evaluation: ScreenChangeEvaluation
    ) -> Bool {
        if let lastEvaluationTimestamp,
           evaluation.timestamp < lastEvaluationTimestamp
        {
            reset()
        }
        lastEvaluationTimestamp = evaluation.timestamp
        if evaluation.significantChange {
            significantChangePending = true
        }
        guard evaluation.state == .stable,
              significantChangePending,
              !attemptInFlight
        else {
            return false
        }
        if let lastAttemptTimestamp,
           evaluation.timestamp - lastAttemptTimestamp
               < minimumRecognitionInterval
        {
            return false
        }
        // Consume only the change represented by this attempt. A newer change
        // observed while capture/OCR is in flight sets the flag again.
        significantChangePending = false
        attemptInFlight = true
        lastAttemptTimestamp = evaluation.timestamp
        return true
    }

    /// Failed privacy, matching, capture, or OCR attempts remain eligible after
    /// the same bounded retry interval; success uses that interval as cooldown.
    public mutating func completeAttempt(succeeded: Bool) {
        guard attemptInFlight else { return }
        attemptInFlight = false
        if !succeeded {
            significantChangePending = true
        }
    }

    public mutating func reset() {
        significantChangePending = false
        attemptInFlight = false
        lastEvaluationTimestamp = nil
        lastAttemptTimestamp = nil
    }
}

final class ScreenTextTriggerController: @unchecked Sendable {
    private let lock = NSLock()
    private var policy: ScreenTextTriggerPolicy

    init(minimumRecognitionInterval: TimeInterval) {
        policy = ScreenTextTriggerPolicy(
            minimumRecognitionInterval: minimumRecognitionInterval
        )
    }

    func shouldRecognize(evaluation: ScreenChangeEvaluation) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return policy.shouldRecognize(evaluation: evaluation)
    }

    func completeAttempt(succeeded: Bool) {
        lock.lock()
        policy.completeAttempt(succeeded: succeeded)
        lock.unlock()
    }

    func reset() {
        lock.lock()
        policy.reset()
        lock.unlock()
    }
}

struct ScreenTextSanitizationResult: Equatable, Sendable {
    let text: String?
    let containsSensitiveText: Bool
}

enum ScreenTextSanitizer {
    static func sanitize(
        _ candidates: [String],
        maximumLineCount: Int,
        maximumCharacterCount: Int
    ) -> String? {
        analyze(
            candidates,
            maximumLineCount: maximumLineCount,
            maximumCharacterCount: maximumCharacterCount
        ).text
    }

    static func analyze(
        _ candidates: [String],
        maximumLineCount: Int,
        maximumCharacterCount: Int
    ) -> ScreenTextSanitizationResult {
        guard maximumLineCount > 0, maximumCharacterCount > 0 else {
            return ScreenTextSanitizationResult(
                text: nil,
                containsSensitiveText: false
            )
        }
        var seen = Set<String>()
        var lines: [String] = []
        var remainingCharacters = maximumCharacterCount
        // Candidate generation is a security decision, so scan the complete
        // OCR result before applying local-observation line/character bounds.
        // A secret below the visible summary cutoff must still block JPEG.
        let containsSensitiveText = candidates.contains { candidate in
            candidate.components(separatedBy: .newlines).contains { rawLine in
                let line = normalizedLine(rawLine)
                return !line.isEmpty && ScreenTextSecretScanner.isSensitive(line)
            }
        }

        candidateLoop: for candidate in candidates {
            for rawLine in candidate.components(separatedBy: .newlines) {
                guard lines.count < maximumLineCount,
                      remainingCharacters > 0
                else { break candidateLoop }
                let normalized = normalizedLine(rawLine)
                guard !normalized.isEmpty else { continue }
                let scan = ScreenTextSecretScanner.redact(normalized)
                let line = scan.text
                let key = line.folding(
                    options: [.caseInsensitive, .diacriticInsensitive],
                    locale: nil
                )
                guard seen.insert(key).inserted else { continue }

                let separatorLength = lines.isEmpty ? 0 : 1
                guard remainingCharacters > separatorLength else {
                    break candidateLoop
                }
                let accepted = String(line.prefix(
                    remainingCharacters - separatorLength
                ))
                guard !accepted.isEmpty else { break candidateLoop }
                lines.append(accepted)
                remainingCharacters -= separatorLength + accepted.count
                if accepted.count < line.count { break candidateLoop }
            }
        }
        return ScreenTextSanitizationResult(
            text: lines.isEmpty ? nil : lines.joined(separator: "\n"),
            containsSensitiveText: containsSensitiveText
        )
    }

    private static func normalizedLine(_ value: String) -> String {
        value
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
    }
}

enum ScreenTextSecretScanner {
    static let redaction = "[敏感内容已隐藏]"

    static func redact(_ line: String) -> (text: String, sensitive: Bool) {
        guard isSensitive(line) else { return (line, false) }
        return (redaction, true)
    }

    static func isSensitive(_ line: String) -> Bool {
        if secretPatterns.contains(where: {
            line.range(of: $0, options: [.regularExpression, .caseInsensitive]) != nil
        }) {
            return true
        }
        return entropyTokens(in: line).contains(where: looksLikeHighEntropySecret)
    }

    private static func entropyTokens(in line: String) -> [String] {
        line.split { character in
            !(character.isLetter
                || character.isNumber
                || "_-/+=.".contains(character))
        }.map(String.init)
    }

    private static func looksLikeHighEntropySecret(_ token: String) -> Bool {
        let bytes = Array(token.utf8)
        guard bytes.count >= 24, bytes.count <= 512 else { return false }
        var classes = 0
        if bytes.contains(where: { (65 ... 90).contains($0) }) { classes += 1 }
        if bytes.contains(where: { (97 ... 122).contains($0) }) { classes += 1 }
        if bytes.contains(where: { (48 ... 57).contains($0) }) { classes += 1 }
        if bytes.contains(where: { !((48 ... 57).contains($0)
            || (65 ... 90).contains($0)
            || (97 ... 122).contains($0)) }) { classes += 1 }
        guard classes >= 2 else { return false }

        var counts: [UInt8: Int] = [:]
        for byte in bytes { counts[byte, default: 0] += 1 }
        let length = Double(bytes.count)
        let entropy = counts.values.reduce(0.0) { result, count in
            let probability = Double(count) / length
            return result - probability * log2(probability)
        }
        return entropy >= 4
    }

    private static let secretPatterns = [
        #"\b[A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*\s*[:=]\s*[^\s]+"#,
        #"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"#,
        #"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"#,
        #"\b(?:sk-(?:proj-)?|hf_|gh[pousr]_|xox[baprs]-|AIza|AKIA|LTAI)[A-Za-z0-9_/-]{8,}\b"#,
        #"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"#,
    ]
}

struct ScreenTextFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let generation: UInt64
    let context: ScreenTextPrivacyContext
    let applicationContext: ScreenTextApplicationContext
    let visionRegionOfInterest: CGRect
}

struct ScreenTextAnalyzedFrame: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let generation: UInt64
    let context: ScreenTextPrivacyContext
    let applicationContext: ScreenTextApplicationContext
    let visionRegionOfInterest: CGRect
    let recognizedText: Bool
    let containsSensitiveText: Bool
}

enum ScreenTextCandidatePolicy {
    static func allowsVisionCandidate(_ frame: ScreenTextAnalyzedFrame) -> Bool {
        frame.recognizedText && !frame.containsSensitiveText
    }
}

/// Latest-only, serial OCR worker. Vision work is isolated from the capture
/// queue and stale generations are checked after recognition before emission.
final class ScreenTextRecognizer: @unchecked Sendable {
    typealias RecognitionHandler = @Sendable (
        CVPixelBuffer,
        CGRect
    ) throws -> [String]
    typealias ObservationHandler = @Sendable (ScreenTextObservation) -> Void
    typealias AnalysisHandler = @Sendable (ScreenTextAnalyzedFrame) -> Void
    typealias CompletionHandler = @Sendable (
        ScreenTextPrivacyContext,
        UInt64,
        Bool
    ) -> Void
    typealias AuthorizationValidator = @Sendable (
        ScreenTextPrivacyContext,
        ScreenTextApplicationContext
    ) -> Bool

    private let configuration: ScreenTextRecognitionConfiguration
    private let recognize: RecognitionHandler
    private let isStillAuthorized: AuthorizationValidator
    private let onObservation: ObservationHandler
    private let onAnalysis: AnalysisHandler
    private let onCompletion: CompletionHandler
    private let analysisQueue: DispatchQueue
    private let frameGate = LatestValueGate<ScreenTextFrame>()
    private let stateLock = NSLock()
    private var activeGeneration: UInt64?
    private var lastEmittedKey: String?

    init(
        configuration: ScreenTextRecognitionConfiguration = .default,
        recognize: @escaping RecognitionHandler = { pixelBuffer, region in
            try ScreenTextRecognizer.performVisionRecognition(
                pixelBuffer,
                regionOfInterest: region
            )
        },
        isStillAuthorized: @escaping AuthorizationValidator = { _, _ in true },
        onAnalysis: @escaping AnalysisHandler = { _ in },
        onCompletion: @escaping CompletionHandler = { _, _, _ in },
        onObservation: @escaping ObservationHandler
    ) {
        self.configuration = configuration
        self.recognize = recognize
        self.isStillAuthorized = isStillAuthorized
        self.onAnalysis = onAnalysis
        self.onCompletion = onCompletion
        self.onObservation = onObservation
        analysisQueue = DispatchQueue(
            label: "app.fishmemory.perception.screen.ocr",
            qos: .utility
        )
    }

    func activate(generation: UInt64) {
        stateLock.lock()
        activeGeneration = generation
        lastEmittedKey = nil
        stateLock.unlock()
        frameGate.reset()
    }

    func invalidate(generation: UInt64? = nil) {
        stateLock.lock()
        if let generation, activeGeneration != generation {
            stateLock.unlock()
            return
        }
        activeGeneration = nil
        lastEmittedKey = nil
        stateLock.unlock()
        frameGate.reset()
    }

    func submit(
        pixelBuffer: CVPixelBuffer,
        generation: UInt64,
        context: ScreenTextPrivacyContext,
        applicationContext: ScreenTextApplicationContext,
        visionRegionOfInterest: CGRect
    ) {
        guard isCurrent(generation: generation) else { return }
        let frame = ScreenTextFrame(
            pixelBuffer: pixelBuffer,
            generation: generation,
            context: context,
            applicationContext: applicationContext,
            visionRegionOfInterest: visionRegionOfInterest
        )
        guard let first = frameGate.submit(frame) else { return }
        analysisQueue.async { [weak self] in
            self?.drainFrames(startingWith: first)
        }
    }

    private func drainFrames(startingWith first: ScreenTextFrame) {
        var current: ScreenTextFrame? = first
        while let frame = current {
            autoreleasepool {
                recognize(frame)
            }
            current = frameGate.takeNextOrFinish()
        }
    }

    private func recognize(_ frame: ScreenTextFrame) {
        let candidates: [String]
        do {
            candidates = try recognize(
                frame.pixelBuffer,
                frame.visionRegionOfInterest
            )
        } catch {
            completeIfCurrent(frame, succeeded: false)
            return
        }
        let sanitized = ScreenTextSanitizer.analyze(
            candidates,
            maximumLineCount: configuration.maximumLineCount,
            maximumCharacterCount: configuration.maximumCharacterCount
        )
        guard isStillAuthorized(frame.context, frame.applicationContext) else {
            completeIfCurrent(frame, succeeded: false)
            return
        }
        emitAnalysisIfCurrent(frame, sanitized: sanitized)
    }

    private func isCurrent(generation: UInt64) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return activeGeneration == generation
    }

    /// The handler runs while the generation lock is held so invalidate()
    /// cannot return and then be followed by a stale observation callback.
    private func emitAnalysisIfCurrent(
        _ frame: ScreenTextFrame,
        sanitized: ScreenTextSanitizationResult
    ) {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard activeGeneration == frame.generation else { return }
        guard isStillAuthorized(frame.context, frame.applicationContext) else {
            onCompletion(frame.context, frame.generation, false)
            return
        }
        if let text = sanitized.text {
            let key = [
                frame.applicationContext.appBundleId,
                frame.applicationContext.contextToken,
                text,
            ].joined(separator: "\n")
            if lastEmittedKey != key {
                lastEmittedKey = key
                onObservation(ScreenTextObservation(
                    revision: frame.context.revision,
                    observedAt: frame.context.observedAt,
                    displayID: frame.context.displayID,
                    text: text,
                    applicationContext: frame.applicationContext
                ))
            }
        }
        onAnalysis(ScreenTextAnalyzedFrame(
            pixelBuffer: frame.pixelBuffer,
            generation: frame.generation,
            context: frame.context,
            applicationContext: frame.applicationContext,
            visionRegionOfInterest: frame.visionRegionOfInterest,
            recognizedText: sanitized.text != nil,
            containsSensitiveText: sanitized.containsSensitiveText
        ))
        onCompletion(frame.context, frame.generation, true)
    }

    private func completeIfCurrent(
        _ frame: ScreenTextFrame,
        succeeded: Bool
    ) {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard activeGeneration == frame.generation else { return }
        onCompletion(frame.context, frame.generation, succeeded)
    }

    private static func performVisionRecognition(
        _ pixelBuffer: CVPixelBuffer,
        regionOfInterest: CGRect
    ) throws -> [String] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .fast
        request.usesLanguageCorrection = false
        request.regionOfInterest = regionOfInterest
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])
        return (request.results ?? []).compactMap {
            $0.topCandidates(1).first?.string
        }
    }
}
