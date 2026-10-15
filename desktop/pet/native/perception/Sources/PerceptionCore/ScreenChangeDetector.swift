import CoreGraphics
import Foundation

public enum ScreenMotionState: String, Codable, Equatable, Sendable {
    case changing
    case stable
    case continuousMotion = "continuous_motion"
}

public struct ScreenChangeDetectorConfiguration: Equatable, Sendable {
    public let analysisInterval: TimeInterval
    public let stableDirtyAreaRatio: Double
    public let stableSampleCount: Int
    public let significantDirtyAreaRatio: Double
    public let continuousMotionDirtyAreaRatio: Double
    public let continuousMotionWindow: TimeInterval
    public let continuousMotionMinimumSamples: Int
    public let continuousMotionMinimumMovingSamples: Int

    public init(
        analysisInterval: TimeInterval = 0.2,
        stableDirtyAreaRatio: Double = 0.015,
        stableSampleCount: Int = 3,
        significantDirtyAreaRatio: Double = 0.08,
        continuousMotionDirtyAreaRatio: Double = 0.15,
        continuousMotionWindow: TimeInterval = 2.0,
        continuousMotionMinimumSamples: Int = 10,
        continuousMotionMinimumMovingSamples: Int = 7
    ) {
        precondition(analysisInterval > 0)
        precondition(stableDirtyAreaRatio >= 0 && stableDirtyAreaRatio <= 1)
        precondition(stableSampleCount > 0)
        precondition(significantDirtyAreaRatio >= 0 && significantDirtyAreaRatio <= 1)
        precondition(continuousMotionDirtyAreaRatio >= 0 && continuousMotionDirtyAreaRatio <= 1)
        precondition(continuousMotionWindow > 0)
        precondition(continuousMotionMinimumSamples > 0)
        precondition(continuousMotionMinimumMovingSamples > 0)
        precondition(continuousMotionMinimumMovingSamples <= continuousMotionMinimumSamples)
        self.analysisInterval = analysisInterval
        self.stableDirtyAreaRatio = stableDirtyAreaRatio
        self.stableSampleCount = stableSampleCount
        self.significantDirtyAreaRatio = significantDirtyAreaRatio
        self.continuousMotionDirtyAreaRatio = continuousMotionDirtyAreaRatio
        self.continuousMotionWindow = continuousMotionWindow
        self.continuousMotionMinimumSamples = continuousMotionMinimumSamples
        self.continuousMotionMinimumMovingSamples = continuousMotionMinimumMovingSamples
    }

    public static let `default` = ScreenChangeDetectorConfiguration()
}

public struct ScreenChangeEvaluation: Codable, Equatable, Sendable {
    public let revision: UInt64
    public let timestamp: TimeInterval
    public let dirtyAreaRatio: Double
    public let dirtyRectCount: Int
    public let state: ScreenMotionState
    public let stateChanged: Bool
    public let significantChange: Bool

    public init(
        revision: UInt64,
        timestamp: TimeInterval,
        dirtyAreaRatio: Double,
        dirtyRectCount: Int,
        state: ScreenMotionState,
        stateChanged: Bool,
        significantChange: Bool
    ) {
        self.revision = revision
        self.timestamp = timestamp
        self.dirtyAreaRatio = dirtyAreaRatio
        self.dirtyRectCount = dirtyRectCount
        self.state = state
        self.stateChanged = stateChanged
        self.significantChange = significantChange
    }
}

/// A deterministic, pixel-free state machine. Screen pixels may be measured by
/// a separate local sampler, but only a bounded change ratio enters this type.
public struct ScreenChangeDetector: Sendable {
    private struct MotionSample: Sendable {
        let timestamp: TimeInterval
        let moving: Bool
    }

    public let configuration: ScreenChangeDetectorConfiguration
    private var revision: UInt64 = 0
    private var lastAnalysisTimestamp: TimeInterval?
    private var stableRun = 0
    private var state: ScreenMotionState = .changing
    private var motionWindow: [MotionSample] = []

    public init(configuration: ScreenChangeDetectorConfiguration = .default) {
        self.configuration = configuration
    }

    public func isAnalysisDue(timestamp: TimeInterval) -> Bool {
        guard timestamp.isFinite else { return false }
        guard let lastAnalysisTimestamp else { return true }
        return timestamp < lastAnalysisTimestamp
            || timestamp - lastAnalysisTimestamp + 1e-9
                >= configuration.analysisInterval
    }

    /// Returns nil for frames between 5 Hz analysis ticks. The first frame at
    /// or after each tick is the newest frame and all intervening frames are
    /// discarded.
    public mutating func ingest(
        timestamp: TimeInterval,
        frameWidth: Int,
        frameHeight: Int,
        dirtyRects: [CGRect]
    ) -> ScreenChangeEvaluation? {
        guard timestamp.isFinite, frameWidth > 0, frameHeight > 0 else { return nil }
        guard beginAnalysis(timestamp: timestamp) else { return nil }

        let frame = CGRect(x: 0, y: 0, width: frameWidth, height: frameHeight)
        var clipped: [CGRect] = []
        for rectangle in dirtyRects {
            let intersection = rectangle.standardized.intersection(frame)
            if !intersection.isNull && !intersection.isEmpty {
                clipped.append(intersection)
            }
        }
        let frameArea = Double(frameWidth) * Double(frameHeight)
        let dirtyAreaRatio = min(1, unionArea(of: clipped) / frameArea)
        return evaluate(
            timestamp: timestamp,
            changedAreaRatio: dirtyAreaRatio,
            changedRegionCount: clipped.count
        )
    }

    public mutating func ingestMeasured(
        timestamp: TimeInterval,
        changedAreaRatio: Double,
        changedRegionCount: Int
    ) -> ScreenChangeEvaluation? {
        guard timestamp.isFinite,
              changedAreaRatio.isFinite,
              changedAreaRatio >= 0,
              changedRegionCount >= 0,
              beginAnalysis(timestamp: timestamp)
        else { return nil }
        return evaluate(
            timestamp: timestamp,
            changedAreaRatio: min(1, changedAreaRatio),
            changedRegionCount: changedRegionCount
        )
    }

    private mutating func beginAnalysis(timestamp: TimeInterval) -> Bool {
        guard isAnalysisDue(timestamp: timestamp) else { return false }
        if let lastAnalysisTimestamp, timestamp < lastAnalysisTimestamp {
            reset()
        }
        lastAnalysisTimestamp = timestamp
        return true
    }

    private mutating func evaluate(
        timestamp: TimeInterval,
        changedAreaRatio: Double,
        changedRegionCount: Int
    ) -> ScreenChangeEvaluation {
        let previousState = state

        if changedAreaRatio <= configuration.stableDirtyAreaRatio {
            stableRun += 1
        } else {
            stableRun = 0
        }

        motionWindow.append(MotionSample(
            timestamp: timestamp,
            moving: changedAreaRatio
                >= configuration.continuousMotionDirtyAreaRatio
        ))
        let cutoff = timestamp - configuration.continuousMotionWindow
        motionWindow.removeAll { $0.timestamp < cutoff }
        var movingSamples = 0
        for sample in motionWindow where sample.moving {
            movingSamples += 1
        }

        if stableRun >= configuration.stableSampleCount {
            state = .stable
        } else if
            motionWindow.count >= configuration.continuousMotionMinimumSamples &&
            movingSamples >= configuration.continuousMotionMinimumMovingSamples
        {
            state = .continuousMotion
        } else if previousState != .continuousMotion
            || changedAreaRatio > configuration.stableDirtyAreaRatio
        {
            state = .changing
        }

        revision += 1
        return ScreenChangeEvaluation(
            revision: revision,
            timestamp: timestamp,
            dirtyAreaRatio: changedAreaRatio,
            dirtyRectCount: changedRegionCount,
            state: state,
            stateChanged: state != previousState,
            significantChange: changedAreaRatio
                >= configuration.significantDirtyAreaRatio
        )
    }

    public mutating func reset() {
        lastAnalysisTimestamp = nil
        stableRun = 0
        state = .changing
        motionWindow.removeAll(keepingCapacity: true)
    }
}

private func unionArea(of rectangles: [CGRect]) -> Double {
    guard !rectangles.isEmpty else { return 0 }
    var xCoordinates: [CGFloat] = []
    xCoordinates.reserveCapacity(rectangles.count * 2)
    for rectangle in rectangles {
        xCoordinates.append(rectangle.minX)
        xCoordinates.append(rectangle.maxX)
    }
    xCoordinates.sort()

    var uniqueX: [CGFloat] = []
    uniqueX.reserveCapacity(xCoordinates.count)
    for value in xCoordinates where uniqueX.last != value {
        uniqueX.append(value)
    }

    var area: CGFloat = 0
    guard uniqueX.count > 1 else { return 0 }
    for index in 0 ..< (uniqueX.count - 1) {
        let left = uniqueX[index]
        let right = uniqueX[index + 1]
        guard right > left else { continue }
        var intervals: [VerticalInterval] = []
        for rectangle in rectangles where rectangle.minX < right && rectangle.maxX > left {
            intervals.append(VerticalInterval(lower: rectangle.minY, upper: rectangle.maxY))
        }
        intervals.sort {
            $0.lower == $1.lower ? $0.upper < $1.upper : $0.lower < $1.lower
        }
        guard var current = intervals.first else { continue }
        var coveredHeight: CGFloat = 0
        for interval in intervals.dropFirst() {
            if interval.lower <= current.upper {
                current.upper = max(current.upper, interval.upper)
            } else {
                coveredHeight += current.upper - current.lower
                current = interval
            }
        }
        coveredHeight += current.upper - current.lower
        area += (right - left) * coveredHeight
    }
    return Double(area)
}

private struct VerticalInterval {
    let lower: CGFloat
    var upper: CGFloat
}
