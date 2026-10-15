import ApplicationServices
import Foundation

public struct ComputerHistoryMonitorConfiguration: Equatable, Sendable {
    public let maximumQueuedEvents: Int
    public let coalescingWindowSeconds: TimeInterval
    public let actionDiffDelaySeconds: TimeInterval
    public let maximumAXNodes: Int
    public let maximumAXDepth: Int
    public let maximumAXDurationSeconds: TimeInterval
    public let fullTreeMaximumBytes: Int
    public let diffMaximumBytes: Int

    public init(
        maximumQueuedEvents: Int = 128,
        coalescingWindowSeconds: TimeInterval = 0.1,
        actionDiffDelaySeconds: TimeInterval = 0.12,
        maximumAXNodes: Int = 1_024,
        maximumAXDepth: Int = 20,
        maximumAXDurationSeconds: TimeInterval = 0.25,
        fullTreeMaximumBytes: Int = 64 * 1_024,
        diffMaximumBytes: Int = 32 * 1_024
    ) {
        precondition(maximumQueuedEvents > 0)
        precondition(coalescingWindowSeconds >= 0)
        precondition(actionDiffDelaySeconds >= 0)
        precondition(maximumAXNodes > 0)
        precondition(maximumAXDepth >= 0)
        precondition(maximumAXDurationSeconds > 0)
        precondition(fullTreeMaximumBytes > 0)
        precondition(diffMaximumBytes > 0)
        self.maximumQueuedEvents = maximumQueuedEvents
        self.coalescingWindowSeconds = coalescingWindowSeconds
        self.actionDiffDelaySeconds = actionDiffDelaySeconds
        self.maximumAXNodes = maximumAXNodes
        self.maximumAXDepth = maximumAXDepth
        self.maximumAXDurationSeconds = maximumAXDurationSeconds
        self.fullTreeMaximumBytes = fullTreeMaximumBytes
        self.diffMaximumBytes = diffMaximumBytes
    }
}

@MainActor
public protocol ComputerHistoryMonitoring: AnyObject {
    func start() -> CapabilityReport
    func stop()
    func currentReport() -> CapabilityReport
}

struct ComputerHistoryContext: Equatable, Sendable {
    let key: String
    let app: ComputerHistoryApplication
    let window: ComputerHistoryWindow
}

struct ComputerHistoryAXCapture: Equatable, Sendable {
    let context: ComputerHistoryContext
    let content: ComputerHistoryAXContent
}

struct ComputerHistoryResolvedTarget: Equatable, Sendable {
    let target: ComputerHistoryTarget
    let secure: Bool
}

@MainActor
protocol ComputerHistoryAXProviding: AnyObject {
    func currentContext() -> ComputerHistoryContext?
    func capture(mode: ComputerHistoryAXMode) async -> ComputerHistoryAXCapture?
    func target(at point: ComputerHistoryPoint?) -> ComputerHistoryResolvedTarget?
    func currentFocusedTarget() -> ComputerHistoryResolvedTarget?
    func currentSelection() -> ComputerHistorySelection?
    func reset()
}

enum ComputerHistoryInteraction: Equatable, Sendable {
    case focusedWindowChanged(timestamp: Double)
    case windowContentChanged(timestamp: Double)
    case privacyBoundary
    case sessionResumed(timestamp: Double)
    case mouseClick(
        timestamp: Double,
        button: ComputerHistoryMouseButton,
        clickCount: Int,
        location: ComputerHistoryPoint
    )
    case mouseContextMenu(
        timestamp: Double,
        button: ComputerHistoryMouseButton,
        clickCount: Int,
        location: ComputerHistoryPoint
    )
    case mouseDrag(
        timestamp: Double,
        button: ComputerHistoryMouseButton,
        origin: ComputerHistoryPoint,
        destination: ComputerHistoryPoint
    )
    case selectionChanged(timestamp: Double)
    case keyboardShortcut(
        timestamp: Double,
        keyEquivalent: String,
        modifiers: [String]
    )
    case keyboardSubmit(
        timestamp: Double,
        keyEquivalent: String,
        modifiers: [String]
    )
    case keyboardTextInput(timestamp: Double, text: String)
}

typealias ComputerHistoryInteractionHandler = @MainActor (
    ComputerHistoryInteraction
) -> Void
typealias ComputerHistoryInputStatusHandler = @MainActor () -> Void

@MainActor
protocol ComputerHistoryInteractionListening: AnyObject {
    var inputReport: ComputerHistoryInputReport { get }
    func start(
        handler: @escaping ComputerHistoryInteractionHandler,
        onInputStatusChanged: @escaping ComputerHistoryInputStatusHandler
    ) -> Bool
    func stop()
}

@MainActor
public final class ComputerHistoryMonitor: ComputerHistoryMonitoring {
    public typealias EventHandler = @MainActor (ComputerHistoryEvent) -> Void
    public typealias SuppressionHandler = @MainActor (
        ComputerHistorySuppressionReport
    ) -> Void
    public typealias InputStatusHandler = @MainActor () -> Void

    private let configuration: ComputerHistoryMonitorConfiguration
    private let trustCheck: @MainActor () -> Bool
    private let listener: any ComputerHistoryInteractionListening
    private let axProvider: any ComputerHistoryAXProviding
    private let onEvent: EventHandler
    private let onSuppressed: SuppressionHandler
    private let onInputStatusChanged: InputStatusHandler
    private var buffer: ComputerHistoryEventBuffer
    private var flushTask: Task<Void, Never>?
    private var windowChangeTask: Task<Void, Never>?
    private var windowChangeTaskID: UUID?
    private var windowUpdateTask: Task<Void, Never>?
    private var windowUpdateTaskID: UUID?
    private var windowUpdateCaptureTaskID: UUID?
    private var pendingWindowUpdateTimestamp: Double?
    private var captureTasks: [UUID: Task<Void, Never>] = [:]
    private var selectionGate = ComputerHistorySelectionGate()
    private var activeContextKey: String?
    private var activeContext: ComputerHistoryContext?
    private var pendingFullTreeContextKey: String?
    private var generation = 0
    private var started = false

    public var suppressedEventCount: Int { buffer.suppressedEventCount }

    public convenience init(
        configuration: ComputerHistoryMonitorConfiguration = .init(),
        trustCheck: @escaping @MainActor () -> Bool = { AXIsProcessTrusted() },
        onSuppressed: @escaping SuppressionHandler = { _ in },
        onInputStatusChanged: @escaping InputStatusHandler = {},
        onEvent: @escaping EventHandler
    ) {
        self.init(
            configuration: configuration,
            trustCheck: trustCheck,
            listener: NativeComputerHistoryInteractionListener(),
            axProvider: NativeComputerHistoryAXProvider(configuration: configuration),
            onSuppressed: onSuppressed,
            onInputStatusChanged: onInputStatusChanged,
            onEvent: onEvent
        )
    }

    init(
        configuration: ComputerHistoryMonitorConfiguration = .init(),
        trustCheck: @escaping @MainActor () -> Bool,
        listener: any ComputerHistoryInteractionListening,
        axProvider: any ComputerHistoryAXProviding,
        onSuppressed: @escaping SuppressionHandler = { _ in },
        onInputStatusChanged: @escaping InputStatusHandler = {},
        onEvent: @escaping EventHandler
    ) {
        self.configuration = configuration
        self.trustCheck = trustCheck
        self.listener = listener
        self.axProvider = axProvider
        self.onSuppressed = onSuppressed
        self.onInputStatusChanged = onInputStatusChanged
        self.onEvent = onEvent
        buffer = ComputerHistoryEventBuffer(
            maximumCount: configuration.maximumQueuedEvents,
            coalescingWindowSeconds: configuration.coalescingWindowSeconds
        )
    }

    @discardableResult
    public func start() -> CapabilityReport {
        guard !started else { return currentReport() }
        guard trustCheck() else {
            return CapabilityReport(
                supported: true,
                requested: true,
                state: .unavailable,
                permission: .notDetermined,
                reason: "accessibility_permission_required",
                computerHistoryInput: listener.inputReport
                    .replacingAccessibilityPermission(.notDetermined)
            )
        }
        started = true
        generation += 1
        let listening = listener.start(
            handler: { [weak self] interaction in self?.accept(interaction) },
            onInputStatusChanged: { [weak self] in
                guard let self, self.started else { return }
                self.onInputStatusChanged()
            }
        )
        guard listening else {
            listener.stop()
            started = false
            return CapabilityReport(
                supported: true,
                requested: true,
                state: .failed,
                permission: .granted,
                reason: "global_event_monitor_unavailable"
            )
        }
        scheduleFullTree(kind: .sessionStarted, timestamp: Self.now())
        return currentReport()
    }

    public func stop() {
        guard started else { return }
        started = false
        generation += 1
        listener.stop()
        cancelCaptureTasks()
        cancelWindowChangeTask()
        cancelWindowUpdateTask()
        cancelFlushAndClearBuffer()
        selectionGate.reset()
        activeContextKey = nil
        activeContext = nil
        pendingFullTreeContextKey = nil
        axProvider.reset()
    }

    func flushPendingEvents() {
        flushTask?.cancel()
        flushTask = nil
        let events = buffer.drain()
        guard started else { return }
        let suppressed = buffer.takeSuppressedEventCount()
        for count in ComputerHistorySuppressionChunking.chunks(for: suppressed) {
            onSuppressed(ComputerHistorySuppressionReport(
                count: count,
                timestamp: Self.now()
            ))
        }
        for event in events {
            onEvent(event)
        }
    }

    public func currentReport() -> CapabilityReport {
        let accessibilityPermission: CapabilityPermission = trustCheck()
            ? .granted
            : .notDetermined
        let input = (started
            ? listener.inputReport
            : ComputerHistoryInputReport.disabled(
                accessibilityPermission: accessibilityPermission
            )
        ).replacingAccessibilityPermission(accessibilityPermission)
        let hasInput = input.eventTapState == .running
        return CapabilityReport(
            supported: true,
            requested: started,
            state: started ? (hasInput ? .running : .unavailable) : .disabled,
            permission: accessibilityPermission,
            reason: started && !hasInput
                ? Self.inputUnavailableReason(for: input)
                : nil,
            computerHistoryInput: input
        )
    }

    private static func inputUnavailableReason(
        for input: ComputerHistoryInputReport
    ) -> String {
        if input.listenEventPermission != .granted {
            return "input_monitoring_permission_required"
        }
        return "global_event_tap_unavailable"
    }

    private func accept(_ interaction: ComputerHistoryInteraction) {
        guard started else { return }
        switch interaction {
        case let .focusedWindowChanged(timestamp):
            let nextContextKey = axProvider.currentContext()?.key ?? ""
            guard nextContextKey != activeContextKey,
                  nextContextKey != pendingFullTreeContextKey
            else { return }
            resetForContextBoundary(clearQueuedEvents: false)
            pendingFullTreeContextKey = nextContextKey
            scheduleWindowChanged(timestamp: timestamp)
        case let .windowContentChanged(timestamp):
            scheduleWindowContentChanged(timestamp: timestamp)
        case .privacyBoundary:
            resetForContextBoundary(clearQueuedEvents: true)
        case let .sessionResumed(timestamp):
            resetForContextBoundary(clearQueuedEvents: true)
            scheduleFullTree(kind: .sessionStarted, timestamp: timestamp)
        case let .mouseClick(timestamp, button, clickCount, location):
            guard let context = axProvider.currentContext() else { return }
            let target = axProvider.target(at: location)?.target
            scheduleAction(
                kind: .mouseClick,
                timestamp: timestamp,
                context: context,
                mouse: ComputerHistoryMouse(
                    button: button,
                    clickCount: clickCount,
                    target: target,
                    location: location
                )
            )
        case let .mouseContextMenu(timestamp, button, clickCount, location):
            guard let context = axProvider.currentContext() else { return }
            let target = axProvider.target(at: location)?.target
            scheduleAction(
                kind: .mouseContextMenu,
                timestamp: timestamp,
                context: context,
                mouse: ComputerHistoryMouse(
                    button: button,
                    clickCount: clickCount,
                    target: target,
                    location: location
                )
            )
        case let .mouseDrag(timestamp, button, origin, destination):
            guard let context = axProvider.currentContext() else { return }
            let target = axProvider.target(at: destination)?.target
            scheduleAction(
                kind: .mouseDrag,
                timestamp: timestamp,
                context: context,
                mouse: ComputerHistoryMouse(
                    button: button,
                    target: target,
                    origin: origin,
                    destination: destination
                )
            )
        case let .selectionChanged(timestamp):
            emitSelection(timestamp: timestamp)
        case let .keyboardShortcut(timestamp, keyEquivalent, modifiers):
            guard let context = axProvider.currentContext() else { return }
            let target = axProvider.currentFocusedTarget()?.target
            scheduleAction(
                kind: .keyboardShortcut,
                timestamp: timestamp,
                context: context,
                keyboard: ComputerHistoryKeyboard(
                    keyEquivalent: keyEquivalent,
                    modifiers: modifiers,
                    target: target
                )
            )
        case let .keyboardSubmit(timestamp, keyEquivalent, modifiers):
            guard let context = axProvider.currentContext() else { return }
            let target = axProvider.currentFocusedTarget()?.target
            scheduleAction(
                kind: .keyboardSubmit,
                timestamp: timestamp,
                context: context,
                keyboard: ComputerHistoryKeyboard(
                    keyEquivalent: keyEquivalent,
                    modifiers: modifiers,
                    target: target
                )
            )
        case let .keyboardTextInput(timestamp, text):
            emitKeyboardText(text, timestamp: timestamp)
        }
    }

    private func scheduleFullTree(
        kind: ComputerHistoryEventKind,
        timestamp: Double
    ) {
        let fallback = axProvider.currentContext() ?? Self.emptyContext
        pendingFullTreeContextKey = fallback.key
        let taskID = UUID()
        let expectedGeneration = generation
        captureTasks[taskID] = Task { @MainActor [weak self] in
            guard let self else { return }
            let capture = await self.axProvider.capture(mode: .fullTree)
            self.captureTasks[taskID] = nil
            guard !Task.isCancelled,
                  self.started,
                  expectedGeneration == self.generation
            else { return }
            let context = capture?.context ?? fallback
            self.activeContextKey = context.key
            self.activeContext = context
            self.pendingFullTreeContextKey = nil
            self.enqueue(ComputerHistoryEvent(
                kind: kind,
                timestamp: timestamp,
                app: context.app,
                window: context.window,
                ax: capture?.content
            ))
        }
    }

    private func scheduleWindowContentChanged(timestamp: Double) {
        if windowUpdateCaptureTaskID != nil {
            pendingWindowUpdateTimestamp = timestamp
            return
        }
        windowUpdateTask?.cancel()
        let debounceTaskID = UUID()
        windowUpdateTaskID = debounceTaskID
        let expectedGeneration = generation
        let delay = configuration.coalescingWindowSeconds
        windowUpdateTask = Task { @MainActor [weak self] in
            if delay > 0 {
                try? await Task.sleep(
                    nanoseconds: UInt64(delay * 1_000_000_000)
                )
            } else {
                await Task.yield()
            }
            guard let self,
                  self.windowUpdateTaskID == debounceTaskID
            else { return }
            self.windowUpdateTask = nil
            self.windowUpdateTaskID = nil
            guard !Task.isCancelled,
                  self.started,
                  expectedGeneration == self.generation,
                  let context = self.axProvider.currentContext(),
                  context.key == self.activeContextKey
            else { return }
            self.captureWindowDiff(timestamp: timestamp, context: context)
        }
    }

    private func captureWindowDiff(
        timestamp: Double,
        context: ComputerHistoryContext
    ) {
        guard captureTasks.count < configuration.maximumQueuedEvents else {
            buffer.recordSuppressed()
            scheduleFlushTaskIfNeeded()
            return
        }
        let taskID = UUID()
        let expectedGeneration = generation
        windowUpdateCaptureTaskID = taskID
        captureTasks[taskID] = Task { @MainActor [weak self] in
            guard let self else { return }
            let capture = await self.axProvider.capture(mode: .diffFromPrevious)
            self.captureTasks[taskID] = nil
            let ownsWindowUpdate = self.windowUpdateCaptureTaskID == taskID
            let pendingTimestamp: Double?
            if ownsWindowUpdate {
                self.windowUpdateCaptureTaskID = nil
                pendingTimestamp = self.pendingWindowUpdateTimestamp
                self.pendingWindowUpdateTimestamp = nil
            } else {
                pendingTimestamp = nil
            }
            defer {
                if ownsWindowUpdate,
                   let pendingTimestamp,
                   self.started,
                   expectedGeneration == self.generation
                {
                    self.scheduleWindowContentChanged(timestamp: pendingTimestamp)
                }
            }
            guard !Task.isCancelled,
                  self.started,
                  expectedGeneration == self.generation
            else { return }
            let content = capture?.context.key == context.key
                ? capture?.content
                : nil
            let resolvedContext = capture?.context.key == context.key
                ? capture?.context ?? context
                : context
            let metadataChanged = resolvedContext.app != self.activeContext?.app
                || resolvedContext.window != self.activeContext?.window
            self.activeContextKey = resolvedContext.key
            self.activeContext = resolvedContext
            guard metadataChanged || !(content?.text.isEmpty ?? true) else { return }
            self.enqueue(ComputerHistoryEvent(
                kind: .windowChanged,
                timestamp: timestamp,
                app: resolvedContext.app,
                window: resolvedContext.window,
                ax: content?.text.isEmpty == false ? content : nil
            ))
        }
    }

    private func scheduleWindowChanged(timestamp: Double) {
        windowChangeTask?.cancel()
        let debounceTaskID = UUID()
        windowChangeTaskID = debounceTaskID
        let expectedGeneration = generation
        let delay = configuration.coalescingWindowSeconds
        windowChangeTask = Task { @MainActor [weak self] in
            if delay > 0 {
                try? await Task.sleep(
                    nanoseconds: UInt64(delay * 1_000_000_000)
                )
            } else {
                await Task.yield()
            }
            guard let self,
                  self.windowChangeTaskID == debounceTaskID
            else { return }
            self.windowChangeTask = nil
            self.windowChangeTaskID = nil
            guard !Task.isCancelled,
                  self.started,
                  expectedGeneration == self.generation
            else { return }
            self.scheduleFullTree(kind: .windowChanged, timestamp: timestamp)
        }
    }

    private func emitKeyboardText(_ text: String, timestamp: Double) {
        guard !text.isEmpty,
              let context = axProvider.currentContext(),
              let focused = axProvider.currentFocusedTarget(),
              !focused.secure
        else { return }
        enqueue(ComputerHistoryEvent(
            kind: .keyboardTextInput,
            timestamp: timestamp,
            app: context.app,
            window: context.window,
            keyboard: ComputerHistoryKeyboard(
                text: text,
                target: focused.target
            )
        ))
    }

    private func emitSelection(timestamp: Double) {
        guard let context = axProvider.currentContext(),
              let selection = axProvider.currentSelection(),
              selectionGate.shouldEmit(
                  contextKey: context.key,
                  selection: selection
              )
        else { return }
        enqueue(ComputerHistoryEvent(
            kind: .selectionChanged,
            timestamp: timestamp,
            app: context.app,
            window: context.window,
            selection: selection
        ))
    }

    private func scheduleAction(
        kind: ComputerHistoryEventKind,
        timestamp: Double,
        context: ComputerHistoryContext,
        mouse: ComputerHistoryMouse? = nil,
        keyboard: ComputerHistoryKeyboard? = nil
    ) {
        guard captureTasks.count < configuration.maximumQueuedEvents else {
            buffer.recordSuppressed()
            scheduleFlushTaskIfNeeded()
            return
        }
        let taskID = UUID()
        let expectedGeneration = generation
        let delay = configuration.actionDiffDelaySeconds
        captureTasks[taskID] = Task { @MainActor [weak self] in
            if delay > 0 {
                try? await Task.sleep(
                    nanoseconds: UInt64(delay * 1_000_000_000)
                )
            } else {
                await Task.yield()
            }
            guard let self else { return }
            guard !Task.isCancelled,
                  self.started,
                  expectedGeneration == self.generation
            else {
                self.captureTasks[taskID] = nil
                return
            }
            let capture = await self.axProvider.capture(mode: .diffFromPrevious)
            self.captureTasks[taskID] = nil
            guard !Task.isCancelled,
                  self.started,
                  expectedGeneration == self.generation
            else { return }
            let ax = capture?.context.key == context.key ? capture?.content : nil
            self.enqueue(ComputerHistoryEvent(
                kind: kind,
                timestamp: timestamp,
                app: context.app,
                window: context.window,
                ax: ax,
                mouse: mouse,
                keyboard: keyboard
            ))
        }
    }

    private func enqueue(_ event: ComputerHistoryEvent) {
        buffer.enqueue(event)
        scheduleFlushTaskIfNeeded()
    }

    private func scheduleFlushTaskIfNeeded() {
        guard flushTask == nil else { return }
        let delay = configuration.coalescingWindowSeconds
        flushTask = Task { @MainActor [weak self] in
            if delay > 0 {
                try? await Task.sleep(
                    nanoseconds: UInt64(delay * 1_000_000_000)
                )
            } else {
                await Task.yield()
            }
            guard !Task.isCancelled else { return }
            self?.attemptScheduledFlush()
        }
    }

    private func attemptScheduledFlush() {
        flushTask = nil
        guard captureTasks.isEmpty,
              windowChangeTask == nil,
              windowUpdateTask == nil
        else {
            scheduleFlushTaskIfNeeded()
            return
        }
        flushPendingEvents()
    }

    private func resetForContextBoundary(clearQueuedEvents: Bool) {
        generation += 1
        cancelCaptureTasks()
        cancelWindowChangeTask()
        cancelWindowUpdateTask()
        selectionGate.reset()
        activeContextKey = nil
        activeContext = nil
        pendingFullTreeContextKey = nil
        axProvider.reset()
        if clearQueuedEvents {
            cancelFlushAndClearBuffer()
        }
    }

    private func cancelCaptureTasks() {
        for task in captureTasks.values { task.cancel() }
        captureTasks.removeAll(keepingCapacity: true)
        windowUpdateCaptureTaskID = nil
        pendingWindowUpdateTimestamp = nil
    }

    private func cancelWindowChangeTask() {
        windowChangeTask?.cancel()
        windowChangeTask = nil
        windowChangeTaskID = nil
    }

    private func cancelWindowUpdateTask() {
        windowUpdateTask?.cancel()
        windowUpdateTask = nil
        windowUpdateTaskID = nil
        cancelWindowUpdateCapture()
        pendingWindowUpdateTimestamp = nil
    }

    private func cancelWindowUpdateCapture() {
        guard let taskID = windowUpdateCaptureTaskID else { return }
        captureTasks[taskID]?.cancel()
        captureTasks[taskID] = nil
        windowUpdateCaptureTaskID = nil
    }

    private func cancelFlushAndClearBuffer() {
        flushTask?.cancel()
        flushTask = nil
        buffer.removeAll()
    }

    private static func now() -> Double {
        Date().timeIntervalSince1970
    }

    private static let emptyContext = ComputerHistoryContext(
        key: "",
        app: ComputerHistoryApplication(bundleIdentifier: nil, name: nil),
        window: ComputerHistoryWindow(title: nil, url: nil)
    )
}
