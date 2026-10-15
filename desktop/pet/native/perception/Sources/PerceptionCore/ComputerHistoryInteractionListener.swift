import AppKit
import ApplicationServices
import Foundation

@MainActor
final class NativeComputerHistoryInteractionListener: NSObject,
    ComputerHistoryInteractionListening
{
    private struct MouseGesture {
        let button: ComputerHistoryMouseButton
        let origin: ComputerHistoryPoint
        let contextMenu: Bool
        var destination: ComputerHistoryPoint
        var dragged: Bool
    }

    private let workspace: NSWorkspace
    private let preflightListenEventAccess: @MainActor () -> Bool
    private let requestListenEventAccess: @MainActor () -> Bool
    private let inputTapStartOverride: (@MainActor (UInt64) -> Bool)?
    private var handler: ComputerHistoryInteractionHandler?
    private var onInputStatusChanged: ComputerHistoryInputStatusHandler?
    private var eventTap: CFMachPort?
    private var eventTapSource: CFRunLoopSource?
    private var eventTapContext: ComputerHistoryEventTapContext?
    private var eventTapState = ComputerHistoryEventTapState.disabled
    private var listenEventPermission = CapabilityPermission.notDetermined
    private var mouseReceivedCount: UInt64 = 0
    private var mouseLastEventAt: Double?
    private var keyboardReceivedCount: UInt64 = 0
    private var keyboardLastEventAt: Double?
    private var tapDisabledCount: UInt64 = 0
    private var eventTapGeneration: UInt64 = 0
    private var eventsSinceStatusNotification: UInt64 = 0
    private var lastStatusNotificationAt: Double?
    private var mouseStatusWasReported = false
    private var keyboardStatusWasReported = false
    private var listenPermissionTask: Task<Void, Never>?
    private var suspendedForPrivacy = false
    private var observer: AXObserver?
    private var applicationElement: AXUIElement?
    private var focusedWindow: AXUIElement?
    private var focusedElement: AXUIElement?
    private var mouseGesture: MouseGesture?
    private var started = false

    init(
        workspace: NSWorkspace = .shared,
        preflightListenEventAccess: @escaping @MainActor () -> Bool = {
            CGPreflightListenEventAccess()
        },
        requestListenEventAccess: @escaping @MainActor () -> Bool = {
            CGRequestListenEventAccess()
        },
        inputTapStartOverride: (@MainActor (UInt64) -> Bool)? = nil
    ) {
        self.workspace = workspace
        self.preflightListenEventAccess = preflightListenEventAccess
        self.requestListenEventAccess = requestListenEventAccess
        self.inputTapStartOverride = inputTapStartOverride
        super.init()
    }

    var inputReport: ComputerHistoryInputReport {
        let channelState: CapabilityState
        switch eventTapState {
        case .disabled:
            channelState = .disabled
        case .running:
            channelState = .running
        case .permissionRequired, .suspended:
            channelState = .unavailable
        case .failed:
            channelState = .failed
        }
        return ComputerHistoryInputReport(
            accessibilityPermission: AXIsProcessTrusted() ? .granted : .notDetermined,
            listenEventPermission: listenEventPermission,
            eventTapState: eventTapState,
            mouse: ComputerHistoryInputChannelReport(
                state: channelState,
                receivedCount: mouseReceivedCount,
                lastEventAt: mouseLastEventAt
            ),
            keyboard: ComputerHistoryInputChannelReport(
                state: channelState,
                receivedCount: keyboardReceivedCount,
                lastEventAt: keyboardLastEventAt
            ),
            tapDisabledCount: tapDisabledCount
        )
    }

    var inputEventGeneration: UInt64 { eventTapGeneration }
    var isPollingListenEventPermission: Bool { listenPermissionTask != nil }

    func start(
        handler: @escaping ComputerHistoryInteractionHandler,
        onInputStatusChanged: @escaping ComputerHistoryInputStatusHandler
    ) -> Bool {
        guard !started else { return true }
        started = true
        self.handler = handler
        self.onInputStatusChanged = onInputStatusChanged
        registerWorkspaceNotifications()
        attachAccessibility(to: workspace.frontmostApplication)
        startInputTap(requestPermissionIfNeeded: true)
        // Accessibility window/selection history remains useful when Input
        // Monitoring is denied. Starting the listener therefore succeeds even
        // when the listen-only tap is in a degraded state.
        return true
    }

    func stop() {
        guard started else { return }
        started = false
        suspendedForPrivacy = false
        eventTapGeneration &+= 1
        listenPermissionTask?.cancel()
        listenPermissionTask = nil
        destroyInputTap(nextState: .disabled)
        workspace.notificationCenter.removeObserver(self)
        detachAccessibility()
        mouseGesture = nil
        handler = nil
        onInputStatusChanged = nil
    }

    private func startInputTap(requestPermissionIfNeeded: Bool) {
        guard started else { return }
        guard !suspendedForPrivacy else {
            eventTapState = .suspended
            return
        }
        listenPermissionTask?.cancel()
        listenPermissionTask = nil

        var permitted = preflightListenEventAccess()
        if !permitted, requestPermissionIfNeeded {
            // This is reached only after the user explicitly enables Computer
            // History. Merely launching the Helper never requests access.
            permitted = requestListenEventAccess()
        }
        listenEventPermission = permitted ? .granted : .denied
        guard permitted else {
            destroyInputTap(nextState: .permissionRequired)
            notifyInputStatusChanged(force: true)
            scheduleListenPermissionPolling()
            return
        }

        destroyInputTap(nextState: .disabled)
        eventTapGeneration &+= 1
        let generation = eventTapGeneration
        if let inputTapStartOverride {
            eventTapState = inputTapStartOverride(generation) ? .running : .failed
            notifyInputStatusChanged(force: true)
            return
        }
        let context = ComputerHistoryEventTapContext(
            monitor: self,
            generation: generation
        )
        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .listenOnly,
            eventsOfInterest: Self.inputEventMask,
            callback: computerHistoryInputEventTapCallback,
            userInfo: Unmanaged.passUnretained(context).toOpaque()
        ) else {
            eventTapState = .failed
            notifyInputStatusChanged(force: true)
            return
        }
        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        eventTap = tap
        eventTapSource = source
        eventTapContext = context
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        guard CGEvent.tapIsEnabled(tap: tap) else {
            destroyInputTap(nextState: .failed)
            notifyInputStatusChanged(force: true)
            return
        }
        eventTapState = .running
        notifyInputStatusChanged(force: true)
    }

    private func destroyInputTap(nextState: ComputerHistoryEventTapState) {
        if let eventTap {
            CGEvent.tapEnable(tap: eventTap, enable: false)
        }
        if let eventTapSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), eventTapSource, .commonModes)
        }
        eventTapSource = nil
        eventTap = nil
        eventTapContext = nil
        eventTapState = nextState
        mouseGesture = nil
    }

    func suspendInputTap() {
        guard started else { return }
        suspendedForPrivacy = true
        listenPermissionTask?.cancel()
        listenPermissionTask = nil
        if let eventTap {
            CGEvent.tapEnable(tap: eventTap, enable: false)
        }
        eventTapState = .suspended
        mouseGesture = nil
        notifyInputStatusChanged(force: true)
    }

    func resumeInputTap() {
        guard started else { return }
        suspendedForPrivacy = false
        guard preflightListenEventAccess() else {
            listenEventPermission = .denied
            destroyInputTap(nextState: .permissionRequired)
            notifyInputStatusChanged(force: true)
            scheduleListenPermissionPolling()
            return
        }
        listenEventPermission = .granted
        if let eventTap {
            CGEvent.tapEnable(tap: eventTap, enable: true)
            if CGEvent.tapIsEnabled(tap: eventTap) {
                eventTapState = .running
                notifyInputStatusChanged(force: true)
                return
            }
        }
        startInputTap(requestPermissionIfNeeded: false)
    }

    func handleInputTapDisabled(generation: UInt64) {
        guard started,
              !suspendedForPrivacy,
              generation == eventTapGeneration
        else { return }
        tapDisabledCount = Self.incremented(tapDisabledCount)
        mouseGesture = nil
        guard preflightListenEventAccess() else {
            listenEventPermission = .denied
            destroyInputTap(nextState: .permissionRequired)
            notifyInputStatusChanged(force: true)
            scheduleListenPermissionPolling()
            return
        }
        listenEventPermission = .granted
        if inputTapStartOverride != nil {
            startInputTap(requestPermissionIfNeeded: false)
            return
        }
        guard let eventTap else {
            eventTapState = .failed
            notifyInputStatusChanged(force: true)
            return
        }
        CGEvent.tapEnable(tap: eventTap, enable: true)
        eventTapState = CGEvent.tapIsEnabled(tap: eventTap) ? .running : .failed
        notifyInputStatusChanged(force: true)
    }

    private func scheduleListenPermissionPolling() {
        guard started,
              !suspendedForPrivacy,
              listenPermissionTask == nil
        else { return }
        let expectedGeneration = eventTapGeneration
        listenPermissionTask = Task { @MainActor [weak self] in
            guard let self else { return }
            for _ in 0 ..< 120 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard !Task.isCancelled,
                      self.started,
                      !self.suspendedForPrivacy,
                      expectedGeneration == self.eventTapGeneration
                else { return }
                if self.preflightListenEventAccess() {
                    self.listenPermissionTask = nil
                    self.startInputTap(requestPermissionIfNeeded: false)
                    return
                }
            }
            self.listenPermissionTask = nil
        }
    }

    private func recordMouseInput(at timestamp: Double) {
        mouseReceivedCount = Self.incremented(mouseReceivedCount)
        mouseLastEventAt = timestamp
        let first = !mouseStatusWasReported
        mouseStatusWasReported = true
        notifyInputStatusChanged(force: first)
    }

    private func recordKeyboardInput(at timestamp: Double) {
        keyboardReceivedCount = Self.incremented(keyboardReceivedCount)
        keyboardLastEventAt = timestamp
        let first = !keyboardStatusWasReported
        keyboardStatusWasReported = true
        notifyInputStatusChanged(force: first)
    }

    private func notifyInputStatusChanged(force: Bool) {
        let now = Self.now()
        eventsSinceStatusNotification = Self.incremented(
            eventsSinceStatusNotification
        )
        let intervalElapsed = lastStatusNotificationAt.map {
            now - $0 >= 1
        } ?? true
        guard force || intervalElapsed || eventsSinceStatusNotification >= 64 else {
            return
        }
        eventsSinceStatusNotification = 0
        lastStatusNotificationAt = now
        onInputStatusChanged?()
    }

    private static func incremented(_ value: UInt64) -> UInt64 {
        value == UInt64.max ? UInt64.max : value + 1
    }

    private func registerWorkspaceNotifications() {
        let center = workspace.notificationCenter
        center.addObserver(
            self,
            selector: #selector(applicationActivated(_:)),
            name: NSWorkspace.didActivateApplicationNotification,
            object: nil
        )
        center.addObserver(
            self,
            selector: #selector(activeSpaceChanged(_:)),
            name: NSWorkspace.activeSpaceDidChangeNotification,
            object: nil
        )
        for name in Self.privacyBoundaryNotifications {
            center.addObserver(
                self,
                selector: #selector(privacyBoundaryReached(_:)),
                name: name,
                object: nil
            )
        }
        for name in Self.resumeNotifications {
            center.addObserver(
                self,
                selector: #selector(sessionResumed(_:)),
                name: name,
                object: nil
            )
        }
    }

    @objc private func applicationActivated(_ notification: Notification) {
        guard started else { return }
        let application = notification.userInfo?[NSWorkspace.applicationUserInfoKey]
            as? NSRunningApplication
        attachAccessibility(to: application ?? workspace.frontmostApplication)
        handler?(.focusedWindowChanged(timestamp: Self.now()))
    }

    @objc private func activeSpaceChanged(_ notification: Notification) {
        guard started else { return }
        _ = notification
        attachAccessibility(to: workspace.frontmostApplication)
        handler?(.focusedWindowChanged(timestamp: Self.now()))
    }

    @objc private func privacyBoundaryReached(_ notification: Notification) {
        guard started else { return }
        _ = notification
        suspendInputTap()
        detachAccessibility()
        mouseGesture = nil
        handler?(.privacyBoundary)
    }

    @objc private func sessionResumed(_ notification: Notification) {
        guard started else { return }
        _ = notification
        attachAccessibility(to: workspace.frontmostApplication)
        resumeInputTap()
        handler?(.sessionResumed(timestamp: Self.now()))
    }

    func handleInputTapEvent(
        _ event: ComputerHistoryCGEventSnapshot,
        generation: UInt64
    ) {
        guard started,
              generation == eventTapGeneration,
              eventTapState == .running
        else { return }
        let timestamp = Self.now()
        switch event.family {
        case .mouse:
            recordMouseInput(at: timestamp)
        case .keyboard:
            recordKeyboardInput(at: timestamp)
        }
        switch event.type {
        case .leftMouseDown, .rightMouseDown, .otherMouseDown:
            let point = event.location
            mouseGesture = MouseGesture(
                button: event.button,
                origin: point,
                contextMenu: event.type == .rightMouseDown
                    || (event.type == .leftMouseDown
                        && event.modifierFlags.contains(.control)),
                destination: point,
                dragged: false
            )
        case .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            guard var gesture = mouseGesture else { return }
            gesture.destination = event.location
            gesture.dragged = true
            mouseGesture = gesture
        case .leftMouseUp, .rightMouseUp, .otherMouseUp:
            guard var gesture = mouseGesture else { return }
            gesture.destination = event.location
            mouseGesture = nil
            if gesture.dragged {
                handler?(.mouseDrag(
                    timestamp: timestamp,
                    button: gesture.button,
                    origin: gesture.origin,
                    destination: gesture.destination
                ))
            } else if gesture.contextMenu {
                handler?(.mouseContextMenu(
                    timestamp: timestamp,
                    button: .right,
                    clickCount: max(event.clickCount, 1),
                    location: gesture.destination
                ))
            } else {
                handler?(.mouseClick(
                    timestamp: timestamp,
                    button: gesture.button,
                    clickCount: max(event.clickCount, 1),
                    location: gesture.destination
                ))
            }
        case .keyDown:
            handleKeyDown(event)
        }
    }

    private func handleKeyDown(_ event: ComputerHistoryCGEventSnapshot) {
        guard let interaction = Self.keyboardInteraction(
            for: event,
            timestamp: Self.now()
        ) else { return }
        handler?(interaction)
    }

    static func keyboardInteraction(
        for event: ComputerHistoryCGEventSnapshot,
        timestamp: Double
    ) -> ComputerHistoryInteraction? {
        let modifiers = event.modifierFlags.names
        if let submit = Self.submitKeyNames[event.keyCode] {
            return .keyboardSubmit(
                timestamp: timestamp,
                keyEquivalent: submit,
                modifiers: modifiers
            )
        }
        if Self.isShortcut(event, modifiers: modifiers) {
            guard let key = Self.keyEquivalent(from: event), !key.isEmpty else {
                return nil
            }
            return .keyboardShortcut(
                timestamp: timestamp,
                keyEquivalent: key,
                modifiers: modifiers
            )
        }
        guard let characters = event.characters else { return nil }
        let normalized = Self.normalizedText(characters)
        guard !normalized.isEmpty else { return nil }
        return .keyboardTextInput(timestamp: timestamp, text: normalized)
    }

    private func attachAccessibility(to application: NSRunningApplication?) {
        detachAccessibility()
        guard started, let application else { return }
        let appElement = AXUIElementCreateApplication(application.processIdentifier)
        AXUIElementSetMessagingTimeout(appElement, 0.05)
        var createdObserver: AXObserver?
        guard AXObserverCreate(
            application.processIdentifier,
            computerHistoryAccessibilityCallback,
            &createdObserver
        ) == .success,
            let createdObserver
        else { return }

        observer = createdObserver
        applicationElement = appElement
        let context = Unmanaged.passUnretained(self).toOpaque()
        for name in Self.applicationNotifications {
            _ = AXObserverAddNotification(
                createdObserver,
                appElement,
                name as CFString,
                context
            )
        }
        CFRunLoopAddSource(
            CFRunLoopGetMain(),
            AXObserverGetRunLoopSource(createdObserver),
            .commonModes
        )
        refreshFocusedSubscriptions()
    }

    private func detachAccessibility() {
        if let observer {
            CFRunLoopRemoveSource(
                CFRunLoopGetMain(),
                AXObserverGetRunLoopSource(observer),
                .commonModes
            )
        }
        observer = nil
        applicationElement = nil
        focusedWindow = nil
        focusedElement = nil
    }

    private func refreshFocusedSubscriptions() {
        guard let observer, let applicationElement else { return }
        let context = Unmanaged.passUnretained(self).toOpaque()
        if let focusedWindow {
            for name in Self.focusedWindowNotifications {
                AXObserverRemoveNotification(
                    observer,
                    focusedWindow,
                    name as CFString
                )
            }
        }
        if let focusedElement {
            AXObserverRemoveNotification(
                observer,
                focusedElement,
                kAXSelectedTextChangedNotification as CFString
            )
        }
        focusedWindow = NativeComputerHistoryAXProvider.elementAttribute(
            kAXFocusedWindowAttribute,
            from: applicationElement
        )
        focusedElement = NativeComputerHistoryAXProvider.elementAttribute(
            kAXFocusedUIElementAttribute,
            from: applicationElement
        )
        if let focusedWindow {
            AXUIElementSetMessagingTimeout(focusedWindow, 0.05)
            for name in Self.focusedWindowNotifications {
                _ = AXObserverAddNotification(
                    observer,
                    focusedWindow,
                    name as CFString,
                    context
                )
            }
        }
        if let focusedElement {
            AXUIElementSetMessagingTimeout(focusedElement, 0.05)
            _ = AXObserverAddNotification(
                observer,
                focusedElement,
                kAXSelectedTextChangedNotification as CFString,
                context
            )
        }
    }

    fileprivate func handleAccessibilityNotification(_ notification: String) {
        guard started else { return }
        switch notification {
        case kAXFocusedWindowChangedNotification,
            kAXWindowCreatedNotification:
            refreshFocusedSubscriptions()
            handler?(.focusedWindowChanged(timestamp: Self.now()))
        case kAXTitleChangedNotification:
            handler?(.windowContentChanged(timestamp: Self.now()))
        case kAXMovedNotification,
            kAXResizedNotification:
            // A soft capture advances the internal visibility baseline. Pure
            // geometry remains silent, while newly revealed semantic nodes can
            // still produce a compact diff.
            handler?(.windowContentChanged(timestamp: Self.now()))
        case kAXFocusedUIElementChangedNotification:
            refreshFocusedSubscriptions()
        case kAXSelectedTextChangedNotification:
            handler?(.selectionChanged(timestamp: Self.now()))
        default:
            break
        }
    }

    private static func isShortcut(
        _ event: ComputerHistoryCGEventSnapshot,
        modifiers: [String]
    ) -> Bool {
        modifiers.contains("command")
            || modifiers.contains("control")
            || modifiers.contains("option")
            || modifiers.contains("function")
            || specialKeyNames[event.keyCode] != nil
    }

    private static func keyEquivalent(
        from event: ComputerHistoryCGEventSnapshot
    ) -> String? {
        if let special = specialKeyNames[event.keyCode] { return special }
        if event.keyCode == 49 { return "space" }
        return event.charactersIgnoringModifiers?
            .trimmingCharacters(in: .controlCharacters)
            .lowercased()
            .prefix(32)
            .description
    }

    private static func normalizedText(_ value: String) -> String {
        var result = ""
        for scalar in value.unicodeScalars {
            if scalar.value == 9 {
                result.append("\t")
            } else if scalar.value == 10 || scalar.value == 13 {
                result.append("\n")
            } else if !CharacterSet.controlCharacters.contains(scalar) {
                result.unicodeScalars.append(scalar)
            }
            if result.count >= 64 { break }
        }
        return result
    }

    private static func now() -> Double {
        Date().timeIntervalSince1970
    }

    private static let applicationNotifications = [
        kAXFocusedWindowChangedNotification as String,
        kAXFocusedUIElementChangedNotification as String,
        kAXWindowCreatedNotification as String,
    ]

    private static let focusedWindowNotifications = [
        kAXTitleChangedNotification as String,
        kAXMovedNotification as String,
        kAXResizedNotification as String,
    ]

    private static let privacyBoundaryNotifications = [
        NSWorkspace.screensDidSleepNotification,
        NSWorkspace.sessionDidResignActiveNotification,
    ]

    private static let resumeNotifications = [
        NSWorkspace.screensDidWakeNotification,
        NSWorkspace.sessionDidBecomeActiveNotification,
    ]

    private static let submitKeyNames: [UInt16: String] = [
        36: "return",
        76: "enter",
    ]

    private static let specialKeyNames: [UInt16: String] = [
        48: "tab",
        51: "delete",
        53: "escape",
        115: "home",
        116: "page_up",
        117: "forward_delete",
        119: "end",
        121: "page_down",
        123: "arrow_left",
        124: "arrow_right",
        125: "arrow_down",
        126: "arrow_up",
    ]

    private static let inputEventMask: CGEventMask = [
        CGEventType.leftMouseDown,
        .leftMouseUp,
        .rightMouseDown,
        .rightMouseUp,
        .otherMouseDown,
        .otherMouseUp,
        .leftMouseDragged,
        .rightMouseDragged,
        .otherMouseDragged,
        .keyDown,
    ].reduce(CGEventMask(0)) { mask, type in
        mask | (CGEventMask(1) << CGEventMask(type.rawValue))
    }
}

enum ComputerHistoryCGEventFamily: Equatable, Sendable {
    case mouse
    case keyboard
}

enum ComputerHistoryCGEventType: Equatable, Sendable {
    case leftMouseDown
    case leftMouseUp
    case rightMouseDown
    case rightMouseUp
    case otherMouseDown
    case otherMouseUp
    case leftMouseDragged
    case rightMouseDragged
    case otherMouseDragged
    case keyDown
}

struct ComputerHistoryEventModifierFlags: OptionSet, Sendable {
    let rawValue: UInt64

    static let command = ComputerHistoryEventModifierFlags(
        rawValue: CGEventFlags.maskCommand.rawValue
    )
    static let control = ComputerHistoryEventModifierFlags(
        rawValue: CGEventFlags.maskControl.rawValue
    )
    static let option = ComputerHistoryEventModifierFlags(
        rawValue: CGEventFlags.maskAlternate.rawValue
    )
    static let shift = ComputerHistoryEventModifierFlags(
        rawValue: CGEventFlags.maskShift.rawValue
    )
    static let capsLock = ComputerHistoryEventModifierFlags(
        rawValue: CGEventFlags.maskAlphaShift.rawValue
    )
    static let function = ComputerHistoryEventModifierFlags(
        rawValue: CGEventFlags.maskSecondaryFn.rawValue
    )

    var names: [String] {
        var result: [String] = []
        if contains(.command) { result.append("command") }
        if contains(.control) { result.append("control") }
        if contains(.option) { result.append("option") }
        if contains(.shift) { result.append("shift") }
        if contains(.capsLock) { result.append("caps_lock") }
        if contains(.function) { result.append("function") }
        return result
    }
}

struct ComputerHistoryCGEventSnapshot: Sendable {
    let type: ComputerHistoryCGEventType
    let location: ComputerHistoryPoint
    let button: ComputerHistoryMouseButton
    let clickCount: Int
    let keyCode: UInt16
    let modifierFlags: ComputerHistoryEventModifierFlags
    let characters: String?
    let charactersIgnoringModifiers: String?

    var family: ComputerHistoryCGEventFamily {
        type == .keyDown ? .keyboard : .mouse
    }

    init?(_ type: CGEventType, event: CGEvent) {
        guard let mappedType = ComputerHistoryCGEventType(type) else { return nil }
        self.type = mappedType
        let location = event.location
        self.location = ComputerHistoryPoint(
            x: Double(location.x),
            y: Double(location.y)
        )
        switch type {
        case .leftMouseDown, .leftMouseUp, .leftMouseDragged:
            button = .left
        case .rightMouseDown, .rightMouseUp, .rightMouseDragged:
            button = .right
        default:
            button = .other
        }
        clickCount = max(
            Int(event.getIntegerValueField(.mouseEventClickState)),
            0
        )
        keyCode = UInt16(clamping: event.getIntegerValueField(.keyboardEventKeycode))
        modifierFlags = ComputerHistoryEventModifierFlags(
            rawValue: event.flags.rawValue
        )
        let nsEvent = mappedType == .keyDown ? NSEvent(cgEvent: event) : nil
        characters = nsEvent?.characters
        charactersIgnoringModifiers = nsEvent?.charactersIgnoringModifiers
    }

    init(
        type: ComputerHistoryCGEventType,
        location: ComputerHistoryPoint = .init(x: 0, y: 0),
        button: ComputerHistoryMouseButton = .left,
        clickCount: Int = 0,
        keyCode: UInt16 = 0,
        modifierFlags: ComputerHistoryEventModifierFlags = [],
        characters: String? = nil,
        charactersIgnoringModifiers: String? = nil
    ) {
        self.type = type
        self.location = location
        self.button = button
        self.clickCount = clickCount
        self.keyCode = keyCode
        self.modifierFlags = modifierFlags
        self.characters = characters
        self.charactersIgnoringModifiers = charactersIgnoringModifiers
    }
}

private extension ComputerHistoryCGEventType {
    init?(_ type: CGEventType) {
        switch type {
        case .leftMouseDown: self = .leftMouseDown
        case .leftMouseUp: self = .leftMouseUp
        case .rightMouseDown: self = .rightMouseDown
        case .rightMouseUp: self = .rightMouseUp
        case .otherMouseDown: self = .otherMouseDown
        case .otherMouseUp: self = .otherMouseUp
        case .leftMouseDragged: self = .leftMouseDragged
        case .rightMouseDragged: self = .rightMouseDragged
        case .otherMouseDragged: self = .otherMouseDragged
        case .keyDown: self = .keyDown
        default: return nil
        }
    }
}

private final class ComputerHistoryEventTapContext: @unchecked Sendable {
    weak var monitor: NativeComputerHistoryInteractionListener?
    let generation: UInt64

    init(
        monitor: NativeComputerHistoryInteractionListener,
        generation: UInt64
    ) {
        self.monitor = monitor
        self.generation = generation
    }
}

private struct ComputerHistoryEventTapTransfer: @unchecked Sendable {
    let monitor: NativeComputerHistoryInteractionListener?
    let generation: UInt64
    let snapshot: ComputerHistoryCGEventSnapshot?
    let disabled: Bool
}

private struct ComputerHistoryAXNotificationTransfer: @unchecked Sendable {
    let monitor: NativeComputerHistoryInteractionListener
    let notification: String
}

private func computerHistoryInputEventTapCallback(
    _ proxy: CGEventTapProxy,
    _ type: CGEventType,
    _ event: CGEvent,
    _ context: UnsafeMutableRawPointer?
) -> Unmanaged<CGEvent>? {
    _ = proxy
    guard let context else { return Unmanaged.passUnretained(event) }
    let tapContext = Unmanaged<ComputerHistoryEventTapContext>
        .fromOpaque(context)
        .takeUnretainedValue()
    let disabled = type == .tapDisabledByTimeout
        || type == .tapDisabledByUserInput
    let transfer = ComputerHistoryEventTapTransfer(
        monitor: tapContext.monitor,
        generation: tapContext.generation,
        snapshot: disabled ? nil : ComputerHistoryCGEventSnapshot(type, event: event),
        disabled: disabled
    )
    DispatchQueue.main.async {
        guard let monitor = transfer.monitor else { return }
        if transfer.disabled {
            monitor.handleInputTapDisabled(generation: transfer.generation)
        } else if let snapshot = transfer.snapshot {
            monitor.handleInputTapEvent(snapshot, generation: transfer.generation)
        }
    }
    return Unmanaged.passUnretained(event)
}

private func computerHistoryAccessibilityCallback(
    _ observer: AXObserver,
    _ element: AXUIElement,
    _ notification: CFString,
    _ context: UnsafeMutableRawPointer?
) {
    _ = observer
    _ = element
    guard let context else { return }
    let monitor = Unmanaged<NativeComputerHistoryInteractionListener>
        .fromOpaque(context)
        .takeUnretainedValue()
    let transfer = ComputerHistoryAXNotificationTransfer(
        monitor: monitor,
        notification: notification as String
    )
    DispatchQueue.main.async {
        transfer.monitor.handleAccessibilityNotification(transfer.notification)
    }
}
