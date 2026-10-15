import AppKit
import ApplicationServices
import Foundation

private let accessibilityTrustedCheckOptionPrompt = "AXTrustedCheckOptionPrompt"

public struct AccessibilityMonitorConfiguration: Equatable, Sendable {
    public let messagingTimeoutSeconds: Float
    public let maximumNodes: Int
    public let maximumDepth: Int
    public let maximumTextLength: Int
    public let valueChangeDebounceSeconds: TimeInterval

    public init(
        messagingTimeoutSeconds: Float = 0.15,
        maximumNodes: Int = 16,
        maximumDepth: Int = 4,
        maximumTextLength: Int = 512,
        valueChangeDebounceSeconds: TimeInterval = 0.2
    ) {
        precondition(messagingTimeoutSeconds > 0)
        precondition(maximumNodes > 0)
        precondition(maximumDepth >= 0)
        precondition(maximumTextLength > 0)
        precondition(valueChangeDebounceSeconds > 0)
        self.messagingTimeoutSeconds = messagingTimeoutSeconds
        self.maximumNodes = maximumNodes
        self.maximumDepth = maximumDepth
        self.maximumTextLength = maximumTextLength
        self.valueChangeDebounceSeconds = valueChangeDebounceSeconds
    }
}

public struct AccessibilityObservation: Codable, Equatable, Sendable {
    public let source: String
    public let revision: UInt64
    public let kind: String
    public let observedAt: Double
    public let processIdentifier: Int32
    public let bundleIdentifier: String?
    public let applicationName: String?
    public let windowTitle: String?
    public let windowFrame: AccessibilityWindowFrame?
    public let focusedRole: String?
    public let focusedSubrole: String?
    public let focusedTitle: String?
    public let focusedValue: String?
    public let contextTitles: [String]
    public let secureField: Bool
    public let nodesInspected: Int
    public let truncated: Bool

    public init(
        revision: UInt64,
        kind: String,
        observedAt: Double,
        processIdentifier: Int32,
        bundleIdentifier: String?,
        applicationName: String?,
        windowTitle: String?,
        windowFrame: AccessibilityWindowFrame? = nil,
        focusedRole: String?,
        focusedSubrole: String?,
        focusedTitle: String?,
        focusedValue: String?,
        contextTitles: [String],
        secureField: Bool,
        nodesInspected: Int,
        truncated: Bool
    ) {
        self.source = "accessibility"
        self.revision = revision
        self.kind = kind
        self.observedAt = observedAt
        self.processIdentifier = processIdentifier
        self.bundleIdentifier = bundleIdentifier
        self.applicationName = applicationName
        self.windowTitle = windowTitle
        self.windowFrame = windowFrame
        self.focusedRole = focusedRole
        self.focusedSubrole = focusedSubrole
        self.focusedTitle = focusedTitle
        self.focusedValue = focusedValue
        self.contextTitles = contextTitles
        self.secureField = secureField
        self.nodesInspected = nodesInspected
        self.truncated = truncated
    }
}

/// AX coordinates are global display points with a top-left origin.
public struct AccessibilityWindowFrame: Codable, Equatable, Sendable {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    public var rect: CGRect {
        CGRect(x: x, y: y, width: width, height: height)
    }
}

@MainActor
public final class AccessibilityMonitor: NSObject {
    public typealias ObservationHandler = @MainActor (AccessibilityObservation) -> Void

    private let workspace: NSWorkspace
    private let configuration: AccessibilityMonitorConfiguration
    private let trustCheck: () -> Bool
    private let onObservation: ObservationHandler
    private let valueChangeDebouncer: LatestAccessibilityEventDebouncer

    private var observer: AXObserver?
    private var applicationElement: AXUIElement?
    private var focusedWindow: AXUIElement?
    private var focusedElement: AXUIElement?
    private var processIdentifier: pid_t?
    private var started = false
    private var revision: UInt64 = 0

    public init(
        workspace: NSWorkspace = .shared,
        configuration: AccessibilityMonitorConfiguration = .init(),
        trustCheck: @escaping () -> Bool = { AXIsProcessTrusted() },
        onObservation: @escaping ObservationHandler
    ) {
        self.workspace = workspace
        self.configuration = configuration
        self.trustCheck = trustCheck
        self.onObservation = onObservation
        valueChangeDebouncer = LatestAccessibilityEventDebouncer(
            delaySeconds: configuration.valueChangeDebounceSeconds
        )
        super.init()
    }

    public static func requestPermissionPrompt() -> Bool {
        let options = [
            accessibilityTrustedCheckOptionPrompt: true,
        ] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }

    @discardableResult
    public func start() -> CapabilityReport {
        guard !started else { return report() }
        guard trustCheck() else {
            return CapabilityReport(
                supported: true,
                requested: true,
                state: .unavailable,
                permission: .notDetermined,
                reason: "accessibility_permission_required"
            )
        }

        started = true
        workspace.notificationCenter.addObserver(
            self,
            selector: #selector(frontmostApplicationChanged(_:)),
            name: NSWorkspace.didActivateApplicationNotification,
            object: nil
        )
        attach(to: workspace.frontmostApplication)
        return report()
    }

    public func stop() {
        guard started else { return }
        started = false
        workspace.notificationCenter.removeObserver(self)
        detachObserver()
    }

    private func report() -> CapabilityReport {
        CapabilityReport(
            supported: true,
            requested: started,
            state: started ? .running : .disabled,
            permission: trustCheck() ? .granted : .notDetermined
        )
    }

    @objc private func frontmostApplicationChanged(_ notification: Notification) {
        let application = notification.userInfo?[NSWorkspace.applicationUserInfoKey]
            as? NSRunningApplication
        attach(to: application ?? workspace.frontmostApplication)
    }

    private func attach(to application: NSRunningApplication?) {
        detachObserver()
        guard started, let application else { return }

        let pid = application.processIdentifier
        let appElement = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(
            appElement,
            configuration.messagingTimeoutSeconds
        )

        var createdObserver: AXObserver?
        let result = AXObserverCreate(
            pid,
            accessibilityObserverCallback,
            &createdObserver
        )
        guard result == .success, let createdObserver else { return }

        observer = createdObserver
        applicationElement = appElement
        processIdentifier = pid
        let context = Unmanaged.passUnretained(self).toOpaque()
        for notificationName in Self.applicationNotifications {
            let error = AXObserverAddNotification(
                createdObserver,
                appElement,
                notificationName as CFString,
                context
            )
            if error != .success && error != .notificationUnsupported {
                continue
            }
        }
        CFRunLoopAddSource(
            CFRunLoopGetMain(),
            AXObserverGetRunLoopSource(createdObserver),
            .commonModes
        )
        refreshFocusedElementSubscription()
        emit(kind: "accessibility_snapshot")
    }

    private func detachObserver() {
        valueChangeDebouncer.cancel()
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
        processIdentifier = nil
    }

    private func handle(element: AXUIElement, notification: String) {
        guard started else { return }
        if AccessibilityNotificationPolicy.shouldDebounce(notification) {
            valueChangeDebouncer.schedule { [weak self] in
                self?.emit(kind: Self.kind(for: notification))
            }
            return
        }
        if notification == kAXFocusedUIElementChangedNotification
            || notification == kAXFocusedWindowChangedNotification
        {
            refreshFocusedElementSubscription()
        }
        AXUIElementSetMessagingTimeout(
            element,
            configuration.messagingTimeoutSeconds
        )
        emit(kind: Self.kind(for: notification))
    }

    private func refreshFocusedElementSubscription() {
        valueChangeDebouncer.cancel()
        guard let observer, let applicationElement else { return }
        if let focusedWindow {
            AXObserverRemoveNotification(
                observer,
                focusedWindow,
                kAXTitleChangedNotification as CFString
            )
        }
        if let focusedElement {
            for name in Self.focusedElementNotifications {
                AXObserverRemoveNotification(observer, focusedElement, name as CFString)
            }
        }
        focusedWindow = elementAttribute(
            kAXFocusedWindowAttribute,
            from: applicationElement
        )
        if let focusedWindow {
            AXUIElementSetMessagingTimeout(
                focusedWindow,
                configuration.messagingTimeoutSeconds
            )
            _ = AXObserverAddNotification(
                observer,
                focusedWindow,
                kAXTitleChangedNotification as CFString,
                Unmanaged.passUnretained(self).toOpaque()
            )
        }
        focusedElement = elementAttribute(
            kAXFocusedUIElementAttribute,
            from: applicationElement
        )
        guard let focusedElement else { return }
        AXUIElementSetMessagingTimeout(
            focusedElement,
            configuration.messagingTimeoutSeconds
        )
        let context = Unmanaged.passUnretained(self).toOpaque()
        for name in Self.focusedElementNotifications {
            let error = AXObserverAddNotification(
                observer,
                focusedElement,
                name as CFString,
                context
            )
            if error != .success && error != .notificationUnsupported {
                continue
            }
        }
    }

    private func emit(kind: String) {
        guard let applicationElement, let processIdentifier else { return }
        let application = NSRunningApplication(processIdentifier: processIdentifier)
        let window = focusedWindow
            ?? elementAttribute(kAXFocusedWindowAttribute, from: applicationElement)
        let rawWindowTitle = stringAttribute(kAXTitleAttribute, from: window)
        let bundleIdentifier = application?.bundleIdentifier

        // A private browser window is a privacy boundary. Stay silent instead of
        // emitting a replacement observation that would itself disclose the mode.
        // This deliberately happens before reading focused text or walking its
        // ancestors. The title heuristic is intentionally narrow and fail-open.
        guard !AccessibilityPrivateBrowsingPolicy.shouldSuppress(
            bundleIdentifier: bundleIdentifier,
            windowTitle: rawWindowTitle
        ) else {
            return
        }

        let focused = elementAttribute(
            kAXFocusedUIElementAttribute,
            from: applicationElement
        ) ?? focusedElement
        let role = stringAttribute(kAXRoleAttribute, from: focused)
        let subrole = stringAttribute(kAXSubroleAttribute, from: focused)
        let secureField = AccessibilityValueFilter.isSecureField(
            role: role,
            subrole: subrole
        )
        let rawValue = secureField
            ? nil
            : stringAttribute(kAXValueAttribute, from: focused)
        let value = AccessibilityValueFilter.bounded(
            rawValue,
            maximumLength: configuration.maximumTextLength
        )
        let title = AccessibilityValueFilter.bounded(
            stringAttribute(kAXTitleAttribute, from: focused),
            maximumLength: configuration.maximumTextLength
        )
        let windowTitle = AccessibilityValueFilter.bounded(
            rawWindowTitle,
            maximumLength: configuration.maximumTextLength
        )
        let windowFrame = accessibilityWindowFrame(from: window)
        let context = collectAncestorTitles(from: focused)
        revision += 1
        onObservation(
            AccessibilityObservation(
                revision: revision,
                kind: kind,
                observedAt: Date().timeIntervalSince1970,
                processIdentifier: processIdentifier,
                bundleIdentifier: bundleIdentifier,
                applicationName: application?.localizedName,
                windowTitle: windowTitle.value,
                windowFrame: windowFrame,
                focusedRole: role,
                focusedSubrole: subrole,
                focusedTitle: title.value,
                focusedValue: value.value,
                contextTitles: context.titles,
                secureField: secureField,
                nodesInspected: context.nodesInspected,
                truncated: value.truncated || title.truncated
                    || windowTitle.truncated || context.truncated
            )
        )
    }

    private func collectAncestorTitles(
        from element: AXUIElement?
    ) -> (titles: [String], nodesInspected: Int, truncated: Bool) {
        var titles: [String] = []
        var current = element
        var depth = 0
        var visited = Set<CFHashCode>()
        var truncated = false

        while let node = current {
            if depth > configuration.maximumDepth
                || visited.count >= configuration.maximumNodes
            {
                truncated = true
                break
            }
            let identifier = CFHash(node)
            guard visited.insert(identifier).inserted else { break }
            if let rawTitle = stringAttribute(kAXTitleAttribute, from: node) {
                let bounded = AccessibilityValueFilter.bounded(
                    rawTitle,
                    maximumLength: configuration.maximumTextLength
                )
                if let title = bounded.value, !title.isEmpty, !titles.contains(title) {
                    titles.append(title)
                }
                truncated = truncated || bounded.truncated
            }
            current = elementAttribute(kAXParentAttribute, from: node)
            depth += 1
        }
        return (titles, visited.count, truncated)
    }

    private func elementAttribute(
        _ name: String,
        from element: AXUIElement?
    ) -> AXUIElement? {
        guard let element else { return nil }
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            element,
            name as CFString,
            &value
        ) == .success else {
            return nil
        }
        guard let value, CFGetTypeID(value) == AXUIElementGetTypeID() else {
            return nil
        }
        return unsafeBitCast(value, to: AXUIElement.self)
    }

    private func stringAttribute(
        _ name: String,
        from element: AXUIElement?
    ) -> String? {
        guard let element else { return nil }
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            element,
            name as CFString,
            &value
        ) == .success else {
            return nil
        }
        return value as? String
    }

    private func accessibilityWindowFrame(
        from element: AXUIElement?
    ) -> AccessibilityWindowFrame? {
        guard let position = pointAttribute(
            kAXPositionAttribute,
            from: element
        ), let size = sizeAttribute(kAXSizeAttribute, from: element)
        else { return nil }
        return AccessibilityWindowFrame(
            x: Double(position.x),
            y: Double(position.y),
            width: Double(size.width),
            height: Double(size.height)
        )
    }

    private func pointAttribute(
        _ name: String,
        from element: AXUIElement?
    ) -> CGPoint? {
        guard let value = axValueAttribute(name, from: element),
              AXValueGetType(value) == .cgPoint
        else { return nil }
        var point = CGPoint.zero
        guard AXValueGetValue(value, .cgPoint, &point) else { return nil }
        return point
    }

    private func sizeAttribute(
        _ name: String,
        from element: AXUIElement?
    ) -> CGSize? {
        guard let value = axValueAttribute(name, from: element),
              AXValueGetType(value) == .cgSize
        else { return nil }
        var size = CGSize.zero
        guard AXValueGetValue(value, .cgSize, &size) else { return nil }
        return size
    }

    private func axValueAttribute(
        _ name: String,
        from element: AXUIElement?
    ) -> AXValue? {
        guard let element else { return nil }
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            element,
            name as CFString,
            &value
        ) == .success,
            let value,
            CFGetTypeID(value) == AXValueGetTypeID()
        else { return nil }
        return unsafeBitCast(value, to: AXValue.self)
    }

    private static func kind(for notification: String) -> String {
        switch notification {
        case kAXFocusedWindowChangedNotification:
            return "accessibility_window_changed"
        case kAXFocusedUIElementChangedNotification:
            return "accessibility_focus_changed"
        case kAXTitleChangedNotification:
            return "accessibility_title_changed"
        case kAXValueChangedNotification,
            kAXSelectedTextChangedNotification:
            return "accessibility_value_changed"
        default:
            return "accessibility_changed"
        }
    }

    private static let applicationNotifications = [
        kAXFocusedWindowChangedNotification as String,
        kAXFocusedUIElementChangedNotification as String,
        kAXWindowCreatedNotification as String,
        kAXTitleChangedNotification as String,
    ]

    private static let focusedElementNotifications = [
        kAXTitleChangedNotification as String,
        kAXValueChangedNotification as String,
        kAXSelectedTextChangedNotification as String,
    ]
}

enum AccessibilityNotificationPolicy {
    static func shouldDebounce(_ notification: String) -> Bool {
        notification == kAXValueChangedNotification
            || notification == kAXSelectedTextChangedNotification
    }
}

enum AccessibilityPrivateBrowsingPolicy {
    private enum BrowserFamily {
        case safari
        case chromium
        case firefox
    }

    static func shouldSuppress(
        bundleIdentifier: String?,
        windowTitle: String?
    ) -> Bool {
        guard let family = browserFamily(for: bundleIdentifier),
              let windowTitle
        else {
            return false
        }

        let title = normalized(windowTitle)
        guard !title.isEmpty else { return false }

        switch family {
        case .safari:
            return containsAny(title, indicators: safariIndicators)
        case .chromium:
            return containsAny(title, indicators: chromiumIndicators)
        case .firefox:
            return containsAny(title, indicators: firefoxIndicators)
        }
    }

    private static func browserFamily(
        for bundleIdentifier: String?
    ) -> BrowserFamily? {
        guard let bundleIdentifier else { return nil }
        let identifier = bundleIdentifier.lowercased()

        if safariBundleIdentifiers.contains(identifier) {
            return .safari
        }
        if chromiumBundleIdentifiers.contains(identifier) {
            return .chromium
        }
        if firefoxBundleIdentifiers.contains(identifier) {
            return .firefox
        }
        return nil
    }

    private static func normalized(_ value: String) -> String {
        value
            .precomposedStringWithCanonicalMapping
            .lowercased()
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
    }

    private static func containsAny(
        _ title: String,
        indicators: [String]
    ) -> Bool {
        indicators.contains(where: title.contains)
    }

    private static let safariBundleIdentifiers: Set<String> = [
        "com.apple.safari",
        "com.apple.safaritechnologypreview",
    ]

    private static let chromiumBundleIdentifiers: Set<String> = [
        "com.brave.browser",
        "com.brave.browser.beta",
        "com.brave.browser.dev",
        "com.brave.browser.nightly",
        "com.google.chrome",
        "com.google.chrome.beta",
        "com.google.chrome.canary",
        "com.google.chrome.dev",
        "com.microsoft.edgemac",
        "com.microsoft.edgemac.beta",
        "com.microsoft.edgemac.canary",
        "com.microsoft.edgemac.dev",
        "com.operasoftware.operagx",
        "com.operasoftware.opera",
        "com.vivaldi.vivaldi",
        "company.thebrowser.browser",
        "org.chromium.chromium",
    ]

    private static let firefoxBundleIdentifiers: Set<String> = [
        "org.mozilla.firefox",
        "org.mozilla.firefoxdeveloperedition",
        "org.mozilla.nightly",
    ]

    private static let safariIndicators = commonPrivateIndicators + [
        "private browsing",
        "private window",
        "隐私浏览",
        "隱私瀏覽",
        "私密浏览",
        "私密瀏覽",
        "私人浏览",
        "私人瀏覽",
        "プライベートブラウズ",
    ]

    private static let chromiumIndicators = commonPrivateIndicators + [
        "incognito",
        "inprivate",
        "无痕模式",
        "無痕模式",
        "隐身模式",
        "隱身模式",
        "シークレット モード",
        "シークレットモード",
    ]

    private static let firefoxIndicators = commonPrivateIndicators + [
        "private browsing",
        "private window",
        "隐私浏览",
        "隱私瀏覽",
        "私密浏览",
        "私密瀏覽",
        "プライベートブラウジング",
    ]

    private static let commonPrivateIndicators = [
        "无痕浏览",
        "無痕瀏覽",
        "navigation privée",
        "navegación privada",
        "navegacao privada",
        "navegação privada",
        "privates surfen",
    ]
}

@MainActor
final class LatestAccessibilityEventDebouncer {
    private let delayNanoseconds: UInt64
    private var task: Task<Void, Never>?

    init(delaySeconds: TimeInterval) {
        precondition(delaySeconds > 0 && delaySeconds.isFinite)
        delayNanoseconds = UInt64(delaySeconds * 1_000_000_000)
    }

    func schedule(_ action: @escaping @MainActor () -> Void) {
        task?.cancel()
        let delayNanoseconds = delayNanoseconds
        task = Task { @MainActor [weak self] in
            do {
                try await Task.sleep(nanoseconds: delayNanoseconds)
            } catch {
                return
            }
            guard !Task.isCancelled, let self else { return }
            self.task = nil
            action()
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
    }
}

private func accessibilityObserverCallback(
    _ observer: AXObserver,
    _ element: AXUIElement,
    _ notification: CFString,
    _ context: UnsafeMutableRawPointer?
) {
    _ = observer
    guard let context else { return }
    let monitor = Unmanaged<AccessibilityMonitor>
        .fromOpaque(context)
        .takeUnretainedValue()
    let notificationName = notification as String
    let transferredElement = AccessibilityElementTransfer(element: element)
    DispatchQueue.main.async {
        monitor.handleFromObserver(
            element: transferredElement.element,
            notification: notificationName
        )
    }
}

private struct AccessibilityElementTransfer: @unchecked Sendable {
    let element: AXUIElement
}

extension AccessibilityMonitor {
    fileprivate func handleFromObserver(
        element: AXUIElement,
        notification: String
    ) {
        handle(element: element, notification: notification)
    }
}

enum AccessibilityValueFilter {
    struct BoundedValue: Equatable {
        let value: String?
        let truncated: Bool
    }

    static func isSecureField(role: String?, subrole: String?) -> Bool {
        role == "AXSecureTextField"
            || subrole == (kAXSecureTextFieldSubrole as String)
    }

    static func bounded(_ value: String?, maximumLength: Int) -> BoundedValue {
        guard let value else { return BoundedValue(value: nil, truncated: false) }
        guard value.count > maximumLength else {
            return BoundedValue(value: value, truncated: false)
        }
        return BoundedValue(
            value: String(value.prefix(maximumLength)),
            truncated: true
        )
    }
}
