import AppKit
import ApplicationServices
import Foundation

@MainActor
final class NativeComputerHistoryAXProvider: ComputerHistoryAXProviding {
    private let workspace: NSWorkspace
    private let configuration: ComputerHistoryMonitorConfiguration
    private let worker: ComputerHistoryAXCaptureWorker
    private var cachedWindowPID: pid_t?
    private var cachedWindowElement: AXUIElement?
    private var cachedWindowToken: String?

    init(
        workspace: NSWorkspace = .shared,
        configuration: ComputerHistoryMonitorConfiguration
    ) {
        self.workspace = workspace
        self.configuration = configuration
        worker = ComputerHistoryAXCaptureWorker(configuration: configuration)
    }

    func currentContext() -> ComputerHistoryContext? {
        makeCaptureRequest()?.context
    }

    func capture(mode: ComputerHistoryAXMode) async -> ComputerHistoryAXCapture? {
        guard let request = makeCaptureRequest() else { return nil }
        return await worker.capture(request, mode: mode)
    }

    func target(at point: ComputerHistoryPoint?) -> ComputerHistoryResolvedTarget? {
        guard let request = makeCaptureRequest() else { return nil }
        let element: AXUIElement?
        if let point {
            var found: AXUIElement?
            let systemWide = AXUIElementCreateSystemWide()
            guard AXUIElementCopyElementAtPosition(
                systemWide,
                Float(point.x),
                Float(point.y),
                &found
            ) == .success,
                let found,
                Self.processIdentifier(of: found) == request.processIdentifier
            else { return nil }
            element = found
        } else {
            element = Self.elementAttribute(
                kAXFocusedUIElementAttribute,
                from: request.applicationElement
            )
        }
        return Self.describeTarget(element)
    }

    func currentFocusedTarget() -> ComputerHistoryResolvedTarget? {
        target(at: nil)
    }

    func currentSelection() -> ComputerHistorySelection? {
        guard let request = makeCaptureRequest(),
              let focused = Self.elementAttribute(
                  kAXFocusedUIElementAttribute,
                  from: request.applicationElement
              ),
              let resolved = Self.describeTarget(focused),
              !resolved.secure
        else { return nil }

        let rawText = Self.stringAttribute(kAXSelectedTextAttribute, from: focused)
        let bounded = ComputerHistoryWireText.payload(
            rawText,
            maximumBytes: 8 * 1_024,
            allowEmpty: true
        )
        return ComputerHistorySelection(
            text: bounded,
            truncated: (rawText?.utf8.count ?? 0) > 8 * 1_024,
            selectedRange: Self.selectedRange(from: focused),
            target: resolved.target
        )
    }

    func reset() {
        worker.reset()
    }

    private func makeCaptureRequest() -> ComputerHistoryAXCaptureRequest? {
        guard let application = workspace.frontmostApplication else { return nil }
        let pid = application.processIdentifier
        let applicationElement = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(
            applicationElement,
            Float(min(configuration.maximumAXDurationSeconds, 0.05))
        )
        let focusedWindow = Self.elementAttribute(
            kAXFocusedWindowAttribute,
            from: applicationElement
        )
        let rawTitle = Self.stringAttribute(kAXTitleAttribute, from: focusedWindow)
        guard !AccessibilityPrivateBrowsingPolicy.shouldSuppress(
            bundleIdentifier: application.bundleIdentifier,
            windowTitle: rawTitle
        ) else { return nil }

        let title = ComputerHistoryWireText.context(
            rawTitle,
            maximumCharacters: 512
        )
        let root = focusedWindow ?? applicationElement
        let url = ComputerHistoryWireText.context(
            Self.urlAttribute(kAXURLAttribute, from: focusedWindow),
            maximumCharacters: 2_048
        )
        let identifier = Self.stringAttribute(
            kAXIdentifierAttribute,
            from: focusedWindow
        )
        let fallbackWindowToken = stableWindowToken(
            processIdentifier: pid,
            window: focusedWindow
        )
        let context = ComputerHistoryContext(
            key: ComputerHistoryContextKey.make(
                processIdentifier: pid,
                windowIdentifier: identifier,
                fallbackWindowToken: fallbackWindowToken
            ),
            app: ComputerHistoryApplication(
                bundleIdentifier: ComputerHistoryWireText.bundleIdentifier(
                    application.bundleIdentifier
                ),
                name: ComputerHistoryWireText.context(
                    application.localizedName,
                    maximumCharacters: 200
                )
            ),
            window: ComputerHistoryWindow(title: title, url: url)
        )
        return ComputerHistoryAXCaptureRequest(
            context: context,
            processIdentifier: pid,
            applicationElement: applicationElement,
            rootElement: root
        )
    }

    private func stableWindowToken(
        processIdentifier: pid_t,
        window: AXUIElement?
    ) -> String {
        guard let window else { return "application" }
        if cachedWindowPID == processIdentifier,
           let cachedWindowElement,
           CFEqual(cachedWindowElement, window),
           let cachedWindowToken
        {
            return cachedWindowToken
        }
        let token = UUID().uuidString
        cachedWindowPID = processIdentifier
        cachedWindowElement = window
        cachedWindowToken = token
        return token
    }

    private static func describeTarget(
        _ element: AXUIElement?
    ) -> ComputerHistoryResolvedTarget? {
        guard let element,
              let role = stringAttribute(kAXRoleAttribute, from: element)
        else { return nil }
        let subrole = stringAttribute(kAXSubroleAttribute, from: element)
        let secure = AccessibilityValueFilter.isSecureField(
            role: role,
            subrole: subrole
        )
        guard let boundedRole = ComputerHistoryWireText.payload(
            role,
            maximumBytes: 128,
            allowEmpty: false
        ) else { return nil }
        let title = ComputerHistoryWireText.payload(
            stringAttribute(kAXTitleAttribute, from: element),
            maximumBytes: 512,
            allowEmpty: true
        )
        let description = ComputerHistoryWireText.payload(
            stringAttribute(kAXDescriptionAttribute, from: element),
            maximumBytes: 1_024,
            allowEmpty: true
        )
        return ComputerHistoryResolvedTarget(
            target: ComputerHistoryTarget(
                role: boundedRole,
                title: title,
                description: description
            ),
            secure: secure
        )
    }

    private static func selectedRange(
        from element: AXUIElement
    ) -> ComputerHistorySelectedRange? {
        guard let value = axValueAttribute(
            kAXSelectedTextRangeAttribute,
            from: element
        ), AXValueGetType(value) == .cfRange
        else { return nil }
        var range = CFRange()
        guard AXValueGetValue(value, .cfRange, &range),
              range.location >= 0,
              range.length >= 0
        else { return nil }
        return ComputerHistorySelectedRange(
            location: range.location,
            length: range.length
        )
    }

    nonisolated static func elementAttribute(
        _ name: String,
        from element: AXUIElement?
    ) -> AXUIElement? {
        guard let element else { return nil }
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            element,
            name as CFString,
            &value
        ) == .success,
            let value,
            CFGetTypeID(value) == AXUIElementGetTypeID()
        else { return nil }
        return unsafeBitCast(value, to: AXUIElement.self)
    }

    nonisolated static func stringAttribute(
        _ name: String,
        from element: AXUIElement?
    ) -> String? {
        guard let element else { return nil }
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            element,
            name as CFString,
            &value
        ) == .success,
            let value
        else { return nil }
        if let string = value as? String { return string }
        if let number = value as? NSNumber { return number.stringValue }
        return nil
    }

    nonisolated static func urlAttribute(
        _ name: String,
        from element: AXUIElement?
    ) -> String? {
        guard let element else { return nil }
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            element,
            name as CFString,
            &value
        ) == .success,
            let value
        else { return nil }
        if let url = value as? URL { return url.absoluteString }
        if let string = value as? String { return string }
        return nil
    }

    nonisolated static func axValueAttribute(
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

    nonisolated static func windowFrame(
        from element: AXUIElement?
    ) -> AccessibilityWindowFrame? {
        guard let position = axValueAttribute(kAXPositionAttribute, from: element),
              AXValueGetType(position) == .cgPoint,
              let size = axValueAttribute(kAXSizeAttribute, from: element),
              AXValueGetType(size) == .cgSize
        else { return nil }
        var point = CGPoint.zero
        var dimensions = CGSize.zero
        guard AXValueGetValue(position, .cgPoint, &point),
              AXValueGetValue(size, .cgSize, &dimensions)
        else { return nil }
        return AccessibilityWindowFrame(
            x: Double(point.x),
            y: Double(point.y),
            width: Double(dimensions.width),
            height: Double(dimensions.height)
        )
    }

    nonisolated private static func processIdentifier(
        of element: AXUIElement
    ) -> pid_t? {
        var pid: pid_t = 0
        guard AXUIElementGetPid(element, &pid) == .success else { return nil }
        return pid
    }
}

private struct ComputerHistoryAXCaptureRequest: @unchecked Sendable {
    let context: ComputerHistoryContext
    let processIdentifier: pid_t
    let applicationElement: AXUIElement
    let rootElement: AXUIElement
}

private final class ComputerHistoryAXCaptureWorker: @unchecked Sendable {
    private let configuration: ComputerHistoryMonitorConfiguration
    private let queue = DispatchQueue(
        label: "com.fishmemory.perception.computer-history-ax",
        qos: .utility
    )
    private let generationLock = NSLock()
    private var generation = 0
    private var state: ComputerHistoryAXSnapshotState
    private var baselineContextKey: String?

    init(configuration: ComputerHistoryMonitorConfiguration) {
        self.configuration = configuration
        state = ComputerHistoryAXSnapshotState(
            fullTreeMaximumBytes: configuration.fullTreeMaximumBytes,
            diffMaximumBytes: configuration.diffMaximumBytes
        )
    }

    func capture(
        _ request: ComputerHistoryAXCaptureRequest,
        mode: ComputerHistoryAXMode
    ) async -> ComputerHistoryAXCapture? {
        let expectedGeneration = currentGeneration()
        return await withCheckedContinuation {
            (continuation: CheckedContinuation<ComputerHistoryAXCapture?, Never>) in
            queue.async { [self] in
                guard isCurrent(expectedGeneration) else {
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: performCapture(request, mode: mode))
            }
        }
    }

    func reset() {
        generationLock.lock()
        generation &+= 1
        generationLock.unlock()
        queue.async { [self] in
            state.reset()
            baselineContextKey = nil
        }
    }

    private func currentGeneration() -> Int {
        generationLock.lock()
        defer { generationLock.unlock() }
        return generation
    }

    private func isCurrent(_ expectedGeneration: Int) -> Bool {
        generationLock.lock()
        defer { generationLock.unlock() }
        return generation == expectedGeneration
    }

    private func performCapture(
        _ request: ComputerHistoryAXCaptureRequest,
        mode: ComputerHistoryAXMode
    ) -> ComputerHistoryAXCapture? {
        if mode == .diffFromPrevious,
           baselineContextKey != request.context.key
        {
            state.reset()
            baselineContextKey = nil
            return nil
        }
        let tree = traverse(root: request.rootElement)
        let content: ComputerHistoryAXContent
        switch mode {
        case .fullTree:
            content = state.captureFullTree(
                nodes: tree.nodes,
                sourceTruncated: tree.truncated
            )
            baselineContextKey = request.context.key
        case .diffFromPrevious:
            content = state.captureDiff(
                nodes: tree.nodes,
                sourceTruncated: tree.truncated
            )
        }
        return ComputerHistoryAXCapture(context: request.context, content: content)
    }

    private func traverse(
        root: AXUIElement
    ) -> (nodes: [ComputerHistoryAXNode], truncated: Bool) {
        let startedAt = CFAbsoluteTimeGetCurrent()
        let rootCore = Self.coreAttributes(
            of: root,
            timeout: Self.timeout(
                startedAt: startedAt,
                maximumDuration: configuration.maximumAXDurationSeconds
            )
        )
        let rootFrame = rootCore.frame
        var stack: [(
            element: AXUIElement,
            path: String,
            depth: Int,
            prefetched: ComputerHistoryAXCoreAttributes?
        )] = [
            (root, "0", 0, rootCore),
        ]
        var visited = Set<CFHashCode>()
        var nodes: [ComputerHistoryAXNode] = []
        nodes.reserveCapacity(min(configuration.maximumAXNodes, 256))
        var truncated = false

        while let item = stack.popLast() {
            guard CFAbsoluteTimeGetCurrent() - startedAt
                < configuration.maximumAXDurationSeconds
            else {
                truncated = true
                break
            }
            guard visited.count < configuration.maximumAXNodes else {
                truncated = true
                break
            }
            let identity = CFHash(item.element)
            guard visited.insert(identity).inserted else { continue }
            let core = item.prefetched ?? Self.coreAttributes(
                of: item.element,
                timeout: Self.timeout(
                    startedAt: startedAt,
                    maximumDuration: configuration.maximumAXDurationSeconds
                )
            )
            guard !core.hidden else { continue }
            let visible = ComputerHistoryAXVisibilityPolicy.shouldInclude(
                hidden: false,
                frame: core.frame,
                rootFrame: rootFrame
            )
            let role = core.role ?? "AXUnknown"
            let subrole = core.subrole
            let secure = AccessibilityValueFilter.isSecureField(
                role: role,
                subrole: subrole
            )
            let text: ComputerHistoryAXTextAttributes
            if !visible || secure {
                text = ComputerHistoryAXTextAttributes(
                    value: nil,
                    selectedText: nil,
                    truncated: false
                )
            } else if CFAbsoluteTimeGetCurrent() - startedAt
                < configuration.maximumAXDurationSeconds
            {
                text = Self.textAttributes(
                    of: item.element,
                    timeout: Self.timeout(
                        startedAt: startedAt,
                        maximumDuration: configuration.maximumAXDurationSeconds
                    )
                )
            } else {
                text = ComputerHistoryAXTextAttributes(
                    value: nil,
                    selectedText: nil,
                    truncated: true
                )
            }
            if visible {
                nodes.append(ComputerHistoryAXNode(
                    path: item.path,
                    role: role,
                    subrole: subrole,
                    title: core.title,
                    description: core.description,
                    value: text.value,
                    url: core.url,
                    selectedText: text.selectedText,
                    frame: core.frame
                ))
            }
            truncated = truncated || core.truncated || text.truncated

            guard CFAbsoluteTimeGetCurrent() - startedAt
                < configuration.maximumAXDurationSeconds
            else {
                truncated = true
                break
            }

            guard item.depth < configuration.maximumAXDepth else {
                truncated = truncated || !core.children.isEmpty
                continue
            }
            for (index, child) in core.children.enumerated().reversed() {
                stack.append((
                    child,
                    "\(item.path)/\(index)",
                    item.depth + 1,
                    nil
                ))
            }
        }
        if !stack.isEmpty { truncated = true }
        return (nodes, truncated)
    }

    private static func coreAttributes(
        of element: AXUIElement,
        timeout: Float
    ) -> ComputerHistoryAXCoreAttributes {
        let values = copyMultiple(
            [
                kAXRoleAttribute,
                kAXSubroleAttribute,
                kAXTitleAttribute,
                kAXDescriptionAttribute,
                kAXURLAttribute,
                kAXPositionAttribute,
                kAXSizeAttribute,
                kAXHiddenAttribute,
                kAXChildrenAttribute,
            ],
            from: element,
            timeout: timeout
        )
        let title = bounded(string(values[safe: 2]))
        let description = bounded(string(values[safe: 3]))
        let url = bounded(url(values[safe: 4]))
        return ComputerHistoryAXCoreAttributes(
            role: string(values[safe: 0]),
            subrole: string(values[safe: 1]),
            title: title.value,
            description: description.value,
            url: url.value,
            frame: frame(position: values[safe: 5], size: values[safe: 6]),
            hidden: boolean(values[safe: 7]) ?? false,
            children: elements(values[safe: 8]),
            truncated: title.truncated || description.truncated || url.truncated
        )
    }

    private static func textAttributes(
        of element: AXUIElement,
        timeout: Float
    ) -> ComputerHistoryAXTextAttributes {
        let values = copyMultiple(
            [kAXValueAttribute, kAXSelectedTextAttribute],
            from: element,
            timeout: timeout
        )
        let value = bounded(string(values[safe: 0]))
        let selected = bounded(string(values[safe: 1]))
        return ComputerHistoryAXTextAttributes(
            value: value.value,
            selectedText: selected.value,
            truncated: value.truncated || selected.truncated
        )
    }

    private static func copyMultiple(
        _ attributes: [String],
        from element: AXUIElement,
        timeout: Float
    ) -> [AnyObject] {
        AXUIElementSetMessagingTimeout(element, max(timeout, 0.001))
        var copied: CFArray?
        guard AXUIElementCopyMultipleAttributeValues(
            element,
            attributes as CFArray,
            AXCopyMultipleAttributeOptions(rawValue: 0),
            &copied
        ) == .success,
            let copied
        else { return [] }
        return copied as [AnyObject]
    }

    private static func bounded(
        _ value: String?
    ) -> AccessibilityValueFilter.BoundedValue {
        AccessibilityValueFilter.bounded(value, maximumLength: 4_096)
    }

    private static func string(_ value: AnyObject?) -> String? {
        if let string = value as? String { return string }
        if let number = value as? NSNumber { return number.stringValue }
        return nil
    }

    private static func url(_ value: AnyObject?) -> String? {
        if let value = value as? URL { return value.absoluteString }
        return value as? String
    }

    private static func boolean(_ value: AnyObject?) -> Bool? {
        (value as? NSNumber)?.boolValue
    }

    private static func elements(_ value: AnyObject?) -> [AXUIElement] {
        value as? [AXUIElement] ?? []
    }

    private static func frame(
        position: AnyObject?,
        size: AnyObject?
    ) -> AccessibilityWindowFrame? {
        guard let position = axValue(position),
              AXValueGetType(position) == .cgPoint,
              let size = axValue(size),
              AXValueGetType(size) == .cgSize
        else { return nil }
        var point = CGPoint.zero
        var dimensions = CGSize.zero
        guard AXValueGetValue(position, .cgPoint, &point),
              AXValueGetValue(size, .cgSize, &dimensions)
        else { return nil }
        return AccessibilityWindowFrame(
            x: Double(point.x),
            y: Double(point.y),
            width: Double(dimensions.width),
            height: Double(dimensions.height)
        )
    }

    private static func axValue(_ value: AnyObject?) -> AXValue? {
        guard let value,
              CFGetTypeID(value) == AXValueGetTypeID()
        else { return nil }
        return unsafeBitCast(value, to: AXValue.self)
    }

    private static func timeout(
        startedAt: CFAbsoluteTime,
        maximumDuration: TimeInterval
    ) -> Float {
        let remaining = maximumDuration - (CFAbsoluteTimeGetCurrent() - startedAt)
        return Float(max(0.001, min(0.02, remaining)))
    }
}

private struct ComputerHistoryAXCoreAttributes {
    let role: String?
    let subrole: String?
    let title: String?
    let description: String?
    let url: String?
    let frame: AccessibilityWindowFrame?
    let hidden: Bool
    let children: [AXUIElement]
    let truncated: Bool
}

private struct ComputerHistoryAXTextAttributes {
    let value: String?
    let selectedText: String?
    let truncated: Bool
}

private extension Array {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
