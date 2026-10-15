import XCTest
@testable import PerceptionCore

final class ComputerHistoryTests: XCTestCase {
    func testDefaultHistoryLimitsMatchResourceBoundaries() {
        let configuration = ComputerHistoryMonitorConfiguration()

        XCTAssertEqual(configuration.maximumQueuedEvents, 128)
        XCTAssertEqual(configuration.coalescingWindowSeconds, 0.1)
        XCTAssertEqual(configuration.actionDiffDelaySeconds, 0.12)
        XCTAssertEqual(configuration.maximumAXNodes, 1_024)
        XCTAssertEqual(configuration.maximumAXDepth, 20)
        XCTAssertEqual(configuration.maximumAXDurationSeconds, 0.25)
        XCTAssertEqual(configuration.fullTreeMaximumBytes, 64 * 1_024)
        XCTAssertEqual(configuration.diffMaximumBytes, 32 * 1_024)
    }

    func testEventEncodesStrictInteractionUnionWithoutNullPlaceholders() throws {
        let event = ComputerHistoryEvent(
            id: "836B635F-A95D-4668-857B-2F4A45808376",
            kind: .keyboardShortcut,
            timestamp: 123.5,
            app: ComputerHistoryApplication(
                bundleIdentifier: "com.example.editor",
                name: "Editor"
            ),
            window: ComputerHistoryWindow(title: "Document", url: nil),
            ax: ComputerHistoryAXContent(
                mode: .diffFromPrevious,
                text: "+ 2. button — Save",
                truncated: false
            ),
            keyboard: ComputerHistoryKeyboard(
                keyEquivalent: "s",
                modifiers: ["command"],
                target: ComputerHistoryTarget(
                    role: "AXTextArea",
                    title: "Editor",
                    description: "Source editor"
                )
            )
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(event))
                as? [String: Any]
        )
        XCTAssertEqual(
            Set(object.keys),
            Set(["id", "kind", "timestamp", "app", "window", "ax", "keyboard"])
        )
        XCTAssertEqual(object["kind"] as? String, "keyboard.shortcut")
        XCTAssertEqual(object["timestamp"] as? Double, 123.5)
        XCTAssertNil(object["mouse"])
        XCTAssertNil(object["selection"])

        let keyboard = try XCTUnwrap(object["keyboard"] as? [String: Any])
        XCTAssertEqual(keyboard["keyEquivalent"] as? String, "s")
        XCTAssertEqual(keyboard["modifiers"] as? [String], ["command"])
        XCTAssertNil(keyboard["text"])
        let target = try XCTUnwrap(keyboard["target"] as? [String: Any])
        XCTAssertEqual(target["role"] as? String, "AXTextArea")
    }

    func testSelectionCarriesRangeAndTarget() throws {
        let selection = ComputerHistorySelection(
            text: "selected",
            truncated: false,
            selectedRange: ComputerHistorySelectedRange(location: 4, length: 8),
            target: ComputerHistoryTarget(
                role: "AXTextArea",
                title: nil,
                description: nil
            )
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(selection))
                as? [String: Any]
        )
        let range = try XCTUnwrap(object["selectedRange"] as? [String: Any])
        XCTAssertEqual(range["location"] as? Int, 4)
        XCTAssertEqual(range["length"] as? Int, 8)
    }

    func testAXStateProducesFullTreeThenBoundedDiff() {
        var state = ComputerHistoryAXSnapshotState(
            fullTreeMaximumBytes: 256,
            diffMaximumBytes: 64
        )
        let initial = [
            ComputerHistoryAXNode(
                path: "0",
                role: "AXWindow",
                title: "Document",
                description: nil,
                value: nil
            ),
            ComputerHistoryAXNode(
                path: "0/0",
                role: "AXButton",
                title: "Save",
                description: nil,
                value: nil
            ),
        ]

        let full = state.captureFullTree(nodes: initial, sourceTruncated: false)
        XCTAssertEqual(full.mode, .fullTree)
        XCTAssertEqual(full.text, "1. window — Document\n  2. button — Save")
        XCTAssertLessThanOrEqual(full.text.utf8.count, 256)
        XCTAssertFalse(full.truncated)

        var changed = initial
        changed[1] = ComputerHistoryAXNode(
            path: "0/0",
            role: "AXButton",
            title: String(repeating: "Save now ", count: 20),
            description: nil,
            value: nil
        )
        let diff = state.captureDiff(nodes: changed, sourceTruncated: false)
        XCTAssertEqual(diff.mode, .diffFromPrevious)
        XCTAssertTrue(diff.text.contains("~ 2. button — Save now"))
        XCTAssertFalse(diff.text.contains("0/0"))
        XCTAssertLessThanOrEqual(diff.text.utf8.count, 64)
        XCTAssertTrue(diff.truncated)
    }

    func testAXTreeCollapsesStructuralNodesAndHidesInternalGeometryAndPaths() {
        var state = ComputerHistoryAXSnapshotState()
        let nodes = [
            ComputerHistoryAXNode(
                path: "0",
                role: "AXGroup",
                title: nil,
                description: nil,
                value: nil
            ),
            ComputerHistoryAXNode(
                path: "0/0",
                role: "AXWindow",
                title: "Document",
                description: nil,
                value: nil
            ),
            ComputerHistoryAXNode(
                path: "0/0/0",
                role: "AXGroup",
                title: nil,
                description: nil,
                value: nil
            ),
            ComputerHistoryAXNode(
                path: "0/0/0/2",
                role: "AXTextField",
                subrole: "AXSearchField",
                title: "Search",
                description: "Site search",
                value: "query",
                url: "https://example.test",
                selectedText: "que",
                frame: AccessibilityWindowFrame(x: 1, y: 2, width: 300, height: 40)
            ),
        ]

        let text = state.captureFullTree(nodes: nodes, sourceTruncated: false).text

        XCTAssertEqual(
            text,
            "1. window — Document\n"
                + "  2. search field — Search | query | Site search | "
                + "https://example.test | selected: que"
        )
        XCTAssertFalse(text.contains("AXGroup"))
        XCTAssertFalse(text.contains("0/0"))
        XCTAssertFalse(text.contains("frame"))
        XCTAssertFalse(text.contains("role="))
    }

    func testAXVisibilityRejectsHiddenAndOffWindowNodes() {
        let root = AccessibilityWindowFrame(x: 0, y: 0, width: 800, height: 600)

        XCTAssertFalse(ComputerHistoryAXVisibilityPolicy.shouldInclude(
            hidden: true,
            frame: nil,
            rootFrame: root
        ))
        XCTAssertFalse(ComputerHistoryAXVisibilityPolicy.shouldInclude(
            hidden: false,
            frame: .init(x: 900, y: 0, width: 100, height: 100),
            rootFrame: root
        ))
        XCTAssertTrue(ComputerHistoryAXVisibilityPolicy.shouldInclude(
            hidden: false,
            frame: .init(x: 799, y: 599, width: 10, height: 10),
            rootFrame: root
        ))
    }

    func testContextKeyPrefersAXIdentifierAndOtherwiseUsesStableWindowToken() {
        XCTAssertEqual(
            ComputerHistoryContextKey.make(
                processIdentifier: 7,
                windowIdentifier: "window-1",
                fallbackWindowToken: "fallback-a"
            ),
            ComputerHistoryContextKey.make(
                processIdentifier: 7,
                windowIdentifier: "window-1",
                fallbackWindowToken: "fallback-b"
            )
        )
        XCTAssertNotEqual(
            ComputerHistoryContextKey.make(
                processIdentifier: 7,
                windowIdentifier: nil,
                fallbackWindowToken: "fallback-a"
            ),
            ComputerHistoryContextKey.make(
                processIdentifier: 7,
                windowIdentifier: nil,
                fallbackWindowToken: "fallback-b"
            )
        )
    }

    func testWireTextKeepsNativeEventsInsideStrictCrossLayerBounds() {
        XCTAssertEqual(
            ComputerHistoryWireText.bundleIdentifier("COM.Example.Editor"),
            "com.example.editor"
        )
        XCTAssertNil(ComputerHistoryWireText.bundleIdentifier("invalid bundle"))
        XCTAssertEqual(
            ComputerHistoryWireText.context("  Line\nTitle  ", maximumCharacters: 32),
            "Line Title"
        )
        let bounded = ComputerHistoryWireText.payload(
            String(repeating: "界", count: 3_000),
            maximumBytes: 8 * 1_024,
            allowEmpty: true
        )
        XCTAssertLessThanOrEqual(bounded?.utf8.count ?? .max, 8 * 1_024)
    }

    func testAXStateDiffReportsDeletionAndResetRemovesBaseline() {
        var state = ComputerHistoryAXSnapshotState()
        let nodes = [
            ComputerHistoryAXNode(
                path: "0/0",
                role: "AXStaticText",
                title: nil,
                description: nil,
                value: "Before"
            ),
        ]
        _ = state.captureFullTree(nodes: nodes, sourceTruncated: false)

        let diff = state.captureDiff(nodes: [], sourceTruncated: false)
        XCTAssertEqual(diff.text, "- 1. static text — Before")
        XCTAssertFalse(diff.text.contains("0/0"))

        state.reset()
        let afterReset = state.captureDiff(nodes: nodes, sourceTruncated: false)
        XCTAssertEqual(afterReset.mode, .diffFromPrevious)
        XCTAssertEqual(afterReset.text, "+ 1. static text — Before")
    }

    func testAXDiffIgnoresGeometryOnlyChanges() {
        var state = ComputerHistoryAXSnapshotState()
        let before = ComputerHistoryAXNode(
            path: "0/0",
            role: "AXButton",
            title: "Save",
            description: nil,
            value: nil,
            frame: .init(x: 1, y: 2, width: 30, height: 20)
        )
        _ = state.captureFullTree(nodes: [before], sourceTruncated: false)
        let after = ComputerHistoryAXNode(
            path: "0/0",
            role: "AXButton",
            title: "Save",
            description: nil,
            value: nil,
            frame: .init(x: 100, y: 200, width: 30, height: 20)
        )

        XCTAssertEqual(
            state.captureDiff(nodes: [after], sourceTruncated: false).text,
            ""
        )
    }

    func testTruncatedAXDiffDoesNotInventDeletionOrDiscardBaselineTail() {
        var state = ComputerHistoryAXSnapshotState()
        let first = ComputerHistoryAXNode(
            path: "0/0",
            role: "AXStaticText",
            title: nil,
            description: nil,
            value: "First"
        )
        let tail = ComputerHistoryAXNode(
            path: "0/1",
            role: "AXStaticText",
            title: nil,
            description: nil,
            value: "Tail"
        )
        _ = state.captureFullTree(nodes: [first, tail], sourceTruncated: false)

        let uncertain = state.captureDiff(nodes: [first], sourceTruncated: true)
        XCTAssertEqual(uncertain.text, "")
        XCTAssertTrue(uncertain.truncated)

        let confirmed = state.captureDiff(nodes: [first], sourceTruncated: false)
        XCTAssertEqual(confirmed.text, "- 2. static text — Tail")
    }

    func testAXSemanticNodeNumbersStayStableAcrossIncrementalInsertions() {
        var state = ComputerHistoryAXSnapshotState()
        let original = [
            ComputerHistoryAXNode(
                path: "0/1",
                role: "AXButton",
                title: "Save",
                description: nil,
                value: nil
            ),
            ComputerHistoryAXNode(
                path: "0/2",
                role: "AXButton",
                title: "Close",
                description: nil,
                value: nil
            ),
        ]
        XCTAssertEqual(
            state.captureFullTree(nodes: original, sourceTruncated: false).text,
            "1. button — Save\n2. button — Close"
        )
        let inserted = ComputerHistoryAXNode(
            path: "0/0",
            role: "AXButton",
            title: "Open",
            description: nil,
            value: nil
        )

        let diff = state.captureDiff(
            nodes: [inserted] + original,
            sourceTruncated: false
        )

        XCTAssertEqual(diff.text, "+ 3. button — Open")
        XCTAssertFalse(diff.text.contains("1. button — Save"))
        XCTAssertFalse(diff.text.contains("2. button — Close"))
    }

    func testSelectionGateRequiresMeaningAndDeduplicatesWithinContext() {
        var gate = ComputerHistorySelectionGate()
        let target = ComputerHistoryTarget(
            role: "AXTextArea",
            title: nil,
            description: nil
        )
        let caret = ComputerHistorySelection(
            text: nil,
            truncated: false,
            selectedRange: .init(location: 5, length: 0),
            target: target
        )
        XCTAssertFalse(gate.shouldEmit(contextKey: "a", selection: caret))

        let selected = ComputerHistorySelection(
            text: "hello",
            truncated: false,
            selectedRange: .init(location: 5, length: 5),
            target: target
        )
        XCTAssertTrue(gate.shouldEmit(contextKey: "a", selection: selected))
        XCTAssertFalse(gate.shouldEmit(contextKey: "a", selection: selected))

        let movedRange = ComputerHistorySelection(
            text: nil,
            truncated: false,
            selectedRange: .init(location: 20, length: 3),
            target: target
        )
        XCTAssertTrue(gate.shouldEmit(contextKey: "a", selection: movedRange))
        XCTAssertFalse(gate.shouldEmit(contextKey: "a", selection: caret))
        XCTAssertFalse(gate.shouldEmit(contextKey: "a", selection: caret))
        XCTAssertTrue(gate.shouldEmit(contextKey: "a", selection: selected))
        XCTAssertFalse(gate.shouldEmit(contextKey: "b", selection: caret))
    }

    func testEventBufferCoalescesSameKindWithinOneHundredMilliseconds() {
        var buffer = ComputerHistoryEventBuffer(
            maximumCount: 128,
            coalescingWindowSeconds: 0.1
        )
        buffer.enqueue(Self.textEvent(id: "one", timestamp: 10, text: "hel"))
        buffer.enqueue(Self.textEvent(id: "two", timestamp: 10.08, text: "lo"))

        let events = buffer.drain()
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events[0].id, "one")
        XCTAssertEqual(events[0].timestamp, 10.08)
        XCTAssertEqual(events[0].keyboard?.text, "hello")
    }

    func testEventBufferDoesNotLetDeselectionEraseRecentSelectedText() {
        var buffer = ComputerHistoryEventBuffer()
        let app = ComputerHistoryApplication(
            bundleIdentifier: "com.example.editor",
            name: "Editor"
        )
        let window = ComputerHistoryWindow(title: "Document", url: nil)
        let target = ComputerHistoryTarget(
            role: "AXTextArea",
            title: nil,
            description: nil
        )
        buffer.enqueue(ComputerHistoryEvent(
            id: "selected",
            kind: .selectionChanged,
            timestamp: 10,
            app: app,
            window: window,
            selection: .init(
                text: "hello",
                truncated: false,
                selectedRange: .init(location: 1, length: 5),
                target: target
            )
        ))
        buffer.enqueue(ComputerHistoryEvent(
            id: "cleared",
            kind: .selectionChanged,
            timestamp: 10.05,
            app: app,
            window: window,
            selection: .init(
                text: nil,
                truncated: false,
                selectedRange: .init(location: 6, length: 0),
                target: target
            )
        ))

        let events = buffer.drain()
        XCTAssertEqual(events.map(\.id), ["selected", "cleared"])
    }

    func testEventBufferKeepsDifferentKindsAndCapsAtOneHundredTwentyEight() {
        var buffer = ComputerHistoryEventBuffer(
            maximumCount: 128,
            coalescingWindowSeconds: 0.1
        )
        for index in 0 ..< 130 {
            buffer.enqueue(ComputerHistoryEvent(
                id: "event-\(index)",
                kind: index.isMultiple(of: 2) ? .mouseClick : .keyboardShortcut,
                timestamp: Double(index),
                app: .init(bundleIdentifier: nil, name: nil),
                window: .init(title: nil, url: nil)
            ))
        }

        let events = buffer.drain()
        XCTAssertEqual(events.count, 128)
        XCTAssertEqual(events.first?.id, "event-2")
        XCTAssertEqual(events.last?.id, "event-129")
        XCTAssertEqual(buffer.suppressedEventCount, 2)
        XCTAssertEqual(buffer.takeSuppressedEventCount(), 2)
        XCTAssertEqual(buffer.suppressedEventCount, 0)
    }

    func testSuppressionCountIsFullyChunkedAtWireMaximum() {
        let chunks = ComputerHistorySuppressionChunking.chunks(for: 2_000_001)

        XCTAssertEqual(chunks, [1_000_000, 1_000_000, 1])
        XCTAssertEqual(chunks.reduce(0, +), 2_000_001)
        XCTAssertTrue(chunks.allSatisfy { 1 ... 1_000_000 ~= $0 })
        XCTAssertEqual(ComputerHistorySuppressionChunking.chunks(for: 0), [])
    }

    func testEventBufferNeverCoalescesAcrossContextOrTextTargets() {
        var buffer = ComputerHistoryEventBuffer()
        buffer.enqueue(Self.textEvent(id: "one", timestamp: 10, text: "a"))
        buffer.enqueue(ComputerHistoryEvent(
            id: "two",
            kind: .keyboardTextInput,
            timestamp: 10.01,
            app: .init(bundleIdentifier: "com.example.other", name: "Other"),
            window: .init(title: "Document", url: nil),
            keyboard: .init(text: "b", target: .init(
                role: "AXTextArea",
                title: nil,
                description: nil
            ))
        ))
        buffer.enqueue(ComputerHistoryEvent(
            id: "three",
            kind: .keyboardTextInput,
            timestamp: 10.02,
            app: .init(bundleIdentifier: "com.example.editor", name: "Editor"),
            window: .init(title: "Document", url: nil),
            keyboard: .init(text: "c", target: .init(
                role: "AXButton",
                title: "Search",
                description: nil
            ))
        ))

        XCTAssertEqual(buffer.drain().count, 3)
    }

    func testAllNineKindsHaveStableWireValues() {
        XCTAssertEqual(
            ComputerHistoryEventKind.allCases.map(\.rawValue),
            [
                "session.started",
                "window.changed",
                "mouse.click",
                "mouse.context_menu",
                "mouse.drag",
                "selection.changed",
                "keyboard.shortcut",
                "keyboard.submit",
                "keyboard.text_input",
            ]
        )
    }

    private static func textEvent(
        id: String,
        timestamp: Double,
        text: String
    ) -> ComputerHistoryEvent {
        ComputerHistoryEvent(
            id: id,
            kind: .keyboardTextInput,
            timestamp: timestamp,
            app: .init(bundleIdentifier: "com.example.editor", name: "Editor"),
            window: .init(title: "Document", url: nil),
            keyboard: .init(text: text)
        )
    }
}

final class ComputerHistoryMonitorTests: XCTestCase {
    @MainActor
    func testCurrentReportSeparatesAXTrustFromInputMonitoringHealth() {
        let listener = TestComputerHistoryListener()
        listener.inputReport = ComputerHistoryInputReport(
            accessibilityPermission: .granted,
            listenEventPermission: .denied,
            eventTapState: .permissionRequired,
            mouse: .init(state: .unavailable, receivedCount: 0, lastEventAt: nil),
            keyboard: .init(state: .unavailable, receivedCount: 0, lastEventAt: nil),
            tapDisabledCount: 0
        )
        let monitor = ComputerHistoryMonitor(
            trustCheck: { true },
            listener: listener,
            axProvider: TestComputerHistoryAXProvider(),
            onEvent: { _ in }
        )

        let degraded = monitor.start()
        XCTAssertEqual(degraded.permission, .granted)
        XCTAssertEqual(degraded.state, .unavailable)
        XCTAssertEqual(degraded.reason, "input_monitoring_permission_required")
        XCTAssertEqual(
            degraded.computerHistoryInput?.accessibilityPermission,
            .granted
        )
        XCTAssertEqual(
            degraded.computerHistoryInput?.listenEventPermission,
            .denied
        )

        listener.inputReport = ComputerHistoryInputReport(
            accessibilityPermission: .granted,
            listenEventPermission: .granted,
            eventTapState: .running,
            mouse: .init(state: .running, receivedCount: 1, lastEventAt: 10),
            keyboard: .init(state: .running, receivedCount: 2, lastEventAt: 11),
            tapDisabledCount: 0
        )
        XCTAssertEqual(monitor.currentReport().state, .running)
        XCTAssertEqual(
            monitor.currentReport().computerHistoryInput?.keyboard.receivedCount,
            2
        )

        monitor.stop()
        XCTAssertEqual(monitor.currentReport().state, .disabled)
    }

    @MainActor
    func testStartEmitsSessionFullTreeAndStopDetachesListener() async {
        let listener = TestComputerHistoryListener()
        let provider = TestComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )

        let report = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()

        XCTAssertEqual(report.state, .running)
        XCTAssertTrue(listener.started)
        XCTAssertEqual(events.map(\.kind), [.sessionStarted])
        XCTAssertEqual(events.first?.ax?.mode, .fullTree)

        monitor.stop()
        XCTAssertTrue(listener.stopped)
        XCTAssertEqual(provider.resetCount, 1)
    }

    @MainActor
    func testTextAndSelectionDoNotForceAXTreeRescanAndSecureTextIsSuppressed() async {
        let listener = TestComputerHistoryListener()
        let provider = TestComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(coalescingWindowSeconds: 0),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()
        provider.captureModes.removeAll()

        listener.emit(.keyboardTextInput(timestamp: 2, text: "hi"))
        listener.emit(.selectionChanged(timestamp: 3))
        monitor.flushPendingEvents()

        XCTAssertEqual(events.map(\.kind), [.keyboardTextInput, .selectionChanged])
        XCTAssertEqual(provider.captureModes, [])
        XCTAssertEqual(events[0].keyboard?.target?.role, "AXTextArea")
        XCTAssertEqual(events[1].selection?.selectedRange?.length, 4)

        listener.emit(.selectionChanged(timestamp: 3.5))
        monitor.flushPendingEvents()
        XCTAssertEqual(events.count, 2)

        provider.selection = ComputerHistorySelection(
            text: nil,
            truncated: false,
            selectedRange: .init(location: 20, length: 0),
            target: provider.focusedTarget.target
        )
        listener.emit(.selectionChanged(timestamp: 3.6))
        monitor.flushPendingEvents()
        XCTAssertEqual(events.map(\.kind), [.keyboardTextInput, .selectionChanged])
        listener.emit(.selectionChanged(timestamp: 3.7))
        monitor.flushPendingEvents()
        XCTAssertEqual(events.count, 2)

        provider.focusedTarget = ComputerHistoryResolvedTarget(
            target: .init(role: "AXSecureTextField", title: nil, description: nil),
            secure: true
        )
        listener.emit(.keyboardTextInput(timestamp: 4, text: "secret"))
        monitor.flushPendingEvents()
        XCTAssertEqual(events.count, 2)
    }

    @MainActor
    func testClickGetsDelayedDiffAndPrivacyBoundaryDropsPendingWork() async {
        let listener = TestComputerHistoryListener()
        let provider = TestComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(actionDiffDelaySeconds: 0),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()

        listener.emit(.mouseClick(
            timestamp: 4,
            button: .left,
            clickCount: 1,
            location: .init(x: 10, y: 20)
        ))
        for _ in 0 ..< 20 {
            if provider.captureModes.contains(.diffFromPrevious) { break }
            await Task.yield()
        }
        monitor.flushPendingEvents()

        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events.first?.kind, .mouseClick)
        XCTAssertEqual(events.first?.ax?.mode, .diffFromPrevious)
        XCTAssertEqual(events.first?.mouse?.location, .init(x: 10, y: 20))

        events.removeAll()
        let delayedListener = TestComputerHistoryListener()
        let delayed = ComputerHistoryMonitor(
            configuration: .init(actionDiffDelaySeconds: 10),
            trustCheck: { true },
            listener: delayedListener,
            axProvider: TestComputerHistoryAXProvider(),
            onEvent: { events.append($0) }
        )
        _ = delayed.start()
        await Task.yield()
        await Task.yield()
        delayed.flushPendingEvents()
        events.removeAll()
        delayedListener.emit(.keyboardShortcut(
            timestamp: 5,
            keyEquivalent: "s",
            modifiers: ["command"]
        ))
        delayedListener.emit(.privacyBoundary)
        delayed.flushPendingEvents()
        XCTAssertTrue(events.isEmpty)
        delayed.stop()
    }

    @MainActor
    func testReturnAndEnterArePersistedAsSubmitInsteadOfShortcut() async {
        let listener = TestComputerHistoryListener()
        let provider = TestComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(actionDiffDelaySeconds: 0),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()

        listener.emit(.keyboardSubmit(
            timestamp: 6,
            keyEquivalent: "return",
            modifiers: []
        ))
        for _ in 0 ..< 30 {
            if provider.captureModes.contains(.diffFromPrevious) { break }
            await Task.yield()
        }
        await Task.yield()
        monitor.flushPendingEvents()

        XCTAssertEqual(events.map(\.kind), [.keyboardSubmit])
        XCTAssertEqual(events.first?.keyboard?.keyEquivalent, "return")
        XCTAssertEqual(events.first?.keyboard?.target?.role, "AXTextArea")
        monitor.stop()
    }

    @MainActor
    func testOnlyActualFocusContextChangeResetsBaselineAndUsesFullTree() async {
        let listener = TestComputerHistoryListener()
        let provider = TestComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(
                coalescingWindowSeconds: 0.01,
                actionDiffDelaySeconds: 0
            ),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()

        listener.emit(.focusedWindowChanged(timestamp: 6))
        listener.emit(.focusedWindowChanged(timestamp: 7))
        try? await Task.sleep(nanoseconds: 20_000_000)
        monitor.flushPendingEvents()
        XCTAssertEqual(events.map(\.kind), [.sessionStarted])
        XCTAssertEqual(provider.resetCount, 0)
        XCTAssertEqual(provider.captureModes, [.fullTree])

        listener.emit(.windowContentChanged(timestamp: 8))
        listener.emit(.windowContentChanged(timestamp: 9))
        try? await Task.sleep(nanoseconds: 20_000_000)
        for _ in 0 ..< 50 {
            if provider.captureModes.count == 2 { break }
            await Task.yield()
        }
        await Task.yield()
        monitor.flushPendingEvents()

        XCTAssertEqual(events.map(\.kind), [.sessionStarted, .windowChanged])
        XCTAssertEqual(events.last?.timestamp, 9)
        XCTAssertEqual(events.last?.ax?.mode, .diffFromPrevious)
        XCTAssertEqual(provider.resetCount, 0)
        XCTAssertEqual(provider.captureModes, [.fullTree, .diffFromPrevious])

        provider.context = ComputerHistoryContext(
            key: "7:other-window",
            app: provider.context.app,
            window: .init(title: "Other", url: nil)
        )
        listener.emit(.focusedWindowChanged(timestamp: 10))
        listener.emit(.focusedWindowChanged(timestamp: 11))
        try? await Task.sleep(nanoseconds: 20_000_000)
        for _ in 0 ..< 50 {
            if provider.captureModes.count == 3 { break }
            await Task.yield()
        }
        await Task.yield()
        monitor.flushPendingEvents()

        XCTAssertEqual(events.map(\.kind), [
            .sessionStarted,
            .windowChanged,
            .windowChanged,
        ])
        XCTAssertEqual(events.last?.timestamp, 10)
        XCTAssertEqual(events.last?.ax?.mode, .fullTree)
        XCTAssertEqual(provider.resetCount, 1)
        XCTAssertEqual(provider.captureModes, [
            .fullTree,
            .diffFromPrevious,
            .fullTree,
        ])
    }

    @MainActor
    func testTitleMetadataChangeEmitsWithoutSemanticAXNoise() async {
        let listener = TestComputerHistoryListener()
        let provider = TestComputerHistoryAXProvider()
        provider.diffText = ""
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(coalescingWindowSeconds: 0),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()

        provider.context = ComputerHistoryContext(
            key: provider.context.key,
            app: provider.context.app,
            window: .init(title: "Renamed", url: nil)
        )
        listener.emit(.windowContentChanged(timestamp: 20))
        for _ in 0 ..< 30 {
            if provider.captureModes.contains(.diffFromPrevious) { break }
            await Task.yield()
        }
        await Task.yield()
        monitor.flushPendingEvents()

        XCTAssertEqual(events.map(\.kind), [.windowChanged])
        XCTAssertEqual(events.first?.window.title, "Renamed")
        XCTAssertNil(events.first?.ax)
        XCTAssertEqual(provider.resetCount, 0)
    }

    @MainActor
    func testWindowSoftCaptureKeepsOnlyLatestInFlightResult() async {
        let listener = TestComputerHistoryListener()
        let provider = MultiSuspendedComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(coalescingWindowSeconds: 0),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()

        listener.emit(.windowContentChanged(timestamp: 30))
        for _ in 0 ..< 30 {
            if provider.pendingDiffCount == 1 { break }
            await Task.yield()
        }
        listener.emit(.windowContentChanged(timestamp: 30.05))
        await Task.yield()
        XCTAssertEqual(provider.pendingDiffCount, 1)

        provider.resumeDiff(at: 0, text: "~ 2. button — Old")
        for _ in 0 ..< 30 {
            if provider.pendingDiffCount == 2 { break }
            await Task.yield()
        }
        XCTAssertEqual(provider.pendingDiffCount, 2)
        provider.resumeDiff(at: 1, text: "~ 2. button — Latest")
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()

        XCTAssertEqual(events.map(\.timestamp), [30, 30.05])
        XCTAssertEqual(events.compactMap(\.ax?.text), [
            "~ 2. button — Old",
            "~ 2. button — Latest",
        ])
    }

    @MainActor
    func testWindowSoftDebounceReplacementKeepsNewestTaskOwnership() async {
        let listener = TestComputerHistoryListener()
        let provider = TestComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(coalescingWindowSeconds: 0.03),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()
        provider.captureModes.removeAll()

        listener.emit(.windowContentChanged(timestamp: 31))
        await Task.yield()
        listener.emit(.windowContentChanged(timestamp: 32))
        for _ in 0 ..< 5 { await Task.yield() }
        listener.emit(.windowContentChanged(timestamp: 33))

        try? await Task.sleep(nanoseconds: 50_000_000)
        for _ in 0 ..< 30 {
            if provider.captureModes == [.diffFromPrevious] { break }
            await Task.yield()
        }
        monitor.flushPendingEvents()

        XCTAssertEqual(provider.captureModes, [.diffFromPrevious])
        XCTAssertEqual(events.map(\.timestamp), [33])
    }

    @MainActor
    func testOldSoftCaptureCannotConsumeNewGenerationPendingUpdate() async {
        let listener = TestComputerHistoryListener()
        let provider = MultiSuspendedComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(coalescingWindowSeconds: 0),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()

        listener.emit(.windowContentChanged(timestamp: 40))
        for _ in 0 ..< 30 {
            if provider.pendingDiffCount == 1 { break }
            await Task.yield()
        }
        XCTAssertEqual(provider.pendingDiffCount, 1)

        listener.emit(.privacyBoundary)
        listener.emit(.sessionResumed(timestamp: 41))
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()

        listener.emit(.windowContentChanged(timestamp: 42))
        for _ in 0 ..< 30 {
            if provider.pendingDiffCount == 2 { break }
            await Task.yield()
        }
        listener.emit(.windowContentChanged(timestamp: 43))
        XCTAssertEqual(provider.pendingDiffCount, 2)

        provider.resumeDiff(at: 0, text: "~ 2. button — Stale")
        await Task.yield()
        provider.resumeDiff(at: 1, text: "~ 2. button — Current")
        for _ in 0 ..< 30 {
            if provider.pendingDiffCount == 3 { break }
            await Task.yield()
        }
        XCTAssertEqual(provider.pendingDiffCount, 3)
        provider.resumeDiff(at: 2, text: "~ 2. button — Latest")
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()

        XCTAssertEqual(events.map(\.timestamp), [42, 43])
        XCTAssertEqual(events.compactMap(\.ax?.text), [
            "~ 2. button — Current",
            "~ 2. button — Latest",
        ])
    }

    @MainActor
    func testOverflowEmitsSuppressionCallbackAndResetsItsCounter() async {
        let listener = TestComputerHistoryListener()
        let provider = TestComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        var reports: [ComputerHistorySuppressionReport] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(maximumQueuedEvents: 1),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onSuppressed: { reports.append($0) },
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()

        listener.emit(.keyboardTextInput(timestamp: 2, text: "a"))
        listener.emit(.selectionChanged(timestamp: 3))
        monitor.flushPendingEvents()

        XCTAssertEqual(events.map(\.kind), [.selectionChanged])
        XCTAssertEqual(reports.map(\.count), [1])
        XCTAssertEqual(monitor.suppressedEventCount, 0)
    }

    @MainActor
    func testPrivacyBoundaryDropsDiffThatWasAlreadyInFlight() async {
        let listener = TestComputerHistoryListener()
        let provider = SuspendedComputerHistoryAXProvider()
        var events: [ComputerHistoryEvent] = []
        let monitor = ComputerHistoryMonitor(
            configuration: .init(actionDiffDelaySeconds: 0),
            trustCheck: { true },
            listener: listener,
            axProvider: provider,
            onEvent: { events.append($0) }
        )
        _ = monitor.start()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()
        events.removeAll()

        listener.emit(.mouseClick(
            timestamp: 10,
            button: .left,
            clickCount: 1,
            location: .init(x: 1, y: 2)
        ))
        for _ in 0 ..< 20 {
            if provider.diffIsPending { break }
            await Task.yield()
        }
        XCTAssertTrue(provider.diffIsPending)

        listener.emit(.privacyBoundary)
        provider.resumeDiff()
        await Task.yield()
        await Task.yield()
        monitor.flushPendingEvents()

        XCTAssertTrue(events.isEmpty)
        monitor.stop()
    }
}

final class NativeComputerHistoryInteractionListenerTests: XCTestCase {
    @MainActor
    func testListenPermissionIsRequestedOnlyAfterExplicitStartAndDegradesAXOnly() {
        var requestCount = 0
        var statusCount = 0
        var permissionGranted = false
        let listener = NativeComputerHistoryInteractionListener(
            preflightListenEventAccess: { permissionGranted },
            requestListenEventAccess: {
                requestCount += 1
                return false
            },
            inputTapStartOverride: { _ in true }
        )

        XCTAssertEqual(requestCount, 0)
        XCTAssertEqual(listener.inputReport.eventTapState, .disabled)
        XCTAssertTrue(listener.start(
            handler: { _ in },
            onInputStatusChanged: { statusCount += 1 }
        ))

        XCTAssertEqual(requestCount, 1)
        XCTAssertEqual(listener.inputReport.listenEventPermission, .denied)
        XCTAssertEqual(listener.inputReport.eventTapState, .permissionRequired)
        XCTAssertEqual(listener.inputReport.mouse.state, .unavailable)
        XCTAssertEqual(listener.inputReport.keyboard.state, .unavailable)
        XCTAssertGreaterThanOrEqual(statusCount, 1)
        XCTAssertTrue(listener.isPollingListenEventPermission)

        listener.suspendInputTap()
        XCTAssertEqual(listener.inputReport.eventTapState, .suspended)
        XCTAssertFalse(listener.isPollingListenEventPermission)
        permissionGranted = true
        XCTAssertEqual(listener.inputReport.eventTapState, .suspended)

        listener.resumeInputTap()
        XCTAssertEqual(listener.inputReport.listenEventPermission, .granted)
        XCTAssertEqual(listener.inputReport.eventTapState, .running)
        XCTAssertFalse(listener.isPollingListenEventPermission)
        listener.stop()
        XCTAssertEqual(listener.inputReport.eventTapState, .disabled)
    }

    @MainActor
    func testDirectTapDeliveryCoversMouseKeyboardHealthAndGenerationSafety() {
        var interactions: [ComputerHistoryInteraction] = []
        var statusCount = 0
        let listener = NativeComputerHistoryInteractionListener(
            preflightListenEventAccess: { true },
            requestListenEventAccess: {
                XCTFail("preflight grant must not request Input Monitoring")
                return false
            },
            inputTapStartOverride: { _ in true }
        )
        XCTAssertTrue(listener.start(
            handler: { interactions.append($0) },
            onInputStatusChanged: { statusCount += 1 }
        ))
        let generation = listener.inputEventGeneration

        listener.handleInputTapEvent(.init(
            type: .leftMouseDown,
            location: .init(x: 10, y: 20),
            button: .left
        ), generation: generation &- 1)
        XCTAssertTrue(interactions.isEmpty)
        XCTAssertEqual(listener.inputReport.mouse.receivedCount, 0)

        listener.handleInputTapEvent(.init(
            type: .leftMouseDown,
            location: .init(x: 10, y: 20),
            button: .left
        ), generation: generation)
        listener.handleInputTapEvent(.init(
            type: .leftMouseUp,
            location: .init(x: 10, y: 20),
            button: .left,
            clickCount: 1
        ), generation: generation)
        listener.handleInputTapEvent(.init(
            type: .leftMouseDown,
            location: .init(x: 1, y: 2),
            button: .left
        ), generation: generation)
        listener.handleInputTapEvent(.init(
            type: .leftMouseDragged,
            location: .init(x: 5, y: 6),
            button: .left
        ), generation: generation)
        listener.handleInputTapEvent(.init(
            type: .leftMouseUp,
            location: .init(x: 7, y: 8),
            button: .left
        ), generation: generation)

        listener.handleInputTapEvent(.init(
            type: .keyDown,
            keyCode: 0,
            characters: "a",
            charactersIgnoringModifiers: "a"
        ), generation: generation)
        listener.handleInputTapEvent(.init(
            type: .keyDown,
            keyCode: 3,
            modifierFlags: [.command],
            characters: "ƒ",
            charactersIgnoringModifiers: "f"
        ), generation: generation)
        listener.handleInputTapEvent(.init(
            type: .keyDown,
            keyCode: 36,
            characters: "\r",
            charactersIgnoringModifiers: "\r"
        ), generation: generation)
        listener.handleInputTapEvent(.init(
            type: .keyDown,
            keyCode: 76,
            characters: "\r",
            charactersIgnoringModifiers: "\r"
        ), generation: generation)

        XCTAssertEqual(interactions.count, 6)
        guard case .mouseClick = interactions[0] else {
            return XCTFail("expected click")
        }
        guard case .mouseDrag = interactions[1] else {
            return XCTFail("expected drag")
        }
        XCTAssertEqual(interactions[2], .keyboardTextInput(
            timestamp: interactions[2].timestampForTesting,
            text: "a"
        ))
        guard case let .keyboardShortcut(_, key, modifiers) = interactions[3] else {
            return XCTFail("expected shortcut")
        }
        XCTAssertEqual(key, "f")
        XCTAssertEqual(modifiers, ["command"])
        guard case let .keyboardSubmit(_, key, _) = interactions[4] else {
            return XCTFail("expected return submit")
        }
        XCTAssertEqual(key, "return")
        guard case let .keyboardSubmit(_, key, _) = interactions[5] else {
            return XCTFail("expected keypad submit")
        }
        XCTAssertEqual(key, "enter")
        XCTAssertEqual(listener.inputReport.mouse.receivedCount, 5)
        XCTAssertEqual(listener.inputReport.keyboard.receivedCount, 4)
        XCTAssertNotNil(listener.inputReport.mouse.lastEventAt)
        XCTAssertNotNil(listener.inputReport.keyboard.lastEventAt)
        XCTAssertGreaterThanOrEqual(statusCount, 3)

        listener.handleInputTapDisabled(generation: generation)
        XCTAssertEqual(listener.inputReport.tapDisabledCount, 1)
        XCTAssertEqual(listener.inputReport.eventTapState, .running)
        XCTAssertNotEqual(listener.inputEventGeneration, generation)
        let countBeforeStaleEvent = interactions.count
        listener.handleInputTapEvent(.init(
            type: .keyDown,
            keyCode: 0,
            characters: "z",
            charactersIgnoringModifiers: "z"
        ), generation: generation)
        XCTAssertEqual(interactions.count, countBeforeStaleEvent)
        listener.stop()
    }

    @MainActor
    func testKeyboardInterpreterKeepsPlainSpaceAsTextAndModifiedSpaceAsShortcut() {
        XCTAssertEqual(
            NativeComputerHistoryInteractionListener.keyboardInteraction(
                for: .init(
                    type: .keyDown,
                    keyCode: 49,
                    characters: " ",
                    charactersIgnoringModifiers: " "
                ),
                timestamp: 1
            ),
            .keyboardTextInput(timestamp: 1, text: " ")
        )
        XCTAssertEqual(
            NativeComputerHistoryInteractionListener.keyboardInteraction(
                for: .init(
                    type: .keyDown,
                    keyCode: 49,
                    modifierFlags: [.command],
                    characters: " ",
                    charactersIgnoringModifiers: " "
                ),
                timestamp: 2
            ),
            .keyboardShortcut(
                timestamp: 2,
                keyEquivalent: "space",
                modifiers: ["command"]
            )
        )
    }
}

private extension ComputerHistoryInteraction {
    var timestampForTesting: Double {
        switch self {
        case let .focusedWindowChanged(timestamp),
            let .windowContentChanged(timestamp),
            let .sessionResumed(timestamp),
            let .selectionChanged(timestamp),
            let .keyboardTextInput(timestamp, _):
            return timestamp
        case .privacyBoundary:
            return 0
        case let .mouseClick(timestamp, _, _, _),
            let .mouseContextMenu(timestamp, _, _, _),
            let .mouseDrag(timestamp, _, _, _),
            let .keyboardShortcut(timestamp, _, _),
            let .keyboardSubmit(timestamp, _, _):
            return timestamp
        }
    }
}

@MainActor
private final class TestComputerHistoryListener: ComputerHistoryInteractionListening {
    var handler: ComputerHistoryInteractionHandler?
    private(set) var started = false
    private(set) var stopped = false
    var inputReport = ComputerHistoryInputReport(
        accessibilityPermission: .granted,
        listenEventPermission: .granted,
        eventTapState: .running,
        mouse: .init(state: .running, receivedCount: 0, lastEventAt: nil),
        keyboard: .init(state: .running, receivedCount: 0, lastEventAt: nil),
        tapDisabledCount: 0
    )

    func start(
        handler: @escaping ComputerHistoryInteractionHandler,
        onInputStatusChanged: @escaping ComputerHistoryInputStatusHandler
    ) -> Bool {
        _ = onInputStatusChanged
        self.handler = handler
        started = true
        return true
    }

    func stop() {
        stopped = true
        handler = nil
    }

    func emit(_ interaction: ComputerHistoryInteraction) {
        handler?(interaction)
    }
}

@MainActor
private final class TestComputerHistoryAXProvider: ComputerHistoryAXProviding {
    var context = ComputerHistoryContext(
        key: "7:window",
        app: .init(bundleIdentifier: "com.example.editor", name: "Editor"),
        window: .init(title: "Document", url: nil)
    )
    var focusedTarget = ComputerHistoryResolvedTarget(
        target: .init(role: "AXTextArea", title: "Editor", description: nil),
        secure: false
    )
    var selection: ComputerHistorySelection?
    var fullTreeText = "1. window — Document"
    var diffText = "~ 2. button — Save"
    var captureModes: [ComputerHistoryAXMode] = []
    private(set) var resetCount = 0

    init() {
        selection = ComputerHistorySelection(
            text: "text",
            truncated: false,
            selectedRange: .init(location: 2, length: 4),
            target: focusedTarget.target
        )
    }

    func currentContext() -> ComputerHistoryContext? { context }

    func capture(mode: ComputerHistoryAXMode) async -> ComputerHistoryAXCapture? {
        captureModes.append(mode)
        return ComputerHistoryAXCapture(
            context: context,
            content: .init(
                mode: mode,
                text: mode == .fullTree ? fullTreeText : diffText,
                truncated: false
            )
        )
    }

    func target(at point: ComputerHistoryPoint?) -> ComputerHistoryResolvedTarget? {
        _ = point
        return focusedTarget
    }

    func currentFocusedTarget() -> ComputerHistoryResolvedTarget? {
        focusedTarget
    }

    func currentSelection() -> ComputerHistorySelection? {
        selection
    }

    func reset() {
        resetCount += 1
    }
}

@MainActor
private final class MultiSuspendedComputerHistoryAXProvider:
    ComputerHistoryAXProviding
{
    private var continuations: [
        CheckedContinuation<ComputerHistoryAXCapture?, Never>?
    ] = []
    let context = ComputerHistoryContext(
        key: "7:window",
        app: .init(bundleIdentifier: "com.example.editor", name: "Editor"),
        window: .init(title: "Document", url: nil)
    )

    var pendingDiffCount: Int { continuations.count }

    func currentContext() -> ComputerHistoryContext? { context }

    func capture(mode: ComputerHistoryAXMode) async -> ComputerHistoryAXCapture? {
        if mode == .fullTree {
            return ComputerHistoryAXCapture(
                context: context,
                content: .init(
                    mode: .fullTree,
                    text: "1. window — Document",
                    truncated: false
                )
            )
        }
        return await withCheckedContinuation { continuation in
            continuations.append(continuation)
        }
    }

    func target(at point: ComputerHistoryPoint?) -> ComputerHistoryResolvedTarget? {
        _ = point
        return nil
    }

    func currentFocusedTarget() -> ComputerHistoryResolvedTarget? { nil }
    func currentSelection() -> ComputerHistorySelection? { nil }
    func reset() {}

    func resumeDiff(at index: Int, text: String) {
        guard continuations.indices.contains(index),
              let continuation = continuations[index]
        else { return }
        continuations[index] = nil
        continuation.resume(returning: ComputerHistoryAXCapture(
            context: context,
            content: .init(
                mode: .diffFromPrevious,
                text: text,
                truncated: false
            )
        ))
    }
}

@MainActor
private final class SuspendedComputerHistoryAXProvider: ComputerHistoryAXProviding {
    private let context = ComputerHistoryContext(
        key: "7:window",
        app: .init(bundleIdentifier: "com.example.editor", name: "Editor"),
        window: .init(title: "Document", url: nil)
    )
    private var diffContinuation: CheckedContinuation<ComputerHistoryAXCapture?, Never>?

    var diffIsPending: Bool { diffContinuation != nil }

    func currentContext() -> ComputerHistoryContext? { context }

    func capture(mode: ComputerHistoryAXMode) async -> ComputerHistoryAXCapture? {
        if mode == .fullTree {
            return ComputerHistoryAXCapture(
                context: context,
                content: .init(
                    mode: .fullTree,
                    text: "1. window — Document",
                    truncated: false
                )
            )
        }
        return await withCheckedContinuation { diffContinuation = $0 }
    }

    func target(at point: ComputerHistoryPoint?) -> ComputerHistoryResolvedTarget? {
        _ = point
        return ComputerHistoryResolvedTarget(
            target: .init(role: "AXButton", title: "Save", description: nil),
            secure: false
        )
    }

    func currentFocusedTarget() -> ComputerHistoryResolvedTarget? {
        target(at: nil)
    }

    func currentSelection() -> ComputerHistorySelection? { nil }

    func reset() {}

    func resumeDiff() {
        diffContinuation?.resume(returning: ComputerHistoryAXCapture(
            context: context,
            content: .init(
                mode: .diffFromPrevious,
                text: "~ 2. button — Save",
                truncated: false
            )
        ))
        diffContinuation = nil
    }
}
