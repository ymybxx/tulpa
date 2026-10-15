import Foundation

public enum ComputerHistoryEventKind: String, Codable, CaseIterable, Sendable {
    case sessionStarted = "session.started"
    case windowChanged = "window.changed"
    case mouseClick = "mouse.click"
    case mouseContextMenu = "mouse.context_menu"
    case mouseDrag = "mouse.drag"
    case selectionChanged = "selection.changed"
    case keyboardShortcut = "keyboard.shortcut"
    case keyboardSubmit = "keyboard.submit"
    case keyboardTextInput = "keyboard.text_input"
}

public struct ComputerHistoryApplication: Codable, Equatable, Sendable {
    public let bundleIdentifier: String?
    public let name: String?

    public init(bundleIdentifier: String?, name: String?) {
        self.bundleIdentifier = bundleIdentifier
        self.name = name
    }
}

public struct ComputerHistoryWindow: Codable, Equatable, Sendable {
    public let title: String?
    public let url: String?

    public init(title: String?, url: String?) {
        self.title = title
        self.url = url
    }
}

public struct ComputerHistoryTarget: Codable, Equatable, Sendable {
    public let role: String
    public let title: String?
    public let description: String?

    public init(role: String, title: String?, description: String?) {
        self.role = role
        self.title = title
        self.description = description
    }
}

public enum ComputerHistoryAXMode: String, Codable, Sendable {
    case fullTree
    case diffFromPrevious
}

public struct ComputerHistoryAXContent: Codable, Equatable, Sendable {
    public let mode: ComputerHistoryAXMode
    public let text: String
    public let truncated: Bool

    public init(mode: ComputerHistoryAXMode, text: String, truncated: Bool) {
        self.mode = mode
        self.text = text
        self.truncated = truncated
    }
}

public enum ComputerHistoryMouseButton: String, Codable, Sendable {
    case left
    case right
    case other
}

public struct ComputerHistoryPoint: Codable, Equatable, Sendable {
    public let x: Double
    public let y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

public struct ComputerHistoryMouse: Codable, Equatable, Sendable {
    public let button: ComputerHistoryMouseButton
    public let clickCount: Int?
    public let target: ComputerHistoryTarget?
    public let location: ComputerHistoryPoint?
    public let origin: ComputerHistoryPoint?
    public let destination: ComputerHistoryPoint?

    public init(
        button: ComputerHistoryMouseButton,
        clickCount: Int? = nil,
        target: ComputerHistoryTarget? = nil,
        location: ComputerHistoryPoint? = nil,
        origin: ComputerHistoryPoint? = nil,
        destination: ComputerHistoryPoint? = nil
    ) {
        self.button = button
        self.clickCount = clickCount
        self.target = target
        self.location = location
        self.origin = origin
        self.destination = destination
    }
}

public struct ComputerHistoryKeyboard: Codable, Equatable, Sendable {
    public let text: String?
    public let keyEquivalent: String?
    public let modifiers: [String]?
    public let target: ComputerHistoryTarget?

    public init(
        text: String? = nil,
        keyEquivalent: String? = nil,
        modifiers: [String]? = nil,
        target: ComputerHistoryTarget? = nil
    ) {
        self.text = text
        self.keyEquivalent = keyEquivalent
        self.modifiers = modifiers
        self.target = target
    }
}

public struct ComputerHistorySelectedRange: Codable, Equatable, Sendable {
    public let location: Int
    public let length: Int

    public init(location: Int, length: Int) {
        self.location = location
        self.length = length
    }
}

public struct ComputerHistorySelection: Codable, Equatable, Sendable {
    public let text: String?
    public let truncated: Bool
    public let selectedRange: ComputerHistorySelectedRange?
    public let target: ComputerHistoryTarget?

    public init(
        text: String?,
        truncated: Bool,
        selectedRange: ComputerHistorySelectedRange? = nil,
        target: ComputerHistoryTarget? = nil
    ) {
        self.text = text
        self.truncated = truncated
        self.selectedRange = selectedRange
        self.target = target
    }
}

struct ComputerHistorySelectionGate {
    private var contextKey: String?
    private var lastSelection: ComputerHistorySelection?

    mutating func shouldEmit(
        contextKey nextContextKey: String,
        selection: ComputerHistorySelection
    ) -> Bool {
        if contextKey != nextContextKey {
            contextKey = nextContextKey
            lastSelection = nil
        }
        guard selection != lastSelection else { return false }

        let hasText = !(selection.text?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty ?? true)
        let hasRange = (selection.selectedRange?.length ?? 0) > 0
        let meaningful = hasText || hasRange
        lastSelection = selection
        return meaningful
    }

    mutating func reset() {
        contextKey = nil
        lastSelection = nil
    }
}

public struct ComputerHistoryEvent: Codable, Equatable, Sendable {
    public let id: String
    public let kind: ComputerHistoryEventKind
    public let timestamp: Double
    public let app: ComputerHistoryApplication
    public let window: ComputerHistoryWindow
    public let ax: ComputerHistoryAXContent?
    public let mouse: ComputerHistoryMouse?
    public let keyboard: ComputerHistoryKeyboard?
    public let selection: ComputerHistorySelection?

    public init(
        id: String = UUID().uuidString,
        kind: ComputerHistoryEventKind,
        timestamp: Double,
        app: ComputerHistoryApplication,
        window: ComputerHistoryWindow,
        ax: ComputerHistoryAXContent? = nil,
        mouse: ComputerHistoryMouse? = nil,
        keyboard: ComputerHistoryKeyboard? = nil,
        selection: ComputerHistorySelection? = nil
    ) {
        self.id = id
        self.kind = kind
        self.timestamp = timestamp
        self.app = app
        self.window = window
        self.ax = ax
        self.mouse = mouse
        self.keyboard = keyboard
        self.selection = selection
    }
}

struct ComputerHistoryEventBuffer {
    private let maximumCount: Int
    private let coalescingWindowSeconds: TimeInterval
    private var events: [ComputerHistoryEvent] = []
    private(set) var suppressedEventCount = 0

    init(maximumCount: Int = 128, coalescingWindowSeconds: TimeInterval = 0.1) {
        precondition(maximumCount > 0)
        precondition(coalescingWindowSeconds >= 0)
        self.maximumCount = maximumCount
        self.coalescingWindowSeconds = coalescingWindowSeconds
    }

    var count: Int { events.count }

    mutating func enqueue(_ event: ComputerHistoryEvent) {
        if let index = events.lastIndex(where: { existing in
            existing.kind == event.kind
                && existing.app == event.app
                && existing.window == event.window
                && Self.targetsPermitCoalescing(existing, event)
                && event.timestamp >= existing.timestamp
                && event.timestamp - existing.timestamp <= coalescingWindowSeconds
        }) {
            events[index] = Self.merge(events[index], event)
        } else {
            events.append(event)
        }
        if events.count > maximumCount {
            let overflow = events.count - maximumCount
            events.removeFirst(overflow)
            suppressedEventCount = suppressedEventCount > Int.max - overflow
                ? Int.max
                : suppressedEventCount + overflow
        }
    }

    mutating func recordSuppressed(_ count: Int = 1) {
        guard count > 0 else { return }
        suppressedEventCount = suppressedEventCount > Int.max - count
            ? Int.max
            : suppressedEventCount + count
    }

    mutating func drain() -> [ComputerHistoryEvent] {
        let drained = events.sorted { left, right in
            if left.timestamp == right.timestamp { return left.id < right.id }
            return left.timestamp < right.timestamp
        }
        events.removeAll(keepingCapacity: true)
        return drained
    }

    mutating func removeAll() {
        events.removeAll(keepingCapacity: true)
        suppressedEventCount = 0
    }

    mutating func takeSuppressedEventCount() -> Int {
        let count = suppressedEventCount
        suppressedEventCount = 0
        return count
    }

    private static func merge(
        _ existing: ComputerHistoryEvent,
        _ incoming: ComputerHistoryEvent
    ) -> ComputerHistoryEvent {
        let keyboard: ComputerHistoryKeyboard?
        if let first = existing.keyboard, let second = incoming.keyboard {
            let combinedText: String?
            if first.text != nil || second.text != nil {
                combinedText = (first.text ?? "") + (second.text ?? "")
            } else {
                combinedText = nil
            }
            keyboard = ComputerHistoryKeyboard(
                text: combinedText,
                keyEquivalent: second.keyEquivalent ?? first.keyEquivalent,
                modifiers: second.modifiers ?? first.modifiers,
                target: second.target ?? first.target
            )
        } else {
            keyboard = incoming.keyboard ?? existing.keyboard
        }

        let mouse: ComputerHistoryMouse?
        if let first = existing.mouse, let second = incoming.mouse {
            mouse = ComputerHistoryMouse(
                button: second.button,
                clickCount: second.clickCount ?? first.clickCount,
                target: second.target ?? first.target,
                location: second.location ?? first.location,
                origin: first.origin ?? second.origin,
                destination: second.destination ?? first.destination
            )
        } else {
            mouse = incoming.mouse ?? existing.mouse
        }

        return ComputerHistoryEvent(
            id: existing.id,
            kind: existing.kind,
            timestamp: incoming.timestamp,
            app: incoming.app,
            window: incoming.window,
            ax: incoming.ax ?? existing.ax,
            mouse: mouse,
            keyboard: keyboard,
            selection: incoming.selection ?? existing.selection
        )
    }

    private static func targetsPermitCoalescing(
        _ existing: ComputerHistoryEvent,
        _ incoming: ComputerHistoryEvent
    ) -> Bool {
        if existing.kind == .selectionChanged {
            guard existing.selection?.target == incoming.selection?.target else {
                return false
            }
            let existingMeaningful = isMeaningful(existing.selection)
            let incomingMeaningful = isMeaningful(incoming.selection)
            // A clear immediately after a real selection is a separate semantic
            // transition. Do not let the 100 ms merge erase the selected text.
            return !existingMeaningful || incomingMeaningful
        }
        guard existing.kind == .keyboardTextInput else { return true }
        guard existing.keyboard?.target == incoming.keyboard?.target else {
            return false
        }
        let combined = (existing.keyboard?.text ?? "")
            + (incoming.keyboard?.text ?? "")
        return combined.utf8.count <= 8 * 1_024
    }

    private static func isMeaningful(
        _ selection: ComputerHistorySelection?
    ) -> Bool {
        guard let selection else { return false }
        let hasText = !(selection.text?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty ?? true)
        return hasText || (selection.selectedRange?.length ?? 0) > 0
    }
}

public struct ComputerHistorySuppressionReport: Codable, Equatable, Sendable {
    public let count: Int
    public let timestamp: Double

    public init(count: Int, timestamp: Double) {
        self.count = count
        self.timestamp = timestamp
    }
}

enum ComputerHistorySuppressionChunking {
    static let maximumChunk = 1_000_000

    static func chunks(for count: Int) -> [Int] {
        guard count > 0 else { return [] }
        var remaining = count
        var chunks: [Int] = []
        while remaining > 0 {
            let chunk = min(remaining, maximumChunk)
            chunks.append(chunk)
            remaining -= chunk
        }
        return chunks
    }
}

enum ComputerHistoryWireText {
    static func context(_ value: String?, maximumCharacters: Int) -> String? {
        guard let value else { return nil }
        let cleanedScalars = value.unicodeScalars.map { scalar -> Character in
            if scalar.value <= 0x1F
                || scalar.value == 0x7F
                || scalar.value == 0x2028
                || scalar.value == 0x2029
            {
                return " "
            }
            return Character(String(scalar))
        }
        let cleaned = String(cleanedScalars)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return nil }
        let bounded = String(
            String.UnicodeScalarView(cleaned.unicodeScalars.prefix(maximumCharacters))
        )
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return bounded.isEmpty ? nil : bounded
    }

    static func payload(
        _ value: String?,
        maximumBytes: Int,
        allowEmpty: Bool
    ) -> String? {
        guard let value else { return nil }
        let cleaned = String(value.unicodeScalars.map { scalar -> Character in
            if scalar.value == 0
                || scalar.value == 0x7F
                || scalar.value == 0x2028
                || scalar.value == 0x2029
            {
                return "�"
            }
            return Character(String(scalar))
        })
        let bounded = boundedUTF8(cleaned, maximumBytes: maximumBytes)
        guard allowEmpty || !bounded.isEmpty else { return nil }
        return bounded
    }

    static func bundleIdentifier(_ value: String?) -> String? {
        guard let candidate = context(value, maximumCharacters: 255)?.lowercased(),
              let first = candidate.unicodeScalars.first,
              Self.isASCIIAlphaNumeric(first),
              candidate.unicodeScalars.allSatisfy({ scalar in
                  Self.isASCIIAlphaNumeric(scalar)
                      || scalar == "."
                      || scalar == "-"
              })
        else { return nil }
        return candidate
    }

    private static func boundedUTF8(_ value: String, maximumBytes: Int) -> String {
        guard value.utf8.count > maximumBytes else { return value }
        var result = ""
        var byteCount = 0
        for character in value {
            let bytes = String(character).utf8.count
            guard byteCount + bytes <= maximumBytes else { break }
            result.append(character)
            byteCount += bytes
        }
        return result
    }

    private static func isASCIIAlphaNumeric(_ scalar: UnicodeScalar) -> Bool {
        (scalar.value >= 48 && scalar.value <= 57)
            || (scalar.value >= 97 && scalar.value <= 122)
    }
}
