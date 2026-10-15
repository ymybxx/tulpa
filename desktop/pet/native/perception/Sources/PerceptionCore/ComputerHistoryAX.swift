import Foundation

struct ComputerHistoryAXNode: Equatable, Sendable {
    let path: String
    let role: String
    let subrole: String?
    let title: String?
    let description: String?
    let value: String?
    let url: String?
    let selectedText: String?
    let frame: AccessibilityWindowFrame?

    init(
        path: String,
        role: String,
        subrole: String? = nil,
        title: String?,
        description: String?,
        value: String?,
        url: String? = nil,
        selectedText: String? = nil,
        frame: AccessibilityWindowFrame? = nil
    ) {
        self.path = path
        self.role = role
        self.subrole = subrole
        self.title = title
        self.description = description
        self.value = value
        self.url = url
        self.selectedText = selectedText
        self.frame = frame
    }

    var semanticRole: String {
        let source = Self.meaningful(subrole)
            ?? Self.meaningful(role)
            ?? "element"
        let withoutPrefix = source.hasPrefix("AX")
            ? String(source.dropFirst(2))
            : source
        let characters = Array(withoutPrefix)
        var result = ""
        for index in characters.indices {
            let character = characters[index]
            if index > characters.startIndex,
               character.isUppercase
            {
                let previous = characters[characters.index(before: index)]
                let nextIndex = characters.index(after: index)
                let nextIsLowercase = nextIndex < characters.endIndex
                    && characters[nextIndex].isLowercase
                if previous.isLowercase || previous.isNumber
                    || (previous.isUppercase && nextIsLowercase)
                {
                    result.append(" ")
                }
            }
            result.append(contentsOf: character.lowercased())
        }
        return result.isEmpty ? "element" : result
    }

    var semanticText: [String] {
        var result: [String] = []
        var seen = Set<String>()
        for (label, candidate) in [
            (nil, title),
            (nil, value),
            (nil, description),
            (nil, url),
            ("selected: ", selectedText),
        ] {
            guard let meaningful = Self.meaningful(candidate) else { continue }
            let key = meaningful.casefolded
            guard seen.insert(key).inserted else { continue }
            result.append((label ?? "") + Self.escape(meaningful))
        }
        return result
    }

    private static func meaningful(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func escape(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\u{0000}", with: "\\0")
            .replacingOccurrences(of: "\u{007F}", with: "\\u007f")
            .replacingOccurrences(of: "\u{2028}", with: "\\u2028")
            .replacingOccurrences(of: "\u{2029}", with: "\\u2029")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
    }
}

private extension String {
    var casefolded: String { folding(options: [.caseInsensitive], locale: nil) }
}

private struct ComputerHistoryRenderedAXNode: Equatable {
    let path: String
    let signature: String
    let indentation: String
    let numberedText: String

    var fullLine: String { indentation + numberedText }

    func diffLine(_ marker: Character) -> String {
        indentation + String(marker) + " " + numberedText
    }
}

struct ComputerHistoryAXSnapshotState {
    private let fullTreeMaximumBytes: Int
    private let diffMaximumBytes: Int
    private var baseline: [String: ComputerHistoryRenderedAXNode] = [:]
    private var semanticIDs: [String: Int] = [:]
    private var nextSemanticID = 1

    init(
        fullTreeMaximumBytes: Int = 64 * 1_024,
        diffMaximumBytes: Int = 32 * 1_024
    ) {
        precondition(fullTreeMaximumBytes > 0)
        precondition(diffMaximumBytes > 0)
        self.fullTreeMaximumBytes = fullTreeMaximumBytes
        self.diffMaximumBytes = diffMaximumBytes
    }

    mutating func captureFullTree(
        nodes: [ComputerHistoryAXNode],
        sourceTruncated: Bool
    ) -> ComputerHistoryAXContent {
        let rendered = render(nodes, rebuildSemanticIDs: true)
        let next = Self.dictionary(rendered)
        baseline = next
        let raw = rendered.map(\.fullLine).joined(separator: "\n")
        let bounded = Self.boundedUTF8(raw, maximumBytes: fullTreeMaximumBytes)
        return ComputerHistoryAXContent(
            mode: .fullTree,
            text: bounded.text,
            truncated: sourceTruncated || bounded.truncated
        )
    }

    mutating func captureDiff(
        nodes: [ComputerHistoryAXNode],
        sourceTruncated: Bool
    ) -> ComputerHistoryAXContent {
        let rendered = render(nodes, rebuildSemanticIDs: false)
        var next = Self.dictionary(rendered)
        if sourceTruncated {
            // A bounded traversal cannot prove that an unseen tail node was
            // deleted. Retain the last complete knowledge until a later full
            // traversal can confirm removal.
            for (path, previous) in baseline where next[path] == nil {
                next[path] = previous
            }
        }
        var lines: [String] = []
        for path in Set(baseline.keys).union(next.keys).sorted(by: Self.pathPrecedes) {
            switch (baseline[path], next[path]) {
            case let (.some(old), .some(new)) where old.signature != new.signature:
                lines.append(new.diffLine("~"))
            case let (.none, .some(new)):
                lines.append(new.diffLine("+"))
            case let (.some(old), .none):
                lines.append(old.diffLine("-"))
            default:
                break
            }
        }
        baseline = next
        let bounded = Self.boundedUTF8(
            lines.joined(separator: "\n"),
            maximumBytes: diffMaximumBytes
        )
        return ComputerHistoryAXContent(
            mode: .diffFromPrevious,
            text: bounded.text,
            truncated: sourceTruncated || bounded.truncated
        )
    }

    mutating func reset() {
        baseline.removeAll(keepingCapacity: true)
        semanticIDs.removeAll(keepingCapacity: true)
        nextSemanticID = 1
    }

    private mutating func render(
        _ nodes: [ComputerHistoryAXNode],
        rebuildSemanticIDs: Bool
    ) -> [ComputerHistoryRenderedAXNode] {
        if rebuildSemanticIDs {
            semanticIDs.removeAll(keepingCapacity: true)
            nextSemanticID = 1
        }
        let semanticNodes = nodes
            .filter { !$0.semanticText.isEmpty }
            .sorted { Self.pathPrecedes($0.path, $1.path) }
        var includedPaths = Set<String>()
        var rendered: [ComputerHistoryRenderedAXNode] = []
        rendered.reserveCapacity(semanticNodes.count)
        for node in semanticNodes {
            let components = node.path.split(separator: "/")
            var semanticDepth = 0
            if components.count > 1 {
                for count in 1 ..< components.count {
                    let ancestor = components.prefix(count).joined(separator: "/")
                    if includedPaths.contains(ancestor) { semanticDepth += 1 }
                }
            }
            includedPaths.insert(node.path)
            let text = node.semanticText.joined(separator: " | ")
            let semanticID: Int
            if let existing = semanticIDs[node.path] {
                semanticID = existing
            } else {
                semanticID = nextSemanticID
                semanticIDs[node.path] = semanticID
                nextSemanticID += 1
            }
            rendered.append(ComputerHistoryRenderedAXNode(
                path: node.path,
                signature: node.semanticRole + "\u{001F}" + text,
                indentation: String(repeating: "  ", count: semanticDepth),
                numberedText: "\(semanticID). \(node.semanticRole) — \(text)"
            ))
        }
        return rendered
    }

    private static func dictionary(
        _ nodes: [ComputerHistoryRenderedAXNode]
    ) -> [String: ComputerHistoryRenderedAXNode] {
        Dictionary(uniqueKeysWithValues: nodes.map { ($0.path, $0) })
    }

    private static func pathPrecedes(_ left: String, _ right: String) -> Bool {
        let leftParts = left.split(separator: "/").map { Int($0) ?? .max }
        let rightParts = right.split(separator: "/").map { Int($0) ?? .max }
        return leftParts.lexicographicallyPrecedes(rightParts)
    }

    private static func boundedUTF8(
        _ value: String,
        maximumBytes: Int
    ) -> (text: String, truncated: Bool) {
        guard value.utf8.count > maximumBytes else {
            return (value, false)
        }
        var result = ""
        result.reserveCapacity(maximumBytes)
        var byteCount = 0
        for character in value {
            let fragment = String(character)
            let fragmentBytes = fragment.utf8.count
            guard byteCount + fragmentBytes <= maximumBytes else { break }
            result.append(character)
            byteCount += fragmentBytes
        }
        return (result, true)
    }
}

enum ComputerHistoryAXVisibilityPolicy {
    static func shouldInclude(
        hidden: Bool,
        frame: AccessibilityWindowFrame?,
        rootFrame: AccessibilityWindowFrame?
    ) -> Bool {
        guard !hidden else { return false }
        guard let frame, let rootFrame else { return true }
        return frame.rect.intersects(rootFrame.rect)
    }
}

enum ComputerHistoryContextKey {
    static func make(
        processIdentifier: pid_t,
        windowIdentifier: String?,
        fallbackWindowToken: String
    ) -> String {
        if let windowIdentifier, !windowIdentifier.isEmpty {
            return "\(processIdentifier):identifier:\(windowIdentifier)"
        }
        return "\(processIdentifier):window:\(fallbackWindowToken)"
    }
}
