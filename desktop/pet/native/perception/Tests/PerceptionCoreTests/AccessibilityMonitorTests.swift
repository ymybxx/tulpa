import ApplicationServices
import XCTest
@testable import PerceptionCore

final class AccessibilityMonitorTests: XCTestCase {
    func testSecureFieldsAreRecognizedWithoutReadingTheirValue() {
        XCTAssertTrue(
            AccessibilityValueFilter.isSecureField(
                role: "AXTextField",
                subrole: kAXSecureTextFieldSubrole as String
            )
        )
        XCTAssertTrue(
            AccessibilityValueFilter.isSecureField(
                role: "AXSecureTextField",
                subrole: nil
            )
        )
        XCTAssertFalse(
            AccessibilityValueFilter.isSecureField(
                role: "AXTextField",
                subrole: nil
            )
        )
    }

    func testTextIsBoundedBeforeItLeavesTheMonitor() {
        XCTAssertEqual(
            AccessibilityValueFilter.bounded("abcdef", maximumLength: 4),
            AccessibilityValueFilter.BoundedValue(value: "abcd", truncated: true)
        )
        XCTAssertEqual(
            AccessibilityValueFilter.bounded("abc", maximumLength: 4),
            AccessibilityValueFilter.BoundedValue(value: "abc", truncated: false)
        )
    }

    func testDefaultLimitsKeepAXReadsSmallAndTimeBounded() {
        let configuration = AccessibilityMonitorConfiguration()

        XCTAssertEqual(configuration.messagingTimeoutSeconds, 0.15)
        XCTAssertEqual(configuration.maximumNodes, 16)
        XCTAssertEqual(configuration.maximumDepth, 4)
        XCTAssertEqual(configuration.maximumTextLength, 512)
        XCTAssertEqual(configuration.valueChangeDebounceSeconds, 0.2)
    }

    func testOnlyValueAndSelectionChangesAreDebounced() {
        XCTAssertTrue(AccessibilityNotificationPolicy.shouldDebounce(
            kAXValueChangedNotification
        ))
        XCTAssertTrue(AccessibilityNotificationPolicy.shouldDebounce(
            kAXSelectedTextChangedNotification
        ))
        XCTAssertFalse(AccessibilityNotificationPolicy.shouldDebounce(
            kAXFocusedUIElementChangedNotification
        ))
        XCTAssertFalse(AccessibilityNotificationPolicy.shouldDebounce(
            kAXFocusedWindowChangedNotification
        ))
    }

    func testKnownBrowserPrivateWindowTitlesAreSuppressed() {
        let cases: [(bundleIdentifier: String, windowTitle: String)] = [
            ("com.apple.Safari", "Private Browsing"),
            ("com.apple.Safari", "无痕浏览"),
            ("com.google.Chrome", "New Incognito Window"),
            ("com.microsoft.edgemac", "InPrivate"),
            ("com.brave.Browser", "無痕模式"),
            ("org.mozilla.firefox", "Private Browsing — Mozilla Firefox"),
        ]

        for item in cases {
            XCTAssertTrue(
                AccessibilityPrivateBrowsingPolicy.shouldSuppress(
                    bundleIdentifier: item.bundleIdentifier,
                    windowTitle: item.windowTitle
                ),
                "Expected suppression for \(item.bundleIdentifier): \(item.windowTitle)"
            )
        }
    }

    func testKnownBrowserOrdinaryWindowTitlesRemainObservable() {
        let cases: [(bundleIdentifier: String?, windowTitle: String?)] = [
            ("com.apple.Safari", "Project dashboard — Safari"),
            ("com.google.Chrome", "Fish Memory — GitHub"),
            ("com.microsoft.edgemac", "Documentation"),
            ("org.mozilla.firefox", "New Window"),
            ("com.google.Chrome", nil),
            (nil, "Incognito"),
        ]

        for item in cases {
            XCTAssertFalse(
                AccessibilityPrivateBrowsingPolicy.shouldSuppress(
                    bundleIdentifier: item.bundleIdentifier,
                    windowTitle: item.windowTitle
                ),
                "Expected ordinary window to remain observable"
            )
        }
    }

    func testPrivateKeywordsDoNotSuppressNonBrowserApplications() {
        let titles = [
            "Incognito launch notes",
            "InPrivate compatibility test",
            "无痕浏览产品需求",
            "Private Browsing translation",
        ]

        for title in titles {
            XCTAssertFalse(
                AccessibilityPrivateBrowsingPolicy.shouldSuppress(
                    bundleIdentifier: "com.example.editor",
                    windowTitle: title
                ),
                "Keyword in a non-browser title must not be treated as private mode"
            )
        }
    }

    @MainActor
    func testValueDebouncerKeepsLatestAndCancelSuppressesPendingWork() async throws {
        let debouncer = LatestAccessibilityEventDebouncer(delaySeconds: 0.01)
        var delivered: [String] = []

        debouncer.schedule { delivered.append("old") }
        debouncer.schedule { delivered.append("latest") }
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(delivered, ["latest"])

        debouncer.schedule { delivered.append("cancelled") }
        debouncer.cancel()
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(delivered, ["latest"])
    }

    func testObservationCarriesAnExplicitSourceDiscriminator() throws {
        let observation = AccessibilityObservation(
            revision: 1,
            kind: "accessibility_snapshot",
            observedAt: 10,
            processIdentifier: 7,
            bundleIdentifier: "com.example.app",
            applicationName: "Example",
            windowTitle: nil,
            focusedRole: nil,
            focusedSubrole: nil,
            focusedTitle: nil,
            focusedValue: nil,
            contextTitles: [],
            secureField: false,
            nodesInspected: 0,
            truncated: false
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(observation))
                as? [String: Any]
        )
        XCTAssertEqual(object["source"] as? String, "accessibility")
    }
}
