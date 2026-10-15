import XCTest
@testable import PerceptionCore

final class ScreenTextPrivacyPolicyTests: XCTestCase {
    func testAllowsMatchingFreshNonSecureFocusedWindow() throws {
        let clock = MutablePrivacyClock(now: 101)
        let store = ScreenTextPrivacyContextStore(clock: { clock.now })
        store.updateWorkspace(application(), invalidateAccessibility: true)
        store.updateAccessibility(accessibilityObservation(observedAt: 100))

        let authorization = try XCTUnwrap(store.authorization(for: candidate()))

        XCTAssertEqual(authorization.processIdentifier, 7)
        XCTAssertEqual(authorization.appBundleId, "com.example.editor")
        XCTAssertEqual(authorization.appName, "Editor")
        XCTAssertEqual(authorization.windowTitle, "Document")
        XCTAssertEqual(
            authorization.windowFrame,
            AccessibilityWindowFrame(x: 10, y: 20, width: 800, height: 600)
        )
    }

    func testRejectsStaleAccessibilityContext() {
        let clock = MutablePrivacyClock(now: 102.01)
        let store = ScreenTextPrivacyContextStore(clock: { clock.now })
        store.updateWorkspace(application(), invalidateAccessibility: true)
        store.updateAccessibility(accessibilityObservation(observedAt: 100))

        XCTAssertNil(store.authorization(for: candidate()))
    }

    func testAuthorizedCaptureSurvivesDelayAndSameWindowValueChanges() throws {
        let clock = MutablePrivacyClock(now: 101)
        let store = ScreenTextPrivacyContextStore(clock: { clock.now })
        store.updateWorkspace(application(), invalidateAccessibility: true)
        store.updateAccessibility(accessibilityObservation(observedAt: 100))
        let authorization = try XCTUnwrap(store.authorization(for: candidate()))

        clock.set(110)
        XCTAssertTrue(store.isStillAuthorized(authorization))

        store.updateAccessibility(accessibilityObservation(observedAt: 110))
        XCTAssertTrue(store.isStillAuthorized(authorization))

        store.updateAccessibility(accessibilityObservation(
            observedAt: 110,
            windowTitle: "Other Document"
        ))
        XCTAssertFalse(store.isStillAuthorized(authorization))
    }

    func testFreshLaterAccessibilityContextCanAuthorizeIndependentCapture() {
        let clock = MutablePrivacyClock(now: 101.6)
        let store = ScreenTextPrivacyContextStore(clock: { clock.now })
        store.updateWorkspace(application(), invalidateAccessibility: true)
        store.updateAccessibility(accessibilityObservation(observedAt: 101.5))

        XCTAssertNotNil(store.authorization(for: ScreenTextPrivacyContext(
            revision: 1,
            observedAt: 101,
            displayID: 1
        )))
    }

    func testRejectsSecureMismatchedAndMissingWindowContexts() {
        let clock = MutablePrivacyClock(now: 101)
        let store = ScreenTextPrivacyContextStore(clock: { clock.now })
        store.updateWorkspace(application(), invalidateAccessibility: true)

        store.updateAccessibility(accessibilityObservation(
            observedAt: 100,
            secureField: true
        ))
        XCTAssertNil(store.authorization(for: candidate()))

        store.updateAccessibility(accessibilityObservation(
            observedAt: 100,
            processIdentifier: 8
        ))
        XCTAssertNil(store.authorization(for: candidate()))

        store.updateAccessibility(accessibilityObservation(
            observedAt: 100,
            windowTitle: nil
        ))
        XCTAssertNil(store.authorization(for: candidate()))

        store.updateAccessibility(accessibilityObservation(
            observedAt: 100,
            windowFrame: nil
        ))
        XCTAssertNil(store.authorization(for: candidate()))
    }

    func testWorkspaceSwitchAndExplicitClearsInvalidateAccessibility() {
        let clock = MutablePrivacyClock(now: 101)
        let store = ScreenTextPrivacyContextStore(clock: { clock.now })
        store.updateWorkspace(application(), invalidateAccessibility: true)
        store.updateAccessibility(accessibilityObservation(observedAt: 100))
        XCTAssertNotNil(store.authorization(for: candidate()))

        store.updateWorkspace(application(), invalidateAccessibility: false)
        XCTAssertNotNil(store.authorization(for: candidate()))

        store.updateWorkspace(application(), invalidateAccessibility: true)
        XCTAssertNil(store.authorization(for: candidate()))

        store.updateAccessibility(accessibilityObservation(observedAt: 100))
        store.clearAccessibility()
        XCTAssertNil(store.authorization(for: candidate()))

        store.updateAccessibility(accessibilityObservation(observedAt: 100))
        store.clearWorkspace()
        XCTAssertNil(store.authorization(for: candidate()))
    }

    func testAllKnownBrowsersAndSensitiveApplicationsAreDenied() {
        let denied: [(String, String)] = [
            ("com.apple.Safari", "Safari"),
            ("com.google.Chrome.canary", "Google Chrome Canary"),
            ("com.microsoft.edgemac", "Microsoft Edge"),
            ("org.mozilla.firefox", "Firefox"),
            ("company.thebrowser.Browser", "Arc"),
            ("com.1password.1Password", "1Password"),
            ("com.bitwarden.desktop", "Bitwarden"),
            ("org.keepassxc.keepassxc", "KeePassXC"),
            ("com.apple.KeychainAccess", "Keychain Access"),
        ]

        for item in denied {
            XCTAssertTrue(ScreenTextSensitiveApplicationPolicy.shouldDeny(
                bundleIdentifier: item.0,
                applicationName: item.1
            ), "Expected deny for \(item.0)")
        }
        XCTAssertFalse(ScreenTextSensitiveApplicationPolicy.shouldDeny(
            bundleIdentifier: "com.example.editor",
            applicationName: "Editor"
        ))
    }

    func testSensitiveAuthenticationAndCredentialWindowTitlesAreDenied() {
        let denied = [
            "Sign in",
            "Authentication Required",
            "Enter Password",
            "API Key Settings",
            "Credential Manager",
            "输入验证码",
            "密钥管理",
        ]
        for title in denied {
            XCTAssertTrue(
                ScreenTextSensitiveWindowTitlePolicy.shouldDeny(title),
                "Expected sensitive title deny for \(title)"
            )
        }
        XCTAssertFalse(ScreenTextSensitiveWindowTitlePolicy.shouldDeny("Document"))
    }

    private func application() -> FrontmostApplication {
        FrontmostApplication(
            bundleIdentifier: "com.example.Editor",
            name: "Editor",
            processIdentifier: 7
        )
    }

    private func accessibilityObservation(
        observedAt: Double,
        processIdentifier: Int32 = 7,
        windowTitle: String? = "Document",
        windowFrame: AccessibilityWindowFrame? = AccessibilityWindowFrame(
            x: 10,
            y: 20,
            width: 800,
            height: 600
        ),
        secureField: Bool = false
    ) -> AccessibilityObservation {
        AccessibilityObservation(
            revision: 1,
            kind: "accessibility_snapshot",
            observedAt: observedAt,
            processIdentifier: processIdentifier,
            bundleIdentifier: "com.example.editor",
            applicationName: "Editor",
            windowTitle: windowTitle,
            windowFrame: windowFrame,
            focusedRole: "AXTextField",
            focusedSubrole: nil,
            focusedTitle: nil,
            focusedValue: nil,
            contextTitles: [],
            secureField: secureField,
            nodesInspected: 1,
            truncated: false
        )
    }

    private func candidate() -> ScreenTextPrivacyContext {
        ScreenTextPrivacyContext(revision: 1, observedAt: 101, displayID: 1)
    }
}

private final class MutablePrivacyClock: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Double

    init(now: Double) {
        value = now
    }

    var now: Double {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func set(_ value: Double) {
        lock.lock()
        self.value = value
        lock.unlock()
    }
}
