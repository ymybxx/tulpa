import Foundation

public typealias ScreenTextAuthorizationValidator = @Sendable (
    ScreenTextPrivacyContext,
    ScreenTextApplicationContext
) -> Bool

public struct ScreenTextApplicationContext: Equatable, Sendable {
    public let processIdentifier: Int32
    public let appBundleId: String
    public let appName: String?
    public let windowTitle: String
    public let windowFrame: AccessibilityWindowFrame
    public let contextToken: String

    public init(
        processIdentifier: Int32,
        appBundleId: String,
        appName: String?,
        windowTitle: String,
        windowFrame: AccessibilityWindowFrame,
        contextToken: String = UUID().uuidString
    ) {
        self.processIdentifier = processIdentifier
        self.appBundleId = appBundleId
        self.appName = appName
        self.windowTitle = windowTitle
        self.windowFrame = windowFrame
        self.contextToken = contextToken
    }
}

/// Thread-safe join of NSWorkspace identity and recent Accessibility context.
/// It deliberately stores no focused value or OCR text.
public final class ScreenTextPrivacyContextStore: @unchecked Sendable {
    public typealias Clock = @Sendable () -> Double

    private struct WorkspaceContext {
        let processIdentifier: Int32
        let bundleIdentifier: String?
        let applicationName: String?
    }

    private struct AccessibilityContext {
        let observedAt: Double
        let processIdentifier: Int32
        let bundleIdentifier: String
        let applicationName: String?
        let windowTitle: String
        let windowFrame: AccessibilityWindowFrame?
        let secureField: Bool
        let contextToken: String
    }

    public let accessibilityFreshnessInterval: TimeInterval
    private let clock: Clock
    private let lock = NSLock()
    private var workspaceContext: WorkspaceContext?
    private var accessibilityContext: AccessibilityContext?

    public init(
        accessibilityFreshnessInterval: TimeInterval = 2,
        clock: @escaping Clock = { Date().timeIntervalSince1970 }
    ) {
        precondition(
            accessibilityFreshnessInterval > 0
                && accessibilityFreshnessInterval.isFinite
        )
        self.accessibilityFreshnessInterval = accessibilityFreshnessInterval
        self.clock = clock
    }

    /// Foreground switches and Space/session boundaries must invalidate the AX
    /// half of the join. A heartbeat for the same process may retain it.
    public func updateWorkspace(
        _ application: FrontmostApplication?,
        invalidateAccessibility: Bool
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard let application else {
            workspaceContext = nil
            accessibilityContext = nil
            return
        }
        let next = WorkspaceContext(
            processIdentifier: application.processIdentifier,
            bundleIdentifier: normalizedBundleIdentifier(
                application.bundleIdentifier
            ),
            applicationName: boundedMetadata(application.name, maximumLength: 128)
        )
        let identityChanged = !sameIdentity(workspaceContext, next)
        workspaceContext = next
        if invalidateAccessibility || identityChanged {
            accessibilityContext = nil
        }
    }

    public func updateAccessibility(_ observation: AccessibilityObservation) {
        lock.lock()
        defer { lock.unlock() }
        guard let workspaceContext,
              let bundleIdentifier = normalizedBundleIdentifier(
                  observation.bundleIdentifier
              ),
              workspaceContext.processIdentifier == observation.processIdentifier,
              workspaceContext.bundleIdentifier == bundleIdentifier
        else {
            accessibilityContext = nil
            return
        }
        let applicationName = boundedMetadata(
            observation.applicationName,
            maximumLength: 128
        )
        let windowTitle = boundedMetadata(
            observation.windowTitle,
            maximumLength: 256
        ) ?? ""
        let windowFrame = observation.windowFrame
        let contextToken: String
        if let current = accessibilityContext,
           current.processIdentifier == observation.processIdentifier,
           current.bundleIdentifier == bundleIdentifier,
           current.windowTitle == windowTitle,
           current.windowFrame == windowFrame,
           current.secureField == observation.secureField
        {
            // Value, selection, or non-secure focus events inside the same
            // window update freshness without revoking an in-flight capture.
            contextToken = current.contextToken
        } else {
            contextToken = UUID().uuidString
        }
        accessibilityContext = AccessibilityContext(
            observedAt: observation.observedAt,
            processIdentifier: observation.processIdentifier,
            bundleIdentifier: bundleIdentifier,
            applicationName: applicationName,
            windowTitle: windowTitle,
            windowFrame: windowFrame,
            secureField: observation.secureField,
            contextToken: contextToken
        )
    }

    public func clearWorkspace() {
        lock.lock()
        workspaceContext = nil
        accessibilityContext = nil
        lock.unlock()
    }

    public func clearAccessibility() {
        lock.lock()
        accessibilityContext = nil
        lock.unlock()
    }

    public func authorization(
        for candidate: ScreenTextPrivacyContext
    ) -> ScreenTextApplicationContext? {
        let now = clock()
        guard now.isFinite, candidate.observedAt.isFinite else { return nil }

        lock.lock()
        defer { lock.unlock() }
        guard let workspaceContext,
              let accessibilityContext,
              let workspaceBundleIdentifier = workspaceContext.bundleIdentifier,
              workspaceContext.processIdentifier
                  == accessibilityContext.processIdentifier,
              workspaceBundleIdentifier == accessibilityContext.bundleIdentifier,
              !accessibilityContext.secureField,
              !accessibilityContext.windowTitle.isEmpty,
              let windowFrame = accessibilityContext.windowFrame,
              isFinitePositiveWindowFrame(windowFrame)
        else { return nil }

        // The display stream is only a trigger. Actual pixels are captured
        // later from the independently matched SCWindow, so the latest AX
        // context may safely be newer than the display-change sample.
        let ageAtValidation = now - accessibilityContext.observedAt
        guard ageAtValidation >= 0,
              ageAtValidation <= accessibilityFreshnessInterval
        else { return nil }
        let appName = workspaceContext.applicationName
            ?? accessibilityContext.applicationName
        guard !ScreenTextSensitiveApplicationPolicy.shouldDeny(
            bundleIdentifier: workspaceBundleIdentifier,
            applicationName: appName
        ), !ScreenTextSensitiveWindowTitlePolicy.shouldDeny(
            accessibilityContext.windowTitle
        ) else { return nil }

        return ScreenTextApplicationContext(
            processIdentifier: accessibilityContext.processIdentifier,
            appBundleId: workspaceBundleIdentifier,
            appName: appName,
            windowTitle: accessibilityContext.windowTitle,
            windowFrame: windowFrame,
            contextToken: accessibilityContext.contextToken
        )
    }

    /// Revalidation for an already-authorized in-flight capture. Freshness is
    /// checked once against the captured frame in authorization(for:). Later
    /// OCR/encoding stages only require that the exact workspace/AX identity
    /// and context token remain current; slow local processing must not revoke
    /// an otherwise unchanged window.
    public func isStillAuthorized(
        _ expected: ScreenTextApplicationContext
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let workspaceContext,
              let accessibilityContext,
              let workspaceBundleIdentifier = workspaceContext.bundleIdentifier,
              workspaceContext.processIdentifier == expected.processIdentifier,
              workspaceContext.processIdentifier
                  == accessibilityContext.processIdentifier,
              workspaceBundleIdentifier == expected.appBundleId,
              workspaceBundleIdentifier == accessibilityContext.bundleIdentifier,
              !accessibilityContext.secureField,
              accessibilityContext.contextToken == expected.contextToken,
              accessibilityContext.windowTitle == expected.windowTitle,
              accessibilityContext.windowFrame == expected.windowFrame,
              !ScreenTextSensitiveApplicationPolicy.shouldDeny(
                bundleIdentifier: workspaceBundleIdentifier,
                applicationName: workspaceContext.applicationName
                    ?? accessibilityContext.applicationName
              ),
              !ScreenTextSensitiveWindowTitlePolicy.shouldDeny(
                accessibilityContext.windowTitle
              )
        else { return false }
        return true
    }

    private func sameIdentity(
        _ current: WorkspaceContext?,
        _ next: WorkspaceContext
    ) -> Bool {
        current?.processIdentifier == next.processIdentifier
            && current?.bundleIdentifier == next.bundleIdentifier
    }
}

enum ScreenTextSensitiveWindowTitlePolicy {
    static func shouldDeny(_ title: String) -> Bool {
        let normalized = title
            .precomposedStringWithCanonicalMapping
            .lowercased()
        return deniedFragments.contains(where: normalized.contains)
    }

    private static let deniedFragments = [
        "authentication",
        "authorization",
        "authorize",
        "api key",
        "credential",
        "credentials",
        "log in",
        "login",
        "one-time password",
        "passcode",
        "password",
        "secret",
        "sign in",
        "two-factor",
        "verification code",
        "2fa",
        "otp",
        "登录",
        "登陆",
        "密码",
        "口令",
        "凭证",
        "憑證",
        "密钥",
        "密鑰",
        "验证码",
        "双重认证",
        "身份验证",
        "授權",
        "授权",
    ]
}

enum ScreenTextSensitiveApplicationPolicy {
    static func shouldDeny(
        bundleIdentifier: String,
        applicationName: String?
    ) -> Bool {
        let bundle = bundleIdentifier.lowercased()
        if deniedBundleIdentifiers.contains(bundle)
            || deniedBundlePrefixes.contains(where: bundle.hasPrefix)
        {
            return true
        }
        guard let applicationName else { return false }
        let name = applicationName
            .precomposedStringWithCanonicalMapping
            .lowercased()
        return deniedApplicationNameFragments.contains(where: name.contains)
    }

    private static let deniedBundleIdentifiers: Set<String> = [
        // Browsers are denied wholesale because private-mode detection is not
        // reliable enough for a whole-display OCR stream.
        "com.apple.safari",
        "com.apple.safaritechnologypreview",
        "com.duckduckgo.macos.browser",
        "com.kagi.kagimacos",
        "company.thebrowser.browser",
        "org.chromium.chromium",
        "org.mozilla.firefox",
        "org.mozilla.firefoxdeveloperedition",
        "org.mozilla.nightly",
        "org.operasoftware.opera",
        "org.operasoftware.operagx",
        // Password, keychain, and authentication surfaces.
        "2bua8c4s2c.com.agilebits.onepassword7",
        "com.1password.1password",
        "com.agilebits.onepassword7",
        "com.apple.keychainaccess",
        "com.apple.loginwindow",
        "com.apple.passwords",
        "com.apple.securityagent",
        "com.bitwarden.desktop",
        "com.callpod.keepermac",
        "com.dashlane.dashlane",
        "com.lastpass.lastpass",
        "com.roboform.roboform",
        "in.sinew.enpass-desktop",
        "me.proton.pass",
        "org.keepassxc.keepassxc",
    ]

    private static let deniedBundlePrefixes = [
        "app.zen-browser.",
        "com.brave.browser",
        "com.google.chrome",
        "com.microsoft.edgemac",
        "com.operasoftware.opera",
        "com.vivaldi.vivaldi",
        "company.thebrowser.",
        "org.mozilla.firefox",
    ]

    private static let deniedApplicationNameFragments = [
        "1password",
        "bitwarden",
        "dashlane",
        "enpass",
        "firefox",
        "google chrome",
        "keychain access",
        "keepass",
        "keeper password",
        "lastpass",
        "microsoft edge",
        "opera gx",
        "password manager",
        "proton pass",
        "safari",
        "vivaldi",
        "钥匙串",
        "密碼管理",
        "密码管理",
    ]
}

private func normalizedBundleIdentifier(_ value: String?) -> String? {
    guard let value else { return nil }
    let normalized = value
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return normalized.isEmpty ? nil : normalized
}

private func boundedMetadata(
    _ value: String?,
    maximumLength: Int
) -> String? {
    guard let value else { return nil }
    let normalized = value
        .split(whereSeparator: { $0.isWhitespace })
        .joined(separator: " ")
    guard !normalized.isEmpty else { return nil }
    return String(normalized.prefix(maximumLength))
}

private func isFinitePositiveWindowFrame(
    _ frame: AccessibilityWindowFrame
) -> Bool {
    frame.x.isFinite && frame.y.isFinite
        && frame.width.isFinite && frame.height.isFinite
        && frame.width > 0 && frame.height > 0
}
