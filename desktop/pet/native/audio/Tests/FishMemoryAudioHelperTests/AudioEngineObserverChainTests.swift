@preconcurrency import AVFoundation
import FishMemoryAudioCore
@testable import FishMemoryLiveKitAudioHelper
import LiveKit
import XCTest

final class AudioEngineObserverChainTests: XCTestCase {
    func testInstallSelectsOnlyCustomObserverAndClearsStaleLinks() {
        let custom = ObserverSpy()
        let fallback = ObserverSpy()
        let staleCustomTail = ObserverSpy()
        let staleFallbackTail = ObserverSpy()
        custom.next = staleCustomTail
        fallback.next = staleFallbackTail
        var selected: [any AudioEngineObserver] = []

        AudioEngineObserverChain.install(
            custom,
            replacing: fallback,
            setObservers: { selected = $0 }
        )

        XCTAssertEqual(selected.count, 1)
        XCTAssertTrue((selected.first as? ObserverSpy) === custom)
        XCTAssertNil(custom.next)
        XCTAssertNil(fallback.next)
    }

    func testRestoreSelectsOnlyDefaultObserverAndClearsBothLinks() {
        let custom = ObserverSpy()
        let fallback = ObserverSpy()
        let staleTail = ObserverSpy()
        custom.next = fallback
        fallback.next = staleTail
        var selected: [any AudioEngineObserver] = []

        AudioEngineObserverChain.restore(
            fallback,
            removing: custom,
            setObservers: { selected = $0 }
        )

        XCTAssertEqual(selected.count, 1)
        XCTAssertTrue((selected.first as? ObserverSpy) === fallback)
        XCTAssertNil(custom.next)
        XCTAssertNil(fallback.next)
    }

    func testBypassControlReportsUnavailableBeforeCaptureStarts() throws {
        let writer = BoundedFrameWriter(capacity: 1) { _ in }
        let capture = LiveKitMicrophoneCapture(
            writer: writer,
            onFatalError: { _ in }
        )

        XCTAssertNil(try capture.setVoiceProcessingBypassed(true))
        XCTAssertNil(try capture.setVoiceProcessingBypassed(false))
    }
}

private final class ObserverSpy: AudioEngineObserver, @unchecked Sendable {
    var next: (any AudioEngineObserver)?
}
