import FishMemoryAudioCore
import XCTest

final class VoiceOutputRouteTests: XCTestCase {
    func testDirectOutputNeedsNoMirror() {
        let route = VoiceOutputRoute(systemOutputUID: "headphones")
        XCTAssertEqual(route.voiceOutputUID, "headphones")
        XCTAssertTrue(route.mirrorOutputUIDs.isEmpty)
    }

    func testMultiOutputPinsVoiceToClockDeviceAndMirrorsOthersOnce() {
        let route = VoiceOutputRoute(
            systemOutputUID: "multi", mainOutputUID: "speakers",
            memberUIDs: ["blackhole", "speakers", "blackhole", "", "monitor"]
        )
        XCTAssertEqual(route.voiceOutputUID, "speakers")
        XCTAssertEqual(route.mirrorOutputUIDs, ["blackhole", "monitor"])
    }

    func testRemovingMemberDoesNotRetainOldMirrorOrInvalidClock() {
        let route = VoiceOutputRoute(
            systemOutputUID: "multi", mainOutputUID: "removed",
            memberUIDs: ["headphones"]
        )
        XCTAssertEqual(route.voiceOutputUID, "headphones")
        XCTAssertTrue(route.mirrorOutputUIDs.isEmpty)
    }
}
