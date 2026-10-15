/// Keep voice processing on one output device and mirror its rendered audio
/// to the other members of a system multi-output device.
public struct VoiceOutputRoute: Equatable, Sendable {
    public let systemOutputUID: String
    public let voiceOutputUID: String
    public let mirrorOutputUIDs: [String]

    public init(
        systemOutputUID: String,
        mainOutputUID: String? = nil,
        memberUIDs: [String] = []
    ) {
        self.systemOutputUID = systemOutputUID
        var seen: Set<String> = []
        let members = memberUIDs.filter { !$0.isEmpty && seen.insert($0).inserted }
        guard !members.isEmpty else {
            voiceOutputUID = systemOutputUID
            mirrorOutputUIDs = []
            return
        }
        let primary = mainOutputUID.flatMap { members.contains($0) ? $0 : nil }
            ?? members[0]
        voiceOutputUID = primary
        mirrorOutputUIDs = members.filter { $0 != primary }
    }
}
