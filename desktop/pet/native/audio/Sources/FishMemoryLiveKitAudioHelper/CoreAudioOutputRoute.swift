@preconcurrency import AVFoundation
import AudioToolbox
import CoreAudio
import FishMemoryAudioCore
import Foundation

enum CoreAudioOutputRoute {
    static func currentInputUID() throws -> String {
        let input = try scalar(
            AudioObjectID(kAudioObjectSystemObject),
            selector: kAudioHardwarePropertyDefaultInputDevice
        )
        return try string(input, selector: kAudioDevicePropertyDeviceUID)
    }

    static func bindVoiceOutput(_ route: VoiceOutputRoute, to engine: AVAudioEngine) throws {
        guard route.voiceOutputUID != route.systemOutputUID else { return }
        guard engine.inputNode.isVoiceProcessingEnabled,
              let unit = engine.inputNode.audioUnit else { return }
        let input = try scalar(
            AudioObjectID(kAudioObjectSystemObject),
            selector: kAudioHardwarePropertyDefaultInputDevice
        )
        let output = try deviceID(uid: route.voiceOutputUID)
        guard let outputUnit = engine.outputNode.audioUnit else {
            throw NSError(domain: "FishMemoryAudioRoute", code: 2)
        }
        // VPIO's output-device property selects its echo reference. The
        // engine's output node can own a distinct unit for actual playout.
        var bindings = [
            (unit, UInt32(1), input), (unit, UInt32(0), output),
        ]
        if outputUnit != unit { bindings.append((outputUnit, UInt32(0), output)) }
        for (target, element, selected) in bindings {
            var current: AudioDeviceID = 0
            var size = UInt32(MemoryLayout<AudioDeviceID>.size)
            try check(AudioUnitGetProperty(
                target, kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global, element, &current, &size
            ))
            if current != selected {
                var device = selected
                try check(AudioUnitSetProperty(
                    target, kAudioOutputUnitProperty_CurrentDevice,
                    kAudioUnitScope_Global, element, &device,
                    UInt32(MemoryLayout<AudioDeviceID>.size)
                ))
            }
        }
        if #available(macOS 14.0, *) {
            // Device binding can recreate VPIO after LiveKit applied these
            // preferences. Reapply the existing minimum-ducking policy to
            // the actual voice unit, without changing microphone processing.
            engine.inputNode.voiceProcessingOtherAudioDuckingConfiguration = .init(
                enableAdvancedDucking: false, duckingLevel: .min
            )
        }
    }

    static func current() throws -> VoiceOutputRoute {
        let output: AudioDeviceID = try scalar(
            AudioObjectID(kAudioObjectSystemObject),
            selector: kAudioHardwarePropertyDefaultOutputDevice
        )
        let uid = try string(output, selector: kAudioDevicePropertyDeviceUID)
        var address = address(kAudioAggregateDevicePropertyComposition)
        guard AudioObjectHasProperty(output, &address) else {
            return VoiceOutputRoute(systemOutputUID: uid)
        }
        let composition: CFDictionary = try object(
            output, selector: kAudioAggregateDevicePropertyComposition
        )
        let values = composition as NSDictionary
        // An aggregate with separate channel groups must keep its normal
        // channel mapping; only a macOS multi-output device mirrors one mix.
        guard (values[kAudioAggregateDeviceIsStackedKey] as? NSNumber)?.boolValue == true,
              let members = values[kAudioAggregateDeviceSubDeviceListKey]
                as? [[String: Any]]
        else { return VoiceOutputRoute(systemOutputUID: uid) }
        return VoiceOutputRoute(
            systemOutputUID: uid,
            mainOutputUID: values[kAudioAggregateDeviceMainSubDeviceKey] as? String,
            memberUIDs: members.compactMap { $0[kAudioSubDeviceUIDKey] as? String }
        )
    }

    static func deviceID(uid: String) throws -> AudioDeviceID {
        var address = address(kAudioHardwarePropertyDevices)
        var size: UInt32 = 0
        try check(AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size
        ))
        var devices = [AudioDeviceID](
            repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size
        )
        try check(devices.withUnsafeMutableBytes { bytes in
            AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject), &address, 0, nil,
                &size, bytes.baseAddress!
            )
        })
        for device in devices {
            if try string(device, selector: kAudioDevicePropertyDeviceUID) == uid {
                return device
            }
        }
        throw NSError(
            domain: "FishMemoryAudioRoute", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Audio output disappeared: \(uid)"]
        )
    }

    private static func address(
        _ selector: AudioObjectPropertySelector
    ) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
    }

    private static func scalar(
        _ device: AudioObjectID, selector: AudioObjectPropertySelector
    ) throws -> UInt32 {
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        var property = address(selector)
        try check(AudioObjectGetPropertyData(device, &property, 0, nil, &size, &value))
        return value
    }

    private static func object<T: AnyObject>(
        _ device: AudioObjectID, selector: AudioObjectPropertySelector
    ) throws -> T {
        var value: Unmanaged<T>?
        var size = UInt32(MemoryLayout<Unmanaged<T>?>.size)
        var property = address(selector)
        try check(AudioObjectGetPropertyData(device, &property, 0, nil, &size, &value))
        guard let value else { throw NSError(domain: NSOSStatusErrorDomain, code: -50) }
        return value.takeRetainedValue()
    }

    private static func string(
        _ device: AudioObjectID, selector: AudioObjectPropertySelector
    ) throws -> String {
        let value: CFString = try object(device, selector: selector)
        return value as String
    }

    private static func check(_ status: OSStatus) throws {
        guard status == noErr else {
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(status))
        }
    }
}
