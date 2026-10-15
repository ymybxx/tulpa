import CoreAudio
import Foundation

/// Device changes are events, never a microphone-frame polling loop. The
/// capture owner compares the resulting route before doing any reconnect.
final class DefaultAudioDeviceMonitor: @unchecked Sendable {
    private let queue = DispatchQueue(label: "fish-memory.default-audio-device")
    private let listener: AudioObjectPropertyListenerBlock
    private var properties: [AudioObjectPropertyAddress] = []

    init(onChange: @escaping @Sendable () -> Void) {
        listener = { _, _ in onChange() }
    }

    func start() throws {
        do {
            for selector in [
                kAudioHardwarePropertyDefaultInputDevice,
                kAudioHardwarePropertyDefaultOutputDevice,
            ] {
                var address = AudioObjectPropertyAddress(
                    mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                    mElement: kAudioObjectPropertyElementMain
                )
                let status = AudioObjectAddPropertyListenerBlock(
                    AudioObjectID(kAudioObjectSystemObject), &address, queue, listener
                )
                guard status == noErr else {
                    throw NSError(domain: NSOSStatusErrorDomain, code: Int(status))
                }
                properties.append(address)
            }
        } catch {
            stop()
            throw error
        }
    }

    func stop() {
        for var address in properties {
            AudioObjectRemovePropertyListenerBlock(
                AudioObjectID(kAudioObjectSystemObject), &address, queue, listener
            )
        }
        properties.removeAll()
    }

    deinit { stop() }
}
