import AudioToolbox
import CoreAudio
import FishMemoryAudioCore
import Foundation

protocol RenderedAudioMirroring: AnyObject, Sendable {
    func append(_ pcm: Data)
    func clear()
}

/// Only a multi-output route needs these lightweight output-only HAL units.
/// The existing LiveKit/VPIO graph remains the sole voice-processing engine
/// and the sole authority for playback acknowledgements.
final class PlaybackOutputMirrors: RenderedAudioMirroring, @unchecked Sendable {
    private let configurationLock = NSLock()
    private let lock = NSLock()
    private var devices: [DevicePlaybackMirror] = []
    private var deviceUIDs: [String] = []

    func configure(_ uids: [String]) throws {
        configurationLock.lock()
        defer { configurationLock.unlock() }
        lock.lock()
        let unchanged = deviceUIDs == uids
        lock.unlock()
        guard !unchanged else { return }

        var replacements: [DevicePlaybackMirror] = []
        do {
            for uid in uids {
                replacements.append(try DevicePlaybackMirror(uid: uid))
            }
        } catch {
            replacements.forEach { $0.stop() }
            throw error
        }
        lock.lock()
        let previous = devices
        devices = replacements
        deviceUIDs = uids
        lock.unlock()
        previous.forEach { $0.stop() }
    }

    func append(_ pcm: Data) {
        for device in snapshot() { device.buffer.append(pcm) }
    }

    func clear() {
        for device in snapshot() { device.buffer.clear() }
    }

    func stop() {
        configurationLock.lock()
        defer { configurationLock.unlock() }
        lock.lock()
        let previous = devices
        devices.removeAll()
        deviceUIDs.removeAll()
        lock.unlock()
        previous.forEach { $0.stop() }
    }

    private func snapshot() -> [DevicePlaybackMirror] {
        lock.lock()
        defer { lock.unlock() }
        return devices
    }
}

private final class DevicePlaybackMirror: @unchecked Sendable {
    let buffer = PCM16MirrorBuffer()
    private var unit: AudioUnit?

    init(uid: String) throws {
        var device = try CoreAudioOutputRoute.deviceID(uid: uid)
        var description = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0, componentFlagsMask: 0
        )
        guard let component = AudioComponentFindNext(nil, &description) else {
            throw NSError(domain: "FishMemoryAudioRoute", code: 3)
        }
        var created: AudioUnit?
        try check(AudioComponentInstanceNew(component, &created))
        guard let created else { throw NSError(domain: "FishMemoryAudioRoute", code: 3) }
        unit = created
        do {
            var disabled: UInt32 = 0
            try check(AudioUnitSetProperty(
                created, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1,
                &disabled, UInt32(MemoryLayout<UInt32>.size)
            ))
            try check(AudioUnitSetProperty(
                created, kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global, 0, &device,
                UInt32(MemoryLayout<AudioDeviceID>.size)
            ))
            var format = AudioStreamBasicDescription(
                mSampleRate: 16_000, mFormatID: kAudioFormatLinearPCM,
                mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                mBytesPerPacket: 8, mFramesPerPacket: 1, mBytesPerFrame: 8,
                mChannelsPerFrame: 2, mBitsPerChannel: 32, mReserved: 0
            )
            try check(AudioUnitSetProperty(
                created, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0,
                &format, UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
            ))
            var callback = AURenderCallbackStruct(
                inputProc: { context, flags, _, _, frameCount, output in
                    guard let output else { return noErr }
                    let buffers = UnsafeMutableAudioBufferListPointer(output)
                    guard buffers.count == 1, buffers[0].mNumberChannels == 2,
                          let data = buffers[0].mData else {
                        return kAudioUnitErr_FormatNotSupported
                    }
                    let buffer = Unmanaged<PCM16MirrorBuffer>.fromOpaque(context)
                        .takeUnretainedValue()
                    let count = buffer.render(
                        into: data.assumingMemoryBound(to: Float.self),
                        frames: Int(frameCount), channels: 2
                    )
                    if count == 0 {
                        flags.pointee.insert(.unitRenderAction_OutputIsSilence)
                    } else {
                        flags.pointee.remove(.unitRenderAction_OutputIsSilence)
                    }
                    return noErr
                },
                inputProcRefCon: Unmanaged.passUnretained(buffer).toOpaque()
            )
            try check(AudioUnitSetProperty(
                created, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0,
                &callback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)
            ))
            try check(AudioUnitInitialize(created))
            try check(AudioOutputUnitStart(created))
            var activeDevice: AudioDeviceID = 0
            var size = UInt32(MemoryLayout<AudioDeviceID>.size)
            try check(AudioUnitGetProperty(
                created, kAudioOutputUnitProperty_CurrentDevice,
                kAudioUnitScope_Global, 0, &activeDevice, &size
            ))
            guard activeDevice == device else {
                throw NSError(domain: "FishMemoryAudioRoute", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "Auxiliary output did not bind to \(uid)"])
            }
        } catch {
            stop()
            throw error
        }
    }

    func stop() {
        if let unit {
            AudioOutputUnitStop(unit)
            AudioUnitUninitialize(unit)
            AudioComponentInstanceDispose(unit)
            self.unit = nil
        }
        buffer.clear()
    }

    deinit { stop() }

    private func check(_ status: OSStatus) throws {
        guard status == noErr else {
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(status))
        }
    }
}
