import CoreAudio
import Darwin
import Foundation
import AVFAudio

struct SystemAudioProcessIdentity: Equatable, Sendable {
    let objectID: AudioObjectID
    let processIdentifier: pid_t
    let parentProcessIdentifier: pid_t?
    let bundleIdentifier: String?
    let executablePath: String?
    let ancestors: [pid_t]
}

struct CoreAudioTapBufferingPolicy: Equatable, Sendable {
    static let totalMemoryLimit = 4 * 1_024 * 1_024
    static let targetBufferedSeconds = 2.0

    let slotCount: Int
    let slotCapacity: Int

    var totalCapacity: Int { slotCount * slotCapacity }

    static func make(
        sampleRate: Double,
        bytesPerFrame: Int,
        currentBufferFrames: Int,
        maximumBufferFrames: Int
    ) -> CoreAudioTapBufferingPolicy? {
        guard sampleRate.isFinite, sampleRate >= 8_000,
              bytesPerFrame > 0,
              currentBufferFrames > 0,
              maximumBufferFrames >= currentBufferFrames else { return nil }
        let requiredCapacity = maximumBufferFrames.multipliedReportingOverflow(
            by: bytesPerFrame
        )
        guard !requiredCapacity.overflow,
              requiredCapacity.partialValue > 0,
              requiredCapacity.partialValue <= totalMemoryLimit else { return nil }
        let slotCapacity = ((requiredCapacity.partialValue + 4_095) / 4_096) * 4_096
        let maximumSlots = totalMemoryLimit / slotCapacity
        guard maximumSlots >= 8 else { return nil }
        let desiredSlots = Int(ceil(
            sampleRate * targetBufferedSeconds / Double(currentBufferFrames)
        ))
        return .init(
            slotCount: min(512, max(8, min(desiredSlots, maximumSlots))),
            slotCapacity: slotCapacity
        )
    }
}

func excludedSystemAudioProcessObjectIDs(
    from processes: [SystemAudioProcessIdentity],
    hostProcessIdentifier: pid_t
) -> [AudioObjectID] {
    processes.compactMap { process in
        shouldExcludeSystemAudioApplication(
            bundleIdentifier: process.bundleIdentifier,
            processIdentifier: process.processIdentifier,
            parentProcessIdentifier: hostProcessIdentifier,
            executablePath: process.executablePath,
            ancestors: process.ancestors
        ) ? process.objectID : nil
    }.sorted()
}

/// The Core Audio process-list callback runs outside the main actor. Keeping
/// its revision in a tiny lock-protected object lets one coalesced refresh read
/// until the process catalog is stable without spawning overlapping scans.
final class SystemAudioProcessListRevision: @unchecked Sendable {
    private let lock = NSLock()
    private var storedValue = 0

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return storedValue
    }

    @discardableResult
    func recordChange() -> Int {
        lock.lock()
        storedValue &+= 1
        let value = storedValue
        lock.unlock()
        return value
    }
}

struct CoreAudioTapCleanupState: Equatable, Sendable {
    var ioStarted: Bool
    var ownsIOProc: Bool
    var ownsAggregateDevice: Bool
    var ownsTap: Bool

    var isComplete: Bool {
        !ioStarted && !ownsIOProc && !ownsAggregateDevice && !ownsTap
    }

    /// HAL objects have a strict dependency order.  A failed step stops this
    /// pass and leaves that handle, plus every dependent handle, owned so a
    /// later retry can finish the cleanup instead of leaking an object.
    mutating func cleanup(using operations: CoreAudioTapCleanupOperations) -> CoreAudioTapCleanupFailure? {
        if ioStarted {
            let status = operations.stopIO()
            guard coreAudioStopSucceeded(status) else {
                return .init(step: "stop_io", status: status)
            }
            ioStarted = false
        }
        if ownsIOProc {
            let status = operations.destroyIOProc()
            guard coreAudioDestroySucceeded(status) else {
                return .init(step: "destroy_io_proc", status: status)
            }
            ownsIOProc = false
        }
        if ownsAggregateDevice {
            let status = operations.destroyAggregateDevice()
            guard coreAudioDestroySucceeded(status) else {
                return .init(step: "destroy_aggregate_device", status: status)
            }
            ownsAggregateDevice = false
        }
        if ownsTap {
            let status = operations.destroyTap()
            guard coreAudioDestroySucceeded(status) else {
                return .init(step: "destroy_process_tap", status: status)
            }
            ownsTap = false
        }
        return nil
    }
}

struct CoreAudioTapCleanupOperations {
    let stopIO: () -> OSStatus
    let destroyIOProc: () -> OSStatus
    let destroyAggregateDevice: () -> OSStatus
    let destroyTap: () -> OSStatus
}

struct CoreAudioTapCleanupFailure: Equatable, Sendable {
    let step: String
    let status: OSStatus
}

struct CoreAudioTapCleanupResult: Sendable {
    let state: CoreAudioTapCleanupState
    let failure: CoreAudioTapCleanupFailure?
}

func coreAudioDestroySucceeded(_ status: OSStatus) -> Bool {
    status == noErr || status == kAudioHardwareBadObjectError
}

func coreAudioStopSucceeded(_ status: OSStatus) -> Bool {
    coreAudioDestroySucceeded(status) || status == kAudioHardwareNotRunningError
}

/// Potentially blocking HAL control calls never execute on the main actor.
/// One serial queue also guarantees that a stop/destroy requested during the
/// system TCC prompt runs only after AudioDeviceStart has returned.
final class CoreAudioHALControlQueue: @unchecked Sendable {
    private let queue = DispatchQueue(
        label: "app.fishmemory.perception.audio.hal-control",
        qos: .userInitiated
    )

    func perform<Result: Sendable>(
        _ operation: @escaping @Sendable () -> Result
    ) async -> Result {
        await withCheckedContinuation { continuation in
            queue.async {
                continuation.resume(returning: operation())
            }
        }
    }
}

/// Recovery, disable, and helper shutdown can all request teardown while an
/// AudioDeviceStart call is suspended in TCC. They must join one cleanup pass;
/// serializing only the HAL calls is insufficient because two callers could
/// otherwise snapshot the same handles and later race their MainActor state
/// writeback.
@MainActor
final class SystemAudioCleanupSingleFlight {
    private var active: Task<Bool, Never>?
    private var epoch = 0

    func run(
        _ operation: @escaping @MainActor @Sendable () async -> Bool
    ) async -> Bool {
        if let active { return await active.value }
        epoch &+= 1
        let token = epoch
        let task = Task { await operation() }
        active = task
        let result = await task.value
        if epoch == token { active = nil }
        return result
    }
}

@available(macOS 14.2, *)
@MainActor
final class CoreAudioProcessTapMonitor: SystemAudioMonitoring {
    private let callbacks: SystemAudioCallbacks
    private let hostProcessIdentifier: pid_t
    private let halControl = CoreAudioHALControlQueue()
    private let cleanupSingleFlight = SystemAudioCleanupSingleFlight()
    private let processChangeQueue = DispatchQueue(
        label: "app.fishmemory.perception.audio.process-list",
        qos: .utility
    )
    private var tapDescription: CATapDescription?
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateDeviceID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private var ioStarted = false
    private var ioStartPending = false
    private var samples: CoreAudioTapSamples?
    private var processListListener: AudioObjectPropertyListenerBlock?
    private var aggregateAliveListener: AudioObjectPropertyListenerBlock?
    private var refreshTask: Task<Void, Never>?
    private var runtimeRecoveryTask: Task<Void, Never>?
    private var excludedProcessObjectIDs: [AudioObjectID] = []
    private var processListRevision = SystemAudioProcessListRevision()
    private var deliveryActive = false
    private var runtimeRecoveryRemaining = 1
    private var lastCleanupFailure: CoreAudioTapCleanupFailure?
    private var generation = 0

    init(callbacks: SystemAudioCallbacks, hostProcessIdentifier: pid_t = getppid()) {
        self.callbacks = callbacks
        self.hostProcessIdentifier = hostProcessIdentifier
    }

    func start() async -> CapabilityReport {
        runtimeRecoveryRemaining = 1
        return await startCapture()
    }

    private func startCapture() async -> CapabilityReport {
        if ioStarted { return report(.running) }
        if hasOwnedCaptureResources, !(await teardownCapture()) {
            return report(.failed, reason: "core_audio_process_tap_cleanup_failed")
        }
        generation += 1
        let token = generation
        processListRevision = SystemAudioProcessListRevision()
        do {
            let loadedIdentities = await Task.detached(priority: .userInitiated) {
                loadSystemAudioProcessIdentities()
            }.value
            guard let identities = loadedIdentities else {
                throw CoreAudioProcessTapError.processListUnavailable
            }
            guard generation == token, !Task.isCancelled else { return report(.disabled) }
            let excluded = excludedSystemAudioProcessObjectIDs(
                from: identities,
                hostProcessIdentifier: hostProcessIdentifier
            )
            let format = try createTapAndAggregate(excluding: excluded)
            try await waitForAggregateDeviceReady(token: token)
            excludedProcessObjectIDs = excluded
            try installProcessListListener(token: token)
            try installAggregateAliveListener(token: token)

            // Subscribe before re-enumerating so the startup snapshot cannot
            // miss a Fish audio helper that appears while the tap is built.
            _ = try await reconcileProcessListUntilStable(token: token)
            try createIO(format: format)
            try await startIO(token: token)
            guard generation == token, !Task.isCancelled else {
                _ = await teardownCapture()
                return report(.disabled)
            }
            try probeAudioTapPermission()

            // AudioDeviceStart is what triggers the system-audio TCC prompt.
            // Samples remain disabled through that call and through this final
            // reconciliation, so no stale-filter chunk can escape.
            _ = try await reconcileProcessListUntilStable(token: token)
            guard generation == token, !Task.isCancelled else {
                _ = await teardownCapture()
                return report(.disabled)
            }
            guard coreAudioDeviceIsAlive(aggregateDeviceID) == true else {
                throw CoreAudioProcessTapError.aggregateDeviceNotReady
            }
            samples?.activate()
            deliveryActive = true
            return report(.running)
        } catch CoreAudioProcessTapError.permissionRequired {
            guard generation == token else { return report(.disabled) }
            _ = await teardownCapture()
            return CapabilityReport(
                supported: true,
                requested: true,
                state: .unavailable,
                permission: .notDetermined,
                reason: "core_audio_process_tap_permission_required"
            )
        } catch {
            guard generation == token else { return report(.disabled) }
            let cleaned = await teardownCapture()
            return report(
                .failed,
                reason: cleaned
                    ? "core_audio_process_tap_start_failed"
                    : "core_audio_process_tap_cleanup_failed"
            )
        }
    }

    func stop() async -> Bool {
        generation += 1
        runtimeRecoveryTask?.cancel()
        runtimeRecoveryTask = nil
        return await teardownCapture()
    }

    private func createTapAndAggregate(
        excluding processObjectIDs: [AudioObjectID]
    ) throws -> AudioStreamBasicDescription {
        precondition(tapID == kAudioObjectUnknown)
        precondition(aggregateDeviceID == kAudioObjectUnknown)
        precondition(ioProcID == nil)

        let description = CATapDescription(
            monoGlobalTapButExcludeProcesses: processObjectIDs
        )
        description.name = "Fish Memory System Audio"
        description.isPrivate = true
        description.muteBehavior = .unmuted

        var nextTapID = AudioObjectID(kAudioObjectUnknown)
        try requireCoreAudioSuccess(
            AudioHardwareCreateProcessTap(description, &nextTapID),
            step: "create_process_tap"
        )
        guard nextTapID != kAudioObjectUnknown else {
            throw CoreAudioProcessTapError.invalidObject("create_process_tap")
        }
        tapDescription = description
        tapID = nextTapID

        let format = try coreAudioTapFormat(tapID: nextTapID)

        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Fish Memory System Audio",
            kAudioAggregateDeviceUIDKey: "app.fishmemory.perception.tap.\(UUID().uuidString)",
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            // Start the IOProc immediately. Auto-start would wait for another
            // process to begin playing, making enable/disable hang on silence.
            kAudioAggregateDeviceTapAutoStartKey: false,
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapUIDKey: description.uuid.uuidString,
                kAudioSubTapDriftCompensationKey: true,
            ]],
        ]
        var nextAggregateID = AudioObjectID(kAudioObjectUnknown)
        try requireCoreAudioSuccess(
            AudioHardwareCreateAggregateDevice(
                aggregateDescription as CFDictionary,
                &nextAggregateID
            ),
            step: "create_aggregate_device"
        )
        guard nextAggregateID != kAudioObjectUnknown else {
            throw CoreAudioProcessTapError.invalidObject("create_aggregate_device")
        }
        aggregateDeviceID = nextAggregateID

        return format
    }

    private func waitForAggregateDeviceReady(token: Int) async throws {
        for _ in 0..<60 {
            guard generation == token, !Task.isCancelled else {
                throw CancellationError()
            }
            if coreAudioDeviceIsAlive(aggregateDeviceID) == true { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw CoreAudioProcessTapError.aggregateDeviceNotReady
    }

    private func createIO(format: AudioStreamBasicDescription) throws {
        let nextAggregateID = aggregateDeviceID
        guard nextAggregateID != kAudioObjectUnknown else {
            throw CoreAudioProcessTapError.invalidObject("create_io_proc")
        }
        let bufferingPolicy = try coreAudioTapBufferingPolicy(
            deviceID: nextAggregateID,
            format: format
        )
        let nextSamples = try CoreAudioTapSamples(
            format: format,
            callbacks: callbacks,
            bufferingPolicy: bufferingPolicy
        )
        var nextIOProcID: AudioDeviceIOProcID?
        try requireCoreAudioSuccess(
            AudioDeviceCreateIOProcIDWithBlock(
                &nextIOProcID,
                nextAggregateID,
                nextSamples.ioQueue
            ) { [weak nextSamples] _, inputData, _, _, _ in
                nextSamples?.capture(inputData: inputData)
            },
            step: "create_io_proc"
        )
        guard let nextIOProcID else {
            throw CoreAudioProcessTapError.invalidObject("create_io_proc")
        }
        samples = nextSamples
        ioProcID = nextIOProcID
    }

    private func startIO(token: Int) async throws {
        guard aggregateDeviceID != kAudioObjectUnknown, let ioProcID else {
            throw CoreAudioProcessTapError.invalidObject("start_io")
        }
        let targetDeviceID = aggregateDeviceID
        ioStartPending = true
        let status = await halControl.perform {
            AudioDeviceStart(targetDeviceID, ioProcID)
        }
        guard generation == token, !Task.isCancelled else {
            // A concurrent stop owns the queued cleanup. Do not publish the
            // result of this stale start or clear its pending-stop marker.
            throw CancellationError()
        }
        ioStartPending = false
        try requireCoreAudioSuccess(status, step: "start_io")
        ioStarted = true
    }

    private func probeAudioTapPermission() throws {
        guard tapID != kAudioObjectUnknown else {
            throw CoreAudioProcessTapError.invalidObject("probe_tap_permission")
        }
        var address = coreAudioPropertyAddress(selector: kAudioTapPropertyDescription)
        var retainedDescription: Unmanaged<CATapDescription>?
        var size = UInt32(MemoryLayout<CATapDescription?>.size)
        let getStatus = AudioObjectGetPropertyData(
            tapID, &address, 0, nil, &size, &retainedDescription
        )
        guard getStatus == noErr, let retainedDescription else {
            throw CoreAudioProcessTapError.permissionRequired
        }
        let description = retainedDescription.takeRetainedValue()
        var value: CATapDescription? = description
        let setStatus = withUnsafePointer(to: &value) { pointer in
            AudioObjectSetPropertyData(
                tapID,
                &address,
                0,
                nil,
                UInt32(MemoryLayout<CATapDescription?>.size),
                pointer
            )
        }
        guard setStatus == noErr else {
            throw CoreAudioProcessTapError.permissionRequired
        }
        tapDescription = description
    }

    private func installProcessListListener(token: Int) throws {
        var address = coreAudioPropertyAddress(
            selector: kAudioHardwarePropertyProcessObjectList
        )
        let revision = processListRevision
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            revision.recordChange()
            Task { @MainActor [weak self] in
                self?.scheduleProcessListRefresh(token: token)
            }
        }
        try requireCoreAudioSuccess(
            AudioObjectAddPropertyListenerBlock(
                AudioObjectID(kAudioObjectSystemObject),
                &address,
                processChangeQueue,
                listener
            ),
            step: "install_process_listener"
        )
        processListListener = listener
    }

    /// Re-read until no subscribed process-list event overlaps the read.  The
    /// listener is installed first, which closes the enumerate/subscribe race.
    /// A bounded retry avoids hanging startup if another application is
    /// continuously creating short-lived audio clients.
    private func reconcileProcessListUntilStable(token: Int) async throws -> Int {
        for _ in 0..<8 {
            guard generation == token, !Task.isCancelled else {
                throw CancellationError()
            }
            let revisionBeforeRead = processListRevision.value
            let identities = await Task.detached(priority: .userInitiated) {
                loadSystemAudioProcessIdentities()
            }.value
            guard generation == token, !Task.isCancelled else {
                throw CancellationError()
            }
            guard let identities else {
                throw CoreAudioProcessTapError.processListUnavailable
            }
            let excluded = excludedSystemAudioProcessObjectIDs(
                from: identities,
                hostProcessIdentifier: hostProcessIdentifier
            )
            if excluded != excludedProcessObjectIDs {
                try updateExcludedProcesses(excluded)
                excludedProcessObjectIDs = excluded
                samples?.markGap(reason: "process_filter_changed")
            }

            // Drain callbacks that were already queued before checking the
            // revision. Events arriving later still invalidate the acceptance
            // revision checked by every HAL callback.
            processChangeQueue.sync {}
            let revisionAfterRead = processListRevision.value
            if revisionAfterRead == revisionBeforeRead {
                return revisionAfterRead
            }
        }
        throw CoreAudioProcessTapError.processListUnstable
    }

    private func scheduleProcessListRefresh(token: Int) {
        // Startup performs its own subscribed reconciliation while delivery is
        // closed. Runtime refreshes begin only after sample delivery is active.
        guard token == generation, ioStarted, deliveryActive else { return }
        // One refresh absorbs all revision changes that arrive while its
        // subscribed reconciliation is running. Detached process scans cannot
        // be cancelled once in proc_pidinfo, so replacing tasks here would pile
        // up scans.
        guard refreshTask == nil else { return }
        refreshTask = Task { [weak self] in
            do {
                guard let self, token == self.generation, self.ioStarted else { return }
                let acceptedRevision = try await self.reconcileProcessListUntilStable(token: token)
                guard token == self.generation, self.ioStarted, self.deliveryActive else { return }
                self.refreshTask = nil
                if self.processListRevision.value != acceptedRevision {
                    self.scheduleProcessListRefresh(token: token)
                }
            } catch is CancellationError {
                guard let self, token == self.generation else { return }
                self.refreshTask = nil
                return
            } catch {
                guard let self, token == self.generation else { return }
                self.refreshTask = nil
                self.beginRuntimeRecovery(reason: "core_audio_process_filter_update_failed")
            }
        }
    }

    private func installAggregateAliveListener(token: Int) throws {
        guard aggregateDeviceID != kAudioObjectUnknown else {
            throw CoreAudioProcessTapError.invalidObject("install_alive_listener")
        }
        var address = coreAudioPropertyAddress(
            selector: kAudioDevicePropertyDeviceIsAlive
        )
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            Task { @MainActor [weak self] in
                self?.handleAggregateAliveChange(token: token)
            }
        }
        try requireCoreAudioSuccess(
            AudioObjectAddPropertyListenerBlock(
                aggregateDeviceID,
                &address,
                processChangeQueue,
                listener
            ),
            step: "install_alive_listener"
        )
        aggregateAliveListener = listener
    }

    private func handleAggregateAliveChange(token: Int) {
        guard token == generation,
              aggregateDeviceID != kAudioObjectUnknown,
              coreAudioDeviceIsAlive(aggregateDeviceID) == false else { return }
        beginRuntimeRecovery(reason: "core_audio_aggregate_device_stopped")
    }

    /// Runtime HAL failures get one serialized rebuild of the same exclusive
    /// tap.  We never switch to ScreenCaptureKit on 14.2+, because a TCC denial
    /// cannot be reliably distinguished from other AudioDeviceStart errors and
    /// SCK would reintroduce the pet's own playback into the captured mix.
    private func beginRuntimeRecovery(reason: String) {
        guard deliveryActive, runtimeRecoveryTask == nil else { return }
        guard runtimeRecoveryRemaining > 0 else {
            callbacks.onStatus(report(.failed, reason: reason))
            return
        }
        runtimeRecoveryRemaining -= 1
        generation += 1
        let recoveryToken = generation
        deliveryActive = false
        samples?.invalidate()
        runtimeRecoveryTask = Task { [weak self] in
            guard let self else { return }
            guard !Task.isCancelled, self.generation == recoveryToken else { return }
            let cleaned = await self.teardownCapture()
            guard !Task.isCancelled, self.generation == recoveryToken else { return }
            guard cleaned else {
                self.runtimeRecoveryTask = nil
                self.callbacks.onStatus(self.report(
                    .failed,
                    reason: "core_audio_process_tap_cleanup_failed"
                ))
                return
            }
            let result = await self.startCapture()
            guard !Task.isCancelled else { return }
            self.runtimeRecoveryTask = nil
            self.callbacks.onStatus(result)
        }
    }

    private func updateExcludedProcesses(_ objectIDs: [AudioObjectID]) throws {
        guard tapID != kAudioObjectUnknown, let tapDescription else {
            throw CoreAudioProcessTapError.invalidObject("update_process_filter")
        }
        tapDescription.processes = objectIDs
        var description: CATapDescription? = tapDescription
        var address = coreAudioPropertyAddress(selector: kAudioTapPropertyDescription)
        let status = withUnsafePointer(to: &description) { pointer in
            AudioObjectSetPropertyData(
                tapID,
                &address,
                0,
                nil,
                UInt32(MemoryLayout<CATapDescription?>.size),
                pointer
            )
        }
        try requireCoreAudioSuccess(status, step: "update_process_filter")
    }

    private var hasOwnedCaptureResources: Bool {
        ioStarted || ioStartPending || ioProcID != nil
            || aggregateDeviceID != kAudioObjectUnknown
            || tapID != kAudioObjectUnknown || processListListener != nil
            || aggregateAliveListener != nil
    }

    /// Core Audio teardown calls can transiently fail while HAL is publishing a
    /// device-state change. Retry a few times, but retain every handle whose
    /// cleanup step did not succeed so a subsequent stop can try again.
    private func teardownCapture(maxAttempts: Int = 3) async -> Bool {
        await cleanupSingleFlight.run { [self] in
            await performTeardown(maxAttempts: maxAttempts)
        }
    }

    private func performTeardown(maxAttempts: Int) async -> Bool {
        for attempt in 0..<maxAttempts {
            if await teardownCaptureOnce() { return true }
            if attempt + 1 < maxAttempts {
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        return !hasOwnedCaptureResources
    }

    private func teardownCaptureOnce() async -> Bool {
        refreshTask?.cancel()
        refreshTask = nil
        deliveryActive = false
        samples?.invalidate()

        let processListenerRemoved = removeProcessListListener()
        let aliveListenerRemoved = removeAggregateAliveListener()
        let currentIOProcID = ioProcID
        let currentAggregateID = aggregateDeviceID
        let currentTapID = tapID
        let initialState = CoreAudioTapCleanupState(
            // If AudioDeviceStart is waiting on TCC, this cleanup pass is
            // queued behind it. Stop unconditionally after it returns; a
            // failed start yields NotRunning, which is an accepted stop state.
            ioStarted: ioStarted || ioStartPending,
            ownsIOProc: currentIOProcID != nil,
            ownsAggregateDevice: currentAggregateID != kAudioObjectUnknown,
            ownsTap: currentTapID != kAudioObjectUnknown
        )
        let cleanup = await halControl.perform {
            var state = initialState
            let failure = state.cleanup(using: .init(
                stopIO: {
                    guard currentAggregateID != kAudioObjectUnknown,
                          let currentIOProcID else { return kAudio_ParamError }
                    return AudioDeviceStop(currentAggregateID, currentIOProcID)
                },
                destroyIOProc: {
                    guard currentAggregateID != kAudioObjectUnknown,
                          let currentIOProcID else { return kAudio_ParamError }
                    return AudioDeviceDestroyIOProcID(currentAggregateID, currentIOProcID)
                },
                destroyAggregateDevice: {
                    guard currentAggregateID != kAudioObjectUnknown else {
                        return kAudio_ParamError
                    }
                    return AudioHardwareDestroyAggregateDevice(currentAggregateID)
                },
                destroyTap: {
                    guard currentTapID != kAudioObjectUnknown else {
                        return kAudio_ParamError
                    }
                    return AudioHardwareDestroyProcessTap(currentTapID)
                }
            ))
            return CoreAudioTapCleanupResult(state: state, failure: failure)
        }
        let state = cleanup.state
        lastCleanupFailure = cleanup.failure

        ioStartPending = false
        ioStarted = state.ioStarted
        if !state.ownsIOProc {
            ioProcID = nil
            samples = nil
        }
        if !state.ownsAggregateDevice {
            aggregateDeviceID = AudioObjectID(kAudioObjectUnknown)
            // Destroying the source object also removes its property listener.
            aggregateAliveListener = nil
        }
        if !state.ownsTap {
            tapID = AudioObjectID(kAudioObjectUnknown)
            tapDescription = nil
            excludedProcessObjectIDs.removeAll(keepingCapacity: false)
        }
        if state.isComplete { lastCleanupFailure = nil }
        return state.isComplete && processListenerRemoved
            && (aliveListenerRemoved || aggregateAliveListener == nil)
    }

    @discardableResult
    private func removeProcessListListener() -> Bool {
        guard let listener = processListListener else { return true }
        var address = coreAudioPropertyAddress(
            selector: kAudioHardwarePropertyProcessObjectList
        )
        let status = AudioObjectRemovePropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            processChangeQueue,
            listener
        )
        guard coreAudioDestroySucceeded(status) else { return false }
        processListListener = nil
        return true
    }

    @discardableResult
    private func removeAggregateAliveListener() -> Bool {
        guard let listener = aggregateAliveListener,
              aggregateDeviceID != kAudioObjectUnknown else {
            if aggregateDeviceID == kAudioObjectUnknown {
                aggregateAliveListener = nil
            }
            return true
        }
        var address = coreAudioPropertyAddress(
            selector: kAudioDevicePropertyDeviceIsAlive
        )
        let status = AudioObjectRemovePropertyListenerBlock(
            aggregateDeviceID,
            &address,
            processChangeQueue,
            listener
        )
        guard coreAudioDestroySucceeded(status) else { return false }
        aggregateAliveListener = nil
        return true
    }

    private func report(_ state: CapabilityState, reason: String? = nil) -> CapabilityReport {
        CapabilityReport(
            supported: true,
            requested: state != .disabled,
            state: state,
            permission: state == .running ? .granted : .notDetermined,
            reason: reason
        )
    }
}

@available(macOS 14.2, *)
private final class CoreAudioTapSamples: @unchecked Sendable {
    let ioQueue = DispatchQueue(
        label: "app.fishmemory.perception.audio.tap-io",
        qos: .userInitiated
    )
    private let processingQueue: DispatchQueue
    private let processor: CoreAudioTapSampleProcessor
    // Owned exclusively by ioQueue. Slots allocate once during initialization;
    // the HAL callback only copies bytes into an existing slot.
    private var availableSlots: [CoreAudioTapInputPacket]
    private var accepting = false
    private var overflowPending = false

    init(
        format: AudioStreamBasicDescription,
        callbacks: SystemAudioCallbacks,
        bufferingPolicy: CoreAudioTapBufferingPolicy
    ) throws {
        let queue = DispatchQueue(
            label: "app.fishmemory.perception.audio.tap-processing",
            qos: .userInitiated
        )
        processingQueue = queue
        processor = try CoreAudioTapSampleProcessor(
            format: format,
            callbacks: callbacks
        )
        availableSlots = (0..<bufferingPolicy.slotCount).map { _ in
            CoreAudioTapInputPacket(capacity: bufferingPolicy.slotCapacity)
        }
    }

    func activate() {
        ioQueue.sync { accepting = true }
    }

    func capture(inputData: UnsafePointer<AudioBufferList>) {
        // This is the HAL callback. Never wait on another queue, write JSON, or
        // invoke the stdout callback here. It only copies a bounded audio block
        // and hands it to the serial converter queue.
        // Keep using the last installed exclusion list while a catalog refresh
        // runs. Unrelated audio clients can appear at any time; pausing every
        // callback for them used to erase whole semantic windows.
        guard accepting else { return }
        guard let packet = availableSlots.popLast() else {
            scheduleOverflowGap()
            return
        }
        guard packet.copy(inputData: inputData) else {
            availableSlots.append(packet)
            return
        }
        processingQueue.async { [weak self] in
            guard let self else { return }
            let shouldConvert = autoreleasepool {
                self.processor.ingest(packet)
            }
            self.ioQueue.async { [weak self] in
                packet.reset()
                self?.availableSlots.append(packet)
            }
            if shouldConvert {
                autoreleasepool { self.processor.convertPendingBatch() }
            }
        }
    }

    func markGap(reason: String) {
        processingQueue.async { [processor] in processor.markGap(reason: reason) }
    }

    func invalidate() {
        processor.invalidateDelivery()
        ioQueue.sync { accepting = false }
        processingQueue.async { [processor] in processor.deactivate() }
    }

    private func scheduleOverflowGap() {
        guard !overflowPending else { return }
        overflowPending = true
        processingQueue.async { [weak self] in
            guard let self else { return }
            self.processor.markGap(reason: "capture_overflow")
            self.ioQueue.async { [weak self] in self?.overflowPending = false }
        }
    }
}

private final class CoreAudioTapInputPacket: @unchecked Sendable {
    struct Buffer: Sendable {
        let offset: Int
        let byteCount: Int
        let channels: Int
    }

    private let capacity: Int
    private var storage: Data
    private(set) var buffers: [Buffer] = []

    init(capacity: Int) {
        self.capacity = capacity
        storage = Data(count: capacity)
        buffers.reserveCapacity(32)
    }

    func copy(inputData: UnsafePointer<AudioBufferList>) -> Bool {
        let list = UnsafeMutableAudioBufferListPointer(
            UnsafeMutablePointer(mutating: inputData)
        )
        guard !list.isEmpty, list.count <= 32 else { return false }
        buffers.removeAll(keepingCapacity: true)
        var copied = true
        storage.withUnsafeMutableBytes { destination in
            var offset = 0
            for buffer in list {
                let byteCount = Int(buffer.mDataByteSize)
                guard byteCount > 0, let source = buffer.mData else { continue }
                guard offset + byteCount <= capacity,
                      let base = destination.baseAddress else {
                    copied = false
                    return
                }
                memcpy(base.advanced(by: offset), source, byteCount)
                buffers.append(.init(
                    offset: offset,
                    byteCount: byteCount,
                    channels: Int(buffer.mNumberChannels)
                ))
                offset += byteCount
            }
        }
        guard copied, !buffers.isEmpty else { reset(); return false }
        return true
    }

    func withUnsafeStorage<Result>(
        _ body: (UnsafeRawBufferPointer) throws -> Result
    ) rethrows -> Result {
        try storage.withUnsafeBytes(body)
    }

    func reset() {
        buffers.removeAll(keepingCapacity: true)
    }
}

private final class CoreAudioTapSampleProcessor: @unchecked Sendable {
    private let format: AudioStreamBasicDescription
    private let delivery: SystemAudioCaptureDelivery
    private let converter: CoreAudioTapSampleRateConverter
    private var monoScratch: [Float] = []
    private var conversionBatch: [Float] = []
    private let conversionBatchSize: Int

    init(
        format: AudioStreamBasicDescription,
        callbacks: SystemAudioCallbacks
    ) throws {
        guard coreAudioTapSupports(format) else {
            throw CoreAudioProcessTapError.unsupportedFormat
        }
        self.format = format
        converter = try CoreAudioTapSampleRateConverter(inputRate: format.mSampleRate)
        // Conversion stays internal and batches about one second at a time.
        // This removes the old four-per-second protocol cadence and leaves
        // ample headroom for the module-owned ten-second window deadline.
        conversionBatchSize = max(1, Int(format.mSampleRate))
        monoScratch.reserveCapacity(min(Int(format.mSampleRate), 65_536))
        conversionBatch.reserveCapacity(conversionBatchSize * 2)
        delivery = SystemAudioCaptureDelivery(callbacks: callbacks)
    }

    func ingest(_ packet: CoreAudioTapInputPacket) -> Bool {
        guard coreAudioTapReadMonoSamples(
            packet: packet,
            format: format,
            into: &monoScratch
        ), !monoScratch.isEmpty else {
            markGap(reason: "invalid_audio_format")
            return false
        }
        conversionBatch.append(contentsOf: monoScratch)
        return conversionBatch.count >= conversionBatchSize
    }

    func convertPendingBatch() {
        guard conversionBatch.count >= conversionBatchSize else { return }
        guard let converted = converter.convert(conversionBatch) else {
            markGap(reason: "audio_conversion_failed")
            return
        }
        conversionBatch.removeAll(keepingCapacity: true)
        guard converted.sampleCount > 0 else { return }
        delivery.submit(.init(pcm: converted.pcm))
    }

    func markGap(reason: String) {
        delivery.submit(.init(reason: reason))
        converter.reset()
        conversionBatch.removeAll(keepingCapacity: true)
    }

    func invalidateDelivery() { delivery.invalidate() }

    func deactivate() {
        converter.reset()
        monoScratch.removeAll(keepingCapacity: true)
        conversionBatch.removeAll(keepingCapacity: true)
    }
}

final class CoreAudioTapSampleRateConverter {
    struct Output {
        let pcm: Data
        let sampleCount: Int
    }

    private let inputFormat: AVAudioFormat
    private let outputFormat: AVAudioFormat
    private let converter: AVAudioConverter
    private let inputBuffer: AVAudioPCMBuffer
    private let outputBuffer: AVAudioPCMBuffer
    private let maximumInputFrames = 32_768

    var configuredQuality: Int { converter.sampleRateConverterQuality }

    init(inputRate: Double) throws {
        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: inputRate,
            channels: 1,
            interleaved: false
        ), let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(SystemAudioWindow.sampleRate),
            channels: 1,
            interleaved: false
        ), let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw CoreAudioProcessTapError.unsupportedFormat
        }
        let outputCapacity = Int(
            ceil(Double(maximumInputFrames) * outputFormat.sampleRate / inputFormat.sampleRate)
        ) + 256
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: AVAudioFrameCount(maximumInputFrames)
        ), let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(outputCapacity)
        ) else { throw CoreAudioProcessTapError.unsupportedFormat }
        self.inputFormat = inputFormat
        self.outputFormat = outputFormat
        self.converter = converter
        self.inputBuffer = inputBuffer
        self.outputBuffer = outputBuffer
        // `high` retains Core Audio's band-limited SRC quality for speech while
        // avoiding the disproportionate CPU cost of the offline-oriented max
        // setting on a continuously running desktop capture.
        converter.sampleRateConverterQuality = AVAudioQuality.high.rawValue
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Normal
    }

    func convert(_ samples: [Float]) -> Output? {
        guard !samples.isEmpty,
              let inputChannels = inputBuffer.floatChannelData else { return nil }
        let estimatedBytes = Int(
            ceil(Double(samples.count) * outputFormat.sampleRate / inputFormat.sampleRate)
        ) * MemoryLayout<Int16>.size
        var pcm = Data()
        pcm.reserveCapacity(estimatedBytes + 512)
        var offset = 0
        while offset < samples.count {
            let count = min(maximumInputFrames, samples.count - offset)
            samples.withUnsafeBufferPointer { source in
                inputChannels[0].update(from: source.baseAddress!.advanced(by: offset), count: count)
            }
            inputBuffer.frameLength = AVAudioFrameCount(count)
            outputBuffer.frameLength = 0
            var supplied = false
            var conversionError: NSError?
            let status = converter.convert(
                to: outputBuffer,
                error: &conversionError
            ) { [inputBuffer] _, inputStatus in
                guard !supplied else {
                    inputStatus.pointee = .noDataNow
                    return nil
                }
                supplied = true
                inputStatus.pointee = .haveData
                return inputBuffer
            }
            guard conversionError == nil, status != .error else { return nil }
            let countOut = Int(outputBuffer.frameLength)
            if countOut > 0, let channels = outputBuffer.int16ChannelData {
                pcm.append(contentsOf:
                    UnsafeRawBufferPointer(
                        start: channels[0],
                        count: countOut * MemoryLayout<Int16>.size
                    )
                )
            }
            offset += count
        }
        return Output(pcm: pcm, sampleCount: pcm.count / MemoryLayout<Int16>.size)
    }

    func reset() { converter.reset() }
}

private func coreAudioTapSupports(_ format: AudioStreamBasicDescription) -> Bool {
    guard format.mFormatID == kAudioFormatLinearPCM,
          format.mSampleRate >= 8_000,
          format.mSampleRate <= 384_000,
          format.mChannelsPerFrame == 1,
          format.mFramesPerPacket == 1,
          format.mFormatFlags & kAudioFormatFlagIsBigEndian == 0,
          format.mFormatFlags & kAudioFormatFlagIsPacked != 0 else { return false }
    let isFloat = format.mFormatFlags & kAudioFormatFlagIsFloat != 0
        && format.mBitsPerChannel == 32
    let isInt16 = format.mFormatFlags & kAudioFormatFlagIsSignedInteger != 0
        && format.mBitsPerChannel == 16
    let bytesPerSample = format.mBitsPerChannel / 8
    return (isFloat || isInt16)
        && format.mBytesPerFrame == bytesPerSample
        && format.mBytesPerPacket == bytesPerSample
}

private func coreAudioTapReadMonoSamples(
    packet: CoreAudioTapInputPacket,
    format: AudioStreamBasicDescription,
    into result: inout [Float]
) -> Bool {
    let bytesPerSample = Int(format.mBitsPerChannel / 8)
    guard bytesPerSample == 2 || bytesPerSample == 4 else { return false }
    let nonInterleaved = format.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
    let declaredChannels = Int(format.mChannelsPerFrame)
    return packet.withUnsafeStorage { storage in
        if nonInterleaved {
            let active = packet.buffers.filter { $0.channels > 0 }
            guard !active.isEmpty else { return false }
            guard active.allSatisfy({
                $0.byteCount.isMultiple(of: bytesPerSample * $0.channels)
            }) else { return false }
            let frameCount = active.map {
                $0.byteCount / (bytesPerSample * $0.channels)
            }.min() ?? 0
            guard frameCount > 0, frameCount <= Int(format.mSampleRate) else { return false }
            resetCoreAudioTapScratch(&result, count: frameCount)
            var channelCount = 0
            for buffer in active {
                for channel in 0..<buffer.channels {
                    channelCount += 1
                    addCoreAudioTapChannel(
                        storage,
                        buffer: buffer,
                        channel: channel,
                        stride: buffer.channels,
                        frameCount: frameCount,
                        format: format,
                        to: &result
                    )
                }
            }
            guard channelCount > 0 else { return false }
            let scale = 1 / Float(channelCount)
            for index in result.indices { result[index] *= scale }
            return true
        }

        guard let first = packet.buffers.first else { return false }
        let channels = first.channels > 0 ? first.channels : declaredChannels
        guard channels > 0 else { return false }
        guard first.byteCount.isMultiple(of: bytesPerSample * channels) else { return false }
        let frameCount = first.byteCount / (bytesPerSample * channels)
        guard frameCount > 0, frameCount <= Int(format.mSampleRate) else { return false }
        resetCoreAudioTapScratch(&result, count: frameCount)
        for channel in 0..<channels {
            addCoreAudioTapChannel(
                storage,
                buffer: first,
                channel: channel,
                stride: channels,
                frameCount: frameCount,
                format: format,
                to: &result
            )
        }
        let scale = 1 / Float(channels)
        for index in result.indices { result[index] *= scale }
        return true
    }
}

private func resetCoreAudioTapScratch(_ values: inout [Float], count: Int) {
    values.removeAll(keepingCapacity: true)
    values.append(contentsOf: repeatElement(0, count: count))
}

private func addCoreAudioTapChannel(
    _ storage: UnsafeRawBufferPointer,
    buffer: CoreAudioTapInputPacket.Buffer,
    channel: Int,
    stride: Int,
    frameCount: Int,
    format: AudioStreamBasicDescription,
    to output: inout [Float]
) {
    guard let baseAddress = storage.baseAddress else { return }
    let start = baseAddress.advanced(by: buffer.offset)
    if format.mFormatFlags & kAudioFormatFlagIsFloat != 0 {
        let values = start.assumingMemoryBound(to: Float.self)
        for frame in 0..<frameCount {
            let value = values[frame * stride + channel]
            if value.isFinite { output[frame] += max(-1, min(1, value)) }
        }
    } else {
        let values = start.assumingMemoryBound(to: Int16.self)
        for frame in 0..<frameCount {
            let value = Int16(littleEndian: values[frame * stride + channel])
            output[frame] += Float(value) / 32_768
        }
    }
}

private enum CoreAudioProcessTapError: Error {
    case operationFailed(step: String, status: OSStatus)
    case invalidObject(String)
    case unsupportedFormat
    case processListUnavailable
    case processListUnstable
    case aggregateDeviceNotReady
    case permissionRequired
    case unsupportedBufferConfiguration
}

private func requireCoreAudioSuccess(_ status: OSStatus, step: String) throws {
    guard status == noErr else {
        throw CoreAudioProcessTapError.operationFailed(step: step, status: status)
    }
}

private func coreAudioPropertyAddress(
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
    element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain
) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: scope,
        mElement: element
    )
}

private func coreAudioTapBufferingPolicy(
    deviceID: AudioObjectID,
    format: AudioStreamBasicDescription
) throws -> CoreAudioTapBufferingPolicy {
    let currentFrames = coreAudioUInt32Property(
        objectID: deviceID,
        selector: kAudioDevicePropertyBufferFrameSize
    ).map(Int.init) ?? 512
    let reportedMaximum = coreAudioValueRangeProperty(
        objectID: deviceID,
        selector: kAudioDevicePropertyBufferFrameSizeRange
    )?.mMaximum
    let maximumFrames = max(
        currentFrames,
        reportedMaximum.flatMap {
            $0.isFinite && $0 > 0 ? Int($0.rounded(.up)) : nil
        } ?? 4_096
    )
    guard let policy = CoreAudioTapBufferingPolicy.make(
        sampleRate: format.mSampleRate,
        bytesPerFrame: Int(format.mBytesPerFrame),
        currentBufferFrames: currentFrames,
        maximumBufferFrames: maximumFrames
    ) else { throw CoreAudioProcessTapError.unsupportedBufferConfiguration }
    return policy
}

private func coreAudioUInt32Property(
    objectID: AudioObjectID,
    selector: AudioObjectPropertySelector
) -> UInt32? {
    var address = coreAudioPropertyAddress(selector: selector)
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    let status = AudioObjectGetPropertyData(
        objectID, &address, 0, nil, &size, &value
    )
    guard status == noErr, size == MemoryLayout<UInt32>.size else { return nil }
    return value
}

private func coreAudioValueRangeProperty(
    objectID: AudioObjectID,
    selector: AudioObjectPropertySelector
) -> AudioValueRange? {
    var address = coreAudioPropertyAddress(selector: selector)
    var value = AudioValueRange()
    var size = UInt32(MemoryLayout<AudioValueRange>.size)
    let status = AudioObjectGetPropertyData(
        objectID, &address, 0, nil, &size, &value
    )
    guard status == noErr, size == MemoryLayout<AudioValueRange>.size else { return nil }
    return value
}

private func coreAudioTapFormat(tapID: AudioObjectID) throws -> AudioStreamBasicDescription {
    var address = coreAudioPropertyAddress(
        selector: kAudioTapPropertyFormat
    )
    var format = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    try requireCoreAudioSuccess(
        AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, &format),
        step: "read_stream_format"
    )
    guard size == MemoryLayout<AudioStreamBasicDescription>.size else {
        throw CoreAudioProcessTapError.unsupportedFormat
    }
    return format
}

private func loadSystemAudioProcessIdentities() -> [SystemAudioProcessIdentity]? {
    guard let processObjectIDs = coreAudioObjectIDList(
        objectID: AudioObjectID(kAudioObjectSystemObject),
        selector: kAudioHardwarePropertyProcessObjectList
    ) else { return nil }
    return processObjectIDs.compactMap { objectID -> SystemAudioProcessIdentity? in
        guard let pid = coreAudioPIDProperty(
            objectID: objectID,
            selector: kAudioProcessPropertyPID
        ), pid > 0 else { return nil }
        return SystemAudioProcessIdentity(
            objectID: objectID,
            processIdentifier: pid,
            parentProcessIdentifier: systemAudioDirectParentProcessID(pid),
            bundleIdentifier: coreAudioStringProperty(
                objectID: objectID,
                selector: kAudioProcessPropertyBundleID
            ),
            executablePath: systemAudioProcessPath(pid),
            ancestors: systemAudioAncestorProcessIDs(pid)
        )
    }
}

private func coreAudioObjectIDList(
    objectID: AudioObjectID,
    selector: AudioObjectPropertySelector
) -> [AudioObjectID]? {
    for _ in 0..<3 {
        var address = coreAudioPropertyAddress(selector: selector)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            objectID, &address, 0, nil, &size
        ) == noErr,
              size.isMultiple(of: UInt32(MemoryLayout<AudioObjectID>.size)) else { continue }
        if size == 0 { return [] }
        var values = [AudioObjectID](
            repeating: AudioObjectID(kAudioObjectUnknown),
            count: Int(size) / MemoryLayout<AudioObjectID>.size
        )
        let status = values.withUnsafeMutableBytes { bytes in
            AudioObjectGetPropertyData(
                objectID, &address, 0, nil, &size, bytes.baseAddress!
            )
        }
        guard status == noErr,
              size.isMultiple(of: UInt32(MemoryLayout<AudioObjectID>.size)) else { continue }
        let count = min(values.count, Int(size) / MemoryLayout<AudioObjectID>.size)
        return values.prefix(count).filter { $0 != kAudioObjectUnknown }
    }
    return nil
}

private func coreAudioPIDProperty(
    objectID: AudioObjectID,
    selector: AudioObjectPropertySelector
) -> pid_t? {
    var address = coreAudioPropertyAddress(selector: selector)
    var value = pid_t(0)
    var size = UInt32(MemoryLayout<pid_t>.size)
    let status = AudioObjectGetPropertyData(
        objectID, &address, 0, nil, &size, &value
    )
    guard status == noErr, size == MemoryLayout<pid_t>.size else { return nil }
    return value
}

private func coreAudioDeviceIsAlive(_ deviceID: AudioObjectID) -> Bool? {
    var address = coreAudioPropertyAddress(
        selector: kAudioDevicePropertyDeviceIsAlive
    )
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    let status = AudioObjectGetPropertyData(
        deviceID, &address, 0, nil, &size, &value
    )
    guard status == noErr, size == MemoryLayout<UInt32>.size else { return nil }
    return value != 0
}

private func coreAudioStringProperty(
    objectID: AudioObjectID,
    selector: AudioObjectPropertySelector
) -> String? {
    var address = coreAudioPropertyAddress(selector: selector)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<CFString?>.size)
    let status = AudioObjectGetPropertyData(
        objectID, &address, 0, nil, &size, &value
    )
    guard status == noErr, let value else { return nil }
    return value.takeRetainedValue() as String
}

private func systemAudioDirectParentProcessID(_ pid: pid_t) -> pid_t? {
    var info = proc_bsdinfo()
    let size = Int32(MemoryLayout<proc_bsdinfo>.size)
    guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else { return nil }
    let parent = pid_t(info.pbi_ppid)
    return parent > 0 ? parent : nil
}
