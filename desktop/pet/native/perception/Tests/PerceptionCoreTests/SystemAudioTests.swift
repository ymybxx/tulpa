import CoreMedia
import CoreAudio
import AVFAudio
import XCTest
@testable import PerceptionCore

final class SystemAudioTests: XCTestCase {
    func testTapBufferingPolicyCoversTwoSecondsWithinFourMegabytes() throws {
        let policy = try XCTUnwrap(CoreAudioTapBufferingPolicy.make(
            sampleRate: 48_000,
            bytesPerFrame: 4,
            currentBufferFrames: 512,
            maximumBufferFrames: 4_096
        ))
        XCTAssertEqual(policy.slotCapacity, 16_384)
        XCTAssertEqual(policy.slotCount, 188)
        XCTAssertLessThanOrEqual(policy.totalCapacity, CoreAudioTapBufferingPolicy.totalMemoryLimit)
        XCTAssertGreaterThanOrEqual(Double(policy.slotCount * 512) / 48_000, 2)
        XCTAssertNil(CoreAudioTapBufferingPolicy.make(
            sampleRate: 48_000,
            bytesPerFrame: 4,
            currentBufferFrames: 512,
            maximumBufferFrames: 2_000_000
        ))
    }

    func testTapSampleRateConverterProducesContinuousPCM() throws {
        let converter = try CoreAudioTapSampleRateConverter(inputRate: 48_000)
        XCTAssertEqual(converter.configuredQuality, AVAudioQuality.high.rawValue)
        let input = [Float](repeating: 0.25, count: 12_000)
        var emittedSamples = 0
        for _ in 0..<20 {
            let converted = try XCTUnwrap(converter.convert(input))
            XCTAssertEqual(converted.pcm.count, converted.sampleCount * 2)
            emittedSamples += converted.sampleCount
        }
        XCTAssertGreaterThan(emittedSamples, 78_000)
        XCTAssertLessThan(emittedSamples, 82_000)
    }

    func testBackendSelectionKeepsOlderMacOSVersionsOnCompatiblePaths() {
        XCTAssertEqual(
            selectSystemAudioBackend(for: .init(majorVersion: 12, minorVersion: 3, patchVersion: 0)),
            .unsupported
        )
        XCTAssertEqual(
            selectSystemAudioBackend(for: .init(majorVersion: 12, minorVersion: 6, patchVersion: 9)),
            .unsupported
        )
        XCTAssertEqual(
            selectSystemAudioBackend(for: .init(majorVersion: 13, minorVersion: 0, patchVersion: 0)),
            .screenCaptureKit
        )
        XCTAssertEqual(
            selectSystemAudioBackend(for: .init(majorVersion: 13, minorVersion: 6, patchVersion: 9)),
            .screenCaptureKit
        )
        XCTAssertEqual(
            selectSystemAudioBackend(for: .init(majorVersion: 14, minorVersion: 1, patchVersion: 9)),
            .screenCaptureKit
        )
    }

    func testBackendSelectionUsesProcessTapStartingAtMacOS14Point2() {
        XCTAssertEqual(
            selectSystemAudioBackend(for: .init(majorVersion: 14, minorVersion: 2, patchVersion: 0)),
            .coreAudioProcessTap
        )
        XCTAssertEqual(
            selectSystemAudioBackend(for: .init(majorVersion: 15, minorVersion: 0, patchVersion: 0)),
            .coreAudioProcessTap
        )
        XCTAssertEqual(
            selectSystemAudioBackend(for: .init(majorVersion: 26, minorVersion: 0, patchVersion: 0)),
            .coreAudioProcessTap
        )
    }

    @MainActor
    func testFactoryConstructsOnlyTheSelectedRuntimeBackend() {
        let callbacks = SystemAudioCallbacks(
            onSamples: { _ in },
            onCaptureGap: { _ in },
            onStatus: { _ in }
        )
        XCTAssertNil(makeSystemAudioMonitor(
            callbacks: callbacks,
            operatingSystemVersion: .init(majorVersion: 12, minorVersion: 6, patchVersion: 0)
        ))

        if #available(macOS 13.0, *) {
            let legacy = makeSystemAudioMonitor(
                callbacks: callbacks,
                operatingSystemVersion: .init(majorVersion: 13, minorVersion: 6, patchVersion: 0)
            )
            XCTAssertTrue(legacy is SystemAudioMonitor)
        }
        if #available(macOS 14.2, *) {
            let modern = makeSystemAudioMonitor(
                callbacks: callbacks,
                operatingSystemVersion: .init(majorVersion: 14, minorVersion: 2, patchVersion: 0)
            )
            // The tap, falling back to ScreenCaptureKit when only Screen Recording is allowed.
            XCTAssertTrue(modern is TapWithScreenCaptureFallback)
        }
    }

    @MainActor
    func testRefusedTapFallsBackToScreenCaptureWhenScreenRecordingIsAllowed() async {
        guard #available(macOS 13.0, *) else { return }
        let tap = FakeAudioMonitor(result: .init(
            supported: true, requested: true, state: .unavailable, permission: .notDetermined,
            reason: TapWithScreenCaptureFallback.tapPermissionRequired))
        let screenCapture = FakeAudioMonitor(result: .init(
            supported: true, requested: true, state: .running, permission: .granted))
        let monitor = TapWithScreenCaptureFallback(
            primary: tap, makeFallback: { screenCapture }, screenRecordingAllowed: { true })

        let report = await monitor.start()
        XCTAssertEqual(report.state, .running)
        XCTAssertEqual(tap.stops, 1)
        XCTAssertEqual(screenCapture.starts, 1)
        _ = await monitor.stop()
        XCTAssertEqual(screenCapture.stops, 1)
    }

    @MainActor
    func testRefusedTapStaysRefusedWithoutScreenRecording() async {
        guard #available(macOS 13.0, *) else { return }
        let tap = FakeAudioMonitor(result: .init(
            supported: true, requested: true, state: .unavailable, permission: .notDetermined,
            reason: TapWithScreenCaptureFallback.tapPermissionRequired))
        let screenCapture = FakeAudioMonitor(result: .init(
            supported: true, requested: true, state: .running, permission: .granted))
        let monitor = TapWithScreenCaptureFallback(
            primary: tap, makeFallback: { screenCapture }, screenRecordingAllowed: { false })

        let report = await monitor.start()
        XCTAssertEqual(report.reason, TapWithScreenCaptureFallback.tapPermissionRequired)
        XCTAssertEqual(screenCapture.starts, 0)
    }

    func testProcessListRevisionChangesMonotonically() {
        let revision = SystemAudioProcessListRevision()
        XCTAssertEqual(revision.value, 0)
        XCTAssertEqual(revision.recordChange(), 1)
        XCTAssertEqual(revision.recordChange(), 2)
        XCTAssertEqual(revision.value, 2)
    }

    func testCoreAudioCleanupRetainsFailedHandleAndDependentResources() {
        var calls: [String] = []
        var state = CoreAudioTapCleanupState(
            ioStarted: true,
            ownsIOProc: true,
            ownsAggregateDevice: true,
            ownsTap: true
        )
        let failure = state.cleanup(using: .init(
            stopIO: { calls.append("stop"); return -1 },
            destroyIOProc: { calls.append("io"); return noErr },
            destroyAggregateDevice: { calls.append("aggregate"); return noErr },
            destroyTap: { calls.append("tap"); return noErr }
        ))

        XCTAssertEqual(failure, .init(step: "stop_io", status: -1))
        XCTAssertEqual(calls, ["stop"])
        XCTAssertEqual(state, .init(
            ioStarted: true,
            ownsIOProc: true,
            ownsAggregateDevice: true,
            ownsTap: true
        ))
    }

    func testCoreAudioCleanupRetriesInOrderAndBecomesIdempotent() {
        var calls: [String] = []
        var failAggregate = true
        var state = CoreAudioTapCleanupState(
            ioStarted: true,
            ownsIOProc: true,
            ownsAggregateDevice: true,
            ownsTap: true
        )
        let operations = CoreAudioTapCleanupOperations(
            stopIO: { calls.append("stop"); return noErr },
            destroyIOProc: { calls.append("io"); return noErr },
            destroyAggregateDevice: {
                calls.append("aggregate")
                if failAggregate { failAggregate = false; return -2 }
                return noErr
            },
            destroyTap: { calls.append("tap"); return noErr }
        )

        XCTAssertEqual(
            state.cleanup(using: operations),
            .init(step: "destroy_aggregate_device", status: -2)
        )
        XCTAssertEqual(calls, ["stop", "io", "aggregate"])
        XCTAssertEqual(state, .init(
            ioStarted: false,
            ownsIOProc: false,
            ownsAggregateDevice: true,
            ownsTap: true
        ))

        XCTAssertNil(state.cleanup(using: operations))
        XCTAssertEqual(calls, ["stop", "io", "aggregate", "aggregate", "tap"])
        XCTAssertTrue(state.isComplete)

        XCTAssertNil(state.cleanup(using: operations))
        XCTAssertEqual(calls, ["stop", "io", "aggregate", "aggregate", "tap"])
    }

    func testCoreAudioCleanupContinuesWhenDeviceAlreadyStopped() {
        var calls: [String] = []
        var state = CoreAudioTapCleanupState(
            ioStarted: true,
            ownsIOProc: true,
            ownsAggregateDevice: true,
            ownsTap: true
        )
        XCTAssertNil(state.cleanup(using: .init(
            stopIO: { calls.append("stop"); return kAudioHardwareNotRunningError },
            destroyIOProc: { calls.append("io"); return noErr },
            destroyAggregateDevice: { calls.append("aggregate"); return noErr },
            destroyTap: { calls.append("tap"); return noErr }
        )))
        XCTAssertEqual(calls, ["stop", "io", "aggregate", "tap"])
        XCTAssertTrue(state.isComplete)
    }

    @MainActor
    func testConcurrentRecoveryAndStopJoinOneMonitorCleanupPass() async {
        let singleFlight = SystemAudioCleanupSingleFlight()
        var calls = 0
        var release: CheckedContinuation<Bool, Never>?
        let recovery = Task { @MainActor in
            await singleFlight.run {
                calls += 1
                return await withCheckedContinuation { release = $0 }
            }
        }
        for _ in 0..<100 where release == nil { await Task.yield() }
        XCTAssertEqual(calls, 1)

        let externalStop = Task { @MainActor in
            await singleFlight.run {
                calls += 100
                return false
            }
        }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertEqual(calls, 1)

        release?.resume(returning: true)
        let recoveryResult = await recovery.value
        let stopResult = await externalStop.value
        XCTAssertTrue(recoveryResult)
        XCTAssertTrue(stopResult)
        XCTAssertEqual(calls, 1)

        let nextResult = await singleFlight.run {
            calls += 1
            return true
        }
        XCTAssertTrue(nextResult)
        XCTAssertEqual(calls, 2)
    }

    func testProcessTapExcludesOnlyFishAudioObjectsAndHostProcessTree() {
        let hostPID: pid_t = 500
        let identities: [SystemAudioProcessIdentity] = [
            .init(
                objectID: 101,
                processIdentifier: 600,
                parentProcessIdentifier: 1,
                bundleIdentifier: nil,
                executablePath: "/helpers/fish-memory-livekit-audio-helper",
                ancestors: []
            ),
            .init(
                objectID: 102,
                processIdentifier: 601,
                parentProcessIdentifier: 1,
                bundleIdentifier: "app.fishmemory.audio-helper",
                executablePath: nil,
                ancestors: []
            ),
            .init(
                objectID: 103,
                processIdentifier: hostPID,
                parentProcessIdentifier: 1,
                bundleIdentifier: nil,
                executablePath: "/helpers/fish-memory-perception-helper",
                ancestors: []
            ),
            .init(
                objectID: 104,
                processIdentifier: 602,
                parentProcessIdentifier: 501,
                bundleIdentifier: nil,
                executablePath: "/venv/bin/python",
                ancestors: [501, hostPID]
            ),
            .init(
                objectID: 201,
                processIdentifier: 700,
                parentProcessIdentifier: 1,
                bundleIdentifier: "com.google.Chrome",
                executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                ancestors: []
            ),
            .init(
                objectID: 202,
                processIdentifier: 701,
                parentProcessIdentifier: 1,
                bundleIdentifier: "org.videolan.vlc",
                executablePath: "/Applications/VLC.app/Contents/MacOS/VLC",
                ancestors: []
            ),
        ]

        XCTAssertEqual(
            Set(excludedSystemAudioProcessObjectIDs(from: identities, hostProcessIdentifier: hostPID)),
            Set([101, 102, 103, 104])
        )
    }

    func testAudioOnlyStreamMonitorsOneDisplayPointAndNoCursor() {
        let configuration = makeSystemAudioStreamConfiguration()
        XCTAssertEqual(configuration.width, 2)
        XCTAssertEqual(configuration.height, 2)
        XCTAssertEqual(
            configuration.sourceRect,
            CGRect(x: 0, y: 0, width: 1, height: 1)
        )
        XCTAssertEqual(
            configuration.minimumFrameInterval,
            CMTime(value: 60, timescale: 1)
        )
        XCTAssertEqual(configuration.queueDepth, 1)
        XCTAssertFalse(configuration.showsCursor)
        XCTAssertTrue(configuration.capturesAudio)
        XCTAssertTrue(configuration.excludesCurrentProcessAudio)
        XCTAssertEqual(configuration.sampleRate, 16_000)
        XCTAssertEqual(configuration.channelCount, 1)
    }

    func testWindowAssemblerProducesExactCompleteAndPartialPayloads() throws {
        var complete = SystemAudioWindowAssembler(
            captureSessionId: "session",
            startedAt: 100,
            startMonotonicMs: 1_000,
            initialUnavailableReason: nil
        )
        for _ in 0..<10 {
            XCTAssertTrue(complete.append(.init(
                pcm: Data(repeating: 7, count: 32_000)
            )))
        }
        let completeWindow = complete.finishNextWindow()
        XCTAssertEqual(completeWindow.captureSessionId, "session")
        XCTAssertEqual(completeWindow.windowIndex, 0)
        XCTAssertEqual(completeWindow.startedAt, 100)
        XCTAssertEqual(completeWindow.endedAt, 110)
        XCTAssertEqual(completeWindow.audioStatus, .complete)
        XCTAssertEqual(completeWindow.audioCapturedSamples, 160_000)
        XCTAssertNil(completeWindow.audioGapReason)
        XCTAssertEqual(Data(base64Encoded: try XCTUnwrap(completeWindow.audioPcmBase64))?.count, 320_000)

        var partial = SystemAudioWindowAssembler(
            captureSessionId: "session",
            startedAt: 100,
            startMonotonicMs: 1_000,
            initialUnavailableReason: nil
        )
        XCTAssertTrue(partial.append(.init(pcm: Data(repeating: 3, count: 40_000))))
        XCTAssertTrue(partial.append(.init(pcm: Data(repeating: 3, count: 40_000))))
        partial.markGap(reason: "capture_overflow")
        let partialWindow = partial.finishNextWindow()
        XCTAssertEqual(partialWindow.audioStatus, .partial)
        XCTAssertEqual(partialWindow.audioCapturedSamples, 40_000)
        XCTAssertEqual(partialWindow.audioGapReason, "capture_overflow")
        XCTAssertEqual(Data(base64Encoded: try XCTUnwrap(partialWindow.audioPcmBase64))?.count, 320_000)
    }

    func testEarlyCutKeepsTenSecondAudioAndTimestampsAligned() throws {
        var assembler = SystemAudioWindowAssembler(
            startedAt: 100, startMonotonicMs: 1_000, initialUnavailableReason: nil
        )
        for value in 1...16 {
            XCTAssertTrue(assembler.append(.init(pcm: Data(repeating: UInt8(value), count: 32_000))))
        }
        let first = assembler.finishNextWindow(sampleCount: 96_000)
        let second = assembler.finishNextWindow()
        XCTAssertEqual(first.startedAt, 100)
        XCTAssertEqual(first.endedAt, 106)
        XCTAssertEqual(first.audioCapturedSamples, 96_000)
        XCTAssertEqual(first.audioStatus, .complete)
        XCTAssertEqual(second.startedAt, first.endedAt)
        XCTAssertEqual(second.endedAt, 116)
        XCTAssertEqual(second.audioCapturedSamples, 160_000)
        XCTAssertEqual(second.audioStatus, .complete)
        let secondPCM = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(second.audioPcmBase64)))
        XCTAssertEqual(secondPCM.count, 320_000)
        XCTAssertEqual(secondPCM.first, 7)
        XCTAssertEqual(secondPCM.last, 16)
    }

    func testWindowAssemblerEmitsUnavailableAndKeepsExactSequentialBounds() {
        var assembler = SystemAudioWindowAssembler(
            captureSessionId: "session",
            startedAt: 100,
            startMonotonicMs: 1_000,
            initialUnavailableReason: "macos_13_required"
        )
        let first = assembler.finishNextWindow()
        let second = assembler.finishNextWindow()
        XCTAssertEqual(first.audioStatus, .unavailable)
        XCTAssertNil(first.audioPcmBase64)
        XCTAssertEqual(first.audioCapturedSamples, 0)
        XCTAssertEqual(first.audioGapReason, "macos_13_required")
        XCTAssertEqual(first.startedAt, 100)
        XCTAssertEqual(first.endedAt, 110)
        XCTAssertEqual(second.windowIndex, 1)
        XCTAssertEqual(second.startedAt, 110)
        XCTAssertEqual(second.endedAt, 120)
    }

    func testWindowAssemblerCarriesSequentialAudioAcrossWindowBoundaries() throws {
        var assembler = SystemAudioWindowAssembler(
            startedAt: 100,
            startMonotonicMs: 1_000,
            periodMilliseconds: 100,
            initialUnavailableReason: nil
        )
        XCTAssertTrue(assembler.append(.init(
            pcm: Data(repeating: 1, count: 3_200)
        )))
        XCTAssertTrue(assembler.append(.init(
            pcm: Data(repeating: 2, count: 1_600)
        )))
        let first = assembler.finishNextWindow()
        let second = assembler.finishNextWindow()
        XCTAssertEqual(first.audioStatus, .complete)
        XCTAssertEqual(first.audioCapturedSamples, 1_600)
        XCTAssertEqual(second.audioStatus, .partial)
        XCTAssertEqual(second.audioCapturedSamples, 800)
        XCTAssertEqual(Data(base64Encoded: try XCTUnwrap(second.audioPcmBase64))?.prefix(1_600),
                       Data(repeating: 2, count: 1_600))
    }

    func testWindowAssemblerBoundsBacklogAtTwoWindows() {
        var assembler = SystemAudioWindowAssembler(
            startedAt: 100,
            startMonotonicMs: 1_000,
            periodMilliseconds: 100,
            initialUnavailableReason: nil
        )
        XCTAssertTrue(assembler.append(.init(pcm: Data(repeating: 1, count: 6_400))))
        XCTAssertFalse(assembler.append(.init(pcm: Data(repeating: 2, count: 2))))

        let first = assembler.finishNextWindow()
        let second = assembler.finishNextWindow()
        XCTAssertEqual(first.audioStatus, .partial)
        XCTAssertEqual(first.audioGapReason, "capture_overflow")
        XCTAssertEqual(first.audioCapturedSamples, 1_600)
        XCTAssertEqual(second.audioStatus, .complete)
        XCTAssertEqual(second.audioCapturedSamples, 1_600)
    }

    func testKnownFishPlaybackProcessesAreExcludedWithoutExcludingOtherPlayers() {
        XCTAssertTrue(shouldExcludeSystemAudioApplication(bundleIdentifier: "app.fishmemory.audio-helper",
            processIdentifier: 5, parentProcessIdentifier: 6, executablePath: nil))
        XCTAssertTrue(shouldExcludeSystemAudioApplication(bundleIdentifier: nil, processIdentifier: 5,
            parentProcessIdentifier: 6, executablePath: "/app/fish-memory-livekit-audio-helper"))
        XCTAssertTrue(shouldExcludeSystemAudioApplication(bundleIdentifier: nil, processIdentifier: 6,
            parentProcessIdentifier: 6, executablePath: nil))
        XCTAssertTrue(shouldExcludeSystemAudioApplication(bundleIdentifier: nil, processIdentifier: 9,
            parentProcessIdentifier: 6, executablePath: "/venv/bin/python", ancestors: [8, 6]))
        XCTAssertFalse(shouldExcludeSystemAudioApplication(bundleIdentifier: "org.videolan.vlc",
            processIdentifier: 8, parentProcessIdentifier: 6, executablePath: "/Applications/VLC.app/Contents/MacOS/VLC"))
    }

    func testFloatPCMConversionAndUnsupportedSampleRate() throws {
        let input: [Float] = [-1, -0.5, 0, 0.5, 1]
        let sample = try makeAudioSample(input, rate: 16_000)
        let bytes = try XCTUnwrap(systemAudioPCM(sample))
        let expected: [Int16] = [-32_768, -16_384, 0, 16_384, 32_767]
        XCTAssertEqual(bytes, expected.withUnsafeBytes { Data($0) })
        XCTAssertNil(systemAudioPCM(try makeAudioSample(input, rate: 48_000)))
        XCTAssertNil(systemAudioPCM(try makeAudioSample([.nan], rate: 16_000)))
    }

    @MainActor
    func testUnsupportedSystemNeverConstructsCaptureAndLegacyProtocolDefaultsOff() throws {
        var calls = 0
        let module = SystemAudioModule(supported: false, factory: { _ in calls += 1; return nil },
                                      onWindow: { _ in }, onStatusChanged: {})
        module.configure(requested: true, screenEnabled: true)
        XCTAssertEqual(calls, 0)
        XCTAssertEqual(module.report.reason, "macos_13_required")
        XCTAssertFalse(module.report.supported)
        let command = try PerceptionProtocolCodec.decodeCommand("""
        {"protocolVersion":1,"command":"configure","modules":{"screen":true,"accessibility":false,"camera":false}}
        """)
        XCTAssertFalse(try XCTUnwrap(command.modules).systemAudio)
    }

    func testChunkBoundaryRoundingContinuesWithoutAnEmptyWindow() async throws {
        let received = expectation(description: "two windows across a chunk boundary")
        let recorder = AudioWindowRecorder()
        let segmenter = SystemAudioWindowSegmenter(
            periodMilliseconds: 50,
            graceMilliseconds: 0,
            wallClock: { 599.97999 },
            handler: { window in
                recorder.append(window)
                if window.windowIndex == 1 { received.fulfill() }
            }
        )
        segmenter.start(initialUnavailableReason: "test")
        await fulfillment(of: [received], timeout: 1)
        segmenter.stop()
        let windows = recorder.events
        XCTAssertGreaterThanOrEqual(windows.count, 2)
        XCTAssertEqual(windows[0].endedAt, 600, accuracy: 0.000_032)
        XCTAssertEqual(windows[1].startedAt, windows[0].endedAt)
        XCTAssertEqual(windows[1].endedAt - windows[1].startedAt, 0.05, accuracy: 0.000_001)
    }

    @MainActor
    func testUnsupportedSystemStillTicksUnavailableWindows() async throws {
        let received = expectation(description: "unavailable native window")
        let recorder = AudioWindowRecorder()
        let module = SystemAudioModule(
            supported: false,
            factory: { _ in XCTFail("unsupported system constructed a backend"); return nil },
            segmenterFactory: { handler in
                SystemAudioWindowSegmenter(
                    periodMilliseconds: 50,
                    graceMilliseconds: 10,
                    handler: handler
                )
            },
            onWindow: {
                recorder.append($0)
                received.fulfill()
            },
            onStatusChanged: {}
        )

        module.configure(requested: true, screenEnabled: true)
        await fulfillment(of: [received], timeout: 1)
        _ = module.stop()
        let window = try XCTUnwrap(recorder.events.first)
        XCTAssertEqual(window.captureSessionId.isEmpty, false)
        XCTAssertEqual(window.windowIndex, 0)
        XCTAssertEqual(window.endedAt - window.startedAt, 0.05, accuracy: 0.000_001)
        XCTAssertEqual(window.audioStatus, .unavailable)
        XCTAssertNil(window.audioPcmBase64)
        XCTAssertEqual(window.audioCapturedSamples, 0)
        XCTAssertEqual(window.audioGapReason, "macos_13_required")
    }

    @MainActor
    func testBackendRecreationKeepsWindowSessionAndSequence() async throws {
        let firstWindow = expectation(description: "first window")
        let secondWindow = expectation(description: "second window")
        let recorder = AudioWindowRecorder()
        let first = ControlledSystemAudioMonitor()
        let second = ControlledSystemAudioMonitor()
        var factoryCalls = 0
        let module = SystemAudioModule(
            supported: true,
            factory: { _ in
                factoryCalls += 1
                return factoryCalls == 1 ? first : second
            },
            segmenterFactory: { handler in
                SystemAudioWindowSegmenter(
                    periodMilliseconds: 50,
                    graceMilliseconds: 10,
                    handler: handler
                )
            },
            onWindow: { window in
                recorder.append(window)
                if window.windowIndex == 0 { firstWindow.fulfill() }
                if window.windowIndex == 1 { secondWindow.fulfill() }
            },
            onStatusChanged: {}
        )

        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where first.continuation == nil { await Task.yield() }
        first.continuation?.resume(returning: .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        ))
        await fulfillment(of: [firstWindow], timeout: 1)

        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where first.stops == 0 { await Task.yield() }
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where second.continuation == nil { await Task.yield() }
        second.continuation?.resume(returning: .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        ))
        await fulfillment(of: [secondWindow], timeout: 1)
        _ = await module.stop()?.value

        let windows = recorder.events
        XCTAssertGreaterThanOrEqual(windows.count, 2)
        XCTAssertEqual(windows[0].captureSessionId, windows[1].captureSessionId)
        XCTAssertEqual(windows[0].windowIndex, 0)
        XCTAssertEqual(windows[1].windowIndex, 1)
    }

    @MainActor
    func testHelloPreservesAdditiveSystemAudioCapabilityWithoutStartingCapture() async throws {
        let pipe = Pipe()
        let writer = PerceptionEventWriter(output: pipe.fileHandleForWriting)
        let runtime = PerceptionRuntime(writer: writer, onShutdown: {})
        runtime.start()
        await writer.flush()
        try pipe.fileHandleForWriting.close()
        let lines = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .split(separator: "\n")
        let hello = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(lines[0].utf8)) as? [String: Any])
        let body = try XCTUnwrap(hello["body"] as? [String: Any])
        let caps = try XCTUnwrap(body["capabilities"] as? [String: Any])
        let audio = try XCTUnwrap(caps["systemAudio"] as? [String: Any])
        XCTAssertEqual(audio["requested"] as? Bool, false)
        XCTAssertEqual(audio["state"] as? String, "disabled")
    }

    @MainActor
    func testAudioOnlyStartAndDisableRevokeLateSamples() async throws {
        let recorder = AudioWindowRecorder()
        let monitor = ControlledSystemAudioMonitor()
        var callbacks: SystemAudioCallbacks?
        let module = SystemAudioModule(supported: true, factory: {
            callbacks = $0
            return monitor
        }, onWindow: { recorder.append($0) }, onStatusChanged: {})
        module.configure(requested: true, screenEnabled: false)
        XCTAssertNotNil(callbacks)
        for _ in 0..<100 where monitor.continuation == nil { await Task.yield() }
        XCTAssertNotNil(monitor.continuation)
        module.configure(requested: false, screenEnabled: false)
        monitor.continuation?.resume(returning: .init(supported: true, requested: true,
                                                       state: .running, permission: .granted))
        for _ in 0..<100 where monitor.stops == 0 { await Task.yield() }
        XCTAssertGreaterThanOrEqual(monitor.stops, 1)
        XCTAssertEqual(module.report.state, .disabled)
        callbacks?.onSamples(.init(pcm: Data(repeating: 0, count: 8_000)))
        XCTAssertTrue(recorder.events.isEmpty)
    }

    @MainActor
    func testScreenOnlyTicksWithoutAudioBackendOnSupportedAndOlderSystems() async throws {
        for supported in [false, true] {
            let received = expectation(description: "screen-only media clock")
            let recorder = AudioWindowRecorder()
            let module = SystemAudioModule(
                supported: supported,
                factory: { _ in XCTFail("screen-only constructed an audio backend"); return nil },
                segmenterFactory: { handler in
                    SystemAudioWindowSegmenter(periodMilliseconds: 50, graceMilliseconds: 0,
                                               handler: handler)
                },
                onWindow: {
                    recorder.append($0)
                    if $0.windowIndex == 0 { received.fulfill() }
                },
                onStatusChanged: {}
            )
            module.configure(requested: false, screenEnabled: true)
            await fulfillment(of: [received], timeout: 1)
            _ = module.stop()
            let window = try XCTUnwrap(recorder.events.first)
            XCTAssertEqual(window.audioStatus, .unavailable)
            XCTAssertEqual(window.audioGapReason, "system_audio_disabled")
            XCTAssertEqual(window.audioCapturedSamples, 0)
            XCTAssertNil(window.audioPcmBase64)
            XCTAssertEqual(window.endedAt - window.startedAt, 0.05, accuracy: 0.000_001)
        }
    }

    @MainActor
    func testScreenToggleDoesNotRestartRunningAudio() async throws {
        let monitor = ControlledSystemAudioMonitor()
        var starts = 0
        let module = SystemAudioModule(supported: true, factory: { _ in
            starts += 1
            return monitor
        }, onWindow: { _ in }, onStatusChanged: {})
        module.configure(requested: true, screenEnabled: false)
        for _ in 0..<100 where monitor.continuation == nil { await Task.yield() }
        monitor.continuation?.resume(returning: .init(supported: true, requested: true,
                                                     state: .running, permission: .granted))
        for _ in 0..<100 where module.report.state != .running { await Task.yield() }
        XCTAssertEqual(module.report.state, .running)
        module.configure(requested: true, screenEnabled: true)
        module.configure(requested: true, screenEnabled: false)
        XCTAssertEqual(starts, 1)
        XCTAssertEqual(monitor.stops, 0)
        if let cleanup = module.stop() { _ = await cleanup.value }
    }

    @MainActor
    func testStopTaskDoesNotFinishBeforeInProgressCaptureStops() async throws {
        let monitor = ControlledSystemAudioMonitor()
        let module = SystemAudioModule(
            supported: true,
            factory: { _ in monitor },
            onWindow: { _ in },
            onStatusChanged: {}
        )
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where monitor.continuation == nil { await Task.yield() }
        monitor.continuation?.resume(returning: .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        ))
        for _ in 0..<100 where module.report.state != .running { await Task.yield() }
        monitor.blockStop = true

        let stop = try XCTUnwrap(module.stop())
        for _ in 0..<100 where monitor.stopContinuation == nil { await Task.yield() }
        XCTAssertFalse(monitor.stopFinished)
        monitor.completeStop()
        _ = await stop.value
        XCTAssertTrue(monitor.stopFinished)
    }

    @MainActor
    func testRepeatedConfigureStartsOnlyOneAudioBackend() async throws {
        let monitor = ControlledSystemAudioMonitor()
        var factoryCalls = 0
        var segmenterCalls = 0
        let module = SystemAudioModule(
            supported: true,
            factory: { _ in
                factoryCalls += 1
                return monitor
            },
            segmenterFactory: { handler in
                segmenterCalls += 1
                return SystemAudioWindowSegmenter(handler: handler)
            },
            onWindow: { _ in },
            onStatusChanged: {}
        )

        module.configure(requested: true, screenEnabled: true)
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where monitor.continuation == nil { await Task.yield() }

        XCTAssertEqual(factoryCalls, 1)
        XCTAssertEqual(segmenterCalls, 1)
        XCTAssertEqual(monitor.starts, 1)

        monitor.continuation?.resume(returning: .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        ))
        for _ in 0..<100 where module.report.state != .running { await Task.yield() }
        _ = await module.stop()?.value
    }

    @MainActor
    func testStopIsIdempotentAfterBackendStarts() async throws {
        let monitor = ControlledSystemAudioMonitor()
        let module = SystemAudioModule(
            supported: true,
            factory: { _ in monitor },
            onWindow: { _ in },
            onStatusChanged: {}
        )
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where monitor.continuation == nil { await Task.yield() }
        monitor.continuation?.resume(returning: .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        ))
        for _ in 0..<100 where module.report.state != .running { await Task.yield() }
        monitor.blockStop = true

        let firstStop = try XCTUnwrap(module.stop())
        let secondStop = try XCTUnwrap(module.stop())
        for _ in 0..<100 where monitor.stopContinuation == nil { await Task.yield() }
        XCTAssertFalse(monitor.stopFinished)
        XCTAssertEqual(monitor.stops, 1)

        monitor.completeStop()
        let firstResult = await firstStop.value
        let secondResult = await secondStop.value
        XCTAssertTrue(firstResult)
        XCTAssertTrue(secondResult)

        XCTAssertEqual(monitor.stops, 1)
        XCTAssertEqual(module.report.state, .disabled)
    }

    @MainActor
    func testShutdownStopJoinsCleanupStartedByDisable() async throws {
        let monitor = ControlledSystemAudioMonitor()
        let module = SystemAudioModule(
            supported: true,
            factory: { _ in monitor },
            onWindow: { _ in },
            onStatusChanged: {}
        )
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where monitor.continuation == nil { await Task.yield() }
        monitor.continuation?.resume(returning: .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        ))
        for _ in 0..<100 where module.report.state != .running { await Task.yield() }
        monitor.blockStop = true

        // configure(false) mirrors the settings update; the following stop is
        // the stopAll call made by helper shutdown.
        module.configure(requested: false, screenEnabled: true)
        let shutdownStop = try XCTUnwrap(module.stop())
        for _ in 0..<100 where monitor.stopContinuation == nil { await Task.yield() }
        var shutdownFinished = false
        let waiter = Task { @MainActor in
            let result = await shutdownStop.value
            shutdownFinished = true
            return result
        }
        for _ in 0..<100 { await Task.yield() }
        XCTAssertFalse(shutdownFinished)

        monitor.completeStop()
        let shutdownResult = await waiter.value
        XCTAssertTrue(shutdownResult)
        XCTAssertTrue(shutdownFinished)
        XCTAssertEqual(monitor.stops, 1)
    }

    @MainActor
    func testQuickDisableThenEnableWaitsForOldBackendToStop() async throws {
        let first = ControlledSystemAudioMonitor()
        let second = ControlledSystemAudioMonitor()
        var factoryCalls = 0
        let module = SystemAudioModule(
            supported: true,
            factory: { _ in
                factoryCalls += 1
                return factoryCalls == 1 ? first : second
            },
            onWindow: { _ in },
            onStatusChanged: {}
        )
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where first.continuation == nil { await Task.yield() }
        first.continuation?.resume(returning: .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        ))
        for _ in 0..<100 where module.report.state != .running { await Task.yield() }
        first.blockStop = true

        module.configure(requested: false, screenEnabled: true)
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where first.stopContinuation == nil { await Task.yield() }

        XCTAssertEqual(factoryCalls, 1)
        XCTAssertEqual(second.starts, 0)
        XCTAssertEqual(module.report.reason, "system_audio_waiting_for_cleanup")

        first.completeStop()
        for _ in 0..<100 where second.continuation == nil { await Task.yield() }
        XCTAssertTrue(first.stopFinished)
        XCTAssertEqual(factoryCalls, 2)
        XCTAssertEqual(second.starts, 1)

        second.continuation?.resume(returning: .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        ))
        for _ in 0..<100 where module.report.state != .running { await Task.yield() }
        XCTAssertEqual(module.report.state, .running)
        _ = await module.stop()?.value
    }

    @MainActor
    func testCleanupFailureBlocksReplacementBackend() async throws {
        let first = ControlledSystemAudioMonitor()
        first.stopSucceeds = false
        let second = ControlledSystemAudioMonitor()
        var factoryCalls = 0
        let module = SystemAudioModule(
            supported: true,
            factory: { _ in
                factoryCalls += 1
                return factoryCalls == 1 ? first : second
            },
            onWindow: { _ in },
            onStatusChanged: {}
        )
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where first.continuation == nil { await Task.yield() }
        first.continuation?.resume(returning: .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        ))
        for _ in 0..<100 where module.report.state != .running { await Task.yield() }

        module.configure(requested: false, screenEnabled: true)
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<100 where module.report.reason != "system_audio_cleanup_failed" {
            await Task.yield()
        }

        XCTAssertEqual(factoryCalls, 1)
        XCTAssertEqual(second.starts, 0)
        XCTAssertEqual(module.report.reason, "system_audio_cleanup_failed")
    }

    @MainActor
    func testBlockingHALStartKeepsMainActorResponsiveAndSerializesStop() async throws {
        let events = StringEventRecorder()
        let startRelease = DispatchSemaphore(value: 0)
        let monitor = BlockingHALSystemAudioMonitor(
            events: events,
            startRelease: startRelease
        )
        let module = SystemAudioModule(
            supported: true,
            factory: { _ in monitor },
            onWindow: { _ in },
            onStatusChanged: {}
        )
        module.configure(requested: true, screenEnabled: true)
        for _ in 0..<1_000 where !events.values.contains("start_entered") {
            await Task.yield()
        }
        XCTAssertTrue(events.values.contains("start_entered"))

        // Reaching this call proves the simulated blocking TCC operation did
        // not occupy MainActor. Stop is accepted now, while its HAL operation
        // remains ordered behind start on the control queue.
        let stop = try XCTUnwrap(module.stop())
        for _ in 0..<1_000 where !monitor.stopRequested { await Task.yield() }
        XCTAssertTrue(monitor.stopRequested)
        XCTAssertFalse(events.values.contains("stop_executed"))

        startRelease.signal()
        let stopped = await stop.value
        XCTAssertTrue(stopped)
        XCTAssertEqual(
            events.values,
            ["start_entered", "start_returned", "stop_executed"]
        )
    }
}

private func makeAudioSample(_ values: [Float], rate: Double) throws -> CMSampleBuffer {
    var format = AudioStreamBasicDescription(mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
        mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
    var description: CMAudioFormatDescription?
    XCTAssertEqual(CMAudioFormatDescriptionCreate(allocator: kCFAllocatorDefault, asbd: &format,
        layoutSize: 0, layout: nil, magicCookieSize: 0, magicCookie: nil,
        extensions: nil, formatDescriptionOut: &description), noErr)
    var block: CMBlockBuffer?
    let data = values.withUnsafeBytes { Data($0) }
    XCTAssertEqual(CMBlockBufferCreateWithMemoryBlock(allocator: kCFAllocatorDefault,
        memoryBlock: nil, blockLength: data.count, blockAllocator: kCFAllocatorDefault,
        customBlockSource: nil, offsetToData: 0, dataLength: data.count,
        flags: 0, blockBufferOut: &block), noErr)
    let validBlock = try XCTUnwrap(block)
    data.withUnsafeBytes {
        XCTAssertEqual(CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: validBlock,
                                                   offsetIntoDestination: 0, dataLength: data.count), noErr)
    }
    var sample: CMSampleBuffer?
    XCTAssertEqual(CMAudioSampleBufferCreateReadyWithPacketDescriptions(allocator: kCFAllocatorDefault,
        dataBuffer: validBlock, formatDescription: try XCTUnwrap(description), sampleCount: values.count,
        presentationTimeStamp: .zero, packetDescriptions: nil, sampleBufferOut: &sample), noErr)
    return try XCTUnwrap(sample)
}

private final class AudioWindowRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [SystemAudioWindow] = []
    var events: [SystemAudioWindow] { lock.lock(); defer { lock.unlock() }; return stored }
    func append(_ event: SystemAudioWindow) { lock.lock(); stored.append(event); lock.unlock() }
}

private final class StringEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [String] = []
    var values: [String] { lock.lock(); defer { lock.unlock() }; return stored }
    func append(_ value: String) { lock.lock(); stored.append(value); lock.unlock() }
}

@MainActor
private final class BlockingHALSystemAudioMonitor: SystemAudioMonitoring {
    private let control = CoreAudioHALControlQueue()
    private let events: StringEventRecorder
    private let startRelease: DispatchSemaphore
    private(set) var stopRequested = false

    init(events: StringEventRecorder, startRelease: DispatchSemaphore) {
        self.events = events
        self.startRelease = startRelease
    }

    func start() async -> CapabilityReport {
        let events = events
        let startRelease = startRelease
        _ = await control.perform {
            events.append("start_entered")
            startRelease.wait()
            events.append("start_returned")
            return noErr
        }
        return .init(
            supported: true,
            requested: true,
            state: .running,
            permission: .granted
        )
    }

    func stop() async -> Bool {
        stopRequested = true
        let events = events
        let status = await control.perform {
            events.append("stop_executed")
            return noErr
        }
        return status == noErr
    }
}

@MainActor
private final class ControlledSystemAudioMonitor: SystemAudioMonitoring {
    var continuation: CheckedContinuation<CapabilityReport, Never>?
    var stopContinuation: CheckedContinuation<Void, Never>?
    var blockStop = false
    var stopSucceeds = true
    var stopFinished = false
    var starts = 0
    var stops = 0
    func start() async -> CapabilityReport {
        starts += 1
        return await withCheckedContinuation { continuation = $0 }
    }
    func stop() async -> Bool {
        stops += 1
        if blockStop {
            await withCheckedContinuation { stopContinuation = $0 }
        }
        stopFinished = true
        return stopSucceeds
    }

    func completeStop() {
        blockStop = false
        stopContinuation?.resume()
        stopContinuation = nil
    }
}

@MainActor
private final class FakeAudioMonitor: SystemAudioMonitoring {
    let result: CapabilityReport
    var starts = 0
    var stops = 0
    init(result: CapabilityReport) { self.result = result }
    func start() async -> CapabilityReport { starts += 1; return result }
    func stop() async -> Bool { stops += 1; return true }
}
