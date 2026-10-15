// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "FishMemoryLiveKitAudio",
    platforms: [
        .macOS("12.3"),
    ],
    products: [
        .executable(
            name: "fish-memory-livekit-audio-helper",
            targets: ["FishMemoryLiveKitAudioHelper"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/livekit/client-sdk-swift.git",
            exact: "2.16.0"
        ),
    ],
    targets: [
        .target(
            name: "FishMemoryAudioCore",
            swiftSettings: [.unsafeFlags(["-warnings-as-errors"])]
        ),
        .executableTarget(
            name: "FishMemoryLiveKitAudioHelper",
            dependencies: [
                "FishMemoryAudioCore",
                .product(name: "LiveKit", package: "client-sdk-swift"),
            ],
            swiftSettings: [.unsafeFlags(["-warnings-as-errors"])]
        ),
        .testTarget(
            name: "FishMemoryAudioCoreTests",
            dependencies: ["FishMemoryAudioCore"],
            swiftSettings: [.unsafeFlags(["-warnings-as-errors"])]
        ),
        .testTarget(
            name: "FishMemoryAudioHelperTests",
            dependencies: ["FishMemoryAudioCore", "FishMemoryLiveKitAudioHelper"],
            swiftSettings: [.unsafeFlags(["-warnings-as-errors"])]
        ),
    ],
    swiftLanguageModes: [.v6]
)
