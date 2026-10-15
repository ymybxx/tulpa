// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "FishMemoryPerception",
    platforms: [
        .macOS("12.3"),
    ],
    products: [
        .executable(
            name: "fish-memory-perception-helper",
            targets: ["FishMemoryPerceptionHelper"]
        ),
    ],
    targets: [
        .target(name: "PerceptionCore"),
        .executableTarget(
            name: "FishMemoryPerceptionHelper",
            dependencies: ["PerceptionCore"]
        ),
        .testTarget(
            name: "PerceptionCoreTests",
            dependencies: ["PerceptionCore"]
        ),
    ]
)
