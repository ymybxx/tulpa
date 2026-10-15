const { build } = require("../package.json");

module.exports = {
  ...build,
  directories: { output: "release-win" },
  extraResources: [
    { from: "runtime/win32-x64/fish-memory-runtime", to: "runtime/fish-memory-runtime" },
    { from: "runtime/win32-x64/fish-plugin-tools", to: "fish-plugin-tools" },
    { from: "runtime/win32-x64/pet-application-activation.node", to: "native/pet-application-activation.node" },
    { from: "runtime/win32-x64/fish-memory-perception-helper.exe", to: "perception/fish-memory-perception-helper.exe" },
    { from: "runtime/win32-x64/models", to: "models" },
    { from: "runtime/win32-x64/licenses", to: "licenses/windows-native" },
    { from: "THIRD_PARTY_NOTICES.md", to: "licenses/THIRD_PARTY_NOTICES.md" },
    { from: "native/windows/NOTICES.md", to: "licenses/Windows-native-NOTICES.md" },
    { from: "native/windows/audio-LICENSE.txt", to: "licenses/Windows-ApplicationLoopback-MIT.txt" },
    { from: "vendor/cubism-web-framework/LICENSE.md", to: "licenses/Live2D-Cubism-Open-Software-License.md" },
  ],
  win: {
    icon: "packaging/brand/tulpa-app-icon.png",
    target: [{ target: "nsis", arch: ["x64"] }],
    executableName: "tulpa",
    requestedExecutionLevel: "asInvoker",
    artifactName: "tulpa-${version}-windows-${arch}-setup.${ext}",
  },
  nsis: {
    oneClick: false,
    perMachine: false,
    allowElevation: false,
    allowToChangeInstallationDirectory: true,
    deleteAppDataOnUninstall: false,
    include: "packaging/windows-installer.nsh",
    installerLanguages: ["en_US", "zh_CN"],
  },
};
