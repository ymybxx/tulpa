# Windows native components

- `delay_load_hook.cpp` adapts node-gyp's Windows delay-load hook, MIT license.
  Source: https://github.com/nodejs/node-gyp/blob/main/src/win_delay_load_hook.cc
  License: `node-gyp-LICENSE.txt`.
- The application loopback implementation follows Microsoft's ApplicationLoopback
  sample design. Its original license is included as `audio-LICENSE.txt`.
- nlohmann/json 3.11.3 is a build dependency under MIT. CMake always installs the
  included license as `licenses/nlohmann-json-LICENSE.txt`.
  Source: https://github.com/nlohmann/json/tree/v3.11.3
- Windows Graphics Capture uses the documented C++/WinRT interoperability and
  free-threaded frame-pool lifecycle illustrated by the Microsoft samples and
  robmikh/Win32CaptureSample. No sample files or model assets are vendored.
  References: https://github.com/microsoft/Windows.UI.Composition-Win32-Samples
  and https://github.com/robmikh/Win32CaptureSample

The implementation does not use undocumented virtual-desktop pinning interfaces.
