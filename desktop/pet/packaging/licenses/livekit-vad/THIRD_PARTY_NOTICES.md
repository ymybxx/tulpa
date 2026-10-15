# Local LiveKit VAD runtime notices

Fish Memory includes the official `livekit-plugins-silero` 1.7.1 plugin and
`livekit-agents` 1.7.1. Only the local Silero VAD plugin is selected; optional
LiveKit cloud/provider plugins are not collected by the packaging hook.

- LiveKit Agents and the Silero integration code are copyright LiveKit, Inc.
  and licensed under Apache-2.0. Upstream source and license:
  https://github.com/livekit/agents/tree/livekit-agents%401.7.1
  https://github.com/livekit/agents/blob/livekit-agents%401.7.1/LICENSE
- The bundled Silero VAD model is copyright 2020-present Silero Team and is
  licensed under MIT. Its full notice is in `Silero-VAD-MIT-LICENSE.txt`.
  Upstream: https://github.com/snakers4/silero-vad/tree/v6.2
- LiveKit RTC includes its upstream license at
  `livekit/rtc/resources/LICENSE.md` in the frozen runtime.
- The required `livekit-local-inference` distribution includes Apache-2.0
  `LICENSE` and a separate `MODEL_LICENSE`; both are retained under its
  `.dist-info/licenses` directory. They are not replaced by the Silero license.
- PyAV 14.2.0 and its bundled FFmpeg libraries are retained as distributed by
  the upstream wheel. PyAV's own BSD license remains in its `.dist-info`
  directory. That BSD license is not the license of all bundled codecs: the
  wheel's `libavcodec` reports `GPL version 3 or later` and its build enables
  GPL components including x264 and x265. The full FFmpeg license texts are
  also retained in the FFmpeg section of `livekit/rtc/resources/LICENSE.md`.
  Upstream build sources and further licensing information are at
  https://github.com/PyAV-Org/PyAV/tree/v14.2.0 and
  https://github.com/PyAV-Org/pyav-ffmpeg.

Metadata, licenses and notices included by the required dependency wheels are
kept in their corresponding `.dist-info` directories. The packaging hook does
not download model files or execute provider/room connections.
