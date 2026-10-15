# Third-party notices

## Live2D Cubism Web Framework

- Upstream: https://github.com/Live2D/CubismWebFramework
- Version: `5-r.5`
- Commit: `198a3769c26ca3d7b600e932590433badd392edd`
- License: Live2D Open Software License; see
  `vendor/cubism-web-framework/LICENSE.md`.

## Live2D Cubism Core and the Mao sample model

The proprietary Live2D Cubism Core (`live2dcubismcore.min.js`, from
`CubismSdkForWeb-5-r.5`) is not stored in this repository. Packaged builds
download it at build time (`scripts/prepare-bundled-live2d.ts`) and ship it
inside the installer; development builds download it from Live2D's official
distribution on first launch. It is used under the
[Live2D Proprietary Software License](https://www.live2d.com/eula/live2d-proprietary-software-license-agreement_en.html).
Distributing an application that contains the Cubism Core may require a Cubism
SDK Release License depending on the distributor; see
https://www.live2d.com/en/sdk/license/.

The official sample model "Mao" is not stored in this repository either. It is
fetched from [Live2D/CubismWebSamples](https://github.com/Live2D/CubismWebSamples)
at a pinned commit, bundled into installers as the default look, and used under
the [Live2D Free Material License](https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html).
`public-runtime/presets/mao-preview.png` is a render of that sample.

Live2D models you import yourself are yours to license; tulpa only stores them
in its data folder.

## LiveKit local audio helper

tulpa's local microphone-processing helper pins LiveKit Swift
`2.16.0`, LiveKit UniFFI XCFramework `0.0.6`, LiveKit WebRTC XCFramework
`144.7559.11`, and SwiftProtobuf `1.34.1`. The helper is used locally without a
LiveKit Room or cloud connection. Its complete Apache-2.0, MIT, WebRTC BSD
3-Clause, and notice texts are packaged inside
`FishMemoryAudioHelper.app/Contents/Resources/Licenses`, with an index in that
helper's `THIRD_PARTY_NOTICES.md`.

## Local system-audio speaker models (downloaded on first use)

System-audio awareness optionally downloads the MIT-licensed
`pyannote/segmentation-3.0` model through the pinned k2-fsa/sherpa-onnx ONNX
export. Its license is retained beside the model as
`speaker-diarization/segmentation-3.0.LICENSE` in the application's data folder.
The model is executed locally with ONNX Runtime; pyannote.audio and PyTorch
are not bundled for this feature.

- Model: https://huggingface.co/pyannote/segmentation-3.0
- Export: https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/pyannote/segmentation
- Speaker embeddings: the Apache-2.0 CAM++ model from
  https://github.com/modelscope/3D-Speaker, using the existing sherpa-onnx
  speaker-embedding runtime. The public model file can be shared with owner
  verification; enrolled owner voice profiles are not used for media speakers.

Cross-window matching references the local/global speaker constraints described
by Diart (https://github.com/juanmc2005/diart). No Diart framework code or model
runtime is included.

## Speech models shipped in the installers

Installers include these models so voice works on first launch
(`scripts/prepare-bundled-models.py`). Each file is verified against the
pinned checksum in `src/fish_memory/model_catalog.py`.

| Model | Used for | Upstream | License |
| --- | --- | --- | --- |
| Silero VAD | voice activity detection | https://github.com/snakers4/silero-vad | MIT (`packaging/licenses/livekit-vad/`) |
| sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 | wake word | https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models | Apache-2.0 (sherpa-onnx) |
| SenseVoice small int8 (sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17) | speech recognition | https://github.com/FunAudioLLM/SenseVoice | FunASR Model Open Source License |
| vocos-16khz-univ | local voice vocoder | https://github.com/k2-fsa/sherpa-onnx/releases/tag/vocoder-models | MIT (Vocos) |
| espeak-ng-data | local voice phonemes | https://github.com/espeak-ng/espeak-ng | GPL-3.0-or-later |

The espeak-ng data files are distributed unmodified, as extracted from the
`matcha-icefall-zh-en` archive of the sherpa-onnx TTS models. Their
corresponding source is the espeak-ng project linked above; the full GPL-3.0
text is at https://www.gnu.org/licenses/gpl-3.0.txt.

## Models downloaded on first use

These are not shipped; the app downloads them from their upstream hosts when a
feature needs them, verifies the pinned checksum, and keeps them in its data
folder. Their licenses are those stated upstream.

- `matcha-icefall-zh-en` local TTS acoustic model —
  https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
- Smart Turn v3 end-of-turn model — https://github.com/pipecat-ai/smart-turn
- `intfloat/multilingual-e5-small` text embeddings —
  https://huggingface.co/intfloat/multilingual-e5-small
- 3D-Speaker CAM++ speaker embeddings (owner voice) —
  https://github.com/modelscope/3D-Speaker
- `pyannote/segmentation-3.0` (see below)

## i18next

Interface localization uses i18next `25.6.3` under the MIT license.
Upstream: https://github.com/i18next/i18next

The MIT License (MIT)

Copyright (c) 2025 i18next

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
