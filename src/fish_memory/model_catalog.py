"""Pinned sources and checksums for the local models.

Checksums are the original GitHub release files. Mirror copies were verified
to be byte-identical; `fetch_file` rejects anything else.
"""

from __future__ import annotations

from fish_memory.model_sources import ModelFile, hf_urls, modelscope_urls

_GITHUB = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# ---------- SenseVoice ASR: bundled; ModelScope, then Hugging Face ----------
_SENSEVOICE_MODELSCOPE = "pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue"
_SENSEVOICE = (
    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    "2365baeacb507f821a0c8120fcee3d484dba7a07",
)
SENSEVOICE_FILES = (
    ModelFile(
        "model.int8.onnx",
        "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
        239_233_841,
        (
            *modelscope_urls(_SENSEVOICE_MODELSCOPE, "model.int8.onnx"),
            *hf_urls(*_SENSEVOICE, "model.int8.onnx"),
        ),
        bundled="asr/sensevoice/model.int8.onnx",
    ),
    ModelFile(
        "tokens.txt",
        "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
        315_894,
        (
            *modelscope_urls(_SENSEVOICE_MODELSCOPE, "tokens.txt"),
            *hf_urls(*_SENSEVOICE, "tokens.txt"),
        ),
        bundled="asr/sensevoice/tokens.txt",
    ),
)

# ---------- Matcha TTS: model via ModelScope or the mirror; vocoder and espeak bundled
# No official ModelScope copy; this community copy was checked byte for byte.
_MATCHA_MODELSCOPE = "bujidc/matcha-icefall-zh-en"
_MATCHA = (
    "csukuangfj/matcha-icefall-zh-en",
    "fbe59d77057428c32111d5f3ea4ed67a7d440c61",
)


def _matcha(name: str, sha256: str, size: int) -> ModelFile:
    return ModelFile(
        name,
        sha256,
        size,
        (*modelscope_urls(_MATCHA_MODELSCOPE, name), *hf_urls(*_MATCHA, name)),
    )


MATCHA_FILES = (
    _matcha(
        "model-steps-3.onnx",
        "524286bf6cf11be74329ae1c682ac69e34d6860c2ea9fd1290319d561540b16a",
        75_717_082,
    ),
    _matcha(
        "lexicon.txt",
        "599efdcdaff4df2a123ce988c2cb90abcd59b919ef609ce36eb587d68b7ca2c0",
        1_400_278,
    ),
    _matcha(
        "tokens.txt",
        "77fee8e5e5dd96b3547119e6159292c648e99e065c54c97777722e3ce710b9a4",
        21_146,
    ),
    _matcha(
        "date-zh.fst",
        "eb8aa079ae3cb81d8f4404992f39d61a0cb990947512b5b8d1e54d1f6980e718",
        59_154,
    ),
    _matcha(
        "number-zh.fst",
        "743f402181fcfebf76cc2f0546b71fa26476e626fbe4e460fb7b4c3a7a8bd5bd",
        64_482,
    ),
    _matcha(
        "phone-zh.fst",
        "1ac2b6fa56b1442320c4de7db08353bab8963a2b57f365eebcdd3a2d3562f8d7",
        88_630,
    ),
    ModelFile(
        "vocos-16khz-univ.onnx",
        "b599142a1fb8ff03de3e84ac35ff537c619e56f4267a6fe894851a42844acf9e",
        53_882_848,
        (f"{_GITHUB}/vocoder-models/vocos-16khz-univ.onnx",),
        bundled="tts/vocos-16khz-univ.onnx",
    ),
)
ESPEAK_BUNDLE = "tts/espeak-ng-data"
ESPEAK_TREE_SHA256 = "1c2ec0747e40d30f8f123b65c93dcba64f2404195b3344ed614df41237df8332"

# ---------- Wake word: bundled (no official mirror) ----------
_KWS = "kws/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"


def kws_files(chunk_size: int) -> tuple[ModelFile, ...]:
    stem = f"epoch-13-avg-2-chunk-{chunk_size}-left-64"
    hashes = {
        8: (
            "2ca84d6bfe73e1ea3c9c49f600f7cad1c9ddd423c53c906b8bfe802444dd78d5",
            4_600_657,
        ),
        16: (
            "408bbd740838c42d5bf6d1c5b80b3c88b616c7860b92d980328b5b068c76ae48",
            4_599_656,
        ),
    }
    encoder_sha, encoder_size = hashes[chunk_size]
    return (
        ModelFile(
            f"encoder-{stem}.int8.onnx",
            encoder_sha,
            encoder_size,
            bundled=f"{_KWS}/encoder-{stem}.int8.onnx",
        ),
        ModelFile(
            f"decoder-{stem}.onnx",
            "63a22dd60f40fff082ac3e09afa507f6787da36df76ded2fbe145fa233e22c21",
            759_829,
            bundled=f"{_KWS}/decoder-{stem}.onnx",
        ),
        ModelFile(
            f"joiner-{stem}.int8.onnx",
            "190d4067b4cc20b72a42a1916e69d92052000fb7051a427ebb1bc72a69207dc1",
            86_629,
            bundled=f"{_KWS}/joiner-{stem}.int8.onnx",
        ),
        ModelFile(
            "tokens.txt",
            "2d3f32311f9b692b964da3c90e830258d3e78e013cb0c992dbfb15cd5a1a71b0",
            1_928,
            bundled=f"{_KWS}/tokens.txt",
        ),
        ModelFile(
            "en.phone",
            "f7000ec3a90544c0c7c16090d8951779c2b322e14dad5006290f498567d439ea",
            3_330_061,
            bundled=f"{_KWS}/en.phone",
        ),
    )


# ---------- Silero VAD: bundled; the mirror only has a different version ----------
SILERO_VAD_FILE = ModelFile(
    "silero_vad.onnx",
    "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    643_854,
    (f"{_GITHUB}/asr-models/silero_vad.onnx",),
    bundled="vad/silero_vad.onnx",
)

# ---------- Speaker models (voiceprint, 影音总结 speakers): mirror first ----------
_SPEAKERS = (
    "csukuangfj/speaker-embedding-models",
    "0743f301363dec56491a490f6d6cbc9d67f9a3bf",
)
_SEGMENTATION_MODELSCOPE = "pengzhendong/sherpa-onnx-pyannote-segmentation-3-0"
_SEGMENTATION = (
    "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
    "9403a6902bb58e3d5ae8c7e77c3422de279db2e0",
)


def speaker_embedding_urls(name: str, origin: str) -> tuple[str, ...]:
    return (*hf_urls(*_SPEAKERS, name), origin)


def segmentation_urls() -> tuple[str, ...]:
    return (
        *modelscope_urls(_SEGMENTATION_MODELSCOPE, "model.onnx"),
        *hf_urls(*_SEGMENTATION, "model.onnx"),
    )


def segmentation_license_urls() -> tuple[str, ...]:
    return hf_urls(*_SEGMENTATION, "LICENSE")


SEGMENTATION_LICENSE_SIZE = 1_061
SEGMENTATION_LICENSE_SHA256 = (
    "14d7016ad68e7394d6e6b78d96cc2ae431c905287b89674cfdf021e79e62b8ba"
)


# ---------- Turn detection and memory search: ModelScope into the Hub cache ----------
SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
SMART_TURN_FILES = (
    ModelFile(
        "smart-turn-v3.2-cpu.onnx",
        "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
        8_679_182,
        modelscope_urls(SMART_TURN_REPO, "smart-turn-v3.2-cpu.onnx"),
    ),
)

E5_SMALL_REPO = "intfloat/multilingual-e5-small"
E5_SMALL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
_E5_SMALL = {
    "onnx/model_qint8_avx512_vnni.onnx": (
        "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
        118_346_824,
    ),
    "config.json": (
        "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
        655,
    ),
    "sentencepiece.bpe.model": (
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
        5_069_051,
    ),
    "special_tokens_map.json": (
        "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
        167,
    ),
    "tokenizer.json": (
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        17_082_730,
    ),
    "tokenizer_config.json": (
        "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
        443,
    ),
}
E5_SMALL_FILES = tuple(
    ModelFile(name, sha256, size, modelscope_urls(E5_SMALL_REPO, name))
    for name, (sha256, size) in _E5_SMALL.items()
)
