"""Collect the models shipped inside the app: wake word, Silero VAD, the TTS
vocoder and espeak-ng data (no official mainland mirror), plus the SenseVoice
speech recognizer so TA can hear on first launch without a 240 MB download.

Usage (from the repository root):
    uv run --locked python desktop/pet/scripts/prepare-bundled-models.py \
        desktop/pet/runtime/mac-arm64/models [--seed <existing models-runtime dir>]

Every file is verified against fish_memory.model_catalog before it is kept.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from fish_memory.kws.models import KWS_MODEL_NAME, KWS_MODEL_URL
from fish_memory.model_catalog import (
    ESPEAK_BUNDLE,
    ESPEAK_TREE_SHA256,
    MATCHA_FILES,
    SENSEVOICE_FILES,
    SILERO_VAD_FILE,
    kws_files,
)
from fish_memory.model_sources import (
    ModelFile,
    _stream,
    fetch_file,
    file_matches,
    tree_digest,
)
from fish_memory.tts.local import MATCHA_MODEL_NAME, MATCHA_MODEL_URL

VOCODER = next(file for file in MATCHA_FILES if file.bundled)
KWS_FILES = {file.name: file for chunk in (8, 16) for file in kws_files(chunk)}


def log(message: str) -> None:
    print(f"[bundled-models] {message}", flush=True)


def place(file: ModelFile, output: Path, seed: Path | None) -> None:
    target = output / str(file.bundled)
    if file_matches(target, file.sha256, file.size):
        return
    if seed is not None:
        for candidate in seed.rglob(file.name):
            if file_matches(candidate, file.sha256, file.size):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, target)
                log(f"copied {file.name} from seed")
                return
    source = ModelFile(file.name, file.sha256, file.size, file.urls)
    log(f"downloading {file.name}")
    fetch_file(source, target)


def extract(url: str, archive_name: str, members: callable, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / archive_name
        log(f"downloading {archive_name}")
        _stream(url, archive, lambda _done, _total: None)
        root = destination.resolve()
        with tarfile.open(archive, "r:bz2") as bundle:
            for member in bundle.getmembers():
                relative = members(member.name)
                if relative is None or not (member.isfile() or member.isdir()):
                    continue
                target = (root / relative).resolve()
                if root not in target.parents and target != root:
                    raise SystemExit(f"unsafe archive path: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                assert source is not None
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def prepare_kws(output: Path, seed: Path | None) -> None:
    missing = [
        file
        for file in KWS_FILES.values()
        if not file_matches(output / str(file.bundled), file.sha256, file.size)
    ]
    if missing and seed is not None:
        for file in missing:
            place(file, output, seed)
        missing = [
            file
            for file in missing
            if not file_matches(output / str(file.bundled), file.sha256, file.size)
        ]
    if not missing:
        return
    prefix = f"{KWS_MODEL_NAME}/"
    extract(
        KWS_MODEL_URL,
        f"{KWS_MODEL_NAME}.tar.bz2",
        lambda name: (
            name
            if name.startswith(prefix) and name[len(prefix) :] in KWS_FILES
            else None
        ),
        output / "kws",
    )
    for file in KWS_FILES.values():
        if not file_matches(output / str(file.bundled), file.sha256, file.size):
            raise SystemExit(f"wake word file failed verification: {file.name}")


def prepare_espeak(output: Path, seed: Path | None) -> None:
    target = output / ESPEAK_BUNDLE
    if target.is_dir() and tree_digest(target) == ESPEAK_TREE_SHA256:
        return
    shutil.rmtree(target, ignore_errors=True)
    if seed is not None:
        for candidate in seed.rglob("espeak-ng-data"):
            if candidate.is_dir() and tree_digest(candidate) == ESPEAK_TREE_SHA256:
                shutil.copytree(candidate, target)
                log("copied espeak-ng-data from seed")
                return
    prefix = f"{MATCHA_MODEL_NAME}/espeak-ng-data"
    extract(
        MATCHA_MODEL_URL,
        f"{MATCHA_MODEL_NAME}.tar.bz2",
        lambda name: (
            "espeak-ng-data" + name[len(prefix) :]
            if name == prefix or name.startswith(prefix + "/")
            else None
        ),
        target.parent,
    )
    if tree_digest(target) != ESPEAK_TREE_SHA256:
        raise SystemExit("espeak-ng-data failed verification")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--seed", type=Path, help="an existing models-runtime directory to copy from"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = args.seed.resolve() if args.seed else None
    place(SILERO_VAD_FILE, output, seed)
    place(VOCODER, output, seed)
    for file in SENSEVOICE_FILES:
        place(file, output, seed)
    prepare_kws(output, seed)
    prepare_espeak(output, seed)
    log(f"ready: {output}")


if __name__ == "__main__":
    sys.exit(main())
