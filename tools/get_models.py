"""
Downloads Whisper-Small for this machine into models/, and the Whisper
vocabulary from Hugging Face:

- on Windows, Qualcomm AI Hub's (w8a16, precompiled for the Snapdragon X
  Elite NPU)
- on macOS, WhisperKit's Core ML models, which it then loads once for the
  Neural Engine: that compiles them for this Mac's Neural Engine, which takes
  minutes the first time, so the app doesn't have to

Usage: uv run tools/get_models.py
"""

import hashlib
import io
import json
import pathlib
import shutil
import socket
import sys
import time
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent / "models"
WHISPER_ZIP = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/"
    "whisper_small_quantized/releases/v0.65.0/"
    "whisper_small_quantized-precompiled_qnn_onnx-w8a16-qualcomm_snapdragon_x_elite.zip"
)
TOKENIZER = "https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json"
# argmaxinc/whisperkit-coreml (MIT), pinned.
COREML_REPO = "argmaxinc/whisperkit-coreml"
COREML_REVISION = "0f63a7800b00dd0226abd051b906c246e1907482"
COREML_MODELS = ("AudioEncoder.mlmodelc", "TextDecoder.mlmodelc")


TIMEOUT_S = 60  # for connecting, and for each read

# urllib tries a host's addresses one at a time, waiting on each, with nothing
# like curl's racing of IPv6 against IPv4 (RFC 8305). Where IPv6 doesn't get
# through (as from one Mac here, to Hugging Face's CDN), that is a minute per
# IPv6 address: try IPv4 first. An IPv6-only network still gets there.
_getaddrinfo = socket.getaddrinfo


def _ipv4_first(*args, **kwargs):
    return sorted(_getaddrinfo(*args, **kwargs), key=lambda info: info[0] != socket.AF_INET)


socket.getaddrinfo = _ipv4_first


def fetch(url: str) -> bytes:
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return response.read()


def get_qnn(target: pathlib.Path) -> None:
    if (target / "decoder.onnx").exists():
        return
    with zipfile.ZipFile(io.BytesIO(fetch(WHISPER_ZIP))) as archive:
        for member in archive.infolist():
            name = pathlib.PurePosixPath(member.filename).name
            if name and not member.is_dir():
                with archive.open(member) as src, open(target / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def get_coreml(target: pathlib.Path) -> None:
    """WhisperKit's encoder and decoder, each file checked against the hash Hugging Face lists."""
    folder = "openai_whisper-small"
    listing = json.loads(
        fetch(f"https://huggingface.co/api/models/{COREML_REPO}/tree/{COREML_REVISION}/{folder}?recursive=true")
    )
    for entry in listing:
        relative = pathlib.PurePosixPath(entry["path"]).relative_to(folder)
        if entry["type"] != "file" or relative.parts[0] not in COREML_MODELS:
            continue
        path = target / relative
        if path.exists() and path.stat().st_size == entry["size"]:
            continue
        # Large files are in LFS, whose oid is their SHA-256; the rest have git's blob SHA-1.
        lfs = entry.get("lfs")
        digest = hashlib.sha256() if lfs else hashlib.sha1(f"blob {entry['size']}\0".encode())
        url = f"https://huggingface.co/{COREML_REPO}/resolve/{COREML_REVISION}/{entry['path']}"
        print(f"downloading {url}", file=sys.stderr)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response, open(partial, "wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != (lfs["oid"] if lfs else entry["oid"]):
            partial.unlink()
            sys.exit(f"{relative}: not what Hugging Face lists")
        partial.rename(path)


def prepare_neural_engine(target: pathlib.Path) -> None:
    """Loads the models for the Neural Engine once, so the app finds them compiled."""
    assert sys.platform == "darwin"  # and type checking for another platform stops here
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from pybricks_voice.macos.whisper_coreml import MLComputeUnitsCPUAndNeuralEngine, load

    for name in COREML_MODELS:
        print(f"preparing {name} for the Neural Engine (minutes, the first time)…", file=sys.stderr)
        t = time.perf_counter()
        load(target / name, MLComputeUnitsCPUAndNeuralEngine)
        print(f"  {time.perf_counter() - t:.1f} s", file=sys.stderr)


def main() -> None:
    vocabulary = ROOT / "whisper-small"
    vocabulary.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        get_qnn(vocabulary)
    elif sys.platform == "darwin":
        get_coreml(ROOT / "whisper-small-coreml")

    vocab = vocabulary / "vocab.json"
    if not vocab.exists():
        # Only the id -> token table is needed; tokenizer.json carries it as model.vocab
        # plus the added (special) tokens.
        data = json.loads(fetch(TOKENIZER))
        table = {token: i for token, i in data["model"]["vocab"].items()}
        for added in data.get("added_tokens", []):
            table[added["content"]] = added["id"]
        vocab.write_text(json.dumps(table), encoding="utf-8")

    if sys.platform == "darwin":
        prepare_neural_engine(ROOT / "whisper-small-coreml")
    print(f"models ready in {ROOT}")


if __name__ == "__main__":
    main()
