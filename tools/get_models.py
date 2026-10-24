"""Downloads Whisper-Small (w8a16, precompiled for the Snapdragon X Elite NPU)
from Qualcomm AI Hub, and the Whisper vocabulary from Hugging Face, into models/.

Usage: uv run tools/get_models.py
"""

import io
import json
import pathlib
import shutil
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent / "models"
WHISPER_ZIP = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/"
    "whisper_small_quantized/releases/v0.65.0/"
    "whisper_small_quantized-precompiled_qnn_onnx-w8a16-qualcomm_snapdragon_x_elite.zip"
)
TOKENIZER = "https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json"


def fetch(url: str) -> bytes:
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as response:
        return response.read()


def main() -> None:
    target = ROOT / "whisper-small"
    target.mkdir(parents=True, exist_ok=True)

    if not (target / "decoder.onnx").exists():
        with zipfile.ZipFile(io.BytesIO(fetch(WHISPER_ZIP))) as archive:
            for member in archive.infolist():
                name = pathlib.PurePosixPath(member.filename).name
                if name and not member.is_dir():
                    with archive.open(member) as src, open(target / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    vocab = target / "vocab.json"
    if not vocab.exists():
        # Only the id -> token table is needed; tokenizer.json carries it as model.vocab
        # plus the added (special) tokens.
        data = json.loads(fetch(TOKENIZER))
        table = {token: i for token, i in data["model"]["vocab"].items()}
        for added in data.get("added_tokens", []):
            table[added["content"]] = added["id"]
        vocab.write_text(json.dumps(table), encoding="utf-8")

    print(f"models ready in {target}")


if __name__ == "__main__":
    main()
