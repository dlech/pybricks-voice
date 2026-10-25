"""Copies the LDraw parts the robot model uses into pybricks_voice/models/ldraw/.

The 3D view draws the robot from LDraw's parts library, which is CC BY 4.0
(see CAreadme.txt there), but needs only a few hundred of its files, so those
are kept in the repository and this script is only for changing the model.
It reads the library's complete.zip, from ldraw.org or a local copy, and takes
the files the models reference, recursively, plus the color table.

    uv run --no-project tools/get_ldraw.py [--zip complete.zip]
"""

import argparse
import io
import pathlib
import re
import shutil
import sys
import urllib.request
import zipfile

URL = "https://library.ldraw.org/library/updates/complete.zip"
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "pybricks_voice" / "models"
OUT = MODELS / "ldraw"
# Parts the 3D view places itself, besides those in the models.
EXTRA = ["39367p01.dat"]  # the wheel
KEEP = ["LDConfig.ldr", "CAreadme.txt", "CAlicense.txt", "CAlicense4.txt"]
REFERENCE = re.compile(r"^\s*1\s+(?:\S+\s+){13}(.+?)\s*$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=pathlib.Path, help="a downloaded complete.zip")
    options = parser.parse_args()
    if options.zip:
        data = options.zip.read_bytes()
    else:
        print(f"downloading {URL}", file=sys.stderr)
        with urllib.request.urlopen(URL) as response:
            data = response.read()

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = {name.lower(): name for name in archive.namelist()}

        def find(reference: str) -> str:
            relative = reference.replace("\\", "/").lower()
            for folder in ("ldraw/parts/", "ldraw/p/"):
                if folder + relative in files:
                    return files[folder + relative]
            raise SystemExit(f"{reference} is not in the library")

        def references(text: str) -> list[str]:
            return [m.group(1) for line in text.splitlines() if (m := REFERENCE.match(line))]

        wanted: dict[str, bytes] = {}
        queue = [r for model in MODELS.glob("*.ldr") for r in references(model.read_text())] + EXTRA
        while queue:
            path = find(queue.pop())
            if path not in wanted:
                wanted[path] = archive.read(path)
                queue += references(wanted[path].decode("utf-8", errors="replace"))
        for name in KEEP:
            wanted[files[f"ldraw/{name}".lower()]] = archive.read(files[f"ldraw/{name}".lower()])

    shutil.rmtree(OUT, ignore_errors=True)
    for path, content in wanted.items():
        target = OUT / path.removeprefix("ldraw/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    size = sum(len(c) for c in wanted.values())
    print(f"wrote {len(wanted)} files, {size / 1e6:.1f} MB, to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
