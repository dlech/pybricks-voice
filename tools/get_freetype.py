"""
Builds the FreeType DLL that pygfx needs, into vendor/.

pygfx imports freetype-py, which bundles FreeType on most platforms but not on
Windows ARM64. This downloads a pinned FreeType release from the official
GitHub mirror, checks it, and builds freetype.dll for this machine with CMake
and Visual Studio (2026 or later, with the C++ workload).

Usage: uv run tools/get_freetype.py
"""

import hashlib
import io
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

VERSION = "2.14.3"
SOURCE = f"https://github.com/freetype/freetype/archive/refs/tags/VER-{VERSION.replace('.', '-')}.tar.gz"
SHA256 = "dc49de6b01a266eef4876a4dd34d9842c475d3e28ff2eff63bd2fb760ab56261"
TARGET = pathlib.Path(__file__).resolve().parent.parent / "vendor" / "freetype.dll"
ARCH = {"ARM64": "ARM64", "AMD64": "x64"}[platform.machine().upper()]


def cmake() -> str:
    found = shutil.which("cmake") or r"C:\Program Files\CMake\bin\cmake.exe"
    if not pathlib.Path(found).exists():
        sys.exit("CMake not found: install it, e.g. winget install Kitware.CMake")
    return found


def main() -> None:
    if TARGET.exists():
        print(f"{TARGET} is already there", file=sys.stderr)
        return
    print(f"downloading {SOURCE}", file=sys.stderr)
    with urllib.request.urlopen(SOURCE) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        sys.exit("the FreeType source didn't match its checksum")

    with tempfile.TemporaryDirectory() as temp:
        work = pathlib.Path(temp)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(work, filter="data")
        source = next(work.glob("freetype-*"))
        build = work / "build"
        options = [
            "-DBUILD_SHARED_LIBS=ON",
            # Only what pygfx's text needs: no compressed or color fonts, no shaping.
            *(f"-DFT_DISABLE_{name}=ON" for name in ("ZLIB", "BZIP2", "PNG", "HARFBUZZ", "BROTLI")),
        ]
        subprocess.run([cmake(), "-S", source, "-B", build, "-A", ARCH, *options], check=True)
        subprocess.run([cmake(), "--build", build, "--config", "Release", "--parallel"], check=True)
        TARGET.parent.mkdir(exist_ok=True)
        shutil.copy2(build / "Release" / "freetype.dll", TARGET)
    print(f"built {TARGET}", file=sys.stderr)


if __name__ == "__main__":
    main()
