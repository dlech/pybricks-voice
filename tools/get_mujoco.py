"""
Builds a MuJoCo wheel for this machine into vendor/wheels/, for a release.

MuJoCo publishes no Windows ARM64 builds, of the library or of its Python
bindings, so this builds both from pinned, checksummed sources: the library
with CMake and Visual Studio, then the bindings from the PyPI source package
against it. The bindings are built with clang-cl, as MuJoCo's own CI builds
its Windows wheels: with MSVC they fail to link, as it gives the code for the
unnamed structs inside mjVisual internal linkage in one file and not another.

pyproject.toml installs the wheel this built from a release of this
repository, so `uv sync` needs nothing built here. For a new MuJoCo version:
change VERSION and the hashes, build it, publish the wheel as a release asset,
and point [tool.uv.sources] at its URL.

Needs CMake, and a Visual Studio with the C++ workload and its "C++ Clang
tools for Windows" component. Run it without the project's environment:

    uv run --no-project tools/get_mujoco.py
"""

import hashlib
import io
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

VERSION = "3.17.0"
LIBRARY = (
    f"https://github.com/google-deepmind/mujoco/archive/refs/tags/{VERSION}.tar.gz",
    "cc96b572d9b8cc257ea963546f39b91a93166f10a3d25839e841e922b84b3905",
)
BINDINGS = (
    f"https://files.pythonhosted.org/packages/source/m/mujoco/mujoco-{VERSION}.tar.gz",
    "5e49c15905e7a39eaa1efb1cae809a81c5f5c2439467f23e7e6e383624d957be",
)
ROOT = pathlib.Path(__file__).resolve().parent.parent
WHEELS = ROOT / "vendor" / "wheels"
PYTHON = ROOT / ".python-version"
ARCH = {"ARM64": ("ARM64", "arm64"), "AMD64": ("x64", "x64")}[platform.machine().upper()]
VSWHERE = pathlib.Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")


def fetch(url: str, sha256: str, into: pathlib.Path) -> pathlib.Path:
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        sys.exit(f"{url} didn't match its checksum")
    into.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(into, filter="data")
    return next(into.iterdir())


def cmake() -> str:
    found = shutil.which("cmake") or r"C:\Program Files\CMake\bin\cmake.exe"
    if not pathlib.Path(found).exists():
        sys.exit("CMake not found: install it, e.g. winget install Kitware.CMake")
    return found


def clang_environment() -> tuple[dict[str, str], pathlib.Path]:
    """
    The developer environment of a Visual Studio that has clang-cl, and
    clang-cl itself.
    """
    installs = subprocess.run(
        [VSWHERE, "-all", "-products", "*", "-property", "installationPath"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for install in map(pathlib.Path, installs):
        clang = install / "VC" / "Tools" / "Llvm" / ("ARM64" if ARCH[0] == "ARM64" else "x64") / "bin" / "clang-cl.exe"
        vcvars = install / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        if clang.exists() and vcvars.exists():
            shown = subprocess.run(
                f'"{vcvars}" {ARCH[1]} >nul && set', shell=True, capture_output=True, text=True, check=True
            ).stdout
            environment = dict(line.split("=", 1) for line in shown.splitlines() if "=" in line)
            return environment, clang
    sys.exit('clang-cl not found: add "C++ Clang tools for Windows" to a Visual Studio install')


def main() -> None:
    if list(WHEELS.glob(f"mujoco-{VERSION}-*.whl")):
        print(f"a MuJoCo {VERSION} wheel is already in {WHEELS}", file=sys.stderr)
        return
    python = PYTHON.read_text().strip() if PYTHON.exists() else sys.executable
    environment, clang = clang_environment()

    # Short paths: MuJoCo's dependencies nest deeply enough to pass MAX_PATH.
    with tempfile.TemporaryDirectory(prefix="mj", dir=os.environ.get("SYSTEMDRIVE", "C:") + "\\") as temp:
        work = pathlib.Path(temp)
        library = fetch(*LIBRARY, work / "lib")
        bindings = fetch(*BINDINGS, work / "py")
        build, install, plugins = work / "b", work / "i", work / "p"

        options = [f"-DMUJOCO_{name}=OFF" for name in ("BUILD_EXAMPLES", "BUILD_SIMULATE", "BUILD_TESTS")]
        options += ["-DMUJOCO_TEST_PYTHON_UTIL=OFF", f"-DCMAKE_INSTALL_PREFIX={install}"]
        subprocess.run([cmake(), "-S", library, "-B", build, "-A", ARCH[0], *options], check=True)
        subprocess.run([cmake(), "--build", build, "--config", "Release", "--parallel"], check=True)
        subprocess.run([cmake(), "--install", build, "--config", "Release"], check=True)
        plugins.mkdir()
        for plugin in ("actuator", "elasticity", "sdf_plugin", "sensor"):
            shutil.copy2(build / "bin" / "Release" / f"{plugin}.dll", plugins)

        environment.update(
            MUJOCO_PATH=str(install),
            MUJOCO_PLUGIN_PATH=str(plugins),
            MUJOCO_CMAKE=cmake(),
            CMAKE_GENERATOR="Ninja",
            MUJOCO_CMAKE_ARGS=(
                f'-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF "-DCMAKE_C_COMPILER:STRING={clang}" '
                f'"-DCMAKE_CXX_COMPILER:STRING={clang}"'
            ),
        )
        WHEELS.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["uv", "build", "--wheel", "--python", python, bindings, "-o", WHEELS], env=environment, check=True
        )
    print(f"built {next(WHEELS.glob(f'mujoco-{VERSION}-*.whl'))}", file=sys.stderr)


if __name__ == "__main__":
    main()
