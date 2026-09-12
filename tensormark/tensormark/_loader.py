"""Native-extension discovery, kept separate so it can be tested without C++."""

from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path


def find_native(package_dir):
    """Prefer packaged binaries in this interpreter's suffix order, then dev.

    Exact filenames avoid selecting extensions tagged for another Python ABI.
    A selected binary's load error must propagate, not silently fall back.
    """
    package_dir = Path(package_dir)
    for directory in (package_dir, package_dir.parent / "tm"):
        for suffix in EXTENSION_SUFFIXES:
            candidate = directory / ("tensormark" + suffix)
            if candidate.is_file():
                return str(candidate)
    raise ImportError(
        "tensormark native extension not found for this interpreter; "
        "run ./build.sh in the repo or `pip install .`"
    )
