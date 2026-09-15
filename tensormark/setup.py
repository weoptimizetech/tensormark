import os
import sysconfig
from pathlib import Path
import shutil

# std::filesystem APIs (neural_demo.cpp) require macOS >= 10.15; pin the
# wheel build target explicitly so pip's cross-env doesn't ship a lower one.
os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")

import pybind11
from setuptools import Extension, setup
from setuptools.command.sdist import sdist


ROOT = Path(__file__).resolve().parent
ENGINE = Path("_native/neural_demo.cpp")
ENGINE_SOURCE = ENGINE if (ROOT / ENGINE).is_file() else Path("../neural_demo.cpp")
HEADERS = ["engine.h", "version.h", "autograd.h", "attention.h", "kda.h",
           "optim.h", "rowops.h"]


class SourceDistribution(sdist):
    """Stage the maintained engine only in the source archive's release tree."""

    def make_release_tree(self, base_dir, files):
        source = ROOT / ENGINE_SOURCE
        if not source.is_file():
            raise FileNotFoundError(
                "Missing neural_demo.cpp: build from a complete checkout or "
                "a TensorMark source distribution."
            )
        super().make_release_tree(base_dir, files)
        destination = Path(base_dir) / ENGINE
        destination.parent.mkdir(parents=True, exist_ok=True)
        # An sdist rebuilt from an sdist may already contain a hard-linked copy.
        if destination.exists():
            destination.unlink()
        shutil.copy2(source, destination)
        # Keep the archive's source manifest complete without modifying the
        # checkout's egg-info (setuptools may hard-link release-tree files).
        for manifest in Path(base_dir).glob("*.egg-info/SOURCES.txt"):
            entries = set(manifest.read_text(encoding="utf-8").splitlines())
            entries.add(ENGINE.as_posix())
            manifest.unlink()
            manifest.write_text("\n".join(sorted(entries)) + "\n", encoding="utf-8")


# TENSORMARK_ARCH overrides the default build-host CPU target when preparing
# wheels for a different supported Apple Silicon machine. Validate portability
# separately; choosing an architecture alone is not a compatibility guarantee.
arch = os.environ.get("TENSORMARK_ARCH", "native")
cflags = [
    "-std=c++23",
    "-O3",
    f"-march={arch}",
    "-undefined", "dynamic_lookup",
]

setup(
    cmdclass={"sdist": SourceDistribution},
    ext_modules=[
        Extension(
            "tensormark.tensormark",
            sources=["tensormark.cpp"],
            depends=[*HEADERS, str(ENGINE_SOURCE)],
            include_dirs=[sysconfig.get_paths()["include"],
                          pybind11.get_include()],
            extra_compile_args=cflags,
            extra_link_args=["-framework", "Accelerate"],
            language="c++",
        )
    ],
)
