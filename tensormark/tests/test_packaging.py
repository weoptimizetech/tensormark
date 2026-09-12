"""Model-free packaging gates: python -m unittest discover -s tests -p test_packaging.py.

Set TENSORMARK_SDIST to check an existing archive; otherwise build an sdist
without isolation (requires build, setuptools, wheel and pybind11 installed).
"""

import email
import importlib.machinery
import importlib.util
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "packaging_loader", ROOT / "tensormark" / "_loader.py"
)
loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loader)


def native_version():
    return re.search(r'^#define TENSORMARK_VERSION "([^"]+)"',
                     (ROOT / "version.h").read_text(), re.M)[1]


def metadata_version():
    # Read only the PEP 621 project table; works on Python 3.10 without tomli.
    project = (ROOT / "pyproject.toml").read_text().split("[project]", 1)[1]
    project = project.split("\n[", 1)[0]
    return re.search(r'^version\s*=\s*"([^"]+)"', project, re.M)[1]


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name) / "tensormark"
        self.package.mkdir()
        self.dev = self.package.parent / "tm"
        self.dev.mkdir()
        self.suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def binary(self, directory, suffix=None):
        path = directory / ("tensormark" + (suffix or self.suffix))
        path.touch()
        return str(path)

    def test_packaged_beats_dev(self):
        expected = self.binary(self.package)
        self.binary(self.dev, ".so")
        self.assertEqual(loader.find_native(self.package), expected)

    def test_current_abi_beats_generic_and_other_abi(self):
        self.binary(self.package, ".so")
        self.binary(self.package, ".cpython-999-darwin.so")
        expected = self.binary(self.package)
        self.assertEqual(loader.find_native(self.package), expected)

    def test_suffix_order_is_interpreter_order(self):
        paths = [self.binary(self.package, suffix)
                 for suffix in reversed(importlib.machinery.EXTENSION_SUFFIXES)]
        self.assertEqual(loader.find_native(self.package), paths[-1])

    def test_generic_packaged_beats_current_dev(self):
        expected = self.binary(self.package, ".so")
        self.binary(self.dev)
        self.assertEqual(loader.find_native(self.package), expected)

    def test_dev_fallback(self):
        expected = self.binary(self.dev, ".so")
        self.assertEqual(loader.find_native(self.package), expected)

    def test_wrong_packaged_abi_allows_dev_fallback(self):
        self.binary(self.package, ".cpython-999-darwin.so")
        expected = self.binary(self.dev, ".so")
        self.assertEqual(loader.find_native(self.package), expected)

    def test_wrong_abis_are_not_candidates(self):
        for directory in (self.package, self.dev):
            self.binary(directory, ".cpython-999-darwin.so")
        with self.assertRaisesRegex(ImportError, "this interpreter"):
            loader.find_native(self.package)

    def test_missing_extension_has_actionable_error(self):
        with self.assertRaisesRegex(ImportError, "build.sh.*pip install"):
            loader.find_native(self.package)

    def test_directory_is_not_extension(self):
        (self.package / ("tensormark" + self.suffix)).mkdir()
        with self.assertRaises(ImportError):
            loader.find_native(self.package)

    def test_broken_packaged_binary_does_not_fall_back(self):
        # Exercise the actual package initializer, not only path discovery.
        for source in ("__init__.py", "_loader.py"):
            (self.package / source).write_bytes((ROOT / "tensormark" / source).read_bytes())
        expected = self.binary(self.package)
        self.binary(self.dev, ".so")
        result = subprocess.run(
            [sys.executable, "-c", "import tensormark"], cwd=self.temp.name,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(expected, result.stderr)
        self.assertNotIn(str(self.dev / "tensormark.so"), result.stderr)


class VersionTests(unittest.TestCase):
    def test_native_header_matches_metadata(self):
        self.assertEqual(native_version(), metadata_version())

    def test_binding_uses_shared_version_header(self):
        source = (ROOT / "tensormark.cpp").read_text()
        self.assertIn('#include "version.h"', source)
        self.assertIn('mod.attr("__version__") = TENSORMARK_VERSION;', source)


class SourceArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing = os.environ.get("TENSORMARK_SDIST")
        if existing:
            archive_path = Path(existing)
        else:
            cls.temp = tempfile.TemporaryDirectory()
            cls.addClassCleanup(cls.temp.cleanup)
            subprocess.run(
                [sys.executable, "-m", "build", "--sdist", "--no-isolation",
                 "--outdir", cls.temp.name, str(ROOT)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            archive_path, = Path(cls.temp.name).glob("*.tar.gz")
        with tarfile.open(archive_path) as archive:
            cls.files = {}
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise AssertionError(f"unsafe archive member: {member.name}")
                if member.isfile():
                    cls.files[str(PurePosixPath(*path.parts[1:]))] = archive.extractfile(member).read()

    def test_required_build_inputs(self):
        required = {
            "setup.py", "pyproject.toml", "MANIFEST.in", "README.md", "build.sh",
            "LICENSE", "NOTICE",
            "tensormark.cpp", "engine.h", "version.h", "autograd.h", "rowops.h",
            "attention.h", "kda.h", "optim.h", "_native/neural_demo.cpp",
            "tensormark/__init__.py", "tensormark/_loader.py", "tensormark/torch.py",
            "tests/test_packaging.py", "tests/smoke_installed.py",
        }
        self.assertFalse(required - self.files.keys(), required - self.files.keys())

    def test_engine_is_identical_to_maintained_source(self):
        source = ROOT / "_native/neural_demo.cpp"
        if not source.is_file():
            source = ROOT.parent / "neural_demo.cpp"
        self.assertEqual(self.files["_native/neural_demo.cpp"], source.read_bytes())

    def test_native_local_include_closure(self):
        # All active archive includes must resolve inside the archive; only
        # engine.h's explicit checkout fallback is allowed to point outside.
        pending, seen = ["tensormark.cpp"], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            text = self.files[name].decode()
            for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, re.M):
                if name == "engine.h" and include == "../neural_demo.cpp":
                    continue
                target = str(PurePosixPath(name).parent / include)
                self.assertIn(target, self.files, f"{name} needs {target}")
                pending.append(target)
        self.assertIn("_native/neural_demo.cpp", seen)
        self.assertIn("rowops.h", seen)

    def test_manifest_and_metadata(self):
        manifest, = [data.decode() for name, data in self.files.items()
                     if name.endswith(".egg-info/SOURCES.txt")]
        self.assertIn("_native/neural_demo.cpp", manifest.splitlines())
        metadata = email.message_from_bytes(self.files["PKG-INFO"])
        self.assertEqual(metadata["Version"], native_version())

    def test_license_metadata_and_texts(self):
        metadata = email.message_from_bytes(self.files["PKG-INFO"])
        self.assertEqual(metadata["License-Expression"], "Apache-2.0")
        self.assertEqual(set(metadata.get_all("License-File")), {"LICENSE", "NOTICE"})
        for name in ("LICENSE", "NOTICE"):
            self.assertEqual(self.files[name], (ROOT / name).read_bytes())
            root_copy = ROOT.parent / name
            if root_copy.is_file():
                self.assertEqual((ROOT / name).read_bytes(), root_copy.read_bytes())

    def test_no_prebuilt_binaries(self):
        self.assertFalse([name for name in self.files
                          if name.endswith((".so", ".pyd", ".dylib", ".pyc"))])


if __name__ == "__main__":
    unittest.main()
