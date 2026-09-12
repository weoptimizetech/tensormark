"""Lossless model cold-storage contracts; fixtures never touch real weights."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("model_archive", Path(__file__).resolve().parents[1] / "model_archive.py")
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class ModelArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "model.mlpackage"
        self.source.mkdir()
        (self.source / "empty").mkdir()
        self.weights = self.source / "weights.bin"
        self.payload = bytes(range(256)) * 4096
        self.weights.write_bytes(self.payload)
        os.utime(self.weights, ns=(1700000000000000000, 1700000000000000000))
        self.packed = self.root / "model.mlpackage.tar.gz"

    def test_directory_roundtrip_prune_preserves_content_and_mtime(self):
        result = archive.pack(self.source, self.packed, remove_source=True)
        self.assertTrue(result["verified"] and result["source_removed"])
        self.assertLess(result["archive_bytes"], result["source_bytes"])
        self.assertFalse(self.source.exists())
        archive.restore(self.packed, self.source)
        self.assertEqual(self.weights.read_bytes(), self.payload)
        self.assertEqual(self.weights.stat().st_mtime_ns, 1700000000000000000)
        self.assertTrue((self.source / "empty").is_dir())
        self.assertTrue(self.packed.exists())

    def test_file_roundtrip_default_retains_source(self):
        result = archive.pack(self.weights, self.packed)
        self.assertFalse(result["source_removed"])
        target = self.root / "restored.bin"
        archive.restore(self.packed, target)
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(target.stat().st_mtime_ns, self.weights.stat().st_mtime_ns)
        with self.assertRaises(FileExistsError):
            archive.restore(self.packed, target)
        with self.assertRaises(FileExistsError):
            archive.pack(self.weights, self.packed)

    def test_source_safety(self):
        with self.assertRaises(ValueError):
            archive.pack(self.source, self.source / "archive.tar.gz", True)
        alias = self.root / "alias"
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            archive.pack(self.source, alias / "archive.tar.gz", True)
        (self.source / "link").symlink_to(self.weights)
        with self.assertRaises(ValueError):
            archive.pack(self.source, self.packed, True)
        self.assertEqual(self.weights.read_bytes(), self.payload)

    def test_verification_failure_never_removes_source(self):
        with mock.patch.object(archive, "inspect_archive", side_effect=ValueError("corrupt")):
            with self.assertRaises(ValueError):
                archive.pack(self.source, self.packed, True)
        self.assertEqual(self.weights.read_bytes(), self.payload)
        self.assertFalse(self.packed.exists())
        self.assertFalse(list(self.root.glob(".model-archive-*")))

    def test_corruption_and_insufficient_space_leave_no_destination(self):
        archive.pack(self.source, self.packed)
        destination = self.root / "restored"
        with mock.patch.object(archive.shutil, "disk_usage", return_value=mock.Mock(free=0)):
            with self.assertRaises(OSError):
                archive.restore(self.packed, destination)
        self.assertFalse(destination.exists())
        with self.packed.open("r+b") as stream:
            stream.seek(-8, 2)
            stream.write(b"XXXXXXXX")
        with self.assertRaises((OSError, ValueError, tarfile.TarError, EOFError)):
            archive.restore(self.packed, destination)
        self.assertFalse(destination.exists())

    def test_unsafe_manifest_paths_rejected(self):
        for name in ("../escape", "/absolute", ".", "C:/escape", "a\\b"):
            with self.subTest(name=name):
                manifest = {"format": "tensormark.model-archive/1", "kind": "directory",
                            "directories": [], "source_mtime_ns": 0,
                            "files": [{"path": name, "size": 1, "mode": 0o600,
                                       "mtime_ns": 0, "sha256": hashlib.sha256(b"x").hexdigest()}]}
                with tarfile.open(self.packed, "w:gz") as tar:
                    body = json.dumps(manifest).encode()
                    member = tarfile.TarInfo("manifest.json")
                    member.size = len(body)
                    tar.addfile(member, io.BytesIO(body))
                with self.assertRaises(ValueError):
                    archive.restore(self.packed, self.root / "destination")
                self.assertFalse((self.root / "destination").exists())


if __name__ == "__main__":
    unittest.main()
