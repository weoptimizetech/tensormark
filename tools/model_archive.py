#!/usr/bin/env python3
"""Lossless, checksum-verified cold storage for model files and package directories.

  python tools/model_archive.py pack model.mlpackage model.mlpackage.tar.gz
  python tools/model_archive.py pack model.mlpackage model.mlpackage.tar.gz --remove-source
  python tools/model_archive.py verify model.mlpackage.tar.gz
  python tools/model_archive.py restore model.mlpackage.tar.gz model.mlpackage

Pack preserves the source by default. --remove-source is for INACTIVE assets only;
it removes the source only after verifying every archived byte and checking that
source metadata has not changed. Restore refuses existing destinations. No model
conversion, quantization, downloads, or third-party Python packages are involved.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

CHUNK = 1024 * 1024
MANIFEST_LIMIT = 8 * 1024 * 1024


def digest(stream):
    checksum = hashlib.sha256()
    size = 0
    while data := stream.read(CHUNK):
        checksum.update(data)
        size += len(data)
    return size, checksum.hexdigest()


def relative_name(name):
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name
            or any(":" in part for part in path.parts) or str(path) != name):
        raise ValueError(f"unsafe archive path: {name!r}")
    return name


def inspect_archive(archive, destination=None):
    """Verify all members; optionally write to a private staging directory."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        by_name = {member.name: member for member in members}
        if len(by_name) != len(members):
            raise ValueError("duplicate archive members")
        meta = by_name.get("manifest.json")
        if not meta or not meta.isfile() or meta.size > MANIFEST_LIMIT:
            raise ValueError("missing or oversized archive manifest")
        manifest = json.load(tar.extractfile(meta))
        if manifest.get("format") != "tensormark.model-archive/1":
            raise ValueError("unsupported model archive format")
        if manifest.get("kind") not in ("file", "directory"):
            raise ValueError("invalid source kind")
        expected = {"manifest.json"}
        directories = set()
        for name in manifest["directories"]:
            relative_name(name)
            if name in directories:
                raise ValueError("duplicate directory")
            directories.add(name)
        total = 0
        for entry in manifest["files"]:
            name = relative_name(entry["path"])
            member_name = "payload/" + name
            if member_name in expected or name in directories:
                raise ValueError("duplicate/conflicting file path")
            expected.add(member_name)
            member = by_name.get(member_name)
            if not member or not member.isfile() or member.size != entry["size"]:
                raise ValueError(f"missing or invalid member: {name}")
            source = tar.extractfile(member)
            checksum = hashlib.sha256()
            size = 0
            target = None
            if destination is not None:
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                target = path.open("xb")
            try:
                while data := source.read(CHUNK):
                    checksum.update(data)
                    size += len(data)
                    if target:
                        target.write(data)
            finally:
                source.close()
                if target:
                    target.close()
            if size != entry["size"] or checksum.hexdigest() != entry["sha256"]:
                raise ValueError(f"checksum mismatch: {name}")
            if destination is not None:
                # Restore ordinary permissions only, never setuid/setgid bits.
                os.chmod(destination / name, entry["mode"] & 0o777)
                timestamp = entry["mtime_ns"]
                os.utime(destination / name, ns=(timestamp, timestamp))
            total += size
        if expected != set(by_name):
            raise ValueError("unlisted archive members")
        # tarfile stops at the tar end marker, before gzip necessarily checks
        # its CRC/footer. Consume the padded tail through gzip EOF as well.
        payload_end = max(member.offset_data + ((member.size + 511) // 512) * 512
                          for member in members)
        tar.fileobj.seek(payload_end)
        while padding := tar.fileobj.read(CHUNK):
            if padding.strip(b"\0"):
                raise ValueError("unexpected trailing archive data")
        if manifest["kind"] == "file" and (len(manifest["files"]) != 1 or directories):
            raise ValueError("invalid single-file archive")
        if destination is not None:
            for name in sorted(directories):
                (destination / name).mkdir(parents=True, exist_ok=True)
        return manifest, total


def pack(source, archive, remove_source=False):
    source, archive = Path(source).absolute(), Path(archive).absolute()
    if source.is_symlink() or not (source.is_file() or source.is_dir()):
        raise ValueError("source must be a regular file or directory, not a symlink")
    if archive.exists() or archive.is_symlink():
        raise FileExistsError(archive)
    source, archive = source.resolve(), archive.resolve()
    if archive == source or source in archive.parents:
        raise ValueError("archive must be outside the source")
    paths = [source] if source.is_file() else sorted(source.rglob("*"))
    manifest = {"format": "tensormark.model-archive/1",
                "kind": "file" if source.is_file() else "directory", "files": [], "directories": [],
                "source_mtime_ns": source.stat().st_mtime_ns}
    snapshots = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"symlink source not supported: {path}")
        name = relative_name(path.name if source.is_file() else path.relative_to(source).as_posix())
        if path.is_dir():
            manifest["directories"].append(name)
            continue
        if not path.is_file():
            raise ValueError(f"non-regular source: {path}")
        stat = path.stat()
        with path.open("rb") as stream:
            size, checksum = digest(stream)
        snapshots[path] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        manifest["files"].append({"path": name, "size": size, "sha256": checksum,
                                  "mode": stat.st_mode & 0o777, "mtime_ns": stat.st_mtime_ns})
    archive.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".model-archive-", dir=archive.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=6) as tar:
            body = json.dumps(manifest, sort_keys=True).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
            for entry in manifest["files"]:
                path = source if source.is_file() else source / entry["path"]
                tar.add(path, arcname="payload/" + entry["path"], recursive=False)
        _, total = inspect_archive(temporary)
        for path, before in snapshots.items():
            stat = path.stat()
            if before != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
                raise RuntimeError("source changed while archiving; original retained")
        current_paths = [source] if source.is_file() else sorted(source.rglob("*"))
        if current_paths != paths:
            raise RuntimeError("source tree changed while archiving; original retained")
        with temporary.open("rb") as persisted:
            os.fsync(persisted.fileno())
        # Publish without overwriting a concurrently created archive.
        os.link(temporary, archive)
        if os.name == "posix":
            directory_fd = os.open(archive.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        size = archive.stat().st_size
        removed = False
        if remove_source and size < total:
            if source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
            removed = True
        return {"archive": str(archive), "source_bytes": total, "archive_bytes": size,
                "saved_bytes": total - size, "source_removed": removed, "verified": True}
    finally:
        temporary.unlink(missing_ok=True)


def restore(archive, destination):
    archive, destination = Path(archive), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    # Verify before allocating an expanded copy, including all checksums.
    manifest, total = inspect_archive(archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < total + 64 * 1024 * 1024:
        raise OSError("insufficient space to restore model; archive retained")
    with tempfile.TemporaryDirectory(prefix=".model-restore-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        inspect_archive(archive, stage)
        expanded = stage if manifest["kind"] == "directory" else stage / manifest["files"][0]["path"]
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        expanded.rename(destination)
        timestamp = manifest["source_mtime_ns"]
        os.utime(destination, ns=(timestamp, timestamp))
    return {"destination": str(destination), "restored_bytes": total, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("pack")
    create.add_argument("source")
    create.add_argument("archive")
    create.add_argument("--remove-source", action="store_true", help="only for inactive assets; verify first")
    verify = commands.add_parser("verify")
    verify.add_argument("archive")
    expand = commands.add_parser("restore")
    expand.add_argument("archive")
    expand.add_argument("destination")
    args = parser.parse_args()
    if args.action == "pack":
        result = pack(args.source, args.archive, args.remove_source)
    elif args.action == "restore":
        result = restore(args.archive, args.destination)
    else:
        _, total = inspect_archive(args.archive)
        result = {"verified": True, "source_bytes": total}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
