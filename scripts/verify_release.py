#!/usr/bin/env python3
"""Verify release identity, archive structure, and every contained file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

CHUNK_SIZE = 1024 * 1024
EXPECTED_REPOSITORY = "NeyanPorras/mangatranslator-ja-es-model"
EXPECTED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
V011_TAG = "v0.1.1-qa"
V011_RUNTIME_REQUIREMENTS = {
    "onnxruntime_android": "1.21.1",
    "onnxruntime_extensions_android": "0.13.0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_archive_path(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        bool(path)
        and not pure.is_absolute()
        and ".." not in pure.parts
        and "\\" not in path
        and not path.endswith("/")
    )


def verify(archive_path: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release = manifest["release"]
    asset = manifest["asset"]
    tag = release["tag"]

    if manifest["schema_version"] != 1:
        raise ValueError("unsupported manifest schema")
    if release["status"] != "experimental_qa_candidate" or release["production_approved"]:
        raise ValueError("manifest must remain an unapproved experimental QA candidate")

    release_prefix = f"https://github.com/{EXPECTED_REPOSITORY}/releases/download/{tag}/"
    manifest_asset_name = f"mangatranslator-ja-es-model-{tag}.manifest.json"
    expected_release_page = f"https://github.com/{EXPECTED_REPOSITORY}/releases/tag/{tag}"
    if release["release_page_url"] != expected_release_page:
        raise ValueError("release page URL is not the pinned public release URL")
    if asset["url"] != release_prefix + asset["name"]:
        raise ValueError("archive release URL is not the pinned public release URL")
    if release["manifest_url"] != release_prefix + manifest_asset_name:
        raise ValueError("manifest release URL is not the pinned public release URL")

    if archive_path.name != asset["name"]:
        raise ValueError("archive filename does not match the manifest")
    if archive_path.stat().st_size != asset["bytes"]:
        raise ValueError("archive byte size does not match the manifest")
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != asset["sha256"]:
        raise ValueError("archive SHA-256 does not match the manifest")

    expected_files = {item["path"]: item for item in asset["files"]}
    if len(expected_files) != len(asset["files"]):
        raise ValueError("manifest contains duplicate archive paths")
    if any(not safe_archive_path(path) for path in expected_files):
        raise ValueError("manifest contains an unsafe archive path")
    root = asset["root_directory"]
    if not safe_archive_path(root) or any(
        PurePosixPath(path).parts[0] != root for path in expected_files
    ):
        raise ValueError("manifest entries do not share the declared root directory")

    runtime_bytes = 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != sorted(expected_files):
            raise ValueError("archive paths are missing, extra, duplicated, or out of order")
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate paths")

        for info in infos:
            expected = expected_files[info.filename]
            if not safe_archive_path(info.filename):
                raise ValueError(f"unsafe archive path: {info.filename}")
            if info.date_time != EXPECTED_ZIP_TIME:
                raise ValueError(f"non-deterministic timestamp: {info.filename}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"unexpected compression: {info.filename}")
            if info.file_size != expected["bytes"]:
                raise ValueError(f"size mismatch: {info.filename}")

            digest = hashlib.sha256()
            with archive.open(info, "r") as handle:
                for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {info.filename}")
            if expected["role"] == "runtime":
                runtime_bytes += info.file_size

    if runtime_bytes != asset["runtime_files_bytes"]:
        raise ValueError("runtime file total does not match the manifest")

    if tag == V011_TAG:
        requirements = manifest["model"]["runtime_requirements"]
        if requirements != V011_RUNTIME_REQUIREMENTS:
            raise ValueError("v0.1.1 runtime requirements are not pinned")
        tokenizers = manifest["model"]["tokenizer_graphs"]
        if len(tokenizers) != 2:
            raise ValueError("v0.1.1 must declare exactly two tokenizer graphs")
        for graph in tokenizers:
            declared = expected_files.get(graph["path"])
            if not declared or declared["role"] != "runtime":
                raise ValueError("tokenizer graph is not a declared runtime file")
            if graph["bytes"] != declared["bytes"] or graph["sha256"] != declared["sha256"]:
                raise ValueError("tokenizer graph identity differs from the archive entry")

    return {
        "archive": str(archive_path),
        "bytes": asset["bytes"],
        "sha256": asset["sha256"],
        "files": len(expected_files),
        "runtime_files_bytes": runtime_bytes,
        "status": "verified",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify(args.archive.resolve(), args.manifest.resolve())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
