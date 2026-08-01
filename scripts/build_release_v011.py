#!/usr/bin/env python3
"""Build v0.1.1 from the verified v0.1.0 archive and pinned tokenizer graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

from verify_release import sha256_file, verify

TAG = "v0.1.1-qa"
ASSET_NAME = "mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.1-qa.zip"
MANIFEST_ASSET_NAME = f"mangatranslator-ja-es-model-{TAG}.manifest.json"
ARCHIVE_ROOT = "mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.1-qa"
RELEASE_BASE = (
    "https://github.com/NeyanPorras/mangatranslator-ja-es-model/releases/download/"
    f"{TAG}"
)
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
CHUNK_SIZE = 1024 * 1024


def checked_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing source file: {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"size mismatch for {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}")


def zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def copy_stream(source, target) -> None:
    shutil.copyfileobj(source, target, CHUNK_SIZE)


def build(
    repo_root: Path,
    base_archive: Path,
    graph_dir: Path,
    output_dir: Path,
    manifest_output: Path,
) -> tuple[Path, Path]:
    provenance = json.loads(
        (repo_root / "provenance" / "bundle-source-v0.1.1-qa.json").read_text(encoding="utf-8")
    )
    base_manifest_path = repo_root / provenance["base_release"]["manifest_path"]
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    benchmark = json.loads(
        (repo_root / "benchmarks" / "quantization-preservation.json").read_text(encoding="utf-8")
    )
    parity = json.loads(
        (repo_root / provenance["parity_evidence"]["path"]).read_text(encoding="utf-8")
    )

    if provenance["release_tag"] != TAG:
        raise ValueError("provenance release tag does not match the builder")
    checked_file(
        base_archive,
        provenance["base_release"]["archive_bytes"],
        provenance["base_release"]["archive_sha256"],
    )
    verify(base_archive, base_manifest_path)

    base_root = provenance["base_release"]["root_directory"]
    base_runtime = [item for item in base_manifest["asset"]["files"] if item["role"] == "runtime"]
    entries: list[dict[str, object]] = []
    for item in base_runtime:
        relative_path = Path(item["path"]).relative_to(base_root).as_posix()
        entries.append(
            {
                "kind": "base_archive",
                "source_path": item["path"],
                "archive_path": f"{ARCHIVE_ROOT}/{relative_path}",
                "role": "runtime",
                "component": "model_runtime",
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
        )

    tokenizer_manifest_entries: list[dict[str, object]] = []
    for descriptor in provenance["tokenizer_graphs"]:
        source = graph_dir / descriptor["source_name"]
        checked_file(source, descriptor["bytes"], descriptor["sha256"])
        entry = {
            "kind": "file",
            "source": source,
            "archive_path": f"{ARCHIVE_ROOT}/runtime/{descriptor['archive_name']}",
            "role": "runtime",
            "component": descriptor["role"],
            "bytes": descriptor["bytes"],
            "sha256": descriptor["sha256"],
        }
        entries.append(entry)
        tokenizer_manifest_entries.append(
            {
                "path": entry["archive_path"],
                "role": descriptor["role"],
                "bytes": descriptor["bytes"],
                "sha256": descriptor["sha256"],
            }
        )

    for filename, role in {
        "LICENSE": "license",
        "NOTICE": "notice",
        "MODEL_CARD.md": "model_card",
    }.items():
        source = repo_root / filename
        entries.append(
            {
                "kind": "file",
                "source": source,
                "archive_path": f"{ARCHIVE_ROOT}/{filename}",
                "role": role,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )

    entries.sort(key=lambda item: str(item["archive_path"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ASSET_NAME
    temporary_archive = output_dir / f".{ASSET_NAME}.tmp"
    if temporary_archive.exists():
        temporary_archive.unlink()

    with zipfile.ZipFile(base_archive, "r") as base_zip, zipfile.ZipFile(
        temporary_archive, "w", allowZip64=True
    ) as output_zip:
        for entry in entries:
            with output_zip.open(zip_info(str(entry["archive_path"])), "w") as target:
                if entry["kind"] == "base_archive":
                    with base_zip.open(str(entry["source_path"]), "r") as source:
                        copy_stream(source, target)
                else:
                    with Path(entry["source"]).open("rb") as source:
                        copy_stream(source, target)
    temporary_archive.replace(archive_path)

    runtime_bytes = sum(int(entry["bytes"]) for entry in entries if entry["role"] == "runtime")
    manifest_files = []
    for entry in entries:
        item = {
            "path": entry["archive_path"],
            "role": entry["role"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        if "component" in entry:
            item["component"] = entry["component"]
        manifest_files.append(item)

    manifest = {
        "schema_version": 1,
        "release": {
            "tag": TAG,
            "status": "experimental_qa_candidate",
            "production_approved": False,
            "release_page_url": (
                "https://github.com/NeyanPorras/mangatranslator-ja-es-model/releases/tag/" + TAG
            ),
            "manifest_url": f"{RELEASE_BASE}/{MANIFEST_ASSET_NAME}",
        },
        "model": {
            "source_repository": provenance["source_model"]["repository"],
            "source_revision": provenance["source_model"]["revision"],
            "source_revision_url": provenance["source_model"]["revision_url"],
            "source_language": provenance["source_model"]["source_language"],
            "target_language": provenance["source_model"]["target_language"],
            "architecture": "Marian",
            "declared_license": provenance["source_model"]["declared_license"],
            "quantization": base_manifest["model"]["quantization"],
            "generation": base_manifest["model"]["generation"],
            "runtime_requirements": provenance["runtime_requirements"],
            "tokenizer_graphs": tokenizer_manifest_entries,
        },
        "asset": {
            "name": ASSET_NAME,
            "url": f"{RELEASE_BASE}/{ASSET_NAME}",
            "archive_format": "zip",
            "compression": "stored",
            "deterministic_timestamp": "1980-01-01T00:00:00Z",
            "root_directory": ARCHIVE_ROOT,
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "runtime_files_bytes": runtime_bytes,
            "files": manifest_files,
        },
        "quality": {
            "status": benchmark["status"],
            "baseline": benchmark["baseline"],
            "corpus": benchmark["corpus"],
            "metrics": benchmark["metrics"],
            "known_regressions": benchmark["known_regressions"],
            "tokenizer_parity": parity,
            "required_gates": benchmark["required_gates"],
        },
        "derivation": {
            "base_release_tag": provenance["base_release"]["tag"],
            "base_archive_url": provenance["base_release"]["archive_url"],
            "base_archive_sha256": provenance["base_release"]["archive_sha256"],
            "base_runtime_entries_preserved": len(base_runtime),
            "added_runtime_entries": 2,
        },
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(manifest_text, encoding="utf-8")
    release_manifest_path = output_dir / MANIFEST_ASSET_NAME
    release_manifest_path.write_text(manifest_text, encoding="utf-8")

    print(f"archive={archive_path}")
    print(f"archive_bytes={archive_path.stat().st_size}")
    print(f"archive_sha256={sha256_file(archive_path)}")
    print(f"manifest={manifest_output}")
    print(f"release_manifest={release_manifest_path}")
    return archive_path, release_manifest_path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=repo_root / "manifests" / f"{TAG}.json",
    )
    args = parser.parse_args()
    try:
        build(
            repo_root,
            args.base_archive.resolve(),
            args.graph_dir.resolve(),
            args.output_dir.resolve(),
            args.manifest_output.resolve(),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        print(f"release build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
