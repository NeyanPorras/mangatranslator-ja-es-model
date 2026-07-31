#!/usr/bin/env python3
"""Build the deterministic experimental model release from pinned local inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

TAG = "v0.1.0-qa"
ASSET_NAME = "mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.0-qa.zip"
MANIFEST_ASSET_NAME = "mangatranslator-ja-es-model-v0.1.0-qa.manifest.json"
ARCHIVE_ROOT = "mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.0-qa"
RELEASE_BASE = (
    "https://github.com/NeyanPorras/mangatranslator-ja-es-model/releases/download/"
    f"{TAG}"
)
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_source(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing source file: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"size mismatch for {path}: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def write_stored_file(archive: zipfile.ZipFile, source: Path, archive_path: str) -> None:
    info = zipfile.ZipInfo(archive_path, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with source.open("rb") as source_handle, archive.open(info, "w") as target_handle:
        shutil.copyfileobj(source_handle, target_handle, CHUNK_SIZE)


def build(repo_root: Path, lab_root: Path, output_dir: Path, manifest_output: Path) -> tuple[Path, Path]:
    provenance_path = repo_root / "provenance" / "bundle-source.json"
    benchmark_path = repo_root / "benchmarks" / "quantization-preservation.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    ort_config_path = repo_root / provenance["quantization"]["ort_config_path"]
    checked_source(
        ort_config_path,
        provenance["quantization"]["ort_config_bytes"],
        provenance["quantization"]["ort_config_sha256"],
    )

    if provenance["release_tag"] != TAG:
        raise ValueError("provenance release tag does not match the builder")

    entries: list[dict[str, object]] = []
    runtime_bytes = 0
    for descriptor in provenance["runtime_files"]:
        source = lab_root / descriptor["source"]
        checked_source(source, descriptor["bytes"], descriptor["sha256"])
        runtime_bytes += descriptor["bytes"]
        entries.append(
            {
                "source": source,
                "archive_path": f"{ARCHIVE_ROOT}/runtime/{descriptor['archive_name']}",
                "role": "runtime",
                "bytes": descriptor["bytes"],
                "sha256": descriptor["sha256"],
            }
        )

    if runtime_bytes != provenance["runtime_files_bytes"]:
        raise ValueError("runtime byte total does not match provenance")

    document_roles = {
        "LICENSE": "license",
        "NOTICE": "notice",
        "MODEL_CARD.md": "model_card",
    }
    for filename, role in document_roles.items():
        source = repo_root / filename
        entries.append(
            {
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
    temporary_archive = archive_path.with_suffix(".zip.tmp")
    if temporary_archive.exists():
        temporary_archive.unlink()

    with zipfile.ZipFile(temporary_archive, "w", allowZip64=True) as archive:
        for entry in entries:
            write_stored_file(
                archive,
                Path(entry["source"]),
                str(entry["archive_path"]),
            )
    temporary_archive.replace(archive_path)

    asset_bytes = archive_path.stat().st_size
    asset_sha256 = sha256_file(archive_path)
    generation_config = json.loads(
        (lab_root / "source-model" / "generation_config.json").read_text(encoding="utf-8")
    )

    manifest = {
        "schema_version": 1,
        "release": {
            "tag": TAG,
            "status": "experimental_qa_candidate",
            "production_approved": False,
            "release_page_url": (
                "https://github.com/NeyanPorras/mangatranslator-ja-es-model/releases/tag/"
                f"{TAG}"
            ),
            "manifest_url": f"{RELEASE_BASE}/{MANIFEST_ASSET_NAME}",
        },
        "model": {
            "source_repository": provenance["source_model"]["repository"],
            "source_revision": provenance["source_model"]["revision"],
            "source_revision_url": provenance["source_model"]["revision_url"],
            "source_language": "ja",
            "target_language": "es",
            "architecture": "Marian",
            "declared_license": provenance["source_model"]["declared_license"],
            "quantization": provenance["quantization"],
            "generation": {
                "num_beams": generation_config["num_beams"],
                "max_length": generation_config["max_length"],
                "decoder_start_token_id": generation_config["decoder_start_token_id"],
                "pad_token_id": generation_config["pad_token_id"],
                "eos_token_id": generation_config["eos_token_id"],
                "renormalize_logits": generation_config["renormalize_logits"],
            },
        },
        "asset": {
            "name": ASSET_NAME,
            "url": f"{RELEASE_BASE}/{ASSET_NAME}",
            "archive_format": "zip",
            "compression": "stored",
            "deterministic_timestamp": "1980-01-01T00:00:00Z",
            "root_directory": ARCHIVE_ROOT,
            "bytes": asset_bytes,
            "sha256": asset_sha256,
            "runtime_files_bytes": runtime_bytes,
            "files": [
                {
                    "path": entry["archive_path"],
                    "role": entry["role"],
                    "bytes": entry["bytes"],
                    "sha256": entry["sha256"],
                }
                for entry in entries
            ],
        },
        "quality": {
            "status": benchmark["status"],
            "baseline": benchmark["baseline"],
            "corpus": benchmark["corpus"],
            "metrics": benchmark["metrics"],
            "known_regressions": benchmark["known_regressions"],
            "required_gates": benchmark["required_gates"],
        },
    }

    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(manifest_text, encoding="utf-8")
    release_manifest_path = output_dir / MANIFEST_ASSET_NAME
    release_manifest_path.write_text(manifest_text, encoding="utf-8")

    print(f"archive={archive_path}")
    print(f"archive_bytes={asset_bytes}")
    print(f"archive_sha256={asset_sha256}")
    print(f"manifest={manifest_output}")
    print(f"release_manifest={release_manifest_path}")
    return archive_path, release_manifest_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-root", type=Path, default=Path("/tmp/manga-model-lab"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/manga-model-release"))
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=repo_root / "manifests" / f"{TAG}.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        build(repo_root, args.lab_root.resolve(), args.output_dir.resolve(), args.manifest_output.resolve())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"release build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
