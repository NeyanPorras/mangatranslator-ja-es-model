from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from verify_release import verify  # noqa: E402

TAG = "v0.1.0-qa"
ASSET = "mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.0-qa.zip"
RELEASE_MANIFEST = "mangatranslator-ja-es-model-v0.1.0-qa.manifest.json"
PAYLOAD = Path("/tmp/manga-model-release")
LAB = Path("/tmp/manga-model-lab")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseTests(unittest.TestCase):
    def test_release_payload_verifies(self) -> None:
        result = verify(PAYLOAD / ASSET, PAYLOAD / RELEASE_MANIFEST)
        self.assertEqual("verified", result["status"])
        self.assertEqual(144594126, result["runtime_files_bytes"])

    def test_rebuild_is_byte_identical(self) -> None:
        if not LAB.is_dir():
            self.skipTest("model lab is not available")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            manifest = Path(directory) / "manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "build_release.py"),
                    "--lab-root",
                    str(LAB),
                    "--output-dir",
                    str(output),
                    "--manifest-output",
                    str(manifest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(sha256(PAYLOAD / ASSET), sha256(output / ASSET))
            self.assertEqual(
                json.loads((PAYLOAD / RELEASE_MANIFEST).read_text(encoding="utf-8")),
                json.loads((output / RELEASE_MANIFEST).read_text(encoding="utf-8")),
            )
            self.assertEqual(
                json.loads((REPO_ROOT / "manifests" / f"{TAG}.json").read_text(encoding="utf-8")),
                json.loads(manifest.read_text(encoding="utf-8")),
            )

    def test_repository_contains_no_model_binary(self) -> None:
        blocked_suffixes = {".onnx", ".spm", ".bin", ".zip", ".tar", ".gz"}
        tracked_candidates = [
            path
            for path in REPO_ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
        ]
        blocked = [path for path in tracked_candidates if path.suffix.lower() in blocked_suffixes]
        self.assertEqual([], blocked)
        oversized = [path for path in tracked_candidates if path.stat().st_size > 1024 * 1024]
        self.assertEqual([], oversized)

    def test_provenance_matches_quantization_config(self) -> None:
        provenance = json.loads(
            (REPO_ROOT / "provenance" / "bundle-source.json").read_text(encoding="utf-8")
        )
        quantization = json.loads(
            (REPO_ROOT / "provenance" / "ort_config.json").read_text(encoding="utf-8")
        )["quantization"]
        self.assertFalse(quantization["is_static"])
        self.assertFalse(quantization["per_channel"])
        self.assertEqual("QOperator", quantization["format"])
        self.assertEqual("QUInt8", quantization["activations_dtype"])
        self.assertEqual("QInt8", quantization["weights_dtype"])
        self.assertEqual(
            provenance["runtime_files_bytes"],
            sum(item["bytes"] for item in provenance["runtime_files"]),
        )
        self.assertEqual(
            provenance["quantization"]["ort_config_sha256"],
            sha256(REPO_ROOT / "provenance" / "ort_config.json"),
        )

    def test_bundle_descriptor_matches_benchmark_evidence(self) -> None:
        benchmark_path = LAB / "quant-benchmark" / "int8-pertensor-qoperator.json"
        if not benchmark_path.is_file():
            self.skipTest("model-lab benchmark evidence is not available")
        benchmark_files = json.loads(benchmark_path.read_text(encoding="utf-8"))["bundle"]["files"]
        provenance = json.loads(
            (REPO_ROOT / "provenance" / "bundle-source.json").read_text(encoding="utf-8")
        )
        actual = [
            {
                "source": Path(item["path"]).relative_to(LAB).as_posix(),
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in benchmark_files
        ]
        declared = [
            {key: item[key] for key in ("source", "bytes", "sha256")}
            for item in provenance["runtime_files"]
        ]
        self.assertEqual(actual, declared)


if __name__ == "__main__":
    unittest.main()
