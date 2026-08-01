from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from verify_release import verify  # noqa: E402

TAG = "v0.1.1-qa"
ASSET = "mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.1-qa.zip"
RELEASE_MANIFEST = f"mangatranslator-ja-es-model-{TAG}.manifest.json"
PAYLOAD = Path(os.environ.get("MANGA_MODEL_V011_PAYLOAD_DIR", "/tmp/manga-model-release-v011"))
BASE_ARCHIVE = Path(
    os.environ.get(
        "MANGA_MODEL_V010_ARCHIVE",
        "/tmp/manga-tokenizer-lab/download/mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.0-qa.zip",
    )
)
GRAPH_DIR = Path(
    os.environ.get("MANGA_MODEL_TOKENIZER_GRAPH_DIR", "/tmp/manga-tokenizer-lab/evidence/ortx-graphs")
)
BASE_RUNTIME = Path(
    os.environ.get(
        "MANGA_MODEL_V010_RUNTIME",
        "/tmp/manga-tokenizer-lab/extracted/mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.0-qa/runtime",
    )
)
REQUIRE_FIXTURES = os.environ.get("CI", "").lower() == "true" or os.environ.get(
    "MANGA_MODEL_REQUIRE_FIXTURES"
) == "1"
EXPECTED_ARCHIVE_SHA256 = "7074a65066466c95187d93319588659854746734dcf76e957d5c5e11805472da"
EXPECTED_MANIFEST_SHA256 = "189da945d9c8f97cf614d5ca71dac3dc5f59e44602d4adaeac3cb1df79918683"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReleaseTests(unittest.TestCase):
    def require_file(self, path: Path, description: str) -> None:
        if path.is_file():
            return
        message = f"{description} is unavailable: {path}"
        if REQUIRE_FIXTURES:
            self.fail(message)
        self.skipTest(message)

    def require_directory(self, path: Path, description: str) -> None:
        if path.is_dir():
            return
        message = f"{description} is unavailable: {path}"
        if REQUIRE_FIXTURES:
            self.fail(message)
        self.skipTest(message)

    def require_dependencies(self, names: tuple[str, ...]) -> None:
        missing = [name for name in names if importlib.util.find_spec(name) is None]
        if not missing:
            return
        message = f"required Python dependencies are unavailable: {', '.join(missing)}"
        if REQUIRE_FIXTURES:
            self.fail(message)
        self.skipTest(message)

    def test_release_payload_verifies(self) -> None:
        self.require_file(PAYLOAD / ASSET, "v0.1.1 release archive")
        self.require_file(PAYLOAD / RELEASE_MANIFEST, "v0.1.1 release manifest")
        result = verify(PAYLOAD / ASSET, PAYLOAD / RELEASE_MANIFEST)
        self.assertEqual("verified", result["status"])
        self.assertEqual(13, result["files"])
        self.assertEqual(146964244, result["runtime_files_bytes"])

    def test_rebuild_is_byte_identical(self) -> None:
        self.require_file(BASE_ARCHIVE, "verified v0.1.0 base archive")
        self.require_directory(GRAPH_DIR, "generated tokenizer graph directory")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            manifest = Path(directory) / "manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "build_release_v011.py"),
                    "--base-archive",
                    str(BASE_ARCHIVE),
                    "--graph-dir",
                    str(GRAPH_DIR),
                    "--output-dir",
                    str(output),
                    "--manifest-output",
                    str(manifest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(EXPECTED_ARCHIVE_SHA256, sha256(output / ASSET))
            self.assertEqual(EXPECTED_MANIFEST_SHA256, sha256(output / RELEASE_MANIFEST))
            self.assertEqual((REPO_ROOT / "manifests" / f"{TAG}.json").read_bytes(), manifest.read_bytes())

    def test_tokenizer_generator_is_byte_identical_and_emits_no_arrays(self) -> None:
        dependencies = ("numpy", "onnx", "sentencepiece")
        self.require_dependencies(dependencies)
        self.require_directory(BASE_RUNTIME, "extracted v0.1.0 runtime")
        provenance = json.loads(
            (REPO_ROOT / "provenance" / "bundle-source-v0.1.1-qa.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "build_ortx_marian_tokenizers.py"),
                    "--runtime-dir",
                    str(BASE_RUNTIME),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                sorted(item["source_name"] for item in provenance["tokenizer_graphs"]),
                sorted(path.name for path in output.iterdir()),
            )
            for expected in provenance["tokenizer_graphs"]:
                graph = output / expected["source_name"]
                self.assertEqual(expected["bytes"], graph.stat().st_size)
                self.assertEqual(expected["sha256"], sha256(graph))

    def test_v010_runtime_entries_are_unchanged_and_only_two_are_added(self) -> None:
        base = json.loads((REPO_ROOT / "manifests" / "v0.1.0-qa.json").read_text(encoding="utf-8"))
        candidate = json.loads((REPO_ROOT / "manifests" / f"{TAG}.json").read_text(encoding="utf-8"))
        old_root = base["asset"]["root_directory"]
        new_root = candidate["asset"]["root_directory"]
        old_runtime = {
            Path(item["path"]).relative_to(old_root).as_posix(): (item["bytes"], item["sha256"])
            for item in base["asset"]["files"]
            if item["role"] == "runtime"
        }
        new_runtime = {
            Path(item["path"]).relative_to(new_root).as_posix(): (item["bytes"], item["sha256"])
            for item in candidate["asset"]["files"]
            if item["role"] == "runtime"
        }
        self.assertEqual(old_runtime, {key: new_runtime[key] for key in old_runtime})
        self.assertEqual(
            {"runtime/marian_source_tokenizer.onnx", "runtime/marian_target_detokenizer.onnx"},
            set(new_runtime) - set(old_runtime),
        )

    def test_verifier_rejects_missing_extra_unsafe_and_hash_mismatch(self) -> None:
        self.require_file(PAYLOAD / ASSET, "v0.1.1 release archive")
        self.require_file(PAYLOAD / RELEASE_MANIFEST, "v0.1.1 release manifest")
        manifest = json.loads((PAYLOAD / RELEASE_MANIFEST).read_text(encoding="utf-8"))
        mutations = []

        missing = copy.deepcopy(manifest)
        missing["asset"]["files"].pop()
        mutations.append(missing)

        extra = copy.deepcopy(manifest)
        extra["asset"]["files"].append(
            {
                "path": f"{extra['asset']['root_directory']}/runtime/extra.bin",
                "role": "runtime",
                "bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        )
        mutations.append(extra)

        unsafe = copy.deepcopy(manifest)
        unsafe["asset"]["files"][0]["path"] = "../escape"
        mutations.append(unsafe)

        wrong_hash = copy.deepcopy(manifest)
        wrong_hash["asset"]["files"][0]["sha256"] = "0" * 64
        mutations.append(wrong_hash)

        with tempfile.TemporaryDirectory() as directory:
            for index, mutated in enumerate(mutations):
                path = Path(directory) / f"manifest-{index}.json"
                path.write_text(json.dumps(mutated), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(ValueError):
                    verify(PAYLOAD / ASSET, path)

    def test_repository_contains_no_release_binary(self) -> None:
        blocked_suffixes = {".onnx", ".spm", ".bin", ".zip", ".tar", ".gz", ".npy"}
        tracked_candidates = [
            path
            for path in REPO_ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
        ]
        self.assertEqual([], [path for path in tracked_candidates if path.suffix.lower() in blocked_suffixes])
        self.assertEqual([], [path for path in tracked_candidates if path.stat().st_size > 1024 * 1024])

    def test_provenance_pins_runtime_and_tokenizer_requirements(self) -> None:
        provenance = json.loads(
            (REPO_ROOT / "provenance" / "bundle-source-v0.1.1-qa.json").read_text(encoding="utf-8")
        )
        self.assertEqual("v0.1.1-qa", provenance["release_tag"])
        self.assertEqual("10da202dfe33204d3752ef7a6eb4721de2418f5a4e3ff1750d253174f6717760", provenance["base_release"]["archive_sha256"])
        self.assertEqual(
            {"onnxruntime_android": "1.21.1", "onnxruntime_extensions_android": "0.13.0"},
            provenance["runtime_requirements"],
        )
        self.assertEqual(2, len(provenance["tokenizer_graphs"]))
        self.assertEqual(sha256(SCRIPTS / "build_ortx_marian_tokenizers.py"), provenance["generator"]["sha256"])
        self.assertFalse(provenance["generator"]["intermediate_arrays_required_at_runtime"])

    def test_committed_manifest_has_expected_identity(self) -> None:
        self.assertEqual(EXPECTED_MANIFEST_SHA256, sha256(REPO_ROOT / "manifests" / f"{TAG}.json"))
        manifest = json.loads((REPO_ROOT / "manifests" / f"{TAG}.json").read_text(encoding="utf-8"))
        self.assertEqual(EXPECTED_ARCHIVE_SHA256, manifest["asset"]["sha256"])


if __name__ == "__main__":
    unittest.main()
