# Experimental Japanese-to-Spanish model for MangaTranslator

This repository distributes a pinned, dynamically quantized Marian ONNX model for **fully local** Japanese-to-Spanish translation. Release `v0.1.1-qa` adds executable tokenizer graphs and remains an **experimental QA candidate, not production-approved**.

## Download and verify

1. Download [`mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.1-qa.zip`](https://github.com/NeyanPorras/mangatranslator-ja-es-model/releases/download/v0.1.1-qa/mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.1-qa.zip).
2. Download the [release manifest](https://github.com/NeyanPorras/mangatranslator-ja-es-model/releases/download/v0.1.1-qa/mangatranslator-ja-es-model-v0.1.1-qa.manifest.json).
3. Verify before extraction:

```bash
python3 scripts/verify_release.py \
  --archive mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.1-qa.zip \
  --manifest mangatranslator-ja-es-model-v0.1.1-qa.manifest.json
```

Do not load an archive that fails verification.

## What is included

| Component | Decision |
|---|---|
| Source | `Helsinki-NLP/opus-mt-ja-es` pinned to `d1693e4d2bc285d02653ff6438fe21e2b3c6025e` |
| Runtime | ONNX encoder plus merged decoder, opset 17 |
| Quantization | Dynamic QOperator/IntegerOps, QUInt8 activations, QInt8 weights, per-tensor |
| Tokenizer | Source and target ONNX graphs with embedded SentencePiece models and Marian ID mappings |
| Android runtime | ONNX Runtime Android 1.21.1 + ONNX Runtime Extensions Android 0.13.0 |
| Generation | Pinned six-beam `generation_config.json`; omitting it changes output |
| Distribution | One deterministic, uncompressed ZIP with runtime files and legal notices |

The model archive is downloaded once. Translation remains on-device; this repository does not provide or require a translation API.

## QA status

The candidate preserved 16/32 FP32 outputs exactly on a synthetic preservation corpus, with mean token similarity 0.8617 and no omission-heuristic flags. It also introduced a known repetition regression: `まさか…` changed from the FP32 baseline `No puede ser.` to `No, no, no.`

Tokenizer parity is exact for 6/6 source cases and 4/4 target cases using real host ONNX graph execution. These checks **do not prove translation quality or Android readiness**. Human semantic review and API 29/36 device profiling are required before production use. See [MODEL_CARD.md](MODEL_CARD.md), [the quantization benchmark](benchmarks/quantization-preservation.md), and [tokenizer parity](benchmarks/tokenizer-golden-parity-v0.1.1-qa.md).

## Rebuild the release

The build verifies every source byte against pinned size and SHA-256 before creating the archive.

```bash
python3 scripts/build_ortx_marian_tokenizers.py \
  --runtime-dir /path/to/v0.1.0/runtime \
  --output-dir /tmp/tokenizer-graphs
python3 scripts/build_release_v011.py \
  --base-archive /path/to/mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.0-qa.zip \
  --graph-dir /tmp/tokenizer-graphs \
  --output-dir /tmp/manga-model-release-v011
python3 -m unittest discover -s tests -v
```

The builder verifies the complete base release and both graph identities before producing a deterministic archive. Model binaries, generated graphs, and release archives are intentionally excluded from Git.

### Automated validation

`.github/workflows/validate.yml` is the authoritative clean-run recipe. It downloads and verifies the pinned `v0.1.0-qa` archive, regenerates both tokenizer graphs, rebuilds `v0.1.1-qa`, and checks the exact archive and manifest hashes.

For a strict local test run, set `MANGA_MODEL_REQUIRE_FIXTURES=1` and provide these paths:

- `MANGA_MODEL_V010_ARCHIVE`
- `MANGA_MODEL_V010_RUNTIME`
- `MANGA_MODEL_TOKENIZER_GRAPH_DIR`
- `MANGA_MODEL_V011_PAYLOAD_DIR`

Without strict mode, fixture-dependent tests may skip so metadata-only contributors can still run the remaining unit tests.

## License

The upstream model card declares Apache-2.0. This repository, quantized graphs, and release metadata are distributed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
