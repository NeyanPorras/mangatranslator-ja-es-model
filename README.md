# Experimental Japanese-to-Spanish model for MangaTranslator

This repository distributes a pinned, dynamically quantized Marian ONNX model for **fully local** Japanese-to-Spanish translation. Release `v0.1.0-qa` is an **experimental QA candidate, not production-approved**.

## Download and verify

1. Download [`mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.0-qa.zip`](https://github.com/NeyanPorras/mangatranslator-ja-es-model/releases/download/v0.1.0-qa/mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.0-qa.zip).
2. Download the [release manifest](https://github.com/NeyanPorras/mangatranslator-ja-es-model/releases/download/v0.1.0-qa/mangatranslator-ja-es-model-v0.1.0-qa.manifest.json).
3. Verify before extraction:

```bash
python3 scripts/verify_release.py \
  --archive mangatranslator-ja-es-int8-pertensor-qoperator-v0.1.0-qa.zip \
  --manifest mangatranslator-ja-es-model-v0.1.0-qa.manifest.json
```

Do not load an archive that fails verification.

## What is included

| Component | Decision |
|---|---|
| Source | `Helsinki-NLP/opus-mt-ja-es` pinned to `d1693e4d2bc285d02653ff6438fe21e2b3c6025e` |
| Runtime | ONNX encoder plus merged decoder, opset 17 |
| Quantization | Dynamic QOperator/IntegerOps, QUInt8 activations, QInt8 weights, per-tensor |
| Tokenizer | Pinned source and target SentencePiece models plus vocabulary |
| Generation | Pinned six-beam `generation_config.json`; omitting it changes output |
| Distribution | One deterministic, uncompressed ZIP with runtime files and legal notices |

The model archive is downloaded once. Translation remains on-device; this repository does not provide or require a translation API.

## QA status

The candidate preserved 16/32 FP32 outputs exactly on a synthetic preservation corpus, with mean token similarity 0.8617 and no omission-heuristic flags. It also introduced a known repetition regression: `まさか…` changed from the FP32 baseline `No puede ser.` to `No, no, no.`

These measurements **do not prove translation quality**. The corpus is synthetic and agent-authored, FP32 is only a numerical baseline, and Android execution has not yet been validated. Human semantic review and API 29/36 device profiling are required before production use. See [MODEL_CARD.md](MODEL_CARD.md) and [the benchmark summary](benchmarks/quantization-preservation.md).

## Rebuild the release

The build verifies every source byte against pinned size and SHA-256 before creating the archive.

```bash
python3 scripts/build_release.py \
  --lab-root /tmp/manga-model-lab \
  --output-dir /tmp/manga-model-release
python3 -m unittest discover -s tests -v
```

Model binaries and release archives are intentionally excluded from Git.

## License

The upstream model card declares Apache-2.0. This repository, quantized graphs, and release metadata are distributed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
