# MangaTranslator ja→es INT8 QA candidate

## Status

**Experimental QA candidate. Not production-approved.** Use only for evaluation until human translation review and Android runtime gates pass.

## Model details

| Field | Value |
|---|---|
| Task | Japanese-to-Spanish machine translation |
| Base model | `Helsinki-NLP/opus-mt-ja-es` |
| Pinned revision | `d1693e4d2bc285d02653ff6438fe21e2b3c6025e` |
| Architecture | Marian encoder-decoder, 6 encoder and 6 decoder layers, `d_model=512` |
| Export | ONNX opset 17 |
| Quantization | Dynamic QOperator/IntegerOps; QUInt8 activations; QInt8 weights; per-tensor |
| Validated host runtime | ONNX Runtime 1.21.1 |
| License | Apache-2.0, as declared by the upstream model card |

The release contains the encoder and merged decoder graphs, exact pinned SentencePiece and vocabulary assets, model config, and six-beam generation config. The runtime bundle is 144,594,126 bytes before archive metadata and legal documents.

## Intended use

This candidate is intended for offline QA of MangaTranslator's Japanese-to-Spanish flow. Images and extracted text can remain on-device. The archive does not include Android generation orchestration or an inference service.

Do not use it for safety-critical, legal, medical, or otherwise consequential translation. Do not represent its output as professionally reviewed.

## Tokenization and generation requirements

- Execute the embedded SentencePiece `nmt_nfkc` normalization; platform NFKC is not a proven substitute.
- Use the global vocabulary mapping and EOS ID 0.
- Apply the included generation config: six beams, cache enabled, maximum length 512, pad/start ID 61917, and EOS 0.
- Implement encoder/merged-decoder beam search and all cache tensors explicitly. ONNX Runtime sessions do not implement Transformers `generate()`.

## Evaluation

A 32-sample, agent-authored synthetic manga-dialogue corpus compared INT8 behavior with FP32 output. FP32 was a numerical baseline, **not a semantic reference translation**.

| Metric | Result |
|---|---:|
| Exact text preservation | 16/32 |
| Mean character similarity | 0.8851 |
| Mean token similarity | 0.8617 |
| Conservative omission flags | 0 |
| Deterministic repeated outputs | 32/32 |
| Host median generation | 122.6 ms |
| Host RSS load delta | 344.9 MiB |
| Host maximum process RSS | 702.7 MiB |

Host performance was measured on WSL2 x86_64 with four ONNX Runtime intra-op threads. It is directional evidence only and must not be presented as Android performance.

## Known limitations

- Quantization changed half of the tested FP32 outputs.
- A repetition regression changed `まさか…` from `No puede ser.` to `No, no, no.`
- Similarity and omission heuristics do not measure semantic correctness.
- The synthetic corpus does not represent manga genres, OCR noise, dialects, names, or long-tail language adequately.
- Android ARM64/x86_64 execution, latency, memory, cancellation, and repeated-page behavior remain unvalidated.

## Required acceptance gates

1. Human semantic scoring on a representative, legally sourced Japanese manga-dialogue evaluation set.
2. Execution on Android API 29 and API 36 with exact tokenizer and generation parity tests.
3. On-device cold-load, latency, peak RSS, cancellation, and repeated-page measurements.
4. Regression handling for repetition and any newly observed omissions.

## Provenance

Source model: <https://huggingface.co/Helsinki-NLP/opus-mt-ja-es/tree/d1693e4d2bc285d02653ff6438fe21e2b3c6025e>

Exact source hashes, quantization settings, and evidence hashes are recorded under [`provenance/`](provenance/) and [`benchmarks/`](benchmarks/).
