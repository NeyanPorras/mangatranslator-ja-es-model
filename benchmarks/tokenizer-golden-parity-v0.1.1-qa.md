# Tokenizer graph parity

The `v0.1.1-qa` tokenizer graphs executed successfully against the pinned host runtimes. This is runtime-parity evidence, **not translation-quality evidence**.

| Check | Result |
|---|---:|
| Source Marian IDs | 6/6 exact |
| Source attention masks | 6/6 exact |
| Target decoded text | 4/4 exact |
| Execution | Real ONNX graphs |

The source cases cover punctuation, `nmt_nfkc` normalization, OCR-like noise, unknown characters, multi-clause text, and a longer input. Target checks include special-token filtering. Full evidence is retained outside Git with SHA-256 `e7a3288e1e6d7047223e8dcc6626b493a15a1547eab6b4872617d8c7a432bcc2`.

Android custom-op loading and device performance remain separate acceptance gates.
