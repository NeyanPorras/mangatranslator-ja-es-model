# Provenance

`bundle-source.json` remains the authoritative mapping for `v0.1.0-qa`. `bundle-source-v0.1.1-qa.json` pins the verified base archive and the two generated tokenizer graph identities. `ort_config.json` records the dynamic per-tensor QOperator configuration used for the translation graphs.

The `v0.1.1-qa` builder first verifies every `v0.1.0-qa` archive entry, then preserves all eight runtime entries byte-for-byte and adds exactly two tokenizer graphs. It refuses missing, extra, resized, rehashed, or path-unsafe content. Model export and quantization remain separate, reviewable processes.
