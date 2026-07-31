# Provenance

`bundle-source.json` is the authoritative source-to-release mapping. Every runtime file is pinned by relative model-lab path, byte size, and SHA-256. `ort_config.json` records the exact dynamic per-tensor QOperator configuration used to produce the graphs.

The release builder refuses missing, resized, or rehashed inputs. It does not download or regenerate the upstream model; model export and quantization remain separate, reviewable processes.
