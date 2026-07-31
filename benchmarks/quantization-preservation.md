# Quantization preservation summary

**Decision:** `int8-pertensor-qoperator` is the current QA candidate because it removed the prior clause-omission flag while remaining smaller than the per-channel candidate. This is preservation evidence, not production approval.

## Results

| Metric | FP32 | INT8 per-tensor |
|---|---:|---:|
| Exact text | 32/32 | 16/32 |
| Mean character similarity | 1.0000 | 0.8851 |
| Mean token similarity | 1.0000 | 0.8617 |
| Omission-heuristic flags | 0 | 0 |
| Deterministic repeats | 32/32 | 32/32 |
| Runtime bundle | 537.0 MiB | 137.9 MiB |
| Host median generation | 197.3 ms | 122.6 ms |
| Host RSS load delta | 925.2 MiB | 344.9 MiB |
| Host maximum process RSS | 1373.4 MiB | 702.7 MiB |

The 32 Japanese inputs were created solely for this benchmark. No manga, subtitle, published benchmark, or Spanish reference translation was copied. FP32 output is a numerical baseline and may itself be wrong or incomplete.

## Known regression

`まさか…` produced `No puede ser.` under FP32 and `No, no, no.` under INT8 per-tensor. This repetition must be evaluated and mitigated before production approval.

## Evidence identity

| Evidence | SHA-256 |
|---|---|
| Feasibility report | `bb4f74534ce3fb04a74e81cda2edadbf5444e5c4f35a12c813fb32ff0a143850` |
| Quantization report | `ddfb01172fc7f1b2fca292f0835acb256bdae62a93c2407c0b011b36f988ca29` |
| Per-tensor benchmark result | `189e2d1b88c31ae7900fd763d7a3509705cbf09f90a633e6ab9e4e6eeef86293` |
| Structured benchmark summary | `d172f2e293beea04fddf5dfd3b4ca8bbe2a1211d35936598934b6207646a33ac` |
| Synthetic corpus | `c6bf13350928da1fa488982f399cdb6eb453baf21645f206cd8be383a10d5adc` |
