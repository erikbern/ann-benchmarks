# CORTEX Fast SIMD HNSW Vector Search Engine

High-performance native similarity search engine integration for the ann-benchmarks suite.

## Overview
- Native standalone vector search engine accelerated for modern x86_64 architectures (AVX2/FMA).
- Supports standard euclidean and angular distance metrics with high-fidelity recall ($M=32, 64$).

## Intellectual Property & License
- Benchmark Wrapper & Configuration (`module.py`, `config.yml`): Distributed under the MIT License.
- Compiled Evaluation Shared Library (`libcortex_ann_silicon.so`): Distributed under the FCSL-1.0 Evaluation License.
- Constructive Patent Notice: Protected under Patent Application CIPO CA 3,322,620. All rights reserved.
