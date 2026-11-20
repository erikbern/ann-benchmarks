# Ice Lake CPU0 succinct measurements

Measured on Alibaba Cloud ecs.r7.16xlarge, Xeon Platinum 8369B. ANN-Benchmarks 01de5eb, succinct dc0f73ef. The core src tree equals 3cb02f17. Reference QPS are user-supplied. Actual recall and RSS match the validation record in g9ibench 07fcd9b. The hardware of that exact reference run still needs its original manifest.

Each setting ran on CPU0 with memory bound to NUMA node 0. Each of three fresh processes performed a full-query warmup and five full-query timing rounds. The table gives the median of 15 QPS values. scores.csv also gives strict ID recall, each best-of-five observation, the median of these three observations and mean latency. All returned IDs were stable and distances were independently verified in FP64. The timer includes Python invocation, query preparation, native QG search and result materialization. Corpus-based distance checks occur between timed queries.

| Dataset | Reference actual recall | Actual ANN recall | Measured QPS | Reference QPS | Measured/reference | Query RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| SIFT | 90.007% | 90.012% | 43,897 | 58,407 | 75.2% | 4.206 |
| SIFT | 95.158% | 95.165% | 36,037 | 47,393 | 76.0% | 4.206 |
| SIFT | 99.019% | 99.020% | 23,625 | 31,750 | 74.4% | 4.206 |
| GIST | 90.250% | 90.260% | 13,168 | 17,775 | 74.1% | 13.220 |
| GIST | 95.740% | 95.790% | 8,818 | 12,041 | 73.2% | 13.220 |
| GIST | 99.430% | 99.460% | 3,216 | 4,231 | 76.0% | 13.220 |
| GLOVE | 90.721% | 90.730% | 15,611 | 22,578 | 69.1% | 5.048 |
| GLOVE | 96.034% | 96.047% | 9,145 | 13,330 | 68.6% | 5.048 |
| GLOVE | 99.189% | 99.213% | 4,244 | 6,008 | 70.6% | 5.048 |

RSS is the median of three process query-window maxima (20 ms samples), including the Python evaluator and retained training corpus. This 5.15 host rejected natural require admission when coverage was unknown. Local measurements use hint, while the reference used a matched graph PMD layout. Local index hashes differ from the reference. Fixed-index process repetitions exclude rebuild variance.

SIFT uses the local M32/400 container. GIST uses the M50/500 substrate previously used for CPU54 CSR export. The GloVe M64/800 streamer graph was built from all 1,183,514 vectors, then encoded through the public QG codec, router and snapshot APIs. Only the independent succinct QG path ran during queries. Inputs and runtime hashes are in inputs.json and each artifact.json. The kernel and codec experiments are separate from the frozen campaign. The archived historical vendor comparison is in ../huawei-gist-20261119/.

## Reproduction

Use the frozen binary manifests and providers for ordered-ID reproduction. Absolute paths below correspond to the measurement host. The replay program accepts relocated data, provider, GloVe snapshot and runtime manifests and verifies their hashes. The adjacent .router file belongs to the same snapshot. The original complete raw rounds, latencies, IDs, sampled RSS and source checkout are under the run directory R.

~~~bash
B=/home/xingrui/workspace/annbenchmark/ann-benchmarks
A=/home/xingrui/workspace/annbenchmark/c7ibench
R="$B/results/cpu0-reference-alignment"
ARCH="$A/ann_benchmarks/algorithms/zvec_qg/validation/cpu0-icelake-20261119"
D="$R/replay-$(date +%Y%m%d-%H%M%S)"
mkdir "$D"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMA="$B/results/r7i-new-api-20261119/local/usr/bin/numactl"
"$B/.venv/bin/python" "$R/core/tools/core/bench/qg_bench.py" isolate \
    --manifest "$D/isolation.json" -- bash "$ARCH/replay-all.sh" "$ARCH" "$D" "$B" "$NUMA"
~~~

The measured bridge sources and exact compiler/link arguments are under runtime/. The build used GCC 11.4, CMake, Ninja, Release, -O3 and -march=icelake-server. The original full source layout and build commands are retained under R. To rebuild the two bridge variants in a new directory T with the same dependencies, place the succinct source at T/core with initialized submodules. Use the Release host's dependency settings and the following commands. Keep each produced module with its own runtime manifest.

~~~bash
cmake -S "$T/core" -B "$T/core/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_TOOLS=ON -DBUILD_TESTING=OFF -DENABLE_ICELAKE=ON -DENABLE_OPENMP=OFF
cmake --build "$T/core/build" --target local_builder
cp "$ARCH/runtime/link.py" "$T/link.py"
cp "$ARCH/runtime/reference.cpp" "$T/direct.cpp"
"$B/.venv/bin/python" "$T/link.py"
mkdir "$T/reference-runtime" "$T/sidecar-runtime"
cp "$T"/_direct_core*.so "$T/reference-runtime/"
cp "$ARCH/runtime/sidecar.cpp" "$T/direct.cpp"
"$B/.venv/bin/python" "$T/link.py"
cp "$T"/_direct_core*.so "$T/sidecar-runtime/"
~~~

The GloVe base builder input is runtime/glove-builder.yaml. For a fresh build, copy the YAML and change IndexPath and DumpPath to new output files. Run local_builder on the YAML and pass its IndexPath streamer file as the provider. Multi-thread construction can yield a different graph. To rebuild the public low-rank snapshot with the measured block4 model on this substrate, set BASE, SNAP and MODULE_DIR to the new streamer path, a new snapshot path and T/sidecar-runtime, respectively.

~~~bash
export BASE SNAP MODULE_DIR
"$T/core/build/bin/local_builder" "$NEW_BUILDER_YAML"
"$B/.venv/bin/python" "$ARCH/runtime/encode-glove.py"
~~~

The newly built graph needs fresh hashes, an identified runtime manifest and a new correctness/performance validation campaign. Replay of the archived results checks the frozen inputs, exact returned IDs and recall each round.


## Comparison with other local measurements

The final frozen table is one campaign. At the stricter GloVe actual recalls of 90.730%, 96.047% and 99.213%, the earlier campaign on this same block4 graph and settings measured 16,376 / 9,596 / 4,382 QPS. Differences between these repeated campaigns are observable even with identical results. A separate run targeting the nominal floors measured 17,328 / 10,884 / 4,832 QPS at 90.007% / 95.013% / 99.011%. The latter meets different actual recall targets and is archived in glove-nominal.json.

In the local historical comparison, Huawei GloVe achieved 16,135 / 9,183 / 3,809 QPS and 01AI achieved 15,456 / 7,948 / 2,592 QPS. On the final frozen run the two lower GloVe cells fell below those historical Huawei values. SIFT and GIST cleared both local historical comparators in all six cells. Supplier measurements and Zvec measurements were taken in separate campaigns. Comparing the GloVe Gaussian block8 and block4 codecs on the same M64 graph favored block4 at all three reference actual-recall floors. GIST achieved higher high-recall throughput on the alternative substrate at the cost of some throughput at the two lower thresholds.

## GloVe query presets

Use `--preset recommended` for the seven nominal recall floors.
It uses the frozen CPU0 GloVe provider, snapshot, router and succinct
binary recorded above. Omitting the flag replays the archived reference run.

| Recommended floor | Actual ANN Recall@10 | QPS |
|---:|---:|---:|
| 30% | 31.633% | 64,953 |
| 50% | 51.362% | 50,734 |
| 70% | 70.572% | 35,343 |
| 85% | 85.003% | 22,270 |
| 90% | 90.007% | 17,275 |
| 95% | 95.013% | 10,916 |
| 99% | 99.011% | 4,855 |

These QPS values come from the individually identified campaigns in the
[preset file](../../configs/r7i-cpu0/glove.json). Each is the median of
three process-level best-of-five QPS values using all 10,000 queries.
Each point also records strict ID recall and the expected ordered-ID digest.

To reproduce the recommended settings, run the following on the measurement
host with `B`, `R`, `ARCH` and `NUMA` set as in the reproduction section.

~~~bash
OUT=$(mktemp -d "$R/glove-replay-XXXXXX")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$B/.venv/bin/python" "$R/core/tools/core/bench/qg_bench.py" isolate \
    --manifest "$OUT/isolation.json" -- bash -ec '
        for rep in 0 1 2; do
            "$1" --physcpubind=0 --membind=0 "$2/.venv/bin/python" \
                "$3/replay.py" glove --preset recommended --repeat "$rep" \
                --data "$2/data/glove-100-angular.hdf5" --output "$4/rep-$rep"
        done
    ' _ "$NUMA" "$B" "$ARCH" "$OUT"
~~~
