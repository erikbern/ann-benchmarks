from ann_benchmarks.plotting.metrics import all_metrics as metrics

all_plot_variants = {
    "recall/time" : (metrics["k-nn"], metrics["qps"]),
    "recall/buildtime" : (metrics["k-nn"], metrics["build"]),
    "recall/indexsize" : (metrics["k-nn"], metrics["indexsize"]),
    "rel/time" : (metrics["rel"], metrics["qps"]),
    "recall/candidates" : (metrics["k-nn"], metrics["candidates"]),
    "eps/time" : (metrics["epsilon"], metrics["qps"])
     }
