from ann_benchmarks.plotting.metrics import all_metrics as metrics

all_plot_variants = {
    "recall/time" : (metrics["k-nn"], metrics["qps"]),
    "recall/buildtime" : (metrics["k-nn"], metrics["build"]),
    "recall/indexsize" : (metrics["k-nn"], metrics["indexsize"]),
    "recall/rel" : (metrics["k-nn"], metrics["rel"]),
    "candidates/recall" : (metrics["candidates"], metrics["k-nn"]),
    "recall/eps" : (metrics["k-nn"], metrics["epsilon"])
     }
