from ann_benchmarks.plotting.metrics import all_metrics as metrics

all_plot_variants = {
    "recall/time": ("k-nn", "qps"),
    "recall/buildtime": ("k-nn", "build"),
    "recall/indexsize": ("k-nn", "indexsize"),
    "recall/distcomps": ("k-nn", "distcomps"),
    "rel/time": ("rel", "qps"),
    "recall/candidates": ("k-nn", "candidates"),
    "recall/qpssize": ("k-nn", "queriessize"),
    "eps/time": ("epsilon", "qps"),
    "largeeps/time": ("largeepsilon", "qps"),
    "recall/p50": ("k-nn", "p50"),
    "recall/p95": ("k-nn", "p95"),
    "recall/p99": ("k-nn", "p99"),
    "recall/p999": ("k-nn", "p999"),
}
