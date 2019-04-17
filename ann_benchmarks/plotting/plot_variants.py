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
    "largeeps/time": ("largeepsilon", "qps")
}
