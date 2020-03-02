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

    "recall_0.8/time": ("k-nn_0.8", "qps"),
    "recall_0.8/buildtime": ("k-nn_0.8", "build"),
    "recall_0.8/indexsize": ("k-nn_0.8", "indexsize"),
    "recall_0.8/distcomps": ("k-nn_0.8", "distcomps"),
    "recall_0.8/candidates": ("k-nn_0.8", "candidates"),
    "recall_0.8/qpssize": ("k-nn_0.8", "queriessize"),
    "eps_0.8/time": ("epsilon_0.8", "qps"),
    "largeeps_0.8/time": ("largeepsilon_0.8", "qps")
}
