import pytest

from ann_benchmarks.plotting.metrics import knn, queries_per_second, index_size, build_time, candidates, epsilon, rel


class DummyMetric:
    def __init__(self):
        self.attrs = {}
        self.d = {}

    def __getitem__(self, key):
        return self.d.get(key, None)

    def __setitem__(self, key, value):
        self.d[key] = value

    def __contains__(self, key):
        return key in self.d

    def create_group(self, name):
        self.d[name] = DummyMetric()
        return self.d[name]


def test_recall():
    exact_queries = [[0.1, 0.25]]
    run1 = [[]]
    run2 = [[0.2, 0.3]]
    run3 = [[0.2]]
    run4 = [[0.2, 0.25]]

    assert knn(exact_queries, run1, 2, DummyMetric()).attrs["mean"] == pytest.approx(0.0)
    assert knn(exact_queries, run2, 2, DummyMetric()).attrs["mean"] == pytest.approx(0.5)
    assert knn(exact_queries, run3, 2, DummyMetric()).attrs["mean"] == pytest.approx(0.5)
    assert knn(exact_queries, run4, 2, DummyMetric()).attrs["mean"] == pytest.approx(1.0)


def test_epsilon_recall():
    exact_queries = [[0.05, 0.08, 0.24, 0.3]]
    run1 = [[]]
    run2 = [[0.1, 0.2, 0.55, 0.7]]

    assert epsilon(exact_queries, run1, 4, DummyMetric(), 1).attrs["mean"] == pytest.approx(0.0)

    assert epsilon(exact_queries, run2, 4, DummyMetric(), 0.0001).attrs["mean"] == pytest.approx(0.5)

    # distance can be off by factor (1 + 1) * 0.3 = 0.6 => recall .75
    assert epsilon(exact_queries, run2, 4, DummyMetric(), 1).attrs["mean"] == pytest.approx(0.75)

    # distance can be off by factor (1 + 2) * 0.3 = 0.9 => recall 1
    assert epsilon(exact_queries, run2, 4, DummyMetric(), 2).attrs["mean"] == pytest.approx(1.0)


def test_relative():
    exact_queries = [[0.1, 0.2, 0.25, 0.3]]
    run1 = []
    run2 = [[0.1, 0.2, 0.25, 0.3]]
    run3 = [[0.1, 0.2, 0.55, 0.9]]

    assert rel(exact_queries, run1, DummyMetric()) == pytest.approx(float("inf"))
    assert rel(exact_queries, run2, DummyMetric()) == pytest.approx(1)

    # total distance exact: 0.85, total distance run3: 1.75
    assert rel(exact_queries, run3, DummyMetric()) == pytest.approx(1.75 / 0.85)


def test_queries_per_second():
    assert queries_per_second([], {"best_search_time": 0.01}) == 100


def test_index_size():
    assert index_size([], {"index_size": 100}) == 100


def test_build_time():
    assert build_time([], {"build_time": 100}) == 100


def test_candidates():
    assert candidates([], {"candidates": 10}) == 10
