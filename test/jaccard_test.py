import pytest
from ann_benchmarks.distance import jaccard


def test_similarity():
    a = [1, 2, 3, 4]
    b = []
    c = [1, 2]
    d = [5, 6]

    assert jaccard(a, b) == pytest.approx(0.0)
    assert jaccard(a, a) == pytest.approx(1.0)
    assert jaccard(a, c) == pytest.approx(0.5)
    assert jaccard(c, d) == pytest.approx(0.0)
