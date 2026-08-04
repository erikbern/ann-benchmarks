import io
import sys

import numpy
import pytest

from ann_benchmarks.datasets import train_test_split


def test_train_test_split_infers_dimension():
    """When dimension is not supplied, it should be inferred from X.shape[1]."""
    X = numpy.random.rand(50, 32).astype(numpy.float32)

    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        train, test = train_test_split(X, test_size=10, dimension=None)
    finally:
        sys.stdout = old_stdout

    assert train.shape == (40, 32)
    assert test.shape == (10, 32)
    assert "Splitting 50*32 into train/test" in captured.getvalue()


def test_train_test_split_uses_explicit_dimension():
    """When dimension is supplied, the print message should use it."""
    X = numpy.random.rand(50, 32).astype(numpy.float32)

    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        train, test = train_test_split(X, test_size=10, dimension=64)
    finally:
        sys.stdout = old_stdout

    assert train.shape == (40, 32)
    assert test.shape == (10, 32)
    assert "Splitting 50*64 into train/test" in captured.getvalue()
