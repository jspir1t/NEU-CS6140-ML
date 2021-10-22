import pytest
import numpy as np
import os
import sys
import time

CURDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURDIR, "../"))
import distances


@pytest.fixture
def get_samples():
    # These can be replaced with actual
    # vectors you want to work with
    x1 = np.random.rand(10000)
    x2 = np.random.rand(100)
    x3 = np.random.rand(10000)
    x4 = np.random.rand(100)
    yield x1, x2, x3, x4


def test_equality(get_samples):
    """
    This is an example test case
    with fixtures and documentation
    GIVEN Two vectors x1, x2
    WHEN they are generated randomly to be of unqeual sizes
    THEN assert that the sizes are actually unequal
    :param get_samples:
    :return:
    """
    x1 = get_samples[0]
    x2 = get_samples[1]
    assert np.size(x1) != np.size(x2)


def test_euclidean_for_loop(get_samples):
    """
    Use this to test your for loop implementation
    :param get_samples:
    :return:
    """
    x1 = get_samples[0]
    x2 = get_samples[1]
    x3 = get_samples[2]
    x4 = get_samples[3]

    start = time.time()
    result = distances.euclidean_for_loop(x1, x3)
    end = time.time()
    print(f'euclidean_for_loop for k=10000 cost: {end - start} seconds')
    assert (np.linalg.norm(x1 - x3) - result) < 1e-10

    start = time.time()
    result = distances.euclidean_for_loop(x2, x4)
    end = time.time()
    print(f'euclidean_for_loop for k=100 cost: {end - start} seconds')
    assert (np.linalg.norm(x2 - x4) - result) < 1e-10


def test_euclidean_vectorized(get_samples):
    """
    Use this to test your for loop implementation
    Note you cannot use numpy.linalg.norm
    :param get_samples:
    :return:
    """
    x1 = get_samples[0]
    x2 = get_samples[1]
    x3 = get_samples[2]
    x4 = get_samples[3]

    start = time.time()
    result = distances.euclidean_vectorized(x1, x3)
    end = time.time()
    print(f'euclidean_for_vectorized for k=10000 cost: {end - start} seconds')
    # Compare the result with np.linalg.norm()'s result. Due to the precision issue of float
    # check by difference smaller than 1e-10
    assert (np.linalg.norm(x1 - x3) - result) < 1e-10

    start = time.time()
    result = distances.euclidean_vectorized(x2, x4)
    end = time.time()
    print(f'euclidean_for_vectorized for k=100 cost: {end - start} seconds')
    assert (np.linalg.norm(x2 - x4) - result) < 1e-10
