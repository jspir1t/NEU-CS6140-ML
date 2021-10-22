import pytest
import numpy as np
from src.stats import calculate_mean

@pytest.fixture
def sample_data():
    x = np.rand(3,3) # You can change this sample data should you wish to
    yield x

@pytest.fixture
def mean_of_data(sample_data):
    yield np.mean(sample_data)

def test_mean(sample_data):
    """
    GIVEN a sample data set
    WHEN the mean
    :param sample_data:
    :return:
    """
    mean = calculate_mean(sample_data)
    assert np.allclose(mean, np.mean(sample_data))


def test_cov(sample_data, mean_of_data):
    NotImplemented