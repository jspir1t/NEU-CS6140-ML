import numpy


def euclidean_for_loop(x1: numpy.ndarray, x2: numpy.ndarray) -> float:
    """
    Add doc string here
    :param x1:
    :param x2:
    :return:
    """
    ans = 0.
    for x in x1 - x2:
        ans += x ** 2
    return numpy.sqrt(ans)


def euclidean_vectorized(x1: numpy.ndarray, x2: numpy.ndarray) -> float:
    """
    Add doc string here
    :param x1:
    :param x2:
    :return:
    """
    return numpy.sqrt(numpy.sum(numpy.power(x1 - x2, 2)))
