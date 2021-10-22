from stats import calculate_mean
from stats import calculate_cov

class GaussianModel:
    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov
        self.d = None # Set this to the feature dimension

    def calculate_log_likelihood(self, x):
        """
        Calculate the log-likelihood for
        :param x:
        :return:
        """
        NotImplemented


