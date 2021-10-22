from stats import calculate_mean, calculate_cov


class PCA:
    def __init__(self, mean=None, cov=None):
        self.mean = None
        self.cov = None

    def calculate_pca(self, x_data):
        """
        Given x_data as the input data calculate
        a matrix to calculate a PCA matrix
        :return: W
        """
        self.mean = calculate_mean(x_data)
        self.cov = calculate_cov(x_data, self.mean)
        NotImplemented
