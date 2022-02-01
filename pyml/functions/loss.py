import numpy as np
from .function import Function


class MeanSquaredErrorLoss(Function):
    """Mean squared error loss function class."""

    def function(self, y, y_hat):
        return np.mean(np.power(y - y_hat, 2))

    def function_derive(self, y, y_hat):
        return 2 * (y_hat - y) / y.size
