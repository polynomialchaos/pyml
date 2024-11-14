import numpy as np
from .loss import Loss


class BinaryCrossEntropyLoss(Loss):
    """ Binary cross entropy loss function class """

    def function(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def function_derive(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return ((1 - y) / (1 - y_hat) - y / y_hat) / np.size(y)


class MeanSquaredErrorLoss(Loss):
    """ Mean squared error loss function class """

    def function(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return np.mean(np.power(y - y_hat, 2))

    def function_derive(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return 2 * (y_hat - y) / y.size
