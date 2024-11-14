import numpy as np
from .activation import Activation


class HyperbolicTangentActivation(Activation):
    """ Hyperbolic tangent activation function class """

    def function(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def function_derive(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2


class SigmoidActivation(Activation):
    """ Sigmoid activation function class """

    def function(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def function_derive(self, x: np.ndarray) -> np.ndarray:
        tmp = self.function(x)
        return tmp * (1 - tmp)
