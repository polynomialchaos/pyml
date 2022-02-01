import imp
import numpy as np
from .function import Function


class HyperbolicTangentActivation(Function):
    """Hyperbolic tangent activation function class."""

    def function(self, x):
        return np.tanh(x)

    def function_derive(self, x):
        return 1 - np.tanh(x)**2

class SigmoidActivation(Function):
    """Sigmoid activation function class."""

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def function_derive(self, x):
        tmp = self.function(x)
        return tmp * (1 - tmp)