import imp
import numpy as np
from .function import Function


class HyperbolicTangent(Function):
    """Hyperbolic tangent activation function class."""

    def function(self, x):
        return np.tanh(x)

    def function_derive(self, x):
        return 1 - np.tanh(x)**2
