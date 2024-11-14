from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """ Layer (base) class """

    def __init__(self):
        self.input: np.ndarray = None
        self.output: np.ndarray = None

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        """ Handler to call backward propagation """
        return self.backward_propagation(output_error, learning_rate)

    @abstractmethod
    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        """ Computes dE/dX for a given output_error=dE/dY (and update parameters if any) """
        raise NotImplementedError

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """ Handler to call forward propagation """
        self.input = input_data
        self.output = self.forward_propagation()
        return self.output

    @abstractmethod
    def forward_propagation(self) -> np.ndarray:
        """ Computes the output Y of a layer for a given input X """
        raise NotImplementedError
