from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """ Function (base) class """

    @abstractmethod
    def function(self, x: np.ndarray) -> np.ndarray:
        """ Function callable """
        raise NotImplementedError

    @abstractmethod
    def function_derive(self, x: np.ndarray) -> np.ndarray:
        """ Function derivative callable """
        raise NotImplementedError
