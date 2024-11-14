from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """ Loss (base) class """

    @abstractmethod
    def function(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """ Function callable """
        raise NotImplementedError

    @abstractmethod
    def function_derive(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """ Function derivative callable """
        raise NotImplementedError
