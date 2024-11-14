import numpy as np
from .convolutional import ConvolutionalLayer
from .helpers import cross_correlate_2d, convolve_2d


class CustomConvolutionalLayer(ConvolutionalLayer):
    """ Custom convolutional layer class """

    def _correlate_2d(self, in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
        return cross_correlate_2d(in1, in2, 'valid')

    def _convolve_2d(self, in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
        return convolve_2d(in1, in2, 'full')
