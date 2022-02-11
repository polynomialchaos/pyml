# MIT License
#
# Copyright (c) 2021 Florian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from ctypes.wintypes import POINT
import os
import numpy as np
from ctypes import cdll, POINTER, c_size_t, c_void_p, c_double

c_size_t_p = POINTER(c_size_t)
c_double_p = POINTER(c_double)
local_path = os.path.dirname(__file__)
lib_path = os.path.join(local_path, 'libexternal.so')


class ExternalLibrary():
    """External library class."""

    def __init__(self):
        self._lib = cdll.LoadLibrary(lib_path)
        self._set_library_argtypes_restype()

    def _set_library_argtypes_restype(self):
        self._lib.cross_correlate_2d.argtypes = [
            c_size_t_p, c_void_p, c_size_t_p, c_void_p, c_size_t_p, c_void_p
        ]
        self._lib.cross_correlate_2d.restype = None

    def convolve_2d(self, matrix, kernel, mode='full', use_lib=True):
        """Achieve the 2D cross correlation."""
        kernel_flip = flip_kernel(kernel)
        return self.cross_correlate_2d(matrix, kernel_flip,
                                       mode=mode, use_lib=use_lib)

    def cross_correlate_2d(self, matrix, kernel, mode='full', use_lib=True):
        """Achieve the 2D cross correlation."""
        nij = matrix.shape
        kij = kernel.shape
        if kij[0] % 2 == 0 or kij[1] % 2 == 0:
            raise ValueError('Kernel shape should be odd!')

        if mode == 'valid':
            pij = 0, 0
            mij = nij[0] - kij[0] + 1, nij[1] - kij[1] + 1
        elif mode == 'same':
            pij = (kij[0] - 1) // 2, (kij[1] - 1) // 2
            mij = nij[0], nij[1]
        elif mode == 'full':
            pij = kij[0] - 1, kij[1] - 1
            mij = nij[0] + kij[0] - 1, nij[1] + kij[1] - 1
        else:
            raise KeyError('Unknown mode!')

        if pij[0] > 0 or pij[1] > 0:
            matrix = add_padding(matrix, pij)

        npij = matrix.shape
        if npij[0] < kij[0] or npij[1] < kij[1]:
            raise ValueError(
                'Kernel can\'t be bigger than matrix in terms of shape!')

        if mij[0] <= 0 or mij[1] <= 0:
            raise ValueError(
                'Can\'t apply input parameters, one of resulting output dimension is non-positive.')

        matrix_out = np.zeros(mij)
        if use_lib:
            self._lib.cross_correlate_2d(
                (c_size_t * 2)(*npij), c_void_p(matrix.ctypes.data),
                (c_size_t * 2)(*kij), c_void_p(kernel.ctypes.data),
                (c_size_t * 2)(*mij), c_void_p(matrix_out.ctypes.data)
            )
        else:
            for i in range(mij[0]):
                indices_x = list(range(i, i + kij[0]))

                for j in range(mij[1]):
                    indices_y = list(range(j, j + kij[1]))

                    submatrix = matrix[indices_x, :][:, indices_y]
                    matrix_out[i][j] = np.sum(np.multiply(submatrix, kernel))

        return matrix_out


def add_padding(matrix, padding):
    """Add padding to given matrix."""
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r: n + r, c: m + c] = matrix

    return padded_matrix


def flip_kernel(kernel):
    """Flip the given kernel."""
    kernel_out = np.copy(kernel)
    kernel_out = np.flipud(np.fliplr(kernel_out))
    kernel_out = kernel_out.ravel().reshape(kernel_out.shape)

    return kernel_out
