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
import numpy as np


def add_padding(matrix, padding):
    """Add padding to given matrix."""
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r: n + r, c: m + c] = matrix

    return padded_matrix


def cross_correlate_2d(matrix, kernel, padding=(0, 0)):
    """Convolve two 2-dimensional arrays."""
    n, m = matrix.shape
    if list(padding) != [0, 0]:
        matrix = add_padding(matrix, padding)

    n_p, m_p = matrix.shape
    k = kernel.shape

    if n_p < k[0] or m_p < k[1]:
        raise(ValueError('Kernel can\'t be bigger than matrix in terms of shape.'))

    h_out = n + 2 * padding[0] - k[0] + 1
    w_out = m + 2 * padding[1] - k[1] + 1

    if h_out <= 0 or w_out <= 0:
        raise(ValueError(
            'Can\'t apply input parameters, one of resulting output dimension is non-positive.'))

    matrix_out = np.zeros((h_out, w_out))

    for i in range(h_out):
        indices_x = list(range(i, i + k[0]))

        for j in range(w_out):
            indices_y = list(range(j, j + k[1]))

            submatrix = matrix[indices_x, :][:, indices_y]
            matrix_out[i][j] = np.sum(np.multiply(submatrix, kernel))

    return matrix_out


def convolve_2d(matrix, kernel, padding=(0, 0)):
    """Convolve two 2-dimensional arrays (flips kernel)."""
    kernel_flip = np.flipud(np.fliplr(kernel))
    return cross_correlate_2d(matrix, kernel_flip, padding=padding)
