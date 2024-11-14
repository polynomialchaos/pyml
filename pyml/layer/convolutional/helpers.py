import numpy as np


def cross_correlate_2d(matrix: np.ndarray, kernel: np.ndarray, padding: tuple[int, int] | None = None):
    """ Convolve two 2-dimensional arrays """
    n, m = matrix.shape
    if padding is None:
        padding = (0, 0)

    matrix = _add_padding(matrix, padding)

    n_p, m_p = matrix.shape
    k = kernel.shape

    if n_p < k[0] or m_p < k[1]:
        raise (ValueError('Kernel can\'t be bigger than matrix in terms of shape.'))

    h_out = n + 2 * padding[0] - k[0] + 1
    w_out = m + 2 * padding[1] - k[1] + 1

    if h_out <= 0 or w_out <= 0:
        raise (ValueError(
            'Can\'t apply input parameters, one of resulting output dimension is non-positive.'))

    matrix_out = np.zeros((h_out, w_out))

    for i in range(h_out):
        indices_x = list(range(i, i + k[0]))

        for j in range(w_out):
            indices_y = list(range(j, j + k[1]))

            submatrix = matrix[indices_x, :][:, indices_y]
            matrix_out[i][j] = np.sum(np.multiply(submatrix, kernel))

    return matrix_out


def convolve_2d(matrix: np.ndarray, kernel: np.ndarray, padding: tuple[int, int] | None = None):
    """ Convolve two 2-dimensional arrays (flips kernel) """
    kernel_flip = np.flipud(np.fliplr(kernel))
    return cross_correlate_2d(matrix, kernel_flip, padding=padding)


def _add_padding(matrix: np.ndarray, padding: tuple[int, int]):
    """ Add padding to given matrix """
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r: n + r, c: m + c] = matrix

    return padded_matrix
