import cupy as cp
import numpy as np


def create_gaussian_cp(x, xs, deltaX, k0, step_size):
    temp = cp.asnumpy(x)
    gaussian = np.exp(-(temp-xs)**2 / (2 * deltaX**2)) \
        * np.exp(1j * k0 * temp)
    gaussian[0] = 0
    gaussian[1] = 0
    gauss_sq = np.abs(gaussian)**2
    gauss_int = np.trapz(gauss_sq, temp, step_size)
    norm_const = 1 / gauss_int
    return cp.asarray(norm_const * gaussian)


def create_gaussian(x, xs, deltaX, k0, step_size):
    temp = x
    gaussian = np.exp(-(temp-xs)**2 / (2 * deltaX**2)) \
        * np.exp(1j * k0 * temp)
    gaussian[0] = 0
    gaussian[1] = 0
    gauss_sq = np.abs(gaussian)**2
    gauss_int = np.trapz(gauss_sq, temp, step_size)
    norm_const = 1 / np.sqrt(gauss_int)
    return norm_const * gaussian

