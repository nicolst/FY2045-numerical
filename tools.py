import numpy as np


def create_gaussian(x, xs, sigX, k0):
    gaussian = np.exp(-(x-xs)**2 / (2 * sigX**2)) * np.exp(1j * k0 * x)
    gaussian[0] = 0
    gaussian[-1] = 0
    gauss_sq = gaussian.real**2 + gaussian.imag**2
    gauss_int = np.trapz(gauss_sq, x)
    norm_const = 1 / np.sqrt(gauss_int)
    return norm_const * gaussian
