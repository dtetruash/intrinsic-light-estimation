"""This file holds a collection of metric evauation functions which are useful
for logging to a time-series but are not used for model training purposes.
"""
import math


def psnr(mse):
    """Compute the Peak Single Noise Ratio from MSE of some data.

    Args:
        mse (float): mean squared error of some data

    Returns:
        PSNR metric
    """
    return -10.0 * math.log10(mse)
