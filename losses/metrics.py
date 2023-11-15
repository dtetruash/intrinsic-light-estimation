"""This file holds a collection of metric evauation functions which are useful
for logging to a time-series but are not used for model training purposes.
"""

import math
from skimage.metrics import structural_similarity, mean_squared_error
import numpy as np
import torch


def psnr(mse):
    """Compute the Peak Single Noise Ratio from MSE of some data.

    Args:
        mse (float): mean squared error of some data

    Returns:
        PSNR metric
    """
    return -10.0 * math.log10(mse)


def dssim(img_pred, img_target):
    """Compute the structural dissimilarity between two image arrays.
    Only supports single image pairs.
    """
    return (1.0 - ssim(img_pred, img_target)) * 0.5


def ssim(img_pred, img_target, **kwargs):
    """Compute the structural similarity between two image arrays.
    Only supports single image pairs.
    """
    data_range = img_target.max() - img_target.min()
    return structural_similarity(img_pred, img_target, data_range=data_range, **kwargs)


def scale_inv_L2(pred, target, lamb=0.5, torch_mode=True):
    """Scale-invariant L2 loss (MSE)
    Is only defined on values above zero. Will clip element-wise between 1e-3 and 1.0
    """
    assert pred.shape == target.shape, (
        f"Given {'tensors' if torch_mode else 'arrays'} "
        "were of differing shapes: {pred.shape, target.shape}."
    )

    lib = torch if torch_mode else np

    dim = target.dim() if torch_mode else target.ndim

    if dim == 2:  # (B, 3)
        num_pixles = target.shape[0]
    elif dim == 3:  # (W,H,3)
        num_pixles = target.shape[0] * target.shape[1]
    else:
        raise ValueError(
            f"Unsupported number of dimsnions in array/tensor {dim}. "
            "Supported 2 (B, 3) or 3 (W,H,3)."
        )

    # Clip and add eps to avoid -ve infs and nans after log
    eps = float(1e-5)
    clipped_pred = lib.clip(pred, eps, 1.0)
    clipped_target = lib.clip(target, eps, 1.0)

    # Get element-wize log difference
    diff = lib.log(clipped_pred) - lib.log(clipped_target)

    if torch_mode:
        assert isinstance(diff, torch.Tensor)
        mse = torch.square(diff).mean()
    else:
        mse = np.square(diff).mean()

    num_pixles_sqr = num_pixles * num_pixles
    scale_inv_mse = lib.square(diff.sum()) / float(num_pixles_sqr)

    return mse - lamb * scale_inv_mse


def local_mse(image1, image2, window_size_factor=0.1):
    """
    Compute local mean square error between two images with a specified window size.
    Note that consecutive windows do not overlap

    Parameters:
    - image1: numpy array representing the first image
    - image2: numpy array representing the second image
    - window_size_factor: factor for determining the window size (default is 0.1)

    Returns:
    - local_mse: numpy array representing the local mean square error map
    """

    # Ensure the images have the same shape
    assert image1.shape == image2.shape, "Images must have the same shape"

    # Compute window size based on the specified factor
    window_size = int(image1.shape[1] * window_size_factor)

    # Compute the local mean square error using a rolling window
    loc_mse_accu = 0.0
    window_ctn = 0

    for i in range(0, image1.shape[0] - window_size + 1, window_size):
        for j in range(0, image1.shape[1] - window_size + 1, window_size):
            window1 = image1[i : i + window_size, j : j + window_size, :]
            window2 = image2[i : i + window_size, j : j + window_size, :]

            mse = mean_squared_error(window1, window2)

            loc_mse_accu += mse
            window_ctn += 1

    return loc_mse_accu / window_ctn


def lpips():
    # Call https://github.com/richzhang/PerceptualSimilarity
    raise NotImplementedError()
