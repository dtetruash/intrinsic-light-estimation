"""This file holds a collection of metric evauation functions which are useful
for logging to a time-series but are not used for model training purposes.
"""

import math
from skimage.metrics import (
    structural_similarity,
    mean_squared_error,
    peak_signal_noise_ratio,
)
import numpy as np
import torch



def mse(img_pred, img_target):
    """Uses the SK-image MSE function: MSE per-element, and not per-pixel"""
    return mean_squared_error(img_pred, img_target)


def psnr(mse):
    """Compute the Peak Single Noise Ratio from MSE of some data.

    Args:
        mse (float): mean squared error of some data

    Returns:
        PSNR metric
    """
    return -10.0 * math.log10(mse)


def psnr_sk(pred, target):
    # Checked!
    return peak_signal_noise_ratio(target, pred, data_range=1.0)


def dssim(img_pred, img_target, channel_axis=2):
    # Believable, checked.
    """Compute the structural dissimilarity between two image arrays.
    Only supports single image pairs.
    """
    return (1.0 - ssim(img_pred, img_target, channel_axis=channel_axis)) * 0.5


def ssim(img_pred, img_target, **kwargs):
    # Checked
    """Compute the structural similarity between two image arrays.
    Only supports single image pairs.
    """
    data_range = img_target.max() - img_target.min()
    return structural_similarity(img_pred, img_target, data_range=data_range, **kwargs)


def scale_inv_L2(pred, target, lamb=0.5, torch_mode=True):
    # FIXME: ALWAYS -Ve
    """Scale-invariant L2 loss (MSE). Assumes 2D input (W,H, ...)
    Is only defined on values above zero. Will clip element-wise between 1e-5 and 1.0
    """
    assert pred.shape == target.shape, (
        f"Given {'tensors' if torch_mode else 'arrays'} "
        "were of differing shapes: {pred.shape, target.shape}."
    )

    assert pred.dtype == np.float32, f"Was {pred.dtype}"
    assert pred.min() >= 0.0, f"Was {pred.min()}"
    assert pred.max() <= 1.0, f"Was {pred.max()}"

    assert target.dtype == np.float32, f"Was {target.dtype}"
    assert target.min() >= 0.0, f"Was {target.min()}"
    assert target.max() <= 1.0, f"Was {target.max()}"

    # ic(pred.shape, pred.dtype, pred.min(), pred.max())
    # ic(target.shape, target.dtype, target.min(), target.max())

    lib = torch if torch_mode else np

    # Clip and add eps to avoid -ve infs and nans after log
    eps = float(1e-5)
    clipped_pred = lib.clip(pred, eps, 1.0)
    clipped_target = lib.clip(target, eps, 1.0)

    # Get element-wize log difference
    diff = lib.log(clipped_pred) - lib.log(clipped_target)

    num_pixles = target.shape[0] * target.shape[1]  # always assuming 2D input
    if torch_mode:
        assert isinstance(diff, torch.Tensor)
        mse = torch.sum(torch.square(diff)) / float(num_pixles)
    else:
        mse = np.sum(np.square(diff)) / float(num_pixles)

    num_pixles_sqr = num_pixles * num_pixles
    scale_inv_mse = lib.square(diff.sum()) / float(num_pixles_sqr)

    # ic("sil2:", mse, scale_inv_mse)

    return mse - lamb * scale_inv_mse


def local_mse(image1, image2, window_size_factor=0.1, window_step=None, use_sil2=False):
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
    # ic(image1.shape, image2.shape)

    # Compute window size based on the specified factor
    window_size = int(image1.shape[1] * window_size_factor)
    if window_step is None:
        window_step = window_size // 2

    if window_step > window_size:
        raise ValueError("The chosen window step is larger than the windiw size.")

    # Compute the local mean square error using a rolling window
    lmse_accu = 0.0
    window_ctn = 0.0
    all_lmse = []
    for i in range(0, image1.shape[0] - window_size + 1, window_step):
        for j in range(0, image1.shape[1] - window_size + 1, window_step):
            window1 = image1[i : i + window_size, j : j + window_size, :]
            window2 = image2[i : i + window_size, j : j + window_size, :]

            # Checked to be 40, 40, 3 at float 32
            # ic(window1.shape, window2.shape, window1.dtype, window2.dtype)

            if use_sil2:
                mse = scale_inv_L2(window1, window2, torch_mode=False)
            else:
                mse = mean_squared_error(window1, window2)
                # ic(mse, np.all(window1 == window2))

            lmse_accu += mse
            window_ctn += 1

    lmse = lmse_accu / window_ctn
    return lmse


def lpips():
    # Call https://github.com/richzhang/PerceptualSimilarity
    raise NotImplementedError()
