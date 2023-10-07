"""This file contains functions and methods which help with reading
image files from the Intrinsic dataset and place them in correct
format for further processing.
"""

import logging

import cv2
import numpy as np
from PIL import Image
from rich.traceback import install as install_rich
from ile_utils.config import Config

install_rich()

config = Config.get_config()


# Image loading functions
def format_image_path(image_number, data_path=None, channel="", light=""):
    if data_path is None:
        data_path = config.get("paths", "data_path")

    img_name = f"{data_path}/r_{image_number:03d}"
    img_name += f"_{channel}" if channel else ""
    img_name += f"_{light}" if light else ""
    return img_name + ".png"


def get_image_paths(image_number, data_path=None, channels=[], lighting=[]):
    """Get an image along with it's given channels."""

    # get combined (fully formed) image names
    # in combined images which are given by lighting
    image_paths = {
        (channel := lt): format_image_path(image_number, data_path, channel)
        for lt in lighting
    }
    image_paths["ground_truth"] = format_image_path(image_number, data_path, "")

    # read in other channels beside combined
    image_paths.update(
        {
            channel: format_image_path(image_number, data_path, channel)
            for channel in channels
        }
    )
    return image_paths


def read_16bit(img_name):
    """Read a 16-bit PNG into a numpy array"""
    bgrd = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    rgbd = np.empty_like(bgrd)
    rgbd[..., :3] = np.flip(bgrd[..., :3], axis=2)
    rgbd[..., -1] = bgrd[..., -1]
    return rgbd


def downsample_image(image_array, downsample_ratio=2):
    return


def load_image_channels(image_number, channels, data_path=None, downsample_ratio=1):
    image_paths = get_image_paths(
        image_number,
        data_path,
        channels,
    )

    # Load image data all as ND arrays
    images = {}
    for channel, path in image_paths.items():
        logging.info(f"Loading {channel} from {path}.")
        try:
            if channel in ["normal"]:
                image = read_16bit(path)
            else:
                image = np.asarray(Image.open(path))

            images[channel] = image
        except ValueError:
            print(
                f"The necessary image channel pass '{channel}' \
                for image '{image_number}' was not found at expeced location {path}"
            )
            raise

    # FIXME: Find a better way to get the W/H
    W, H = np.asarray(list(images.items())[0][1]).shape[:-1]

    # apply downscaling if needed
    if downsample_ratio > 1:
        W, H = W // downsample_ratio, H // downsample_ratio

        downsampled_images = {}
        for channel, image in images.items():
            downsampled_images[channel] = cv2.resize(
                image, (W, H), interpolation=cv2.INTER_NEAREST
            )

        images = downsampled_images

    return W, H, images
