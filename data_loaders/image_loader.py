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
def format_image_path(frame_number, data_path=None, channel="", light=""):
    """Construct the correct image path for an image in the intrinsic dataset
    given a frame number and the channel to find the path to.

    Args:
        frame_number (int): the number of the frame
        data_path (str or pathlike): the absolute base path to all images
        channel (str): the name of the channel (e.g., normal, albedo, depth...)
        light (str): the name of the light used to generate the image if any

    Returns:
        The string path to the image with given parameters.
    """
    if data_path is None:
        data_path = config.get("paths", "data_path")

    img_name = f"{data_path}/r_{frame_number:03d}"
    img_name += f"_{channel}" if channel else ""
    img_name += f"_{light}" if light else ""
    return img_name + ".png"


# FIXME: Remove the lights info from the paths, or see if they are still needed.
def get_image_paths(frame_number, data_path=None, channels=[], lighting=[]):
    """Get paths to images of specified channels for a given frame.

    Args:
        frame_number (int): the number of the frame
        data_path (str or pathlike): the absolute base path to all images
        channels (List[str]): the names of the channels (e.g., normal, albedo, depth...)
        lighting (List[str]): the names of lights used to generate the images if any

    Returns:
        List of string paths to the images with given parameters.
    """

    # get combined (fully formed) image names
    # in combined images which are given by lighting
    image_paths = {
        (channel := lt): format_image_path(frame_number, data_path, channel)
        for lt in lighting
    }

    # Full channel is the name for the full render of the frame
    # it has no sufix in the datset, and therefore must be treated differently.
    full_channel_name = "full"
    if full_channel_name in channels:
        image_paths[full_channel_name] = format_image_path(frame_number, data_path, "")

    # read in other channels beside full
    image_paths.update(
        {
            channel: format_image_path(frame_number, data_path, channel)
            for channel in channels
            if channel != full_channel_name
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


def resample_image(image_array, new_W, new_H):
    """Resample image to new dimentions without inrerpolation."""
    return cv2.resize(image_array, (new_W, new_H), interpolation=cv2.INTER_NEAREST)


def load_frame_channels(frame_number, channels, data_path=None, downsample_ratio=1):
    """Load the given channels of a frame as image arrays

    Args:
        frame_number (int): the number of the frame
        channels (List[str]): the names of the channels (e.g., normal, albedo, depth...)
        data_path (str or pathlike): the absolute base path to all images
        downsample_ratio (int): Positive downsampling factor

    Returns:
        (image width, image height, a channel-name indexed dict of image arrays)
    """
    image_paths = get_image_paths(
        frame_number,
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
                for image '{frame_number}' was not found at expeced location {path}"
            )
            raise

    # FIXME: Find a better way to get the W/H
    W, H = np.asarray(list(images.items())[0][1]).shape[:-1]

    # apply downscaling if needed
    if downsample_ratio > 1:
        W, H = W // downsample_ratio, H // downsample_ratio

        downsampled_images = {}
        for channel, image in images.items():
            downsampled_images[channel] = resample_image(image, W, H)
        images = downsampled_images

    return W, H, images
