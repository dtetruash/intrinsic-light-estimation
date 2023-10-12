""" This file implements dataloaders for the intrinsics dataset
Currently, this dataloader loads all the relevant data, and then computes a stream
of point-lit pixels without shadowing lit from a single light in the scene.
"""


import json
import logging
from configparser import NoOptionError

import data_loaders.image_loader as il
import data_loaders.olat_render as ro
import numpy as np
import torch
from ile_utils.config import Config
from torch.utils.data import Dataset
from tqdm import tqdm


logging.basicConfig(filename="datasets.log", encoding="utf-8", level=logging.INFO)
# If you wish to log to stdin as well:
# logging.getLogger().addHandler(logging.StreamHandler())


def _get_light_info(config, is_single_olat=False):
    """Get a dict of light names and light lications. If the sigle olat option is set
    the dictionary will only have said light given it is set in the config
    and exists in the scene.

    Args:
        config (utils_ile.config.Config): config object
        is_single_olat (bool): if we should only set the single light's location

    Returns:
        Dict of light names (lower case) to light locations ndarrays (3,)
    """
    light_file_path = config.get("paths", "lights_file")
    with open(light_file_path, "r") as lf:
        lights_info = json.loads(lf.read())

    logging.info(
        f"OLAT dataset: get_lights_info: Loaded {light_file_path}. \
        Number of lights is {len(lights_info['lights'])}"
    )

    def get_location(light):
        return ro.to_translation(np.array(light["transformation_matrix"]))

    light_locations = {
        light["name_light"].lower(): get_location(light)
        for light in lights_info["lights"]
    }

    if is_single_olat:
        # if only a single light is required for the dataset, then find it
        # and override the location dict.
        try:
            single_olat_light = config.get("dataset", "single_olat_light").lower()
        except NoOptionError:
            raise ValueError(
                "If the 'single_olat' dataset option is set, \
            the specification the 'single_olat_light' option is required in the config."
            )
        light_locations = {single_olat_light: light_locations[single_olat_light]}

    return light_locations


def validate_split(config):
    if not config.has_option("paths", "split"):
        logging.info("Split option in config.ini is not set. Using fallback 'train'")
        config.set("paths", "split", "train")
    else:
        split = config.get("paths", "split", fallback="train")
        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Split must be either 'train' or 'val' or 'test' and was {split}"
            )


# TODO: Rename illumination to shading.


# No OLAT needed: Should have the same base class, which just load different channels
# Intrinsic item:       albedo, normal, combined shading, combined image
# DiffuseIntrinsicitem: albedo, normal, combined shading, combined diffuse image
# Intrinsic Attrs:      get_frame(number) -> image or dicr of images
#
# Needs to make OLATS:
# OLAT item:            albedo, normal, OLAT Shading, OLAT Image (for all lights)
# Single OLAT item:     albedo, normal, OLAT Shading, OLAT Image (for one light)
# OLAT Attrs:      get_frame(number, Optional[light]) -> image or dicr of images
#
#
# The attrs function is needed to recostruct images after the fact, for validation
# etc.


def image2stream(image, occupancy_mask):
    """Convert image array to a flat array of just the occupied pixels

    Args:
        image (ndarray): RGBA image array
        occupancy_mask (ndarray ): binary array of the same size as the image array

    Returns:
        A flat ndarray of just the occupied pixels in hte image array.
    """
    return ro.remove_alpha(image)[occupancy_mask]


def gather_image_metadata(config):
    """Gather the meta-data of the scene.

    Args:
        config (utils_ile.config.Config): config dict-like of the run

    Returns:
        Frame transforms dictionary, downsample ratio, number of frames to load.
    """
    with open(config.get("paths", "transforms_file"), "r") as tf:
        frame_transforms = json.loads(tf.read())

    downsample_ratio = int(config.get("parameters", "downsample_ratio"))

    return frame_transforms, downsample_ratio


class IntrinsicDataset(Dataset):
    """Base class for the intrinsics dataset. It will load and deliver the passed
    channels in the oder they are passed in.

    Attributes:
        num_frames: number of frames loaded
        stored_chanels: channels loaded in order of their return by self.__getitem__
    """

    def __init__(self, config, channels):
        # Value checking and validation
        validate_split(config)

        # load information about the scene.
        # Get image transforms_file
        frame_transforms, downsample_ratio = gather_image_metadata(config)
        self.num_frames = len(frame_transforms)

        # 2. Scene images
        # We must always load the depth channel since that is used to determine
        # pixel occupancy
        channels_to_load = list(set(channels + ["depth"]))

        # for each frame in the split
        frames = []
        pixel_streams = {channel_name: [] for channel_name in channels_to_load}
        data_path = config.get("paths", "data_path")
        for frame_number, frame in tqdm(
            enumerate(frame_transforms["frames"]),
            desc=f"Loading dataset from {data_path}",
            total=self.num_frames,
        ):
            logging.info(f"Loading data from frame {frame['file_path']}")

            # get all of the data attrs and load in the image cache
            W, H, images = il.load_frame_channels(
                frame_number,
                channels=channels_to_load,
                data_path=data_path,
                downsample_ratio=downsample_ratio,
            )

            # Assuing that all images have the same dimensions.
            if self.dim is None:
                self.dim = W, H

            # Save the dist for visualization use later.
            frames.append(images)

            # for each channel, extract the occupied pixels and place them in the stream
            occupancy_mask = ro.get_occupancy(images["depth"])

            for image_channel, image in images.items():
                # NOTE: We might want the depth buffer later on. Could include
                # an option for it later.
                if image_channel == "depth":
                    continue
                pixel_stream = image2stream(image, occupancy_mask)
                pixel_streams[image_channel].append(pixel_stream)

        # Concatenate all pixel streams and form the feature tensor
        # combine loaded steams in the order given by the channels input
        self._feats = torch.stack(
            [
                torch.as_tensor(np.concatenate(pixel_streams[channel]))
                for channel in channels
            ],
            dim=1,
        ).float()

        # Tracking for documentation and meta-data
        self.chanels = channels
        self._decompositions = pixel_streams
        self._frames = frames

        def __getitem__(self, idx):
            return self._feats[idx]

        def __len__(self):
            return self._len

        def get_frame_images(self, frame_number):
            """Get the loaded frame's channels by its number.

            Args:
            frame_number (int): number of the frame whose image dict to get

            Returns:
                Dictionary of unaltered image channels of the frame
            """
            return self.frames[frame_number]

        def get_frame_decomposition(frame_number):
            """Get the loaded frame's channels as flattened data streams. Useful
            for recombining with infered intrinsic attributes like lighting.

            Args:
                frame_number (int): number of the frame whose stream dict to get

            Returns:
                Dictionary of flattened (only occupied) pixel streams for the frame
            """
            return {
                channel: stream_list[frame_number]
                for channel, stream_list in self._decompositions.items()
            }


class IntrinsicGlobalDataset(IntrinsicDataset):
    """Torch dataset for the Intrinsic Dataset.
    Items loaded are in the form: [full, albedo, shading, normal]
    """

    def __init__(self, config):
        super().__init__(config, ["full", "albedo", "shading", "normal"])


class IntrinsicDiffuseDataset(IntrinsicDataset):
    """Torch dataset for the Intrinsic Dataset.
    Items loaded are in the form: [diffuse, albedo, shading, normal]
    """

    def __init__(self, config):
        super().__init__(config, ["diffuse", "albedo", "shading", "normal"])


class OLATDataset(Dataset):
    def __init__(self, config, is_single_olat=False):
        validate_split(config)

        # init empty containers
        # These are used for the constuction of the overall Tensor at the end
        # of this method
        # NOTE: In the other dataset this is a dict of lists.
        world_normals_list = []
        aldedo_list = []
        olat_pixelstream_list = []
        target_list = []
        occupancy_list = []
        end_indexes = []
        self._len = 0

        # Is this a single OLAT dataset?
        self._is_single_OLAT = is_single_olat

        # Get image transforms_file
        frame_transforms, downsample_ratio = gather_image_metadata(config)

        self.num_frames = len(frame_transforms["frames"])

        # Get light transform file
        light_locations = _get_light_info(config, self._is_single_OLAT)
        logging.debug(f"lights were: {light_locations}")
        if self._is_single_OLAT:
            logging.debug("Only one light used since single_olat_light is set.")

        self._lights_info = light_locations
        self._num_lights = len(light_locations)

        data_path = config.get("paths", "data_path")

        # List of dicts of {albedo, normal}
        self._frame_attributes = []
        # List of dicts of {albedo stream, normal stream}
        self._attribute_pixelstreams = []

        # Dict of lists of dicts ([light][frane]{shading, image})
        self._frame_olats = {light: [] for light in light_locations}
        # Dict of lists of dicts ([light][frane]{shading stream, image stream})
        self._olat_pixelstreams = {light: [] for light in light_locations}

        for frame_number, frame in tqdm(
            enumerate(frame_transforms["frames"]),
            desc=f"Loading dataset from {data_path}",
            total=self.num_frames,
        ):
            logging.info(f"Loading data from image {frame['file_path']}")

            # BEGIN COMPUTING OLAT:

            # STEP 1: Load the components
            # get all of the data attrs and load in the image cache
            (
                W,
                H,
                images,
                albedo,
                world_normals,
                posed_points,
                occupancy_mask,
            ) = ro.gather_intrinsic_components(
                frame_number, frame_transforms, downsample_ratio=downsample_ratio
            )

            self._frame_attributes.append(images)
            self._attribute_pixelstreams.append(
                {
                    "albedo": albedo,
                    "normal": world_normals,
                }
            )

            num_samples_per_light = posed_points.shape[0]

            # STEP 2: Compute the OLAT samples
            # Also this collects the attribute per image for reconstructions.
            for light in light_locations:
                # create OLAT images of pixels for the loaded image
                light_loc = light_locations[light]
                (
                    olat_image_pixels,
                    olat_shading_pixels,
                    (light_vectors, _),
                    _,
                ) = ro.render_OLAT_pixelstream(
                    world_normals,
                    albedo,
                    posed_points,
                    light_loc,
                    return_light_vectors=True,
                    return_shading=True,
                )
                olat_pixelstream_list += [olat_image_pixels]

                # set the target. Keep consistent with inputs
                # must be computed per pixel using its posed_point
                target_list += [light_vectors]

                self._olat_pixelstreams[light].append(
                    {"shading": olat_shading_pixels, "image": olat_image_pixels}
                )

                shading_image = ro.reconstruct_image(
                    W, H, olat_shading_pixels, occupancy_mask, add_alpha=True
                )
                olat_image = ro.reconstruct_image(
                    W, H, olat_image_pixels, occupancy_mask, add_alpha=True
                )
                self._frame_olats[light].append(
                    {"shading": shading_image, "image": olat_image}
                )

                # break out of the loop, since the chosen light is first in order
                if self._is_single_OLAT:
                    break

            # save output
            added_samples = num_samples_per_light * self._num_lights
            logging.info(
                f"Appending {num_samples_per_light} samples \
                from image {frame_number} per light."
            )
            logging.info(
                f"Appending {added_samples} samples \
                from image {frame_number}."
            )

            # append to list, we will concat later
            world_normals_list += [world_normals] * self._num_lights
            aldedo_list += [albedo] * self._num_lights
            occupancy_list += [occupancy_mask] * self._num_lights

            # compute loaded lengths
            self._len += added_samples

            logging.info(
                f"New number of samples \
                after loading image {frame_number} is {self._len}"
            )

        # concat the outputs and make them tensors
        self._world_normals = torch.as_tensor(np.concatenate(world_normals_list))
        self._albedo = torch.as_tensor(np.concatenate(aldedo_list))
        self._raster_images = torch.as_tensor(np.concatenate(olat_pixelstream_list))

        self.feats = torch.stack(
            [self._world_normals, self._albedo, self._raster_images], dim=1
        ).float()
        self.target = torch.as_tensor(np.concatenate(target_list)).float()

        self._occupancy_mask = np.stack(occupancy_list)  # TODO: Remove after debugging
        self._end_indexes = end_indexes

        logging.debug(f"Final number of samples is {self._len}")
        logging.debug(f"World normals array has shape {self._world_normals.shape}")
        logging.debug(f"Albedo array has shape {self._albedo.shape}")
        logging.debug(f"Raster image array has shape {self._raster_images.shape}")
        logging.debug(f"Target array has shape {self.target.shape}")

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        """Get the features of an image pixel as well as the direction from where it is
        lit form by the index within its image's occupied pixels
        """
        # NOTE: Leaving target vectors for back-compatability
        feats = self.feats[index]
        target = self.target[index]
        return feats, target

    def _check_light_name(self, light_name):
        if self._is_single_OLAT:
            if light_name is not None:
                raise ValueError(
                    "A single OLAT sample dataset does not contain \
                    light name information. Call with light_name=None"
                )
            light_name = list(self._lights_info.keys())[0]

        assert (
            light_name in self._lights_info.keys()
        ), f"Light with name {light_name} is not loaded in _light_info"

        return light_name

    def get_frame_images(self, frame_number, light_name=None):
        """Get the loaded frame's channels by its number and the light used
        to render them.

        Args:
            frame_number (int): number of the frame whose image dict to get
            light_name (str): name of the light to use.
                If None, assumed to be the only possible light

        Returns:
            Dictionary of unaltered image channels of the frame
        """
        light_name = self._check_light_name(light_name)
        frame_attrs = self._frame_attributes[frame_number]
        frame_olat = self._frame_olats[light_name][frame_number]

        return frame_olat | frame_attrs

    def get_frame_decomposition(self, frame_number, light_name=None):
        """Get the loaded frame's channels as flattened data streams. Useful
        for recombining with infered intrinsic attributes like lighting.

        Args:
            frame_number (int): number of the frame whose stream dict to get
            light_name (str): name of the light to use.
                If None, assumed to be the only possible light

        Returns:
            Dictionary of flattened (only occupied) pixel streams for the frame
        """
        light_name = self._check_light_name(light_name)
        attr_pixels = self._attribute_pixelstreams[frame_number]
        olat_pixels = self._olat_pixelstreams[light_name][frame_number]
        return attr_pixels | olat_pixels


if __name__ == "__main__":
    # TODO: Test by creating all datasets, and reconstructing a random frame from it
    # via the stream and via the loaded frames themselves?
    config = Config.get_config()
    ds = OLATDataset(config)
    print(f"Loaded dataset has {len(ds)} samples.")
