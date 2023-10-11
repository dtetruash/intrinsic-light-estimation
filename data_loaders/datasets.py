""" This file implements dataloaders for the intrinsics dataset
Currently, this dataloader loads all the relevant data, and then computes a stream
of point-lit pixels without shadowing lit from a single light in the scene.
"""


import logging
import torch
from torch.utils.data import Dataset
import json
import numpy as np
from tqdm import tqdm

from configparser import NoOptionError

import data_loaders.image_loader as il
from ile_utils.config import Config

import data_loaders.raster_relight as rr

config = Config.get_config()

logging.basicConfig(filename="datasets.log", encoding="utf-8", level=logging.INFO)
# If you wish to log to stdin as well:
# logging.getLogger().addHandler(logging.StreamHandler())


def _get_light_info(config):
    light_file_path = config.get("paths", "lights_file")
    with open(light_file_path, "r") as lf:
        lights_info = json.loads(lf.read())

    logging.info(
        f"raster dataloader: get_lights_info: Loaded {light_file_path}. \
        Number of lights is {len(lights_info['lights'])}"
    )

    light_transforms = {
        light["name_light"].lower(): np.array(light["transformation_matrix"])
        for light in lights_info["lights"]
    }
    light_locations = {
        name: rr.to_translation(transform)
        for (name, transform) in light_transforms.items()
    }

    return light_locations


def validate_split(config):
    if config.has_option("paths", "split"):
        config.set("paths", "split", "train")
        split = "train"

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
    return rr.remove_alpha(image)[occupancy_mask]


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
        self.frames = []
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

            # Save the dist for visualization use later.
            self.frames.append(images)

            # for each channel, extract the occupied pixels and place them in the stream
            rr.get_occupancy(images["depth"])

            for image_channel, image in images.items():
                if image_channel == "depth":
                    continue
                pixel_stream = image2stream(image)
                pixel_streams[image_channel].append(pixel_stream)

        # Concatenate all pixel streams and form the feature tensor
        for image_channel in pixel_streams:
            streams = pixel_streams[image_channel]
            pixel_streams[image_channel] = torch.as_tensor(np.concatenate(streams))

        # combine loaded steams in the order given by the channels input
        self._feats = torch.stack(
            [pixel_streams[channel] for channel in channels], dim=1
        ).float()

        # Tracking for documentation and meta-data
        self.chanels = channels
        self.dim = (W, H)  # Assuing that all images have the same dimensions.

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
            return self.decomposition[frame_number]


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
    def __init__(self, split="train"):
        validate_split(config)

        # init empty containers
        # These are used for the constuction of the overall Tensor at the end
        # of this method
        world_normals_list = []
        aldedo_list = []
        raster_images_list = []
        target_list = []
        occupancy_list = []
        end_indexes = []
        self._len = 0

        # Is this a single OLAT dataset?
        # NOTE: Could also do this by asking if single_olat_light is set?
        self._is_single_OLAT = config.getboolean(
            "dataset", "single_olat", fallback=False
        )
        if self._is_single_OLAT:
            try:
                self._single_OLAT_light = config.get("dataset", "single_olat_light")
            except NoOptionError as e:
                raise type(e)(
                    e.message
                    + " If the 'single_olat' dataset option is set, \
                the specification the 'single_olat_light' option is required."
                )

        # Get image transforms_file
        frame_transforms, downsample_ratio = gather_image_metadata(config)

        self.num_frames = len(frame_transforms["frames"])
        frame_transforms["camera_angle_x"]

        light_locations = _get_light_info(config)
        logging.debug(f"lights were: {light_locations}")

        self._lights_info = light_locations
        self._num_lights = len(light_locations)

        # An attribute which collects dicts of tuples of tensors accosiated
        # with each frame and light
        self.attributes = []

        data_path = config.get("paths", "data_path")

        # The structure of this code block should be:
        # for frame in set:
        #       Load those attrs which are needed to compute OLAT
        #           get_intrinsic_components() -> albedo, normals, depth, posed_points
        #       Compute the OLAT (per light or for one given light)
        #           compute_olat_for_frame(components, light_info) -> [olat_smaples]
        #       Record the arrts and the OLAT

        for image_number, frame in tqdm(
            enumerate(frame_transforms["frames"]),
            desc=f"Loading dataset from {data_path}",
            total=self.num_frames,
        ):
            logging.info(f"Loading data from image {frame['file_path']}")

            # get all of the data attrs and load in the image cache
            (
                W,
                H,
                albedo,
                world_normals,
                posed_points,
                occupancy_mask,
            ) = rr.gather_intrinsic_components(
                image_number, downsample_ratio=downsample_ratio
            )

            # TODO: Call rr.compute_OLAT_pixels here

            # BEGIN COMPUTING OLAT:

            # STEP 1: Load the components

            num_samples_per_light = posed_points.shape[0]

            # For reconstruction purposes
            # Used to see where in the stream the content of each frame begins
            # and ends. More favourable solution than the self._attributes
            # structure.
            # FIXME: Remove the image_attributes section, and implement a function
            # which loads the attrs from via __get_item__ and end_idxs
            # + the shared data like width and height.
            if len(end_indexes) == 0:
                end_indexes.append(num_samples_per_light)
            else:
                end_indexes.append(end_indexes[-1] + num_samples_per_light)

            # TODO: Add switch to either load a single OLAT or all of them
            if self._is_single_OLAT:
                # don't do for loop and don't off-set but number of lights
                raise NotImplementedError()

            # STEP 2: Compute the OLAT samples
            # Also this collects the attribute per image for reconstructions.
            image_attributes = {}
            for light in light_locations:
                # create raster images of pixels for the loaded image
                light_loc = light_locations[light]
                (
                    raster_image_pixels,
                    (light_vectors, _),
                    _,
                ) = rr.compute_OLAT_pixelstream(
                    world_normals, albedo, posed_points, light_loc, return_norms=True
                )
                raster_images_list += [raster_image_pixels]

                # set the target. Keep consistent with inputs
                # must be computed per pixel using its posed_point
                target_list += [light_vectors]

                # add attributed and raster for later use
                image_attributes[light] = (
                    W,
                    H,
                    raster_image_pixels,
                    world_normals,
                    albedo,
                    posed_points,
                    light_vectors,
                    occupancy_mask,
                )

            # TODO: Add interface to this to change everywhere else and abstract from
            # the use do the raw atributes structure.
            self.attributes.append(image_attributes)

            # save output
            added_samples = num_samples_per_light * self._num_lights
            logging.info(
                f"Appending {num_samples_per_light} samples \
                from image {image_number} per light."
            )
            logging.info(
                f"Appending {added_samples} samples \
                from image {image_number}."
            )

            # append to list, we will concat later
            world_normals_list += [world_normals] * self._num_lights
            aldedo_list += [albedo] * self._num_lights
            occupancy_list += [occupancy_mask] * self._num_lights

            # compute loaded lengths
            self._len += added_samples

            logging.info(
                f"New number of samples \
                after loading image {image_number} is {self._len}"
            )

        # concat the outputs and make them tensors
        self._world_normals = torch.as_tensor(np.concatenate(world_normals_list))
        self._albedo = torch.as_tensor(np.concatenate(aldedo_list))
        self._raster_images = torch.as_tensor(np.concatenate(raster_images_list))

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
        feats = self.feats[index]

        if self.target:
            target = self.target[index]
            return feats, target
        else:
            return feats

    def get_frame_attributes(self, frame_number, light_name=None):
        """Get the attributes of a given frame, and optionally under a given light.

        Args:
            frame_number (int): frame whose attributes to get
            light_name (str): the name identifier of the light_name
                under which the image should be reperoduced.

        Raises:
            ValueError: if the dataset does not record per light information
                while a light name is given.
        """
        raise NotImplementedError()

        if self.single_olat and light_name is not None:
            raise ValueError(
                "A single OLAT sample dataset does not contain light name information."
            )

        def index_into_frame(frame_number):
            pass

        def index_into_light(light_name, frame_attributes):
            pass

        frame_attributes = index_into_frame(frame_number)
        if not light_name:
            return frame_attributes

        return index_into_light(light_name, frame_attributes)


if __name__ == "__main__":
    ds = OLATDataset()
    print(f"Loaded dataset has {len(ds)} samples.")
