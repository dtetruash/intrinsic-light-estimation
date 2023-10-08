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


class IntrinsicDataset(Dataset):
    """Base class for the intrinsic dataset.
    This should have the logic of loading images from
    the dataset directory, placing these into the appropriate datastructures,
    and being fexible to deliver the needed by the dirived classes data.

    - Load channels from config, or load all channels
    - have a __getitem__ which can change based on which things are belign loaded?
    """

    def __init__(self, config=None):
        # Value checking and validation
        validate_split(config)

        # load information about the scene.
        # 1. Scene metadata

        # Get image transforms_file
        with open(config.get("paths", "transforms_file"), "r") as tf:
            image_transforms = json.loads(tf.read())

        downsample_ratio = int(config.get("parameters", "downsample_ratio"))

        light_locations = _get_light_info(config)
        logging.debug(f"lights were: {light_locations}")

        self._lights_info = light_locations
        self._num_lights = len(light_locations)

        self.num_frames = len(image_transforms["frames"])

        # 2. Scene images
        # We must always load the depth channel since that is used to determine
        # NOTE: We may need to explicitly load the comnined channel as well.
        # pixel occupancy
        channels_to_load = config.get("images", "channels") + ["depth"]

        # for each frame in the split
        data_path = config.get("paths", "data_path")
        for image_number, frame in tqdm(
            enumerate(image_transforms["frames"]),
            desc=f"Loading dataset from {data_path}",
            total=self.num_frames,
        ):
            logging.info(f"Loading data from image {frame['file_path']}")

            # get all of the data attrs and load in the image cache
            W, H, images = rr.load_image_channels(
                image_number,
                channels=channels_to_load,
                data_path=data_path,
                downsample_ratio=downsample_ratio,
            )

            # for each channel, extract the occupied pixels and place them in the stream
            occupancy_mask = rr.get_occupancy(images["depth"])

            pixel_streams = {channel_name: [] for channel_name in images.keys()}
            for image_channel in images:
                if image_channel == "depth":
                    continue
                no_alpha_image = images[image_channel][..., :-1]
                pixel_streams[image_channel].append(no_alpha_image[occupancy_mask])

        # Concatenate all pixel streams and form the feature tensor
        for image_channel in pixel_streams:
            streams = pixel_streams[image_channel]
            pixel_streams[image_channel] = torch.as_tensor(np.concatenate(streams))

        # combine loaded steams in alphabetical order into an (Pixels, Channels) tensor
        stored_channels = sorted(pixel_streams)
        self._feats = torch.stack(
            [pixel_streams[channel] for channel in stored_channels], dim=1
        ).float()

        # Tracking for documentation and meta-data
        self.stored_chanels = stored_channels

        # TODO:
        # If we need to compute OLAT samples here if set
        # Ensure that all the correct channels are loaded then.
        # Maybe should see if that is the case first.

        # There should be a config option to set the target vector.
        # Target can be:
        #   Image pixel
        #   Global Shading (maybe)

        def __getitem__(self, idx):
            return self._feats[idx]

        def __len__(self):
            return self._len


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
        self._is_single_OLAT = config.get("dataset", "single_olat", fallback=False)
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
        # TODO: Move to a function as this is used in all datasets
        with open(config.get("paths", "transforms_file"), "r") as tf:
            image_transforms = json.loads(tf.read())

        downsample_ratio = int(config.get("parameters", "downsample_ratio"))

        light_locations = _get_light_info(config)
        logging.debug(f"lights were: {light_locations}")

        camera_angle_x = image_transforms["camera_angle_x"]

        self._lights_info = light_locations
        self._num_lights = len(light_locations)

        self.num_frames = len(image_transforms["frames"])

        # An attribute which collects dicts of tuples of tensors accosiated
        # with each frame and light
        self.attributes = []

        data_path = config.get("paths", "data_path")

        # The structure of this code block should be:
        # for frame in set:
        #       Load those attrs which are needed to compute OLAT
        #           get_frame_components() -> albedo, normals, depth, posed_points
        #       Compute the OLAT (per light or for one given light)
        #           compute_olat_for_frame(components, light_info) -> [olat_smaples]
        #       Record the arrts and the OLAT

        for image_number, frame in tqdm(
            enumerate(image_transforms["frames"]),
            desc=f"Loading dataset from {data_path}",
            total=self.num_frames,
        ):
            logging.info(f"Loading data from image {frame['file_path']}")

            # get all of the data attrs and load in the image cache
            (
                W,
                H,
                image_albedo,
                normal_pixels,
                depth,
                posed_points,
                occupancy_mask,
            ) = rr.gather_intrinsic_components(
                image_number, downsample_ratio=downsample_ratio
            )

            # TODO: Call rr.compute_OLAT_pixels here

            # BEGIN COMPUTING OLAT:

            # STEP 1: Load the components

            # load image parameters
            intrinsics, c2w, R, T = rr.get_camera_parameters(
                W, H, camera_angle_x, image_number, image_transforms
            )

            # Transform notmal pixel valuse from pixel values to world normals
            logging.debug(f"Normals pixels shape: {normal_pixels.shape}")
            camera_normals = rr.get_camera_space_normals_from_pixels(normal_pixels)
            world_normals = rr.to_world_space_normals(camera_normals, R)

            # project depth to get 3d coords
            # For now, and testing, we save the full point cloud object
            posed_points = rr.project_and_pose_3d_points_via_rgbd(rgbd, intrinsics, c2w)
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

            # normalize albedo
            albedo = image_albedo / 255.0

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
                raster_image_pixels, (light_vectors, _), _ = rr.compute_OLAT_pixels(
                    world_normals, albedo, posed_points, light_loc, T
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
        target = self.target[index]
        # print(F"DEBUG: feats_concat: {self.feats[index]}, {self.feats[index].shape}")
        # print(F"DEBUG: feats: {feats}, {feats.shape}")
        # print(f'DEBUG: target: {target.shape}')
        return feats, target


if __name__ == "__main__":
    ds = OLATDataset()
    print(f"Loaded dataset has {len(ds)} samples.")
