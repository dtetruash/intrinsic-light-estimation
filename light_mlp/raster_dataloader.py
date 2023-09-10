import logging
import torch
from torch.utils.data import Dataset
import json
import numpy as np
from tqdm import tqdm

import raster_relight as rr

logging.basicConfig(filename='raster_dataloader.log', encoding='utf-8', level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler())

def get_light_info(config):
    with open(config['paths']['lights_file'], 'r') as lf:
        lights_info = json.loads(lf.read())

    logging.info(f"raster dataloader: get_lights_info: Loaded {config['paths']['lights_file']}. Number of lights is {len(lights_info['lights'])}")

    light_transforms = {light['name_light'].lower(): np.array(light['transformation_matrix']) for light in lights_info['lights']}
    light_locations = {name: rr.to_translation(transform) for (name, transform) in light_transforms.items()}

    return light_locations


# +
# The dataset should be loading the given images, and flattening these to those pixels which are occupied
# We end up with an array of pixel features [normal, albedo, image]
# note that depth is needed for internal funcitons

# The target tensor is then whatever the image feature was rastered from
# We can achieve this by rastering using onlye a sinlge given light direction for now (the one we have, since later we will not be doing rastering)

class RasterDataset(Dataset):
    def __init__(self, split='train'):
        config = rr.parse_config()

        if split:
            config['paths']['split'] = split
        else:
            raise ValueError(f"Split must be either 'train' or 'val' or 'test'.")

        # init empty containers
        world_normals_list = []
        aldedo_list = []
        raster_images_list = []
        target_list = []
        occupancy_list = []
        end_indexes = []
        self._len = 0

        # Get image transforms_file
        with open(config['paths']['transforms_file'], 'r') as tf:
            image_transforms = json.loads(tf.read())

        camera_angle_x = image_transforms["camera_angle_x"]

        downsample_ratio = int(config['parameters']['downsample_ratio'])

        light_locations = get_light_info(config)
        logging.debug(f"lights were: {light_locations}")

        self._lights_info = light_locations
        self._num_lights = len(light_locations)

        self.num_frames = len(image_transforms['frames'])

        self.attributes = []

        for i, frame in tqdm(enumerate(image_transforms['frames']),
                             desc=f"Loading dataset from {config['paths']['scene']}/{config['paths']['split']}", total=self.num_frames):

            logging.info(f"Loading data from image {frame['file_path']}")

            image_number = i

            # get all of the data attrs and load in the image cache
            W, H, image_albedo, normal_pixels, depth, rgbd, occupancy_mask = rr.load_images(image_number, downsample_ratio=downsample_ratio)

            logging.debug(f"Normal pixels range from {normal_pixels.min()} to {normal_pixels.max()}")

            # load image parameters
            intrinsics, c2w, R, T = rr.get_camera_parameters(W, H,
                                                             camera_angle_x,
                                                             image_number,
                                                             image_transforms)

            self._camera_center = T

            # Transform notmal pixel valuse from pixel values to world normals
            logging.debug(f"Normals pixels shape: {normal_pixels.shape}")
            camera_normals = rr.get_camera_space_normals(normal_pixels)
            world_normals = rr.get_world_space_normals(camera_normals, R)

            # project depth to get 3d coords
            # For now, and testing, we save the full point cloud object
            posed_points = rr.project_and_pose_3d_points_via_rgbd(rgbd, intrinsics, c2w)
            num_samples_per_light = posed_points.shape[0]

            # For reconstruction purposes
            if len(end_indexes) == 0:
                end_indexes.append(num_samples_per_light)
            else:
                end_indexes.append(end_indexes[-1] + num_samples_per_light)

            # normalize albedo
            albedo = image_albedo / 255.0

            image_attributes = {}
            for light in light_locations:
                # create raster images of pixels for the loaded image
                light_loc = light_locations[light]
                raster_image_pixels, _, _ = rr.compute_raster(world_normals, albedo, posed_points, light_loc, T)
                raster_images_list += [raster_image_pixels]

                # set the target. Keep consistent with inputs
                target_list += [np.broadcast_to(light_loc, (num_samples_per_light, 3))]

                # add attributed and raster for later use
                image_attributes[light] = (W, H, raster_image_pixels, world_normals, albedo, occupancy_mask)

            self.attributes.append(image_attributes)

            # save output
            added_samples = num_samples_per_light * self._num_lights
            logging.info(f"Appending {num_samples_per_light} samples from image {i} per light.")
            logging.info(f"Appending {added_samples} samples from image {i}.")

            # append to list, we will concat later
            world_normals_list += [world_normals] * self._num_lights
            aldedo_list += [albedo] * self._num_lights
            occupancy_list += [occupancy_mask] * self._num_lights

            # compute loaded lengths
            self._len += added_samples

            logging.info(f"New number of samples after loading image {i} is {self._len}")

        # concat the outputs and make them tensors
        self._world_normals = torch.as_tensor(np.concatenate(world_normals_list))
        self._albedo = torch.as_tensor(np.concatenate(aldedo_list))
        self._raster_images = torch.as_tensor(np.concatenate(raster_images_list))

        self.feats = torch.stack([self._world_normals, self._albedo, self._raster_images], dim=1).float()
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
        """Get the features of an image pixel as well as the direction from where it is lit form by the index within its image's occupied pixels"""
        # feats = torch.cat([self._world_normals[index], self._albedo[index], self._raster_images[index]])
        feats = self.feats[index]
        target = self.target[index]
        # print(F"DEBUG: feats_concat: {self.feats[index]}, {self.feats[index].shape}")
        # print(F"DEBUG: feats: {feats}, {feats.shape}")
        # print(f'DEBUG: target: {target.shape}')
        return feats, target


if __name__ == "__main__":
    ds = RasterDataset()
    print(f"Loaded dataset has {len(ds)} samples.")
