from matplotlib import pyplot as plt
import logging
from torch.utils.data import Dataset
import json
import numpy as np
import raster_relight as rr

logging.basicConfig(filename='raster_dataloader.log', encoding='utf-8', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

def get_light_info(config):
    with open(config['paths']['lights_file'], 'r') as lf:
        lights_info = json.loads(lf.read())

    print(f"Loaded {config['paths']['lights_file']}. Number of lights is {len(lights_info['lights'])}")

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
    def __init__(self):
        config = rr.parse_config()

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

        # get light location
        # TODO: Specify the light name through the config
        # TODO: loop over all light locations to get all OLAT samples for training
        light_locations = get_light_info(config)
        logging.debug(f"lights were: {light_locations}")
        # light_name = config['lighting']['light_name']
        # light_loc = light_locations[light_name]
        self._num_lights = len(light_locations)

        for i, frame in enumerate(image_transforms['frames']):
            logging.info(f"Loading data from image {frame['file_path']}")

            image_number = i

            # get all of the data attrs and load in the image cache
            W, H, image_albedo, normal_pixels, depth, rgbd, occupancy_mask = rr.load_images(image_number)

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

            for light in light_locations:
                # create raster images of pixels for the loaded image
                light_loc = light_locations[light]
                raster_image_pixels, _, _ = rr.compute_raster(world_normals, albedo, posed_points, light_loc, T)
                raster_images_list += [raster_image_pixels]

                # set the target. Keep consistent with inputs
                target_list += [np.broadcast_to(light_loc, (num_samples_per_light, 3))]

            # view raster for sanity:
            # TODO: remove after testing
            # raster_save = np.ones((W, H, 3))
            # raster_save[self._occupancy_mask] = raster_image_pixels
            # plt.imsave(f'sanity/{i}_{light_name}.png', raster_save)

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

        # concat the outputs
        self._world_normals = np.concatenate(world_normals_list)
        self._albedo = np.concatenate(aldedo_list)
        self._raster_images = np.concatenate(raster_images_list)
        self.target = np.concatenate(target_list)

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
        return self._world_normals[index], self._aldedo[index], self._raster_images[index], self.target[index]


if __name__ == "__main__":
    ds = RasterDataset()
    print(f"Loaded dataset has {len(ds)} samples.")
