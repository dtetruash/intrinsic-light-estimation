from pprint import pprint
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
import raster_relight as rr
import open3d as o3d


def load_images(image_number, depth_scale=1/8, depth_trunc=8):
    """ Load images into (N, ...) ndarrays of needed dtype of only occupied pixels.
    Return a tuple of ndarrays with flattened and occupied pixel attributes and information and handful of specialized formats
    Also return an rbd image for later use in projecting depth.
    """
    image_paths = rr.get_image_paths(image_number, ['normal', 'albedo', 'depth'])

    # Load image data
    images = {}
    for channel, path in image_paths.items():
        print(f"loading {channel} from {path}")
        try:
            if channel in ['normal']:
                image = rr.read_16bit(path)
            else:
                image = np.asarray(Image.open(path))

            images[channel] = image
        except ValueError:
            print(f"The necessary image channel pass '{channel}' for image '{image_number}' was not found at expeced location {path}")
            raise

    # Normal
    image_normal = images['normal'][..., :-1]

    # albdo
    image_albedo = images['albedo'][..., :-1]

    # get only the alpha from the depth image
    depth_alpha = images['depth'][..., -1].astype(np.uint8)

    # invert the depth image to be black -> white as depth increases
    # TODO: upgrade depth images to 16-bit
    depth_remapped = rr.remap_depth_black2white(depth_alpha)

    # Make RGBD image intput
    image_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(images['combined'][..., :-1]),
        o3d.geometry.Image(depth_remapped),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False)

    # FIXME: Find a better way to get the W/H
    W, H = np.asarray(list(images.items())[0][1]).shape[:-1]

    # get image occupancy
    occupancy_mask = rr.get_occupancy(depth_alpha)
    print(f"Occumpancy mask of shape {occupancy_mask.shape}, and with {occupancy_mask.sum()} occupied pixels.")

    # return image data (minux alpha) for only occupied pixels for further processing

    return W, H, image_albedo[occupancy_mask], image_normal[occupancy_mask], depth_remapped[occupancy_mask], image_rgbd, occupancy_mask


# +
#
# get the light location per scene
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

        # TOOD: make the image_number range
        image_number = 0
        # get all of the data attrs and load in the image cache
        W, H, albedo, normal_pixels, depth, rgbd, self._occupancy_mask = load_images(image_number)

        print(f"Normal pixels range from {normal_pixels.min()} to {normal_pixels.max()}")

        img_path = 'dl_img'

        albedo_img = np.ones((800, 800, 3)) * 255
        albedo_img[self._occupancy_mask] = albedo
        plt.imsave(img_path+"/albedo.png", albedo_img.astype(np.uint8))

        normal_img = np.ones((800, 800, 3)) * 255
        normal_img[self._occupancy_mask] = normal_pixels
        plt.imsave(img_path+"/normal.png", normal_img / (2**16 - 1))

        depth_img = np.ones((800, 800, 3)) * 255
        depth_img[self._occupancy_mask] = depth[..., np.newaxis]
        plt.imsave(img_path+"/depth.png", depth_img.astype(np.uint8))

        with open(config['paths']['transforms_file'], 'r') as tf:
            image_transforms = json.loads(tf.read())

        # load image parameters
        intrinsics, c2w, R, T = rr.get_camera_parameters(W, H,
                                                         image_transforms["camera_angle_x"],
                                                         image_number,
                                                         image_transforms)

        # Transform notmal pixel valuse from pixel values to world normals
        print(f"Normals pixels shape: {normal_pixels.shape}")
        camera_normals = rr.get_camera_space_normals(normal_pixels)

        camera_normals_img = np.ones((800, 800, 3)) * 255
        camera_normals_img[self._occupancy_mask] = (camera_normals + 1) / 2 * 255
        plt.imsave(img_path+"/camera.png", camera_normals_img.astype(np.uint8))

        world_normals = rr.get_world_space_normals(camera_normals, R)

        world_normals_img = np.ones((800, 800, 3)) * 255
        world_normals_img[self._occupancy_mask] = (world_normals + 1) / 2 * 255
        plt.imsave(img_path+"/world.png", world_normals_img.astype(np.uint8))

        # project depth to get 3d coords
        # For now, and testing, we save the full point cloud object
        # FIXME: Posed points should not be dense and should give occupied pixels only
        # TODO: Remove full point cloud being stored here
        pcd = rr.project_and_pose_3d_points_via_rgbd(rgbd, intrinsics, c2w, return_array=False)
        posed_points = np.asarray(pcd.points)
        print(f"posed_points returned by rr has shape {posed_points.shape}")
        posed_points = posed_points.reshape((800, 800, 3))[self._occupancy_mask]  # FIXME: Here I could equiv. flatten the mask to index the posed_points

        pcd.points = o3d.utility.Vector3dVector(posed_points)
        pcd.normals = o3d.utility.Vector3dVector(world_normals)
        self._pcd = pcd

        # get light location
        # TODO: make samples depenent on which light we use.
        # TODO: Iterate through OLAT samples
        light_locations = get_light_info(config)
        light_loc = list(light_locations.values())[0]

        # create raster images of pixels
        # TODO: remove ligth vecs from here...
        raster_images, (light_vecs, light_vecs_norms_sq) = rr.compute_raster(world_normals, albedo, posed_points, light_loc)
        self._light_vecs = light_vecs
        self._light_vecs_norms = np.sqrt(light_vecs_norms_sq)

        # stack attrs together, we should get a (A,N) array, in the order normal, albedo, image
        # transpose such that the shape (N,F,3)
        self._world_normals = world_normals
        self._aldedo = albedo
        self._raster_images = raster_images

        # set the target
        self.target = light_loc

        # compute loaded lengths
        # TODO: genezalize to when multiple images are loaded, as a running sum in the for loop
        self._len = posed_points.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        """Get the features of an image pixel as well as the direction from where it is lit form by the index within its image's occupied pixels"""
        return self._world_normals[index], self._aldedo[index], self._raster_images[index], self.target


if __name__ == "__main__":
    ds = RasterDataset()
    pprint(list(DataLoader(ds, num_workers=0))[:3])
    print(len(ds))
