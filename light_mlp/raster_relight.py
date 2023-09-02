import os
import numpy as np
import cv2
import json
import open3d as o3d
import configparser

from PIL import Image
from matplotlib import pyplot as plt

from rich.traceback import install as install_rich
install_rich()


# ### Constants and Settings
#
def parse_config(config_name='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_name)

    DATASET_PATH = config['paths']['dataset_path']
    SCENE = config['paths']['scene']
    SPLIT = config['paths']['split']

    config['paths']['scene_path'] = SCENE_PATH = f"{DATASET_PATH}/{SCENE}"
    config['paths']['data_path'] = f"{SCENE_PATH}/{SPLIT}" if SPLIT else SCENE_PATH
    config['paths']['transforms_file'] = f"{SCENE_PATH}/transforms_{SPLIT}.json" if SPLIT else f"{SCENE_PATH}/transforms.json"
    config['paths']['lights_file'] = f"{SCENE_PATH}/lights_{SCENE}.json"
    return config


config = parse_config()

# TODO: mote this to a config dict? or config file
# Constnats and paths

# SCENE = 'intrinsic_tester_sphere'
# SPLIT = ''
# DATASET_PATH = "/home/dtetruash/Thesis/datasets/nerf-blender/nerf_synthetic"
# SCENE_PATH = f"{DATASET_PATH}/{SCENE}"
# data_path = f"{SCENE_PATH}/{SPLIT}" if SPLIT else SCENE_PATH
#
# LIGHT_ORIENTATION = "bottom"
#
# lights_file = f"{SCENE_PATH}/lights_{SCENE}.json"
# transforms_file = f"{SCENE_PATH}/transforms_{SPLIT}.json" if SPLIT else f"{SCENE_PATH}/transforms.json"
#
# image_number = 0
# -

# Get the transform for the given image
def get_c2w(image_number, image_transforms):
    return np.array(image_transforms['frames'][image_number]['transform_matrix'])

# Transfomation matrix functions
def to_translation(c2w):
    return c2w[:3, -1]

def to_rotation(c2w):
    return c2w[:3, :3]

# Image loading functions
def format_image_path(image_number, channel='', light=''):
    data_path = config['paths']['data_path']

    img_name = f"{data_path}/r_{image_number:03d}"
    img_name += f'_{channel}' if channel else ''
    img_name += f'_{light}' if light else ''
    return img_name + '.png'

# TODO: Add suport for a dynamic open function.
def get_image_paths(image_number, channels=[], lighting=[], open_func=Image.open):
    """Get an image along with it's given channels."""

    # get combined (fully formed) image names in combined images which are given by lighting
    image_paths = {(channel := f'combined_{lt}'): format_image_path(image_number, channel) for lt in lighting}
    image_paths['combined'] = format_image_path(image_number, 'combined')

    # read in other channels beside combined
    image_paths.update({channel: format_image_path(image_number, channel) for channel in channels})
    return image_paths

def read_16bit(img_name):
    """Read a 16-bit PNG into a numpy array"""
    bgrd = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    rgbd = np.empty_like(bgrd)
    rgbd[..., :3] = np.flip(bgrd[..., :3], axis=2)
    rgbd[..., -1] = bgrd[..., -1]
    return rgbd

def load_images(image_number, channels=[], lighting=[], depth_scale=1/8, depth_trunc=8):
    """ Return loaded images.
    Return a dicts of images loaded via Image.open, and handful of specialized formats
    """
    image_paths = get_image_paths(image_number, channels, lighting)

    # load in color and data images, compute RGBD image, then return
    images = {}
    for channel, path in image_paths.items():
        if os.path.exists(path):
            image = Image.open(path)
            images[channel] = image

    # FIXME: normals will not be loaded unless 'normals' in channels ... dumb design of a function

    # load normals
    if 'normal' in image_paths.keys():
        image_normal = read_16bit(image_paths['normal'])

    # Reloading to meet the RGBd creation function requirements on image channels
    image_combined_rgb = o3d.geometry.Image(np.asarray(images['combined'].convert("RGB")))

    # invert the depth image to be black -> white as depth increases
    image_depth = np.asarray(images['depth'])[:, :, -1].astype(np.uint8)
    depth_raw = remap_depth_black2white(image_depth)

    image_depth = o3d.geometry.Image(depth_raw)
    o3d.visualization.draw_geometries([image_depth])

    # Make RGBD image intput
    image_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_combined_rgb, image_depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False)

    W, H = np.asarray(list(images.items())[0][1]).shape[:-1]

    return W, H, images, image_normal, image_rgbd

def remap_depth_black2white(depth_array):
    # FIXME: Should work on notmalized arrays in [0,1] since then the color depth is irrelevant
    """Remap an 8-bit depth image where 255 -> near and 0 -> far
    to the convention that 0 -> near and 255 -> far.
    """
    bit_depth = 8 if depth_array.dtype == np.uint8 else 16
    print(f"Bit depth used for depth remapping was {bit_depth}")
    return -depth_array + (bit_depth**2 - 1)

def get_focal(width, camera_angle_x):
    return .5 * width / np.tan(.5 * camera_angle_x)

def denormalize_depth(depth_image_array, from_min=0, from_max=8):
    raise Exception("This function is deprecated.")
    """Inverse the depth image mapping which we did in blender during compositing.
    The default forrward mapping is (0,8) -> (1,0).
    Return depth of each pixel in """
    depth_from_alpha = depth_image_array[..., -1]
    return (depth_from_alpha/255. - 1) * -(from_max - from_min), depth_from_alpha > 0

def get_camera_parameters(W, H, camera_angle_x, image_number, image_transforms):
    focal = get_focal(W, camera_angle_x)
    c2w = get_c2w(image_number, image_transforms)
    T = to_translation(c2w)
    R = to_rotation(c2w)

    # Set intrinsic matrix
    f_x = f_y = focal
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, f_x, f_y, W/2., H/2.)

    return intrinsics, c2w, R, T

def project_and_pose_3d_points_via_rgbd(image_rgbd, intrinsics, c2w, return_array=True):
    """Projct image point via depth and camera parameters
    image_rgbd: open3d RGBD image to porject
    intrinsics: open3d intrinsics matrix object
    c2w: ndarray (4,4) containing the camera-to-world transform
    return_array: (default True) will return ndarray if set, and open3d point cloud object otherwise
    """
    # TODO: Replace this with a function which just does the projection without the color
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image_rgbd,
        o3d.camera.PinholeCameraIntrinsic(intrinsics)
    )

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform(
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]])

    # Pose the point cloud with the camera transform
    pcd.transform(c2w)
    points_posed = np.asarray(pcd.points)

    print(f"Posed_points shape in porject func: {points_posed.shape}")

    return points_posed if return_array else pcd

def check_rotation(R):
    # Check the rotation matrix
    det_R = np.linalg.det(R)
    print(f"det(R) = {det_R}")
    assert np.abs(det_R - 1.0) < 1e-5

    Rt_R = R.T @ R
    print("R^T @ R = ")
    print(Rt_R)
    assert np.all(np.abs(Rt_R - np.eye(3)) < 1e-5)

def get_camera_space_normals(camera_normal_pixels, shift=np.array([-0.5]*3), scale=np.array([2.0]*3), color_depth=16):
    """
    camera_normal_pixels : ndarray (N,3) of normals encoded as pixel values
    """
    # ### Load camera-space normals and scale them to the correct ranges
    camera_normals_pixels_normalized = camera_normal_pixels / (2**color_depth - 1)  # [0,1]

    # Translation and scaling needed to remap to [-1,1], [-1,1], and [-1,0] resp.
    # (doing it like this for debug reasons)
    camera_normals = (camera_normals_pixels_normalized + shift) * scale

    # Normalize the normals and zero non-occupied ones.
    camera_normals_norms = np.linalg.norm(camera_normals, axis=-1)
    camera_normals_normalized = camera_normals / camera_normals_norms[..., np.newaxis]

    return camera_normals_normalized

def get_world_space_normals(camera_normals, R):
    """
    camera_normals : ndarray (N,3) normalized normal vectors in camera-space
    Transform camera-space normals in inhomogeneous coordinates
    """
    # Transform the vectors to world space
    world_normals = np.dot(R, camera_normals.T)
    print(f"world_normals_shape after posing: {world_normals.shape}")

    # Reshape the transformed vectors back to the original image shape
    world_normals = world_normals.T.reshape(camera_normals.shape)
    print(f"world_normals shape after reshaping to camera_normals.shape: {world_normals.shape}")

    return world_normals

def compute_light_vectors(posed_points, light_location):
    """Compute vectors from the posed points to the given light location then normalize,
    as well as their squared norms."""
    # a[occupied_mask].shape == posed_points.shape
    light_vectors = light_location - posed_points
    print(f"shapes: l_v:{light_vectors}, l_l : {light_location.shape}, p_p : {posed_points.shape}")

    # Get norms of the light vectors
    light_norms = np.linalg.norm(light_vectors, axis=-1)
    light_norms_sqr = np.square(light_norms)

    return light_vectors / light_norms[..., np.newaxis], light_norms_sqr

def compute_viewing_vectors(posed_points, camera_center):
    # FIXME: Merge with the above function
    viewing_vectors = camera_center - posed_points
    viewing_norms = np.linalg.norm(viewing_vectors, axis=-1)
    return viewing_vectors / viewing_norms[..., np.newaxis], viewing_norms

def compute_clipped_dot_prod(vecs_1, vecs_2):
    assert vecs_1.shape == vecs_2.shape
    # Compute max(n dot l,0) i.e., shading
    dot = np.sum(vecs_2.reshape((-1, 3)) * vecs_1.reshape((-1, 3)), axis=-1)

    return np.maximum(dot, 0)

def shade_albedo(albedo, shading):
    # Compute reaster image

    # TODO: add viewing direction cosine.

    image = albedo * shading[..., np.newaxis]

    return image

def compute_raster(world_normals, albedo, posed_points, light_location, camera_center, light_power=50):
    """
    world_normals : ndarray of per-pixel normals (N, 3)
    albedo        : ndattay of per-pixels albedo (N, 3)
    occupied_mask : ndarray of per-pixel occupancy (N,)
    posed_points  : ndarray of per-pixel projected 3d locations (N, 3)
    light_location: ndarray of the location of the light to render with (3,)
    """

    # shapes
    assert world_normals.shape == albedo.shape
    shape = world_normals.shape[:-1]

    # #### Get Light Distances for First Light

    print(f'output shape is {shape}')
    light_vectors, light_vector_norms_sqr = compute_light_vectors(posed_points, light_location)
    viewing_vectors, viewing_norms = compute_light_vectors(posed_points, camera_center)

    shading = compute_clipped_dot_prod(light_vectors, world_normals)
    viewing_foreshortening = compute_clipped_dot_prod(viewing_vectors, world_normals)

    # TODO: Add geometric attentuation via light_vector_norms_sqr
    # geometric_term = np.zeros(shape)
    # geometric_term[occupied_mask] = light_power / (light_vector_norms_sqr[occupied_mask] * np.pi * 4.0 + 1e-5)

    return shade_albedo(albedo, shading) * viewing_foreshortening[..., np.newaxis], (light_vectors, light_vector_norms_sqr)

def get_occupancy(depth_image):
    """Get the occupancy of the image sample i.e., there where there is finite depth and therefore a surface.
    depth_image : nparray of the depth image where +ve values in the alpha channel indicate finite depth
    """
    if depth_image.ndim == 3:
        depth_alpha = depth_image[..., -1]
    elif depth_image.ndim == 2:
        depth_alpha = depth_image
    else:
        raise ValueError("Depth image of incompatible shape. Must be RGB or RGBD with 3 dimensions, or grayscale with 2.")

    return depth_alpha > 0.0

def create_raster_images(flatten=False):
    """Renders raster images given the config"""
    # Load Transforms
    with open(config['paths']['transforms_file'], 'r') as tf:
        image_transforms = json.loads(tf.read())

    image_number = int(config['images']['image_number'])

    # Loading Images
    W, H, images, image_normal, image_rgbd = load_images(
        image_number,
        channels=['albedo', 'depth', 'normal'])

    # Loading Lights and Light Transforms
    with open(config['paths']['lights_file'], 'r') as lf:
        lights_info = json.loads(lf.read())

    print(f"Loaded {config['paths']['lights_file']}. Number of lights is {len(lights_info['lights'])}")

    light_transforms = {light['name_light'].lower(): np.array(light['transformation_matrix']) for light in lights_info['lights']}
    light_locations = {name: to_translation(transform) for (name, transform) in light_transforms.items()}

    # Compute Camera Parameters
    intrinsics, c2w, R, T = get_camera_parameters(W, H,
                                                  image_transforms["camera_angle_x"],
                                                  image_number,
                                                  image_transforms)
    check_rotation(R)

    print("Using the following intrinsics matrix:")
    print("K = ")
    print(intrinsics.intrinsic_matrix)

    # ## Creating Point Cloud
    occupied_mask = get_occupancy(np.asarray(images['depth']))
    print(f"There are {occupied_mask.sum()} points with finite depth in the image.")

    # ### Using Open3D RGBD + Point Cloud
    posed_points = project_and_pose_3d_points_via_rgbd(image_rgbd, intrinsics, c2w)

    # impose the shape to be (N, 3)
    posed_points = posed_points[occupied_mask.reshape(-1)].reshape(-1, 3)
    print(f"Posed_points shape in raster: {posed_points.shape}")

    # Raster Rendering
    albedo = np.asarray(images['albedo'])[..., :3].astype(np.float32) / 255.

    # Get Normals
    camera_normals = get_camera_space_normals(image_normal) * occupied_mask[..., np.newaxis]
    world_normals = get_world_space_normals(camera_normals, R)

    # do this for the given light orientations, populate dict for output
    # output_rasters = {}
    # for light_name, light_location in light_locations.items():
    #     raster_image = compute_raster(world_normals, albedo, occupied_mask, posed_points, light_location)
    #     output_rasters.update(light_name, raster_image)

    output_rasters = [
        [light_name, compute_raster(world_normals, albedo, occupied_mask, posed_points, light_location, T)]
        for (light_name, light_location) in light_locations.items()
    ]

    return output_rasters


if __name__ == "__main__":

    raster_images = create_raster_images()

    # TODO: Add output name
    for light_name, image in raster_images.items():
        output_name = f"raster_{light_name}.png"
        plt.imsave('raster.png', image)
