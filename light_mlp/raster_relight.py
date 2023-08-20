import depth_transformations as dt
from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
import cv2
from rich.traceback import install as install_rich
import json
from matplotlib import pyplot as plt
import open3d as o3d
install_rich()


np.set_printoptions(precision=3, suppress=True)
plt.rcParams['figure.figsize'] = 10, 20


# #### Helpers and Plotters

def two_plot(im0, im1, label0="", label1=""):
    plt.subplot(1, 2, 1)
    plt.title(label0)
    plt.imshow(im0)
    plt.subplot(1, 2, 2)
    plt.title(label1)
    plt.imshow(im1)
    plt.show()


def check_values(a, name=""):
    if name:
        print(name)
    print(f"min: {a.min()}, mean: {a.mean()}, max: {a.max()}")


# ### Constants and Settings

# Constnats and paths
SCENE = 'intrinsic_tester_sphere'
SPLIT = ''
DATASET_PATH = "/home/dtetruash/Thesis/datasets/nerf-blender/nerf_synthetic"
SCENE_PATH = f"{DATASET_PATH}/{SCENE}"
DATA_PATH = f"{SCENE_PATH}/{SPLIT}" if SPLIT else SCENE_PATH

LIGHT_ORIENTATION = "bottom"

LIGHTS_FILE = f"{SCENE_PATH}/lights_{SCENE}.json"
TRANSFORMS_FILE = f"{SCENE_PATH}/transforms_{SPLIT}.json" if SPLIT else f"{SCENE_PATH}/transforms.json"

IMAGE_NUMBER = 0
# -

# Get the transform for the given image
def get_c2w(image_number):
    return np.array(image_transforms['frames'][image_number]['transform_matrix'])

# Transfomation matrix functions
def to_translation(c2w):
    return c2w[:3, -1]

def to_rotation(c2w):
    return c2w[:3, :3]

# Image loading functions
def get_image_name(image_number, channel=''):
    img_name = f"{DATA_PATH}/r_{image_number:03d}"
    img_name += f'_{channel}' if channel else ''
    img_name += f'_{LIGHT_ORIENTATION}' if LIGHT_ORIENTATION else ''
    return img_name + '.png'

# TODO: Add suport for a synamic open function.
def read_images(image_number, channels=[], open_func=Image.open):
    """Get an image along with it's given channels."""
    images = {'combined': get_image_name(image_number, channel='combined')}
    images.update({channel: get_image_name(image_number, channel) for channel in channels})
    return {k: open_func(v) for (k, v) in images.items()}

def read_16bit(img_name):
    """Read a 16-bit PNG into a numpy array"""
    bgrd = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    rgbd = np.empty_like(bgrd)
    rgbd[..., :3] = np.flip(bgrd[..., :3], axis=2)
    rgbd[..., -1] = bgrd[..., -1]
    return rgbd

def remap_depth_black2white(depth_array):
    """Remap an 8-bit depth image where 255 -> near and 0 -> far
    to the convention that 0 -> near and 255 -> far.
    """
    bit_depth = 8 if depth_array.dtype == np.uint8 else 16
    return -depth_array + (bit_depth**2 - 1)

def get_focal(width, camera_angle_x):
    return .5 * width / np.tan(.5 * camera_angle_x)

def denormalize_depth(depth_image_array, from_min=0, from_max=8):
    # DEPRICATED
    """Inverse the depth image mapping which we did in blender during compositing.
    The default forrward mapping is (0,8) -> (1,0).
    Return depth of each pixel in """
    depth_from_alpha = depth_image_array[..., -1]
    return (depth_from_alpha/255. - 1) * -(from_max - from_min), depth_from_alpha > 0


if __name__ == "__main__":
    # Load Transforms
    with open(TRANSFORMS_FILE, 'r') as tf:
        image_transforms = json.loads(tf.read())

    # Loading Images & Transforms
    # Get needed image passes
    images = read_images(IMAGE_NUMBER, ['albedo', 'depth', 'normal'])

    # Read normal as a 16-bit image
    image_normal = read_16bit(get_image_name(IMAGE_NUMBER, channel='normal'))

    # Reloading to meet the RGBd creation function requirements on image channels
    image_combined_rgb = o3d.geometry.Image(np.asarray(images['combined'].convert("RGB")))

    # invert the depth image to be black -> white as depth increases
    depth_raw = remap_depth_black2white(np.asarray(images['depth'])[:, :, -1].astype(np.uint8))
    image_depth = o3d.geometry.Image(depth_raw)

    # Lights and Light Transforms

    # Get lights information
    with open(LIGHTS_FILE, 'r') as lf:
        lights_info = json.loads(lf.read())
    print(f"Loaded {LIGHTS_FILE}. Number of lights is {len(lights_info['lights'])}")

    # Get the location of the lights
    light_transforms = [np.array(light['transformation_matrix']) for light in lights_info['lights']]
    light_locations = [to_translation(transform) for transform in light_transforms]

    # ### Camera Parameters
    H, W = np.asarray(images['combined']).shape[:-1]
    focal = get_focal(W, image_transforms["camera_angle_x"])
    c2w = get_c2w(IMAGE_NUMBER)
    T = to_translation(c2w)
    R = to_rotation(c2w)

    # Set intrinsic matrix
    f_x = f_y = focal
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, f_x, f_y, W/2., H/2.)
    print("Using the following intrinsics matrix:")
    print("K = ")
    print(intrinsics.intrinsic_matrix)

    # ## Creating Point Cloud
    # TODO: Rename depth_maks to pixel_mask or occupied_mask.
    depth_mask = np.asarray(images['depth'])[..., -1] > 0.0
    num_points = depth_mask.sum()
    print(f"There are {num_points} points with finite depth in the image.")

# ### Using Open3D RGBD + Point Cloud

    # Make RGBD image intput
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_combined_rgb, image_depth,
        depth_scale=1/8, depth_trunc=8,
        convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(intrinsics)
    )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Check that the point cloud has all of the points.
    points = np.asarray(pcd.points)
    assert num_points == points.shape[0]

    # Pose the point cloud with the camera transform
    pcd.transform(c2w)
    points_posed = np.asarray(pcd.points)

    # Raster Rendering

    # Constants and containers for later
    raster_image = np.zeros((W, H, 3))

    # Check the rotation matrix
    det_R = np.linalg.det(R)
    print(f"det(R) = {det_R}")
    assert np.abs(det_R - 1.0) < 1e-5

    Rt_R = R.T @ R
    print("R^T @ R =")
    print(Rt_R)
    assert np.all(np.abs(Rt_R - np.eye(3)) < 1e-5)

    # ### Load camera-space normals and scale them to the correct ranges
    camera_normals_pixels = image_normal[..., :-1] / (2**16 - 1)

    # Translation and scaling needed to remap to [-1,1], [-1,1], and [-1,0] resp.
    # (doing it like this for debug reasons)
    normals_shift = np.array([-0.5]*3)
    normals_scale = np.array([2.0]*3)

    camera_normals = (camera_normals_pixels + normals_shift) * normals_scale

    # Normalize the normals and zero non-occupied ones.
    camera_normals_norms = np.linalg.norm(camera_normals, axis=2)
    camera_normals_normalized = camera_normals / camera_normals_norms[..., np.newaxis] * depth_mask[..., np.newaxis]

    # #### Get World Normals

    # Reshape the vectors to be a list of 3D points
    camera_normals_flattened = camera_normals_normalized.reshape(-1, 3).T

    # Transform the vectors to world space
    world_normals_flattened = np.dot(R, camera_normals_flattened)

    # Reshape the transformed vectors back to the original image shape
    world_normals = world_normals_flattened.T.reshape(camera_normals.shape)

    # #### Get Light Distances for First Light

    # TODO: Make this dynamic w.r.t. the light location; encapsulate
    light_vectors = np.zeros_like(raster_image)
    light_vectors[depth_mask] = light_locations[0] - points_posed

    # Get norms of the light vectors
    light_norms = np.zeros(raster_image.shape[:-1])
    light_norms[depth_mask] = np.linalg.norm(light_vectors[depth_mask], axis=1)
    light_norms_sqr = np.square(light_norms)

    # Normalize light vectors
    light_vectors[depth_mask] = light_vectors[depth_mask] / light_norms[depth_mask][..., np.newaxis]

    # Compute max(n dot l,0) i.e., shading
    normal_dot_light = np.zeros(raster_image.shape[:-1])
    normal_dot_light = np.sum(world_normals.reshape((-1, 3)) * light_vectors.reshape((-1, 3)), axis=1)

    normal_dot_light = np.maximum(normal_dot_light.reshape((W, H)), 0) * depth_mask

    # Compute reaster image
    albedo = np.asarray(images['albedo'])[..., :3].astype(np.float32) / 255.

    raster_image = albedo * normal_dot_light[:, :, np.newaxis]

    raster_image_vis = raster_image.copy()
    raster_image_vis[~depth_mask] = 1

    # TODO: Add output name
    plt.imsave('raster.png', raster_image_vis)
