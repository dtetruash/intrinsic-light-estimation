"""This script holds code to manimulate the intrinsic dataset images
and to generate point-lit OLAT sampels from existing sample attributes.
"""

import json
from log import get_logger
from configparser import NoOptionError

import data_loaders.image_loader as il
import numpy as np
import open3d as o3d
import torch
from ile_utils.config import Config
from log import get_logger
from matplotlib import pyplot as plt
from rich.traceback import install as install_rich

install_rich()

config = Config.get_config()
logger = get_logger(__file__)


# Get the transform for the given image
def get_c2w(frame_number, frame_transforms):
    """Extract camera transform for the given frame

    Args:
        frame_number (int): frame number
        frame_transforms (dict): loaded dict of image transforms

    Returns:
        Array of the camera transform
    """
    return np.array(frame_transforms["frames"][frame_number]["transform_matrix"])


# Transfomation matrix functions
def to_translation(c2w):
    return c2w[:3, -1]


def to_rotation(c2w):
    return c2w[:3, :3]


def remove_alpha(image_array):
    """Strip the alpha channel from RGBA image array."""
    # Need more robust value checking:
    if image_array.shape[-1] == 4:
        return image_array[..., :-1]
    else:
        raise ValueError(
            "Given image array has unexpected number of dimensions         along the"
            f" final axis. Was {image_array.shape[-1]} and expected 4."
        )


def gather_intrinsic_components(frame_number, frame_transforms, downsample_ratio=1):
    """Gather intrinsin components of a frame needed to combine into OLAT samples.

    Here, we load needed images of a frame, compute the 3d point cloud of each pixel,
    and return them as streams of occupied pixels (N, 3).

    Args:
        frame_number (int): fame to gather components for
        depth_scale (float): inverse scale factor to scale the depth with
            (will be divided by)
        depth_trunc (float): maximum depth value after which depth is considered inf
        downsample_ratio (int): positive downsampling ratio

    Returns:
        Width of the frame,
        Height of the frame,
        Dictionary of the unaltered images for normal, albedo, and depth
        albedo stream (N, 3),
        world-space normal vectors stream (N, 3),
        posed point cloud (N, 3),
        occupancy mask for the WxH image (W,H, 1),
    """

    # Load raw image data
    # This also determines the with/height of the images used
    W, H, images = il.load_frame_channels(
        frame_number,
        data_path=None,
        channels=["normal", "albedo", "depth"],
        downsample_ratio=downsample_ratio,
    )

    # Compute Camera Parameters
    intrinsics, c2w = get_camera_parameters(W, H, frame_number, frame_transforms)

    # Albdo
    albedo = remove_alpha(images["albedo"])
    logger.debug(f"Albedo range from {albedo.min()} to {albedo.max()}")

    # Normal
    normal_pixels = remove_alpha(images["normal"])
    logger.debug(
        f"Normal pixels range from {normal_pixels.min()} to {normal_pixels.max()}"
    )
    world_normals = get_world_space_normals_from_pixels(normal_pixels, c2w)

    # get only the alpha from the depth image
    depth_alpha = images["depth"][..., -1]  # This would be an array in [0,255]
    logger.debug(
        f"Got depth_alpha from {depth_alpha.min()} to {depth_alpha.max()}"
        f" with mean of {depth_alpha.mean()}. The datatype is { depth_alpha.dtype }"
    )

    # invert the depth image to be black -> white as depth increases
    depth_remapped, depth_normalization_constant = remap_depth_black2white(depth_alpha)
    logger.debug(
        f"Ater remapping got depth_remapped from {depth_remapped.min()}"
        f" to {depth_remapped.max()} with mean of {depth_remapped.mean()}."
        f" The datatype is { depth_remapped.dtype }"
    )

    # read-in parameters for RGBD image creation
    # if only one is set, we must error, since fallbacks only work together
    try:
        depth_scale = config.get("parameters", "depth_scale")
        depth_trunc = config.get("parameters", "depth_trunc")
        logger.info(
            f"Read-in depth_scale {depth_scale} and"
            f" depth_trunc {depth_trunc} from config."
        )
    except NoOptionError:
        depth_scale = 1 / 8
        depth_trunc = 8.0
        logger.warning(
            "Parameter options depth_scale and depth_trunc not set in config!\n"
            f"Using defaults {depth_scale} and {depth_trunc} respectively."
            " These might not be correct for the depth maps used, so do check."
        )

    # Make RGBD image intput
    # TODO: Replace this with just the projection and no color
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image((albedo * 255).astype(np.uint8)),
        o3d.geometry.Image(depth_remapped),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    # Normalize depth to [0,1] after the rgbd image is created
    depth_normalized = depth_remapped / depth_normalization_constant
    logger.debug(
        f"After normalizing got depth_normalized from {depth_normalized.min()}        "
        f" to {depth_normalized.max()} with mean of {depth_normalized.mean()}.        "
        f" The datatype is { depth_normalized.dtype }"
    )

    # create the posed point cloud
    posed_points = project_and_pose_3d_points_via_rgbd(rgbd, intrinsics, c2w)

    # get image occupancy
    occupancy_mask = get_occupancy(depth_alpha)
    logger.debug(
        f"Occumpancy mask of shape {occupancy_mask.shape},         and with"
        f" {occupancy_mask.sum()} occupied pixels."
    )

    return (
        W,
        H,
        images,
        albedo[occupancy_mask],
        world_normals[occupancy_mask],
        posed_points,
        occupancy_mask,
    )


def remap_depth_black2white(depth_array):
    """Remap an 8-bit or 16-bit depth image where white -> near and black -> far
    to the convention that black -> near and white -> far.
    """
    bit_depth = 8 if depth_array.dtype == np.uint8 else 16
    max_value = 2**bit_depth - 1
    logger.debug(
        f"Bit depth used for depth remapping was {bit_depth}         and a max values of"
        f" {max_value}"
    )
    return -depth_array + max_value, max_value


def get_focal(width, camera_angle_x):
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    return focal


def denormalize_depth(depth_image_array, from_min=0, from_max=8):
    raise Exception("This function is deprecated.")
    """Inverse the depth image mapping which we did in blender during compositing.
    The default forrward mapping is (0,8) -> (1,0).
    Return depth of each pixel in """
    depth_from_alpha = depth_image_array[..., -1]
    return (depth_from_alpha / 255.0 - 1) * -(from_max - from_min), depth_from_alpha > 0


def get_camera_parameters(W, H, frame_number, frame_transforms):
    """Compose the intrinsic and extrinsic camera matrecies for the given frame

    Args:
        W (int): width of the frame
        H (int): height of the frame
        frame_number (int): index of the frame
        frame_transforms (dict): dictionar of frame parameters from intrinsic database

    Returns:
        o3d.camera.PinholeCameraIntrinsic, and ndarray of camera extrinsics
    """
    c2w = get_c2w(frame_number, frame_transforms)

    # Check that the rotation matrix is valid
    check_rotation(to_rotation(c2w))

    # Set intrinsic matrix
    camera_angle_x = frame_transforms["camera_angle_x"]
    focal = get_focal(W, camera_angle_x)
    f_x = f_y = focal
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, f_x, f_y, W / 2.0, H / 2.0)

    return intrinsics, c2w


def project_and_pose_3d_points_via_rgbd(
    image_rgbd, camara_intrinsics, c2w, return_array=True
):
    """Projct image point via depth and camera parameters
    image_rgbd: open3d RGBD image to porject
    intrinsics: open3d intrinsics matrix object
    c2w: ndarray (4,4) containing the camera-to-world transform
    return_array: (default True) will return ndarray if set,
                and open3d point cloud object otherwise
    """
    # TODO: Replace this with a function
    # which just does the projection without the color
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image_rgbd, o3d.camera.PinholeCameraIntrinsic(camara_intrinsics)
    )

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Pose the point cloud with the camera transform
    pcd.transform(c2w)
    points_posed = np.asarray(pcd.points)

    logger.debug(f"Posed_points shape in porject func: {points_posed.shape}")

    return points_posed if return_array else pcd


def check_rotation(R):
    # Check the rotation matrix
    det_R = np.linalg.det(R)
    logger.debug(f"det(R) = {det_R}")
    assert np.abs(det_R - 1.0) < 1e-5, (
        "Rotation check failed. Expected rotation matrix determinant to be 1,     and"
        f" was {det_R}."
    )

    Rt_R = R.T @ R
    logger.debug("R^T @ R = ")
    logger.debug(Rt_R)
    assert np.all(
        np.abs(Rt_R - np.eye(3)) < 1e-5
    ), "Rotation check failed. Rotation matrix was not orthonormal."


def get_camera_space_normals_from_pixels(
    camera_normal_pixels, shift=np.array([-0.5] * 3), scale=np.array([2.0] * 3)
):
    """
    Computes normalized normal vectors in camera-space from normalized pixel values.

    camera_normal_pixels : ndarray (N,3) of camera-space normals encoded as pixel values in [0..1]

    Returns:
        Normalized R^3 normal vectors in camera-space.
    """
    # ### Load camera-space normals and scale them to the correct ranges
    if camera_normal_pixels.dtype != np.float32:
        raise ValueError(
            "The datatype of given camera normal pixel vaues"
            f" is {camera_normal_pixels.dtype}. Only np.float32 supported."
        )
    if camera_normal_pixels.shape[-1] != 3:
        raise ValueError(
            "Last axis of camera_normal_pixels must be 3 (RGB)."
            f" Was {camera_normal_pixels.shape[-1]}."
        )
    if camera_normal_pixels.max() > 1.0 or camera_normal_pixels.min() < 0.0:
        raise ValueError(
            "The given camera normal pixel values are not normalized.\n"
            "Expected values between 0.0 and 1.0, got values between"
            f" {camera_normal_pixels.min()} and {camera_normal_pixels.max()}."
        )

    # Translation and scaling needed to remap to [-1,1], [-1,1], and [-1,0] resp.
    # (doing it like this for debug reasons)
    camera_normals = (camera_normal_pixels + shift) * scale

    # Normalize the normals and zero non-occupied ones.
    camera_normals_norms = np.linalg.norm(camera_normals, axis=-1)
    camera_normals_normalized = camera_normals / camera_normals_norms[..., np.newaxis]

    return camera_normals_normalized


def to_world_space_normals(camera_normals, R):
    """
    camera_normals : ndarray (N,3) normalized normal vectors in camera-space
    Transform camera-space normals in inhomogeneous coordinates
    """
    # Transform the vectors to world space
    world_normals = np.dot(R, camera_normals.T)
    logger.debug(f"world_normals_shape after posing: {world_normals.shape}")

    # Reshape the transformed vectors back to the original image shape
    world_normals = world_normals.T.reshape(camera_normals.shape)
    logger.debug(
        "world_normals shape after reshaping to camera_normals.shape:"
        f" {world_normals.shape}"
    )

    return world_normals


def to_camera_space_normals(world_normals, R):
    """
    world_normals: ndarray (N,3) normalized normal vectors in camera-space
    Transform camera-space normals in inhomogeneous coordinates
    """
    camera_normals = np.dot(R.T, world_normals.T)

    # Reshape the transformed vectors back to the original image shape
    camera_normals = camera_normals.T.reshape(world_normals.shape)

    return camera_normals


def get_world_space_normals_from_pixels(camera_normal_pixels, c2w):
    """Transform normalized pixels carrying normal information in camera-space in [0..1]
    to world-space normal vectors.

    Args:
        camera_normal_pixels (ndarray): array of RGB pixels carrying normal vector information
        c2w (ndarray): camera to world transform calibrated for above pixels

    Returns:
        NDArray of world normals computed from the pixels and the transform.
    """
    camera_normals = get_camera_space_normals_from_pixels(camera_normal_pixels)

    R = to_rotation(c2w)
    check_rotation(R)

    world_normals = to_world_space_normals(camera_normals, R)

    return world_normals


def get_pixels_from_world_normals(world_normals, c2w):
    R = to_rotation(c2w)
    check_rotation(R)

    camera_normals = to_camera_space_normals(world_normals, R)

    pixels = get_pixels_from_camera_space_normals(camera_normals)
    return pixels.astype(np.float32)


def get_pixels_from_camera_space_normals(camera_normals):
    shift = np.array([1] * 3)
    scale = np.array([2.0] * 3)
    return (camera_normals + shift) / scale


def compute_light_vectors(posed_points, light_location):
    """Compute vectors from the posed points to the given light location then normalize,
    as well as their squared norms."""
    # a[occupied_mask].shape == posed_points.shape
    light_vectors = light_location - posed_points
    logger.debug(
        f"shapes: l_v:{light_vectors.shape}, l_l : {light_location.shape}, "
        f" p_p : {posed_points.shape}"
    )

    # Get norms of the light vectors
    light_norms = np.linalg.norm(light_vectors, axis=-1)
    light_norms_sqr = np.square(light_norms)

    normalized_light_vectors = light_vectors / light_norms[..., np.newaxis]

    return normalized_light_vectors, light_norms_sqr


def compute_viewing_vectors(posed_points, camera_center):
    # FIXME: Merge with the above function
    viewing_vectors = camera_center - posed_points
    viewing_norms = np.linalg.norm(viewing_vectors, axis=-1)
    return viewing_vectors / viewing_norms[..., np.newaxis], viewing_norms


def compute_clipped_dot_prod(vecs_1, vecs_2):
    assert vecs_1.shape == vecs_2.shape, (
        "Clipped dot prod.: Invalid inputs do not have identical shapes. Were"
        f" {vecs_1.shape} and {vecs_2.shape}"
    )
    # Compute max(n dot l,0) i.e., shading
    dot = np.sum(vecs_2.reshape((-1, 3)) * vecs_1.reshape((-1, 3)), axis=-1)

    return np.maximum(dot, 0)


def compute_clipped_dot_prod_torch(vecs_1, vecs_2):
    assert vecs_1.shape == vecs_2.shape, (
        "Clipped dot prod.: Invalid inputs do not have identical shapes. Were"
        f" {vecs_1.shape} and {vecs_2.shape}"
    )
    # Compute max(n dot l,0) i.e., shading
    dot = torch.sum(vecs_2.reshape((-1, 3)) * vecs_1.reshape((-1, 3)), dim=-1)

    return torch.maximum(dot, torch.tensor([0.0])).float()


def shade_albedo(albedo, shading, torch_mode=True):
    """Shade the albedo color channel according to achromatic shading of the
    same size.

    Args:
        albedo (ndarray): array of albedo RGB pixels (N,3)
        shading (ndarray): array of shading values in [0..1] (N,)

    Returns:
        ndarray of shaded pixels = albedo ⊙ shading
    """
    # Compute reaster image
    if torch_mode:
        image = torch.multiply(albedo, shading[:, np.newaxis])
    else:
        image = np.empty_like(albedo)
        np.multiply(albedo, shading[:, np.newaxis], image).astype(np.float32)

    return image


def shade_albedo_torch(albedo, shading):
    # Compute reaster image
    return shade_albedo(albedo, shading, torch_mode=True)


def render_OLAT_pixelstream(
    world_normals,
    albedo,
    posed_points,
    light_location,
    return_light_vectors=False,
    return_shading=False,
):
    """Compute the OLAT rendering of a posed and oriented point cloud
    with some albedo lit by a pointlight at a known location.

    Args:
        world_normals (ndarray):  per-pixel normals (N, 3) in range [-1,1]^3
        albedo (ndarray):  per-pixels albedo (N, 3) in range [0,1]^3
        posed_points (ndarray): per-pixel projected 3d locations (N, 3)
        light_location (ndarray):  per-pixel projected 3d locations (N, 3)
        return_light_vectors (bool): return the light vectors and squared norms
        return_shading (bool): return the shading pixels

    Returns:
        Flat OLAT rendered pixel stream,
        optionally the intermediary shading used for the rendered pixel stream,
        optionally a tuple of light vectors and their squared norms.
        Returned in this order.
    """
    # shapes
    assert world_normals.shape == albedo.shape, (
        "Compute raster failed. World normals and albedo shape mismatch. Were"
        f" {world_normals.shape} and {albedo.shape}."
    )

    logger.debug(
        f"In compute_OLAT_pixelstream: posed_points - {posed_points.shape},"
        f" light_location: {light_location.shape}"
    )

    light_vectors, light_vector_norms_sqr = compute_light_vectors(
        posed_points, light_location
    )

    shading = compute_clipped_dot_prod(light_vectors, world_normals)
    render = shade_albedo(albedo, shading)

    logger.debug(
        f"in raster: albedo type is {albedo.dtype} and range in"
        f" [{albedo.min(), albedo.max()}]"
    )
    logger.debug(
        f"in raster: shading type is {shading.dtype} and range in"
        f" [{shading.min(), shading.max()}]"
    )
    logger.debug(
        f"in raster: raster type is {render.dtype} and range in"
        f" [{render.min(), render.max()}]"
    )

    ret = [render]

    if return_shading:
        ret.append(shading)

    if return_light_vectors:
        ret.append((light_vectors, light_vector_norms_sqr))

    return tuple(ret)


def render_from_directions(light_dirs, albedo, world_normals, return_shading=False):
    shading = compute_clipped_dot_prod(light_dirs, world_normals)
    raster = shade_albedo(albedo, shading)
    return (raster, shading) if return_shading else raster


def render_from_directions_torch(
    light_dirs, albedo, world_normals, return_shading=False
):
    shading = compute_clipped_dot_prod_torch(light_dirs, world_normals)
    raster = shade_albedo_torch(albedo, shading)
    return (raster, shading) if return_shading else raster


def get_occupancy(depth_image):
    """Get the occupancy of the image sample i.e.,
    there where there is finite depth and therefore a surface.

    depth_image : ndarray of the depth image where +ve values in the alpha channel
                  indicate finite depth. Dim(3) RGB(A) or Dim(2) Alpha accepted.
    """
    if depth_image.ndim == 3:
        depth_alpha = depth_image[..., -1]
    elif depth_image.ndim == 2:
        depth_alpha = depth_image
    else:
        raise ValueError(
            "Depth image of incompatible shape. Must be RGB(A) with 3 dimensions,"
            "or grayscale/alpha with 2."
        )

    return depth_alpha > 0.0


def reconstruct_image(W, H, pixels, occupancy, add_alpha=False):
    """Reconstruct image from pixel and occupancy as RGB or RGBA.

    Args:
        W (int): width of image
        H (int): height of image
        pixels (ndarray): array of image pixels (N,3) or (N,)
        occupancy (ndarray): binary array of pixel occupancy
        add_alpha (bool): add alpha channel from occupancy

    Returns:
        ndarray image array with given pixels in RGB or RGBA mode
    """
    d = pixels.dim() if isinstance(pixels, torch.Tensor) else pixels.ndim
    msg = f"Provided pixels have shape {pixels.shape} and dim {d}."
    assert pixels.shape[-1] == 3 or d == 1, (
        msg,
        "Expected 3 in last dimension or to be scalar array.",
    )

    logger.debug(msg)

    image_channels = 4 if add_alpha else 3
    image_container = np.ones((W, H, image_channels))
    if d == 1:
        image_container[..., 0:3][occupancy] = pixels[..., np.newaxis]
    else:
        image_container[..., 0:3][occupancy] = pixels

    if add_alpha:
        image_container[..., -1] = occupancy.astype(np.float32)

    return image_container


def create_OLAT_samples_for_frame():
    """Renders raster images given the config for one image frame"""
    # Load Transforms
    with open(config.get("dataset", "transforms_file"), "r") as tf:
        frame_transforms = json.loads(tf.read())

    frame_number = int(config.get("images", "image_number"))

    downsample_ratio = int(config.get("dataset", "downsample_ratio"))

    # Loading Images
    (
        W,
        H,
        _,  # Don't need images in this method
        albedo,
        normals,
        posed_points,
        occupency_mask,
    ) = gather_intrinsic_components(
        frame_number, frame_transforms, downsample_ratio=downsample_ratio
    )
    logger.info(f"Loaded images of size ({W},{H}).")

    # Loading Lights and Light Transforms
    lights_file_path = config.get("dataset", "lights_file")
    with open(lights_file_path, "r") as lf:
        lights_info = json.loads(lf.read())
    logger.info(
        f"Loaded {lights_file_path}.         Number of lights is"
        f" {len(lights_info['lights'])}"
    )

    light_transforms = {
        light["name_light"].lower(): np.array(light["transformation_matrix"])
        for light in lights_info["lights"]
    }
    light_locations = {
        name: to_translation(transform) for (name, transform) in light_transforms.items()
    }

    output_OLATs = [
        (
            light_name,
            render_OLAT_pixelstream(normals, albedo, posed_points, light_location),
        )
        for (light_name, light_location) in light_locations.items()
    ]

    return output_OLATs, occupency_mask, W, H


if __name__ == "__main__":
    olat_pixel_arrays, occupancy_mask, W, H = create_OLAT_samples_for_frame()

    for light_name, pixels in olat_pixel_arrays:
        assert isinstance(pixels, np.ndarray)

        logger.debug(f"{light_name}: {pixels.min()}, {pixels.mean()}, {pixels.max()}")
        output_name = f"olat_{light_name}.png"
        image_container = reconstruct_image(W, H, pixels, occupancy_mask)
        plt.imsave(output_name, image_container)
