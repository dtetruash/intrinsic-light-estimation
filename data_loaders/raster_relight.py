"""This script hold code to manimulate the intrinsic dataset
and to generate point-lit OLAT sampels (dubbed rasters for convenience) from
existing sample attributes.
"""

import logging
import numpy as np
import json
import open3d as o3d
import torch


from ile_utils.config import Config
import data_loaders.image_loader as il

from matplotlib import pyplot as plt

from rich.traceback import install as install_rich

install_rich()

config = Config.get_config()


# Get the transform for the given image
def get_c2w(image_number, image_transforms):
    return np.array(image_transforms["frames"][image_number]["transform_matrix"])


# Transfomation matrix functions
def to_translation(c2w):
    return c2w[:3, -1]


def to_rotation(c2w):
    return c2w[:3, :3]


def load_images(image_number, depth_scale=(1 / 8), depth_trunc=8.0, downsample_ratio=1):
    """Load images into (N, ...) ndarrays of needed dtype of only occupied pixels.
    Return a tuple of ndarrays with flattened and occupied pixel attributes and
    information and handful of specialized formats
    Also return an rbd image for later use in projecting depth.
    """

    images = il.load_image_channels(
        image_number,
        data_path=None,
        channels=["normal", "albedo", "depth"],
        downsample_ratio=downsample_ratio,
    )

    # Normal
    image_normal = images["normal"][..., :-1]

    # albdo
    image_albedo = images["albedo"][..., :-1]
    # plt.imsave('albedo_in_relight_load.png', image_albedo)
    # get only the alpha from the depth image
    depth_alpha = images["depth"][..., -1]  # This would be an array in [0,255]

    logging.debug(
        f"Got depth_alpha from {depth_alpha.min()} to {depth_alpha.max()} \
        with mean of {depth_alpha.mean()}. The datatype is { depth_alpha.dtype }"
    )

    # plt.imsave('depth_alpha.png', depth_alpha, vmin=0, vmax=255, cmap='gray')

    # invert the depth image to be black -> white as depth increases
    depth_remapped, depth_normalization_constant = remap_depth_black2white(depth_alpha)

    logging.debug(
        f"Ater remapping got depth_remapped from {depth_remapped.min()} \
        to {depth_remapped.max()} with mean of {depth_remapped.mean()}. \
        The datatype is { depth_remapped.dtype }"
    )

    # plt.imsave('depth_remapped.png', depth_remapped, vmin=0, vmax=255, cmap='gray')

    # Make RGBD image intput
    # TODO: Replace this with just the projection and no color
    image_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(images["albedo"][..., :-1]),
        o3d.geometry.Image(depth_remapped),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    # o3d.visualization.draw_geometries(
    # [o3d.geometry.Image(images['albedo'][..., :-1]),
    # o3d.geometry.Image(depth_remapped)])

    # Normalize depth to [0,1] after the rgbd image is created
    depth_normalized = depth_remapped / depth_normalization_constant
    logging.debug(
        f"After normalizing got depth_normalized from {depth_normalized.min()} \
        to {depth_normalized.max()} with mean of {depth_normalized.mean()}. \
        The datatype is { depth_normalized.dtype }"
    )

    # get image occupancy
    occupancy_mask = get_occupancy(depth_alpha)
    logging.debug(
        f"Occumpancy mask of shape {occupancy_mask.shape}, \
        and with {occupancy_mask.sum()} occupied pixels."
    )

    # return image data (minux alpha) for only occupied pixels for further processing

    return (
        W,
        H,
        image_albedo[occupancy_mask],
        image_normal[occupancy_mask],
        depth_normalized[occupancy_mask],
        image_rgbd,
        occupancy_mask,
    )


def remap_depth_black2white(depth_array):
    """Remap an 8-bit or 16-bit depth image where white -> near and black -> far
    to the convention that black -> near and white -> far.
    """
    bit_depth = 8 if depth_array.dtype == np.uint8 else 16
    max_value = 2**bit_depth - 1
    logging.debug(
        f"Bit depth used for depth remapping was {bit_depth} \
        and a max values of {max_value}"
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


def get_camera_parameters(W, H, camera_angle_x, image_number, image_transforms):
    focal = get_focal(W, camera_angle_x)

    c2w = get_c2w(image_number, image_transforms)
    T = to_translation(c2w)
    R = to_rotation(c2w)

    # Set intrinsic matrix
    f_x = f_y = focal
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, f_x, f_y, W / 2.0, H / 2.0)

    return intrinsics, c2w, R, T


def project_and_pose_3d_points_via_rgbd(image_rgbd, intrinsics, c2w, return_array=True):
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
        image_rgbd, o3d.camera.PinholeCameraIntrinsic(intrinsics)
    )

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Pose the point cloud with the camera transform
    pcd.transform(c2w)
    points_posed = np.asarray(pcd.points)

    logging.debug(f"Posed_points shape in porject func: {points_posed.shape}")

    return points_posed if return_array else pcd


def check_rotation(R):
    # Check the rotation matrix
    det_R = np.linalg.det(R)
    logging.info(f"det(R) = {det_R}")
    assert (
        np.abs(det_R - 1.0) < 1e-5
    ), f"Rotation check failed. Expected rotation matrix determinant to be 1, \
    and was {det_R}."

    Rt_R = R.T @ R
    logging.info("R^T @ R = ")
    logging.info(Rt_R)
    assert np.all(
        np.abs(Rt_R - np.eye(3)) < 1e-5
    ), "Rotation check failed. Rotation matrix was not orthonormal."


def get_camera_space_normals(
    camera_normal_pixels, shift=np.array([-0.5] * 3), scale=np.array([2.0] * 3)
):
    """
    camera_normal_pixels : ndarray (N,3) of normals encoded as pixel values
    """
    # ### Load camera-space normals and scale them to the correct ranges
    if camera_normal_pixels.dtype == np.uint8:
        color_depth = 8
    elif camera_normal_pixels.dtype == np.uint16:
        color_depth = 16
    else:
        raise ValueError(
            f"The datatype of given camera normal pixel vaues \
            is {camera_normal_pixels.dtype}. Only uint8 and uint16 are supported."
        )

    logging.info(f"Normals were loaded using a color depth of {color_depth}.")

    camera_normals_pixels_normalized = camera_normal_pixels / (
        2**color_depth - 1
    )  # [0,1]

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
    logging.debug(f"world_normals_shape after posing: {world_normals.shape}")

    # Reshape the transformed vectors back to the original image shape
    world_normals = world_normals.T.reshape(camera_normals.shape)
    logging.debug(
        f"world_normals shape after reshaping to \
            camera_normals.shape: {world_normals.shape}"
    )

    return world_normals


def compute_light_vectors(posed_points, light_location):
    """Compute vectors from the posed points to the given light location then normalize,
    as well as their squared norms."""
    # a[occupied_mask].shape == posed_points.shape
    light_vectors = light_location - posed_points
    logging.debug(
        f"shapes: l_v:{light_vectors.shape}, l_l : {light_location.shape}, \
            p_p : {posed_points.shape}"
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
    assert (
        vecs_1.shape == vecs_2.shape
    ), f"Clipped dot prod.: Invalid inputs do not have identical shapes. \
        Were {vecs_1.shape} and {vecs_2.shape}"
    # Compute max(n dot l,0) i.e., shading
    dot = np.sum(vecs_2.reshape((-1, 3)) * vecs_1.reshape((-1, 3)), axis=-1)

    return np.maximum(dot, 0)


def compute_clipped_dot_prod_torch(vecs_1, vecs_2):
    assert (
        vecs_1.shape == vecs_2.shape
    ), f"Clipped dot prod.: Invalid inputs do not have identical shapes. \
        Were {vecs_1.shape} and {vecs_2.shape}"
    # Compute max(n dot l,0) i.e., shading
    dot = torch.sum(vecs_2.reshape((-1, 3)) * vecs_1.reshape((-1, 3)), dim=-1)

    return torch.maximum(dot, torch.tensor([0.0])).float()


def shade_albedo(albedo, shading):
    # Compute reaster image
    image = np.empty_like(albedo)
    np.multiply(albedo, shading[:, np.newaxis], image)
    return image


def shade_albedo_torch(albedo, shading):
    # Compute reaster image
    image = torch.multiply(albedo, shading[:, np.newaxis])
    assert (
        image.shape == albedo.shape
    ), f"Shade albedo failed. Image and albedo shape mismatch. \
        Were {albedo.shape} and {image.shape}."
    return image.float()


def compute_raster(
    world_normals,
    albedo,
    posed_points,
    light_location,
    camera_center,
    light_power=50,
    apply_viewing_cosine=False,
):
    """
    Compute the raster rendering of a collection of posed points with some albedo lit
    by a pointlight at a know direction.

    world_normals : ndarray of per-pixel normals (N, 3) in range [-1,1]^3
    albedo        : ndattay of per-pixels albedo (N, 3) in range [0,1]^3
    occupied_mask : ndarray of per-pixel occupancy (N,) binary
    posed_points  : ndarray of per-pixel projected 3d locations (N, 3)
    light_location: ndarray of the location of the light to render with (3,)
    """

    # shapes
    assert (
        world_normals.shape == albedo.shape
    ), f"Compute raster failed. World normals and albedo shape mismatch. \
        Were {world_normals.shape} and {albedo.shape}."

    logging.debug(
        f"in compute_raster: posed_points - {posed_points.shape}, \
        light_location: {light_location.shape}"
    )
    # #### Get Light Distances for First Light

    light_vectors, light_vector_norms_sqr = compute_light_vectors(
        posed_points, light_location
    )
    viewing_vectors, viewing_norms = compute_viewing_vectors(
        posed_points, camera_center
    )

    shading = compute_clipped_dot_prod(light_vectors, world_normals)
    viewing_foreshortening = compute_clipped_dot_prod(viewing_vectors, world_normals)

    raster = shade_albedo(albedo, shading)

    if apply_viewing_cosine:
        raster *= viewing_foreshortening[..., np.newaxis]

    logging.debug(
        f"in raster: albedo type is {albedo.dtype} and range \
        in [{albedo.min(), albedo.max()}]"
    )
    logging.debug(
        f"in raster: shading type is {shading.dtype} and range \
        in [{shading.min(), shading.max()}]"
    )
    logging.debug(
        f"in raster: raster type is {raster.dtype} and range \
        in [{raster.min(), raster.max()}]"
    )

    return (
        raster,
        (light_vectors, light_vector_norms_sqr),
        (viewing_vectors, viewing_norms),
    )


def raster_from_directions(light_dirs, albedo, world_normals, return_shading=False):
    shading = compute_clipped_dot_prod(light_dirs, world_normals)
    raster = shade_albedo(albedo, shading)
    return (raster, shading) if return_shading else raster


def raster_from_directions_torch(
    light_dirs, albedo, world_normals, return_shading=False
):
    shading = compute_clipped_dot_prod_torch(light_dirs, world_normals)
    raster = shade_albedo_torch(albedo, shading)
    return (raster, shading) if return_shading else raster


def get_occupancy(depth_image):
    """Get the occupancy of the image sample i.e.,
    there where there is finite depth and therefore a surface.
    depth_image : nparray of the depth image where +ve values in the alpha channel
                  indicate finite depth
    """
    if depth_image.ndim == 3:
        depth_alpha = depth_image[..., -1]
    elif depth_image.ndim == 2:
        depth_alpha = depth_image
    else:
        raise ValueError(
            "Depth image of incompatible shape. Must be RGB or RGBD with 3 dimensions, \
            or grayscale with 2."
        )

    return depth_alpha > 0.0


def create_OLAT_samples_for_frame(flatten=False):
    """Renders raster images given the config for one image frame"""
    # Load Transforms
    with open(config.get("paths", "transforms_file"), "r") as tf:
        image_transforms = json.loads(tf.read())

    image_number = int(config.get("images", "image_number"))

    downsample_ratio = int(config.get("parameters", "downsample_ratio"))

    # Loading Images
    (
        W,
        H,
        image_albedo,
        image_normal,
        depth_remapped,
        image_rgbd,
        occupency_mask,
    ) = load_images(image_number, downsample_ratio=downsample_ratio)
    logging.info(f"Loaded images of size ({W},{H}).")

    # Loading Lights and Light Transforms
    lights_file_path = config.get("paths", "lights_file")
    with open(lights_file_path, "r") as lf:
        lights_info = json.loads(lf.read())

    logging.info(
        f"Loaded {lights_file_path}. \
        Number of lights is {len(lights_info['lights'])}"
    )

    light_transforms = {
        light["name_light"].lower(): np.array(light["transformation_matrix"])
        for light in lights_info["lights"]
    }
    light_locations = {
        name: to_translation(transform)
        for (name, transform) in light_transforms.items()
    }

    # Compute Camera Parameters
    intrinsics, c2w, R, T = get_camera_parameters(
        W, H, image_transforms["camera_angle_x"], image_number, image_transforms
    )
    check_rotation(R)

    logging.info("Using the following intrinsics matrix:")
    logging.info("K = ")
    logging.info(intrinsics.intrinsic_matrix)

    # ## Creating Point Cloud
    logging.info(
        f"There are {occupency_mask.sum()} points with finite depth in the image."
    )

    # ### Using Open3D RGBD + Point Cloud
    posed_points = project_and_pose_3d_points_via_rgbd(image_rgbd, intrinsics, c2w)
    logging.debug(
        f"After posing, we have posed_points of shape {posed_points.shape} \
        and size {posed_points.size}"
    )

    # Raster Rendering

    # Get Normals
    camera_normals = get_camera_space_normals(image_normal)
    world_normals = get_world_space_normals(camera_normals, R)

    # do this for the given light orientations, populate dict for output
    # output_rasters = {}
    # for light_name, light_location in light_locations.items():
    #     raster_image = compute_raster(world_normals, albedo, occupied_mask,
    #         posed_points, light_location)
    #     output_rasters.update(light_name, raster_image)

    # Normalize Albedo to [0.1]
    albedo = image_albedo / 255.0
    logging.debug(
        f"in outter: albedo is {albedo.dtype} \
        in range [{albedo.min()}, { albedo.max() }]"
    )
    output_rasters = [
        [
            light_name,
            compute_raster(world_normals, albedo, posed_points, light_location, T)[0],
        ]
        for (light_name, light_location) in light_locations.items()
    ]

    return output_rasters, occupency_mask, W, H


if __name__ == "__main__":
    raster_images, occupency_mask, W, H = create_OLAT_samples_for_frame()

    for light_name, image in raster_images:
        logging.debug(f"{light_name}: {image.min()}, {image.mean()}, {image.max()}")
        output_name = f"raster_{light_name}.png"
        image_container = np.ones((W, H, 3))
        image_container[occupency_mask] = image
        plt.imsave(output_name, image_container)
