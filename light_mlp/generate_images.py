import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib.colors import Normalize

from data_loaders import olat_render as ro
from spherical_harmonics import spherical_harmonics as sh
from losses.loss_functions import cosine_similarity_module

rng = np.random.default_rng(882723)

heatmap_image_numbers = None
heatmap_light_name = None


def set_heatmap_params(number_of_images, light_names):
    global heatmap_image_numbers
    if heatmap_image_numbers is None:
        heatmap_image_numbers = rng.integers(number_of_images, size=6)

    global heatmap_light_name
    if heatmap_light_name is None:
        heatmap_light_name = light_names[rng.integers(len(light_names))]


def generate_validation_image(model, model_class, valid_dataset):
    """Generate an image comparing a ground truth image with one generated using the model.
    model: MLP which outputs light direction vectors"""
    model.eval()
    with torch.inference_mode():
        # Randomly choose which image from the validation set to reconstruct
        image_number = rng.integers(valid_dataset.num_frames)
        # randomly choose a light in the scene
        light_names = list(valid_dataset._lights_info.keys())
        light_name = light_names[rng.integers(valid_dataset._num_lights)]
        print(
            f"Generating OLAT validation image {image_number} with light {light_name}..."
        )

        # load attributes of this validation image
        (
            W,
            H,
            gt_raster_pixels,
            world_normals,
            albedo,
            _,
            gt_light_vectors,
            occupancy_mask,
        ) = valid_dataset.attributes[image_number][light_name]

        gt_raster_pixels = gt_raster_pixels.astype(np.float32)
        world_normals = world_normals.astype(np.float32)
        albedo = albedo.astype(np.float32)
        gt_light_vectors = gt_light_vectors.astype(np.float32)

        # prepare inputs for inference
        feats = np.stack([world_normals, albedo, gt_raster_pixels], axis=1)
        inputs = torch.flatten(torch.as_tensor(feats).float(), start_dim=1)

        # Do inference to get light representation
        outputs = model(inputs).numpy().astype(np.float32)

        # Construct a normals image
        img_size = (W, H, 3)

        val_raster_image = np.ones(img_size, dtype=np.float32)
        val_light_dir_image = np.ones(img_size, dtype=np.float32)
        val_shading_image = np.ones(img_size, dtype=np.float32)

        if model_class == "light-mlp":
            light_vectors = outputs

            # Light direciton images
            val_light_colors = 0.5 * light_vectors + 0.5
            val_light_dir_image[occupancy_mask] = val_light_colors

            # Raster pixel images
            val_raster_pixels, val_shading = ro.render_from_directions(
                light_vectors, albedo, world_normals, return_shading=True
            )

        elif model_class == "light-sh":
            light_harmonics = outputs

            # Raster pixel image
            val_shading = sh.render_second_order_SH(light_harmonics, world_normals)
            val_raster_pixels = ro.shade_albedo(albedo, val_shading)

        else:
            raise ValueError(f"Unknown model class {model_class}")

        # TODO: Replace all of these with ro.reconstruct_image function

        # raster pixel image
        val_raster_image[occupancy_mask] = val_raster_pixels

        # Shading images
        val_shading_image[occupancy_mask] = val_shading[..., np.newaxis]

        gt_light_dir_image = np.ones(img_size, dtype=np.float32)
        gt_light_colors = 0.5 * gt_light_vectors + 0.5
        gt_light_dir_image[occupancy_mask] = gt_light_colors

        gt_raster_image = np.ones(img_size, dtype=np.float32)
        gt_raster_image[occupancy_mask] = gt_raster_pixels

        gt_shading_image = np.ones(img_size, dtype=np.float32)
        gt_shading = ro.compute_clipped_dot_prod(world_normals, gt_light_vectors)
        gt_shading_image[occupancy_mask] = gt_shading[..., np.newaxis]

        # TODO: add error image loss.
        # heatmap_image = np.concatenate([generate_heatmap_image(model, valid_dataset, image_number, light_name), np.ones(img_size)], axis=0)

        # Stick them together
        validation_images = np.concatenate(
            [val_raster_image, val_shading_image, val_light_dir_image], axis=1
        )

        gt_images = np.concatenate(
            [gt_raster_image, gt_shading_image, gt_light_dir_image], axis=1
        )

        image_array = np.concatenate([validation_images, gt_images], axis=0)
        # image_array = np.concatenate([image_array, heatmap_image], axis=1)

        image_caption = "Top row : Inference. Bottom: GT.\nLeft to right: Render, Shading, Light directions."

        return image_array, image_caption


def get_heatmap_color_array(data_array, cmap_name="hot", dmin=-1.0, dmax=1.0):
    """Color the scalar data_array acording to a color map."""
    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=dmin, vmax=dmax)
    rgb_values = cmap(norm(data_array))[..., :-1]
    return rgb_values


def generate_heatmap_image(model, valid_dataset, image_number, light_name):
    """Create a single heatmap image via matplotlib (for now)"""
    # FIXME: Write this method once the global-sh script works.
    raise NotImplementedError("This method must be reworked.")
    model.eval()
    with torch.inference_mode():
        # get light dirs for given image
        (
            W,
            H,
            gt_raster_pixels,
            world_normals,
            albedo,
            _,
            gt_light_vectors,
            occupancy_mask,
        ) = valid_dataset.attributes[image_number][light_name]

        feats = np.stack([world_normals, albedo, gt_raster_pixels], axis=1)
        inputs = torch.flatten(torch.as_tensor(feats).float(), start_dim=1)

        # Do inference to get light vectors
        light_vectors = model(inputs)
        light_vectors = light_vectors.numpy().astype(np.float32)

        # compute per pixel cosine similarity
        cosine_similarity = (
            cosine_similarity_module(light_vectors, gt_light_vectors).float().numpy()
        )

        # colorize the similarity to get image array,
        colorized_cosine = get_heatmap_color_array(cosine_similarity)

        img_size = (W, H, 3)
        heatmap_image = np.ones(img_size, dtype=np.float32)
        heatmap_image[occupancy_mask] = colorized_cosine

        return heatmap_image


def generate_heatmap_image_grid(model, valid_dataset):
    raise NotImplementedError("This method must be reworked")
    image_number_max = valid_dataset.num_frames
    light_names = list(valid_dataset._lights_info.keys())
    set_heatmap_params(image_number_max, light_names)

    return np.concatenate(
        [
            generate_heatmap_image(model, valid_dataset, im, heatmap_light_name)
            for im in heatmap_image_numbers
        ],
        axis=0,
    )
