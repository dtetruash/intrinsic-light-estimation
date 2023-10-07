"""Directly optimize the coeficients of a global spherical basis function
to be able to reproduce global illumination, foregoing an MLP core.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from data_loaders import raster_relight as rr
from data_loaders.get_dataloader import get_dataloader
from data_loaders.datasets import OLATDataset
from ile_utils.get_device import get_device
from ile_utils.config import Config
from rich.traceback import install as install_rich
from sperical_harmonics import spherical_harmonics as sh
from tqdm import tqdm

install_rich()

device = get_device()
config = Config.get_config()

rng = np.random.default_rng(882723)


# TODO: move this to losses
def psnr(mse):
    return -10.0 * math.log10(mse)


def train_epoch(epoch, train_dl, sh_coeff, optimizer, n_batches_per_epoch):
    cumu_loss = 0.0
    cumu_psnr = 0.0
    pbar = tqdm(
        enumerate(train_dl),
        total=len(train_dl),
        desc="Batch",
        position=1,
        leave=True,
        colour="red",
    )
    for batch, (feats, _) in pbar:
        # Move to device
        feats = feats.to(device)

        # forward pass
        batch_train_loss, batch_psnr = do_forward_pass(sh_coeff, feats)
        cumu_loss += batch_train_loss.item()
        cumu_psnr = batch_psnr

        # Optimization step
        optimizer.zero_grad()
        batch_train_loss.backward()
        optimizer.step()

        # Collect metrics
        epoch_step = batch + 1 + (n_batches_per_epoch * epoch)
        metrics = {
            "train/train_loss": batch_train_loss,
            "train/train_psnr": batch_psnr,
            "train/epoch": epoch_step / n_batches_per_epoch,
        }

        if batch + 1 < n_batches_per_epoch:
            # 🐝 Log train metrics to wandb
            wandb.log(metrics)

    avg_batch_loss = cumu_loss / len(train_dl)
    avg_psnr = cumu_psnr / len(train_dl)

    return avg_batch_loss, avg_psnr


# TODO: move this to SH module
def render_pixel_from_sh(sh_coeff, normals, albedo):
    shading = sh.render_second_order_SH(sh_coeff, normals)
    pixel = rr.shade_albedo_torch(albedo, shading)
    return pixel


def do_forward_pass(sh_coeff, feats):
    # Deconstruct feats into components, albedo, normal, shading, raster_img
    normals, albedo, gt_rgb = feats.unbind(-1)
    # print(f"normals were {normals.shape}")
    # print(f"albedo were {albedo.shape}")
    # print(f"gt_rgb were {gt_rgb.shape}")

    # render pixel with SH:
    # Do this by computing the shading via the sh code,
    # and calling the raster render function
    # NOTE: We should broadcast the coeffs to num of normals here,
    # or hope it will be done automatically.
    pred_rgb = render_pixel_from_sh(sh_coeff, normals, albedo)
    # Compute reconstruction loss:
    train_loss = F.mse_loss(gt_rgb, pred_rgb)
    train_psnr = psnr(train_loss)

    return train_loss, train_psnr


# TODO: move to image gen
def generate_validation_image(sh_coeff, valid_dataset):
    """Generate an image comparing a ground truth image
    with one generated using the model.
    model: MLP which outputs light direction vectors"""
    sh_coeff.requires_grad_(False)
    with torch.inference_mode():
        # Randomly choose which image from the validation set to reconstruct
        image_number = rng.integers(valid_dataset.num_frames)
        # randomly choose a light in the scene
        light_names = list(valid_dataset._lights_info.keys())
        light_name = light_names[rng.integers(valid_dataset._num_lights)]
        print(
            f"Generating OLAT validation image {image_number} \
            with light {light_name}..."
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
        torch.flatten(torch.as_tensor(feats).float(), start_dim=1)

        # Construct a normals image
        img_size = (W, H, 3)

        val_raster_image = np.ones(img_size, dtype=np.float32)
        val_light_dir_image = np.ones(img_size, dtype=np.float32)
        val_shading_image = np.ones(img_size, dtype=np.float32)

        # Raster pixel image
        val_shading = sh.render_second_order_SH(sh_coeff, world_normals)
        val_raster_pixels = rr.shade_albedo(albedo, val_shading)

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
        gt_shading = rr.compute_clipped_dot_prod(world_normals, gt_light_vectors)
        gt_shading_image[occupancy_mask] = gt_shading[..., np.newaxis]

        # TODO: add error image loss.
        # heatmap_image = np.concatenate(
        #     [
        #         generate_heatmap_image(
        #            model,
        #            valid_dataset,
        #            image_number,
        #            light_name
        #         ),
        #         np.ones(img_size),
        #     ],
        #     axis=0,
        # )

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


def get_dataset():
    raise NotImplementedError()
    dataset_option = config.get("sperical_harmonics", "dataset", fallback="intrinsic")
    if dataset_option == "single_OLAT":
        # make dataset which the id of the light to be used.
        pass
    elif dataset_option == "intrinsic":
        pass


def main():
    config = wandb.config
    # parse global config
    epochs = 1

    # TODO: Need a way to make this modular w.r.t. which dataset is used.
    # Something like: get_dataset between single OLAT or combined
    dataset = get_dataset()
    # create the data loader and load the data
    train_dl = get_dataloader(
        dataset,
        batch_size=config["batch_size"],
        subset_fraction=1,
    )

    valid_dataset = OLATDataset(split="val")

    print(
        f"Loaded train dataset with {len(train_dl)} \
        batches and {len(train_dl.dataset)} samples."
    )

    # TODO: Move this to Models
    # initialize SH coefficients
    sh_coeff = torch.zeros(9)
    nn.init.normal_(sh_coeff)

    # TODO: Add LR scheduling
    # Set the coeffs as parameters for optimization
    optimizer = torch.optim.RMSprop([sh_coeff])

    n_batches_per_epoch = math.ceil(len(train_dl.dataset) / config["batch_size"])

    # for each epoch, for each batch,
    # render the pixel using the normal and the coeficients of the SH,
    # produce the reconstruction loss.
    for epoch in tqdm(range(config["epochs"]), total=epochs, desc="Epoch"):
        # Set coeffs in training mode
        sh_coeff.requires_grad_()

        avg_loss, avg_psnr = train_epoch(
            epoch, train_dl, sh_coeff, optimizer, n_batches_per_epoch
        )
        wandb.log({"train/avg_loss": avg_loss, "train/avg_psnr": avg_loss})

        # TODO: Add validation loop here to eval loss and
        # Render a validation image and log it to wandb
        # FIXME: This produces falty images probably due to uncplipped values
        val_image_array, image_caption = generate_validation_image(
            sh_coeff, valid_dataset
        )
        val_image = wandb.Image(val_image_array, caption=image_caption)

        val_metrics = {"val/images": val_image}
        wandb.log(val_metrics)


enable_wandb = True

if __name__ == "__main__":
    if enable_wandb:
        wandb.login()

        with wandb.init(project="direct-opt-global-sh") as run:
            wandb.config = {
                "epochs": 1,
                "batch_size": 1024,
            }
            main()
    else:
        main()
