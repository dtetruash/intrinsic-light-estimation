"""Directly optimize the coeficients of a global spherical basis function
to be able to reproduce global illumination, foregoing an MLP core.
"""
import math

import numpy as np
import torch
import torch.autograd.anomaly_mode
import torch.nn as nn
import torch.nn.functional as F
import wandb
from data_loaders import olat_render as ro
from data_loaders.datasets import (
    IntrinsicDiffuseDataset,
    IntrinsicGlobalDataset,
    IntrinsicDataset,
    OLATDataset,
    unpack_item,
)
from data_loaders.get_dataloader import get_dataloader
from icecream import ic
from ile_utils.config import Config
from ile_utils.get_device import get_device
from log import get_logger
from losses.metrics import psnr
from rich.traceback import install as install_rich
from tqdm import tqdm

from spherical_harmonics import spherical_harmonics as sh

install_rich()

device = get_device()
config = Config.get_config()
# Each main optimization file should have a logger.
# TODO: This logget should be passed down to all calls below this one somehow.
logger = get_logger(__file__)
rng = np.random.default_rng(882723)

Config.log_config(logger)

ic.configureOutput(outputFunction=lambda s: logger.info(s))


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

    for batch, item in pbar:
        # Decompose items in dataset
        feats, _ = unpack_item(item, type(train_dl.dataset))

        # Move to device
        feats = feats.to(device)

        with torch.autograd.anomaly_mode.detect_anomaly():
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
def render_pixel_from_sh(
    sh_coeff, normals, albedo, torch_mode=True, return_shading=False
):
    """Render a pixel color from a second order spherical harmonics basis function
    normal and albedo at the surface.

    Args:
        sh_coeff (torch.Tensor or ndarray): second order spherical harmonic coefficients
        normals (torch.Tensor or ndarray): world normals at each pixel location (N,3)
        albedo (torch.Tensor or ndarray): albedo at each pixel location (N,3)
        torch_mode (bool): flag switch to use torch instead of numpy

    Returns:
        Torch.tensor of rendered pixel colors (clipped to [0..1] in each channel).
    """
    lib = torch if torch_mode else np
    shading = sh.render_second_order_SH(sh_coeff, normals, torch_mode)
    clipped_shading = lib.clip(shading, 0.0, 1.0)
    pixel = ro.shade_albedo(albedo, clipped_shading, torch_mode)
    return (pixel, clipped_shading) if return_shading else pixel


def do_forward_pass(sh_coeff, feats):
    # Deconstruct feats into components, albedo, normal, shading, raster_img
    # FIXME: Check this against the current output of the datasets.
    normals, albedo, gt_rgb = feats.unbind(-1)
    # print(f"normals were {normals.shape}")
    # print(f"albedo were {albedo.shape}")
    # print(f"gt_rgb were {gt_rgb.shape}")

    # render pixel with SH:
    # Do this by computing the shading via the sh code,
    # and calling the render function
    # NOTE: We should broadcast the coeffs to num of normals here,
    # or hope it will be done automatically.
    pred_rgb = render_pixel_from_sh(sh_coeff, normals, albedo)
    assert isinstance(pred_rgb, torch.Tensor)

    # Compute reconstruction loss:
    train_loss = F.mse_loss(gt_rgb, pred_rgb)
    train_psnr = psnr(train_loss)

    return train_loss, train_psnr


# TODO: move to image gen
def generate_validation_image_from_global_sh(sh_coeff, valid_dataset):
    """Generate an image comparing a ground truth image
    with one generated using the model.
    model: MLP which outputs light direction vectors"""
    sh_coeff.requires_grad_(False)
    with torch.inference_mode():
        # Randomly choose which image from the validation set to reconstruct
        frame_number = rng.integers(valid_dataset.num_frames)

        # if this is a single light dataset,
        # NOTE: We need to suply the light when we have only a single OLAT
        # Dataset, right?

        # load attributes of this validation image
        gt_images = valid_dataset.get_frame_images(frame_number)
        gt_attributes, occupancy_mask = valid_dataset.get_frame_decomposition(
            frame_number
        )

        world_normals = gt_attributes["normal"]
        albedo = gt_attributes["albedo"]

        # Raster pixel image
        val_raster_pixels, val_shading = render_pixel_from_sh(
            sh_coeff.numpy(),
            world_normals,
            albedo,
            torch_mode=False,
            return_shading=True,
        )

        assert valid_dataset.dim is not None
        W, H = valid_dataset.dim

        val_shading_image = ro.reconstruct_image(W, H, val_shading, occupancy_mask)
        val_render_image = ro.reconstruct_image(W, H, val_raster_pixels, occupancy_mask)

        ic(
            val_shading_image.min(),
            val_shading_image.max(),
            val_shading_image.dtype,
        )
        ic(
            val_shading_image.min(),
            val_shading_image.max(),
            val_shading_image.dtype,
        )
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
        validation_row = np.concatenate([val_render_image, val_shading_image], axis=1)

        logger.debug(f"Image dict has keys: {list(gt_images.keys())}")

        if isinstance(valid_dataset, IntrinsicDiffuseDataset):
            combined_index = "diffuse"
        elif isinstance(valid_dataset, IntrinsicGlobalDataset):
            combined_index = "full"
        else:
            # FIXME: Check that this is the correct index
            combined_index = "image"

        gt_raster_image = ro.remove_alpha(gt_images[combined_index])
        gt_shading_image = ro.remove_alpha(gt_images["shading"])

        ic(
            gt_raster_image.min(),
            gt_raster_image.max(),
            gt_raster_image.dtype,
        )

        ic(
            gt_shading_image.min(),
            gt_shading_image.max(),
            gt_shading_image.dtype,
        )

        gt_row = np.concatenate([gt_raster_image, gt_shading_image], axis=1)

        image_array = np.concatenate([validation_row, gt_row], axis=0)

        image_caption = "Top row : Inference. Bottom: GT.\n\
        Left to right: Render, Shading."

        return image_array, image_caption


def get_dataset(config, split="train"):
    dataset_option = config.get(
        "spherical_harmonics", "dataset", fallback="intrinsic-global"
    )
    logger.debug(f"Dataset option is {type(dataset_option)}:{dataset_option}")

    if dataset_option == "single_OLAT":
        return OLATDataset(config, split, is_single_olat=True)

    if dataset_option == "intrinsic-global":
        return IntrinsicGlobalDataset(config, split)

    if dataset_option == "intrinsic-diffuse":
        return IntrinsicDiffuseDataset(config, split)

    raise ValueError(
        (
            f"Dataset option {dataset_option} is unsuported in this optimzation."
            "Use one of ['single_OLAT', 'intrinsic-diffuse', 'intrinsic-global']."
        )
    )


def main():
    # TODO: Need a way to make this modular w.r.t. which dataset is used.
    # Something like: get_dataset between single OLAT or combined
    # create the data loader and load the data
    train_dl = get_dataloader(
        get_dataset(config),
        batch_size=wandb.config["batch_size"],
        subset_fraction=1,
    )
    assert isinstance(train_dl.dataset, IntrinsicDataset) or isinstance(
        train_dl.dataset, OLATDataset
    )  # appease the type checker
    logger.info(f"Frames: {train_dl.dataset.num_frames}.")
    logger.info(
        f"Loaded train dataset with {len(train_dl)}"
        f" batches and {len(train_dl.dataset)} samples"
        f" and {train_dl.dataset.num_frames} frames."
    )

    valid_dl = get_dataloader(  # noqa: F841
        get_dataset(config, split="val"),
        batch_size=wandb.config["batch_size"],
        subset_fraction=1,
    )

    # TODO: Move this to Models
    # initialize SH coefficients
    sh_coeff = torch.zeros(9)
    nn.init.normal_(sh_coeff)

    # TODO: Add LR scheduling
    # Set the coeffs as parameters for optimization
    optimizer = torch.optim.RMSprop([sh_coeff])

    n_batches_per_epoch = math.ceil(len(train_dl.dataset) / wandb.config["batch_size"])

    # for each epoch, for each batch,
    # render the pixel using the normal and the coeficients of the SH,
    # produce the reconstruction loss.
    for epoch in tqdm(
        range(wandb.config["epochs"]), total=wandb.config["epochs"], desc="Epoch"
    ):
        # Set coeffs in training mode
        sh_coeff.requires_grad_()

        avg_loss, avg_psnr = train_epoch(
            epoch, train_dl, sh_coeff, optimizer, n_batches_per_epoch
        )
        wandb.log({"train/avg_loss": avg_loss, "train/avg_psnr": avg_psnr})

        # TODO: Add validation loop here to eval loss and
        # Render a validation image and log it to wandb
        # FIXME: This produces falty images probably due to uncplipped values
        val_image_array, image_caption = generate_validation_image_from_global_sh(
            sh_coeff, valid_dl.dataset
        )

        # TODO: Add alert if the images were strage?

        val_image = wandb.Image(val_image_array, caption=image_caption)

        val_metrics = {"val/images": val_image}
        wandb.log(val_metrics)


enable_wandb = True

if __name__ == "__main__":
    wandb.config = {
        "epochs": 1,
        "batch_size": 1024,
    }
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
