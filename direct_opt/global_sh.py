"""Directly optimize the coeficients of a global spherical basis functions
to be able to reproduce global illumination, foregoing an MLP core.
"""

import math
from matplotlib import pyplot as plt
from rich.table import Table as RichTable
from rich.console import Console
import argparse

import numpy as np
import torch
import torch.autograd.anomaly_mode
import torch.nn.functional as F
import wandb
import wandb.plot
from rich.traceback import install as install_rich
from tqdm import tqdm

from ile_utils.config import Config
from ile_utils.get_device import get_device

# Must read in the config before other imports override it
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_path", help="Path to the config file", required=True
    )

    args = parser.parse_args()

    # Readin the proper config.
    config = Config.get_config(args.config_path)


from data_loaders import olat_render as ro
from data_loaders.datasets import (
    IntrinsicDiffuseDataset,
    IntrinsicGlobalDataset,
    OLATDataset,
    unpack_item,
)
from data_loaders.get_dataloader import get_dataloader
from data_loaders.metadata_loader import get_camera_orientation
from icecream import ic
from log import get_logger
from losses.metrics import psnr
from spherical_harmonics.sph_harm import render_second_order_SH
from spherical_harmonics.visualize import visualie_SH_on_3D_sphere, evaluate_SH_on_sphere
from spherical_harmonics.sampling import sample_uniform_sphere

install_rich()

device = get_device()
rng = np.random.default_rng(882723)

logger = get_logger(__file__)
Config.log_config(logger)

# Amend the icecream printing function to
ic.configureOutput(outputFunction=lambda s: logger.info(s))


def train_epoch(
    epoch, train_dl, sh_coeff, optimizer, n_batches_per_epoch, coeff_evolution_data
):
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
        feats, _ = unpack_item(item, type(train_dataset))

        # Move to device
        feats = feats.to(device)

        with torch.autograd.anomaly_mode.detect_anomaly():
            # forward pass
            batch_train_loss, batch_psnr = do_forward_pass(
                sh_coeff, feats, type(train_dataset)
            )
            cumu_loss += batch_train_loss.item()
            cumu_psnr += batch_psnr

            # Optimization step
            optimizer.zero_grad()
            batch_train_loss.backward()
            optimizer.step()

        # Collect metrics
        epoch_step = batch + 1 + (n_batches_per_epoch * epoch)
        epoch_progress = epoch_step / n_batches_per_epoch
        metrics = {
            "train/train_loss": batch_train_loss,
            "train/train_psnr": batch_psnr,
            "train/epoch": epoch_progress,
        }

        # Log the evolution of the coeffs themselves for later
        if epoch_step % wandb.config["record_freq"] == 0:
            coeff_evolution_data["xs"].append(epoch_progress)
            coeff_evolution_data["ys"].append(sh_coeff.clone().detach().cpu().numpy())

        if batch + 1 < n_batches_per_epoch:
            # 🐝 Log train metrics to wandb
            wandb.log(metrics)

    # Epoch metrics
    avg_batch_loss = cumu_loss / len(train_dl)
    avg_psnr = cumu_psnr / len(train_dl)
    epoch_metrics = {"train/avg_loss": avg_batch_loss, "train/avg_psnr": avg_psnr}
    wandb.log(epoch_metrics)


def validate_model(sh_coeff, valid_dl):
    """Compute performance of the model on the validation dataset and log a wandb.Table

    Args:
        sh_coeff (torch.Tensor): tensor of second order spherical harmonics coefficients
        valid_dl (torch.Dataloader): dataloader of the validation dataset

    """
    val_loss_acc = 0.0
    val_psnr_acc = 0.0
    with torch.inference_mode():
        for item in tqdm(valid_dl, desc="Validating model", total=len(valid_dl)):
            feats, _ = unpack_item(item, type(valid_dataset))

            feats = feats.to(device)
            samples_in_batch = feats.size(0)

            # Forward pass
            val_loss, val_psnr = do_forward_pass(sh_coeff, feats, type(valid_dataset))

            val_loss_acc += val_loss.item() * samples_in_batch
            val_psnr_acc += val_psnr * samples_in_batch

    samples_in_set = len(valid_dl.dataset)

    # Log the metrics
    val_metrics = {
        "val/loss": val_loss_acc / samples_in_set,
        "val/psnr": val_psnr_acc / samples_in_set,
    }
    wandb.log(val_metrics)


def nornalize_to_canonical_range(x):
    """Scale and shift the array to the canonical range [0..1] with full support.
    Output's range will be [0..1], always.

    Args:
        x (ndarray or torch.Tensor): array of values

    Returns:
        Array/Tensor of values in x scaled to [0..1]. Will always have full range.
    """
    xmax, xmin = x.max(), x.min()
    return (x - xmin) / (xmax - xmin)


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
    shading = render_second_order_SH(sh_coeff, normals, torch_mode)
    clipped_shading = lib.clip(shading, 0.0, 1.0)
    pixel = ro.shade_albedo(albedo, clipped_shading, torch_mode)
    return (pixel, shading) if return_shading else pixel


def do_forward_pass(sh_coeff, feats, dataset_type):
    # Deconstruct feats into components, albedo, normal, shading, raster_img

    gt_rgb, albedo, _, normals = dataset_type.unpack_item_batch(feats)

    # render pixel with SH:
    # NOTE: We should broadcast the coeffs to num of normals here,
    # or hope it will be done automatically.
    pred_rgb = render_pixel_from_sh(sh_coeff, normals, albedo)
    assert isinstance(pred_rgb, torch.Tensor)

    # Compute reconstruction loss:
    train_loss = F.mse_loss(gt_rgb, pred_rgb)

    # Non negativity constraint
    if wandb.config.get("non_negativity_loss", False):
        uniform_on_sphere = sample_uniform_sphere(rng)
        uniform_SH_evaluations = render_second_order_SH(
            sh_coeff, torch.tensor(uniform_on_sphere)
        )
        non_negativity_loss = torch.mean(
            torch.square(
                torch.minimum(
                    torch.zeros_like(uniform_SH_evaluations), uniform_SH_evaluations
                )
            )
        )
        train_loss += non_negativity_loss

        # Log the non-neg loss to see evolution
        wandb.log({"train/non-neg-loss": non_negativity_loss})

    train_psnr = psnr(train_loss.item())

    return train_loss, train_psnr


def visualize_scene(frame_number, sh_coeff, dataset):
    # Make top row of infered images
    # load attributes of this validation image
    gt_attributes, occupancy_mask = dataset.get_frame_decomposition(frame_number)

    _, albedo, _, world_normals = gt_attributes

    val_render_pixels, val_shading = render_pixel_from_sh(
        sh_coeff.numpy(),
        world_normals.numpy(),
        albedo.numpy(),
        torch_mode=False,
        return_shading=True,
    )

    assert dataset.dim is not None
    W, H = dataset.dim

    val_render_image = ro.reconstruct_image(
        W, H, val_render_pixels, occupancy_mask, add_alpha=True
    )

    val_shading = np.clip(
        val_shading, 0.0, 1.0
    )  # Clip the shading for proper visualization.
    val_shading_image = ro.reconstruct_image(
        W, H, val_shading, occupancy_mask, add_alpha=True
    )

    # Stick them together
    gt_render_image, _, gt_shading_image, _ = dataset.get_frame_images(frame_number)

    shading_col = np.concatenate([val_shading_image, gt_shading_image], axis=0)
    render_col = np.concatenate([val_render_image, gt_render_image], axis=0)

    return shading_col, render_col


# TODO: move to image gen
def generate_validation_artefacts_from_global_sh(
    val_step, sh_coeff, test_dataset, shading_histogram_data
):
    """Generate and log post-epoch metrics and visualizations

    Args:
        val_step (int): epoch index
        sh_coeff (tensor): tensor of SH coefficients
        test_dataset (dataset object): dataset to source visualization frames from
    """
    sh_coeff.requires_grad_(False)
    with torch.inference_mode():
        # Randomly choose which image from the validation set to reconstruct
        # frame_number = rng.integers(valid_dataset.num_frames)
        vis_split = config.get("visualization", "split", fallback="test")
        front_frame, back_frame, *_ = [
            int(i.strip())
            for i in config.get("visualization", "indexes", fallback="39,89").split(",")
        ]

        # SH VIS ON SPHERE
        R_front = get_camera_orientation(front_frame, split=vis_split)
        R_back = get_camera_orientation(back_frame, split=vis_split)
        fig = visualie_SH_on_3D_sphere(
            sh_coeff.numpy(), camera_orientations=[R_front, R_back], show_extremes=True
        )
        wandb.log(
            {
                "vis/sh_sphere": wandb.Image(
                    fig, caption=f"SH Evaluation on sphere after epoch {val_step}."
                )
            }
        )
        # END SH VIS ON SPHERE

        # HISTOGRAM OF SHADING DATA
        sh_values, _ = evaluate_SH_on_sphere(sh_coeff.numpy())
        shading_histogram_data.append(sh_values)
        # END HISTOGRAM

        # RENDER IMAGES OF SCENE
        front_shading, front_render = visualize_scene(
            front_frame, sh_coeff, test_dataset
        )
        back_shading, back_render = visualize_scene(back_frame, sh_coeff, test_dataset)
        shading_image_array = np.concatenate([front_shading, back_shading], axis=1)
        render_image_array = np.concatenate([front_render, back_render], axis=1)

        image_caption = f"Top: Inference. Bottom: GT.\n(After epoch {val_step})."

        shading_image = wandb.Image(
            shading_image_array, caption="Clipped Shading Images\n" + image_caption
        )
        render_image = wandb.Image(
            render_image_array, caption="Renreded Images\n" + image_caption
        )

        val_metrics = {
            "vis/shading_images": shading_image,
            "vis/render_images": render_image,
        }

        wandb.log(val_metrics)


def get_dataset(config, downsample_ratio=1, split="train"):
    dataset_option = config.get("dataset", "type", fallback="intrinsic-global")
    logger.debug(f"Dataset option is {dataset_option}")

    if dataset_option == "single_OLAT":
        return OLATDataset(config, split, is_single_olat=True)

    if dataset_option == "intrinsic-global":
        return IntrinsicGlobalDataset(config, split)

    if dataset_option == "intrinsic-diffuse":
        return IntrinsicDiffuseDataset(config, split)

    raise ValueError(
        f"Dataset option {dataset_option} is unsuported in this optimzation."
        "Use one of ['single_OLAT', 'intrinsic-diffuse', 'intrinsic-global']."
    )


def experiment_run():
    subset_fraction = wandb.config["subset_fraction"]
    logger.info(f"Training using {1.0/subset_fraction*100:.2f}% of the data.")

    # Training Set
    train_dl = get_dataloader(
        train_dataset,
        batch_size=wandb.config["batch_size"],
        subset_fraction=subset_fraction,
        shuffle=wandb.config["shuffle_train"],
    )
    logger.info(
        f"Train dataloader with {len(train_dl)} batches"
        f" and {len(train_dl.dataset)}/{len(train_dataset)} samples"
        f" from {train_dataset.num_frames} frames."
    )

    # Validation Set
    valid_dl = get_dataloader(
        valid_dataset,
        batch_size=wandb.config["batch_size"],
        subset_fraction=subset_fraction,
    )
    logger.info(
        f"Loaded validation dataloader with {len(valid_dl)} batches"
        f" and {len(valid_dl.dataset)}/{len(valid_dataset)} samples"
        f" from {valid_dataset.num_frames} frames."
    )

    # Test set
    test_dl = get_dataloader(
        test_dataset,
        batch_size=wandb.config["batch_size"],
        subset_fraction=subset_fraction,
    )
    logger.info(
        f"Loaded test dataset with {len(test_dl)} batches"
        f" and {len(test_dl.dataset)}/{len(test_dataset)} samples"
        f" from {test_dataset.num_frames} frames."
    )

    # Populate the table with data

    # TODO: Move this to Models
    # initialize SH coefficients
    # TODO: Add initialization options (experiemnt configs)

    # SH Coeffs initialization
    init_file = wandb.config.get("sh_init", None)
    if init_file is None:
        sh_coeff = torch.zeros(9)  # TODO: Make this be dept on an order setting
    else:
        sh_coeff = torch.tensor(np.fromfile(init_file))
        assert (
            sh_coeff.shape[0] == 9
        ), f"SH inisialization must be of length 9, was {sh_coeff.shape[0]}"

    # Log initializatoin
    column_names = [f"C{i}" for i in range(len(sh_coeff))]

    # Create a Rich Table to dislay SH coeffs  at the end
    sh_coeff_display_table = RichTable("Row", *column_names, title="SH Coefficients")
    sh_coeff_display_table.add_row(
        "Initial", *[f"{float(c):.3f}" for c in sh_coeff.tolist()]
    )

    # Initialization pertubation
    # If it is set to zero, it still couts as enabled
    if "init_pertubation" in wandb.config:
        init_perturb_strength = wandb.config["init_pertubation"]
        pertubation = torch.rand_like(sh_coeff) * init_perturb_strength
        sh_coeff += pertubation

        sh_coeff_display_table.add_row(
            "Pertubation", *[f"{float(c)}" for c in pertubation.tolist()]
        )

        sh_coeff_display_table.add_row(
            "Init + Pertubation", *[f"{float(c):.3f}" for c in sh_coeff.tolist()]
        )

    # nn.init.normal_(sh_coeff)
    logger.info("Initializing Spherical Harmonics coefficients to:")
    logger.info(sh_coeff)

    # TODO: Add LR scheduling
    # Set the coeffs as parameters for optimization
    optimizer = torch.optim.RMSprop([sh_coeff])

    n_batches_per_epoch = math.ceil(len(train_dl.dataset) / wandb.config["batch_size"])

    # Make table to hold post-epoch coeffs
    coeff_evolution_data = {"xs": [], "ys": []}
    shading_histogram_data = []

    # for each epoch, for each batch,
    # render the pixel using the normal and the coeficients of the SH,
    # produce the reconstruction loss.
    for epoch in tqdm(
        range(wandb.config["epochs"]), total=wandb.config["epochs"], desc="Epoch"
    ):
        # Set coeffs in training mode
        sh_coeff.requires_grad_()

        # Run and log the training on an epoch
        train_epoch(
            epoch,
            train_dl,
            sh_coeff,
            optimizer,
            n_batches_per_epoch,
            coeff_evolution_data,
        )

        # Run and log the validation of the model after epoch
        validate_model(sh_coeff, valid_dl)

        # Render a validation image and log it to wandb
        generate_validation_artefacts_from_global_sh(
            epoch, sh_coeff, test_dataset, shading_histogram_data
        )

    # Compute the test error and metrics
    # TODO: Add test metric loop

    # Histogram plotting
    shading_histogram_data = np.stack(shading_histogram_data, axis=1)
    ic(shading_histogram_data.shape)
    logger.info("Making shading values histogram...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    _, bins, _ = ax.hist(
        shading_histogram_data,
        stacked=True,
        histtype="bar",
        ec="black",
        label=[f"Epoch {i}" for i in range(wandb.config["epochs"])],
    )
    ax.set_xticks(bins)
    ax.legend(prop={"size": 10})
    ax.set_title(r"Shading values over $S^2$")
    wandb.log({"val/shading_hist": wandb.Image(fig)})
    logger.info("Done.")

    # Ceoff evolution plot:
    wandb.log(
        {
            "train/coeff_evolution": wandb.plot.line_series(
                coeff_evolution_data["xs"],
                np.stack(coeff_evolution_data["ys"]).T,
                keys=column_names,
                title="Coefficient Evolution",
                xname="Epoch",
            )
        }
    )

    # Save the coefficients produced
    # FIXME: Should index theses with the Y_lm notation
    optimized_coeff_table = wandb.Table(columns=column_names)
    optimized_coeff_table.add_data(
        *list(sh_coeff.numpy())
    )  # Only add the optimized values
    wandb.log({"Post-training SH Coefficinets": optimized_coeff_table})

    sh_coeff_display_table.add_row(
        "Optimized", *[f"{float(c):.3f}" for c in sh_coeff.tolist()]
    )
    Console().print(sh_coeff_display_table)


if __name__ == "__main__":
    wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_path", help="Path to the config file", required=True
    )

    args = parser.parse_args()

    # Readin the proper config.
    config = Config.get_config(args.config_path)
    Config.log_config(logger)

    # Experiment meta-data confids
    project_name = config.get("experiment", "project_name")
    run_name_prefix = config.get("experiment", "run_name_prefix")
    num_runs = config.getint("experiment", "num_runs", fallback=1)
    record_freq = config.getint("experiment", "record_frequency", fallback=50)

    # Dataset configs
    dataset_subset_fraction = config.getint("dataset", "subset_fraction", fallback=1)
    dataset_downsample_ratio = config.getint("dataset", "downsample_ratio", fallback=2)

    num_epochs = config.getint("training", "epochs", fallback=1)
    shuffle_train = config.getboolean("training", "shuffle_train")

    # SH configs
    initialization_vector_file = config.get(
        "global_spherical_harmonics", "sh_initialization", fallback=None
    )
    # Get pertubation if any
    pertubations = config.get(
        "global_spherical_harmonics", "sh_initialization_purtubation", fallback="0.0"
    )
    pertubations = [float(p.strip()) for p in pertubations.split(",")]

    # get the non-negativity constraint switch
    non_negativity_constraint = config.getboolean(
        "global_spherical_harmonics", "non_negativity_constraint", fallback=True
    )

    # Load in the datasets
    train_dataset = get_dataset(config)
    valid_dataset = get_dataset(config, split="val")
    test_dataset = get_dataset(config, split="test")

    for str_i, ptrb in enumerate(pertubations):
        run_name = run_name_prefix
        if len(pertubations) > 1:
            run_name += f"_purtstr{str_i}"

        # Set this run's confg
        run_config = {
            "epochs": num_epochs,
            "batch_size": 1024,
            "subset_fraction": dataset_subset_fraction,
            "downsample_ratio": dataset_downsample_ratio,
            "non_negativity_constraint": non_negativity_constraint,
            "shuffle_train": shuffle_train,
            "sh_init": initialization_vector_file,
            "init_pertubation": ptrb,
            "record_freq": record_freq,
        }

        for run_i in range(num_runs):
            run = wandb.init(
                project=project_name, name=f"{run_name}_run{run_i}", config=run_config
            )
            assert run is not None

            with run:
                experiment_run()
