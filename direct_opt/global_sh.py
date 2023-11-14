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
from spherical_harmonics.sph_harm import (
    evaluate_second_order_SH,
    render_pixel_from_sh_unclipped,
)
from spherical_harmonics.visualize import (
    visualie_SH_on_3D_sphere,
    evaluate_SH_on_sphere,
    visualize_scene_frame_from_sh,
)
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


def do_forward_pass(sh_coeff, feats, dataset_type):
    # Deconstruct feats into components, albedo, normal, shading, raster_img

    gt_rgb, albedo, _, normals = dataset_type.unpack_item_batch(feats)

    # render pixel with SH:
    # NOTE: We should broadcast the coeffs to num of normals here,
    # or hope it will be done automatically.
    pred_rgb = render_pixel_from_sh_unclipped(sh_coeff, normals, albedo)
    assert isinstance(pred_rgb, torch.Tensor)

    # Compute reconstruction loss:
    train_loss = F.mse_loss(gt_rgb, pred_rgb)

    # Non negativity constraint
    if wandb.config.get("non_negativity_loss", False):
        uniform_on_sphere = sample_uniform_sphere(rng)
        uniform_SH_evaluations = evaluate_second_order_SH(
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
        front_shading, front_render = visualize_scene_frame_from_sh(
            front_frame, sh_coeff, test_dataset
        )
        back_shading, back_render = visualize_scene_frame_from_sh(
            back_frame, sh_coeff, test_dataset
        )
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

    # SH Coeffs initialization, if none given use zeros
    init_sh_file = wandb.config["sh_init"]
    if init_sh_file is not None:
        sh_coeff = torch.tensor(np.fromfile(init_sh_file))
        assert (
            sh_coeff.shape[0] == 9
        ), f"SH inisialization must be of length 9, was {sh_coeff.shape[0]}"
    else:
        sh_coeff = torch.zeros(9)

    # Log initializatoin
    column_names = [f"C{i}" for i in range(len(sh_coeff))]

    # Create a Rich Table to dislay SH coeffs  at the end
    sh_coeff_display_table = RichTable("Row", *column_names, title="SH Coefficients")
    sh_coeff_display_table.add_row(
        "Initial", *[f"{float(c):.3f}" for c in sh_coeff.tolist()]
    )

    # Initialization pertubation
    # If it is set to zero, it still couts as enabled
    perturb_init_file = wandb.config["pertubation_init"]
    if perturb_init_file is not None:
        perturb_strength = wandb.config["pertubation_strength"]
        ic(pertubations_file)

        pertubation = torch.tensor(np.fromfile(perturb_init_file)) * perturb_strength
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

    # Set the coeffs as parameters for optimization
    if "lr" in wandb.config:
        learning_rate = wandb.config["lr"]
    else:
        learning_rate = 0.01  # Default of RSMProp

    optimizer = torch.optim.RMSprop([sh_coeff], lr=learning_rate)
    logger.info(f"Using RSMProp optimizer with learning rate {learning_rate}.")

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
    shading_histogram_data_array = np.stack(shading_histogram_data, axis=1)
    # ic(shading_histogram_data.shape)
    logger.info("Making shading values histogram...")
    ic(shading_histogram_data_array.shape)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    _, bins, _ = ax.hist(
        shading_histogram_data_array,
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

    # log the histogram data
    # TODO: Test and fix me
    hist_table = wandb.Table(
        columns=[f"Epoch {i}" for i in range(wandb.config["epochs"])]
    )
    for i in range(shading_histogram_data_array.shape[0]):
        hist_table.add_data(*shading_histogram_data_array[i, :])

    wandb.log({"val/shading_hist_table": hist_table})

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
    wandb.log({"optimized_sh_coeff": optimized_coeff_table})

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

    # Set this run's confg
    run_config = {
        "batch_size": 1024,
        "record_freq": record_freq,
    }

    # Dataset Settings
    dataset_subset_fraction = config.getint("dataset", "subset_fraction", fallback=1)
    dataset_downsample_ratio = config.getint("dataset", "downsample_ratio", fallback=2)
    run_config.update(
        {
            "subset_fraction": dataset_subset_fraction,
            "downsample_ratio": dataset_downsample_ratio,
        }
    )

    # Training Settings
    num_epochs = config.getint("training", "epochs", fallback=1)
    shuffle_train = config.getboolean("training", "shuffle_train", fallback=False)
    run_config.update({"epochs": num_epochs, "shuffle_train": shuffle_train})

    learning_rate = config.get("training", "learning_rate", fallback=0.01)
    run_config.update(lr=float(learning_rate))

    # SH Settings
    initialization_vector_file = config.get(
        "global_spherical_harmonics", "sh_initialization", fallback=None
    )
    run_config.update(sh_init=initialization_vector_file)

    # Get pertubation if any
    pertubations_file = config.get(
        "global_spherical_harmonics", "sh_initialization_purtubation", fallback=None
    )
    run_config.update(
        pertubation_init=pertubations_file,
    )

    # these settings used only outside of the experiment function
    pertubations_min = config.getfloat(
        "global_spherical_harmonics",
        "sh_initialization_purtubation_strength_min",
        fallback=0,
    )
    pertubations_max = config.getfloat(
        "global_spherical_harmonics",
        "sh_initialization_purtubation_strength_max",
        fallback=1,
    )

    prtb_strs = (
        [0]
        if pertubations_file is None
        else np.linspace(pertubations_min, pertubations_max, num=8)
    )

    # get the non-negativity constraint switch
    non_negativity_constraint = config.getboolean(
        "global_spherical_harmonics", "non_negativity_constraint", fallback=True
    )
    run_config.update(non_negativity_constraint=non_negativity_constraint)

    # Load in the datasets
    train_dataset = get_dataset(config)
    valid_dataset = get_dataset(config, split="val")
    test_dataset = get_dataset(config, split="test")

    for prtb_str in prtb_strs:  # [0] is no ptrb file set
        run_name = run_name_prefix
        if pertubations_file is not None:
            run_name += f"_ptrbstr-{prtb_str}"

        run_config.update(pertubation_strength=prtb_str)

        for run_i in range(num_runs):
            run = wandb.init(project=project_name, config=run_config)
            assert run is not None

            # Keep the autogenerated name with a prefix
            wandb_run_name = run.name
            run.name = "_".join([run_name, wandb_run_name])

            with run:
                experiment_run()
