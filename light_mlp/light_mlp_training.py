# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import argparse
import math
import os

import generate_images as genimg
import torch
import torch.nn.functional as F
import wandb
from model import LightMLP, LightSH
from rich.traceback import install as install_rich
from tqdm import tqdm

from ile.data_loaders.datasets import RasterDataset
from ile.losses.loss_functions import (
    cosine_similarity_loss,
    photometric_loss,
    raster_photometric_loss,
    sh_photometric_loss,
    unitarity_loss,
)
from ile_utils.config import Config

install_rich()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_current_step(epoch, batch, batch_size, batches_per_epoch):
    return batch_size * (batches_per_epoch * epoch + batch + 1)


# Get the dataloader
def get_dataloader(batch_size, split="train", subset_fraction=1):
    "Get a dataloader for training or testing"

    is_train = split == "train"

    full_dataset = RasterDataset(split)

    if subset_fraction > 1:
        sub_dataset = torch.utils.data.Subset(
            full_dataset, indices=range(0, len(full_dataset), subset_fraction)
        )
    else:
        sub_dataset = full_dataset

    loader = torch.utils.data.DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=0,
    )
    return loader


def get_optimizer(optimizer, model, learning_rate):
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    return optimizer


def get_model(model_type, num_feats, num_layers, layer_size, dropout):
    # TODO: Add flag to see which model to create, mlp or sh
    if model_type == "light-mlp":
        model_class = LightMLP
    elif model_type == "light-sh":
        model_class = LightSH
    else:
        raise ValueError(f"Unsupported model type {model_type}")

    hidden_channels = [layer_size] * num_layers
    hidden_channels[-1] = hidden_channels[-1] // 2
    return model_class(num_feats, hidden_channels, dropout=dropout)


def get_loss_func(loss_func, unitarity_lambda):
    print(f"Loss func name was: {loss_func}.")
    if loss_func == "cosine":
        return lambda x, y: cosine_similarity_loss(
            x, y
        ) + unitarity_lambda * unitarity_loss(x)
    elif loss_func == "mse":
        return lambda x, y: F.mse_loss(x, y) + unitarity_lambda * unitarity_loss(x)
    elif loss_func == "photometric_raster":
        return lambda x, y: raster_photometric_loss(x, y)
    elif loss_func == "photometric_sh":
        return lambda x, y: sh_photometric_loss(x, y)
    elif loss_func == "photometric":
        return lambda x, y: photometric_loss(x, y)
    else:
        raise ValueError(f"Not suported loss function type '{loss_func}'.")


def do_forward_pass(model, feats, target, loss_func, loss_func_name):
    inputs = torch.flatten(feats, start_dim=1)  # (batch_len, )
    outputs = model(inputs)  # should be (batch_len, 3)

    if loss_func_name.startswith("photometric"):
        # NOTE: if the dataloader changes its output (eg, adding posiiton) this will
        # need to change.
        train_loss = loss_func(
            outputs, feats
        )  # for photometric this also needs some of the inputs
    else:
        train_loss = loss_func(outputs, target)

    return train_loss


def validate_model(model, valid_dl, loss_func, loss_func_name):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    val_loss_acc = 0.0
    with torch.inference_mode():
        for step, (feats, target_vector) in tqdm(
            enumerate(valid_dl), desc="Validating model", total=len(valid_dl)
        ):
            feats, target_vector = feats.to(device), target_vector.to(device)
            samples_in_batch = target_vector.size(0)

            # Forward pass
            val_loss = do_forward_pass(
                model, feats, target_vector, loss_func, loss_func_name
            ).item()

            val_loss_acc += val_loss * samples_in_batch

    return val_loss_acc / len(valid_dl.dataset)


def train_epoch(
    epoch, train_dl, model, optimizer, n_batches_per_epoch, loss_func, loss_func_name
):
    cumu_loss = 0.0
    for batch, (feats, target_vector) in tqdm(
        enumerate(train_dl),
        total=len(train_dl),
        desc="Batch",
        position=1,
        leave=True,
        colour="red",
    ):
        # Move to device
        feats, target_vector = feats.to(device), target_vector.to(device)

        # forward pass
        batch_train_loss = do_forward_pass(
            model, feats, target_vector, loss_func, loss_func_name
        )
        cumu_loss += batch_train_loss.item()

        # Optimization step
        optimizer.zero_grad()
        batch_train_loss.backward()
        optimizer.step()

        # Collect metrics
        epoch_step = batch + 1 + (n_batches_per_epoch * epoch)
        metrics = {
            "train/train_loss": batch_train_loss,
            "train/epoch": epoch_step / n_batches_per_epoch,
        }

        # step = get_current_step(
        # epoch, batch, train_dl.batch_size, n_batches_per_epoch
        # )

        if batch + 1 < n_batches_per_epoch:
            # 🐝 Log train metrics to wandb
            wandb.log(metrics)

    avg_batch_loss = cumu_loss / len(train_dl)
    return avg_batch_loss


# Main training function
def carry_out_training():
    with wandb.init() as current_run:
        # wandb.define_metric("train/step")
        # wandb.define_metric("train/*", step_metric="train/step")
        #
        # wandb.define_metric("epoch/step")
        # wandb.define_metric("val/*", step_metric="val/step")

        # Copy your config
        config = current_run.config

        print(f"config object was: {list(config.items())}")

        train_dl = get_dataloader(
            batch_size=config.batch_size, subset_fraction=config.data_subset_fraction
        )
        print(
            f"Loaded train dataset with {len(train_dl)} batches \
            and {len(train_dl.dataset)} samples."
        )

        valid_dl = get_dataloader(batch_size=2 * config.batch_size, split="val")
        print(
            f"Loaded valid dataset with {len(valid_dl)} batches \
            and {len(valid_dl.dataset)} samples."
        )

        # make output dirs
        if not os.path.exists(config.model_checkpoint_path):
            os.makedirs(config.model_checkpoint_path)

        if not os.path.exists(config.model_trained_path):
            os.makedirs(config.model_trained_path)

        # -
        n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

        # Construct model, optimizer, and loss
        # model = LightMLP(
        #     config.num_feats, config.hidden_channels, dropout=config.dropout
        # )
        model = get_model(
            config.model_class,
            config.num_feats,
            config.num_layers,
            config.layer_size,
            config.dropout,
        ).to(device)
        optimizer = get_optimizer(config.optimizer, model, config.learning_rate)
        loss_func = get_loss_func(config.loss_func, config.unitarity_lambda)

        print("Model was:", model)

        # +
        # Training

        for epoch in tqdm(
            range(config.epochs + 1),
            desc="Epoch",
            total=config.epochs,
            position=0,
            leave=True,
            colour="green",
        ):
            model.train()

            # Train an epoch
            print(f"Training epoch {epoch}")
            avg_train_loss = train_epoch(
                epoch,
                train_dl,
                model,
                optimizer,
                n_steps_per_epoch,
                loss_func,
                config.loss_func,
            )
            wandb.log(
                {
                    "train/avg_loss": avg_train_loss,
                }
            )

            # Validation
            model.eval()
            print(f"Validating model after epoch {epoch}")
            val_loss = validate_model(model, valid_dl, loss_func, config.loss_func)
            print("Done validating.")

            # Render a validation image
            print("Creating validation image.")
            val_image_array, image_caption = genimg.generate_validation_image(
                model, config.model_class, valid_dl.dataset
            )
            val_image = wandb.Image(val_image_array, caption=image_caption)

            # TODO: create error heatmaps
            # error_heatmap_image, heatmap_caption = genimg.generate_heatmap_image_grid(
            #     model, valid_dl.dataset
            # )

            # 🐝 Log train and validation metrics to wandb
            val_metrics = {"validation_loss": val_loss, "val/images": val_image}
            wandb.log(val_metrics)

            print(f"Train Loss: {avg_train_loss:.3f} Valid Loss: {val_loss:3f}")

            # Save model artifacts
            to_save = {
                "model_state_dict": model.state_dict(),
                "optimized_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "learning_rate": config.learning_rate,
                "dropout": config.dropout,
                "num_feats": config.num_feats,
                "num_layers": config.num_layers,
                "layer_size": config.layer_size,
            }

            # save the model and upload it to wandb
            if epoch + 1 == config.epochs:
                # save trained model
                trained_file = f"{config.model_trained_path}/{current_run.project}_{current_run.name}.pth"
                torch.save(to_save, trained_file)
                wandb.save(trained_file, policy="now")
            else:
                # save checkpoint
                check_point_file = f"{config.model_checkpoint_path}/{current_run.project}_{current_run.name}_ckpt.pth"
                torch.save(to_save, check_point_file)
                wandb.save(check_point_file, policy="live")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    project_name = "light-sh-selfsupervised-sweep"

    raster_config = Config.get_config()

    sweep_configuration = {
        "method": "random",
        "metric": {
            "name": "validation_loss",
            "goal": "minimize",
        },
        "parameters": {
            "batch_size": {
                "distribution": "q_log_uniform_values",
                "q": 8,
                "min": 32,
                "max": 256,
            },
            "learning_rate": {
                "min": 0.0001,
                "max": 0.1,
            },
            "loss_func": {"value": "photometric_sh"},
            "unitarity_lambda": {"value": 0.0},
            "optimizer": {"values": ["adam", "sgd"]},
        },
    }

    # model parameters
    sweep_configuration["parameters"].update(
        {
            "model_class": {"value": "light-sh"},
            "num_feats": {"value": 3},
            "num_layers": {
                "values": [3, 6, 10],
            },
            "layer_size": {"values": [64, 128, 256, 512]},
            "dropout": {"values": [0.15, 0.2, 0.25, 0.3, 0.4]},
        }
    )

    # non-sweep parameters
    sweep_configuration["parameters"].update(
        {
            "epochs": {"value": 3},
            "data_subset_fraction": {"value": 1},
            "model_checkpoint_path": {"value": "model_checkpoints"},
            "model_trained_path": {"value": "model_trained"},
            "scene": {"value": raster_config["paths"]["scene"]},
        }
    )

    if args.debug:
        pass

    wandb.login()

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

    wandb.agent(sweep_id, function=carry_out_training, count=10)
