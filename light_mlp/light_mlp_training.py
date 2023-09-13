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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import math
from tqdm import tqdm
import os
from model import LightMLP

import raster_relight as rr
import raster_dataloader as rd
# -

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
def get_dataloader(batch_size, split='train', subset_fraction=1):
    "Get a dataloader for training or testing"

    is_train = split == 'train'

    full_dataset = rd.RasterDataset(split)

    if subset_fraction > 1:
        sub_dataset = torch.utils.data.Subset(full_dataset, indices=range(0, len(full_dataset), subset_fraction))
    else:
        sub_dataset = full_dataset

    loader = torch.utils.data.DataLoader(dataset=sub_dataset,
                                         batch_size=batch_size,
                                         shuffle=is_train,
                                         pin_memory=True, num_workers=0)
    return loader

def get_optimizer(optimizer, model, learning_rate):
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    return optimizer

def get_model(num_feats, num_layers, layer_size, dropout):
    hidden_channels = [layer_size] * num_layers
    hidden_channels[-1] = hidden_channels[-1] // 2
    return LightMLP(num_feats, hidden_channels, dropout=dropout)

def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    val_loss = 0.
    with torch.inference_mode():
        for step, (feats, target_vector) in tqdm(enumerate(valid_dl),
                                                 desc="Validating model",
                                                 total=len(valid_dl)):
            feats, target_vector = feats.to(device), target_vector.to(device)

            # Forward pass
            inputs = torch.flatten(feats, start_dim=1)
            outputs = model(inputs)  # should be (batch_len, 3)
            val_loss += loss_func(outputs, target_vector)*target_vector.size(0)

    return val_loss / len(valid_dl.dataset)


def generate_validation_image(model, valid_dataset):
    """Generate an image comparing a ground truth image with one generated using the model.
    model: MLP which outputs light direction vectors"""
    model.eval()
    with torch.inference_mode():
        # Randomly choose which image from the validation set to reconstruct
        image_number = np.random.randint(valid_dataset.num_frames)
        # randomly choose a light in the scene
        light_names = list(valid_dataset._lights_info.keys())
        light_name = light_names[np.random.randint(valid_dataset._num_lights)]
        print(f"Generating OLAT validation image {image_number} with light {light_name}...")

        # load attributes of this validation image
        W, H, gt_raster_pixels, world_normals, albedo, _, gt_light_vectors, occupancy_mask = valid_dataset.attributes[image_number][light_name]

        gt_raster_pixels = gt_raster_pixels.astype(np.float32)
        world_normals = world_normals.astype(np.float32)
        albedo = albedo.astype(np.float32)
        gt_light_vectors = gt_light_vectors.astype(np.float32)

        # prepare inputs for inference
        feats = np.stack([world_normals, albedo, gt_raster_pixels], axis=1)
        inputs = torch.flatten(torch.as_tensor(feats).float(), start_dim=1)

        # Do inference to get light vectors
        light_vectors = model(inputs)
        light_vectors = light_vectors.numpy().astype(np.float32)

        # Construct a normals image
        img_size = (W, H, 3)

        # Light direciton images
        val_light_dir_image = np.ones(img_size, dtype=np.float32)
        val_light_dirs = 0.5*light_vectors + 0.5
        val_light_dir_image[occupancy_mask] = val_light_dirs

        gt_light_dir_image = np.ones(img_size, dtype=np.float32)
        gt_light_colors = 0.5*gt_light_vectors + 0.5
        gt_light_dir_image[occupancy_mask] = gt_light_colors

        # Raster pixel images
        val_raster_image = np.ones(img_size, dtype=np.float32)
        val_raster_pixels, val_shading = rr.raster_from_directions(light_vectors, albedo, world_normals, return_shading=True)
        val_raster_image[occupancy_mask] = val_raster_pixels

        gt_raster_image = np.ones(img_size, dtype=np.float32)
        gt_raster_image[occupancy_mask] = gt_raster_pixels

        # Shading images
        val_shading_image = np.ones(img_size, dtype=np.float32)
        val_shading_image[occupancy_mask] = val_shading[..., np.newaxis]

        gt_shading_image = np.ones(img_size, dtype=np.float32)
        gt_shading = rr.compute_clipped_dot_prod(world_normals, gt_light_vectors)
        gt_shading_image[occupancy_mask] = gt_shading[..., np.newaxis]

        # Stick them together
        validation_images = np.concatenate([val_raster_image, val_shading_image, val_light_dir_image], axis=1)

        gt_images = np.concatenate([gt_raster_image, gt_shading_image, gt_light_dir_image], axis=1)

        image_array = np.concatenate([validation_images, gt_images], axis=0)

        image_caption = "Top row : Inference. Bottom: GT.\nLeft to right: Render, Shading, Light directions."

        return image_array, image_caption


def train_epoch(epoch, train_dl, model, loss_func, optimizer, n_batches_per_epoch):
    cumu_loss = 0.0
    for batch, (feats, target_vector) in tqdm(enumerate(train_dl),
                                              total=len(train_dl),
                                              desc="Batch",
                                              position=1, leave=False, colour='red'):
        # Move to device
        feats, target_vector = feats.to(device), target_vector.to(device)

        # Forward pass
        inputs = torch.flatten(feats, start_dim=1)  # (batch_len, )
        outputs = model(inputs)  # should be (batch_len, 3)
        train_loss = loss_func(outputs, target_vector)
        cumu_loss += train_loss.item()

        # Optimization step
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Collect metrics
        metrics = {"train/train_loss": train_loss,
                   "train/epoch": (batch + 1 + (n_batches_per_epoch * epoch)) / n_batches_per_epoch,
                   }

        # step = get_current_step(epoch, batch, train_dl.batch_size, n_batches_per_epoch)

        if batch + 1 < n_batches_per_epoch:
            # 🐝 Log train metrics to wandb
            wandb.log(metrics)

    return cumu_loss / len(train_dl)


# Loss functions!

cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
def cosine_similarity_loss(x, y, lamb=2.0):
    cosine_similarity = cosine(x, y)
    similarity_target = torch.tensor([1.0]).broadcast_to(cosine_similarity.size()).to(device)
    similarity_term = F.mse_loss(cosine_similarity, similarity_target)

    return similarity_term

def unitarity_loss(x):
    x_norms = torch.linalg.norm(x, dim=-1)
    unitarity_term = F.mse_loss(x_norms,
                                torch.tensor([1.0]).broadcast_to(x_norms.size()).to(device))

    return unitarity_term

def get_loss_func(loss_func, unitarity_lambda):
    if loss_func == 'cosine':
        return lambda x, y: cosine_similarity_loss(x, y) + unitarity_lambda * unitarity_loss(x)
    elif loss_func == 'mse':
        return lambda x, y: F.mse_loss(x, y) + unitarity_lambda * unitarity_loss(x)


# Main training function
def carry_out_training():

    with wandb.init() as current_run:

        # Copy your config
        config = current_run.config

        print(f"config object was: {list(config.items())}")

        train_dl = get_dataloader(batch_size=config.batch_size, subset_fraction=config.data_subset_fraction)
        print(f"Loaded train dataset with {len(train_dl)} batches and {len(train_dl.dataset)} samples.")

        valid_dl = get_dataloader(batch_size=2*config.batch_size, split='val')
        print(f"Loaded valid dataset with {len(valid_dl)} batches and {len(valid_dl.dataset)} samples.")

        # make output dirs
        if not os.path.exists(config.model_checkpoint_path):
            os.makedirs(config.model_checkpoint_path)

        if not os.path.exists(config.model_trained_path):
            os.makedirs(config.model_trained_path)

        # -
        n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

        # Construct model, optimizer, and loss
        # model = LightMLP(config.num_feats, config.hidden_channels, dropout=config.dropout)
        model = get_model(config.num_feats, config.num_layers, config.layer_size, config.dropout)
        optimizer = get_optimizer(config.optimizer, model, config.learning_rate)
        loss_func = get_loss_func(config.loss_func, config.unitarity_lambda)

        print("Model was:", model)

        # +
        # Training
        for epoch in tqdm(range(config.epochs + 1),
                          desc="Epoch",
                          total=config.epochs,
                          position=0, leave=True, colour='green'):
            model.train()

            # Train an epoch
            print(f"Training epoch {epoch}")
            avg_train_loss = train_epoch(epoch, train_dl, model, loss_func, optimizer, n_steps_per_epoch)
            wandb.log({"train/avg_loss": avg_train_loss})

            # Validation
            model.eval()
            print(f"Validating model after epoch {epoch}")
            val_loss = validate_model(model, valid_dl, loss_func)
            print("Done validating.")

            # Render a validation image
            print("Creating validation image.")
            val_image_array, image_caption = generate_validation_image(model, valid_dl.dataset)
            val_image = wandb.Image(val_image_array, caption=image_caption)

            # 🐝 Log train and validation metrics to wandb
            val_metrics = {"validation_loss": val_loss,
                           "val/images": val_image}
            wandb.log(val_metrics)

            print(f"Train Loss: {avg_train_loss:.3f} Valid Loss: {val_loss:3f}")

            # Save model artifacts
            to_save = {
                'model_state_dict': model.state_dict(),
                'optimized_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'learning_rate': config.learning_rate,
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
                wandb.save(trained_file, policy='now')
            else:
                # save checkpoint
                check_point_file = f"{config.model_checkpoint_path}/{current_run.project}_{current_run.name}_ckpt.pth"
                torch.save(to_save, check_point_file)
                wandb.save(check_point_file, policy='live')


if __name__ == "__main__":

    project_name = "light-mlp-supervised-cosine"

    raster_config = rr.parse_config()

    sweep_configuration = {
        'method': 'random',
        'metric': {
            'name': 'validation_loss',
            'goal': 'minimize',
        },
        'parameters': {
            'batch_size': {
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': 32,
                'max': 256,
            },
            'learning_rate': {
                'min': 0.0001,
                'max': 0.1,
            },
            'loss_func': {'values': ['cosine', 'mse']},
            'unitarity_lambda': {

                'min': 0.0,
                'max': 2.0,
            },
            'optimizer': {'values': ['adam', 'sgd']},
        },
    }

    # model parameters
    sweep_configuration['parameters'].update({
        'num_feats': {'value': 3},
        'num_layers': {
            'values': [3, 6, 10],
        },
        'layer_size': {'values': [128, 256, 512]},
        'dropout': {'values': [0.15, 0.2, 0.25, 0.3, 0.4]},
    })

    # non-sweep parameters
    sweep_configuration['parameters'].update({
        'epochs': {'value': 3},
        'data_subset_fraction': {'value': 64},
        'model_checkpoint_path': {'value': 'model_checkpoints'},
        'model_trained_path': {'value': 'model_trained'},
        'scene': {'value': raster_config['paths']['scene']}
    })

    wandb.login()

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

    wandb.agent(sweep_id, function=carry_out_training, count=15)
