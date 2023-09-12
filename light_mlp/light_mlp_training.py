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
import torchvision
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


# Get the dataloader
def get_dataloader(batch_size, split='train'):
    is_train = split == 'train'
    "Get a dataloader for training or testing"
    full_dataset = rd.RasterDataset(split)  # uses a config to decide what to load.
    loader = torch.utils.data.DataLoader(dataset=full_dataset,
                                         batch_size=batch_size,
                                         shuffle=is_train,
                                         pin_memory=True, num_workers=0)
    return loader


def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for step, (feats, target_vector) in tqdm(enumerate(valid_dl),
                                                 desc="Validating model",
                                                 total=len(valid_dl)):
            feats, target_vector = feats.to(device), target_vector.to(device)

            # Forward pass
            inputs = torch.flatten(feats, start_dim=1)
            outputs = model(inputs)  # should be (batch_len, 3)
            val_loss += loss_func(outputs, target_vector)*target_vector.size(0)

            break

    return val_loss / len(valid_dl.dataset)


def generate_validation_image(model, valid_dataset):
    """Generate an image comparing a ground truth image with one generated using the model.
    model: MLP which outputs light direction vectors"""
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


def train_epoch(epoch, train_dl, model, loss_func):
    cumu_loss = 0.0
    for step, (feats, target_vector) in tqdm(enumerate(train_dl),
                                             total=len(train_dl),
                                             desc="Step",
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
                   "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                   }

        if step + 1 < n_steps_per_epoch:
            # 🐝 Log train metrics to wandb
            wandb.log(metrics)

        break

    return cumu_loss / len(train_dl)


# Loss functions!

cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
def cosine_similarity_loss_func(x, y, lamb=2.0):
    cosine_similarity = cosine(x, y)
    similarity_target = torch.tensor([1.0]).broadcast_to(cosine_similarity.size()).to(device)
    similarity_term = F.mse_loss(cosine_similarity, similarity_target)

    return similarity_term

def unitarity_loss(x):
    x_norms = torch.linalg.norm(x, dim=-1)
    unitarity_term = F.mse_loss(x_norms,
                                torch.tensor([1.0]).broadcast_to(x_norms.size()).to(device))

    return unitarity_term


def loss_func(x, y, lamb=2.0):
    return cosine_similarity_loss_func(x, y) + lamb * unitarity_loss(x)


# +
# Get the data
batch_size = 1024
train_dl = get_dataloader(batch_size=batch_size)
print("Loaded train dataset.")

valid_dl = get_dataloader(batch_size=2*batch_size, split='val')
print("Loaded validation dataset.")

# +
# 🐝 initialise a wandb run
# NOTE: The model checkpoint path should be to scratch on the cluster

raster_config = rr.parse_config()

current_run = wandb.init(
    project="light-mlp-supervised-cosine",
    config={
        "epochs": 5,
        "batch_size": batch_size,
        "lr": 1e-3,
        "dropout": 0.0,  # random.uniform(0.01, 0.80),
        "num_feats": 3,
        "hidden_channels": [256]*4 + [128],
        "model_checkpoint_path": 'model_checkpoints',
        'model_trained_path': 'model_trained',
        'scene': raster_config['paths']['scene']
    })

# Copy your config
config = wandb.config

# make output dirs
if not os.path.exists(config.model_checkpoint_path):
    os.makedirs(config.model_checkpoint_path)

if not os.path.exists(config.model_trained_path):
    os.makedirs(config.model_trained_path)

n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
# -

# MLP model
model = LightMLP(config.num_feats, config.hidden_channels, dropout=config.dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# +
# Training
for epoch in tqdm(range(config.epochs),
                  desc="Epoch",
                  total=config.epochs,
                  position=0, leave=True, colour='green'):
    model.train()

    # Train an epoch
    print(f"Training epoch {epoch}")
    avg_train_loss = train_epoch(epoch, train_dl, model, cosine_similarity_loss_func)
    wandb.log({"train/avg_loss": avg_train_loss})

    # Validation
    print(f"Validating model after epoch {epoch}")
    val_loss = validate_model(model, valid_dl, cosine_similarity_loss_func)
    print("Done validating.")

    # Render a validation image
    print("Creating validation image.")
    val_image_array, image_caption = generate_validation_image(model, valid_dl.dataset)
    # val_image = PILImage.fromarray(val_image_array, mode="RGB")
    # print(f"image type {type(val_image)}")
    # val_image.save(f"validation_image_{epoch:03}.png")

    val_image = wandb.Image(val_image_array, caption=image_caption)
    # 🐝 Log train and validation metrics to wandb
    val_metrics = {"val/val_loss": val_loss,
                   "val/images": val_image}
    wandb.log(val_metrics)

    print(f"Train Loss: {avg_train_loss:.3f} Valid Loss: {val_loss:3f}")

    to_save = {
        'model_state_dict': model.state_dict(),
        'optimized_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'lr': config.lr,
        "dropout": config.dropout,
        "num_feats": config.num_feats,
        "hidden_channels": config.hidden_channels,

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


# 🐝 Close your wandb run
wandb.finish()
# -

model
