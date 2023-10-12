# Loss functions!
import torch
import torch.nn as nn
import torch.nn.functional as F
import raster_relight as rr

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def decompose_feats(feats):
    # assuming that we have (B, F, 3) with normals, albedo, then normals
    return feats[:, 0], feats[:, 1], feats[:, 2]


cosine_similarity_module = nn.CosineSimilarity(dim=1, eps=1e-6)


def cosine_similarity_loss(x, y, lamb=2.0):
    cosine_similarity = cosine_similarity_module(x, y)
    similarity_target = (
        torch.tensor([1.0]).broadcast_to(cosine_similarity.size()).to(device)
    )
    similarity_term = F.mse_loss(cosine_similarity, similarity_target)

    return similarity_term


def unitarity_loss(x):
    x_norms = torch.linalg.norm(x, dim=-1)
    unitarity_term = F.mse_loss(
        x_norms, torch.tensor([1.0]).broadcast_to(x_norms.size()).to(device)
    )

    return unitarity_term


def photometric_loss(x, feats):
    """Take in a rendered pixel image from model parameters and generate the loss
    between it and a ranster renderer of that pixel.
    """

    # NOTE: This function will need to change
    # The comparison method, as well as the rendering method.
    # E.g., when computing the loss against diffuse combined render, we would not need
    # to recompute the target image pixel.

    # decompose feats to albedo, normals, and images
    normals, albedo, images = decompose_feats(feats)

    pred_light_vectors = x
    pred_images = rr.render_from_directions_torch(pred_light_vectors, albedo, normals)

    return F.mse_loss(images, pred_images)
