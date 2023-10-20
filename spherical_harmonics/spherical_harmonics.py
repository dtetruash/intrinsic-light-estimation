"""Methods for rendering pixels via used ligting models"""

import torch
import math
import numpy as np

Y_0_coeff = 1.0 / math.sqrt(4.0 * math.pi)
Y_1_coeff = math.sqrt((3.0 / (4.0 * math.pi)))
Y_2_coeff = 3.0 * math.sqrt(5.0 / (12.0 * math.pi))
Y_20_coeff = 0.5 * math.sqrt(5.0 / (4.0 * math.pi))


def get_SH_alpha(n):
    return math.sqrt((4.0 * math.pi) / (2.0 * n + 1.0))


sh_alphas = torch.concat(
    [torch.tensor([get_SH_alpha(n)] * m) for n, m in enumerate([1, 3, 5])]
)


def get_SH_basis(normals, torch_mode=True):
    """This basis assumes we take Y^o to have -ve sh_coefficients while Y^e to have non -ve sh_coefficients"""
    # Force to use a tensor (e.g., when validating)
    if torch_mode and not isinstance(normals, torch.Tensor):
        normals = torch.as_tensor(normals)

    lib = torch if torch_mode else np

    # these are now (B, 1)
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]

    if torch_mode:
        y00 = get_SH_alpha(0) * torch.tensor([Y_0_coeff]).broadcast_to(x.shape)
    else:
        y00 = get_SH_alpha(0) * np.broadcast_to(np.array([Y_0_coeff]), x.shape)

    yo11 = get_SH_alpha(1) * Y_1_coeff * y
    y_10 = get_SH_alpha(1) * Y_1_coeff * z
    ye11 = get_SH_alpha(1) * Y_1_coeff * x

    yo22 = get_SH_alpha(2) * Y_2_coeff * x * y
    yo21 = get_SH_alpha(2) * Y_2_coeff * y * z
    y_20 = get_SH_alpha(2) * Y_20_coeff * (3.0 * lib.square(z) - 1.0)
    ye21 = get_SH_alpha(2) * Y_2_coeff * x * z
    ye22 = get_SH_alpha(2) * 0.5 * Y_2_coeff * (lib.square(x) - lib.square(y))

    return lib.stack([y00, yo11, y_10, ye11, yo22, yo21, y_20, ye21, ye22]).T


def render_second_order_SH(sh_coefficients, normals, torch_mode=True):
    """Render pixel images given spherical harmonics sh_coefficients (per pixel) and normals (per pixel)
    sh_coefficients : ndarray (B, 9)
    normals         : ndarray (B, 3)
    torch_mode (bool): flag switch to use torch instead of numpy
    """

    # Should have the shape (B, 9)
    sh_basis = get_SH_basis(normals, torch_mode)

    # print(f"basis shape was {sh_basis.shape}")

    # FIXME Bound this to [0,1] (by force) or normalize outside per image
    prod = (sh_alphas if torch_mode else sh_alphas.numpy()) * sh_basis * sh_coefficients
    if torch_mode:
        return torch.sum(prod, dim=-1).float()  # should be in [0,1]
    else:
        return np.sum(prod, axis=-1).astype(np.float32)


def evaluate_harmonic(sh_coefficients, theta, phi):
    raise NotImplementedError()


if __name__ == "__main__":
    B = 15
    coeff = torch.rand((B, 9))
    normals = torch.nn.functional.normalize(2.0 * torch.rand((B, 3)) - 1.0, dim=-1)

    print(
        f"Rendered shading was {(shading:=render_second_order_SH(coeff, normals))} with"
        f" shape {shading.shape}"
    )
