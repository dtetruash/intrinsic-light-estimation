import torch

import spherical_harmonics as sh
from sampling import sample_uniform_sphere

def SH_non_negativity(sh_coeff):
    thetas, phis = sample_uniform_sphere(K=6414)
    intensity_samples = sh.evaluate_harmonic(sh_coeff, thetas, phis)

    return torch.minimum(torch.tensor([0]), intensity_samples).square().mean()
