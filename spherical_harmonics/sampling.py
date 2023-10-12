import numpy as np
import math

def get_uniform_samples(rng, K=1, dim=2):
    return rng.random((K, dim), dtype=np.float32)

def sample_uniform_sphere(samples):
    """Return uniformally sampled direction on a sphere in polar coordinates
    """
    assert samples.shape[-1] == 2, f"Given samples must be 2 dementional, given were {samples.shape[-1]} dimentional. Overall samples shape was {samples.shape}"

    theta = samples[..., 1] * math.pi
    phi = samples[..., 0] * 2. * math.pi
    return theta, phi

def sample_uniform_cylinder(samples):
    theta = samples[..., 0] * 2.0 * math.pi
    height = samples[..., 1]
    return theta, height
