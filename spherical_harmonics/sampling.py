from icecream import ic
import numpy as np
import math

from rich.traceback import install

install()


def get_uniform_samples(rng=None, K=1, dim=2):
    if rng is None:
        rng = np.random.default_rng()

    return rng.random((K, dim), dtype=np.float32)


def square_to_uniform_cylinder(samples):
    theta = samples[..., 0] * 2.0 * math.pi
    height = samples[..., 1]

    return np.stack([np.cos(theta), np.sin(theta), height], axis=1)


def cylinder_to_unit_spheroid(cyl_samples):
    r = np.sqrt(1.0 - np.square(cyl_samples[..., -1]))
    spheroid_samples = np.copy(cyl_samples)
    spheroid_samples[..., 0:2] *= r[np.newaxis].T
    return spheroid_samples


def square_to_uniform_sphere(samples):
    """Return uniformally sampled direction on a sphere in polar coordinates"""
    assert samples.shape[-1] == 2, (
        "Given samples must be 2 dementional, given were"
        f" {samples.shape[-1]} dimentional. Overall samples shape was {samples.shape}"
    )
    cyl = square_to_uniform_cylinder(samples)
    cyl[..., -1] *= 2.0
    cyl[..., -1] -= 1.0
    return cylinder_to_unit_spheroid(cyl)


def sample_uniform_sphere(rng=None, K=1000):
    samples = get_uniform_samples(rng, K)
    return square_to_uniform_sphere(samples)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))

    # Validation wireframe of sphere
    theta, phi = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi)
    X = np.sin(PHI) * np.cos(THETA)
    Y = np.sin(PHI) * np.sin(THETA)
    Z = np.cos(PHI)
    ax.plot_wireframe(X, Y, Z, colors="black", alpha=0.5, linestyle=":")

    # 1000 samples on sphere
    sphere_samples = sample_uniform_sphere()
    ic(sphere_samples.shape)
    ax.scatter(*sphere_samples.T)

    plt.show()
