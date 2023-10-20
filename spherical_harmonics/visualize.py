from scipy.special import sph_harm
import numpy as np
from matplotlib import pyplot as plt
from icecream import ic
from spherical_harmonics import render_second_order_SH, get_SH_alpha
from matplotlib import cm, colors  # noqa: F401
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def generate_harmonic_sphere_image(sh_coeffs, camera_transform):
    """Create a visualization of the shading represented by spherical harmonics
    from a given camera-angle.

    Args:
        sh_coeffs (torch.Tensor or ndarray): spherical harmonics coefficients of second order
        camera_transform (ndarray): extrinsic camera transform.

    Raises:
        NotImplementedError: [TODO:throw]
    """
    raise NotImplementedError()


def generate_harmonic_latlong_image(sh_coeffs, image_dim=100):
    """Create a plot of the shading represented by spherical harmonics
    over the entire surface of the sphere as a latitude/longitude plot.

    Args:
        sh_coeffs (torch.Tensor or ndarray): spherical harmonics coefficients of second order

    Raises:
        NotImplementedError: [TODO:throw]
    """
    x, y, z = get_sphere_surface_cartesian(image_dim)
    cart_normals = np.stack([x, y, z]).transpose((1, 2, 0))
    ic(cart_normals.shape)
    cart_normals = cart_normals.reshape(-1, 3)
    ic(cart_normals.shape)

    # rendred each point in cart_normals
    shading_rendering = render_second_order_SH(sh_coeffs, cart_normals, torch_mode=False)
    ic(shading_rendering.shape)
    return shading_rendering.reshape(image_dim, image_dim, -1)


def generate_harmonic_latlong_image_scipy(m, n, image_dim=100):
    # Validation:
    phi = np.linspace(0, np.pi, image_dim)
    theta = np.linspace(0, 2 * np.pi, image_dim)
    phi, theta = np.meshgrid(phi, theta)

    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    fcolors = sph_harm(m, n, theta, phi)

    if m >= 0:
        return fcolors.real
    else:
        return fcolors.imag


def get_sphere_surface_polar(resolution=100):
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)
    return phi, theta


def get_sphere_surface_cartesian(resolution=100):
    phi, theta = get_sphere_surface_polar(resolution)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z


def visualie_SH_on_3D_sphere(sh_coeffs):
    x, y, z = get_sphere_surface_cartesian()
    cart_normals = np.stack([x, y, z]).transpose((1, 2, 0))
    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    fcolors = render_second_order_SH(sh_coeffs, cart_normals, torch_mode=False)
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin) / (fmax - fmin)

    # Set the aspect ratio to 1 so our sphere looks spherical
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
    # Turn off the axis planes
    ax.set_axis_off()
    plt.show()


def visualize_SH_validation_with_scipy():
    sh_coeffs = np.eye(9)
    fig, axes = plt.subplots(9, 3)

    ns = [0, 1, 1, 1, 2, 2, 2, 2, 2]
    ms = [0, -1, 0, 1, -2, -1, 0, 1, 2]
    for i in range(9):
        shading = generate_harmonic_latlong_image(sh_coeffs[i]) * (-1.0) ** ms[i]
        ax = axes[i, 0]
        ax.imshow(shading)
        ax.tick_params(left=False, bottom=False)
        ax.text(
            -0.01,
            0.5,
            f"n={ns[i]};m={ms[i]}",
            va="center",
            ha="right",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.text(
            -0.01,
            0,
            f"Min: {shading.min():.3f}\nMax: {shading.max():.3f}",
            va="center",
            ha="right",
            fontsize=8,
            transform=ax.transAxes,
        )
        if i == 0:
            ax.text(
                0.5,
                1.5,
                "Ours * (-1)^m",
                va="center",
                ha="center",
                fontsize=10,
                transform=ax.transAxes,
            )
        ax.axis("off")

    for i in range(9):
        sh = generate_harmonic_latlong_image_scipy(ms[i], ns[i])

        ax = axes[i, 1]
        ax.imshow(sh)
        ax.tick_params(left=False, bottom=False)
        ax.text(
            -0.01,
            0,
            f"Min: {sh.min():.3f}\nMax: {sh.max():.3f}",
            va="center",
            ha="right",
            fontsize=8,
            transform=ax.transAxes,
        )
        if i == 0:
            ax.text(
                0.5,
                1.5,
                "Scipy",
                va="center",
                ha="center",
                fontsize=10,
                transform=ax.transAxes,
            )
        ax.axis("off")

        alpha_n = get_SH_alpha(ns[i])
        ic(alpha_n, ns[i])
        sh *= alpha_n
        ax = axes[i, 2]
        ax.imshow(sh)
        ax.tick_params(left=False, bottom=False)
        ax.text(
            -0.01,
            0,
            f"Min: {sh.min():.3f}\nMax: {sh.max():.3f}",
            va="center",
            ha="right",
            fontsize=8,
            transform=ax.transAxes,
        )
        if i == 0:
            ax.text(
                0.5,
                1.5,
                "Scipy * alpha_nm",
                va="center",
                ha="center",
                fontsize=10,
                transform=ax.transAxes,
            )
        ax.axis("off")

    # fig.align_labels()
    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # axes.set_yticks(np.arange(len(ns)), labels=[f"n={n};m={m}" for n, m in zip(ns, ms)])
    visualie_SH_on_3D_sphere(np.eye(9)[2])
