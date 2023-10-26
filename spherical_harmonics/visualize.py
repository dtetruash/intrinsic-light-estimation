import json

import copy
import matplotlib.style as mplstyle
import numpy as np
from icecream import ic
from ile_utils.config import Config
from matplotlib import cm  # noqa: F401
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.special import sph_harm
from data_loaders import olat_render as ro

from spherical_harmonics.spherical_harmonics import get_SH_alpha, render_second_order_SH

mplstyle.use("fast")

from rich.traceback import install

install()


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


def meshgrid_to_matrix(*grid):
    """Take a meshgrid to an (N,M) matrix.
        *grid: mesh grid coordianate arrays (like x,y,z)

    Returns:
        Single matrix N,M where M=grid(i).size and N=len(grid)
    """
    return np.stack([g.ravel() for g in grid])


def matrix_to_meshgrid(matrix, res=100):
    return tuple(row.reshape(res, res) for row in np.vsplit(matrix, 3))


def draw_3D_axis(ax, rot_matrix=np.eye(3)):
    # Draw a set of x, y, z axes for reference.
    # Axes are stores as columns of the rotation matrix
    for vector, color in zip(rot_matrix.T, ["r", "g", "b"]):
        ax.quiver(
            *[0, 0, 0],
            *vector,
            length=0.75,
            color=color,
            pivot="tail",
            arrow_length_ratio=0.05,
            zorder=2,
        )


def make_equal_axis_aspect(ax):
    ax.set_box_aspect((1, 1, 1), zoom=1.65)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_poles(ax, rot_matrix=np.eye(3)):
    north_pole = rot_matrix @ np.array([0, 0, 1])
    south_pole = rot_matrix @ np.array([0, 0, -1])

    nx, ny, nz = north_pole
    sx, sy, sz = south_pole

    ax.scatter([nx], [ny], [nz], s=4, c="white", edgecolors=["black"], zorder=3)
    ax.scatter([sx], [sy], [sz], s=4, c="black", edgecolors=["white"], zorder=3)


def visualie_SH_on_3D_sphere(
    sh_coeffs,
    camera_orientation=np.eye(3),
    draw_gizmos=True,
    resolution=100,
    bg_color="white",
    show_extremes=False,
):
    """Visualize a unit sphere shaded by SH lighting.

    Args:
        sh_coeffs (ndarray): second order SH coefficients
        rot_matrix (ndarray): 3x3 camera-to-world rotation matrix
        draw_gizmos (bool): draw world-axes and pole gizmos
        resolution (int): resolution of the sphere's surface along phi and theta
        bg_color (str): figure background color
        show_extremes (bool): highligh values outside [0,1] with red and blue resp.
    """

    cart_normals = meshgrid_to_matrix(*get_sphere_surface_cartesian(resolution))
    ic(cart_normals.shape)

    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    fcolors = render_second_order_SH(sh_coeffs, cart_normals.T, torch_mode=False)
    ic(fcolors.shape, fcolors.min(), fcolors.max())
    fcolors = fcolors.reshape(resolution, resolution)

    # Set the aspect ratio to 1 so our sphere looks spherical
    ax = plt.axes(projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    make_equal_axis_aspect(ax)

    # Draw the axis guides and poles
    if draw_gizmos:
        draw_3D_axis(ax)
        plot_poles(ax)

    # set the background color
    ax.set_facecolor(bg_color)

    ax.set_axis_off()
    plt.tight_layout()

    # Set the plot camera to correcpond to the given camera orientation
    # Decompose the rotation matrix into elevation and azimuth from the camera direction axis
    camera_direction = -np.dot(camera_orientation, np.array([0, 0, -1]))
    ic(camera_direction)

    # Calculate azimuth and elevation angles
    cx, cy, cz = camera_direction
    azim = np.degrees(np.arctan2(cy, cx))
    elev = np.degrees(np.arcsin(cz))

    ic(elev, azim)
    ax.view_init(roll=0, elev=elev, azim=azim)

    # Plot the shpere's surface

    # Get the shading of each surface face
    cmap = copy.copy(plt.get_cmap("gray"))
    if show_extremes:
        cmap.set_under("red")
        cmap.set_over("blue")
        face_colors = cmap(fcolors)
    else:
        norm = Normalize(vmin=fcolors.min(), vmax=fcolors.max())
        face_colors = cmap(norm(fcolors))

    # Plot the sphrer with the right colors
    ax.plot_surface(
        *matrix_to_meshgrid(cart_normals, res=resolution),
        facecolors=face_colors,
        linewidth=0,
        antialiased=False,
        rstride=1,
        cstride=1,
        zorder=1,
        shade=False,
    )

    return plt.gcf()


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


def plot2npimage(fig):
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer

    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(height, width, 3)
    return img


if __name__ == "__main__":
    # Read the c2w from file.
    config = Config.get_config()
    file_path = config.get("paths", "transforms_file")
    ic(file_path)
    with open(file_path, "r") as tf:
        frame_transforms = json.loads(tf.read())

    # I am supplying the C2W matrix here. Rotates objects from the camera to world.
    R_img = ro.to_rotation(ro.get_c2w(74, frame_transforms))
    fig = visualie_SH_on_3D_sphere(
        np.eye(9)[5], camera_orientation=R_img, bg_color="black", resolution=400
    )

    plt.savefig("sphere.png", bbox_inches="tight", pad_inches=0)
    plt.show()
