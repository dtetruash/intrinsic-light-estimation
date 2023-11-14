import copy
import json

import matplotlib.style as mplstyle
import numpy as np
from data_loaders import olat_render as ro
from icecream import ic
from ile_utils.config import Config
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rich.traceback import install
from scipy.special import sph_harm

from spherical_harmonics.sph_harm import evaluate_second_order_SH, render_pixel_from_sh

install()

mplstyle.use("fast")


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
    shading_rendering = evaluate_second_order_SH(
        sh_coeffs, cart_normals, torch_mode=False
    )
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
            linewidth=1,
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


def plot_SH_sphere_on_axis(
    ax,
    surface_points,
    surface_values,
    camera_orientation,
    draw_gizmos=True,
    bg_color="white",
    show_extremes=False,
):
    # Set the aspect ratio to 1 so our sphere looks spherical
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
    orientate_3d_axes(ax, camera_orientation)

    # Get the shading of each surface face
    cmap = copy.copy(plt.get_cmap("gray"))
    if show_extremes:
        cmap.set_under("orange")
        cmap.set_over("blue")
        face_colors = cmap(surface_values)
    else:
        norm = Normalize(vmin=surface_values.min(), vmax=surface_values.max())
        face_colors = cmap(norm(surface_values))

    # Plot the sphrer with the right colors
    ax.plot_surface(
        *surface_points,
        facecolors=face_colors,
        linewidth=0,
        antialiased=False,
        rstride=1,
        cstride=1,
        zorder=1,
        shade=False,
    )


def visualize_scene_frame_from_sh(frame_number, sh_coeff, dataset, torch_mode=True):
    # Make top row of infered images

    # load attributes of this validation image
    gt_attributes, occupancy_mask = dataset.get_frame_decomposition(frame_number)

    _, albedo, _, world_normals = gt_attributes

    val_render_pixels, val_shading = render_pixel_from_sh(
        sh_coeff,
        world_normals,
        albedo,
        torch_mode,
        return_shading=True,
    )

    assert dataset.dim is not None
    W, H = dataset.dim

    val_render_image = ro.reconstruct_image(
        W, H, val_render_pixels, occupancy_mask, add_alpha=True
    )

    val_shading = np.clip(
        val_shading, 0.0, 1.0
    )  # Clip the shading for proper visualization.
    val_shading_image = ro.reconstruct_image(
        W, H, val_shading, occupancy_mask, add_alpha=True
    )

    # Stick them together
    gt_render_image, _, gt_shading_image, _ = dataset.get_frame_images(frame_number)

    shading_col = np.concatenate([val_shading_image, gt_shading_image], axis=0)
    render_col = np.concatenate([val_render_image, gt_render_image], axis=0)

    return shading_col, render_col


def visualie_SH_on_3D_sphere(
    sh_coeffs,
    camera_orientations=[np.eye(3)],
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
    # Calculate the spherical harmonic Y(l,m)
    surface_values, surface_points = evaluate_SH_on_sphere(sh_coeffs, resolution)
    surface_values = surface_values.reshape(resolution, resolution)

    ic(surface_values.shape, surface_values.min(), surface_values.max())

    fig = plt.figure()
    axes_kwargs = {"projection": "3d", "computed_zorder": False}
    num_plots = len(camera_orientations)
    for i in range(num_plots):
        ax = fig.add_subplot(1, num_plots, i + 1, **axes_kwargs)
        plot_SH_sphere_on_axis(
            ax,
            surface_points,
            surface_values,
            camera_orientations[i],
            draw_gizmos,
            bg_color,
            show_extremes,
        )

    return fig


def evaluate_SH_on_sphere(sh_coeffs, resolution=100):
    mesh_grid = get_sphere_surface_cartesian(resolution)
    cart_normals = meshgrid_to_matrix(*mesh_grid)

    # Calculate the spherical harmonic Y(l,m)
    return (
        evaluate_second_order_SH(sh_coeffs, cart_normals.T, torch_mode=False),
        mesh_grid,
    )


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

        alpha_n = sh.get_SH_alpha(ns[i])
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

    return fig


def plot2npimage(fig):
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer

    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(height, width, 3)
    return img


def orientate_3d_axes(ax, camera_orientation):
    # Decompose the rotation matrix into elevation and azimuth from the camera direction axis
    camera_direction = -np.dot(camera_orientation, np.array([0, 0, -1]))

    # Calculate azimuth and elevation angles
    cx, cy, cz = camera_direction
    azim = np.degrees(np.arctan2(cy, cx))
    elev = np.degrees(np.arcsin(cz))

    ic(elev, azim)
    ax.view_init(roll=0, elev=elev, azim=azim)


if __name__ == "__main__":
    # Read the c2w from file.
    config = Config.get_config()
    split = "test"
    file_path = config.get("dataset", "scene_path") + f"/transforms_{split}.json"

    ic(file_path)
    with open(file_path, "r") as tf:
        frame_transforms = json.loads(tf.read())

    # I am supplying the C2W matrix here. Rotates objects from the camera to world.
    R_front = ro.to_rotation(ro.get_c2w(39, frame_transforms))
    R_back = ro.to_rotation(ro.get_c2w(89, frame_transforms))
    Rs = [R_front, R_back]

    # Good example coeffs from chair int
    # sh_coeff = np.array(
    #     [0.07135, -0.1003, 0.162, -0.09631, -0.04226, 0.02891, 0.1266, -0.04299, -0.1085]
    # )

    # Bad init perturbed
    sh_coeff = np.array(
        [-0.7476, -0.5516, -0.3432, -0.2094, 0.9305, -1.1122, 0.9675, 1.2823, -1.3073]
    )

    fig = visualie_SH_on_3D_sphere(
        sh_coeff,
        camera_orientations=[R_front, R_back],
        bg_color="black",
        resolution=100,
        show_extremes=True,
    )

    # plt.savefig("sphere.png", bbox_inches="tight", pad_inches=0)
    plt.show()
