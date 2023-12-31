import glob
import copy
import json
import argparse
import os
import pandas as pd
from tqdm import tqdm

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


def render_image_from_sh(
    frame_number, sh_coeff, dataset, add_alpha=True, torch_mode=True
):
    # this method produces sh-rendered images of the render and shading
    # of a given frame of a dataset.
    # The images should be well-formed on the [0,1] domain

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

    render_image = ro.reconstruct_image(
        W, H, val_render_pixels, occupancy_mask, add_alpha=add_alpha
    )

    val_shading = np.clip(
        val_shading, 0.0, 1.0
    )  # Clip the shading for proper visualization.
    shading_image = ro.reconstruct_image(
        W, H, val_shading, occupancy_mask, add_alpha=add_alpha
    )

    return render_image, shading_image


def visualize_scene_frame_from_sh(frame_number, sh_coeff, dataset, torch_mode=True):
    # This method makes column comparison between sh-render and gt.
    # It makes columns for both the shading and for the render

    # Make top row of infered images
    val_render_image, val_shading_image = render_image_from_sh(
        frame_number, sh_coeff, dataset, add_alpha=True, torch_mode=torch_mode
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
    plot_mode="row",
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

    fig = plt.figure(figsize=(4.31, 4.31))
    axes_kwargs = {"projection": "3d", "computed_zorder": False}
    num_plots = len(camera_orientations)
    for i in range(num_plots):
        if plot_mode == "row":
            ax = fig.add_subplot(1, num_plots, i + 1, **axes_kwargs)
        elif plot_mode == "col":
            ax = fig.add_subplot(num_plots, 1, i + 1, **axes_kwargs)
        else:
            raise ValueError(f"Unknown plot_mode {plot_mode}")

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

    parser = argparse.ArgumentParser(description="Your script description here")

    # Mode argument
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="row",
        choices=["row", "col", "sep"],
        help="Specify the mode (default: row)",
    )

    # Views argument
    parser.add_argument(
        "-v",
        "--views",
        nargs="+",
        type=int,
        default=[39, 89],
        required=True,
        help="List of integers representing views (at least one element is required)",
    )

    # Sh-files argument
    parser.add_argument(
        "-f",
        "--sh-files",
        nargs="+",
        type=str,
        required=True,
        help="List of file paths to CSV files",
    )

    # show overflow
    parser.add_argument(
        "--show-overflow", action="store_true", help="Render overflow on plots"
    )

    # show overflow
    parser.add_argument(
        "-r", "--res", type=int, default=200, help="Resolution of the sphere surface"
    )

    # Output path argument
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output path for the results"
    )

    args = parser.parse_args()

    # Set default value for --output based on -f
    if args.output is None:
        args.output = [
            os.path.join(os.path.dirname(file_path), "sh_vis")
            for file_path in args.sh_files
        ]

    # I am supplying the C2W matrix here. Rotates objects from the camera to world.
    views_Rs = [ro.to_rotation(ro.get_c2w(v, frame_transforms)) for v in args.views]

    #
    args.sh_files = [
        file_path for pattern in args.sh_files for file_path in glob.glob(pattern)
    ]

    ic(args.mode)
    ic(args.views)
    ic(args.sh_files)
    ic(args.output)

    # Get SH from csv file
    sh_coeff_files = args.sh_files
    for sh_i, sh_file in tqdm(enumerate(sh_coeff_files), desc="File"):
        output_dir = args.output[sh_i]
        os.makedirs(output_dir, exist_ok=True)

        output_name_prefix = f"sh_vis_{config.get('dataset','scene')}_{split}"
        if args.show_overflow:
            output_name_prefix += "_show-overflow"

        # read the csv:
        df = pd.read_csv(sh_file)
        df = df.apply(pd.to_numeric, errors="coerce")
        sh_coeff = df.to_numpy()[0]

        # Now, if 'sep' then for loop over -v and render each image
        if args.mode == "sep":
            for v_i, v_R in tqdm(enumerate(views_Rs), desc="View"):
                # make a scphere and export it
                fig = visualie_SH_on_3D_sphere(
                    sh_coeff,
                    camera_orientations=[v_R],
                    bg_color="white",
                    resolution=args.res,
                    show_extremes=args.show_overflow,
                )

                # export image
                output_name = f"{output_name_prefix}_{args.views[v_i]:03d}.png"
                fig.savefig(
                    f"{output_dir}/{output_name}", bbox_inches="tight", pad_inches=0
                )

                plt.close()

        else:
            fig = visualie_SH_on_3D_sphere(
                sh_coeff,
                camera_orientations=views_Rs,
                bg_color="white",
                resolution=args.res,
                show_extremes=args.show_overflow,
                plot_mode=args.mode,
            )

            output_name = f"{output_name_prefix}_{args.mode}_{'-'.join(args.views)}.png"
            fig.savefig(f"{output_dir}/{output_name}", bbox_inches="tight", pad_inches=0)

            plt.close()

    # plt.savefig("sphere.png", bbox_inches="tight", pad_inches=0)
    plt.show()
