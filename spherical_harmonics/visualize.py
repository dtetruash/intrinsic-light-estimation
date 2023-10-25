from scipy.special import sph_harm
import numpy as np
from matplotlib import pyplot as plt
from icecream import ic
from spherical_harmonics import render_second_order_SH, get_SH_alpha
from matplotlib import cm  # noqa: F401
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.style as mplstyle

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
            length=1.5,
            color=color,
            pivot="middle",
            arrow_length_ratio=0.05,
            zorder=2,
        )


def visualie_SH_on_3D_sphere(sh_coeffs, rot_matrix=np.eye(3)):
    """Visualize a unit sphere shaded by SH lighting.

    Args:
        sh_coeffs (ndarray): second order SH coefficients
        rot_matrix (ndarray): 3x3 world-to-camera rotation matrix
    """
    cart_normals = meshgrid_to_matrix(*get_sphere_surface_cartesian())
    ic(cart_normals.shape)

    rot_cart_normals = rot_matrix @ cart_normals
    ic(rot_cart_normals.shape)
    ic(rot_cart_normals.T.shape)

    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    fcolors = render_second_order_SH(sh_coeffs, rot_cart_normals.T, torch_mode=False)
    ic(fcolors.shape, fcolors.min(), fcolors.max())
    fcolors = fcolors.reshape(100, 100)

    # Set the aspect ratio to 1 so our sphere looks spherical
    ax = plt.axes(projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # Set the plot camera to mimic canonical camera-space
    ax.view_init(elev=90, azim=-90)  # Look at the +XY plane

    x, y, z = matrix_to_meshgrid(rot_cart_normals)
    ic(x.shape, y.shape, z.shape)
    ic(x.min(), x.max())
    ic(y.min(), y.max())
    ic(z.min(), z.max())
    norm = Normalize(vmin=fcolors.min(), vmax=fcolors.max())
    face_colors = cm.viridis(norm(fcolors))
    surface = ax.plot_surface(
        x,
        y,
        z,
        facecolors=face_colors,
        linewidth=0,
        antialiased=False,
        rstride=1,
        cstride=1,
        zorder=1,
    )

    # Draw the axis guides
    draw_3D_axis(ax, rot_matrix)
    # ax.set_axis_off()

    # Turn off the axis planes
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
