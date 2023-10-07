"""This file was authored by Partha Das.
Included with permission.
"""
import numpy as np
import open3d as o3d


class PCDPoint:
    """
    Convenience class for 3d point management.
    """

    def __init__(self, coord=np.zeros((1, 3)), colour=np.zeros((1, 3))):
        self.coord = coord
        self.colour = colour

    def xyz(self):
        return self.coord[0, 0], self.coord[0, 1], self.coord[0, 2]

    def rgb(self):
        return self.colour[0, 0], self.colour[0, 1], self.colour[0, 2]


def depth2pcd_point(depth_uvz, cam_int, rgb_colour=np.ones((1, 3)) * 255):
    """
    Returns the 3d world point for a given depth point value.

    depth_uvz : coordinates of the depth pixel, ie (*pixel_coord, z-value)
    cam_int: 4x4 intrinsics matrix
    rgb_colour: 8-bit RGB values
    """
    f_x = cam_int[0][0]
    f_y = cam_int[1][1]
    c_x = cam_int[0][2]
    c_y = cam_int[1][2]

    u, v, z = depth_uvz
    x = ((v - c_x) * z) / f_x
    y = ((u - c_y) * z) / f_y

    return PCDPoint(np.array([x, y, z]).reshape(1, 3), rgb_colour / 255)


def depth2pcd(depth, img, cam_intrinsics):
    """
    Iteratively creates a pointcloud from a depth image. Is slower, because
    each point in the depth image is iterated. But useful for cases when,
    each point needs some custom transformation before accumulating into a
    point cloud. Without pose information.
    """
    h, w = depth.shape
    total_points = h * w

    total_pcd_array = np.zeros((total_points, 3))
    total_colour_array = np.zeros((total_points, 3))

    cnt = 0
    for u in range(h):
        for v in range(w):
            uvz_point = [u, v, depth[u, v]]
            rgb_colour_point = img[u, v, :].reshape(1, 3)
            depth_point = depth2pcd_point(uvz_point, cam_intrinsics, rgb_colour_point)
            x, y, z = depth_point.xyz()
            r, g, b = depth_point.rgb()

            total_pcd_array[cnt, :] = np.array([x, y, z])
            total_colour_array[cnt, :] = np.array([r, g, b])
            cnt += 1

    total_pcd = o3d.geometry.PointCloud()
    total_pcd.points = o3d.utility.Vector3dVector(np.asarray(total_pcd_array))
    total_pcd.colors = o3d.utility.Vector3dVector(np.asarray(total_colour_array))
    return total_pcd


def depth2pcd_posed(depth, img, cam_intrinsics, cam_pose):
    """
    Iteratively creates a pointcloud from a depth image. Is slower, because
    each point in the depth image is iterated. But useful for cases when,
    each point needs some custom transformation before accumulating into a
    point cloud. Without pose information. With pose information.
    """
    h, w = depth.shape
    total_points = h * w

    total_pcd_array = np.zeros((total_points, 3))
    total_colour_array = np.zeros((total_points, 3))

    cnt = 0
    for u in range(h):
        for v in range(w):
            uvz_point = [u, v, depth[u, v]]
            rgb_colour_point = img[u, v, :].reshape(1, 3)
            depth_point = depth2pcd_point(uvz_point, cam_intrinsics, rgb_colour_point)
            x, y, z = depth_point.xyz()
            r, g, b = depth_point.rgb()

            total_pcd_array[cnt, :] = np.array([x, y, z])
            total_colour_array[cnt, :] = np.array([r, g, b])
            cnt += 1

    total_pcd_array_aug = np.concatenate(
        [total_pcd_array, np.ones((total_points, 1))], axis=1
    )
    total_pcd_transformed = np.dot(total_pcd_array_aug, cam_pose)

    total_pcd = o3d.geometry.PointCloud()
    total_pcd.points = o3d.utility.Vector3dVector(
        # np.asarray(total_pcd_array)
        np.asarray(total_pcd_transformed[:, :3])
    )
    total_pcd.colors = o3d.utility.Vector3dVector(np.asarray(total_colour_array))
    return total_pcd


def create_point_cloud(
    frame_depth, frame_img, frame_intrinsics, frame_pose, save_name="cluster1.ply"
):
    """
    Wrapper class to create a point cloud using the other functions.
    If you want multiframe for a complete scan, add a glob here to
    iterate through all the files and loop over and accumulate the
    point clouds into a larger scan.

    frame_depth: np.ndarray of depth data
    frame_img: np.ndarray of image data
    frame_intrinsics : np.ndarray (4x4 intrinsics)
    frame_pose : np.ndarray (4x4 extrinsics)
    """

    frame_A_pcd = o3d.geometry.PointCloud()

    frame_pcd = depth2pcd_posed(frame_depth, frame_img, frame_intrinsics, frame_pose)
    frame_A_pcd += frame_pcd

    o3d.io.write_point_cloud(save_name, frame_A_pcd)

    o3d.visualization.draw_geometries([frame_A_pcd])


def create_point_cloud_open3d(frame_depth, frame_img, frame_intrinsics, frame_pose):
    """
    Same as above, but uses open3d to create a point cloud from depth
    and rgb image.

    frame_depth: np.ndarray of depth data
    frame_img: np.ndarray of image data
    frame_intrinsics : np.ndarray (4x4 intrinsics)
    """
    frame_A_pcd = o3d.geometry.PointCloud()

    colour_raw = o3d.geometry.Image(frame_img)
    depth_raw = o3d.geometry.Image(frame_depth)

    cam_int_obj = o3d.camera.PinholeCameraIntrinsic()
    cam_int_obj.intrinsic_matrix = frame_intrinsics

    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colour_raw, depth_raw, convert_rgb_to_intensity=False
    )

    frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, cam_int_obj)

    frame_A_pcd += frame_pcd.transform(frame_pose)

    o3d.visualization.draw_geometries([frame_A_pcd])


if __name__ == "__main__":
    create_point_cloud()
