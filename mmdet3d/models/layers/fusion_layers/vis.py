import argparse
import sys, time
import cv2
import numpy as np


import open3d as o3d


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]])


class Visualizer():
    def __init__(self, mode='3d'):
        self.scene_2D_width = 750
        self.user_press = None

        if mode != '3d':
            return

        self.__visualizer = o3d.visualization.Visualizer()
        self.__visualizer.create_window(width=1280, height=720)
        self.__pcd = o3d.geometry.PointCloud()
        self.__visualizer.add_geometry(self.__pcd)

        opt = self.__visualizer.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        self.zoom = 0.3  # smaller is zoomer

        self.__view_control = self.__visualizer.get_view_control()
        self.__view_control.translate(30, 0)

        self.R = rotx(-np.pi / 2.5) @ rotz(np.pi / 2)

    def visuallize_pointcloud(self, points,colors=None,blocking=True):


        if colors is not None:
            colors = colors / 255
            self.__pcd.colors = o3d.utility.Vector3dVector(colors)
        self.__pcd.points = o3d.utility.Vector3dVector(points)
        self.__pcd.rotate(self.R, self.__pcd.get_center())
        if blocking:
            o3d.visualization.draw_geometries([self.__pcd])
        else:
            # non blocking visualization
            self.__visualizer.add_geometry(self.__pcd)

            # control the view camera (must be after add_geometry())
            # self.__view_control.translate(30,0)
            self.__view_control.set_zoom(self.zoom)

            self.__visualizer.update_renderer()
            self.__visualizer.poll_events()

        screenshot = self.__visualizer.capture_screen_float_buffer()
        return (np.array(screenshot) * 255).astype(np.uint8)[:, :, ::-1]


if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='2d', help='mode of visualization can be 2d or 3d')
    args = parser.parse_args()

    visualizer = Visualizer(mode=args.mode)

    path = '../tensorrt_inference/data/results_pointclouds'
    paths = sorted(os.listdir(path))
    pointclouds_paths = [os.path.join(path, pointcloud_path) for pointcloud_path in paths]

    for pointcloud_path in pointclouds_paths:
        pointcloud = np.fromfile(pointcloud_path, dtype=np.float32).reshape((-1, 4))
        bev = visualizer.visualize_painted_pointcloud(pointcloud=pointcloud)

        print("pointcloud.shape ", pointcloud.shape)
        semantic_channel = pointcloud[:, 3]
        semantic_channel = semantic_channel[semantic_channel != 255]
        semantic_df = pd.DataFrame(semantic_channel)
        # semantic_df.hist(bins=100)
        print(semantic_df.value_counts())

        if args.mode == '2d':
            cv2.imshow("bev", bev)
            plt.show()
            if cv2.waitKey(0) == 27:
                exit()
        else:
            visualizer.visuallize_pointcloud(pointcloud, True)

