#!/usr/bin/env python

import argparse
import glplotlib.glplot as glt

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("example_idx", type=int)

    args = parser.parse_args()

    data_dict = np.load(args.file)

    pre_band = data_dict["{}/band_pre".format(args.example_idx)]
    post_band = data_dict["{}/band_post".format(args.example_idx)]
    local_environment = data_dict["{}/local_env".format(args.example_idx)]

    glt.axis_generic()
    glt.grid_generic()
    glt.show(persistent=False)

    plot_voxel_grid(local_environment, [1, 1, 1])
    plot_voxel_grid(pre_band, [0, 1, 0])
    plot_voxel_grid(post_band, [0, 0, 1])


def plot_voxel_grid(voxel_grid, color):
    r = np.arange(voxel_grid.shape[0])
    points = np.stack(np.meshgrid(r, r, r), axis=3).reshape(-1, 3)

    occupied_points = []
    for point in points:
        if voxel_grid[point[0], point[1], point[2]] > 0.5:
            occupied_points.append(point)
    occupied_points = np.array(occupied_points)
    color.append(0.5)
    glt.scatter_generic(occupied_points, color=color, size=7)


if __name__ == '__main__':
    main()
