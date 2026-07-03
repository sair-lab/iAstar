"""Generating data for shortest path experiments
Author: Mohammadamin Barekatain, Ryo Yonetani
Affiliation: OMRON SINIC X
"""

from __future__ import print_function
import argparse
import random
import os
import re

import numpy as np
from skimage.util import montage

from planning_datasets_utils import (
    load_maze_from_directory,
    get_goalMaps_optPolicies_optDists,
    get_mechanism,
)


def generate_data(input_path, train_size, valid_size, test_size, mechanism,
                  maze_size, edge_size, tile_size, output_filename):
    mazes = [[] for _ in range(3)]
    goal = [[] for _ in range(3)]
    opt_policies = [[] for _ in range(3)]
    opt_dists = [[] for _ in range(3)]
    for i, (split, num_images) in enumerate(
            zip(["train", "validation", "test"],
                [train_size, valid_size, test_size])):
        tmp = load_maze_from_directory(input_path, split, maze_size)
        if tile_size == 1:
            mazes[i] = tmp
        else:
            tmp_montage = np.zeros(
                (num_images, maze_size * tile_size, maze_size * tile_size))
            for t in range(num_images):
                idx = np.random.choice(len(tmp), tile_size * tile_size)
                tmp_montage[t] = montage([tmp[j] for j in idx])
            mazes[i] = tmp_montage
        goal[i], opt_policies[i], opt_dists[
            i] = get_goalMaps_optPolicies_optDists(mazes[i],
                                                   mechanism,
                                                   from_largest=True,
                                                   edge_size=edge_size,
                                                   input_path=input_path)

    np.savez_compressed(
        output_filename,
        mazes[0],
        goal[0],
        opt_policies[0],
        opt_dists[0],
        mazes[1],
        goal[1],
        opt_policies[1],
        opt_dists[1],
        mazes[2],
        goal[2],
        opt_policies[2],
        opt_dists[2],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",
                        type=str,
                        required=True,
                        help="Directory to the dataset.")
    parser.add_argument("--output-path",
                        type=str,
                        required=True,
                        help="Directory for the output data.")
    parser.add_argument("--maze-size",
                        type=int,
                        required=True,
                        help="Size of mazes.")
    parser.add_argument(
        "--mechanism",
        type=str,
        required=True,
        help="Maze transition mechanism. (news|diffdrive|moore)",
    )
    parser.add_argument(
        "--edge-ratio",
        type=float,
        default=0,
        help="Size of edge regions to sample goals (0 to disable it)")
    parser.add_argument("--train-size",
                        type=int,
                        default=800,
                        help="Number of training mazes.")
    parser.add_argument("--valid-size",
                        type=int,
                        default=100,
                        help="Number of validation mazes.")
    parser.add_argument("--test-size",
                        type=int,
                        default=100,
                        help="Number of test mazes.")
    parser.add_argument("--seed",
                        type=int,
                        default=1372,
                        help="Random generator seed")
    parser.add_argument("--tile-size",
                        type=int,
                        default=1,
                        help="Tiling maps (experimental).")

    args = parser.parse_args()

    mechanism = get_mechanism(args.mechanism)

    np.random.seed(args.seed)
    random.seed(args.seed)

    input_path = args.input_path if args.input_path[
        -1] != "/" else args.input_path[:-1]
    maze_name = re.split("/", input_path)[-1]
    if '*' in maze_name:
        maze_name = 'all'

    maze_size = int(args.maze_size)
    tile_size = int(args.tile_size)
    edge_size = int(args.edge_ratio * maze_size * tile_size)
    train_size = args.train_size
    valid_size = args.valid_size
    test_size = args.test_size
    output_filename = os.path.join(
        args.output_path,
        "{}_{:03d}_{}_c{}.npz".format(maze_name, maze_size * tile_size,
                                      args.mechanism, edge_size),
    )

    print("input:{} output:{}".format(input_path, output_filename))

    generate_data(
        input_path,
        train_size,
        valid_size,
        test_size,
        mechanism,
        maze_size,
        edge_size,
        tile_size,
        output_filename,
    )


if __name__ == "__main__":
    main()
