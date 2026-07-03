"""problem instance generation utils
Author: Mohammadamin Barekatain, Ryo Yonetani
Affiliation: OMRON SINIC X
Part of this script has been copied from https://github.com/RLAgent/gated-path-planning-networks
"""

# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
from typing import Tuple

import numpy as np
from natsort import natsorted
from skimage.measure import label
from skimage.filters import threshold_otsu
from PIL import Image
import glob
import random
from tqdm import tqdm

from .dijkstra import dijkstra_dist
from .mechanism import Mechanism


def extract_policy(maze: np.ndarray, mechanism: Mechanism,
                   value: np.ndarray) -> np.ndarray:
    """
    Extracts the policy from the given values.

    Args:
        maze (np.ndarray): maze data
        mechanism (Mechanism): one of news (4 neighbors) or moore (8 neighbors)
        value (np.ndarray): optimal distance map obtained using dijkstra

    Returns:
        np.ndarray: policy map
    """
    policy = np.zeros((mechanism.num_actions, value.shape[0], value.shape[1],
                       value.shape[2]))
    for p_orient in range(value.shape[0]):
        for p_y in range(value.shape[1]):
            for p_x in range(value.shape[2]):
                # Find the neighbor w/ max value (assuming deterministic
                # transitions)
                max_val = -sys.maxsize
                max_acts = [0]
                neighbors = mechanism.neighbors_func(maze, p_orient, p_y, p_x)
                for i in range(len(neighbors)):
                    n = neighbors[i]
                    nval = value[n[0]][n[1]][n[2]]
                    if nval > max_val:
                        max_val = nval
                        max_acts = [i]
                    elif nval == max_val:
                        max_acts.append(i)

                # Choose max actions if several w/ same value
                max_act = max_acts[np.random.randint(len(max_acts))]
                policy[max_act][p_orient][p_y][p_x] = 1.0
    return policy


def load_maze_from_directory(input_path: str, split: str,
                             size: int) -> np.ndarray:
    """
    Load a set of maze maps from a specified directory

    Args:
        input_path (str): path to the directory
        split (str): one of train/validation/test
        size (int): map size

    Returns:
        np.ndarray: a set of maze maps
    """

    assert split in ["train", "validation", "test"]

    mazes = []
    image_paths = natsorted(glob.glob(os.path.join(input_path, split,
                                                   "*.png")))
    for image_path in image_paths:
        image = np.asarray(
            Image.open(image_path).convert("L").resize((size, size)),
            dtype=np.float32,
        )
        th = threshold_otsu(image)
        image_out = np.zeros_like(image)
        image_out[image > th] = 1.0
        mazes.append(image_out)

    return np.array(mazes)


def get_goalMaps_optPolicies_optDists(
        mazes: np.ndarray,
        mechanism: Mechanism,
        from_largest: bool = True,
        edge_size: int = 0,
        input_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get goal maps as well as optimal policies and distances from each location to the goal

    Args:
        mazes (np.ndarray): maze maps 
        mechanism (Mechanism): one of news (4 neighbors) or moore (8 neighbors)
        from_largest (bool, optional): whether to pick a goal from the largest passable region. Defaults to True.
        edge_size (int, optional): the width of edge from which goals are picked. Defaults to 0.
        input_path (str, optional): path to the original maze data. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: goal maps, optimal policy maps, and optimal distance maps
    """

    data_size, maze_size = mazes.shape[0], mazes.shape[1]

    goal_maps = np.zeros(
        (data_size, mechanism.num_orient, maze_size, maze_size))
    opt_policies = np.zeros((data_size, mechanism.num_actions,
                             mechanism.num_orient, maze_size, maze_size))
    opt_dists = np.zeros(
        (data_size, mechanism.num_orient, maze_size, maze_size))

    for i, maze in tqdm(enumerate(mazes)):
        # select a random goal which is not an obstacle
        if from_largest:
            limage = label(maze, background=0, connectivity=1)
            num_pixels = np.bincount(limage.flatten())
            num_pixels[0] = 0
            cond = limage == np.argmax(num_pixels)
            if edge_size > 0:  # supperss goal locations to be sampled from center regions
                corner_image = np.ones_like(cond) * True
                corner_image[edge_size:-edge_size, :] = False
                corner_image[:, edge_size:-edge_size] = False
                if np.any(cond & corner_image):
                    cond = cond & corner_image
                else:
                    print('no regions found around any corner ({}, size: {})'.
                          format(input_path, maze_size))
            none_zeros = np.nonzero(cond)
        else:
            none_zeros = np.nonzero(maze > 0.5)

        none_zeros = [(i, j) for i, j in zip(none_zeros[0], none_zeros[1])]
        goal_pos = random.choice(none_zeros)
        goal_orient = np.random.randint(mechanism.num_orient)
        goal_loc = (goal_orient, goal_pos[0], goal_pos[1])

        # update the goal map
        goal_maps[i, goal_loc[0], goal_loc[1], goal_loc[2]] = 1.0

        # Use Dijkstra's to construct the optimal policy
        opt_value = dijkstra_dist(maze, mechanism, goal_loc)
        opt_policy = extract_policy(maze, mechanism, opt_value)

        opt_policies[i, :, :, :, :] = opt_policy
        opt_dists[i, :, :, :] = opt_value

    return goal_maps, opt_policies, opt_dists
