"""Generating instances for pathfinding
Author: Mohammadamin Barekatain, Ryo Yonetani
Affiliation: OMRON SINIC X
Reference: Path Planning using Neural A* Search
"""

from __future__ import print_function
import argparse
import random
import os
import re
import cv2
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Tuple
from natsort import natsorted
from skimage.util import montage
from skimage.measure import label
from skimage.filters import threshold_otsu

def load_tsdf_from_directory(input_path: str, size: int):
    """
    Load a set of tsdf maps from a specified directory

    Args:
        input_path (str): path to the directory
        split (str): one of train/validation/test
        size (int): map size

    Returns:
        np.ndarray: a set of maze maps
    """

    tsdfs = []
    tsdf_paths = natsorted(glob.glob(os.path.join(input_path, "*.npy")))
    for tsdf_path in tsdf_paths:
        tsdf_arr = np.load(tsdf_path)
        tsdf = cv2.resize(cv2.Mat(tsdf_arr),(size,size))
        tsdfs.append(tsdf)
    return np.array(tsdfs)

def load_maze_data_from_directory(input_path: str, size: int):
    """
    Load a set of tsdf maps from a specified directory

    Args:
        input_path (str): path to the directory
        split (str): one of train/validation/test
        size (int): map size

    Returns:
        np.ndarray: a set of maze maps
    """
    maze_file = glob.glob(os.path.join(input_path, f"{str(size)}/*.npz"))[0]
    mazes = np.load(maze_file)

    return mazes

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

def get_goalMaps(
        mazes: np.ndarray,
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
        Tuple[np.ndarray, np.ndarray, np.ndarray]: goal maps
    """

    data_size, maze_size = mazes.shape[0], mazes.shape[1]

    goal_maps = np.zeros(
        (data_size, 1, maze_size, maze_size))

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
        # update the goal map
        goal_maps[i, 0, goal_pos[0], goal_pos[1]] = 1.0

    return goal_maps

def generate_data(input_path, train_size, valid_size, test_size,
                  maze_size, edge_size, tile_size, output_filename, dataset, thre = 0.2):
    assert dataset in ["matterport", "maze", "mpd"]
    mazes = [[] for _ in range(3)]
    goal = [[] for _ in range(3)]
    tsdfs = [[] for _ in range(3)]
    for i, (split, num_images) in enumerate(
            zip(["train", "validation", "test"],
                [train_size, valid_size, test_size])):
        if dataset == "matterport":        
            tmp_tsdf = load_tsdf_from_directory(input_path, maze_size)
            tmp_maze = np.zeros_like(tmp_tsdf)
            tmp_maze[tmp_tsdf<=thre] = 1.0
            tsdfs[i] = tmp_tsdf
        if dataset == "maze":
            tmp_maze = load_maze_data_from_directory(input_path, maze_size)
        else:
            tmp_maze = load_maze_from_directory(input_path, split, maze_size)
        if tmp_maze.shape[0] != num_images:
            random_idxs = np.random.randint(0, len(tmp_maze), num_images)
            rand_mazes = tmp_maze[random_idxs]
            tmp_maze = rand_mazes
        if tile_size == 1:
            mazes[i] = tmp_maze
        else:
            tmp_montage = np.zeros((num_images, maze_size * tile_size, maze_size * tile_size))
            for t in range(num_images):
                idx = np.random.choice(len(tmp_maze), tile_size * tile_size)
                tmp_montage[t] = montage([tmp_maze[j] for j in idx])
            mazes[i] = tmp_montage
        goal[i] = get_goalMaps(mazes[i],
                            from_largest=True,
                            edge_size=edge_size,
                            input_path=input_path)
    np.savez_compressed(
        output_filename,
        mazes[0],
        goal[0],
        tsdfs[0],
        mazes[1],
        goal[1],
        tsdfs[1],
        mazes[2],
        goal[2],
        tsdfs[2]
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
    parser.add_argument("--edge-ratio",
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
    parser.add_argument("--dataset",
                        type=str,
                        default="matterport",
                        help="dataset")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    input_path = args.input_path if args.input_path[
        -1] != "/" else args.input_path[:-1]
    maze_name = re.split("/", input_path)[-1]
    if '*' in maze_name:
        maze_name = 'all'

    maze_size = int(args.maze_size)
    tile_size = args.tile_size
    edge_size = int(args.edge_ratio * maze_size * tile_size)
    train_size = args.train_size
    valid_size = args.valid_size
    test_size = args.test_size
    output_filename = os.path.join(
        args.output_path,
        "{}_{:03d}_c{}.npz".format(maze_name, maze_size * tile_size, edge_size),
    )
    dataset = args.dataset

    print("input:{} output:{}".format(input_path, output_filename))

    generate_data(
        input_path,
        train_size,
        valid_size,
        test_size,
        maze_size,
        edge_size,
        tile_size,
        output_filename,
        dataset        
    )


if __name__ == "__main__":
    main()
