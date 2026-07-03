"""Generating data from SDD dataset 
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import argparse

from glob import glob
import numpy as np

from planning_datasets_utils.SDD import create_sample_from_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', '-w', type=int, default=64)
    parser.add_argument('--min-len-ratio', '-ml', type=float, default=0.5)
    parser.add_argument('--min-max-size', '-mm,', type=int, default=128)
    parser.add_argument('--max-step', '-ms,', type=int, default=300)
    parser.add_argument('--save-dir', '-s', type=str, default='data/sdd')

    args = parser.parse_args()
    W = int(args.size)
    min_len_ratio = float(args.min_len_ratio)
    min_max_size = int(args.min_max_size)
    max_step = int(args.max_step)
    save_dir = "%s/s%03d_%0.1f_%d_%d" % (args.save_dir, W, min_len_ratio,
                                         min_max_size, max_step)

    np.random.seed(0)

    for data_dir in sorted(glob('data/sdd/original/*/video*')):
        print(data_dir)
        create_sample_from_dataset(data_dir, save_dir, W, min_len_ratio, min_max_size, max_step)
