"""problem instance generation utils for SDD
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import os
import re

import numpy as np
from PIL import Image
import pandas as pd
from scipy.ndimage import label as scipy_label


def create_sample_from_dataset(data_dirname: str,
                               save_dir: str,
                               W: int = 64,
                               min_len_ratio: float = 0.5,
                               min_max_size: int = 128,
                               max_step: int = 600):
    """
    Create samples consisting raw images and ground truth pedestrian trajectories.

    Args:
        data_dirname (str): directory containing reference.jpg and annotations.txt
        save_dir (str): directory to store samples
        W (int, optional): cropped image size. Defaults to 64. 
        min_len_ratio (float, optional): threshold parameter for trajectory complexity. Defaults to 0.5.
        min_max_size (int, optional): minimum size for the bounding box that encompasses a trajectory. Defaults to 128.
        max_step (int, optional): maximum number of steps in trajectories. Defaults to 600.
    """
    dirname_split = re.split('/', data_dirname)
    os.makedirs('%s/%s/%s' % (save_dir, dirname_split[-2], dirname_split[-1]),
                exist_ok=True)

    label_all = {
        'Pedestrian': 0,
        'Biker': 1,
        'Skater': 2,
        'Car': 3,
        'Cart': 4,
        'Bus': 5
    }
    ref_image = Image.open(os.path.join(data_dirname, 'reference.jpg'))
    data = pd.read_csv(os.path.join(data_dirname, 'annotations.txt'), sep=' ', \
                        names=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame_id', 'lost', 'occluded', 'generated', 'label'])
    unique_id = data['id'].unique()
    num_samples = len(unique_id)
    for id in unique_id:
        sample = data[data['id'] == id][[
            'xmin', 'ymin', 'xmax', 'ymax', 'occluded', 'lost'
        ]].to_numpy()
        visible_label = scipy_label(sample[:, -1] == 0)[0]
        for vl in range(1, visible_label.max() + 1):
            sample_ = sample[visible_label == vl]
            traj = ((sample_[:, :2] + sample_[:, 2:4]) / 2.)
            if (len(traj) > max_step):
                start_idx = np.random.randint(0, len(traj) - max_step)
                traj = traj[start_idx:start_idx + max_step, :]
            max_size = np.max(traj.max(axis=0) - traj.min(axis=0), 0)
            mean_loc = traj.min(
                axis=0) + (traj.max(axis=0) - traj.min(axis=0)) / 2
            if (np.all(mean_loc - max_size / 2 - 50 > 0)):
                min_loc = np.maximum(mean_loc - max_size / 2 - 50, 0)
                max_loc = min_loc + max_size + 100
            else:
                max_loc = np.minimum(mean_loc + max_size / 2 + 50,
                                     ref_image.size)
                min_loc = max_loc - max_size - 100
            roi = [min_loc[0], min_loc[1], max_loc[0], max_loc[1]]
            all_length = np.abs(np.diff(traj, axis=0)).sum()
            sg_length = np.abs(np.diff(traj[[0, -1]], axis=0)).sum()
            length_ratio = sg_length / (all_length + 1e-5)
            if ((sg_length > 0) & (length_ratio > min_len_ratio) &
                (max_size.min() > min_max_size)):
                label = label_all[data[data['id'] == id]['label'].unique()[0]]
                traj_resized = ((traj - min_loc) / (max_size + 100) * W)
                traj_resized[traj_resized > W - 1] = W - 1
                traj_resized[traj_resized < 0] = 0
                # densify trajectries to ensure traj_image to be created correctly
                traj0 = np.interp(range(len(traj_resized) * 5),
                                  range(0, 5 * len(traj_resized), 5),
                                  traj_resized[:, 0])
                traj1 = np.interp(range(len(traj_resized) * 5),
                                  range(0, 5 * len(traj_resized), 5),
                                  traj_resized[:, 1])

                traj_image = (np.histogram2d(traj1, traj0, range(W + 1))[0] >
                              0) * 1.
                start_image = np.zeros_like(traj_image)
                start_image[int(traj_resized[0, 1]),
                            int(traj_resized[0, 0])] = 1
                goal_image = np.zeros_like(traj_image)
                goal_image[int(traj_resized[-1, 1]),
                           int(traj_resized[-1, 0])] = 1
                image = np.array(ref_image.crop(roi).resize((W, W)))
                np.savez_compressed(
                    '%s/%s/%s/%08d_%02d.npz' %
                    (save_dir, dirname_split[-2], dirname_split[-1], id, vl),
                    image=image,
                    start_image=start_image,
                    goal_image=goal_image,
                    traj_image=traj_image,
                    traj=traj_resized,
                    label=label,
                    length_ratio=length_ratio)
