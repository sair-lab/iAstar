"""preprocessing script for CSM dataset
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import os
import re

from glob import glob
import numpy as np
from PIL import Image


def crop_images(inputdir: str = "data/street/original/all",
                outputdir: str = "data/street/original/mixed",
                W: int = 128,
                Ns: list = [160, 20, 40]) -> None:
    train_filenames = glob("%s/*[0,1]_256.png" % inputdir)
    test_filenames = glob("%s/*2_256.png" % inputdir)
    for filename in train_filenames:
        fileid = re.split("_256.png", re.split("/", filename)[-1])[0]
        img = Image.open(filename).convert("L").resize((256, 256))
        lt = np.random.randint(0, 256 - W, [Ns[0], 2])
        boxes = np.hstack((lt, lt + W))
        for n, box in enumerate(boxes):
            cropped = img.crop(box)
            cropped.save(
                os.path.join(outputdir, 'train', "%s_%05d.png" % (fileid, n)))
    for filename in train_filenames:
        fileid = re.split("_256.png", re.split("/", filename)[-1])[0]
        img = Image.open(filename).convert("L").resize((256, 256))
        lt = np.random.randint(0, 256 - W, [Ns[1], 2])
        boxes = np.hstack((lt, lt + W))
        for n, box in enumerate(boxes):
            cropped = img.crop(box)
            cropped.save(
                os.path.join(outputdir, 'validation',
                             "%s_%05d.png" % (fileid, n)))
    for filename in test_filenames:
        fileid = re.split("_256.png", re.split("/", filename)[-1])[0]
        img = Image.open(filename).convert("L").resize((256, 256))
        lt = np.random.randint(0, 256 - W, [Ns[2], 2])
        boxes = np.hstack((lt, lt + W))
        for n, box in enumerate(boxes):
            cropped = img.crop(box)
            cropped.save(
                os.path.join(outputdir, 'test', "%s_%05d.png" % (fileid, n)))


if __name__ == "__main__":
    np.random.seed(1372)
    crop_images()
