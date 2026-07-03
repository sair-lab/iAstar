#!/usr/bin/env python
"""
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

from setuptools import setup, find_packages

setup(name="planning_datasets_utils",
      version="0.1.0",
      description="Path Planning using Neural A* Search",
      author="Ryo Yonetani",
      author_email="ryo.yonetani@sinicx.com",
      url="https://github.com/omron-sinicx/planning-datasets",
      install_requires=[
          "torch>=1.5.0",  # use 1.5.0 for full reproduciblity 
          "numpy>=1.22",   # to address a security vulnerability in numpy<= 1.21.6. Full reproducibility requires numpy==1.18.4 
          "tqdm==4.42.1",
          "natsort==7.0.1",
          "ipython>=7.16.3",
          "scikit-image>=0.17.2",  # use 0.17.2 for full reproducibility
          "pandas>=1.0.3",
      ],
      packages=find_packages())
