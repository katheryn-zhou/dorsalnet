# Calculate tuning curves for drifting gratings
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
import paths
sys.path.append(paths.CPC_DPC)
sys.path.append(paths.CPC_BACKBONE)

from python_dict_wrapper import wrap

import os
from tqdm import tqdm
import collections
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.optimize
# import jax

# import jax.numpy as jnp
# from jax import grad, jit, vmap

import tables
from pathlib import Path

f = tables.open_file('../data_derived/airsim/batch2/nh_fall/2021-02-02T005329/output.h5')
print(f)
print(f.get_node('/short_videos')) # (2827, 10, 3, 112, 112) = (trials, timesteps, C, H, W)
images = f.get_node('/short_videos')[6, :, :, :, :]
print(images.shape) # (10, 3, 112, 112)

for i in range(3):
    plt.subplot(131 + i)
    plt.imshow(images[i, :, :, :].transpose((1, 2, 0)))
    print(images[i, :, :, :].transpose((1, 2, 0)).shape)
    plt.axis('off')
    plt.savefig("example_stimuli")