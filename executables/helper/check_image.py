
from training_utils.trainer_frontend_helper import TrainerFrontendHelper
from utils import utils
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from utils.image_utils import plot_image_observation
import numpy as np
import pickle
import sys

"""
Plot images from pkl file
"""
imagez_merged=[]
#filename = '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_filter_obstacle_0.25/area5a/success_v2_44k/img_data_rgb_1024_1024_3_90.00_90.00_0.01_20.00_0.22_18_10_100_80_-45_1.000/file1.pkl'
filename = '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/1030-SVM1/file0.pkl'
with open(filename, 'rb') as handle:
    data = pickle.load(handle)
    imgs_nmkd = data['image']
    imgs_nmkd = imgs_nmkd[:10, :, :, :]
    imagez_merged = np.concatenate(imgs_nmkd)
    imagez_merged = np.expand_dims(imagez_merged, axis=0)

    fig, _, axs = utils.subplot2(plt, (len(imgs_nmkd), 1), (8, 8), (.4, .4))
    axs = axs[::-1]
    for idx, img_mkd in enumerate(imagez_merged):
        ax = axs[idx]
        size = img_mkd.shape[0] * 0.05
        plot_image_observation(ax, img_mkd, size)
        ax.set_title('Img: {:d}'.format(idx))

    #figdir = os.path.join('/home/anjianl/Desktop/', 'imgs')
    figdir = os.path.join('/local-scratch/tara/', 'imgs')

    utils.mkdir_if_missing(figdir)
    figname = os.path.join(figdir, '{:d}.pdf'.format(2))
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)