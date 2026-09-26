#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 18:57:28 2018
Last update on May 2024
@author: arpmstaff
"""

from PIL import Image
from matplotlib.offsetbox import (OffsetImage, AnchoredOffsetbox)
from matplotlib._pylab_helpers import Gcf
import matplotlib.pyplot as plt
import numpy as np
import os.path as path


DEF_HISTPLOT_LABELS = [
    'Density',
    'Probability',
    'Count',
    'Frequency',
    'Proportion',
    'Percent'
]

DEF_JOINTPLOT_LABELS = [
    'None'
]


def get_bottom_right_axis(fig):
    anchor = np.array([1, 0])
    min_dist = np.Inf
    bottom_right_ax = None
    for i, x in enumerate(fig.get_axes()):
        bbox_dots = x.get_position()
        dots = bbox_dots.get_points()
        se_point = np.array([dots[1, 0], dots[0, 1]])
        if min_dist > np.linalg.norm(se_point - anchor):
            min_dist = np.linalg.norm(se_point - anchor)
            bottom_right_ax = x
    return bottom_right_ax


def add_watermark_to_ax(fig, ax, location, size_fraction, alpha1):
    img = Image.open(path.join(path.dirname(__file__), 'logo.png'))
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    # make the watermark width and height 1/16 of the x axis
    wm_width = int(width * size_fraction)
    scaling = wm_width / float(img.size[0])
    wm_height = int(float(img.size[1])*float(scaling))

    if wm_height > height/4:
        wm_height = int(height/4)
        scaling = wm_height/float(img.size[1])
        wm_width = int(img.size[0]*scaling)

    img = img.resize((wm_width, wm_height), Image.LANCZOS)

    imagebox = OffsetImage(img, zoom=1, alpha=alpha1)
    imagebox.image.axes = ax

    ao = AnchoredOffsetbox(location, pad=0.01, borderpad=0., child=imagebox)
    ao.patch.set_alpha(0)
    ax.add_artist(ao)


def add_logo(fig=None, axis=None, location=4, set_fig_size=False,
             size_frac_x=1/16, alpha=0.4, legend=False, type=None, data_for_lim=None):
    sns_marker = None
    if fig is None:
        sns_marker = True
        fig = plt.gcf()  # get current figure if not provided
    if set_fig_size:
        fig.set_size_inches(10, 8)
    if axis is None:
        ax = get_bottom_right_axis(fig)
        add_watermark_to_ax(fig, ax, location, size_frac_x, alpha)
    else:
        add_watermark_to_ax(fig, axis, location, size_frac_x, alpha)
    if legend:
        plt.legend()
    if type == 'hist':
        if plt.gca().get_ylabel() in DEF_HISTPLOT_LABELS:
            plt.ylabel(" ")
    elif type == 'joint':
        if plt.gca().get_ylabel() in DEF_JOINTPLOT_LABELS:
            plt.ylabel(" ")
        if plt.gca().get_xlabel() in DEF_JOINTPLOT_LABELS:
            plt.xlabel(" ")
    elif type == 'geo':
        ax.set_aspect('equal')
        if isinstance(data_for_lim, np.ndarray):
            x = data_for_lim[0]
            y = data_for_lim[1]
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            padding = (max_val - min_val)*0.1
            min_val -= padding
            max_val += padding
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
    if sns_marker:
        plt.show()
