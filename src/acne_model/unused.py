#for currently unused functions that may be useful later
#imports

import matplotlib
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle as rect
from itertools import permutations
from colorsys import rgb_to_hls, hls_to_rgb
import pandas as pd
import scipy as sp
from scipy import optimize
import numpy as np
import seaborn as sns
import copy
from collections import defaultdict, Counter
from matplotlib.cm import viridis
import statsmodels.api as sm
from scipy.stats import dirichlet
from scipy.stats import beta
from scipy.special import gammaln, psi
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
import json
import filterpy as fp
import numdifftools as nd
from functools import partial

def blend_old_and_new_color(old_color, new_color, alpha):
    """Another function used by display_plots_of_dataset. Simply blends 2 colors with a given alpha value."""
    # normalize both old and new colors into decimals
    o = np.array(old_color)
    n = np.array(new_color)

    o_h, o_l, o_s = rgb_to_hls(*o)
    n_h, n_l, n_s = rgb_to_hls(*n)

    # circular interpolation of hue for more effective blending
    hue_diff = ((n_h - o_h + 0.5) % 1.0) - 0.5
    blended_h = (o_h + alpha * hue_diff) % 1.0
    blended_l = (1 - alpha) * o_l + alpha * n_l
    blended_s = (1 - alpha) * o_s + alpha * n_s

    blended_rgb = np.array(hls_to_rgb(blended_h, blended_l, blended_s))
    return tuple(blended_rgb)

def linear_generator_starting_colors(number_treatments, base_colormap = "viridis"):
    """A helper function for display_plots_of_dataset below. Generates n RGB tuples on a linear scale, given the number of treatments.
    Each of these is used as a starting point for a regular map of treatments by day and a blended heatmap of treatments by day, with
    lower lightness for each consecutive day of the same treatment, blended with the last color corresponding to the days of the
    previous treatments."""

    cmap = plt.get_cmap(base_colormap)
    default_colors = [cmap(i / (number_treatments - 1))[:3] for i in range(number_treatments)]
    return default_colors

def display_plots_of_dataset(separate_dfs, patient_intro_days, alpha=.3):
    """This function plots the raw data for inspection.
    Each patient corresponds to a series of rectangles, with face color corresponding to treatment.
    The facecolor changes brightness based on the number of days of a consecutive treatment are applied,
    setting up potential for conditional dependency analysis between treatment history distributions later
    on. It uses a colorsys mapping between a given starting set of RGB tuples, one for each type of
    treatment."""

    x_dim = 200
    y_dim = 200

    data_fig = plt.figure(figsize=(x_dim, y_dim))
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)

    mainPanelHeight = 3.75
    mainPanelWidth = 5

    otherMainPanelHeight = 3.75
    otherMainPanelWidth = 5

    legendPanelHeight = 3
    legendPanelWidth = .5

    sidePanelHeight = 3

    sidePanelWidth = .25

    # setting up the panels and placing the proper positions
    firstMainPanel = plt.axes([.05 / x_dim, .375 / y_dim, mainPanelWidth / x_dim, mainPanelHeight /
                               y_dim])
    firstMainPanel.set_xlabel("Treatment Day")
    firstMainPanel.set_ylabel("Patient ID")
    firstMainPanel.set_title("Dataset Overview")

    otherMainPanel = plt.axes(
        [.05 / x_dim, (1.25 + mainPanelHeight) / y_dim, otherMainPanelWidth / x_dim, otherMainPanelHeight /
         y_dim])
    otherMainPanel.set_xlabel("Treatment Day")
    otherMainPanel.set_ylabel("Patient ID")
    otherMainPanel.set_title("Dataset Overview (Blended)")

    main_bottom = 0.375 / y_dim
    main_height = mainPanelHeight / y_dim
    main_top = main_bottom + main_height

    other_bottom = (1.25 + mainPanelHeight) / y_dim

    legend_bottom = (main_top + other_bottom) / 2

    # setting up the legend panel
    legendRight = plt.axes([(1.5 + otherMainPanelWidth) / x_dim, legend_bottom - (.5 * legendPanelHeight) / y_dim,
                            legendPanelWidth / x_dim, legendPanelHeight / y_dim])
    # seting ticks of legend
    legendRight.set_title("Treatments (Base Color)")
    legendRight.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True, right=False, labelright=False,
                            top=False, labeltop=False)

    all_patient_IDS = set()

    starting_colors_dict = defaultdict(tuple)
    # getting the set of all treatments
    all_treatments_all_patients = set()  # mental note: trying to make the mapping work for default colors for each treatment with default dict
    for separated_dataframe in separate_dfs:
        patient_ID = set(separated_dataframe["patient_id"].tolist())

        all_patient_IDS.update(patient_ID)
        treatments_series_set = set(separated_dataframe["treatment"].tolist())
        all_treatments_all_patients.update(treatments_series_set)
    # finding the set of n RGB tuples on a linear scale given all of the n treatments that were collected
    initial_colors = linear_generator_starting_colors(len(all_treatments_all_patients))

    for treatment_type, each_unlightened_color in zip(all_treatments_all_patients, initial_colors):
        starting_colors_dict[treatment_type] = each_unlightened_color

    for index, separated_dataframe in enumerate(separate_dfs):
        treatment_series_list = separated_dataframe["treatment"].tolist()
        current_treatment = treatment_series_list[0]
        treatment_streak_length = 1
        last_facecolor = starting_colors_dict[treatment_series_list[0]]
        for day, which_treatment in enumerate(treatment_series_list):
            bar_width = 1
            lightness_factor = max(0.7, 1 - 0.05 * (
                        treatment_streak_length - 1))  # used to darken base facecolors for a given treatment with repeated treatment
            if which_treatment == current_treatment:
                treatment_streak_length += 1
            else:
                current_treatment = which_treatment
                treatment_streak_length = 1

            # unblended rectangle's facecolor and plotting
            unblended_facecolor = starting_colors_dict[which_treatment]
            rect_to_add = rect((day, index), width=1, height=1, facecolor=unblended_facecolor, edgecolor='black',
                               linewidth=0.25)
            firstMainPanel.add_patch(rect_to_add)

            # blended rectangle's facecolor and plotting
            r, g, b = unblended_facecolor[0], unblended_facecolor[1], unblended_facecolor[2]
            facecolor_unblended_hls = rgb_to_hls(r, g, b)
            streak_aware_facecolor_unblended = hls_to_rgb(facecolor_unblended_hls[0],
                                                          facecolor_unblended_hls[1] * lightness_factor,
                                                          facecolor_unblended_hls[2])
            blended_streak_aware_facecolor = blend_old_and_new_color(last_facecolor, streak_aware_facecolor_unblended,
                                                                     alpha)

            blended_rect_to_add = rect((day, index), width=1, height=1, facecolor=blended_streak_aware_facecolor,
                                       edgecolor='black', linewidth=0.25)
            otherMainPanel.add_patch(blended_rect_to_add)
            last_facecolor = blended_streak_aware_facecolor

    # setting limits of first panel and second panel
    max_days = max(len(df["treatment"]) for df in separate_dfs)
    firstMainPanel.set_xlim(0, max_days)
    firstMainPanel.set_ylim(0, len(separate_dfs) - 0.5)
    firstMainPanel.set_yticks([index for index in range(len(separate_dfs))])
    firstMainPanel.set_yticklabels(all_patient_IDS)
    firstMainPanel.invert_yaxis()

    otherMainPanel.set_xlim(0, max_days)
    otherMainPanel.set_ylim(0, len(separate_dfs) - 0.5)
    otherMainPanel.set_yticks([index for index in range(len(separate_dfs))])
    otherMainPanel.set_yticklabels(all_patient_IDS)
    otherMainPanel.invert_yaxis()

    # adding base colors to the legend panel
    legendRight.set_xlim(0, .1)
    legendRight.set_ylim(0, len(all_treatments_all_patients))

    which_y = 0
    for treatment, starting_color in starting_colors_dict.items():
        given_rectangle = rect((0, which_y), width=1, height=1, facecolor=starting_color, edgecolor='black',
                               linewidth=0.25)
        legendRight.add_patch(given_rectangle)
        which_y += 1
    legendRight.set_yticks([index + .5 for index in range(len(starting_colors_dict.keys()))])
    legendRight.tick_params(axis='y', labelsize=10)
    legendRight.set_yticklabels(starting_colors_dict.keys())

    plt.show()

def fit_piecewise_regression_and_plot_unused(points, splits):
    """This function actually splits a given array (here, cumulative KL divergence) into seperate regression models,
    using splits to partition the array and then fit a linear regression model to each one.
    It then calls plot_piecewise_regression_segments to plot the segments over the cumulative KL divergence Curve. Unused in this version."""
    split_points_and_indices = [(points[split[0]:split[1] + 1], split) for split in splits]
    linear_model = LinearRegression()
    consecutive_models = [
        LinearRegression().fit(np.arange(len(one_split_points[0])).reshape(-1, 1), one_split_points[0].reshape(-1, 1))
        for one_split_points in split_points_and_indices]
    slopes = [which_model.coef_[0] for which_model in consecutive_models]