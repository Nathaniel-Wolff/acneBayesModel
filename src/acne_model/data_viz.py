#for visualization of data after parsing

from acne_model.raw_data_analysis import build_histograms, build_Dirichlet
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
from collections import defaultdict
from matplotlib.cm import viridis
import statsmodels.api as sm
from scipy.stats import dirichlet
from scipy.stats import beta
from scipy.special import gammaln, psi
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
from acne_model.raw_data_analysis import assign_states_to_mdfs

def check_confidence_intervals(metadata_DFs, quantiles):
    """Checking the confidence intervals of the quantiles to ensure they don't overlap."""
    #collecting all severities, converting all to positive values, flattening as we go
    all_severities = []
    for df in metadata_DFs:
        all_severities.extend(df["AcneSeverity"] * -1)
    boostrapped_CIs = bootstrap_CI_margin_of_error(all_severities, quantiles)
    return boostrapped_CIs

def plot_histograms(first_order_probabilities):
    """This function plots a striped heatmap to inspect the distributions of normalized acne severity states for each treatment history.
    It uses a viridis heatmap implementation from my other repository figuresAndViewers.
    In lieu of using the actual treatment histories themselves as x labels, the x label is the index of the history in the sequence.
    """

    x_dim = 100
    y_dim = 200

    fig = plt.figure(figsize=(x_dim, y_dim))
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)

    # plotting histograms of each context dependent model of acne treatment severity
    bar_width = 0.3
    spacing = 0.05

    # sorting the distributions by the state name in reverse

    real_first_order_probabilities = dict(sorted(first_order_probabilities.items(), reverse=True))

    mainPanelHeight = 15
    mainPanelWidth = 20

    legendPanelHeight = .25
    legendPanelWidth = .5

    sidePanelHeight = 3
    sidePanelWidth = .25

    # setting up the panels and placing the proper positions
    firstMainPanel = plt.axes([.05 / x_dim, .375 / y_dim, mainPanelWidth / x_dim, mainPanelHeight /
                               y_dim])
    firstMainPanel.set_xlabel("Treatment History Index")
    firstMainPanel.set_title("Raw Distributions of Normalized Acne Severity States")

    # setting up the legend panel
    legendRight = plt.axes(
        [(1 + mainPanelWidth) / x_dim, legendPanelHeight / y_dim, legendPanelWidth / x_dim, mainPanelHeight / y_dim])
    # seting ticks of legend
    legendRight.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True, right=False, labelright=False,
                            top=False, labeltop=False)

    legendRight.set_xlim(0, .1)
    legendRight.set_ylim(0, 20)
    legendRight.set_yticks([0, 20], ['0', '1'])

    # looping through to construct a heatmap for all distributions
    entries = len(real_first_order_probabilities)
    bar_width = 1 / entries
    firstMainPanel.set_xlim(0, 1)
    firstMainPanel.set_ylim(0, 1)

    x_pos = 0
    for history, raw_distribution in real_first_order_probabilities.items():
        distribution = defaultdict(float, raw_distribution)
        scaled_x_pos = x_pos * bar_width

        high_fc = viridis(distribution["High Severity"])[:3]
        med_fc = viridis(distribution["Medium Severity"])[:3]
        low_fc = viridis(distribution["Low Severity"])[:3]

        firstMainPanel.add_patch(
            rect([scaled_x_pos, 2 / 3], width=bar_width, height=1 / 3, facecolor=high_fc, edgecolor='black',
                 linewidth=0.25))
        firstMainPanel.add_patch(
            rect([scaled_x_pos, 1 / 3], width=bar_width, height=1 / 3, facecolor=med_fc, edgecolor='black',
                 linewidth=0.25))
        firstMainPanel.add_patch(
            rect([scaled_x_pos, 0], width=bar_width, height=1 / 3, facecolor=low_fc, edgecolor='black', linewidth=0.25))

        x_actual_spot = scaled_x_pos + bar_width / 2

        x_pos += 1
    num_rects = len(real_first_order_probabilities)
    tick_positions = [i * bar_width + bar_width / 2 for i in range(num_rects)]

    firstMainPanel.set_xticks(tick_positions)
    firstMainPanel.set_xticklabels(['' for _ in tick_positions])
    firstMainPanel.set_yticks([1 / 6, 0.5, 5 / 6])
    firstMainPanel.set_yticklabels(["Low Severity", "Medium Severity", "High Severity"], fontsize=15)

    # plotting viridis heatmap in the sidebar
    # color map tuple pair linspaces, viridis values
    vvLin1Red = np.linspace(68 / 255, 59 / 255, 5)
    vvLin2Red = np.linspace(59 / 255, 33 / 255, 6)
    vvLin3Red = np.linspace(33 / 255, 94 / 255, 6)
    vvLin4Red = np.linspace(94 / 255, 253 / 255, 6)

    vvLin1Green = np.linspace(1 / 255, 82 / 255, 5)
    vvLin2Green = np.linspace(82 / 255, 145 / 255, 6)
    vvLin3Green = np.linspace(145 / 255, 201 / 255, 6)
    vvLin4Green = np.linspace(201 / 255, 231 / 255, 6)

    vvLin1Blue = np.linspace(84 / 255, 139 / 255, 5)
    vvLin2Blue = np.linspace(139 / 255, 140 / 255, 6)
    vvLin3Blue = np.linspace(140 / 255, 98 / 255, 6)
    vvLin4Blue = np.linspace(98 / 255, 37 / 255, 6)

    plLin4Red = np.linspace(245 / 255, 237 / 255, 5)
    plLin3Red = np.linspace(190 / 255, 245 / 255, 6)
    plLin2Red = np.linspace(87 / 255, 190 / 255, 6)
    plLin1Red = np.linspace(15 / 255, 87 / 255, 6)

    plLin4Green = np.linspace(135 / 255, 252 / 255, 5)
    plLin3Green = np.linspace(48 / 255, 135 / 255, 6)
    plLin2Green = np.linspace(0 / 255, 48 / 255, 6)
    plLin1Green = np.linspace(0 / 255, 0 / 255, 6)

    plLin4Blue = np.linspace(48 / 255, 27 / 255, 5)
    plLin3Blue = np.linspace(101 / 255, 48 / 255, 6)
    plLin2Blue = np.linspace(151 / 255, 101 / 255, 6)
    plLin1Blue = np.linspace(118 / 255, 151 / 255, 6)

    # total linspaces for all tuple pairs, viridis values
    vvListOfRedLins = list(vvLin1Red) + list(vvLin2Red) + list(vvLin3Red) + list(vvLin4Red)
    vvListOfGreenLins = list(vvLin1Green) + list(vvLin2Green) + list(vvLin3Green) + list(vvLin4Green)
    vvListOfBlueLins = list(vvLin1Blue) + list(vvLin2Blue) + list(vvLin3Blue) + list(vvLin4Blue)

    orderedVVRed = list(dict.fromkeys(vvListOfRedLins))
    orderedVVGreen = list(dict.fromkeys(vvListOfGreenLins))
    orderedVVBlue = list(dict.fromkeys(vvListOfBlueLins))

    # viridis heatmaps into the legend panel
    for index in range(0, 20, 1):
        colorPaletteVV = (orderedVVRed[index], orderedVVGreen[index], orderedVVBlue[index])
        vvGradeRect = rect([0, index], .1, 8, facecolor=colorPaletteVV, edgecolor='black', linewidth=0)
        legendRight.add_patch(vvGradeRect)

    plt.savefig("Raw_Striped_Heatmap")
    plt.close()

def find_dirichlet_marginal_cis(alphas, confidence_level=.95):
    """Function that is wrapped by the below function. Calculates the upper and lower quantiles supplied by confidence interval
    for each marginal beta distribution of a given dirichlet. Returns each confidence interval indexed by the respective alpha."""
    confidence_intervals = {}
    top_density = (1 - confidence_level) / 2
    bottom_density = 1 - top_density
    total_alpha = np.sum(alphas)

    for index, alpha in enumerate(alphas):
        other_alpha_sum = total_alpha - alpha
        lower_bound = beta.ppf(bottom_density, alpha, other_alpha_sum)
        upper_bound = beta.ppf(top_density, alpha, other_alpha_sum)
        confidence_intervals[index] = (lower_bound, upper_bound)

    return confidence_intervals

def plot_Dirichlets_credible_interals(histories_and_dirichlets, ordered_categories, confidence_level=.95):
    """This function finds the 95% credible intervals for each component of the Dirichlet distribution for all treatment histories.
    It ensures that a stacked plot is made."""

    # checking to see if no categories are supplied; just uses indices of each state to name categories in that case
    if ordered_categories is None:
        ordered_categories = [f"State {i}" for i in range(len(histories_and_dirichlets[0]))]

    fig, ax = plt.subplots(len(ordered_categories), 1, figsize=(len(ordered_categories) * 3.5, 7), sharex=True)
    ax[-1].set_xlabel("History Index")
    ax[len(ax) - 1].set_title("95% Credible Intervals for marginal acne severity state distributions")
    jitter_spacing = .5
    left_end = 0
    for history, alphas in histories_and_dirichlets.items():

        these_confidence_intervals = find_dirichlet_marginal_cis(alphas)
        for subplot_index, ordered_confidence_interval in these_confidence_intervals.items():
            x = left_end + (subplot_index - len(
                these_confidence_intervals) / 2) * jitter_spacing  # adding some x offset for the error bars
            center = (ordered_confidence_interval[1] + ordered_confidence_interval[0]) / 2
            width = ordered_confidence_interval[0] - ordered_confidence_interval[1]

            ax[len(ordered_categories) - subplot_index - 1].errorbar(x, center, yerr=width / 2, fmt='o', color='C0',
                                                                     capsize=5)

        left_end += 1

    for i, name in enumerate(reversed(ordered_categories)):
        ax[i].set_ylabel(name)

    plt.tight_layout()
    plt.show()
    plt.close()

def calculate_KL_Divergence_vectorized(histories_and_posteriors):
    """This vectorized function calculates the Kullback-Leibler divergence between adjacent
    3 dimensional Dirichlet distributions, each of which is indexed by its alpha parameter.
    It does this for a full array, computing KL Divergence for each term alpha[i] and alpha[i+1]. So, it requires
    you to supply two arrays of parameters, but really, just the same one read forwards and backwards.
    """
    history_labels = list(histories_and_posteriors.keys())
    x_vals = np.arange(1, len(history_labels))

    alphas = np.array(list(histories_and_posteriors.values()))
    alphas_backward = alphas[:-1]
    alphas_forward = alphas[1:]

    #ensuring alphas are non-0 up front
    sum_forward = np.sum(alphas_forward, axis=1)
    sum_backward = np.sum(alphas_backward, axis=1)

    first_term = gammaln(sum_forward) - gammaln(sum_backward)
    second_term = np.sum(gammaln(alphas_backward) - gammaln(alphas_forward), axis=1)
    third_term = np.sum(
        (alphas_forward - alphas_backward) *
        (psi(alphas_forward) - psi(sum_forward)[:, None]),
        axis=1
    )

    kl_div = first_term + second_term + third_term
    cumulative_kl = np.cumsum(kl_div)
    cumulative_kl = np.clip(cumulative_kl, 1e-10, None)  # Avoid log(0)

    return x_vals, np.log(cumulative_kl), cumulative_kl

def split_for_piecewise_regression(xs, cumulative_kls, percentile_cutoff=80, split_index=None, gaussian_sigma=1,
                                   slope_threshold=0.2,
                                   min_segment_length=3):
    """This algorithm uses a brute force method to find the inflection points to fit piecewise regression models to the data.
    But first...
    It dynamically chooses the right number of inflection points to fit each model to via the following...
    1) It smoothes the cumulative KL divergence array with a Gaussian filter (by default is standard normal).
    2) It computes pairwise slopes between adjacent cumulative KL values (with each x interval = 1/number of steps in treatment history)
    ^check that later
    3) Computes differences between adjacent slopes, returning a 2nd derivative approximation for the entire array
    4) (Currently commented out) Plots the distribution of the magnitudes for inspection.
    5) A cutoff for inflection points is chosen as a percentile of the 2nd derivative magntitdes (default is 90th).

    Then, it iterates through the found inflection points, dividing them into subarrays of consecutive points. Once it does this, it does
    one final check to see if each subarray should be further divided. It does this by iterating through each array, comparing the difference in
    adjacent slopes pairwise, checking their difference against the provided slope threshold parameter. It then makes a list of lists of
    inflection points where the regression models should start and end.

    It then uses the result of this to further divide the points into subarrays that a regression model can be fit to.
    At last, it checks the length of each subarray against the minimum segment length, and discards ones that are too small.

    """

    smoothed_kls = gaussian_filter1d(cumulative_kls, sigma=gaussian_sigma)
    slopes = np.diff(smoothed_kls)
    second_derivatives_magnitudes = np.abs(np.diff(slopes))
    threshold = np.percentile(second_derivatives_magnitudes, percentile_cutoff)
    inflection_points = np.where(second_derivatives_magnitudes > threshold)[0] + 1

    differences = np.diff(inflection_points)
    non_consecutive_indices = np.where(differences != 1)[0] + 1
    separated_inflection_points = np.array_split(inflection_points, non_consecutive_indices)

    # adding the end of the array to the non_consecutive indices for easier splitting later
    # separated_inflection_points[-1] = np.append(separated_inflection_points[len(separated_inflection_points)-1], len(smoothed_kls)-1)
    # ^above is causing issues
    all_breaks = []

    for consecutives_array in separated_inflection_points:
        index_consecutives_array = 0
        last_inflection_pt = 0
        last_slope = 0

        consecutives_breaks = []

        for an_inflection_pt in consecutives_array:
            if an_inflection_pt >= len(slopes):
                continue
            slope_current = slopes[an_inflection_pt]

            slope_difference = np.abs(slope_current - last_slope)

            if slope_difference > slope_threshold:
                consecutives_breaks.append(an_inflection_pt)

            last_slope = slope_current
            last_inflection_pt = an_inflection_pt

        all_breaks.append((consecutives_breaks, index_consecutives_array))

        index_consecutives_array += 1

    fixed_consecutives = []
    for breaks, index in all_breaks:

        if len(breaks) != 0:
            consecutives_array_to_split = separated_inflection_points[index]
            last_breaking_index = breaks[0] - 1
            for breaking_index in breaks:
                broken_consecutives_array = consecutives_array_to_split[last_breaking_index: breaking_index + 1]
                fixed_consecutives.append(broken_consecutives_array)
                last_breaking_index = breaking_index

    # clipping the appropriate array in the inflection point array of arrays
    clipped_inflection_points = []
    last_consecutive_break = None

    # Find the last break if it exists
    if fixed_consecutives:
        last_consecutive_break = fixed_consecutives[-1][-1]

    # Search for that break in the separated inflection points
    where_clipping = None
    for i, again_consecutives_array in enumerate(separated_inflection_points):
        if last_consecutive_break in again_consecutives_array:
            idx = np.where(again_consecutives_array == last_consecutive_break)[0][0]
            clipped_inflection_points.append(again_consecutives_array[idx + 1:])
            where_clipping = i
            break  # stop after finding the first match

    # Combine final splits
    if where_clipping is not None:
        full_splits = fixed_consecutives + clipped_inflection_points + separated_inflection_points[where_clipping + 1:]
    else:
        full_splits = fixed_consecutives
    return full_splits

def plot_piecewise_regression_segments(curve, splits, ax, all_treatment_histories, color="blue", linewidth=2):
    """This function is wrapped by piecewise_regression_and_plot below. It plots the actual line segments piecewise
    over a rendered cumulative KL Divergence plot. """

    results_pvals_r2_betas = {}
    all_results = []
    # for now, just taking the longest treatment history and using that for indexing
    full_list_of_histories_all_patients = []
    for treatment_history in all_treatment_histories:
        for history in treatment_history:
            full_list_of_histories_all_patients.append(tuple(history))

    unique_list_of_histories = list(set(full_list_of_histories_all_patients))
    unique_list_of_histories.sort()

    if splits is None:
        print("No splits to plot!")
        return

    # removing any splits of size 7 (too small to be fitted)
    removed_splits = [split for split in splits if len(split) > 10]

    # setting up subplots
    top = len(removed_splits) // 2
    bottom = top + (len(removed_splits) % 2)
    fig, regression_axes = plt.subplots(2, max(top, bottom), figsize=(20, 20))
    fig.suptitle("Segments of Non-Diminishing KL Divergence")

    first_plot_index = 0
    all_indices_array = np.arange(0, len(removed_splits))

    for i, other_splits in enumerate(removed_splits):
        fixed_x_ticks = []

        other_split_start, other_split_end = other_splits[0], other_splits[len(other_splits) - 1]
        these_x_ticks_unfixed = unique_list_of_histories[other_split_start: other_split_end + 1]

        for other_i in range(len(other_splits)):
            joined_history_piece = " ".join(str(item) + " " for item in these_x_ticks_unfixed[other_i])

            fixed_x_ticks.append(joined_history_piece)

        y_segment = curve[other_split_start: other_split_end + 1]
        x_segment = np.arange(other_split_start, other_split_end + 1).reshape(-1, 1)
        regression_piece = LinearRegression().fit(x_segment, y_segment.reshape(-1, 1))

        y_pred = regression_piece.predict(x_segment)
        axis_to_plot_on = regression_axes.flatten()[i]
        axis_to_plot_on.scatter(x_segment, y_segment)
        axis_to_plot_on.plot(x_segment.flatten(), y_pred.flatten(), color=color, linewidth=linewidth)
        axis_to_plot_on.set_ylabel("KL Divergence (log scale)")
        axis_to_plot_on.set_xlabel("Treatment History")

        axis_to_plot_on.set_xticks(x_segment.flatten())

        axis_to_plot_on.set_xticklabels(fixed_x_ticks, fontsize=7, rotation=45)

        # also returning the report for each segement (including p values for slope) using statsmodels package
        x = sm.add_constant(x_segment)
        this_model = sm.OLS(y_segment.reshape(-1, 1), x).fit()
        slope = this_model.params[1]
        pval = this_model.pvalues[1]
        r2 = this_model.rsquared

        results_pvals_r2_betas[i] = {
            "Segment": i,
            "Start": (other_split_start, unique_list_of_histories[other_split_start]),
            "End": (other_split_end, unique_list_of_histories[other_split_end]),
            "Num Points": len(y_segment),
            "Slope": float(slope),
            "p-value": float(pval),
            "R²": r2}

        all_results.append({
            "Segment": i,
            "Start": other_split_start, "Starting History": unique_list_of_histories[other_split_start],
            "End": other_split_end, "Ending History": unique_list_of_histories[other_split_end],
            "Num Points": len(y_segment),
            "Slope": slope.round(4),
            "p-value": pval,
            "R²": r2.round(4)})

    results_df = pd.DataFrame(all_results)
    results_df["p-value"] = results_df["p-value"].apply(lambda p: f"{p:.3e}" if p < 0.001 else f"{p:.3f}")
    results_df.to_csv("segment_regression_results.csv", index=False)

def bootstrap_KLD_spline_and_dKL(already_fitted_spline, deriv_ax, x, observed_KLDs, dy_smooth, x_smooth, n_boot=300):
    """Another bootstrap algorithm that computes the 95% CIs of the cumulative KL fitted spline and the derivative of it,
    based on the residuals of each of n_boot (parameter above) points of the already fitted spline.
    The input x refers to the x values used to fit the spline, the input observed_KLDs refers to the real ones.
    Unused - dy_smooth is the smoothed derivative of the already fitted spline"""

    n_boot = 300
    y_boot_preds = []
    dy_boot_preds = []

    for i in range(n_boot):
        # resample residuals
        y_fit = already_fitted_spline(x)
        residuals = observed_KLDs - y_fit
        resampled_y = y_fit + np.random.choice(residuals, size=len(observed_KLDs), replace=True)

        boot_spline = UnivariateSpline(x, resampled_y, s=len(x) * np.var(observed_KLDs) * 0.002)
        y_boot_preds.append(boot_spline(x))
        dy_boot_preds.append(boot_spline.derivative()(x))

    y_boot_preds = np.array(y_boot_preds)
    dy_boot_preds = np.array(dy_boot_preds)

    # Step 1: compute mean derivative across bootstraps
    dy_mean = np.mean(dy_boot_preds, axis=0)

    # Step 2: compute deviations from the mean
    dy_dev = dy_boot_preds - dy_mean

    # Step 3: compute percentile-based CI of deviations, then recenter around dy_mean
    dy_lower = dy_mean + np.percentile(dy_dev, 2.5, axis=0)
    dy_upper = dy_mean + np.percentile(dy_dev, 97.5, axis=0)

    # Step 4: plot
    # fig, ax = plt.subplots(figsize=(8,5))
    deriv_ax.plot(x_smooth, dy_smooth, 'r--', label='Derivative (original fit)')
    deriv_ax.fill_between(x, dy_lower, dy_upper, color='red', alpha=0.3, label='95% CI')
    deriv_ax.set_xlabel('Treatment index')
    deriv_ax.set_ylabel('d(KL)/d(treatment)')
    deriv_ax.legend()
    plt.show()

    return dy_boot_preds, y_boot_preds

def determine_dy_intervals_subject_to_threshold(deriv_boot_preds, p_val_threshold=.95):
    """Function to determine if fraction of bootstrap predictions of the derivative of the cumulative KLD curve
    exceeds a threshold for a the amount of slopes that should be so."""
    deriv_boot_preds = np.array(deriv_boot_preds)  # (n_boot, n_points)

    found_indices = []
    for i in range(deriv_boot_preds.shape[1]):
        fraction_positive = np.mean(deriv_boot_preds[:, i] > 0)
        fraction_negative = np.mean(deriv_boot_preds[:, i] < 0)
        if fraction_positive >= p_val_threshold or fraction_negative >= p_val_threshold:
            found_indices.append(i)

    dy_splits = split_at_non_consecutives(found_indices)
    return dy_splits

def split_at_non_consecutives(arr):
    """Simple function to split an array into separate arrays of nonconsecutive points."""
    if not arr:
        return []

    result = []
    current_sub_array = [arr[0]]

    for i in range(1, len(arr)):
        # Define "consecutive" - here, arr[i] is one greater than arr[i-1]
        if arr[i] == arr[i - 1] + 1:
            current_sub_array.append(arr[i])
        else:
            result.append(current_sub_array)
            current_sub_array = [arr[i]]

    result.append(current_sub_array)  # Add the last sub-array
    return result

def make_KL_Divergence_plot(x_vals, kl_divergences, cumulative_kl, splits, all_treatment_histories):
    """This function plots the Kullback-Leibler Divergence between each consecutive Dirichlet posterior distribution
    as a line plot. This is a wrapper of calculate_KL_Divergence_vectorized.
    It also plots the cumulative KL Divergence over the KL divergence between individual distributions.
    It ends up plotting the linear regression models over the plot as well. """

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(x_vals[0:], kl_divergences[0:], marker='o', label='KL Divergence')

    # approximating curve smoothly with spline
    # sorting if needed
    sorting_indices = np.argsort(x_vals)
    x_to_plot = x_vals[sorting_indices]
    cumulative_kls_to_plot = cumulative_kl[sorting_indices]

    # fitting smoothed spline, smoothing by a factor of the variance and the number of samples
    factor_for_smoothing = 0.002
    smoothing = len(x_to_plot) * np.var(cumulative_kls_to_plot) * factor_for_smoothing
    fitted_spline = UnivariateSpline(x_to_plot, cumulative_kls_to_plot, s=smoothing)

    # evaluating the fitted smoothed spline and its derivative
    x_smooth = np.linspace(x_to_plot.min(), x_to_plot.max(), 500)
    y_smooth = fitted_spline(x_smooth)
    dy_smooth = fitted_spline.derivative()(x_smooth)

    ax1.plot(x_smooth, y_smooth, 'o', label='Observed cumulative KL', alpha=0.6)
    ax1.plot(x_smooth, y_smooth, 'r-', label='Smoothing spline')
    ax1.set_xlabel("Treatment index")
    ax1.set_ylabel("Cumulative KL divergence")
    ax1.legend(loc='upper left')

    # Secondary axis for derivative
    ax2 = ax1.twinx()
    ax2.plot(x_smooth, dy_smooth, 'r--')
    ax2.set_ylabel("d(KL)/d(treatment)")
    ax2.legend(loc='upper right')

    ax1.set_title('KL Divergence and Cumulative Information Gain (Log Scale)')
    ax1.set_xlabel('Treatment History Index')
    ax1.set_ylabel('KL Divergence (log scale)')
    ax1.legend()

    # plotting the confidence intervals of the spline and its derivative using the bootstrap function above
    these_dy_boot_preds, these_y_boot_preds = bootstrap_KLD_spline_and_dKL(fitted_spline, ax2, x_to_plot,
                                                                           cumulative_kls_to_plot, dy_smooth, x_smooth)

    plt.tight_layout()
    plt.show()
    plt.close()

    these_determined_splits = determine_dy_intervals_subject_to_threshold(these_dy_boot_preds)
    linear_models = plot_piecewise_regression_segments(cumulative_kls_to_plot, these_determined_splits, ax1,
                                                       all_treatment_histories, linewidth=1)

    # mental note: just need to collect the p values for the fit segments, Dubin Watson statistic for autocorrelation, maybe a bootstrap CI
    # if not just use the confidence interval spit out by the linear model fitting module above (use chatgpt thread)

    return x_vals, np.log(cumulative_kl)

def validate_state_assignments_with_bootstrapping_result(metadata_DFs, state_names, orig_state_ranges,
                                                         boot_middle_qs_dict):
    """Function that validates if state assignments are valid in the context of hard binning (in particular around
    the saddle point)."""
    # Helper: extract the original labels once
    # Original state assignments
    orig_mdfs = copy.deepcopy(metadata_DFs)
    orig_mdfs = assign_states_to_mdfs(orig_mdfs, state_names, orig_state_ranges)
    orig_labels = np.concatenate([df["State"].values for df in orig_mdfs])

    fracs_changed = {}

    # Iterate over numeric middle cutpoints (q̂) in the bootstrap dict
    for qhat, boot_info in boot_middle_qs_dict.items():
        frac_changed = []

        # Iterate over bootstrap samples of the middle cutpoint
        for q in boot_info['bootstrap_samples']:
            # Ensure numeric
            q = float(q)

            # Update the middle cutpoint only
            low, high = float(orig_state_ranges[0]), float(orig_state_ranges[2])
            middle = float(q)
            epsilon = 1e-6
            middle = max(low + epsilon, min(middle, high - epsilon))
            new_ranges = [low, middle, high]

            # Reassign states with the adjusted middle cut
            mdfs_try = copy.deepcopy(metadata_DFs)
            mdfs_try = assign_states_to_mdfs(mdfs_try, state_names, new_ranges)
            new_labels = np.concatenate([df["State"].values for df in mdfs_try])

            # Fraction of labels that changed
            diff_frac = np.mean(new_labels != orig_labels)
            frac_changed.append(diff_frac)

        frac_changed = np.array(frac_changed)

        print(f"For middle cutpoint q̂={qhat:.4f}")
        print(f"Median fraction changed: {np.median(frac_changed):.3f}")
        print(
            f"5th–95th percentile range: ({np.percentile(frac_changed, 5):.3f}, {np.percentile(frac_changed, 95):.3f})\n")

        fracs_changed[qhat] = frac_changed

    return fracs_changed

def view_KDE(metadata_DFs):
    """Function to view the KDE of the data by itself."""
    """Determines the quantile cutoff determining the quantiles corresponding to categorical acne severity states (low, medium, and high), by
        1) Computing the KDE of the distribution of all normalized acne severity scores over all patients and treatment histories.
        2) Using optimization to find the saddle point of the distribution and consolidating with the modes."""
    fig, axes = plt.subplots(figsize=(5, 5))
    # collecting all severities, converting all to positive values, flattening as we go
    all_severities = []
    for df in metadata_DFs:
        all_severities.extend(df["AcneSeverity"] * -1)

    # check for normal character by plotting histogram
    #severities_histo = np.histogram(all_severities, density=True)
    axes.hist(all_severities, bins  = 30, density = True, alpha = .25)
    axes.set_xlabel("Normalized Severity Score")
    axes.set_ylabel("Density")
    axes.set_title("KDE of Acne Severity Distribution")
    axes.legend()

    # fitting a kernel density estimate to the data
    sns.kdeplot(all_severities, fill=True, ax=axes)

    # extracting the equation of the pdf and finding the local minimum in between the two modes
    kde_pdf = sp.stats.gaussian_kde(all_severities)

    # using max and min of pdf to find saddle point in between 2 modes, sampling 1000 points
    severity_grid = np.linspace(np.min(all_severities), np.max(all_severities), 1000)
    neg_kde = lambda x: -kde_pdf(x.reshape(1, -1))
    # finding the two main modes using optimization, with first 2 mode guesses at the .2 and .8 quantiles
    guesses = np.percentile(all_severities, [20, 80])

    modes = []
    for guess in guesses:
        better = optimize.minimize(neg_kde, np.array([guess]))
        modes.append(better.x[0])
    modes = np.array(modes)

    # finding the saddle point in between the two modes, using that as cutoff for the two patient states
    initial_guess = np.mean(modes)  # average of the modes
    bds = [(min(modes) + 1, max(modes) - 1)]  # This is a list of two tuples for each mode

    saddle_pt = optimize.minimize(kde_pdf, [initial_guess], bounds=bds)
    state_ranges = [modes[0], saddle_pt.x[0], modes[1]]
    state_names = ["High Severity", "Medium Severity", "Low Severity"]

    # plotting modes and saddle point over the distribution
    axes.scatter(modes[0], kde_pdf(modes[0]), color="red", label="Lower Mode")
    axes.scatter(modes[1], kde_pdf(modes[1]), color="red", label="Upper Mode")
    axes.scatter([saddle_pt.x[0]], kde_pdf([saddle_pt.x[0]]), color='green', label="Saddle Point")

    fig.show()
    plt.close(fig)


def data_visualization(these_assigned_md_DFs, this_treatment_history, this_md_DFs, these_ranges):
    """Function that actually plots all relevant plots for the observed data."""
    these_checked_CIs = check_confidence_intervals(this_md_DFs, these_ranges)  # returns a dictionary now
    built_histograms, raw_probabilities = build_histograms(these_assigned_md_DFs)
    this_KDE = view_KDE(these_assigned_md_DFs)
    #plotted_histogram = plot_histograms(raw_probabilities)

    uninformative_prior = [1, 1, 1]  # with a1 corresponding to low severity, a2 corresponding to medium, and a3 corresponding to high

    these_Dirichlets, these_categories = build_Dirichlet(uninformative_prior, built_histograms)

    dirichlet_credible_intervals = plot_Dirichlets_credible_interals(these_Dirichlets, these_categories)

    these_xs, this_log_cum_kls, this_kls = calculate_KL_Divergence_vectorized(these_Dirichlets)
    splits = split_for_piecewise_regression(these_xs, this_log_cum_kls)
    final_plot = make_KL_Divergence_plot(these_xs, this_log_cum_kls, this_kls, splits, this_treatment_history)

    return these_Dirichlets, these_categories

def bootstrap_CI_margin_of_error(actual_values, observed_quantile_values, B=5000, bandwidth=0.05):
    """Short function to approximate the margin of error (MOE) for the bimodal KDE presented here. Uses Bootstrapping to approximate
    the 95% confidence interval for each quantile, and halves each of them to give MOE."""

    data = np.asarray(actual_values)
    n = len(data)
    kde = KernelDensity(bandwidth=bandwidth).fit(data[:, None])
    rng = np.random.default_rng(123)

    results = {}

    for qhat in observed_quantile_values:
        boot_q = np.empty(B)
        # compute percentile of qhat in original data
        p = (data < qhat).mean()
        for b in range(B):
            smpl = kde.sample(n_samples=n, random_state=123 + b).ravel()
            boot_q[b] = np.quantile(smpl, p)  # same percentile in resampled data

        ci_lower, ci_upper = np.percentile(boot_q, [2.5, 97.5])
        moe = 0.5 * (ci_upper - ci_lower)

        results[qhat] = {
            'CI': (ci_lower, ci_upper),
            'MOE': moe,
            'bootstrap_samples': boot_q
        }

        # plot histogram
        plt.figure()
        plt.hist(boot_q, bins=80)
        plt.axvline(qhat, color='k', linestyle='--', label='observed cutpoint')
        plt.title(f'Smoothed bootstrap distribution for cutpoint {qhat:.2f}')
        plt.legend()
        plt.show()
        plt.close()

        print(f"Cutpoint={qhat:.4f}, 95% CI=({ci_lower:.4f},{ci_upper:.4f}), MOE={moe:.4f}")

    return results

