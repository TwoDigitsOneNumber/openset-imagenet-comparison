"""Training of all models for the paper"""

import argparse
import multiprocessing
import collections
import subprocess
import pathlib
from pathlib import Path
import openset_imagenet
import os, sys
import torch
import numpy
from loguru import logger

from collections import defaultdict
import numpy as np
import pandas as pd

import yaml

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm, colors
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, LogLocator
from matplotlib.lines import Line2D

# from openset_imagenet.util import STYLES, COLORS, normalize_array

clrs = cm.tab10(range(10))

COLORS = {
    # benchmarks
    "softmax": clrs[0],
    "entropic": clrs[0],
    "objectosphere": clrs[0],
    "garbage": clrs[0],
    # face losses (HFN)
    "sphereface": clrs[3],
    "cosface": clrs[1],
    "arcface": clrs[4],
    # face losses (SFN)
    'norm_sfn': clrs[1],
    'cosface_sfn': clrs[2],
    'arcface_sfn': clrs[3],
    # margin-OS (SFN)
    'softmax_os': clrs[1],
    'cos_os': clrs[2],
    'arc_os': clrs[3],
    # margin-eos (HFN)
    "norm_eos": clrs[1],
    "cos_eos": clrs[2],
    "arc_eos": clrs[3],
    # early tries
    "cos_eos_sfn": clrs[3],
    "arc_eos_sfn": clrs[2],
    "cos_os_non_symmetric": clrs[6],
    "arc_os_non_symmetric": clrs[7],
    "sm_softmax": clrs[0]
}

STYLES = {
    # benchmarks
    "softmax": "solid",
    "entropic": "solid",
    "objectosphere": "solid",
    "garbage": "solid",
    # face losses (HFN)
    "sphereface": "solid",
    "cosface": "solid",
    "arcface": "solid",
    # face losses (SFN)
    'norm_sfn': "solid",
    'cosface_sfn': "solid",
    'arcface_sfn': "solid",
    # margin-OS (SFN)
    'softmax_os': 'solid',
    'cos_os': 'solid',
    'arc_os': 'solid',
    # margin-eos (HFN)
    "norm_eos": "solid",
    "cos_eos": "solid",
    "arc_eos": "solid",
    # margin-eos (SFN)
    "cos_eos_sfn": "solid",
    "arc_eos_sfn": "solid",
    # early tries
    "cos_os_non_symmetric": "solid",
    "arc_os_non_symmetric": "solid",
    "sm_softmax": "solid",
    # protocols
    "p1": "dashed",
    "p2": "dotted",
    "p3": "solid",
    "p0": "dashdot"
}

NAMES = {
    # benchmarks
    "softmax": "Softmax",
    "entropic": "EOS",
    "objectosphere": "Objectosphere",
    "garbage": "Garbage",
    # face losses (HFN)
    "sphereface": "SphereFace",
    "cosface": "CosFace",
    "arcface": "ArcFace",
    # face losses (SFN)
    'norm_sfn': "SFN-Norm",
    'cosface_sfn': "SFN-CosFace",
    'arcface_sfn': "SFN-ArcFace",
    # margin-OS (SFN)
    'softmax_os': 'Norm-OS',
    'cos_os': 'Cos-OS',
    'arc_os': 'Arc-OS',
    # margin-eos (HFN)
    "norm_eos": "Norm-EOS",
    "cos_eos": "Cos-EOS",
    "arc_eos": "Arc-EOS",
    # early tries
    "cos_eos_sfn": "Cos-EOS (SFN)",
    "arc_eos_sfn": "Arc-EOS (SFN)",
    "cos_os_non_symmetric": "Cos-OS (non-sym.)",
    "arc_os_non_symmetric": "Arc-OS (non-sym.)",
    "sm_softmax": "SM-Softmax",
    # algorithms
    "threshold": "Threshold",
    "openmax": "OpenMax",
    "proser": "PROSER",
    "evm": "EVM",
    "maxlogits": "MaxLogits",
    # other
    "p1": "P_1",
    "p2": "P_2",
    "p3": "P_3",
    "p0": "P_0",
    1: "$P_1$",
    2: "$P_2$",
    3: "$P_3$",
    0: "$P_0$"
}



def command_line_options(command_line_arguments=None):
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Plotting", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--protocols", "-p",
        type=int,
        choices = (1,2,3,0,10),
        nargs="+",
        default = (1,2,3,0,10),
        help="Select the protocols that should be evaluated. Set 0 for toy data."
    )
    parser.add_argument(
        "--losses", "-l",
        nargs = "+",
		choices = (
			'softmax', 'entropic', 'garbage', 'objectosphere',  	# benchmarks
			'sphereface', 'cosface', 'arcface',  					# face losses (HFN)
			'norm_sfn', 'cosface_sfn', 'arcface_sfn',  				# face losses (SFN)
			'softmax_os', 'cos_os', 'arc_os',  						# margin-OS (SFN)
			'norm_eos', 'cos_eos', 'arc_eos',  						# margin-EOS (HFN)
			'cos_eos_sfn', 'arc_eos_sfn',     						# margin-EOS (SFN)
            'cos_os_non_symmetric', 'arc_os_non_symmetric', 'sm_softmax'
		),
        default = (
			'softmax', 'entropic', 'objectosphere',               	# benchmarks
			'sphereface', 'cosface', 'arcface',  					# face losses (HFN)
			'norm_sfn', 'cosface_sfn', 'arcface_sfn',  				# face losses (SFN)
			'softmax_os', 'cos_os', 'arc_os',  						# margin-OS (SFN)
			'norm_eos', 'cos_eos', 'arc_eos',  						# margin-EOS (HFN)
			'cos_eos_sfn', 'arc_eos_sfn',     						# margin-EOS (SFN)
            'cos_os_non_symmetric', 'arc_os_non_symmetric', 'sm_softmax'
        ),
        help = "Select the loss functions that should be included into the plot"
    )
    parser.add_argument(
        "--algorithms", "-a",
        choices = ["threshold", "openmax", "evm", "proser", "maxlogits"],
        nargs = "+",
        default = ["threshold", "maxlogits"],
        help = "Which algorithm to include into the plot. Specific parameters should be in the yaml file"
    )
    parser.add_argument(
        "--configuration", "-c",
        type = pathlib.Path,
        default = pathlib.Path("config/test.yaml"),
        help = "The configuration file that defines some basic information"
    )
    parser.add_argument(
        "--use-best",
        action = "store_true",
        help = "If selected, the best model is selected from the validation set. Otherwise, the last model is used"
    )
    parser.add_argument(
        "--force", "-f",
        action = "store_true",
        help = "If set, score files will be recomputed even if they already exist"
    )
    parser.add_argument(
        "--plots",
        help = "Select where to write the plots into"
    )
    parser.add_argument(
        "--tables",
        default = "results/Results_Protocol{}_{}.tex",
        help = "Select the files where to write the CCR table into"
    )
    parser.add_argument(
        "--fpr-thresholds", "-t",
        type = float,
        nargs="+",
        default = [1e-3, 1e-2, 1e-1, 1.],
        help = "Select the thresholds for which the CCR validation metric should be tabled"
    )

    args = parser.parse_args(command_line_arguments)

    args.plots = args.plots or f"results/Results_{'best' if args.use_best else 'last'}.pdf"
#    args.table = args.table or f"results/Results_{suffix}.tex"
    return args


def load_training_scores(args, cfg):
    # we sort them as follows: protocol, loss, algorithm
    training_scores = defaultdict(lambda: defaultdict(dict))
    for protocol in args.protocols:
        for loss in args.losses:
            output_directory = pathlib.Path(cfg.output_directory) / f"Protocol_{protocol}"
            file_path = Path(output_directory) / f"{loss}_train_arr.npz"
            # load data
            if os.path.exists(file_path):
                data = numpy.load(file_path)
                for key in data.keys():  # keys are the data description, e.g., train_loss, val_conf_unk
                    training_scores[protocol][loss][key] = data[key]
            else:
                logger.warning(f"Did not find file {file_path}")

    return training_scores



def load_scores(args, cfg):
    # collect all result files;
    suffix = "best" if args.use_best else "curr"
    # we sort them as follows: protocol, loss, algorithm
    scores = defaultdict(lambda: defaultdict(dict))
    features = defaultdict(lambda: defaultdict(dict))
    logits = defaultdict(lambda: defaultdict(dict))
    angles = defaultdict(lambda: defaultdict(dict))
    ground_truths = {}

    # always load maxlogits
    algs = args.algorithms.copy()
    # if 'maxlogits' not in algs:
    #     algs.append('maxlogits')

#    epoch = {p:{} for p in args.protocols}
    for protocol in args.protocols:
        for loss in args.losses:
            for algorithm in algs:
                output_directory = pathlib.Path(cfg.output_directory) / f"Protocol_{protocol}"
                alg = "threshold" if algorithm == "maxlogits" else algorithm
                scr = "logits" if algorithm == "maxlogits" else "scores"
                score_file = output_directory / f"{loss}_{alg}_test_arr_{suffix}.npz"

                if os.path.exists(score_file):
                    # remember files
                    results = numpy.load(score_file)

                    scores[protocol][loss][algorithm] = results[scr]
                    logits[protocol][loss][algorithm] = results['logits']
                    features[protocol][loss][algorithm] = results['features']
                    angles[protocol][loss][algorithm] = results['angles']

                    if protocol not in ground_truths:
                        ground_truths[protocol] = results["gt"].astype(int)
                    else:
                        assert numpy.all(results["gt"] == ground_truths[protocol])

                    logger.info(f"Loaded score file {score_file} for protocol {protocol}, {loss}, {algorithm}")
                else:
                    logger.warning(f"Did not find score file {score_file} for protocol {protocol}, {loss}, {algorithm}")

    return scores, ground_truths, features, logits, angles

def normalize_array(array, axis, ord=2):
    """only works on axis = 1"""
    norms = np.linalg.norm(array, axis=axis, ord=ord)
    norms = norms.reshape((-1,1))  # reshape norms to same dim as array
    array = np.divide(array, norms)  # normalize
    return array


def plot_training_metrics(args, training_scores, pdf):
    """plot loss curves, confidences of knowns and negatives, and average confidence (all as function of the training epochs training)."""
    P = len(args.protocols)
    fig = pyplot.figure(figsize=(12,3*P))
    gs = fig.add_gridspec(P, 3, hspace=0.25, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=False)
    axs = axs.flat
    font_size = 15
    linewidth = 1.1

    colors = cm.tab10(range(10))
    lines = ['Training Loss', 'Validation Loss', 'Conf. Known', 'Conf. Unknown', 'Avg. Conf.']
    LINE_COLORS = {line:colors[i] for i, line in enumerate(lines)}

    # add lines to fig (index 0 for loss curves; index 1 for confidences)
    for index, protocol in enumerate(args.protocols):

        max_loss = 0

        for loss_function, loss_function_arrays in training_scores[protocol].items():
            epochs = loss_function_arrays['epochs']

            # find max loss value of all losses per protocol
            max_train_loss_curr = numpy.amax(loss_function_arrays['train_loss'])
            max_val_loss_curr = numpy.amax(loss_function_arrays['val_loss'])
            max_curr = max(max_train_loss_curr, max_val_loss_curr)
            if max_curr > max_loss:
                max_loss = max_curr

            # plot loss curves on the left hand side
            axs[2*index].plot(
                epochs, loss_function_arrays['train_loss'],
                linestyle=STYLES[loss_function], color=LINE_COLORS['Training Loss'], linewidth=linewidth)
            axs[2*index].plot(
                epochs, loss_function_arrays['val_loss'], 
                linestyle=STYLES[loss_function], color=LINE_COLORS['Validation Loss'], linewidth=linewidth)
            
            # plot confidence curves on middle plot
            axs[2*index+1].plot(
                epochs, loss_function_arrays['val_conf_kn'],
                linestyle=STYLES[loss_function], color=LINE_COLORS['Conf. Known'], linewidth=linewidth)
            axs[2*index+1].plot(
                epochs, loss_function_arrays['val_conf_unk'], 
                linestyle=STYLES[loss_function], color=LINE_COLORS['Conf. Unknown'], linewidth=linewidth)

            # plot avg confidence curves
            axs[2*index+2].plot(
                epochs, (loss_function_arrays['val_conf_unk']+loss_function_arrays['val_conf_kn'])*0.5, 
                linestyle=STYLES[loss_function], color=LINE_COLORS['Avg. Conf.'], linewidth=linewidth)

        # add titles
        axs[2*index].set_title("Loss Curves", fontsize=13)
        axs[2*index+1].set_title("Validation Confidence Curves", fontsize=13)
        axs[2*index+2].set_title("Avg. Validation Confidence", fontsize=13)

        # axis formating
        axs[2*index].set_ylim(0,min(40, max_loss*1.1))
        axs[2*index+1].set_ylim(0,1)
        axs[2*index+2].set_ylim(0,1)

    # fugure labels
    fig.text(0.5, -0.05, 'Epochs', ha='center', fontsize=font_size)
    fig.text(0.08, 0.5, 'Loss', va='center', rotation='vertical', fontsize=font_size)
    fig.text(0.92, 0.5, 'Confidence', va='center', rotation='vertical', fontsize=font_size)

    # add legend
    openset_imagenet.util.training_scores_legend(args.losses, lines, LINE_COLORS, fig,
        bbox_to_anchor=(0.5,-0.3), handletextpad=0.6, columnspacing=1.5,
        title="How to Read: Line Style -> Loss Function; Color -> What"
    )

    pdf.savefig(bbox_inches='tight', pad_inches = 0)


def plot_feature_magnitudes(args, features, ground_truths, pdf):
    """plot feature magnitudes grouped by known, negative and unknown."""
    P = len(args.protocols)
    algs = [a for a in args.algorithms if a != 'maxlogits']
    A = len(algs)
    # A = len(args.algorithms)
    font_size = 15
    linewidth = 1.1

    # Manual colors
    color_categories = {
        'knowns': colors.to_rgba('tab:blue', 1),
        'unknowns': colors.to_rgba('indianred', 1),
        'negatives': colors.to_rgba('tab:green', 1)

    }

    for loss in args.losses:

        fig = pyplot.figure(figsize=(6*A,3*P))
        gs = fig.add_gridspec(P, A, hspace=0.25, wspace=0.1)
        axs = gs.subplots(sharex=True, sharey=True)

        for index, protocol in enumerate(args.protocols):
            for a, algorithm in enumerate(args.algorithms):

                # compute feature magnitudes
                distributions = openset_imagenet.util.get_feature_magnitude_distribution(
                    features[protocol][loss][algorithm],
                    ground_truths[protocol]
                )

                if P==1 and A==1:
                    ax = axs
                else:
                    ax = axs[2*index+a]

                for category in ['knowns', 'unknowns', 'negatives']:
                    ax.plot(
                        distributions[category][1][1:], distributions[category][0],
                        color=color_categories[category], linewidth=linewidth
                    )
                    # if threshold and maxlogit evaluation, then add both names for clarity (but only one plot, because they are identical)
                    if 'threshold' in args.algorithms and 'maxlogits' in args.algorithms:
                        ax.set_title(f"{NAMES[protocol]} {NAMES['threshold']}/{NAMES['maxlogits']}")
                    else:
                        ax.set_title(f"{NAMES[protocol]} {NAMES[algorithm]}")
                
                # format ticks
                ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)

        # Manual legend
        legend_elements = [Line2D([None], [None], color=color_categories["knowns"]),
                           Line2D([None], [None], color=color_categories["negatives"]),
                           Line2D([None], [None], color=color_categories["unknowns"])]
        legend_labels = ["Known", "Negative", "Unknown"]
        fig.legend(handles=legend_elements, labels=legend_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5,-0.15))

        # X label
        fig.text(0.5, -0.05, f'{NAMES[loss]} Deep Feature Magnitudes', ha='center', fontsize=font_size)
        fig.text(0.01, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=font_size)

        pdf.savefig(bbox_inches='tight', pad_inches = 0)



def angle_legend(categories, category_names, category_colors, figure, **kwargs):
    """Creates a legend with the different line style and colors"""
    # add dummy plots for the different styles
    from matplotlib.lines import Line2D

    # create legend elements
    empty_legend = Line2D([None], [None], marker=".", visible=False)
    padding = 0
    l_padding = max(padding, 0)

    # add legend elements with sufficient padding
    legend_elements = \
            [empty_legend]*(l_padding//2) + \
            [Line2D([None], [None], linestyle='solid', color=category_colors[cat]) for cat in categories] + \
            [empty_legend]*(l_padding//2 + l_padding%2)
            # [empty_legend]*(a_padding//2) + \
            # [Line2D([None], [None], linestyle="solid", color=COLORS[algorithm]) for algorithm in algorithms] + \
            # [empty_legend]*(a_padding//2 + + a_padding%2)

    labels = \
            [""] *(l_padding//2) + \
            [category_names[cat] for cat in categories] + \
            [""]*(l_padding//2 + l_padding%2)
            # [""] *(a_padding//2) + \
            # [NAMES[algorithm] for algorithm in algorithms] + \
            # [""]*(a_padding//2 + + a_padding%2)

    # re-order row-first to column-first
    columns = len(categories)

    # indexes = [i for j in range(columns) for i in (j, j+columns)]
    indexes = [j for j in range(columns)]
    legend_elements = [legend_elements[index] for index in indexes]
    labels = [labels[index] for index in indexes]

    figure.legend(handles=legend_elements, labels=labels, loc="lower center", ncol=columns, **kwargs)


# probability curves
def round_to_one(d):
    """find smallest number n < 1 that rounds to 1 when rounding to d decimals."""
    numerator = (10**np.arange(1,d+1)).sum()*9 + 5
    denominator = 10**(d+1)
    return numerator/denominator

def find_s(d, C, alpha):
    """
    find lower strict bound on upper bound (beta) of logits s.t. the softmax for a class can still reach 1.

    logit_i in [alpha, beta] forall i in {1, ..., C}

    parameters
        d(int): decimals to which round softmax
        C(int): nr of classes, i.e., dimension of softmax
        alpha(float): lower bound of logits.
    """
    rounding_precision = round_to_one(d)
    return np.log(rounding_precision/(1-rounding_precision)) + np.log(C-1) + alpha

def l_ada(decimals, C):
    """lower bound of s implicitely given by AdaCos paper."""
    p_w = round_to_one(decimals)
    return  np.log((C-1)*p_w/(1-p_w))

def l_cos(decimals, C):
    """lower bound of s according to CosFace paper."""
    return l_ada(decimals, C) * ((C-1) / C)

def prob_cosine_margin(theta, s, m, C):
    """probability with cosine margin as function of s, m, C"""
    logit_j = np.exp(s * (np.cos(theta) - m))
    return logit_j / (logit_j + (C-1))

def prob_angular_margin(theta, s, m, C):
    """probability with angular margin as function of s, m, C"""
    logit_j = np.exp(s * np.cos(theta + m))
    return logit_j / (logit_j + (C-1))


def plot_angle_distributions(losses, protocols, angles, ground_truths):
    """plot distributions of angles to class centers. depicts angles of deep features to:
    - true class (for knowns)
    - closest class (for unknowns and negatives respectively)
    """
    class_per_protocol = {
        1: 116,
        2: 30,
        3: 151,
    }
    decimals = 5
    thetas = np.linspace(0, np.pi, 100)

    density=True
    P = len(protocols)
    L = len(losses)
    algorithm = 'threshold'
    categories = ['known_true', 'unknown_smallest', 'negative_smallest'] 
    # algs = [a for a in args.algorithms if a != 'maxlogits']
    # A = len(algs) + 3

    # fig = pyplot.figure(figsize=(6*(A+1),3*P))
    # gs = fig.add_gridspec(P, A+1, hspace=0.25, wspace=0.1)
    # axs = gs.subplots(sharex=True, sharey=False)
    # if P>1 or A>1:
    #     axs = axs.flat
    font_size = 15
    linewidth = 1.1


    step_size = numpy.pi/4
    x_tick = numpy.arange(0, numpy.pi + step_size, step_size)
    x_label = [r'0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']

    # Manual colors
    category_color = {
        'known_true': colors.to_rgba('tab:blue', 1),
        'unknown_smallest': colors.to_rgba('indianred', 1),
        'negative_smallest': colors.to_rgba('tab:green', 1),
        'curve': 'black',
    }
    category_color_fill = {
        'known_true': colors.to_rgba('tab:blue', .04),
        'unknown_smallest': colors.to_rgba('indianred', .04),
        'negative_smallest': colors.to_rgba('tab:green', .04),
    }
    category_name = {
        'known_true': "Known",
        'unknown_smallest': "Unknown",
        'negative_smallest': "Negative",
        'curve': 'Probability Curve',
    }
    
    fig = pyplot.figure(figsize=(4*P,3*L))
    gs = fig.add_gridspec(L, P, hspace=0.45, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)
    axs = axs.flat

    for index_l, loss in enumerate(losses):


        for index_p, protocol in enumerate(protocols):
            # for a, algorithm in enumerate(algs):

            distributions = openset_imagenet.util.get_angle_distributions(angles[protocol][loss][algorithm], ground_truths[protocol], density=density)

            ax = axs[3*index_l+index_p]
            
            for cat in categories:
                ax.stairs(
                    distributions[cat][0], distributions[cat][1], 
                    fill=True, color=category_color_fill[cat], edgecolor=category_color[cat],
                    label=category_name[cat], linewidth=linewidth
                )

            title = f'$P_{protocol}$ {NAMES[loss]}'
            ax.set_title(title, fontsize=font_size)

            # # axis formating
            ax.set_xlim(0, numpy.pi)
            if density:
                ax.set_ylim(0, 7)
            else:
                ax.set_ylim(0, 2000)
            # ax.legend(fontsize=6)
            ax.set_xticks(x_tick)
            ax.set_xticklabels(x_label)


            if loss not in ['softmax', 'entropic', 'objectosphere']:
                # superimpose probability curve
                ax2 = ax.twinx()

                C = class_per_protocol[protocol]
                s_cos = l_cos(decimals=decimals, C=C)
                s = np.ceil(s_cos)
                zero_curve = [prob_angular_margin(theta, s=s, m=0, C=C) for theta in thetas]
                if loss in ['cosface_sfn', 'cos_os', 'cos_eos']:  # cosine margin losses
                    curve = [prob_cosine_margin(theta, s=s, m=0.35, C=C) for theta in thetas]
                    ax2.plot(thetas, curve, color='black', linestyle='dashed', linewidth=linewidth)
                elif loss in ['arcface_sfn', 'arc_os', 'arc_eos']:  # angular margin losses
                    curve = [prob_angular_margin(theta, s=s, m=0.5, C=C) for theta in thetas]
                    ax2.plot(thetas, curve, color='black', linestyle='dashed', linewidth=linewidth)

                ax2.plot(thetas, zero_curve, color='black', linewidth=linewidth)
                if protocol != 3:
                    ax2.set_yticklabels([])


    # add legend
    anchor_legend_x, anchor_legend_y = 0.5, 0.02
    cats = categories.copy()
    cats.append('curve')
    angle_legend(
        cats, category_name, category_color, fig,
        bbox_to_anchor=(anchor_legend_x,anchor_legend_y), handletextpad=0.6, columnspacing=1.5, fontsize=font_size
    )

    # X label
    fig.text(0.5, 0.07, f'Angle of Sample to Class Center (in radians)', ha='center', fontsize=font_size)
    if density:
        y_label = 'Frequency'
    else:
        y_label = 'Density'
    fig.text(0.07, 0.5, y_label, va='center', rotation='vertical', fontsize=font_size)
    fig.text(1-0.06, 0.5, 'Softmax Score', va='center', rotation='vertical', fontsize=font_size)


def plot_CCR_FPR(args, scores, ground_truths, pdf):
    """plot CCR and FPR separately as functions of the threshold."""
    P = len(args.protocols)
    # plot_maxlogits = 'maxlogits' in args.algorithms
    # algs = [a for a in args.algorithms if a != 'maxlogits']

    for protocol in args.protocols:

        A = len(args.algorithms)
        # A = 2
        fig = pyplot.figure(figsize=(4*(A*2),3*2))
        gs = fig.add_gridspec(2, 2*A, hspace=0.25, wspace=0.1)
        axs = gs.subplots(sharex='col', sharey='row')
        axs = axs.flat
        font_size = 15
        linewidth = 1.1

        max_ccr = 0

        labels = [-1, -2]
        label_names = ["Negative", "Unknown"]

        for loss_function in args.losses:
            for alg_idx, algorithm in enumerate(args.algorithms):

                for label_idx, label in enumerate(labels):  # negative labels, unknown label

                    ccr_plot_idx = 2*alg_idx+label_idx
                    fpr_plot_idx = 2*A+2*alg_idx+label_idx

                    ccr, fpr, thresholds = openset_imagenet.util.calculate_oscr(ground_truths[protocol], scores[protocol][loss_function][algorithm], unk_label=label, return_thresholds=True)
                    max_ccr_curr = numpy.amax(ccr)
                    if max_ccr_curr > max_ccr:
                        max_ccr = max_ccr_curr

                    axs[ccr_plot_idx].plot(thresholds, ccr, 
                        linestyle=STYLES[loss_function], color=COLORS[algorithm],
                        linewidth=linewidth
                    )
                    axs[fpr_plot_idx].plot(thresholds, fpr, 
                        linestyle=STYLES[loss_function], color=COLORS[algorithm],
                        linewidth=linewidth
                    )

                    # add titles
                    axs[ccr_plot_idx].set_title(f"CCR {label_names[label_idx]}", fontsize=font_size)
                    axs[fpr_plot_idx].set_title(f"FPR {label_names[label_idx]}", fontsize=font_size)

                    # axis formating
                    max_ccr = min(1, max_ccr*1.1)
                    axs[ccr_plot_idx].set_ylim(0,max(max_ccr, 0.8))
                    axs[fpr_plot_idx].set_ylim(8*1e-5, 1.4)
                    axs[fpr_plot_idx].set_yscale('log')

        # figure labels
        fig.text(0.5, 0.05, 'Threshold', ha='center', fontsize=font_size)
        # fig.text(0.08, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font_size)
        # fig.text(0.92, 0.5, 'FPR', va='center', rotation='vertical', fontsize=font_size)

        # add legend
        openset_imagenet.util.oscr_legend(args.losses, args.algorithms, fig,
            bbox_to_anchor=(0.5,-0.1), handletextpad=0.6, columnspacing=1.5,
            title="How to Read: Line Style -> Loss; Color -> Algorithm"
        )

        pdf.savefig(bbox_inches='tight', pad_inches = 0)


def plot_single_oscr(fpr, ccr, ax, loss, algorithm, scale, max_ccr, lower_bound_CCR=None, lower_bound_FPR=None):

    linewidth = 1.1
    max_ccr = min(1, max_ccr*1.1)

    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Manual limits
        ax.set_ylim(0.09, 1)
        ax.set_xlim(8 * 1e-5, 1.4)
        # Manual ticks
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=100))
        locmin = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    elif scale == 'semilog':
        ax.set_xscale('log')

        # Manual limits
        if lower_bound_CCR is not None and lower_bound_CCR < max(0.8, max_ccr):
            ax.set_ylim(lower_bound_CCR, max(0.8, max_ccr))
        else:
            ax.set_ylim(0.0, max(0.8, max_ccr))

        if lower_bound_FPR is not None:
            ax.set_xlim(lower_bound_FPR, 1.01)
        else:
            ax.set_xlim(8 * 1e-5, 1.4)

        # Manual ticks
        if lower_bound_CCR is not None and lower_bound_CCR < max(0.8, max_ccr):
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))  # MaxNLocator(7))  #, prune='lower'))
        else:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # MaxNLocator(7))  #, prune='lower'))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        locmin = ticker.LogLocator(base=10.0, subs=np.linspace(0, 1, 10, False), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.set_ylim(0.0, max(0.8, max_ccr))
        # ax.set_xlim(None, None)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # , prune='lower'))
    # Remove fpr=0 since it cause errors with different ccrs and logscale.
#    if len(x):
#        non_zero = x != 0
#        x = x[non_zero]
#        y = y[non_zero]
    ax.plot(fpr,
            ccr,
            linestyle=STYLES[loss],
            color=COLORS[loss],
            linewidth=linewidth)  # marker='2', markersize=1
    return ax


def calculate_oscr(gt, scores, unk_label=-1, return_thresholds=False):
    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (ccr, fpr) for every score.
    Args:
        gt (np.array): Integer array of target class labels.
        scores (np.array): Float array of dim [N_samples, N_classes]
        unk_label (int): Label to calculate the fpr, either negatives or unknowns. Defaults to -1 (negatives)
    Returns: Two lists first one for ccr, second for fpr.
    """
    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = gt.astype(int)
    kn = gt >= 0
    unk = gt == unk_label

    # Get total number of samples of each type
    total_kn = np.sum(kn)
    total_unk = np.sum(unk)

    ccr, fpr = [], []
    # get predicted class for known samples
    pred_class = np.argmax(scores, axis=1)[kn]
    correctly_predicted = pred_class == gt[kn]
    target_score = scores[kn][range(kn.sum()), gt[kn]]

    # TODO: why are the scores only considered of the unknowns? because we plot only on the unknowns?
    # get maximum scores for unknown samples
    max_score = np.max(scores, axis=1)[unk]

    # Any max score can be a threshold
    # TODO: add 0 to the thresholds? why are only unknowns considered? shouldn't it be: thresholds = np.unique(np.max(scores, axis=1))
    thresholds = np.unique(max_score)

    #print(target_score) #HB
    for tau in thresholds:
        # compute CCR value
        val = (correctly_predicted & (target_score >= tau)).sum() / total_kn
        ccr.append(val)

        val = (max_score >= tau).sum() / total_unk
        fpr.append(val)

    ccr = np.array(ccr)
    fpr = np.array(fpr)
    if return_thresholds:
        return ccr, fpr, thresholds
    else:
        return ccr, fpr


def plot_oscr(algorithms, losses, arrays, gt, scale='linear', title=None, ax_label_font=13, ax=None, unk_label=-1, lower_bound_CCR=None, lower_bound_FPR=None):
    """Plots OSCR curves for all given scores.
    The scores are stored as arrays: Float array of dim [N_samples, N_classes].
    The arrays contain scores for various loss functions and algorithms as arrays[loss][algorithm].
    """

    max_ccr = 0

    # for loss, loss_arrays in arrays.items():
    for loss in losses:
        loss_arrays = arrays[loss]
        for algorithm in algorithms:
        # for algorithm, scores in loss_arrays.items():
            scores = loss_arrays[algorithm]
            ccr, fpr = calculate_oscr(gt, scores, unk_label)

            max_ccr_curr = np.amax(ccr)
            if max_ccr_curr > max_ccr:
                max_ccr = max_ccr_curr
            
            ax = plot_single_oscr(fpr, ccr,
                              ax=ax,
                              loss=loss,
                              algorithm=algorithm,
                              scale=scale,
                              max_ccr=max_ccr,
                              lower_bound_CCR=lower_bound_CCR,
                              lower_bound_FPR=lower_bound_FPR
                              )

    if title is not None:
        ax.set_title(title, fontsize=ax_label_font)
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False, labelsize=ax_label_font)

    return ax


def oscr_legend(losses, algorithms, figure, **kwargs):
    """Creates a legend with the different line style and colors"""
    # add dummy plots for the different styles
    from matplotlib.lines import Line2D

    # create legend elements
    empty_legend = Line2D([None], [None], marker=".", visible=False)
    padding = len(algorithms) - len(losses)
    l_padding = max(padding, 0)

    # add legend elements with sufficient padding
    legend_elements = \
            [empty_legend]*(l_padding//2) + \
            [Line2D([None], [None], linestyle=STYLES[loss], color=COLORS[loss]) for loss in losses] + \
            [empty_legend]*(l_padding//2 + l_padding%2)
            # [empty_legend]*(a_padding//2) + \
            # [Line2D([None], [None], linestyle="solid", color=COLORS[algorithm]) for algorithm in algorithms] + \
            # [empty_legend]*(a_padding//2 + + a_padding%2)

    labels = \
            [""] *(l_padding//2) + \
            [NAMES[loss] for loss in losses] + \
            [""]*(l_padding//2 + l_padding%2)
            # [""] *(a_padding//2) + \
            # [NAMES[algorithm] for algorithm in algorithms] + \
            # [""]*(a_padding//2 + + a_padding%2)

    # re-order row-first to column-first
    columns = len(losses)

    # indexes = [i for j in range(columns) for i in (j, j+columns)]
    indexes = [j for j in range(columns)]
    legend_elements = [legend_elements[index] for index in indexes]
    labels = [labels[index] for index in indexes]

    figure.legend(handles=legend_elements, labels=labels, loc="lower center", ncol=columns, **kwargs)



def plot_OSCR(protocols, algorithms, losses, scores, ground_truths, lower_bound_CCR=None, lower_bound_FPR=None):
    # plot OSCR
    P = len(protocols)
    fig = pyplot.figure(figsize=(8,3*P))
    gs = fig.add_gridspec(P, 2, hspace=0.25, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)
    axs = axs.flat
    fontsize = 12

    if P == 1:
        fig.subplots_adjust(bottom=0.3)

    for index, protocol in enumerate(protocols):
        plot_oscr(
            algorithms=algorithms,
            losses=losses,
            arrays=scores[protocol], gt=ground_truths[protocol], 
            scale="semilog", title=f'$P_{protocol}$ Negative',
            ax_label_font=fontsize, ax=axs[2*index], unk_label=-1,
            lower_bound_CCR=lower_bound_CCR, lower_bound_FPR=lower_bound_FPR
        )
        plot_oscr(
            algorithms=algorithms,
            losses=losses,
            arrays=scores[protocol], gt=ground_truths[protocol], 
            scale="semilog", title=f'$P_{protocol}$ Unknown',
            ax_label_font=fontsize, ax=axs[2*index+1], unk_label=-2,
            lower_bound_CCR=lower_bound_CCR, lower_bound_FPR=lower_bound_FPR
        )
    # Axis properties
    for ax in axs:
        ax.label_outer()
        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')

    # positioning of legend and axis labels
    if len(protocols) == 3:
        anchor_legend_x, anchor_legend_y = 0.5, 0.02
        anchor_text_x, anchor_text_y = 0.5, 0.06
    if len(protocols) == 1:
        anchor_legend_x, anchor_legend_y = 0.5, 0.01
        anchor_text_x, anchor_text_y = 0.5, 0.15

    # Figure labels
    fig.text(0.06, 0.5, 'CCR', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(anchor_text_x, anchor_text_y, 'FPR', ha='center', fontsize=fontsize)

    # add legend
    oscr_legend(
        losses, algorithms, fig,
        bbox_to_anchor=(anchor_legend_x,anchor_legend_y), handletextpad=0.6, columnspacing=1.5
    )




# from openset_imagenet.util import NAMES
def plot_score_distributions(args, scores, ground_truths, pdf):

    font_size = 15
    bins = 30
    P = len(args.protocols)
    L = len(args.losses)
    # A = len(args.algorithms)
    algorithms = ['threshold', 'maxlogits']
    A = len(algorithms)

    # Manual colors
    edge_unknown = colors.to_rgba('indianred', 1)
    fill_unknown = colors.to_rgba('firebrick', 0.04)
    edge_known = colors.to_rgba('tab:blue', 1)
    fill_known = colors.to_rgba('tab:blue', 0.04)
    edge_negative = colors.to_rgba('tab:green', 1)
    fill_negative = colors.to_rgba('tab:green', 0.04)

    for loss in args.losses:

        fig = pyplot.figure(figsize=(3*(A+1), 2*P))
        gs = fig.add_gridspec(P, A+1, hspace=0.2, wspace=0.08)
        axs = gs.subplots(sharex=False, sharey=False)
        if P>1 or A>1:
            axs = axs.flat

        for p, protocol in enumerate(args.protocols):

            for a, algorithm in enumerate(algorithms):
            # for a, algorithm in enumerate(args.algorithms):
                # compute histogram
                if scores[protocol][loss][algorithm] is not None:
                    histograms = openset_imagenet.util.get_histogram(
                        scores[protocol][loss][algorithm],
                        ground_truths[protocol],
                        bins=bins
                    )

                # original version (caused error)
                # ax = axs[p,a]

                # my workaround to handle plots for only 1 algorithm. 
                # if A > 1 and P > 1:
                #     ax = axs[p,a]
                # elif A > 1:
                #     ax = axs[a]
                # elif P > 1:
                #     ax = axs[p]
                # else:
                #     ax = axs

                if P==1 and A==1:
                    ax = axs
                else:
                    ax = axs[2*p+a]

                # plot softmax/maxlogit scores
                ax.stairs(histograms["known"][0], histograms["known"][1], fill=True, color=fill_known, edgecolor=edge_known, linewidth=1)
                ax.stairs(histograms["unknown_max"][0], histograms["unknown_max"][1], fill=True, color=fill_unknown, edgecolor=edge_unknown, linewidth=1)
                ax.stairs(histograms["negative_max"][0], histograms["negative_max"][1], fill=True, color=fill_negative, edgecolor=edge_negative, linewidth=1)
                if algorithm == 'threshold':
                    ax.set_xlim((0,1))


                ax.set_title(f"{NAMES[protocol]} {NAMES[algorithm]}")

                # set tick locator
                ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True, labelsize=font_size)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.label_outer()

            # plot mean score distributions

            histograms = openset_imagenet.util.get_histogram(
                scores[protocol][loss]['maxlogits'],
                ground_truths[protocol],
                bins=bins
            )

            ax = axs[2*p+A]
            ax.stairs(histograms["known_wrong_mean"][0], histograms["known_wrong_mean"][1], fill=True, color=fill_known, edgecolor=edge_known, linewidth=1)
            ax.stairs(histograms["unknown_mean"][0], histograms["unknown_mean"][1], fill=True, color=fill_unknown, edgecolor=edge_unknown, linewidth=1)
            ax.stairs(histograms["negative_mean"][0], histograms["negative_mean"][1], fill=True, color=fill_negative, edgecolor=edge_negative, linewidth=1)

            ax.set_title(f"{NAMES[protocol]} AvgLogits (wrong class)")

            # set tick locator
            ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True, labelsize=font_size)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.label_outer()

        # Manual legend
        legend_elements = [Line2D([None], [None], color=edge_known),
                           Line2D([None], [None], color=edge_negative),
                           Line2D([None], [None], color=edge_unknown)]
        legend_labels = ["Known", "Negative", "Unknown"]
        fig.legend(handles=legend_elements, labels=legend_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5,-0.3))

        # X label
        fig.text(0.5, -0.08, f'{NAMES[loss]} Scores', ha='center', fontsize=font_size)

        pdf.savefig(bbox_inches='tight', pad_inches = 0)


from openset_imagenet.script.parameter_selection import THRESHOLDS
def ccr_table(groups, protocols, fpr_thresholds, scores, gt, highlight_overall_best=True):

    def nonemax(a,b):
        b = numpy.array([v if v is not None else numpy.nan for v in b])
        return numpy.where(numpy.logical_and(numpy.logical_not(numpy.isnan(b)), b >= a), b, a)

    alg = 'threshold'

    for protocol in protocols:
        # latex_file = args.tables.format(protocol, 'best' if use_best else 'last')
        latex_file = f'results/performance_tables/p{protocol}_results_thesis.tex'
        print("Writing CCR tables for protocol", protocol, "to file", latex_file)
        # compute all CCR values and store maximum values
        results = collections.defaultdict(dict)
        max_total = numpy.zeros(len(fpr_thresholds))
        max_by_grp = collections.defaultdict(lambda:numpy.zeros(len(fpr_thresholds)))
        # max_by_loss = collections.defaultdict(lambda:numpy.zeros(len(fpr_thresholds)))
        # for algorithm in algorithms:
        for group, losses in groups.items():
            for loss in losses:
                ccrs = openset_imagenet.util.ccr_at_fpr(gt[protocol], scores[protocol][loss][alg], fpr_thresholds, unk_label=-2)
                results[group][loss] = ccrs
                max_total = nonemax(max_total, ccrs)
                max_by_grp[group] = nonemax(max_by_grp[group], ccrs)
                # max_by_loss[loss] = nonemax(max_by_loss[loss], ccrs)


        with open(latex_file, "w") as f:
            # write header
            # f.write("\\begin{table}[t]\n")
            # f.write("\\centering\n")

            f.write("\\begin{tabularx}{.7\\textwidth}{|c|c||ccc|c|}\n")
            f.write("\\cline{1-6}\n")
            f.write("\multirow{2}{*}{\\bf Group} & \\multirow{2}{*}{\\bf Loss} & \\multicolumn{3}{c|}{\\bf CCR@FPR} & \\bf {Acc} \\\\ \\cline{3-6}\n")
            # f.write("\\bf Group & \\bf Loss & ")
            f.write("& &" + " & ".join([THRESHOLDS[t] for t in fpr_thresholds]))
            f.write("\\\\\\cline{1-6}\\cline{1-6}\n")
            for group, losses in groups.items():
                f.write(f"\\multirow{{{len(losses)}}}{{*}}{{{group}}}")
                for loss in losses:
                    f.write(f" & {NAMES[loss]}")
                    for i, v in enumerate(results[group][loss]):
                        if v is None: f.write(" &")
                        elif v == max_total[i]: 
                            if highlight_overall_best:
                                f.write(f" & \\textcolor{{blue}}{{\\bf {v:.4f}}}")
                            else:
                                f.write(f" & \\underline {{{v:.4f}}}")
                        elif v == max_by_grp[group][i]: f.write(f" & \\underline {{{v:.4f}}}")
                        # elif v == max_by_loss[loss][i]: f.write(f" & \\underline{{{v:.4f}}}")
                        else: f.write(f" & {v:.4f}")
                    f.write("\\\\\n")
                f.write("\\cline{1-6}\n")
            f.write("\\end{tabularx}\n")
            # f.write("\\end{table}")


def plot_feature_distributions(losses, features, ground_truths):
    """plot deep features."""
    # algs = [a for a in args.algorithms if a != 'maxlogits']
    # only compute this for threshold to allow for arangement with normalized representation
    algorithm = 'threshold'
    # visualization = ['threshold', 'normalized']
    # V = len(visualization)
    magnitude = 10
    protocol = 10
    font_size = 15

    clrs = cm.tab10(range(10))
    color_map = {i:clrs[i] for i in range(10) if i!=7}  # use tab10 colors except gray
    color_map[7] = 'teal'
    color_map[-1] = 'gray'
    color_map[-2] = 'black'

    known_colors = [color_map[i] for i in range(10)]

    target_colors = [color_map[i] for i in ground_truths[0]]

    known_only_idx = ground_truths[0] >= 0
    known_with_negative_idx = numpy.logical_or(ground_truths[0] == -1, known_only_idx)
    known_with_unknown_idx = numpy.logical_or(ground_truths[0] == -2, known_only_idx)
    subsets = [known_only_idx, known_with_negative_idx, known_with_unknown_idx, known_only_idx]
    # subset_names = ["Known Classes", "Known and Negative Classes", "Known and Unknown Classes", "Normalized Feature Magnitude"]
    subset_names = ["known", "negative", "unknown", "normalized"]
    S = len(subsets)


    for loss in losses:

        if features[protocol][loss][algorithm].shape[1] == 2:

            # font_size = 15
            linewidth = 1.1

            model_save = torch.load(f'experiments/Protocol_{protocol}/{loss}_threshold_curr.pth', map_location=torch.device('cpu'))
            W = model_save['model_state_dict']['logits.w'].numpy()

            abs_max = numpy.amax(numpy.abs(features[protocol][loss][algorithm])) * 1.05  # add 10% margin
            abs_max = numpy.ceil(abs_max)

            # plot without knowns negatives and without unknowns, with negatives, and only with unknowns
            for i, subset in enumerate(subsets):

                fig = pyplot.figure(figsize=(4,4))
                ax = fig.subplots()

                feats = features[protocol][loss][algorithm]
                if i == len(subsets)-1:
                    feats = normalize_array(feats, axis=1, ord=2) * magnitude  # multiply by 10 for better visual scale
                    W_norm = normalize_array(W.T, axis=1, ord=2).T * magnitude

                color = [target_colors[k] for k in numpy.arange(len(target_colors)) if subset[k]]

                ax.scatter(
                    x=feats[:,0][subset],
                    y=feats[:,1][subset],
                    c=color,
                    s=5,
                    linewidth=linewidth,
                    alpha=.1,
                    marker='.'
                )

                if i == len(subsets)-1:
                    for j, c in enumerate(known_colors):
                        ax.plot([0, W_norm[0,j]], [0, W_norm[1,j]], color=c)


                if i == len(subsets)-1:
                    abs_max = numpy.amax(numpy.abs(feats)) * 1.05  # add 10% margin
                    abs_max = numpy.ceil(abs_max)
                ax.set_xlim((-abs_max, abs_max))
                ax.set_ylim((-abs_max, abs_max))
                ax.tick_params(labelsize=font_size)
                # ax.set_xlabel(subset_names[i], ha='center', fontsize=font_size)
                if i == len(subsets)-1:
                    ax.axis('off')

                plt.savefig(f'results/deep_features/df_{loss}_{subset_names[i]}.png')
                plt.close(fig)
                

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def main(command_line_arguments = None):
    args = command_line_options(command_line_arguments)
    cfg = openset_imagenet.util.load_yaml(args.configuration)

    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])

    print("Extracting and loading scores")
    scores, ground_truths, features, logits, angles = load_scores(args, cfg)
    training_scores = load_training_scores(args, cfg)

    create_folder('results/deep_features')
    create_folder('results/performance_plots')
    create_folder('results/performance_tables')

    # ========== main experiments ==========

    # plot OSCR for protocols 1-3 (SFN Margin Losses)
    print('Plotting OSCR curves and angle distributions for SFN Margin losses')
    losses = ['softmax', 'norm_sfn', 'cosface_sfn', 'arcface_sfn']
    protocols = [1,2,3]

    plot_OSCR(protocols, ['threshold'], losses, scores, ground_truths, lower_bound_CCR=None, lower_bound_FPR=None)
    pyplot.savefig('results/performance_plots/sfn-margin-oscr.png')
    plot_OSCR(protocols, ['maxlogits'], losses, scores, ground_truths, lower_bound_CCR=None, lower_bound_FPR=None)
    pyplot.savefig('results/performance_plots/sfn-margin-oscr-maxlogits.png')

    plot_angle_distributions(losses, protocols, angles, ground_truths)
    pyplot.savefig('results/performance_plots/sfn-margin-angles.png')

    # plot OSCR for protocols 1-3 (Margin-OS)
    print('Plotting OSCR curves and angle distributions for Margin-OS losses')
    losses = ['objectosphere', 'softmax_os', 'cos_os', 'arc_os']
    protocols = [1,2,3]

    plot_OSCR(protocols, ['threshold'], losses, scores, ground_truths, lower_bound_CCR=None, lower_bound_FPR=None)
    pyplot.savefig('results/performance_plots/margin-os-oscr.png')
    plot_OSCR(protocols, ['maxlogits'], losses, scores, ground_truths, lower_bound_CCR=None, lower_bound_FPR=None)
    pyplot.savefig('results/performance_plots/margin-os-oscr-maxlogits.png')

    plot_angle_distributions(losses, protocols, angles, ground_truths)
    pyplot.savefig('results/performance_plots/margin-os-angles.png')

    # plot OSCR for protocols 1-3 (Margin-EOS)
    print('Plotting OSCR curves and angle distributions for Margin-EOS losses')
    losses = ['objectosphere', 'norm_eos', 'cos_eos', 'arc_eos']
    protocols = [1,2,3]

    plot_OSCR(protocols, ['threshold'], losses, scores, ground_truths, lower_bound_CCR=None, lower_bound_FPR=None)
    pyplot.savefig('results/performance_plots/margin-eos-oscr.png')
    plot_OSCR(protocols, ['maxlogits'], losses, scores, ground_truths, lower_bound_CCR=None, lower_bound_FPR=None)
    pyplot.savefig('results/performance_plots/margin-eos-oscr-maxlogits.png')

    plot_angle_distributions(losses, protocols, angles, ground_truths)
    pyplot.savefig('results/performance_plots/margin-eos-angles.png')


    # create table for results on protocols 1-3 (split into sections per methods)
    groups = {
        'Benchmarks': ['softmax', 'entropic', 'objectosphere'],
        'SFN Margin': ['norm_sfn', 'cosface_sfn', 'arcface_sfn'],
        'Margin-OS' : ['softmax_os', 'cos_os', 'arc_os'],
        'Margin-EOS': ['norm_eos', 'cos_eos', 'arc_eos']
    }
    ccr_table(groups=groups, protocols=[1,2,3], fpr_thresholds=args.fpr_thresholds, scores=scores, gt=ground_truths)


    # ========== preliminary experiments ==========

    # comparison of preliminary experiments
    groups = {
        'Margin Types'  : ['sm_softmax', 'sphereface', 'cosface', 'arcface'],
        'HFN vs. SFN (CosFace)': ['cosface', 'cosface_sfn'],
        'HFN vs. SFN (ArcFace)': ['arcface', 'arcface_sfn'],
        'HFN vs. SFN (Cos-EOS)': ['cos_eos', 'cos_eos_sfn'],
        'HFN vs. SFN (Arc-EOS)': ['arc_eos', 'arc_eos_sfn'],
        'OS vs. sym. OS (Cos-OS)': ['cos_os', 'cos_os_non_symmetric'],
        'OS vs. sym. OS (Arc-OS)': ['arc_os', 'arc_os_non_symmetric']
    }
    ccr_table(groups=groups, protocols=[0], fpr_thresholds=args.fpr_thresholds, scores=scores, gt=ground_truths, highlight_overall_best=False)

    for group, losses in groups.items():
        print(f'Plotting OSCR curves for "{group}"')
        plot_OSCR([0], ['threshold'], losses, scores, ground_truths, lower_bound_CCR=None, lower_bound_FPR=None)
        pyplot.savefig(f'results/performance_plots/early_{group}.png')


    # ========== df visualizations ==========

    # plot deep feature visualizations
    print("Plotting deep feature distributions")
    losses = [
        'softmax', 'entropic', 'objectosphere',               	# benchmarks
        'sphereface', 'cosface', 'arcface',  					# face losses (HFN)
        'norm_sfn', 'cosface_sfn', 'arcface_sfn',  				# face losses (SFN)
        'softmax_os', 'cos_os', 'arc_os',  						# margin-OS (SFN)
    ]
    plot_feature_distributions(losses, features, ground_truths)


    # ==========================================

    # print("Writing file", args.plots)
    # pdf = PdfPages(args.plots)
    # try:
        # plot OSCR (actually not required for best case)
    #     print("Plotting OSCR curves")
    #     plot_OSCR(args, scores, ground_truths)
    #     pdf.savefig(bbox_inches='tight', pad_inches = 0)
    #     plot_OSCR(args, scores, ground_truths, lower_bound_CCR=.97, lower_bound_FPR=0.1)
    #     pdf.savefig(bbox_inches='tight', pad_inches = 0)

    #     print("Plotting CCR and FPR curves")
    #     plot_CCR_FPR(args, scores, ground_truths, pdf)

    #     """
    #     if not args.linear and not args.use_best and not args.sort_by_loss:
    #       # plot confidences
    #       print("Plotting confidence plots")
    #       plot_confidences(args)
    #       pdf.savefig(bbox_inches='tight', pad_inches = 0)
    #     """

    #     # plot histograms
    #     print("Plotting score distribution histograms")
    #     plot_score_distributions(args, scores, ground_truths, pdf)

    #     # plot feature magnitude distributions
    #     print("Plotting feature magnitude distributions")
    #     plot_feature_magnitudes(args, features, ground_truths, pdf)

    #     # plot angle distribution
    #     print("Plotting angle distribution")
    #     plot_angle_distributions(args, angles, ground_truths, pdf)

    #     # plot training scores
    #     print("Plotting training metrics")
    #     plot_training_metrics(args, training_scores, pdf)

    #     if 0 in args.protocols:
    #         print("Plotting deep feature distributions")
    #         plot_feature_distributions(args, features, ground_truths, pdf)
    # finally:
    #     pdf.close()

    # # create result table
    # print("Creating CCR Tables")
    # ccr_table(args, scores, ground_truths)

if __name__ == '__main__':
    main()