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

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from collections import defaultdict

from matplotlib import pyplot, cm, colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, LogLocator

from openset_imagenet.util import STYLES, COLORS

#def train_one(cmd):
#  print(cmd)
#  print(" ".join(cmd))

def command_line_options(command_line_arguments=None):
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Plotting", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--protocols", "-p",
        type=int,
        choices = (1,2,3),
        nargs="+",
        default = (1,2,3),
        help="Select the protocols that should be evaluated"
    )
    parser.add_argument(
        "--losses", "-l",
        nargs = "+",
        choices = ('softmax', 'garbage', 'entropic', 'sphereface', 'cosface', 'arcface', 'magface', 'cosos', 'coseos'),
        default = ('softmax', 'entropic', 'cosface', 'cosos'),
        help = "Select the loss functions that should be included into the plot"
    )
    parser.add_argument(
        "--algorithms", "-a",
        choices = ["threshold", "openmax", "evm", "proser", "maxlogits"],
        nargs = "+",
        default = ["threshold", "openmax", "evm", "proser", "maxlogits"],
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
    ground_truths = {}
#    epoch = {p:{} for p in args.protocols}
    for protocol in args.protocols:
        for loss in args.losses:
            for algorithm in args.algorithms:
                output_directory = pathlib.Path(cfg.output_directory) / f"Protocol_{protocol}"
                alg = "threshold" if algorithm == "maxlogits" else algorithm
                scr = "logits" if algorithm == "maxlogits" else "scores"
                score_file = output_directory / f"{loss}_{alg}_test_arr_{suffix}.npz"

                if os.path.exists(score_file):
                    # remember files
                    results = numpy.load(score_file)

                    scores[protocol][loss][algorithm] = results[scr]

                    if protocol not in ground_truths:
                        ground_truths[protocol] = results["gt"].astype(int)
                    else:
                        assert numpy.all(results["gt"] == ground_truths[protocol])

                    logger.info(f"Loaded score file {score_file} for protocol {protocol}, {loss}, {algorithm}")
                else:
                    logger.warning(f"Did not find score file {score_file} for protocol {protocol}, {loss}, {algorithm}")

    return scores, ground_truths


def plot_training_scores(args, training_scores, pdf):
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
            max_train_loss_curr = numpy.max(loss_function_arrays['train_loss'])
            max_val_loss_curr = numpy.max(loss_function_arrays['val_loss'])
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
        axs[2*index].set_ylim(0,min(40, round(max_loss)))
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


def plot_CCR_FPR(args, scores, ground_truths, pdf):
    """plot CCR and FPR separately as functions of the threshold."""
    P = len(args.protocols)
    fig = pyplot.figure(figsize=(12,3*P))
    gs = fig.add_gridspec(P, 2, hspace=0.25, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=False)
    axs = axs.flat
    font_size = 15
    linewidth = 1.1

    unk_label = -1

    for index, protocol in enumerate(args.protocols):
        for loss_function, loss_function_arrays in scores[protocol].items():
            for algorithm, scores_array in loss_function_arrays.items():

                ccr, fpr, thresholds = openset_imagenet.util.calculate_oscr(ground_truths[protocol], scores_array, unk_label, return_thresholds=True)

                axs[2*index].plot(thresholds, ccr, 
                    linestyle=STYLES[loss_function], color=COLORS[algorithm],
                    linewidth=linewidth
                )
                axs[2*index+1].plot(thresholds, ccr, 
                    linestyle=STYLES[loss_function], color=COLORS[algorithm],
                    linewidth=linewidth
                )

        # add titles
        axs[2*index].set_title("CCR", fontsize=font_size)
        axs[2*index+1].set_title("FPR", fontsize=font_size)

        # axis formating
        axs[2*index].set_ylim(0,0.8)
        axs[2*index+1].set_ylim(8*1e-5, 1.4)
        axs[2*index+1].set_yscale('log')

    # figure labels
    fig.text(0.5, -0.05, 'Threshold', ha='center', fontsize=font_size)
    fig.text(0.08, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font_size)
    fig.text(0.92, 0.5, 'FPR', va='center', rotation='vertical', fontsize=font_size)

    # add legend
    openset_imagenet.util.oscr_legend(args.losses, args.algorithms, fig,
        bbox_to_anchor=(0.5,-0.3), handletextpad=0.6, columnspacing=1.5,
        title="How to Read: Line Style -> Loss; Color -> Algorithm"
    )

    pdf.savefig(bbox_inches='tight', pad_inches = 0)


def plot_OSCR(args, scores, ground_truths):
    # plot OSCR
    P = len(args.protocols)
    fig = pyplot.figure(figsize=(8,3*P))
    gs = fig.add_gridspec(P, 2, hspace=0.25, wspace=0.1)
    axs = gs.subplots(sharex=True, sharey=True)
    axs = axs.flat
    font = 15

    for index, protocol in enumerate(args.protocols):
        openset_imagenet.util.plot_oscr(arrays=scores[protocol], gt=ground_truths[protocol], scale="semilog", title=f'$P_{protocol}$ Negative',
                    ax_label_font=font, ax=axs[2*index], unk_label=-1,)
        openset_imagenet.util.plot_oscr(arrays=scores[protocol], gt=ground_truths[protocol], scale="semilog", title=f'$P_{protocol}$ Unknown',
                    ax_label_font=font, ax=axs[2*index+1], unk_label=-2,)
    # Axis properties
    for ax in axs:
        ax.label_outer()
        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')

    # Figure labels
    fig.text(0.5, -0.05, 'FPR', ha='center', fontsize=font)
    fig.text(0.04, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)

    # add legend
    openset_imagenet.util.oscr_legend(
        args.losses, args.algorithms, fig,
        bbox_to_anchor=(0.5,-0.3), handletextpad=0.6, columnspacing=1.5,
        title="How to Read: Line Style -> Loss; Color -> Algorithm"
    )


from openset_imagenet.util import NAMES
def plot_score_distributions(args, scores, ground_truths, pdf):

    font_size = 15
    bins = 30
    P = len(args.protocols)
    L = len(args.losses)
    algorithms = [a for a in args.algorithms if a != "maxlogits"]
#    algorithms = args.algorithms
    A = len(algorithms)

    # Manual colors
    edge_unknown = colors.to_rgba('indianred', 1)
    fill_unknown = colors.to_rgba('firebrick', 0.04)
    edge_known = colors.to_rgba('tab:blue', 1)
    fill_known = colors.to_rgba('tab:blue', 0.04)
    edge_negative = colors.to_rgba('tab:green', 1)
    fill_negative = colors.to_rgba('tab:green', 0.04)

    for loss in args.losses:

        fig = pyplot.figure(figsize=(3*A+1, 2*P))
        gs = fig.add_gridspec(P, A, hspace=0.2, wspace=0.08)
        axs = gs.subplots(sharex=True, sharey=False)

        for p, protocol in enumerate(args.protocols):
            max_a = (0, 0)

            for a, algorithm in enumerate(algorithms):
                # Calculate histogram
                # workaround to handle plots for only 1 algorithm
                if A > 1 and P > 1:
                    ax = axs[p,a]
                elif A > 1:
                    ax = axs[a]
                elif P > 1:
                    ax = axs[p]
                else:
                    ax = axs

                if scores[protocol][loss][algorithm] is not None:
                    histograms = openset_imagenet.util.get_histogram(
                        scores[protocol][loss][algorithm],
                        ground_truths[protocol],
                        bins=bins
                    )
                    # Plot histograms
                    ax.stairs(histograms["known"][0], histograms["known"][1], fill=True, color=fill_known, edgecolor=edge_known, linewidth=1)
                    ax.stairs(histograms["unknown"][0], histograms["unknown"][1], fill=True, color=fill_unknown, edgecolor=edge_unknown, linewidth=1)
                    ax.stairs(histograms["negative"][0], histograms["negative"][1], fill=True, color=fill_negative, edgecolor=edge_negative, linewidth=1)

                ax.set_title(f"{NAMES[protocol]} {NAMES[algorithm]}")

                # set tick locator
                max_a = max(max_a, (max(numpy.max(h[0]) for h in histograms.values()), a))
                ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.label_outer()

            # share axis of the maximum value
            for a in range(A):
                if a != max_a[1]:
#                    axs[p,a].set_ylim([0, max_a[0]])
                    if A > 1 and P > 1:
                        axs[p,a].sharey(axs[p,max_a[1]])
                    elif A > 1:
                        axs[a].sharey(axs[max_a[1]])
                    elif P > 1:
                        axs[p].sharey(axs[p])
                    # else:
                    #     ax = axs


        # Manual legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([None], [None], color=edge_known),
                           Line2D([None], [None], color=edge_negative),
                           Line2D([None], [None], color=edge_unknown)]
        legend_labels = ["Known", "Negative", "Unknown"]
        fig.legend(handles=legend_elements, labels=legend_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5,-0.3))


        # X label
        fig.text(0.5, -0.08, f'{NAMES[loss]} Scores', ha='center', fontsize=font_size)

        pdf.savefig(bbox_inches='tight', pad_inches = 0)


from .parameter_selection import THRESHOLDS
def ccr_table(args, scores, gt):

    def nonemax(a,b):
        b = numpy.array([v if v is not None else numpy.nan for v in b])
        return numpy.where(numpy.logical_and(numpy.logical_not(numpy.isnan(b)), b >= a), b, a)
    for protocol in args.protocols:
        latex_file = args.tables.format(protocol, 'best' if args.use_best else 'last')
        print("Writing CCR tables for protocol", protocol, "to file", latex_file)
        # compute all CCR values and store maximum values
        results = collections.defaultdict(dict)
        max_total = numpy.zeros(len(args.fpr_thresholds))
        max_by_alg = collections.defaultdict(lambda:numpy.zeros(len(args.fpr_thresholds)))
        max_by_loss = collections.defaultdict(lambda:numpy.zeros(len(args.fpr_thresholds)))
        for algorithm in args.algorithms:
            for loss in args.losses:
                ccrs = openset_imagenet.util.ccr_at_fpr(gt[protocol], scores[protocol][loss][algorithm], args.fpr_thresholds, unk_label=-2)
                results[algorithm][loss] = ccrs
                max_total = nonemax(max_total, ccrs)
                max_by_alg[algorithm] = nonemax(max_by_alg[algorithm], ccrs)
                max_by_loss[loss] = nonemax(max_by_loss[loss], ccrs)


        with open(latex_file, "w") as f:
            # write header
            f.write("\\bf Algorithm & \\bf Loss & ")
            f.write(" & ".join([THRESHOLDS[t] for t in args.fpr_thresholds]))
            f.write("\\\\\\hline\\hline\n")
            for algorithm in args.algorithms:
                f.write(f"\\multirow{{{len(args.losses)}}}{{*}}{{{NAMES[algorithm]}}}")
                for loss in args.losses:
                    f.write(f" & {NAMES[loss]}")
                    for i, v in enumerate(results[algorithm][loss]):
                        if v is None: f.write(" &")
                        elif v == max_total[i]: f.write(f" & \\textcolor{{blue}}{{\\bf {v:.4f}}}")
                        elif v == max_by_alg[algorithm][i]: f.write(f" & \\it {v:.4f}")
                        elif v == max_by_loss[loss][i]: f.write(f" & \\underline{{{v:.4f}}}")
                        else: f.write(f" & {v:.4f}")
                    f.write("\\\\\n")
                f.write("\\hline\n")


def main(command_line_arguments = None):
    args = command_line_options(command_line_arguments)
    cfg = openset_imagenet.util.load_yaml(args.configuration)

    msg_format = "{time:DD_MM_HH:mm} {name} {level}: {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO", "format": msg_format}])
#    logger.add(
#        sink = sys.stdout,
#        format=msg_format,
#        level="INFO")

    print("Extracting and loading scores")
    scores, ground_truths = load_scores(args, cfg)
    training_scores = load_training_scores(args, cfg)

    print("Writing file", args.plots)
    pdf = PdfPages(args.plots)
    try:
        # plot OSCR (actually not required for best case)
        print("Plotting OSCR curves, and CCR and FPR curves")
        plot_OSCR(args, scores, ground_truths)
        pdf.savefig(bbox_inches='tight', pad_inches = 0)

        plot_CCR_FPR(args, scores, ground_truths, pdf)

        """
        if not args.linear and not args.use_best and not args.sort_by_loss:
          # plot confidences
          print("Plotting confidence plots")
          plot_confidences(args)
          pdf.savefig(bbox_inches='tight', pad_inches = 0)
        """

        # plot histograms
        print("Plotting score distribution histograms")
        plot_score_distributions(args, scores, ground_truths, pdf)

        # plot training scores
        print("Plotting training metrics")
        plot_training_scores(args, training_scores, pdf)
    finally:
        pdf.close()

    # create result table
    print("Creating CCR Tables")
    ccr_table(args, scores, ground_truths)
