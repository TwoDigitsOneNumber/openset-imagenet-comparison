# Notes on the fork

The loss functions names used in the files and used as command line arguments are:
'softmax', 'entropic', 'objectosphere',               	# benchmarks
'sphereface', 'cosface', 'arcface',  					# face losses (HFN)
'norm_sfn', 'cosface_sfn', 'arcface_sfn',  				# SFN-Margin losses
'softmax_os', 'cos_os', 'arc_os',  						# Margin-OS losses (SFN)
'norm_eos', 'cos_eos', 'arc_eos',  						# Margin-EOS losses (HFN)
**Note**: The implementation for Norm-OS is called softmax_os.

Below I provide a (non exhaustive) list of the most important changes made to the original code:
- Added new model class for the [LeNet++ network](/openset_imagenet/model.py) `LeNetBottleneck`
- Restructured how the model classes work:
    - Extended model "interface" with new arguments in the constructor, in particular: `loss_type`
    - Added new file [`logit_variants.py`](/openset_imagenet/logits_variants.py) which contains classes that implement various logits (i.e., `Linear`, `SphereFace`, `CosFace`, and `ArcFace`) as well as some generalized versions of the margin-based losses that run a bit slower but allow handling of negative samples (i.e., `CosineMargin`, `AngularMargin`, and `LogitMargin`). This file also includes a function `set_logits`, which instantiates the correct logit type given the loss type (e.g., `CosineMargin` logits for `loss_type == 'cos_eos'`).
    - Models now return logits, deep features, and optionally the angles of each feature to each class center during a forward pass. Note, this breaks compatibility of the code with the PROSER model.
    - Loss and protocol specific hyperparameters are automatically applied (see below).
- Added new loss functions `RingLoss` and `ObjectosphereLoss` (built from scratch and compared to `objectoSphere_loss` from [here](https://github.com/Vastlab/vast/blob/main/vast/losses/losses.py)) as well as a wrapper to compute joint losses (`JointLoss`).
- Added toy protocol:
    - Added dataset class `OSCToyDataset` for the toy protocol in [`dataset.py`](/openset_imagenet/dataset.py) as subclass of `ImagenetDataset`. It is designed to reuse as much of the code of the parent class as possible, to guarantee compatible behavior in all scripts.
    - Due to the implementation as subclass, I added a dataset generation script [`OSCToyDataGeneration.py`](/openset_imagenet/script/OSCToyDataGeneration.py) which creates the partitions required for the toy protocol and mirrors the ImageNet protocol structure to guarantee compatibility with the existing code and minimize code duplication.
    - The toy protocol for preliminary experiments ($K=C$) is considered `p0` and the toy protocol for visualization ($K=2$) is considered `p10`. They work exactly the same apart from the deep feature dimension, but considering them as two protocols simplified many things, e.g., storing results/plots and choosing different hyperparameters.
- Separated hyperparameters from the files and moved them to dedicated protocol specific files, e.g., [`p2_hyperparameters.yaml`](/config/p2_hyperparameters.yaml). This allows to change hyperparameters specific for each loss function and protocol in a single location. All parameters then are applied to training and evaluation where relevant. 
- Modified training script [`train.py`](/openset_imagenet/train.py):
    - Training now categorizes loss functions based on various factors (e.g., does it handle negative samples? Does it require the deep features as inputs? etc.) such that the loss type must only be categorized into the respective lists in the beginning of the file, to reduce likelihood of introducing small mistakes (e.g., forget to remove negatives from the training data for certain loss functions).
    - The only thing to adapt in the training file (except the categorization) are new losses, which are all bundeled together and apply the hyperparameters for the specific loss functions and protocols.
    - Now many "training scores" (e.g., training and validation loss, and validation confidences) get stored for visualizations (see `write_training_scores` and `get_arrays`)
- During evaluation in [`evaluate_algs.py`](/openset_imagenet/script/evaluate_algs.py) more information gets stored for visualizations, most notably, the angles of test deep features to all class centers.
- Added many new plots for diagnosing training and behavior of the various loss functions to the [`plot_all.py`](/openset_imagenet/script/plot_all.py) script including many helper functions in [`util.py`](/openset_imagenet/util.py). New plots include: `plot_training_metrics`, `plot_feature_magnitudes`, `plot_angle_distributions`, `plot_CCR_FPR`, and `plot_feature_distributions`.
- Added a new script [`plot_thesis.py`](/openset_imagenet/script/plot_thesis.py) (copied and adapted from [`plot_all.py`](/openset_imagenet/script/plot_all.py)) to generate plots specificly to be used in the thesis. This script places all figures and tables into three folders in `/results`: `deep_features`, `performance_tables`, and `performance_plots`. The script is called from the active conda environment (see below) via
```python
python openset_imagenet/script/plot_thesis.py
```
- The repository also contains a file [`test.ipynb`](/test.ipynb) which was used by me to sporatically analyze certain results or verify implementations. It is also used to compare the datasets MNIST and EMNITS and create the corresponding figures used in the thesis. While this is only of minor importance I choose to leave it in for completeness sake.


All information below is from the original code base and unaltered in its meaning.

# Open set on ImageNet
Implementation of the experiments performed in Large-Scale Open-Set Classification Protocols for ImageNet, which has been accepted for publication in WACV 2023.
You can find a [pre-print of the paper including our supplemental material on arXiv](https://arxiv.org/abs/2210.06789).
If you make use of our evaluation protocols or this implementation, please cite the following paper:

    @inproceedings{palechor2023openset,
        author       = {Palechor Anacona, Jesus Andres and Bhoumik, Annesha and G\"unther, Manuel},
        booktitle    = {Winter Conference on Applications of Computer Vision (WACV)},
        title        = {Large-Scale Open-Set Classification Protocols for {ImageNet}},
        year         = {2023},
        organization = {IEEE}
    }

## LICENSE
This code package is open-source based on the BSD license.
Please see `LICENSE` for details.

## Data

All scripts rely on the ImageNet dataset using the ILSVRC 2012 data.
If you do not have a copy yet, it can be downloaded from Kaggle (untested): https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview.
The protocols rely on the `robustness` library, which in turn relies on some files that have been distributed with the ImageNet dataset some time ago, but they are not available anymore.
With a bit of luck, you can find the files somewhere online:

* imagenet_class_index.json
* wordnet.is_a.txt
* words.txt

If not, you can also rely on the pre-computed protocol files, which can be found in the provided `protocols.zip` file and extracted via:

    unzip protocols.zip


## Setup

We provide a conda installation script to install all the dependencies.
Please run:

    conda env create -f environment.yaml

Afterward, activate the environment via:

    conda activate openset-imagenet-comparison

## Scripts

The directory `openset_imagenet/script` includes several scripts, which are automatically installed and runnable.

### Protocols

You can generate the protocol files using the command `imagenet_protocols.py`.
Please refer to its help for details:

    protocols_imagenet.py --help

Basically, you have to provide the original directory for your ImageNet images, and the directory containing the files for the `robustness` library.
The other options should be changed rarely.

### Training of one base model

The training can be performed using the `train_imagenet.py` script.
It relies on a configuration file as can be found in `config/threshold.yaml`.
Please set all parameters as required (the default values are as used in the paper), and run:

    train_imagenet.py [config] [protocol] -g GPU

where `[config]` is the configuration file, `[protocol]` one of the three protocols.
The `-g` option can be used to specify that the training should be performed on the GPU (**highly recommended**), and you can also specify a GPU index in case you have several GPUs at your disposal.

### Running different algorithms using that model

The other algorithms (EVM, OpenMax, PROSER) can be executed with exactly the same `train_imagenet.py` script.
Simply provide another configuration file from the `config/` directory.
Again, you might want to adapt some parameters in those configuration files, but they are all set according to the results in the paper.

.. note::
   Please make sure that you have run the base model training before executing other algorithms.

### Training of all the models with all of the algorithms in the paper

The `train_imagenet_all.py` script provides a shortcut to train a model with three different loss functions on three different protocols.
It relies on the same configuration files from the `config/` directory where some parts are modified during execution, and read the config files with the names according to the `--algorithms`.
You can run:

    train_imagenet_all.py -g [list-of-gpus]

where `[config]` is the directory containing all configuration files, which is by default `config/`.
You can also select some of the `--protocols` to run on, as well as some of the `--loss-functions`, and some of the `--algorithms`.
The `-g` option can take several GPU indexes, and trainings will be executed in parallel if more than one GPU index is specified.
In case the training stops early for unknown reasons, you can safely use the `--continue` option to continue training from the last epoch -- this option also works for the PROSER training.

When you have a single GPU available, start the script and book a trip to Hawaii, results will finish in about a week.
The more GPUs you can spare, the faster the training will end.
However, make sure that the `threshold` algorithm is always executed first, maybe by running:

    train_imagenet_all.py -g [list-of-gpus] --algorithms threshold
    train_imagenet_all.py -g [list-of-gpus] --algorithms openmax evm proser

### Parameter Optimization

Some of our algorithms will require to adapt the parameters to the different loss functions and protocols.
Particularly, EVM and OpenMax have a set of parameters that should be optimized.
Due to the nature of the algorithms, the `train_imagenet_all.py` script has already trained and saved all parameter combinations as provided in the configuration files of these two algorithms, here the task is only to evaluate the algorithms on unseen data.
Naturally, we will make use of the known and the negative samples of the validation set to perform the parameter optimization.

The parameter optimization will be done via the `parameter_selection.py` script.
It will read the configuration files of the EVM and OpenMax algorithms, load the images from the validation set, extract features with all trained base networks (as given by the `--losses` parameter), and evaluate the different parameter settings of the algorithms.
Particularly, the CCR values at various FPR thresholds will be computed.
Depending on the protocol, this might require several minutes to hours.
Finally, it will write a separate LaTeX table file per protocol/algorithm/loss combination, and summary LaTeX tables including the best parameters for each algorithm.

.. note::
   Note that also the PROSER algorithm has parameters which we might want to optimize.
   However, since this would require a complete network finetuning for each parameter/protocol/algorithm combination, we do not include PROSER in this script.

The optimized parameters should also be transferred into the `config/test.yaml`, we have done this already for you.

### Evaluation

In order to evaluate all models on the test sets, you can make use of the `evaluate_imagenet.py` script.
This script will use all trained models (as resulting from the `train_imagenet_all.py` script) and extract the features, logits and scores for the test set.
A detailed list of algorithms and parameters is read from the `config/test.yaml` file, which is the default for the `--configuration` option of the `evaluate_imagenet.py` script.
Any model that has not been trained will automatically be skipped.
Otherwise, you can restrict the numbers of `--losses` and `--algorithms`, as well as selecting single `--protocols`.
It is also recommended to run feature extraction on a `--gpu`.
For more options and details on the options, please refer to:

    evaluate_imagenet.py --help

### Plotting

Finally, the `plot_imagenet.py` script can be used to create the plots and result tables from the test set as we have them in the paper.
The script will take information from the same `config/test.yaml` configuration file and make use of all results are generated by the `evaluate_imagenet.py` script.
It will plot all results into a single PDF file (`Results_last.pdf` by default), containing multiple pages.
Page 1 will display all OSCR plots for all algorithms applied to networks trained with all loss functions, where both negative and unknown samples are evaluated for each of the three protocols.
The following three pages will contain score distribution plots of the different algorithms (excluding MaxLogits), separated for the three loss functions.

Again, results that do not exist are skipped automatically.
Since the list of algorithms and loss functions will make the plot very busy, you can try to sub-select several `--losses`, `--algorithms`, or `--protocols` to reduce the number of lines in the plots.

You can also modify other parameters, see:

    plot_imagenet.py --help

Additionally, the script will produce three tables, one for each protocol, where the CCR values at various FPR values are tabularized, for an easier comparison and reference.

## Getting help

In case of trouble, feel free to contact us under [siebenkopf@googlemail.com](mailto:siebenkopf@googlemail.com?subject=Open-Set%20ImageNet%20Protocols)
