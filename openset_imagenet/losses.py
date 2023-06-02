""" Code taken from the vast library https://github.com/Vastlab/vast"""
from torch.nn import functional as F
import torch
import numpy as np
from vast import tools


class EntropicOpensetLoss:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, unk_weight=1):
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.unknowns_multiplier = unk_weight / self.class_count
        self.ones = tools.device(torch.ones(self.class_count)) * self.unknowns_multiplier
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, targets):
        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = targets < 0
        kn_idx = ~unk_idx
        # check if there are known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[targets[kn_idx]]

        categorical_targets[unk_idx, :] = (
            self.ones.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )
        return self.cross_entropy(logits, categorical_targets)


class MagFaceLoss:
    def __init__(self, lambda_g=35, u_a=110):
        self.lambda_g = lambda_g
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.u_a = u_a

    def calc_loss_G(self, a):
        """a: vector of norms of feature vectors, dim: Nx1, returns float"""
        g = a/(self.u_a**2) + 1/(a)
        return torch.mean(g)

    def __call__(self, logits, targets, features):
        a = torch.linalg.norm(features, ord=2, dim=1)

        return self.cross_entropy(logits, targets) + self.lambda_g * self.calc_loss_G(a)


class ObjectosphereLoss:
    """Computes only the objectosphere loss, i.e.: (J_R - J_E) / lambda. See equation (2) in "Reducing Network Agnostophobia (2018)" by Akshay Raj Dhamija, Manuel Günther, Terrance E. Boult.

    params:
        xi (float): target lower boundary of feature magnitude for knowns.
    """
    def __init__(self, xi):
        self.xi = xi

    def __call__(self, logits, targets, features):
        # distinguish knowns from negatives/unknowns (via boolean mask)
        unk_idx = targets < 0
        kn_idx = ~unk_idx

        # compute feature magnitudes a
        a = torch.linalg.norm(features, ord=2, dim=1)

        # compute loss accordingly
        error_knowns = torch.square(torch.maximum(self.xi-a[kn_idx], torch.zeros_like(a[kn_idx]))) 
        error_unknowns = torch.square(a[unk_idx])

        # reduce via mean and then return
        return torch.mean(torch.cat((error_knowns, error_unknowns)))


class ObjectosphereLossWrapper:
    """
    Loss that when called appends the objectosphere loss to the input loss.

    params:
        prepended_loss (function): 
            loss function (f) that gets combined with objectosphere loss (o) as follows: prepended_loss(logits, targets) + lambda_os * objecto_sphere(logits, targets, features). 
            It must be able to be called via: prepended_loss(logits, targets). If it needs the features as additional argument it must be called as prepended_loss(logits, targets, features) and you must set prepended_loss_requires_features=True.
        lambda_os (float): weight for the objectosphere loss
        xi (float): target lower boundary of feature magnitude for knowns.
        prepended_loss_requires_features (bool): set true if prepended_loss requires the features as input.
    """
    def __init__(self, prepended_loss, lambda_os, xi, prepended_loss_requires_features=False):
        self.prepended_loss = prepended_loss
        self.lambda_os = lambda_os
        self.xi = xi
        self.prepended_loss_requires_features = prepended_loss_requires_features
        self.objectosphere_loss = ObjectosphereLoss(self.xi)

    def __call__(self, logits, targets, features):
        if self.prepended_loss_requires_features:
            return self.prepended_loss(logits, targets, features) + self.lambda_os * self.objectosphere_loss(logits, targets, features)
        else:
            return self.prepended_loss(logits, targets) + self.lambda_os * self.objectosphere_loss(logits, targets, features)


class AverageMeter(object):
    """ Computes and stores the average and current value. Taken from
    https://github.com/pytorch/examples/tree/master/imagenet
    """
    def __init__(self):
        self.val, self.avg, self.sum, self.count = None, None, None, None
        self.reset()

    def reset(self):
        """ Sets all values to 0. """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        """ Update metric values.

        Args:
            val (flat): Current value.
            count (int): Number of samples represented by val. Defaults to 1.
        """
        # reason for avoiding update if val=nan:
        # when training CosOS we ignore index -1 and in certain batches it happens that all indices are -1 resulting in loss=nan. This issue is unavoidable for this implementation and this workaround prevents the average training loss from being nan if only a few batch losses are nan
        if not np.isnan(val):
            self.val = val
            self.sum += val * count
            self.count += count
            self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.avg:3.3f}"


# Taken from:
# https://github.com/Lance0218/Pytorch-DistributedDataParallel-Training-Tricks/
class EarlyStopping:
    """ Stops the training if validation loss/metrics doesn't improve after a given patience"""
    def __init__(self, patience=100, delta=0):
        """
        Args:
            patience(int): How long wait after last time validation loss improved. Default: 100
            delta(float): Minimum change in the monitored quantity to qualify as an improvement
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        if loss is True:
            score = -metrics
        else:
            score = metrics

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
