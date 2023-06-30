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


# class EntropicOpensetLoss:
#     """ Taken from vast, modified to accept mini batches without positive examples."""
#     def __init__(self, num_of_classes, unk_weight=1, known_weights=None):
#         self.class_count = num_of_classes
#         self.eye = tools.device(torch.eye(self.class_count))
#         self.unknowns_multiplier = unk_weight / self.class_count
#         self.ones = tools.device(torch.ones(self.class_count)) * self.unknowns_multiplier
#         self.cross_entropy = torch.nn.CrossEntropyLoss(weight=known_weights)
#         # self.unk_weight = unk_weight
#         # self.known_weights = known_weights

#     def __call__(self, logits, targets):
#         categorical_targets = tools.device(torch.zeros(logits.shape))
#         unk_idx = targets < 0
#         kn_idx = ~unk_idx
#         # check if there are known samples in the batch
#         if torch.any(kn_idx):
#             categorical_targets[kn_idx, :] = self.eye[targets[kn_idx]]

#         categorical_targets[unk_idx, :] = (
#             self.ones.expand(
#                 torch.sum(unk_idx).item(), self.class_count
#             )
#         )
#         return self.cross_entropy(logits, categorical_targets)


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


class RingLoss:
    """Computes ring loss"""
    def __init__(self, xi):
        self.xi = xi

    def __call__(self, logits, targets, features):
        # compute feature magnitudes a
        a = torch.linalg.norm(features, ord=2, dim=1)
        return torch.mean(torch.square(self.xi-a))


class ObjectosphereLoss:
    """Computes only the objectosphere loss, i.e.: (J_R - J_E) / lambda. See equation (2) in "Reducing Network Agnostophobia (2018)" by Akshay Raj Dhamija, Manuel Günther, Terrance E. Boult.

    params:
        xi (float): target lower boundary of feature magnitude for knowns.
        symmetric (bool): if symmetric, penalize deviation of magnitude to xi of knowns symmetrically, not via max.
    """
    def __init__(self, xi, symmetric=False):
        self.xi = xi
        self.symmetric = symmetric

    def __call__(self, logits, targets, features):
        # distinguish knowns from negatives/unknowns (via boolean mask)
        unk_idx = targets < 0
        kn_idx = ~unk_idx

        # compute feature magnitudes a
        a = torch.linalg.norm(features, ord=2, dim=1)

        # compute loss accordingly
        if self.symmetric:
            error_knowns = torch.square(self.xi-a[kn_idx])
        else:
            error_knowns = torch.square(torch.maximum(self.xi-a[kn_idx], torch.zeros_like(a[kn_idx]))) 
        error_unknowns = torch.square(a[unk_idx])

        # reduce via mean and then return
        return torch.mean(torch.cat((error_knowns, error_unknowns)))


class objectoSphere_loss:
    """taken (and slightly adapted for compatibility) from https://github.com/Vastlab/vast/blob/main/vast/losses/losses.py. identical results compared to ObjectosphereLoss."""
    def __init__(self, xi=50.0):
        self.knownsMinimumMag = xi

    def __call__(self, logits, target, features, sample_weights=None):
        # compute feature magnitude
        mag = features.norm(p=2, dim=1)
        # For knowns we want a certain magnitude
        mag_diff_from_ring = torch.clamp(self.knownsMinimumMag - mag, min=0.0)

        # Loss per sample
        loss = tools.device(torch.zeros(features.shape[0]))
        known_indexes = target != -1
        unknown_indexes = ~known_indexes
        # knowns: punish if magnitude is inside of ring
        loss[known_indexes] = mag_diff_from_ring[known_indexes]
        # unknowns: punish any magnitude
        loss[unknown_indexes] = mag[unknown_indexes]
        loss = torch.pow(loss, 2)
        if sample_weights is not None:
            loss = sample_weights * loss
        return torch.mean(loss)


class JointLoss:
    """
    Loss that when called appends loss_2 to loss_1 and weights loss_2 with lmbd.

    loss functions loss_i (i={1,2}) get combined with loss_1 as follows: loss_1(logits, targets) + lmbd * loss_2(logits, targets, features). 
    It must be able to be called via: loss_i(logits, targets). If it needs the features as additional argument it must be called as loss_i(logits, targets, features) and you must set loss_i_requires_features=True.

    params:
        loss_1 (function): 
        loss_2 (function): 
        lambda_os (float): weight for the objectosphere loss
        xi (float): target lower boundary of feature magnitude for knowns.
        loss_1_requires_features (bool): set true if loss_1 requires the features as input.
        loss_2_requires_features (bool): set true if loss_2 requires the features as input.
    """
    def __init__(self, loss_1, loss_2, lmbd, loss_1_requires_features=False, loss_2_requires_features=False):
        self.loss_1 = loss_1
        self.loss_2 = loss_2
        self.lmbd = lmbd
        self.loss_1_requires_features = loss_1_requires_features
        self.loss_2_requires_features = loss_2_requires_features

    def __call__(self, logits, targets, features):
        if self.loss_1_requires_features and self.loss_2_requires_features:
            return self.loss_1(logits, targets, features) + \
                self.lmbd * self.loss_2(logits, targets, features)

        elif self.loss_1_requires_features and not self.loss_2_requires_features:
            return self.loss_1(logits, targets, features) + \
                self.lmbd * self.loss_2(logits, targets)

        elif not self.loss_1_requires_features and self.loss_2_requires_features:
            return self.loss_1(logits, targets) + \
                self.lmbd * self.loss_2(logits, targets, features)

        elif not self.loss_1_requires_features and not self.loss_2_requires_features:
            return self.loss_1(logits, targets) + \
                self.lmbd * self.loss_2(logits, targets)


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
