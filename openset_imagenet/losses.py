""" Code taken from the vast library https://github.com/Vastlab/vast"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vast import tools
import math


class EntropicOpensetLoss:
    """ Taken from vast, modified to accept mini batches without positive examples."""
    def __init__(self, num_of_classes, fc_layer_dim, out_features, logit_bias=True, unk_weight=1):
        """
        args:
            num_of_classes:
            fc_layer_dim: dimensionality of features, i.e., size of deep feature layer
            out_features: logits dimension
            logit_bias(bool): True to use bias term in the logits layer.
            unk_weight:
        """
        self.class_count = num_of_classes
        self.eye = tools.device(torch.eye(self.class_count))
        self.unknowns_multiplier = unk_weight / self.class_count
        self.ones = tools.device(torch.ones(self.class_count)) * self.unknowns_multiplier
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.logit_layer = nn.Linear(
            in_features=fc_layer_dim,
            out_features=out_features,
            bias=logit_bias)


    def __call__(self, features, targets):
        logits = self.logit_layer(features)

        categorical_targets = tools.device(torch.zeros(logits.shape))
        unk_idx = targets < 0
        kn_idx = ~unk_idx
        # check if there is known samples in the batch
        if torch.any(kn_idx):
            categorical_targets[kn_idx, :] = self.eye[targets[kn_idx]]

        categorical_targets[unk_idx, :] = (
            self.ones.expand(
                torch.sum(unk_idx).item(), self.class_count
            )
        )
        return self.cross_entropy(logits, categorical_targets)



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


# from: https://github.com/ydwen/opensphere/blob/032c31b8918fb7639d3b34ac7433bfd537c6c518/model/head/arcface.py but adapted
class ArcFace:
    """ reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, feat_dim, num_class, s=64., m=0.5):
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def __call__(self, features, targets):
        """computes logits and loss"""
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(features, dim=1).mm(self.w)
        with torch.no_grad():
            theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
            theta_m.scatter_(1, targets.view(-1, 1), self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, targets)

        return loss


# from: https://github.com/ydwen/opensphere/blob/032c31b8918fb7639d3b34ac7433bfd537c6c518/model/head/cosface.py but adapted
class CosFace:
    """reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
       reference2: <Additive Margin Softmax for Face Verification>
    """
    def __init__(self, feat_dim, num_class, s=64., m=0.35):
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def __call__(self, features, targets):
        """computes logits and loss"""
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(features, dim=1).mm(self.w)
        with torch.no_grad():
            d_theta = torch.zeros_like(cos_theta)
            d_theta.scatter_(1, targets.view(-1, 1), -self.m, reduce='add')

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, targets)

        return loss


# from: https://github.com/ydwen/opensphere/blob/032c31b8918fb7639d3b34ac7433bfd537c6c518/model/head/sphereface.py but adapted
class SphereFace:
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
        It also used characteristic gradient detachment tricks proposed in
        <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, feat_dim, num_class, s=30., m=1.5):
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def __call__(self, features, targets):
        """computes logits and loss"""
        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(features, dim=1).mm(self.w)
        with torch.no_grad():
            m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
            m_theta.scatter_(
                1, targets.view(-1, 1), self.m, reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - cos_theta

        logits = self.s * (cos_theta + d_theta)
        loss = F.cross_entropy(logits, targets)

        return loss



