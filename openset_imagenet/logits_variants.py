import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def set_logits(loss_type, in_features, out_features, logit_bias):
    """return logits based on the specified loss type
    
    parameters:
        loss_type (str): name of loss used, this determines which type of logits to use
        in_features (int): dimension of deep features, i.e., dimension of input for the logits 
        out_features (int): output dimension of the logits
        logit_bias (bool): currently only here for compatibility, has no effect

        **kwargs: add other logit_variant specific parameters as keyword arguments. If not specified, the defaults of the logit class are used. Can be set to multiple arguments even if not all are available for all logits, e.g., when running 'cosface' and 'cosos' in parallel, can set 's' (which affects both since the parameter is called the same) and can also set 'variable_magnitude_during_testing' (which is only available for 'cosos' but does not break the call for 'cosface').
            -> if a parameter should be set differently for different logit_variants, then the parameter must be renamed in the respective logit class to a distinct name
    """

    logit_map = {
        'linear': Linear,
        'sphereface': SphereFace,
        'cosface': CosFace,
        'arcface': ArcFace,
        'magface': MagFace,
        'cosos': CosineMargin
    }

    # if loss_type not in ["sphereface", "cosface", "arcface", "magface", 'cosos-f', 'cosos-v', 'cosos-m', 'coseos']:

    # use linear logits for softmax and entropic loss
    if loss_type in ['softmax', 'entropic']:
        loss_type = 'linear'



    # ==================================================
    # ============== SET LOGIT PARAMETERS ==============
    # ==================================================
    # set parameters for the logits to use during training




    s = 64

    # pick appart type from its variant
    if '-' in loss_type:
        loss_type, variant = loss_type.split('-')

    # handle CosOS variants
    variable_magnitude_during_testing = True
    if loss_type == 'cosos':
        if variant == 'v':
            s = None
            variable_magnitude_during_testing = True
        elif variant == 'f':
            variable_magnitude_during_testing = False
        elif variant == 'm':
            variable_magnitude_during_testing = True
        else:
            raise ValueError(f"cosos-{variant} is not a valid option!")




    # ==================================================
    # ==================================================
    # ==================================================



    return logit_map[loss_type](
        in_features=in_features, 
        out_features=out_features, 
        logit_bias=logit_bias,
        s = s,
        variable_magnitude_during_testing = variable_magnitude_during_testing
    )


class Linear(nn.Module):
    """Wrapper for compatibility of second argument of forward function."""
    def __init__(self, in_features, out_features, logit_bias, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)
        print('Using Linear logits')

    def forward(self, features, labels):
        logits = features.mm(self.w)
        return logits


class SphereFace(nn.Module):
    """Taken from: https://github.com/ydwen/opensphere/blob/main/model/head/sphereface.py and adapted.
       reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
       It also used characteristic gradient detachment tricks proposed in
       <SphereFace Revived: Unifying Hyperspherical Face Recognition>.

        logit_bias argument in constructor soley for compatibility.
    """
    def __init__(self, in_features, out_features, logit_bias, s=30., m=1.5, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)
        print('Using SphereFace logits')

    def forward(self, features, labels):
        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(features, dim=1).mm(self.w)

        # their (maybe incorrect) version
        with torch.no_grad():
            if labels is None:  # forward pass at test time
                # mathematically equivalent to setting m=1 and k=0.
                # this way avoids rewriting scatter_ without the use of label.
                d_theta = torch.zeros_like(cos_theta)
            else:
                m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
                m_theta.scatter_(1, labels.view(-1, 1), self.m, reduce='multiply')
                k = (m_theta / math.pi).floor()
                sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
                phi_theta = sign * torch.cos(m_theta) - 2. * k
                d_theta = phi_theta - cos_theta

        # hard feature normalization (using self.s)
        logits = self.s * (cos_theta + d_theta)
        return logits
        
        # # my (supposedly correct) version:
        # with torch.no_grad():
        #     m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
        #     if labels is not None:  # training time forward pass
        #         m_theta.scatter_(1, labels.view(-1, 1), self.m, reduce='multiply')
        #         k = (m_theta / math.pi).floor()
        #         sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
        #         phi_theta = sign * torch.cos(m_theta) - 2. * k
        #         cos_theta.scatter_(1, labels.view(-1, 1), phi_theta) 
        #     # else just use cos_theta, i.e., pass no margin (m=1) and cnange no sign (k=0). In practice this means just skipping the above if statement

        # # hard feature normalization (using self.s)
        # logits = self.s * cos_theta
        # return logits


class CosFace(nn.Module):
    """
        Taken from https://github.com/ydwen/opensphere/blob/main/model/head/cosface.py and adapted.
        reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
        reference2: <Additive Margin Softmax for Face Verification>

        logit_bias argument in constructor soley for compatibility.
    """
    def __init__(self, in_features, out_features, logit_bias, s=64., m=0.35, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)
        print('Using CosFace logits')

    def forward(self, features, labels):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(features, dim=1).mm(self.w)

        # # their version
        # with torch.no_grad():
        #     d_theta = torch.zeros_like(cos_theta)
        #     if labels is not None:  # training time forward pass
        #         d_theta.scatter_(1, labels.view(-1, 1), -self.m, reduce='add')
        # logits = self.s * (cos_theta + d_theta)
        # return logits
        
        # my version
        with torch.no_grad():
            if labels is not None:  # training time forward pass
                cos_theta.scatter_(1, labels.view(-1, 1), -self.m, reduce='add')

        logits = self.s * cos_theta
        return logits


class ArcFace(nn.Module):
    """
        Taken from https://github.com/ydwen/opensphere/blob/main/model/head/arcface.py and adapted.
        reference: <Additive Angular Margin Loss for Deep Face Recognition>

        logit_bias argument in constructor soley for compatibility.
    """
    def __init__(self, in_features, out_features, logit_bias, s=64., m=0.5, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)
        print('Using ArcFace logits')

    def forward(self, features, labels):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(features, dim=1).mm(self.w)

        # their version (should be correct)
        with torch.no_grad():
            if labels is None:  # testing time forward pass
                d_theta = torch.zeros_like(cos_theta)
            else:  # training time forward pass
                theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
                theta_m.scatter_(1, labels.view(-1, 1), self.m, reduce='add')
                theta_m.clamp_(1e-5, math.pi)
                d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)
        return logits

        # # my version (no unnecessary tensor initialization for d_theta)
        # with torch.no_grad():
        #     if labels is not None:  # training time forward pass
        #         theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
        #         theta_m.scatter_(1, labels.view(-1, 1), self.m, reduce='add')
        #         theta_m.clamp_(1e-5, math.pi)
        #         cos_theta = torch.cos(theta_m)
        #     # else just use cos_theta, i.e., pass no margin (m=0). In practice this means just skipping the above if statement
        # logits = self.s * cos_theta
        # return logits


class MagFace(nn.Module):
    """
    Inspired by https://github.com/IrvingMeng/MagFace but mostly adapted from ArcFace implementation.
    MagFace Loss.

        logit_bias argument in constructor soley for compatibility.
    """
    def __init__(self, in_features, out_features, logit_bias, s=64., l_a=10, u_a=110, l_m=.4, u_m=.8, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)
        print('Using MagFace logits')

    def _margin(self, a):
        """compute adaptive margin m(a_i) but for all vectors simultaneously. a is vector of norms of feature vectors. """
        margin = (a-self.l_a) * (self.u_m-self.l_m) / (self.u_a-self.l_a) + self.l_m
        return margin

    def forward(self, features, labels):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(features, dim=1).mm(self.w)

        with torch.no_grad():
            if labels is None:  # testing time forward pass
                d_theta = torch.zeros_like(cos_theta)
            else:  # training time forward pass

                # compute norms of feature vectors
                a = torch.linalg.norm(features, ord=2, dim=1)

                theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
                theta_m.scatter_(1, labels.view(-1, 1), self._margin(a).view(-1,1), reduce='add')
                theta_m.clamp_(1e-5, math.pi)
                d_theta = torch.cos(theta_m) - cos_theta

        #     # else just use cos_theta, i.e., pass no margin (m(a_i)=0 for all a_i). In practice this means just skipping the above if statement
        logits = self.s * (cos_theta + d_theta)
        return logits


class CosineMargin(nn.Module):
    """
        CosFace adapted with optional feature normalization and such that it ignores negative samples (data points with target < 0).

        Taken from https://github.com/ydwen/opensphere/blob/main/model/head/cosface.py and adapted.
        reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
        reference2: <Additive Margin Softmax for Face Verification>

    """
    def __init__(self, in_features, out_features, logit_bias, s=None, m=0.35, variable_magnitude_during_testing=False, **kwargs):
        """
        parameters:
            s (int): Feature magnitude in the deep feature space. For s=64 equal to CosFace. Use s=None for no feature normalization.
            logit_bias argument in constructor soley for compatibility, has no effect.
            variable_magnitude_during_testing (bool): lets feature magnitudes be variable during testing/validation, allows to keep feature magnitude fixed only during training. Only has an effect when s is not None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        self.variable_magnitude_during_testing = variable_magnitude_during_testing
        nn.init.xavier_normal_(self.w)
        print('Using CosineMargin logits')

    def forward(self, features, labels):
        """set labels to None during evaluation, i.e., testing time forward pass."""
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(features, dim=1).mm(self.w)  # \cos(\theta_{i,j})

        with torch.no_grad():
            if labels is not None:  # training time forward pass
                # distinguish knowns from negatives/unknowns (via boolean mask) and only add margin to knowns
                kn_idx = labels >= 0
                cos_theta[kn_idx,:].scatter_(1, labels[kn_idx].view(-1, 1), -self.m, reduce='add')

        # variable feature magnitude if 
        if (self.s is None) or (self.s is not None and labels is None and self.variable_magnitude_during_testing):
            a = torch.linalg.norm(features, ord=2, dim=1)
            logits = torch.mul(a.view(-1,1), cos_theta)
        else:  # fixed feature magnitude
            logits = self.s * cos_theta

        return logits

