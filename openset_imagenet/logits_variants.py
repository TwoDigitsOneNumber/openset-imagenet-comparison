import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Linear(nn.Module):
    """Wrapper for compatibility of second argument of forward function."""
    def __init__(self, in_features, out_features, logit_bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.logit_bias = logit_bias

        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)
        # self.logits = nn.Linear(
        #     in_features=self.in_features,
        #     out_features=self.out_features,
        #     bias=self.logit_bias)

    def forward(self, features, labels):
        logits = features.mm(self.w)
        return logits


class SphereFace(nn.Module):
    """Taken from: https://github.com/ydwen/opensphere/blob/main/model/head/sphereface.py and adapted.
       reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
       It also used characteristic gradient detachment tricks proposed in
       <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, in_features, out_features, logit_bias, s=30., m=1.5):
        """logit_bias argument soley for compatibility."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)

    def forward(self, features, labels):
        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(features, dim=1).mm(self.w)

        # # their (maybe incorrect) version
        # with torch.no_grad():
        #     m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
        #     if labels is None:  # forward pass at test time
        #         # mathematically equivalent to setting m=1 and k=0.
        #         # this way avoids rewriting scatter_ without the use of label.
        #         d_theta = torch.zeros_like(cos_theta)
        #     else:
        #         m_theta.scatter_(1, labels.view(-1, 1), self.m, reduce='multiply')
        #         k = (m_theta / math.pi).floor()
        #         sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
        #         phi_theta = sign * torch.cos(m_theta) - 2. * k
        #         d_theta = phi_theta - cos_theta

        # # hard feature normalization (using self.s)
        # logits = self.s * (cos_theta + d_theta)
        # return logits
        
        # my (supposedly correct) version:
        with torch.no_grad():
            m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
            if labels is not None:  # training time forward pass
                m_theta.scatter_(1, labels.view(-1, 1), self.m, reduce='multiply')
                k = (m_theta / math.pi).floor()
                sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
                phi_theta = sign * torch.cos(m_theta) - 2. * k
                cos_theta.scatter_(1, labels.view(-1, 1), phi_theta) 
            # else just use cos_theta, i.e., pass no margin (m=1) and cnange no sign (k=0). In practice this means just skipping the above if statement

        # hard feature normalization (using self.s)
        logits = self.s * cos_theta
        return logits


class CosFace(nn.Module):
    """
        Taken from https://github.com/ydwen/opensphere/blob/main/model/head/cosface.py and adapted.
        reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
        reference2: <Additive Margin Softmax for Face Verification>
    """
    def __init__(self, in_features, out_features, logit_bias, s=64., m=0.35):
        """logit_bias argument soley for compatibility."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)

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
    """
    def __init__(self, in_features, out_features, logit_bias, s=64., m=0.5):
        """logit_bias argument soley for compatibility."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_normal_(self.w)

    def forward(self, features, labels):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(features, dim=1).mm(self.w)

        # # their version (should be correct)
        # with torch.no_grad():
        #     if labels is None:  # testing time forward pass
        #         d_theta = torch.zeros_like(cos_theta)
        #     else:  # training time forward pass
        #         theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
        #         theta_m.scatter_(1, labels.view(-1, 1), self.m, reduce='add')
        #         theta_m.clamp_(1e-5, math.pi)
        #         d_theta = torch.cos(theta_m) - cos_theta

        # logits = self.s * (cos_theta + d_theta)
        # return logits

        # my version (no unnecessary tensor initialization for d_theta)
        with torch.no_grad():
            if labels is not None:  # training time forward pass
                theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
                theta_m.scatter_(1, labels.view(-1, 1), self.m, reduce='add')
                theta_m.clamp_(1e-5, math.pi)
                cos_theta = torch.cos(theta_m)
            # else just use cos_theta, i.e., pass no margin (m=0). In practice this means just skipping the above if statement
        logits = self.s * cos_theta
        return logits


class MagFace(nn.Module):
    """
    Inspired by https://github.com/IrvingMeng/MagFace but mostly adapted from ArcFace implementation.
    MagFace Loss.
    """
    def __init__(self, in_features, out_features, logit_bias, s=64., l_a=10, u_a=110, l_m=.4, u_m=.8):
        """logit_bias argument soley for compatibility."""
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

    def _margin(self, a):
        """compute adaptive margin m(a_i) but for all vectors simultaneously. a is vector of norms of feature vectors. """
        margin = (a-self.l_a) * (self.u_m-self.l_m) / (self.u_a-self.l_a) + self.l_m
        return margin

    def forward(self, features, labels):

        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(features, dim=1).mm(self.w)
        with torch.no_grad():
            if labels is not None:  # training time forward pass
                # compute norms of feature vectors
                a = torch.linalg.norm(features, ord=2, dim=1)

                theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
                theta_m.scatter_(1, labels.view(-1, 1), self._margin(a).view(-1,1), reduce='add')
                theta_m.clamp_(1e-5, math.pi)
                cos_theta = torch.cos(theta_m)

        #     # else just use cos_theta, i.e., pass no margin (m(a_i)=0 for all a_i). In practice this means just skipping the above if statement
        logits = self.s * cos_theta
        return logits
