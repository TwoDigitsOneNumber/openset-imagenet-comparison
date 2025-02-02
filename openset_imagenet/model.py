""" ResNet50, parts taken from VAST: https://github.com/Vastlab/vast/tree/main/vast/architectures"""
from torchvision import models
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torch
from torch import nn
import random
import numpy as np
import pathlib
import vast
from .logits_variants import set_logits

class ResNet50(nn.Module):
    """Represents a ResNet50 model"""

    def __init__(self, protocol, loss_type, fc_layer_dim=1000, out_features=1000, logit_bias=True):
        """ Builds a ResNet model, with deep features and logits layers.

        Args:
            loss_type(str): name of loss used, determines type of logits to compute.
            fc_layer_dim(int): Deep features dimension.
        """
        super(ResNet50, self).__init__()

        self.number_of_classes = out_features
        self.loss_type = loss_type

        # Change the dimension of out features
        self.resnet_base = models.resnet50(pretrained=False)
        fc_in_features = self.resnet_base.fc.in_features
        self.resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.logits, self.logit_type = set_logits(
            protocol=protocol,
            loss_type=self.loss_type,
            in_features=fc_layer_dim,
            out_features=out_features,
            logit_bias=False
        )


    def forward(self, image, labels, return_angles=False):
        """ Forward pass

        Args:
            image(tensor): Tensor with input samples
            labels(tensor): Labels for the input samples (needed for margin computation in 'face losses')
            return_angles(bool): set true to return angles of input data (i.e. its deep feature) and all class centers.

        Returns:
            Deep features of the samples.
        """
        deep_features = self.resnet_base(image)
        logits = self.logits(deep_features, labels)

        if return_angles:
            # compute angles
            with torch.no_grad():
                cos_theta = F.normalize(deep_features, dim=1).mm(F.normalize(self.logits.w, dim=0))
                angles = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
            return logits, deep_features, angles
        else:
            return logits, deep_features


class ResNet50Proser(nn.Module):
    """Implements functionality for the PROSER approach into ResNet50"""
    def __init__(self, dummy_count, fc_layer_dim, resnet_base, loss_type):
        super(ResNet50Proser, self).__init__()
        self.dummy_count = dummy_count
        # add a dummy classifier for unknown classes
        self.dummy_classifier = nn.Linear(fc_layer_dim, dummy_count)
        self.resnet_base = resnet_base
        self.loss_type=loss_type

    def first_blocks(self, x):
        """Calls the first three blocks of the model
        This repeats some functionality of the original ResNet implementation found here:
        https://github.com/pytorch/vision/blob/ad2eceabf0dcdb17a25d84da62492825a2c770a2/torchvision/models/resnet.py

        Note: for consistency reasons with the original source code of Zhou et al. (2021), the
        manifold mixup is performed after the third group of blocks (i.e. layer3). By following
        this approach, the manifold mixup is performed after the penultimate group/layer
        """

        x = self.resnet_base.resnet_base.conv1(x)
        x = self.resnet_base.resnet_base.bn1(x)
        x = self.resnet_base.resnet_base.relu(x)
        x = self.resnet_base.resnet_base.maxpool(x)

        x = self.resnet_base.resnet_base.layer1(x)
        x = self.resnet_base.resnet_base.layer2(x)
        x = self.resnet_base.resnet_base.layer3(x)


        return x

    def last_blocks(self, x):
        """Calls the last blocks of the model, and returns the deep features, the logits and the results of the dummy classifier
        This repeats some functionality of the original ResNet implementation found here:
        https://github.com/pytorch/vision/blob/ad2eceabf0dcdb17a25d84da62492825a2c770a2/torchvision/models/resnet.py

        Note: for consistency reasons with the original source code of Zhou et al. (2021), the
        manifold mixup is performed after the third group of blocks (i.e. layer3). By following
        this approach, the manifold mixup is performed after the penultimate group/layer
        """
        x = self.resnet_base.resnet_base.layer4(x)

        x = self.resnet_base.resnet_base.avgpool(x)

        x = torch.flatten(x, 1)

        features = self.resnet_base.resnet_base.fc(x)

        # apply our standard output layer
        logits = self.resnet_base.logits(features)

        if self.loss_type == "garbage":
            # for garbage class, we remove the logit for the unknown class -- since we will add another one below
            logits = logits[:,:-1]

        # apply our dummy layer, get only the maximum output
        dummy = torch.max(self.dummy_classifier(features), dim=1)[0]
        return logits, dummy, features

    def forward(self, image):
        """Extracts the logits, the dummy classiifers and the deep features for the given input """
        intermediate_features = self.first_blocks(image)
        return self.last_blocks(intermediate_features)


class LeNetBottleneck(nn.Module):
    """ Builds a LeNet model with deep features and logits, which includes a bottleneck in the deep features (deep_feature_dim). 

    parameters: see ResNet50 in this file.
    """

    def __init__(self, protocol, loss_type, deep_feature_dim=2, out_features=10, logit_bias=True):
        super(LeNetBottleneck, self).__init__()

        self.number_of_classes = out_features
        self.deep_feature_dim = deep_feature_dim
        self.loss_type = loss_type

        self.feature_extractor = nn.Sequential(            
            # 28x28 -> 14x14
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 14x14 -> 7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 7x7 -> 3x3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding='same'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.deep_feature_extractor = nn.Sequential(
            nn.Linear(128*3*3, self.deep_feature_dim)
        )

        self.logits, self.logit_type = set_logits(
            protocol=protocol,
            loss_type=self.loss_type,
            in_features=self.deep_feature_dim,
            out_features=self.number_of_classes,
            logit_bias=False
        )


    def forward(self, image, labels, return_angles=False):
        """ Forward pass

        Args:
            image(tensor): Tensor with input samples
            labels(tensor): Labels for the input samples (needed for margin computation in 'face losses')
            return_angles(bool): set true to return angles of input data (i.e. its deep feature) and all class centers.

        Returns:
            Logits and deep features of the samples.
        """
        x = self.feature_extractor(image)
        x = torch.flatten(x, 1)
        deep_features = self.deep_feature_extractor(x)
        logits = self.logits(deep_features, labels)

        if return_angles:
            # compute angles
            with torch.no_grad():
                cos_theta = F.normalize(deep_features, dim=1).mm(F.normalize(self.logits.w, dim=0))
                angles = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
            return logits, deep_features, angles
        else:
            return logits, deep_features


def set_seeds(seed):
    """ Sets the seed for different sources of randomness.

    Args:
        seed(int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = Falsegg



def save_checkpoint(f_name, model, epoch, opt, best_score_, scheduler=None):
    """ Saves a training checkpoint.

    Args:
        f_name(str): File name.
        model(torch module): Pytorch model.
        epoch(int): Current epoch.
        opt(torch optimizer): Current optimizer.
        best_score_(float): Current best score.
        scheduler(torch lr_scheduler): Pytorch scheduler.
    """
    # If model is DistributedDataParallel extracts the module.
    if isinstance(model, DistributedDataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    data = {"epoch": epoch + 1,
            "model_state_dict": state,
            "opt_state_dict": opt.state_dict(),
            "best_score": best_score_}
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    torch.save(data, f_name)


def load_checkpoint(model, checkpoint, opt=None, scheduler=None):
    """ Loads a checkpoint, if the model was saved using DistributedDataParallel, removes the word
    'module' from the state_dictionary keys to load it in a single device. If fine-tuning model then
    optimizer should be none to start from clean optimizer parameters.

    Args:
        model (torch nn.module): Requires a model to load the state dictionary.
        checkpoint (Path): File path.
        opt (torch optimizer): An optimizer to load the state dictionary. Defaults to None.
        device (str): Device to load the checkpoint. Defaults to 'cpu'.
        scheduler (torch lr_scheduler): Learning rate scheduler. Defaults to None.
    """
    file_path = pathlib.Path(checkpoint)
    if file_path.is_file():  # First check if file exists
        checkpoint = torch.load(file_path, map_location=vast.tools._device)
        key = list(checkpoint["model_state_dict"].keys())[0]
        # If module was saved as DistributedDataParallel then removes the world "module"
        # from dictionary keys
        if key[:6] == "module":
            new_state_dict = OrderedDict()
            for k_i, v_i in checkpoint["model_state_dict"].items():
                key = k_i[7:]  # remove "module"
                new_state_dict[key] = v_i
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if opt is not None:  # Load optimizer state
            opt.load_state_dict(checkpoint["opt_state_dict"])

        if scheduler is not None:  # Load scheduler state
            scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["best_score"]
        return start_epoch, best_score
    else:
        raise Exception(f"Checkpoint file '{checkpoint}' not found")
