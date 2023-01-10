from enum import Enum
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss, FocalLoss, LovaszLoss, TverskyLoss, \
    SoftBCEWithLogitsLoss, SoftCrossEntropyLoss, MCCLoss
from torch import nn
from torch.nn.modules.loss import _Loss, CrossEntropyLoss


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


class Losses(Enum):
    jaccard_loss = 'jaccard_loss'
    dice_loss = 'dice_loss'
    focal_loss = 'focal_loss'
    lovasz_loss = 'lovasz_loss'
    cross_entropy_loss = 'cross_entropy_loss'
    # tversky_loss = 'tversky_loss'
    # soft_bce_with_logits_loss = 'soft_bce_with_logits_loss'
    # soft_cross_entropy_loss = 'soft_cross_entropy_loss'
    # mcc_loss = 'mcc_loss'

    def get_loss(self):
        return {
            'jaccard_loss': JaccardLoss,
            'dice_loss': DiceLoss,
            'focal_loss': FocalLoss,
            'lovasz_loss': LovaszLoss,
            'tversky_loss': TverskyLoss,
            'soft_bce_with_logits_loss': SoftBCEWithLogitsLoss,
            'soft_cross_entropy_loss': SoftCrossEntropyLoss,
            'mcc_loss': MCCLoss,
            'cross_entropy_loss': nn.CrossEntropyLoss,
        }[self.value]
