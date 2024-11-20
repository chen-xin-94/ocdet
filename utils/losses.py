import torch
import torch.nn.functional as F


def mse_loss_with_logits(inputs, targets):
    """
    MSE loss with logits

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs.
                 The target values for each example.
    Returns:
        Loss tensor
    """
    return F.mse_loss(inputs.sigmoid(), targets)


def weighted_mse_loss_with_logits(inputs, targets, alpha):
    """
    Weighted MSE loss

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs.
                 The target values for each example.
        alpha: A scalar, the weight for the positive class.
    Returns:
        Loss tensor
    """
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    else:
        alpha_t = 1
    return (alpha_t * (inputs.sigmoid() - targets) ** 2).mean()


def focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of shape (N, C, H ,W) where C = number of classes.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def balanced_continuous_focal_loss(
    inputs, targets, alpha: float = 0.25, gamma: float = 2
):
    """
    Compared to FL, it uses continuous targets and therefore different scale
    Compared to QFL, it uses an additional scale (alpha) to balance positive and negative samples

    Args:
        inputs: A float tensor of shape (N, C, H ,W) where C = number of classes.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the continuous
                 target probability between [0,1] for each element in inputs
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. alpha = -1  means no weighting.
                Should be default to the inverse class frequency (i.e. 1 - num_pos / num_neg)
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    scale = (prob - targets).abs() ** gamma
    loss = ce_loss * scale

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def quality_focal_loss(inputs, targets, gamma: float = 2):
    """
    compared to continuous focal loss, it doen't use alpha to balance positive and negative samples
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    scale = (prob - targets).abs() ** gamma
    loss = ce_loss * scale

    return loss.mean(1).sum()


def dice_loss(inputs, targets, threshold=0.5):
    """
    Compute the DICE loss. Targets are forced to be binary
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid().flatten(1)
    # force binary targets
    targets = (targets > threshold).float().flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum()


def continous_dice_loss(inputs, targets):
    """
    Compute the DICE loss without forcing binary targets
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid().flatten(1)
    # targets can be continuous
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum()


# combine the losses into two classes
class FocalDiceLoss(torch.nn.Module):
    def __init__(self, ratio_focal: float = 20, alpha: float = 0.25, gamma: float = 2):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight_focal = ratio_focal / (ratio_focal + 1)
        self.weight_dice = 1 / (ratio_focal + 1)

    def forward(self, inputs, targets):
        return self.weight_focal * focal_loss(
            inputs, targets, self.alpha, self.gamma
        ) + self.weight_dice * dice_loss(inputs, targets)


class CFocalCDiceLoss(torch.nn.Module):
    def __init__(self, ratio_focal: float = 20, alpha: float = 0.25, gamma: float = 2):
        super(CFocalCDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight_focal = ratio_focal / (ratio_focal + 1)
        self.weight_dice = 1 / (ratio_focal + 1)

    def forward(self, inputs, targets):
        return self.weight_focal * balanced_continuous_focal_loss(
            inputs, targets, self.alpha, self.gamma
        ) + self.weight_dice * continous_dice_loss(inputs, targets)


class CFocalDiceLoss(torch.nn.Module):
    def __init__(self, ratio_focal: float = 20, alpha: float = 0.25, gamma: float = 2):
        super(CFocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight_focal = ratio_focal / (ratio_focal + 1)
        self.weight_dice = 1 / (ratio_focal + 1)

    def forward(self, inputs, targets):
        return self.weight_focal * balanced_continuous_focal_loss(
            inputs, targets, self.alpha, self.gamma
        ) + self.weight_dice * dice_loss(inputs, targets)


class FocalCDiceLoss(torch.nn.Module):
    def __init__(self, ratio_focal: float = 20, alpha: float = 0.25, gamma: float = 2):
        super(FocalCDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight_focal = ratio_focal / (ratio_focal + 1)
        self.weight_dice = 1 / (ratio_focal + 1)

    def forward(self, inputs, targets):
        return self.weight_focal * focal_loss(
            inputs, targets, self.alpha, self.gamma
        ) + self.weight_dice * continous_dice_loss(inputs, targets)


if __name__ == "__main__":
    # Test the losses
    inputs = torch.randn(2, 1, 4, 4)
    targets = (torch.randn(2, 1, 4, 4) > 0.5).float()
    targets_continuous = torch.rand(2, 1, 4, 4)
    print(focal_loss(inputs, targets))
    print(balanced_continuous_focal_loss(inputs, targets_continuous))
    print(dice_loss(inputs, targets_continuous))
    print(continous_dice_loss(inputs, targets_continuous))
