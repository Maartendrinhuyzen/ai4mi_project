#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class BalancedCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = kwargs.get('class_weights', None)
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        if self.class_weights is not None:
            # Move weights to the device of predictions and reshape
            weights = self.class_weights[self.idk].to(pred_softmax.device)
            weights = weights.view(1, -1, 1, 1)  # Shape: (1, C, 1, 1)
            
            # Use broadcasting for element-wise multiplication
            weighted_loss = -weights * mask * log_p
            loss = weighted_loss.sum() / (mask.sum() + 1e-10)
        else:
            loss = - (mask * log_p).sum() / (mask.sum() + 1e-10)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, idk, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.idk = idk
        self.smooth = smooth

    def forward(self, pred_softmax, weak_target):
        """
        Computes the Dice loss.

        Args:
            pred_softmax (Tensor): Predicted probabilities after softmax. Shape: [B, C, H, W]
            weak_target (Tensor): Ground truth one-hot encoded masks. Shape: [B, C, H, W]

        Returns:
            Tensor: Computed Dice loss.
        """
        # Flatten predictions and targets
        pred = pred_softmax[:, self.idk, :, :].contiguous().view(-1)
        target = weak_target[:, self.idk, :, :].contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, idk, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initializes the Focal Loss.

        Args:
            idk (list[int]): List of class indices to include.
            alpha (list[float], optional): Balancing factor for class-specific loss.
            gamma (float, optional): Focusing parameter.
            reduction (str, optional): Reduction method ('mean' or 'sum').
        """
        super(FocalLoss, self).__init__()
        self.idk = idk
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_softmax, weak_target):
        """
        Computes the Focal loss.

        Args:
            pred_softmax (Tensor): Predicted probabilities after softmax. Shape: [B, C, H, W]
            weak_target (Tensor): Ground truth one-hot encoded masks. Shape: [B, C, H, W]

        Returns:
            Tensor: Computed Focal loss.
        """
        if self.alpha is not None:
            alpha = self.alpha.to(pred_softmax.device)
            alpha = alpha.view(1, -1, 1, 1)  # Shape: [1, C, 1, 1]
            alpha = alpha[:, self.idk, :, :]
        else:
            alpha = 1.0

        # Compute cross-entropy
        ce = -weak_target * torch.log(pred_softmax + 1e-7)
        weight = alpha * torch.pow(1 - pred_softmax, self.gamma)
        focal_loss = weight * ce
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss        