# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrazor.registry import MODELS

@MODELS.register_module()
class HCLoss(nn.Module):
    """
    Args:
        loss_weight (float): Weight of this loss item. Defaults to 1.0.
        scale_weights (list[float]): Weights for different scales. 
            Defaults to [1.0, 0.5, 0.25] corresponding to scales [1, 1/2, 1/4].
        base_reduction (str): Reduction method for base scale loss.
            Options are 'mean', 'sum', 'none'. Defaults to 'mean'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 scale_weights: list = [ 0.5, 0.25,0.125],
                 base_reduction: str = 'mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.scale_weights = scale_weights 
        self.base_reduction = base_reduction

        # Validate scale weights
        if len(self.scale_weights) != 3:
            raise ValueError('scale_weights must have length 3 (for 1x, 1/2x, 1/4x scales)')

    def forward(self,
                fstudent: torch.Tensor,
                fteacher: torch.Tensor,
                ) -> torch.Tensor:
        """Forward function.

        Args:
            fstudent (torch.Tensor | list[torch.Tensor]): Student features.
            fteacher (torch.Tensor | list[torch.Tensor]): Teacher features.
        Returns:
            torch.Tensor: The calculated loss.
        """
        # Handle single feature map case
        if isinstance(fstudent, torch.Tensor):
            fstudent = [fstudent]
            fteacher = [fteacher]

        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            # Base scale (original resolution)
            n, c, h, w = fs.shape
            loss = F.mse_loss(fs, ft, reduction=self.base_reduction) * self.scale_weights[0]

            # Multi-scale comparison
            for scale, weight in zip([4, 2, 1], self.scale_weights[0:]):
                if scale >= h:
                    continue  # Skip if feature map is smaller than target scale
                
                # Adaptive pooling to target scale
                tmp_fs = F.adaptive_avg_pool2d(fs, (scale, scale))
                tmp_ft = F.adaptive_avg_pool2d(ft, (scale, scale))
                
                # Add scaled loss
                loss += F.mse_loss(tmp_fs, tmp_ft, reduction=self.base_reduction) * weight

            # Normalize by total weights
            total_weight = sum(self.scale_weights)
            loss = loss / total_weight
            loss_all += loss

        return self.loss_weight * loss_all

    def __repr__(self):
        """str: Returns a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(loss_weight={self.loss_weight}, '
        repr_str += f'scale_weights={self.scale_weights}, '
        repr_str += f'base_reduction={self.base_reduction})'
        return repr_str 
