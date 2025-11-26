# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS

#mse+hcl
@MODELS.register_module()
class new_PKDLoss(nn.Module):
    """PyTorch version of `PKD: General Distillation Framework for Object
    Detectors via Pearson Correlation Coefficient.

    <https://arxiv.org/abs/2207.02039>`_.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        resize_stu (bool): If True, we'll down/up sample the features of the
            student model to the spatial size of those of the teacher model if
            their spatial sizes are different. And vice versa. Defaults to
            True.
    """
    

    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(new_PKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def hcl(self,fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            loss = F.mse_loss(fs, ft, reduction='mean')
            cnt = 1.0
            tot = 1.0
            for l in [4,2,1]:
                tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                tmpft = F.adaptive_avg_pool2d(ft, (l,l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all
    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Forward computation.

        Args:
            preds_S (torch.Tensor | Tuple[torch.Tensor]): The student model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).
            preds_T (torch.Tensor | Tuple[torch.Tensor]): The teacher model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.

        for pred_S, pred_T in zip(preds_S, preds_T):
            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S[0] != size_T[0]:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear')
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear')
            assert pred_S.shape == pred_T.shape

            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)
            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.

            
            loss += self.hcl(norm_S, norm_T)

        return loss * self.loss_weight

@MODELS.register_module()
class hcl_wo_mseLoss(nn.Module):
    """PyTorch version of `PKD: General Distillation Framework for Object
    Detectors via Pearson Correlation Coefficient.

    <https://arxiv.org/abs/2207.02039>`_.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        resize_stu (bool): If True, we'll down/up sample the features of the
            student model to the spatial size of those of the teacher model if
            their spatial sizes are different. And vice versa. Defaults to
            True.
    """
    

    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(hcl_wo_mseLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def hcl(self,fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            loss = F.mse_loss(fs, ft, reduction='mean')
            cnt = 1.0
            tot = 1.0
            for l in [4,2,1]:
                tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                tmpft = F.adaptive_avg_pool2d(ft, (l,l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all
    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Forward computation.

        Args:
            preds_S (torch.Tensor | Tuple[torch.Tensor]): The student model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).
            preds_T (torch.Tensor | Tuple[torch.Tensor]): The teacher model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.

        for pred_S, pred_T in zip(preds_S, preds_T):
            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S[0] != size_T[0]:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear')
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear')
            assert pred_S.shape == pred_T.shape

            
            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.
            loss += self.hcl(pred_S, pred_T)

        return loss * self.loss_weight
    

@MODELS.register_module()
class yolopkdLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(yolopkdLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def hcl(self,fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            loss = F.mse_loss(fs, ft, reduction='mean')
            cnt = 1.0
            tot = 1.0
            for l in [4,2,1]:
                tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                tmpft = F.adaptive_avg_pool2d(ft, (l,l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all
    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Forward computation.

        Args:
            preds_S (torch.Tensor | Tuple[torch.Tensor]): The student model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).
            preds_T (torch.Tensor | Tuple[torch.Tensor]): The teacher model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.

        for pred_S, pred_T in zip(preds_S, preds_T):
            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S[0] != size_T[0]:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear')
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear')
            assert pred_S.shape[1] == pred_T.shape[1]

            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)            
            loss += self.hcl(norm_S, norm_T)

        return loss * self.loss_weight
    

    
@MODELS.register_module()
class MixALoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(MixALoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def MixA(self,fstudent, fteacher):
        loss_all = 0
        divisor = 2
        n,c,h,w = fstudent.shape
        tmpft0 = F.adaptive_avg_pool2d(fteacher, (h,w))
        loss = F.mse_loss(fstudent, tmpft0, reduction='mean')
        loss_lower = 0
        cnt = 0.5
        cnt_lower = 0.5
        tot = 1.0
        tot_lower =1.0
        while True:
            h = h // divisor
            w = w // divisor
            if h < 5 or w < 5:
                break
            cnt /= divisor
            tot += cnt
            tmpfs = F.adaptive_avg_pool2d(fstudent, (h,w))
            tmpft = F.adaptive_avg_pool2d(fteacher, (h,w))
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        loss = loss / tot
        loss_all = loss_all + loss
        for l in [4,2,1]:
            tmpfs = F.adaptive_max_pool2d(fstudent, (l,l))
            tmpft = F.adaptive_max_pool2d(fteacher, (l,l))
            cnt_lower /= 2.0
            loss_lower += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt_lower
            tot_lower += cnt_lower
        loss_lower = loss_lower / tot_lower
        loss_all = loss_all + loss_lower
        return loss_all

    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:

        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.

        for pred_S, pred_T in zip(preds_S, preds_T):
            if pred_S.shape[1] != pred_T.shape[1]:
                align_conv = nn.Conv2d(pred_T.shape[1], pred_S.shape[1], kernel_size=1).to(pred_T.device)
                pred_T = align_conv(pred_T)
            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)
            loss += self.MixA(norm_S, norm_T)/2

        return loss * self.loss_weight

@MODELS.register_module()
class stuchanelmixaLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(stuchanelmixaLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def MixA(self,fstudent, fteacher):
        loss_all = 0
        divisor = 2
        n,c,h,w = fstudent.shape
        tmpft0 = F.adaptive_avg_pool2d(fteacher, (h,w))
        loss = F.mse_loss(fstudent, tmpft0, reduction='mean')
        loss_lower = 0
        cnt = 0.5
        cnt_lower = 0.5
        tot = 1.0
        tot_lower =1.0
        while True:
            h = h // divisor
            w = w // divisor
            if h < 5 or w < 5:
                break
            cnt /= divisor
            tot += cnt
            tmpfs = F.adaptive_avg_pool2d(fstudent, (h,w))
            tmpft = F.adaptive_avg_pool2d(fteacher, (h,w))
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        loss = loss / tot
        loss_all = loss_all + loss
        for l in [4,2,1]:
            tmpfs = F.adaptive_max_pool2d(fstudent, (l,l))
            tmpft = F.adaptive_max_pool2d(fteacher, (l,l))
            cnt_lower /= 2.0
            loss_lower += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt_lower
            tot_lower += cnt_lower
        loss_lower = loss_lower / tot_lower
        loss_all = loss_all + loss_lower
        return loss_all

    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:

        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.

        for pred_S, pred_T in zip(preds_S, preds_T):
            if pred_S.shape[1] != pred_T.shape[1]:
                align_conv = nn.Conv2d(pred_S.shape[1], pred_T.shape[1], kernel_size=1).to(pred_S.device)
                pred_T = align_conv(pred_S)
            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)
            loss += self.MixA(norm_S, norm_T)/2

        return loss * self.loss_weight