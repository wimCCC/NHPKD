from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS

@MODELS.register_module()
class SAAPLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(SAAPLoss, self).__init__()
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

    def SAAP(self,fstudent, fteacher):
        loss_all = 0
        divisor = 2
        n,c,h,w = fstudent.shape
        tmpft0 = F.adaptive_avg_pool2d(fteacher, (h,w))
        loss = F.mse_loss(fstudent, tmpft0, reduction='mean')
        loss_lower = 0
        # cnt = 0.5
        cnt_lower = 0.5
        # tot = 1.0
        tot_lower =1.0
        # while True:
        #     h = h // divisor
        #     w = w // divisor
        #     if h < 5 or w < 5:
        #         break
        #     cnt /= divisor
        #     tot += cnt
        #     tmpfs = F.adaptive_avg_pool2d(fstudent, (h,w))
        #     tmpft = F.adaptive_avg_pool2d(fteacher, (h,w))
        #     loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        # loss = loss / tot
        # loss_all = loss_all + loss
        for l in [4,2,1]:
            tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
            tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
            cnt_lower /= 2.0
            loss_lower += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt_lower
            tot_lower += cnt_lower
        loss_lower = loss_lower / tot_lower
        loss_all = loss_all + loss_lower
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
            if pred_S.shape[1] != pred_T.shape[1]:
                align_conv = nn.Conv2d(pred_T.shape[1], pred_S.shape[1], kernel_size=1).to(pred_T.device)
                pred_T = align_conv(pred_T)
            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)
            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.

            
            loss += self.SAAP(norm_S, norm_T)

        return loss * self.loss_weight
    

@MODELS.register_module()
class LAMPLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(LAMPLoss, self).__init__()
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

    def LAMP(self,fstudent, fteacher):
        loss_all = 0
        divisor = 2
        n,c,h,w = fstudent.shape
        tmpft0 = F.adaptive_avg_pool2d(fteacher, (h,w))
        loss = F.mse_loss(fstudent, tmpft0, reduction='mean')
        loss_lower = 0
        cnt = 0.5
        # cnt_lower = 0.5
        tot = 1.0
        # tot_lower =1.0
        while True:
            h = h // divisor
            w = w // divisor
            if h < 5 or w < 5:
                break
            cnt /= divisor
            tot += cnt
            tmpfs = F.adaptive_max_pool2d(fstudent, (h,w))
            tmpft = F.adaptive_max_pool2d(fteacher, (h,w))
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        loss = loss / tot
        loss_all = loss_all + loss
        # for l in [4,2,1]:
        #     tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        #     tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        #     cnt_lower /= 2.0
        #     loss_lower += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt_lower
        #     tot_lower += cnt_lower
        # loss_lower = loss_lower / tot_lower
        loss_all = loss_all + loss_lower
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
            if pred_S.shape[1] != pred_T.shape[1]:
                align_conv = nn.Conv2d(pred_T.shape[1], pred_S.shape[1], kernel_size=1).to(pred_T.device)
                pred_T = align_conv(pred_T)
            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)
            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.

            
            loss += self.LAMP(norm_S, norm_T)

        return loss * self.loss_weight
    
@MODELS.register_module()
class LAAPLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(LAAPLoss, self).__init__()
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

    def LAAP(self,fstudent, fteacher):
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
        # for l in [4,2,1]:
        #     tmpfs = F.adaptive_max_pool2d(fstudent, (l,l))
        #     tmpft = F.adaptive_max_pool2d(fteacher, (l,l))
        #     cnt_lower /= 2.0
        #     loss_lower += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt_lower
        #     tot_lower += cnt_lower
        # loss_lower = loss_lower / tot_lower
        # loss_all = loss_all + loss_lower
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
            loss += self.LAAP(norm_S, norm_T)/2

        return loss * self.loss_weight
    
@MODELS.register_module()
class SAMPLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(SAMPLoss, self).__init__()
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

    def SAMP(self,fstudent, fteacher):
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
        # while True:
        #     h = h // divisor
        #     w = w // divisor
        #     if h < 5 or w < 5:
        #         break
        #     cnt /= divisor
        #     tot += cnt
        #     tmpfs = F.adaptive_avg_pool2d(fstudent, (h,w))
        #     tmpft = F.adaptive_avg_pool2d(fteacher, (h,w))
        #     loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        # loss = loss / tot
        # loss_all = loss_all + loss
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
            loss += self.SAMP(norm_S, norm_T)/2

        return loss * self.loss_weight
    
@MODELS.register_module()
class PAALoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(PAALoss, self).__init__()
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

    def PAA(self,fstudent, fteacher):
        loss_all = 0
        divisor = 2
        n,c,h,w = fstudent.shape
        tmpft0 = F.adaptive_avg_pool2d(fteacher, (h,w))
        loss = F.mse_loss(fstudent, tmpft0, reduction='mean')
        cnt = 1.0
        tot = 1.0
        while True:
            h = h // divisor
            w = w // divisor
            if h < 1 or w < 1:
                break
            cnt /= divisor
            tot += cnt
            tmpfs = F.adaptive_avg_pool2d(fstudent, (h,w))
            tmpft = F.adaptive_avg_pool2d(fteacher, (h,w))
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
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
            if pred_S.shape[1] != pred_T.shape[1]:
                align_conv = nn.Conv2d(pred_T.shape[1], pred_S.shape[1], kernel_size=1).to(pred_T.device)
                pred_T = align_conv(pred_T)
            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)
            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.

            
            loss += self.PAA(norm_S, norm_T)

        return loss * self.loss_weight
    

@MODELS.register_module()
class PAMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(PAMLoss, self).__init__()
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

    def PAM(self,fstudent, fteacher):
        loss_all = 0
        divisor = 2
        n,c,h,w = fstudent.shape
        tmpft0 = F.adaptive_avg_pool2d(fteacher, (h,w))
        loss = F.mse_loss(fstudent, tmpft0, reduction='mean')
        cnt = 1.0
        tot = 1.0
        while True:
            h = h // divisor
            w = w // divisor
            if h < 1 or w < 1:
                break
            cnt /= divisor
            tot += cnt
            tmpfs = F.adaptive_max_pool2d(fstudent, (h,w))
            tmpft = F.adaptive_max_pool2d(fteacher, (h,w))
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
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
            if pred_S.shape[1] != pred_T.shape[1]:
                align_conv = nn.Conv2d(pred_T.shape[1], pred_S.shape[1], kernel_size=1).to(pred_T.device)
                pred_T = align_conv(pred_T)
            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)
            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.

            
            loss += self.PAM(norm_S, norm_T)

        return loss * self.loss_weight