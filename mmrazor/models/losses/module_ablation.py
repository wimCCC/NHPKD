from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
@MODELS.register_module()
class Scale8Loss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Scale8Loss, self).__init__()
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
            if h < 9 or w < 9:
                break
            cnt /= divisor
            tot += cnt
            tmpfs = F.adaptive_avg_pool2d(fstudent, (h,w))
            tmpft = F.adaptive_avg_pool2d(fteacher, (h,w))
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        loss = loss / tot
        loss_all = loss_all + loss
        for l in [8,4,2,1]:
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
class Scale16Loss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Scale16Loss, self).__init__()
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
            if h < 17 or w < 17:
                break
            cnt /= divisor
            tot += cnt
            tmpfs = F.adaptive_avg_pool2d(fstudent, (h,w))
            tmpft = F.adaptive_avg_pool2d(fteacher, (h,w))
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        loss = loss / tot
        loss_all = loss_all + loss
        for l in [16,8,4,2,1]:
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
class Scale32Loss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Scale32Loss, self).__init__()
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
            if h < 33 or w < 33:
                break
            cnt /= divisor
            tot += cnt
            tmpfs = F.adaptive_avg_pool2d(fstudent, (h,w))
            tmpft = F.adaptive_avg_pool2d(fteacher, (h,w))
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        loss = loss / tot
        loss_all = loss_all + loss
        for l in [32,16,8,4,2,1]:
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
class Ablation_AAP_4(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Ablation_AAP_4, self).__init__()
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
        cnt = 0.5
        tot = 1.0
        while True:
            h = h // divisor
            w = w // divisor
            if h < 4 or w < 4:
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
class Ablation_AAP_8(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Ablation_AAP_8, self).__init__()
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
        cnt = 0.5
        tot = 1.0
        while True:
            h = h // divisor
            w = w // divisor
            if h < 8 or w < 8:
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
class Ablation_AAP_16(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Ablation_AAP_16, self).__init__()
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
        cnt = 0.5
        tot = 1.0
        while True:
            h = h // divisor
            w = w // divisor
            if h < 16 or w < 16:
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
class Ablation_AAP_32(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Ablation_AAP_32, self).__init__()
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
        cnt = 0.5
        tot = 1.0
        while True:
            h = h // divisor
            w = w // divisor
            if h < 32 or w < 32:
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
class Ablation_AMP_4(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Ablation_AMP_4, self).__init__()
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
        cnt_lower = 0.5
        tot_lower =1.0
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
class Ablation_AMP_2(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Ablation_AMP_2, self).__init__()
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
        cnt_lower = 0.5
        tot_lower =1.0
        loss_all = loss_all + loss
        for l in [2,1]:
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
class Ablation_AMP_8(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Ablation_AMP_8, self).__init__()
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
        cnt_lower = 0.5
        tot_lower =1.0
        loss_all = loss_all + loss
        for l in [8,4,2,1]:
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
class Ablation_AMP_16(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Ablation_AMP_16, self).__init__()
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
        cnt_lower = 0.5
        tot_lower =1.0
        loss_all = loss_all + loss
        for l in [16,8,4,2,1]:
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
class Ablation_AMP_32(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(Ablation_AMP_32, self).__init__()
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
        cnt_lower = 0.5
        tot_lower =1.0
        loss_all = loss_all + loss
        for l in [32,16,8,4,2,1]:
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