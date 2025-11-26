# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class INSLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(INSLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu
    
    def mask_insdist(self, gt_bboxes, backbone_feat, featmap_size, featmap_stride, threshold):
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        with torch.no_grad():
            mask_batch = []
            for batch in range(len(gt_bboxes)):
                
                h, w = featmap_size[0], featmap_size[1]
                mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
                
                for ins in range(gt_bboxes[batch].shape[0]):
                    gt_level_map = gt_bboxes[batch][ins] / featmap_stride
                    
                    lx = min(max(0, int(gt_level_map[0])), w - 1)
                    rx = min(max(0, int(gt_level_map[2])), w - 1)
                    ly = min(max(0, int(gt_level_map[1])), h - 1)
                    ry = min(max(0, int(gt_level_map[3])), h - 1)
                    
                    if (lx == rx) or (ly == ry):
                        mask_per_img[ly, lx] += 1
                    else:
                        x = backbone_feat[batch].view(-1, h * w).permute(1, 0)
                        feature_gt = avgpool(backbone_feat[batch][:, ly:(ry + 1), lx:(rx + 1)]).squeeze(-1)
                        energy = torch.mm(x, feature_gt)
                        
                        min_ = torch.min(energy)
                        max_ = torch.max(energy)
                        assert max_ != 0 
                        energy = (energy - min_) / max_
                        attention = energy.view(h, w)
                        
                        attention = (attention > threshold).double()
                        mask_per_img += attention
                mask_per_img = (mask_per_img > 0).double()
                mask_batch.append(mask_per_img)
                
        return torch.stack(mask_batch, dim=0)
      

    def dist_insdist(self, tensor_a, tensor_b, mask):
        diff = (tensor_a - tensor_b) ** 2
    
        mask_gt = mask.unsqueeze(1).repeat(1, tensor_a.size(1), 1, 1).cuda()
        diff_gt = diff * mask_gt
        diff_gt = (torch.sum(diff_gt) + 1e-8) ** 0.5
        
        mask_bg = (1 - mask_gt)
        diff_bg = diff * mask_bg
        diff_bg = (torch.sum(diff_bg) + 1e-8) ** 0.5
        
        return diff_gt, diff_bg

    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple],
                anchor_list: Union[torch.Tensor, Tuple]
                ) -> torch.Tensor:
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
        
        all_bboxes = []

        # 遍历每个 DetDataSample 对象
        for det_data_sample in anchor_list:
            # 提取 gt_instances 中的真实边界框
            if hasattr(det_data_sample, 'gt_instances'):
                gt_bboxes = det_data_sample.gt_instances.bboxes  # 获取真实边界框
                all_bboxes.append(gt_bboxes)
        # all_bboxes_tensor = torch.stack(all_bboxes)
        _mask = self.mask_insdist(all_bboxes, preds_T[-1], featmap_size=preds_T[-1].shape[2:], featmap_stride = 32, threshold=0.6).unsqueeze(1)
        feat_loss = 0.
        for i in range(0, len(preds_T)):
            d_size = preds_T[i].shape[2:]
            mask = F.interpolate(_mask, d_size).squeeze(1)
            loss_gt, loss_bg = self.dist_insdist(preds_T[i], preds_S[i], mask)
            feat_loss += (loss_gt * 1 + loss_bg * 1)
        feat_loss = feat_loss * self.loss_weight
        # losses.update({'kd_feat_loss': feat_loss})
        
        
        # loss += F.mse_loss(norm_S, norm_T, reduction='sum') / 2
        
        return feat_loss