# Copyright (c) OpenMMLab. All rights reserved.
from .ab_loss import ABLoss
from .at_loss import ATLoss
from .crd_loss import CRDLoss
from .cross_entropy_loss import CrossEntropyLoss
from .cwd import ChannelWiseDivergence
from .dafl_loss import ActivationLoss, InformationEntropyLoss, OnehotLikeLoss
from .decoupled_kd import DKDLoss
from .dist_loss import DISTLoss
from .factor_transfer_loss import FTLoss
from .fbkd_loss import FBKDLoss
from .kd_soft_ce_loss import KDSoftCELoss
from .kl_divergence import KLDivergence
from .l1_loss import L1Loss
from .l2_loss import L2Loss
from .mgd_loss import MGDLoss
from .ofd_loss import OFDLoss
from .pkd_loss import PKDLoss,yolo_real_PKDLoss
from .relational_kd import AngleWiseRKD, DistanceWiseRKD
from .weighted_soft_label_distillation import WSLD

from .my_pkd import new_PKDLoss,hcl_wo_mseLoss,MixALoss,stuchanelmixaLoss
from .Focal_loss import FocalLoss
from .hcl import HCLoss
from .my_pkd import yolopkdLoss
from .insdist_loss import INSLoss
from .hypothesis import LAAPLoss,SAAPLoss,LAMPLoss,SAMPLoss,PAALoss,PAMLoss
from .module_ablation import Scale8Loss,Scale16Loss,Scale32Loss,Ablation_AMP_32,Ablation_AMP_8,Ablation_AMP_16,Ablation_AMP_4,Ablation_AAP_32,Ablation_AAP_16,Ablation_AAP_8,Ablation_AAP_4,Ablation_AMP_2
__all__ = [
    'ChannelWiseDivergence', 'KLDivergence', 'AngleWiseRKD', 'DistanceWiseRKD',
    'WSLD', 'L2Loss', 'ABLoss', 'DKDLoss', 'KDSoftCELoss', 'ActivationLoss',
    'OnehotLikeLoss', 'InformationEntropyLoss', 'FTLoss', 'ATLoss', 'OFDLoss',
    'L1Loss', 'FBKDLoss', 'CRDLoss', 'CrossEntropyLoss', 'PKDLoss', 'MGDLoss',
    'DISTLoss', 'new_PKDLoss', 'FocalLoss', 'HCLoss', 'hcl_wo_mseLoss','yolopkdLoss','PAALoss','MixALoss',
    'INSLoss', 'LAAPLoss', 'SAAPLoss', 'LAMPLoss', 'SAMPLoss', 'PAMLoss','yolo_real_PKDLoss','Scale8Loss','Scale16Loss','Scale32Loss','Ablation_AMP_32','Ablation_AMP_16','Ablation_AMP_8','Ablation_AMP_4','Ablation_AAP_32','Ablation_AAP_16','Ablation_AAP_8','Ablation_AAP_4',
    'stuchanelmixaLoss','Ablation_AMP_2'
]