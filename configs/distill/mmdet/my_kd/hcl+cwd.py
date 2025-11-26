_base_ = ['mmdet::retinanet/sirst_retinanet_r50.py']

student = _base_.model
del _base_.model

teacher_ckpt = '/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth'  # noqa: E501
default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=20, type='CheckpointHook'))
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::retinanet/sirst_retinanet_r50.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::retinanet/sirst_retinanet_r101.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            neck_s3=dict(type='ModuleOutputs',
                         source='neck.fpn_convs.3.conv'),
            fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            neck_s3=dict(type='ModuleOutputs',
                         source='neck.fpn_convs.3.conv'),
            fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_cwd_fpn0=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn1=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn2=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn3=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn4=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_pkd_fpn0=dict(loss_weight=6, type='new_PKDLoss'),
            loss_pkd_fpn1=dict(loss_weight=6, type='new_PKDLoss'),
            loss_pkd_fpn2=dict(loss_weight=6, type='new_PKDLoss'),
            loss_pkd_fpn3=dict(loss_weight=6, type='new_PKDLoss')),
        connectors=dict(
            loss_s0_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s0_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s1_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s1_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s2_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True),
            loss_s2_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True),
            loss_s3_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True),
            loss_s3_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=256,
                mode='dot_product',
                sub_sample=True)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(data_idx=0, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=0, from_student=False, recorder='fpn')),
            loss_pkd_fpn1=dict(
                preds_S=dict(data_idx=1, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=1, from_student=False, recorder='fpn')),
            loss_pkd_fpn2=dict(
                preds_S=dict(data_idx=2, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=2, from_student=False, recorder='fpn')),
            loss_pkd_fpn3=dict(
                preds_S=dict(data_idx=3, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=3, from_student=False, recorder='fpn')),
            loss_cwd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_cwd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_cwd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_cwd_fpn3=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=3)),
            loss_cwd_fpn4=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=4),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=4)))
    )
)
load_from = None
resume = True
find_unused_parameters = True
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
train_cfg = dict(
    _scope_='mmdet',
    max_epochs=400,
    type='EpochBasedTrainLoop',
    val_interval=10)
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
work_dir = './work_dirs/retinanet_r50_cwd+hcl'