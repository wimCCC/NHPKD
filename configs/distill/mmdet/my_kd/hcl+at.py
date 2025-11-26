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
            loss_s1=dict(type='ATLoss', loss_weight=2500.0),
            loss_s2=dict(type='ATLoss', loss_weight=2500.0),
            loss_s3=dict(type='ATLoss', loss_weight=2500.0),
            loss_s4=dict(type='ATLoss', loss_weight=2500.0),
            loss_pkd_fpn0=dict(loss_weight=6, type='new_PKDLoss'),
            loss_pkd_fpn1=dict(loss_weight=6, type='new_PKDLoss'),
            loss_pkd_fpn2=dict(loss_weight=6, type='new_PKDLoss'),
            loss_pkd_fpn3=dict(loss_weight=6, type='new_PKDLoss')),
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
            loss_s1=dict(
                s_feature=dict(
                    data_idx=0, from_student=True, recorder='fpn'),
                t_feature=dict(
                    data_idx=0, from_student=False, recorder='fpn')),
            loss_s2=dict(
                s_feature=dict(
                    data_idx=1, from_student=True, recorder='fpn'),
                t_feature=dict(
                    data_idx=1, from_student=False, recorder='fpn')),
            loss_s3=dict(
                s_feature=dict(
                    data_idx=2, from_student=True, recorder='fpn'),
                t_feature=dict(
                    data_idx=2, from_student=False, recorder='fpn')),
            loss_s4=dict(
                s_feature=dict(
                    data_idx=3, from_student=True, recorder='fpn'),
                t_feature=dict(
                    data_idx=3, from_student=False, recorder='fpn')
                )
            )
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
work_dir = './work_dirs/retinanet_r50_at+hcl'