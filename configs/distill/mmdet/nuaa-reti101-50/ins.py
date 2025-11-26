_base_ = ['mmdet::retinanet/sirst_retinanet_r50.py']
del _base_.model
teacher_ckpt = '/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth'

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::retinanet/sirst_retinanet_r50.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::retinanet/sirst_retinanet_r101.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='INSLoss', loss_weight=0.001),
            loss_pkd_fpn1=dict(type='INSLoss', loss_weight=0.001),
            loss_pkd_fpn2=dict(type='INSLoss', loss_weight=0.001),
            loss_pkd_fpn3=dict(type='INSLoss', loss_weight=0.001)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_pkd_fpn3=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=3)))))

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
# optimizer
default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=100, type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))

load_from = None
resume = False
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    type='OptimWrapper')
train_cfg = dict(
    _scope_='mmdet',
    max_epochs=500,
    type='EpochBasedTrainLoop',
    val_interval=5)
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
work_dir = './work_dirs/retinanet_r50_inskd'