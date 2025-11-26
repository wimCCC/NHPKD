_base_ = ['mmdet::retinanet/sirst_retinanet_r50.py']
del _base_.model
teacher_ckpt = '/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth'

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
            neck_s3=dict(type='ModuleOutputs', source='neck.fpn_convs.3.conv')),
        teacher_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            neck_s3=dict(type='ModuleOutputs', source='neck.fpn_convs.3.conv')),
        distill_losses=dict(
            loss_s0=dict(type='new_PKDLoss'),
            loss_s1=dict(type='new_PKDLoss'),
            loss_s2=dict(type='new_PKDLoss'),
            loss_s3=dict(type='new_PKDLoss')),
        connector=dict(
            fuse_s0=dict(
                type='FuseConnector',
                in_channel=256,
                mid_channel=256,
                out_channel=256,
                fuse=True),
            fuse_s1=dict( 
                type='FuseConnector',
                in_channel=256,
                mid_channel=256,
                out_channel=256,
                fuse=True),
            fuse_s2=dict(
                type='FuseConnector',
                in_channel=256,
                mid_channel=256,
                out_channel=256,
                fuse=True)),
        loss_forward_mappings=dict(
            loss_s3=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='neck_s3'),
                preds_T=dict(
                    from_student=False,
                    recorder='neck_s3')),
            loss_s2=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='neck_s2'),
                preds_T=dict(
                    from_student=False,
                    recorder=['neck_s2','neck_s3'])),
            loss_s1=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='neck_s1'),
                preds_T=dict(
                    from_student=False,
                    recorder=['neck_s1','neck_s2'])),
            loss_s0=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='neck_s0'),
                preds_T=dict(
                    from_student=False,
                    recorder=['neck_s0','neck_s1'])))
            
    )
)
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
load_from = None
resume = False
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    type='OptimWrapper')
train_cfg = dict(
    _scope_='mmdet',
    max_epochs=400,
    type='EpochBasedTrainLoop',
    val_interval=10)
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
work_dir = './work_dirs/retinanet_r50_review_kd'

