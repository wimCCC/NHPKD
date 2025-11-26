_base_ = ['mmdet::retinanet/sirst_retinanet_r50.py']

teacher_ckpt = '/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth'
auto_scale_lr = dict(base_batch_size=16, enable=True)
student = _base_.model
student.backbone.init_cfg = None
student.neck.init_cfg = dict(
    type='Pretrained', prefix='neck.', checkpoint=teacher_ckpt)
student.bbox_head.init_cfg = dict(
    type='Pretrained', prefix='bbox_head.', checkpoint=teacher_ckpt)
del student.backbone.init_cfg
model = dict(
    _scope_='mmrazor',
    _delete_=True,
    type='FpnTeacherDistill',
    architecture=student,
    teacher=dict(
        cfg_path='mmdet::retinanet/sirst_retinanet_r101.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            fpn3=dict(type='ModuleOutputs', source='neck.fpn_convs.3.conv'),
            fpn4=dict(type='ModuleOutputs', source='neck.fpn_convs.4.conv')),
        teacher_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            fpn3=dict(type='ModuleOutputs', source='neck.fpn_convs.3.conv'),
            fpn4=dict(type='ModuleOutputs', source='neck.fpn_convs.4.conv')),
        connectors=dict(
            s_fpn0_connector=dict(
                type='MGDConnector',
                student_channels=256,
                teacher_channels=256,
                lambda_mgd=0.65),
            s_fpn1_connector=dict(
                type='MGDConnector',
                student_channels=256,
                teacher_channels=256,
                lambda_mgd=0.65),
            s_fpn2_connector=dict(
                type='MGDConnector',
                student_channels=256,
                teacher_channels=256,
                lambda_mgd=0.65),
            s_fpn3_connector=dict(
                type='MGDConnector',
                student_channels=256,
                teacher_channels=256,
                lambda_mgd=0.65),
            s_fpn4_connector=dict(
                type='MGDConnector',
                student_channels=256,
                teacher_channels=256,
                lambda_mgd=0.65)),
        distill_losses=dict(
            loss_mgd_fpn0=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_fpn1=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_fpn2=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_fpn3=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_fpn4=dict(type='MGDLoss', alpha_mgd=0.00002)),
        loss_forward_mappings=dict(
            loss_mgd_fpn0=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='fpn0',
                    connector='s_fpn0_connector'),
                preds_T=dict(from_student=False, recorder='fpn0')),
            loss_mgd_fpn1=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='fpn1',
                    connector='s_fpn1_connector'),
                preds_T=dict(from_student=False, recorder='fpn1')),
            loss_mgd_fpn2=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='fpn2',
                    connector='s_fpn2_connector'),
                preds_T=dict(from_student=False, recorder='fpn2')),
            loss_mgd_fpn3=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='fpn3',
                    connector='s_fpn3_connector'),
                preds_T=dict(from_student=False, recorder='fpn3')),
            loss_mgd_fpn4=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='fpn4',
                    connector='s_fpn4_connector'),
                preds_T=dict(from_student=False, recorder='fpn4')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

load_from = None
resume = True
find_unused_parameters = True
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.003, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
train_cfg = dict(
    _scope_='mmdet',
    max_epochs=600,
    type='EpochBasedTrainLoop',
    val_interval=10)
work_dir = './work_dirs/retinanet_r50_mgd'
