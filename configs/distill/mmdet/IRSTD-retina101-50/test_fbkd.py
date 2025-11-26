_base_ = ['./mse+hcl.py']

student = _base_.model
del _base_.model

teacher_ckpt = '/home/caiwm/mmrazor/checkpoint/teacher/IRSTED1_reti101_e90.pth'
default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=200, type='CheckpointHook'))
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
                         source='neck.fpn_convs.3.conv')),
        teacher_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.fpn_convs.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.fpn_convs.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.fpn_convs.2.conv'),
            neck_s3=dict(type='ModuleOutputs',
                         source='neck.fpn_convs.3.conv')),
        distill_losses=dict(
            loss_s0=dict(type='FBKDLoss'),
            loss_s1=dict(type='FBKDLoss'),
            loss_s2=dict(type='FBKDLoss'),
            loss_s3=dict(type='FBKDLoss')),
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
            loss_s0=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s0',
                    connector='loss_s0_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s0',
                    connector='loss_s0_tfeat')),
            loss_s1=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s1',
                    connector='loss_s1_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s1',
                    connector='loss_s1_tfeat')),
            loss_s2=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s2',
                    connector='loss_s2_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s2',
                    connector='loss_s2_tfeat')),
            loss_s3=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s3',
                    connector='loss_s3_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s3',
                    connector='loss_s3_tfeat')))))
load_from = None
resume = True
find_unused_parameters = True
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.003, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
train_cfg = dict(
    _scope_='mmdet',
    max_epochs=400,
    type='EpochBasedTrainLoop',
    val_interval=5)
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
work_dir = './work_dirs/IRSTD-retinanet_r101-50_fbkd'