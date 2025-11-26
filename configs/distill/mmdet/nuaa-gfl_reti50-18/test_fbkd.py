auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = '/home4/caiwm/IRSTD-1k'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=200, type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
img_scales = [
    (
        1333,
        800,
    ),
    (
        666,
        400,
    ),
    (
        2000,
        1200,
    ),
]
launcher = 'none'
load_from = None
log_config = dict(
    hooks=[
        dict(_scope_='mmdet', type='TextLoggerHook'),
        dict(
            _scope_='mmdet',
            init_kwargs=dict(project='retinanet_r50_fpn_1x_coco'),
            type='WandbLoggerHook'),
    ],
    interval=100)
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=('0', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    _scope_='mmrazor',
    architecture=dict(
        cfg_path='mmdet::retinanet/sirst_retinanet_r18.py', pretrained=False),
    distiller=dict(
        connectors=dict(
            loss_s0_sfeat=dict(
                in_channels=256,
                maxpool_stride=8,
                mode='dot_product',
                reduction=4,
                sub_sample=True,
                type='FBKDStudentConnector'),
            loss_s0_tfeat=dict(
                in_channels=256,
                maxpool_stride=8,
                mode='dot_product',
                reduction=4,
                sub_sample=True,
                type='FBKDTeacherConnector'),
            loss_s1_sfeat=dict(
                in_channels=256,
                maxpool_stride=4,
                mode='dot_product',
                reduction=4,
                sub_sample=True,
                type='FBKDStudentConnector'),
            loss_s1_tfeat=dict(
                in_channels=256,
                maxpool_stride=4,
                mode='dot_product',
                reduction=4,
                sub_sample=True,
                type='FBKDTeacherConnector'),
            loss_s2_sfeat=dict(
                in_channels=256,
                mode='dot_product',
                sub_sample=True,
                type='FBKDStudentConnector'),
            loss_s2_tfeat=dict(
                in_channels=256,
                mode='dot_product',
                sub_sample=True,
                type='FBKDTeacherConnector'),
            loss_s3_sfeat=dict(
                in_channels=256,
                mode='dot_product',
                sub_sample=True,
                type='FBKDStudentConnector'),
            loss_s3_tfeat=dict(
                in_channels=256,
                mode='dot_product',
                sub_sample=True,
                type='FBKDTeacherConnector')),
        distill_losses=dict(
            loss_s0=dict(type='FBKDLoss'),
            loss_s1=dict(type='FBKDLoss'),
            loss_s2=dict(type='FBKDLoss'),
            loss_s3=dict(type='FBKDLoss')),
        loss_forward_mappings=dict(
            loss_s0=dict(
                s_input=dict(
                    connector='loss_s0_sfeat',
                    from_student=True,
                    recorder='neck_s0'),
                t_input=dict(
                    connector='loss_s0_tfeat',
                    from_student=False,
                    recorder='neck_s0')),
            loss_s1=dict(
                s_input=dict(
                    connector='loss_s1_sfeat',
                    from_student=True,
                    recorder='neck_s1'),
                t_input=dict(
                    connector='loss_s1_tfeat',
                    from_student=False,
                    recorder='neck_s1')),
            loss_s2=dict(
                s_input=dict(
                    connector='loss_s2_sfeat',
                    from_student=True,
                    recorder='neck_s2'),
                t_input=dict(
                    connector='loss_s2_tfeat',
                    from_student=False,
                    recorder='neck_s2')),
            loss_s3=dict(
                s_input=dict(
                    connector='loss_s3_sfeat',
                    from_student=True,
                    recorder='neck_s3'),
                t_input=dict(
                    connector='loss_s3_tfeat',
                    from_student=False,
                    recorder='neck_s3'))),
        student_recorders=dict(
            neck_s0=dict(source='neck.fpn_convs.0.conv', type='ModuleOutputs'),
            neck_s1=dict(source='neck.fpn_convs.1.conv', type='ModuleOutputs'),
            neck_s2=dict(source='neck.fpn_convs.2.conv', type='ModuleOutputs'),
            neck_s3=dict(source='neck.fpn_convs.3.conv',
                         type='ModuleOutputs')),
        teacher_recorders=dict(
            neck_s0=dict(source='neck.fpn_convs.0.conv', type='ModuleOutputs'),
            neck_s1=dict(source='neck.fpn_convs.1.conv', type='ModuleOutputs'),
            neck_s2=dict(source='neck.fpn_convs.2.conv', type='ModuleOutputs'),
            neck_s3=dict(source='neck.fpn_convs.3.conv',
                         type='ModuleOutputs')),
        type='ConfigurableDistiller'),
    teacher=dict(
        cfg_path='mmdet::retinanet/sirst_retinanet_r50.py', pretrained=False),
    teacher_ckpt=
    '/home/caiwm/mmrazor/checkpoint/teacher/IRSTD1_reti50_e150.pth',
    type='SingleTeacherDistill')
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        _scope_='mmdet',
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        _scope_='mmdet',
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
student = dict(
    _scope_='mmdet',
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        pretrained=None,
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=1,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.3),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')
teacher_ckpt = '/home/caiwm/mmrazor/checkpoint/teacher/IRSTD1_reti50_e150.pth'
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='/home4/caiwm/IRSTD-1k',
        metainfo=dict(classes=('0', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        _scope_='mmdet',
        ann_file='/home4/caiwm/IRSTD-1k/annotations/test.json',
        format_only=False,
        metric='bbox',
        type='PrecisionRecallF1Metric'),
]
test_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmdet',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmdet',
    max_epochs=400,
    type='EpochBasedTrainLoop',
    val_interval=5)
train_dataloader = dict(
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='/home4/caiwm/IRSTD-1k',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=('0', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(_scope_='mmdet', prob=0.5, type='RandomFlip'),
    dict(_scope_='mmdet', type='PackDetInputs'),
]
tta_model = dict(
    _scope_='mmdet',
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.5, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(
        _scope_='mmdet',
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    1333,
                    800,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    666,
                    400,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    2000,
                    1200,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='/home4/caiwm/IRSTD-1k',
        metainfo=dict(classes=('0', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        _scope_='mmdet',
        ann_file='/home4/caiwm/IRSTD-1k/annotations/test.json',
        format_only=False,
        metric='bbox',
        type='PrecisionRecallF1Metric'),
]
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'),
    ])
work_dir = './work_dirs/IRSTD-reti_r50-18_fbkd'
