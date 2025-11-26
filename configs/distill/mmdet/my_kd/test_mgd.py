auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = '/home4/caiwm/nuaa-sirst'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=5, type='CheckpointHook'),
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
        _scope_='mmdet',
        backbone=dict(
            depth=50,
            frozen_stages=1,
            # init_cfg=dict(
            #     checkpoint='torchvision://resnet50', type=None),
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
            init_cfg=dict(
                checkpoint=
                '/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth',
                prefix='bbox_head.',
                type='Pretrained'),
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
            init_cfg=dict(
                checkpoint=
                '/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth',
                prefix='neck.',
                type='Pretrained'),
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
        type='RetinaNet'),
    distiller=dict(
        connectors=dict(
            s_fpn0_connector=dict(
                lambda_mgd=0.65,
                student_channels=256,
                teacher_channels=256,
                type='MGDConnector'),
            s_fpn1_connector=dict(
                lambda_mgd=0.65,
                student_channels=256,
                teacher_channels=256,
                type='MGDConnector'),
            s_fpn2_connector=dict(
                lambda_mgd=0.65,
                student_channels=256,
                teacher_channels=256,
                type='MGDConnector'),
            s_fpn3_connector=dict(
                lambda_mgd=0.65,
                student_channels=256,
                teacher_channels=256,
                type='MGDConnector'),
            s_fpn4_connector=dict(
                lambda_mgd=0.65,
                student_channels=256,
                teacher_channels=256,
                type='MGDConnector')),
        distill_losses=dict(
            loss_mgd_fpn0=dict(alpha_mgd=2e-05, type='MGDLoss'),
            loss_mgd_fpn1=dict(alpha_mgd=2e-05, type='MGDLoss'),
            loss_mgd_fpn2=dict(alpha_mgd=2e-05, type='MGDLoss'),
            loss_mgd_fpn3=dict(alpha_mgd=2e-05, type='MGDLoss'),
            loss_mgd_fpn4=dict(alpha_mgd=2e-05, type='MGDLoss')),
        loss_forward_mappings=dict(
            loss_mgd_fpn0=dict(
                preds_S=dict(
                    connector='s_fpn0_connector',
                    from_student=True,
                    recorder='fpn0'),
                preds_T=dict(from_student=False, recorder='fpn0')),
            loss_mgd_fpn1=dict(
                preds_S=dict(
                    connector='s_fpn1_connector',
                    from_student=True,
                    recorder='fpn1'),
                preds_T=dict(from_student=False, recorder='fpn1')),
            loss_mgd_fpn2=dict(
                preds_S=dict(
                    connector='s_fpn2_connector',
                    from_student=True,
                    recorder='fpn2'),
                preds_T=dict(from_student=False, recorder='fpn2')),
            loss_mgd_fpn3=dict(
                preds_S=dict(
                    connector='s_fpn3_connector',
                    from_student=True,
                    recorder='fpn3'),
                preds_T=dict(from_student=False, recorder='fpn3')),
            loss_mgd_fpn4=dict(
                preds_S=dict(
                    connector='s_fpn4_connector',
                    from_student=True,
                    recorder='fpn4'),
                preds_T=dict(from_student=False, recorder='fpn4'))),
        student_recorders=dict(
            fpn0=dict(source='neck.fpn_convs.0.conv', type='ModuleOutputs'),
            fpn1=dict(source='neck.fpn_convs.1.conv', type='ModuleOutputs'),
            fpn2=dict(source='neck.fpn_convs.2.conv', type='ModuleOutputs'),
            fpn3=dict(source='neck.fpn_convs.3.conv', type='ModuleOutputs'),
            fpn4=dict(source='neck.fpn_convs.4.conv', type='ModuleOutputs')),
        teacher_recorders=dict(
            fpn0=dict(source='neck.fpn_convs.0.conv', type='ModuleOutputs'),
            fpn1=dict(source='neck.fpn_convs.1.conv', type='ModuleOutputs'),
            fpn2=dict(source='neck.fpn_convs.2.conv', type='ModuleOutputs'),
            fpn3=dict(source='neck.fpn_convs.3.conv', type='ModuleOutputs'),
            fpn4=dict(source='neck.fpn_convs.4.conv', type='ModuleOutputs')),
        type='ConfigurableDistiller'),
    teacher=dict(
        cfg_path='mmdet::retinanet/sirst_retinanet_r101.py', pretrained=False),
    teacher_ckpt='/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth',
    type='FpnTeacherDistill')
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.0003, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
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
        init_cfg=dict(
            checkpoint=
            '/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth',
            prefix='bbox_head.',
            type='Pretrained'),
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
        init_cfg=dict(
            checkpoint=
            '/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth',
            prefix='neck.',
            type='Pretrained'),
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
teacher_ckpt = '/home/caiwm/mmrazor/checkpoint/teacher/retina101_epoch_500.pth'
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='/home4/caiwm/nuaa-sirst',
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
        ann_file='/home4/caiwm/nuaa-sirst/annotations/test.json',
        metric='bbox',
        format_only=False,
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
    max_epochs=50,
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
        data_root='/home4/caiwm/nuaa-sirst',
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
val_cfg = dict(_delete_=True,type='mmrazor.SingleTeacherDistillValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='/home4/caiwm/nuaa-sirst',
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
        ann_file='/home4/caiwm/nuaa-sirst/annotations/test.json',
        metric='bbox',
        format_only=False,
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
work_dir = './work_dirs/test_mgd'
