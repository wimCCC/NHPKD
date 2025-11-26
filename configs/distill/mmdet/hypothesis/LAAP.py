auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = '/home4/caiwm/nuaa-sirst'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(_scope_='mmdet', interval=100, type='CheckpointHook'),
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
        cfg_path='mmdet::yolox/sirst_yolox_t.py', pretrained=False),
    distiller=dict(
        distill_losses=dict(
            loss_pkd_fpn0=dict(loss_weight=2, type='LAAPLoss'),
            loss_pkd_fpn1=dict(loss_weight=2, type='LAAPLoss'),
            loss_pkd_fpn2=dict(loss_weight=2, type='LAAPLoss'),
            loss_pkd_fpn3=dict(loss_weight=2, type='LAAPLoss')),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(data_idx=0, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=0, from_student=False, recorder='fpn')),
            loss_pkd_fpn1=dict(
                preds_S=dict(data_idx=1, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=1, from_student=False, recorder='fpn')),
            loss_pkd_fpn2=dict(
                preds_S=dict(data_idx=2, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=2, from_student=False, recorder='fpn'))),
        student_recorders=dict(fpn=dict(source='neck', type='ModuleOutputs')),
        teacher_recorders=dict(fpn=dict(source='neck', type='ModuleOutputs')),
        type='ConfigurableDistiller'),
    teacher=dict(
        cfg_path='mmdet::retinanet/sirst_retinanet_r50.py', pretrained=False),
    teacher_ckpt=
    'checkpoint/teacher/retina50_epoch_280.pth',
    type='FpnTeacherDistill')
optim_wrapper = dict(
    _scope_='mmdet',
    clip_grad=dict(max_norm=2, norm_type=2),
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
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
val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
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
    ])
work_dir = '/home4/caiwm/hypothesis/LAAP'
