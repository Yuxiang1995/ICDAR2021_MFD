dataset_type = 'CocoDataset'
data_root = '/data4/dataset/formula_icdar2021/'
classes = ('embedded', 'isolated')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1600, 1440)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1583, 2048),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
        type=dataset_type,
        ann_file=data_root + 'Tr00/train_coco_sdk4.json',
        img_prefix=data_root + 'Tr00/img/',
        classes=classes,
        pipeline=train_pipeline),
        dict(
        type=dataset_type,
        ann_file=data_root + 'Tr01/train_coco_sdk4.json',
        img_prefix=data_root + 'Tr01/img/',
        classes=classes,
        pipeline=train_pipeline),
        dict(
        type=dataset_type,
        ann_file=data_root + 'Tr10/train_coco_sdk4.json',
        img_prefix=data_root + 'Tr10/img/',
        classes=classes,
        pipeline=train_pipeline),
        dict(
        type=dataset_type,
        ann_file=data_root + 'Va00/train_coco_sdk4.json',
        img_prefix=data_root + 'Va00/img/',
        classes=classes,
        pipeline=train_pipeline),
        dict(
        type=dataset_type,
        ann_file=data_root + 'Va01/train_coco_sdk4.json',
        img_prefix=data_root + 'Va01/img/',
        classes=classes,
        pipeline=train_pipeline),
        dict(
        type=dataset_type,
        ann_file=data_root + 'Ts00/train_coco_sdk4.json',
        img_prefix=data_root + 'Ts00/img/',
        classes=classes,
        pipeline=train_pipeline),
        dict(
        type=dataset_type,
        ann_file=data_root + 'Ts01/train_coco_sdk4.json',
        img_prefix=data_root + 'Ts01/img/',
        classes=classes,
        pipeline=train_pipeline),
        ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Ts01/train_coco_sdk4.json',
        img_prefix=data_root + 'Ts01/img/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Ts10/train_coco_sdk4.json',
        img_prefix=data_root + 'Ts10/img/',
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
