_base_ = [
    '../_base_/datasets/formula_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='GFL',
    pretrained='open-mmlab://resnest50',
    backbone=dict(
        type='ResNeSt',
        stem_channels=64,
        depth=50,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        with_cp=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=24, # 16
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)
# batch size per GPU
data = dict(
    samples_per_gpu=3,)
# use ResNeSt img_norm
img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
# optimizer
optimizier = dict(type='Ranger', lr=0.001)