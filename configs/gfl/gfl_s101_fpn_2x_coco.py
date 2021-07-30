_base_ = './gfl_s50_fpn_2x_coco.py'
model = dict(
    pretrained='open-mmlab://resnest101',
    backbone=dict(stem_channels=128, depth=101)
)
# fp16 settings
fp16 = dict(loss_scale='dynamic')