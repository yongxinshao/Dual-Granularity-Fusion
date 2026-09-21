_base_ = './cascade-mask-rcnn_r50_fpn_1x_nuim.py'

# learning policy
lr_config = dict(step=[16, 19])
runner = dict(max_epochs=20)
load_from = None  # checkpoint path intentionally omitted from the public release

