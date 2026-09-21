# 鍚庣鏀寔

鎴戜滑鏀寔涓嶅悓鐨勬枃浠跺鎴风鍚庣锛氱鐩樸€丆eph 鍜?LMDB 绛夈€備笅闈㈡槸淇敼閰嶇疆浣夸箣浠?Ceph 鍔犺浇鍜屼繚瀛樻暟鎹殑绀轰緥銆?
## 浠?Ceph 璇诲彇鏁版嵁鍜屾爣娉ㄦ枃浠?
鎴戜滑鏀寔浠?Ceph 鍔犺浇鏁版嵁鍜岀敓鎴愮殑鏍囨敞淇℃伅鏂囦欢锛坧kl 鍜?json锛夛細

```python
# set file client backends as Ceph
backend_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/nuscenes/':
        's3://openmmlab/datasets/detection3d/nuscenes/', # replace the path with your data path on Ceph
        'data/nuscenes/':
        's3://openmmlab/datasets/detection3d/nuscenes/' # replace the path with your data path on Ceph
    }))

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    sample_groups=dict(Car=15),
    classes=class_names,
    # set file client for points loader to load training data
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # set file client for data base sampler to load db info file
    backend_args=backend_args)

train_pipeline = [
    # set file client for loading training data
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    # set file client for loading training data annotations
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, backend_args=backend_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    # set file client for loading validation/testing data
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    # set file client for loading training info files (.pkl)
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(pipeline=train_pipeline, classes=class_names, backend_args=backend_args)),
    # set file client for loading validation info files (.pkl)
    val=dict(pipeline=test_pipeline, classes=class_names,backend_args=backend_args),
    # set file client for loading testing info files (.pkl)
    test=dict(pipeline=test_pipeline, classes=class_names, backend_args=backend_args))
```

## 浠?Ceph 璇诲彇棰勮缁冩ā鍨?
```python
model = dict(
    pts_backbone=dict(
        _delete_=True,
        type='NoStemRegNet',
        arch='regnetx_1.6gf',
        init_cfg=dict(
            type='Pretrained', checkpoint='s3://openmmlab/checkpoints/mmdetection3d/regnetx_1.6gf'), # replace the path with your pretrained model path on Ceph
        ...
```

## 浠?Ceph 璇诲彇妯″瀷鏉冮噸鏂囦欢

```python
# replace the path with your checkpoint path on Ceph
load_from = None  # checkpoint path intentionally omitted from the public release
resume_from = None
workflow = [('train', 1)]
```

## 淇濆瓨妯″瀷鏉冮噸鏂囦欢鑷?Ceph

```python
# checkpoint saving
# replace the path with your checkpoint saving path on Ceph
checkpoint_config = dict(interval=1, max_keep_ckpts=2, out_dir='s3://openmmlab/mmdetection3d')
```

## EvalHook 淇濆瓨鏈€浼樻ā鍨嬫潈閲嶆枃浠惰嚦 Ceph

```python
# replace the path with your checkpoint saving path on Ceph
evaluation = dict(interval=1, save_best='bbox', out_dir='s3://openmmlab/mmdetection3d')
```

## 璁粌鏃ュ織淇濆瓨鑷?Ceph

璁粌鍚庣殑璁粌鏃ュ織浼氬浠藉埌鎸囧畾鐨?Ceph 璺緞銆?
```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', out_dir='s3://openmmlab/mmdetection3d'),
    ])
```

鎮ㄨ繕鍙互閫氳繃璁剧疆 `keep_local = False` 澶囦唤鍒版寚瀹氱殑 Ceph 璺緞鍚庡垹闄ゆ湰鍦拌缁冩棩蹇椼€?
```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', out_dir='s3://openmmlab/mmdetection3d', keep_local=False),
    ])
```

