# 瀛︿範閰嶇疆鏂囦欢

MMDetection3D 鍜屽叾浠?OpenMMLab 浠撳簱浣跨敤 [MMEngine 鐨勯厤缃枃浠剁郴缁焆(https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html)銆傚畠鍏锋湁妯″潡鍖栧拰缁ф壙鎬ц璁★紝浠ヤ究浜庤繘琛屽悇绉嶅疄楠屻€?
## 閰嶇疆鏂囦欢鐨勫唴瀹?
MMDetection3D 閲囩敤妯″潡鍖栬璁★紝鎵€鏈夊姛鑳界殑妯″潡鍙互閫氳繃閰嶇疆鏂囦欢杩涜閰嶇疆銆備互 PointPillars 涓轰緥锛屾垜浠皢鏍规嵁涓嶅悓鐨勫姛鑳芥ā鍧椾粙缁嶉厤缃枃浠剁殑鍚勪釜瀛楁銆?
### 妯″瀷閰嶇疆

鍦?MMDetection3D 鐨勯厤缃腑锛屾垜浠娇鐢?`model` 瀛楁鏉ラ厤缃娴嬬畻娉曠殑缁勪欢銆傞櫎浜?`voxel_encoder`锛宍backbone` 绛夌缁忕綉缁滅粍浠跺锛岃繕闇€瑕?`data_preprocessor`锛宍train_cfg` 鍜?`test_cfg`銆俙data_preprocessor` 璐熻矗瀵规暟鎹姞杞藉櫒锛坉ataloader锛夎緭鍑虹殑姣忎竴鎵规暟鎹繘琛岄澶勭悊銆傛ā鍨嬮厤缃腑鐨?`train_cfg` 鍜?`test_cfg` 鐢ㄤ簬璁剧疆璁粌鍜屾祴璇曠粍浠剁殑瓒呭弬鏁般€?
```python
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            voxel_size=[0.16, 0.16, 4],
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.16, 0.16, 4],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    train_cfg=dict(
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
```

### 鏁版嵁闆嗗拰璇勬祴鍣ㄩ厤缃?
鍦ㄤ娇鐢╗鎵ц鍣紙Runner锛塢(https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html)杩涜璁粌銆佹祴璇曞拰楠岃瘉鏃讹紝鎴戜滑闇€瑕侀厤缃甗鏁版嵁鍔犺浇鍣╙(https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader)銆傛瀯寤烘暟鎹姞杞藉櫒闇€瑕佽缃暟鎹泦鍜屾暟鎹鐞嗘祦绋嬨€傜敱浜庤繖閮ㄥ垎鐨勯厤缃緝涓哄鏉傦紝鎴戜滑浣跨敤涓棿鍙橀噺鏉ョ畝鍖栨暟鎹姞杞藉櫒閰嶇疆鐨勭紪鍐欍€?
```python
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=15, Cyclist=15),
    points_loader=dict(
        type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4))

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
```

[璇勬祴鍣╙(https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluation.html)鐢ㄤ簬璁＄畻璁粌妯″瀷鍦ㄩ獙璇佸拰娴嬭瘯鏁版嵁闆嗕笂鐨勬寚鏍囥€傝瘎娴嬪櫒鐨勯厤缃敱涓€涓垨涓€缁勮瘎浠锋寚鏍囬厤缃粍鎴愶細

```python
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator
```

鐢变簬娴嬭瘯鏁版嵁闆嗘病鏈夋爣娉ㄦ枃浠讹紝鍥犳 MMDetection3D 涓殑 test_dataloader 鍜?test_evaluator 閰嶇疆閫氬父绛変簬 val銆傚鏋滄偍鎯宠淇濆瓨鍦ㄦ祴璇曟暟鎹泦涓婄殑妫€娴嬬粨鏋滐紝鍒欏彲浠ュ儚杩欐牱缂栧啓閰嶇疆锛?
```python
# 鍦ㄦ祴璇曢泦涓婃帹鐞嗭紝
# 骞跺皢妫€娴嬬粨鏋滆浆鎹㈡牸寮忎互鐢ㄤ簬鎻愪氦缁撴灉
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='testing/velodyne_reduced'),
        ann_file='kitti_infos_test.pkl',
        load_eval_anns=False,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
test_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_test.pkl',
    metric='bbox',
    format_only=True,
    submission_prefix='results/kitti-3class/kitti_results')
```

### 璁粌鍜屾祴璇曢厤缃?
MMEngine 鐨勬墽琛屽櫒浣跨敤寰幆锛圠oop锛夋潵鎺у埗璁粌锛岄獙璇佸拰娴嬭瘯杩囩▼銆傜敤鎴峰彲浠ヤ娇鐢ㄨ繖浜涘瓧娈佃缃渶澶ц缁冭疆娆″拰楠岃瘉闂撮殧锛?
```python
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

### 浼樺寲閰嶇疆

`optim_wrapper` 鏄厤缃紭鍖栫浉鍏宠缃殑瀛楁銆備紭鍖栧櫒灏佽涓嶄粎鎻愪緵浜嗕紭鍖栧櫒鐨勫姛鑳斤紝杩樻敮鎸佹搴﹁鍓€佹贩鍚堢簿搴﹁缁冪瓑鍔熻兘銆傛洿澶氬唴瀹硅鐪媅浼樺寲鍣ㄥ皝瑁呮暀绋媇(https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html)銆?
```python
optim_wrapper = dict(  # 浼樺寲鍣ㄥ皝瑁呴厤缃?    type='OptimWrapper',  # 浼樺寲鍣ㄥ皝瑁呯被鍨嬶紝鍒囨崲鍒?AmpOptimWrapper 鍚姩娣峰悎绮惧害璁粌
    optimizer=dict(  # 浼樺寲鍣ㄩ厤缃€傛敮鎸?PyTorch 鐨勫悇绉嶄紭鍖栧櫒锛岃鍙傝€?https://pytorch.org/docs/stable/optim.html#algorithms
        type='AdamW', lr=0.001, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))  # 姊害瑁佸壀閫夐」銆傝缃负 None 绂佺敤姊害瑁佸壀銆備娇鐢ㄦ柟娉曡瑙?https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html
```

`param_scheduler` 鏄厤缃皟鏁翠紭鍖栧櫒瓒呭弬鏁帮紙渚嬪瀛︿範鐜囧拰鍔ㄩ噺锛夌殑瀛楁銆傜敤鎴峰彲浠ョ粍鍚堝涓皟搴﹀櫒鏉ュ垱寤烘墍闇€瑕佺殑鍙傛暟璋冩暣绛栫暐銆傛洿澶氫俊鎭鍙傝€僛鍙傛暟璋冨害鍣ㄦ暀绋媇(https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html)鍜孾鍙傛暟璋冨害鍣?API 鏂囨。](https://mmengine.readthedocs.io/zh_CN/latest/api/optim.html#scheduler)銆?
```python
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=32,
        eta_min=0.01,
        begin=0,
        end=32,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=48,
        eta_min=1.0000000000000001e-07,
        begin=32,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=32,
        eta_min=0.8947368421052632,
        begin=0,
        end=32,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=48,
        eta_min=1,
        begin=32,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
]
```

### 閽╁瓙閰嶇疆

鐢ㄦ埛鍙互鍦ㄨ缁冦€侀獙璇佸拰娴嬭瘯寰幆涓婃坊鍔犻挬瀛愶紝浠庤€屽湪杩愯鏈熼棿鎻掑叆涓€浜涙搷浣溿€傛湁涓ょ涓嶅悓鐨勯挬瀛愬瓧娈碉紝涓€绉嶆槸 `default_hooks`锛屽彟涓€绉嶆槸 `custom_hooks`銆?
`default_hooks` 鏄竴涓挬瀛愰厤缃瓧鍏革紝骞朵笖杩欎簺閽╁瓙鏄繍琛屾椂鎵€闇€瑕佺殑銆傚畠浠叿鏈夐粯璁や紭鍏堢骇锛屾槸涓嶉渶瑕佷慨鏀圭殑銆傚鏋滄湭璁剧疆锛屾墽琛屽櫒灏嗕娇鐢ㄩ粯璁ゅ€笺€傚鏋滆绂佺敤榛樿閽╁瓙锛岀敤鎴峰彲浠ュ皢鍏堕厤缃缃负 `None`銆?
```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
```

`custom_hooks` 鏄竴涓敱鍏朵粬閽╁瓙閰嶇疆缁勬垚鐨勫垪琛ㄣ€傜敤鎴峰彲浠ュ紑鍙戣嚜宸辩殑閽╁瓙骞跺皢鍏舵彃鍏ュ埌璇ュ瓧娈典腑銆?
```python
custom_hooks = []
```

### 杩愯閰嶇疆

```python
default_scope = 'mmdet3d'  # 瀵绘壘妯″潡鐨勯粯璁ゆ敞鍐屽櫒鍩熴€傝鍙傝€?https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=False,  # 鏄惁鍚敤 cudnn benchmark
    mp_cfg=dict(  # 澶氳繘绋嬮厤缃?        mp_start_method='fork',  # 浣跨敤 fork 鏉ュ惎鍔ㄥ杩涚▼銆?fork' 閫氬父姣?'spawn' 鏇村揩锛屼絾鍙兘涓嶅畨鍏ㄣ€傝鍙傝€?https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # 鍏抽棴 opencv 鐨勫杩涚▼浠ラ伩鍏嶇郴缁熻秴璐熻嵎
    dist_cfg=dict(backend='nccl'))  # 鍒嗗竷寮忛厤缃?
vis_backends = [dict(type='LocalVisBackend')]  # 鍙鍖栧悗绔€傝鍙傝€?https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(
    type='LogProcessor',  # 鏃ュ織澶勭悊鍣ㄧ敤浜庡鐞嗚繍琛屾椂鏃ュ織
    window_size=50,  # 鏃ュ織鏁板€肩殑骞虫粦绐楀彛
    by_epoch=True)  # 鏄惁浣跨敤 epoch 鏍煎紡鐨勬棩蹇椼€傞渶瑕佷笌璁粌寰幆鐨勭被鍨嬩繚鎸佷竴鑷?
log_level = 'INFO'  # 鏃ュ織绛夌骇
load_from = None  # checkpoint path intentionally omitted from the public release

## 閰嶇疆鏂囦欢缁ф壙

鍦?`configs/_base_` 鏂囦欢澶逛笅鏈?4 涓熀鏈粍浠剁被鍨嬶紝鍒嗗埆鏄細鏁版嵁闆嗭紙dataset锛夛紝妯″瀷锛坢odel锛夛紝璁粌绛栫暐锛坰chedule锛夊拰杩愯鏃剁殑榛樿璁剧疆锛坉efault runtime锛夈€傝澶氭柟娉曪紝濡?SECOND銆丳ointPillars銆丳artA2銆乂oteNet 閮借兘澶熷緢瀹规槗鍦版瀯寤哄嚭鏉ャ€傜敱 `_base_` 涓嬬殑缁勪欢缁勬垚鐨勯厤缃紝琚垜浠О涓?_鍘熷閰嶇疆锛坧rimitive锛塤銆?
瀵逛簬鍚屼竴涓枃浠跺す涓嬬殑鎵€鏈夐厤缃紝鎺ㄨ崘**鍙湁涓€涓?*瀵瑰簲鐨?_鍘熷閰嶇疆_ 鏂囦欢銆傛墍鏈夊叾浠栫殑閰嶇疆鏂囦欢閮藉簲璇ョ户鎵胯嚜杩欎釜 _鍘熷閰嶇疆_ 鏂囦欢銆傝繖鏍峰氨鑳戒繚璇侀厤缃枃浠剁殑鏈€澶х户鎵挎繁搴︿负 3銆?
涓轰簡渚夸簬鐞嗚В锛屾垜浠缓璁础鐚€呯户鎵跨幇鏈夋柟娉曘€備緥濡傦紝濡傛灉鍦?PointPillars 鐨勫熀纭€涓婂仛浜嗕竴浜涗慨鏀癸紝鐢ㄦ埛鍙互棣栧厛閫氳繃鎸囧畾 `_base_ = '../pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'` 鏉ョ户鎵垮熀纭€鐨?PointPillars 缁撴瀯锛岀劧鍚庝慨鏀归厤缃枃浠朵腑鐨勫繀瑕佸弬鏁颁互瀹屾垚缁ф壙銆?
濡傛灉鎮ㄥ湪鏋勫缓涓€涓笌浠讳綍鐜版湁鏂规硶閮戒笉鍏变韩鐨勫叏鏂版柟娉曪紝閭ｄ箞鍙互鍦?`configs` 鏂囦欢澶逛笅鍒涘缓涓€涓柊鐨勪緥濡?`xxx_rcnn` 鏂囦欢澶广€?
鏇村缁嗚妭璇峰弬鑰?[MMEngine 閰嶇疆鏂囦欢鏁欑▼](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html)銆?
閫氳繃璁剧疆 `_base_` 瀛楁锛屾垜浠彲浠ヨ缃綋鍓嶉厤缃枃浠剁户鎵胯嚜鍝簺鏂囦欢銆?
褰?`_base_` 涓烘枃浠惰矾寰勫瓧绗︿覆鏃讹紝琛ㄧず缁ф壙涓€涓厤缃枃浠剁殑鍐呭銆?
```python
_base_ = './pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
```

褰?`_base_` 鏄涓枃浠惰矾寰勭粍鎴愮殑鍒楄〃寮忥紝琛ㄧず缁ф壙澶氫釜鏂囦欢銆?
```python
_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]
```

濡傛灉闇€瑕佹娴嬮厤缃枃浠讹紝鍙互閫氳繃杩愯 `python tools/misc/print_config.py /PATH/TO/CONFIG` 鏉ユ煡鐪嬪畬鏁寸殑閰嶇疆銆?
### 蹇界暐鍩虹閰嶇疆鏂囦欢閲岀殑閮ㄥ垎瀛楁

鏈夋椂锛屾偍涔熻浼氳缃?`_delete_=True` 鍘诲拷鐣ュ熀纭€閰嶇疆鏂囦欢閲岀殑涓€浜涘瓧娈点€傛偍鍙互鍙傝€?[MMEngine 閰嶇疆鏂囦欢鏁欑▼](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html) 鏉ヨ幏寰椾竴浜涚畝鍗曠殑鎸囧銆?
鍦?MMDetection3D 閲岋紝渚嬪锛屼慨鏀逛互涓?PointPillars 閰嶇疆涓殑棰堥儴缃戠粶锛?
```python
model = dict(
    type='MVXFasterRCNN',
    data_preprocessor=dict(voxel_layer=dict(...)),
    pts_voxel_encoder=dict(...),
    pts_middle_encoder=dict(...),
    pts_backbone=dict(...),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_bbox_head=dict(...))
```

`FPN` 鍜?`SECONDFPN` 浣跨敤涓嶅悓鐨勫叧閿瓧鏉ユ瀯寤猴細

```python
_base_ = '../_base_/models/pointpillars_hv_fpn_nus.py'
model = dict(
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(...))
```

`_delete_=True` 灏嗕娇鐢ㄦ柊鐨勯敭鍘绘浛鎹?`pts_neck` 瀛楁鍐呮墍鏈夋棫鐨勯敭銆?
### 鍦ㄩ厤缃枃浠堕噷浣跨敤涓棿鍙橀噺

閰嶇疆鏂囦欢閲屼細浣跨敤涓€浜涗腑闂村彉閲忥紝渚嬪鏁版嵁闆嗛噷鐨?`train_pipeline`/`test_pipeline`銆傞渶瑕佹敞鎰忕殑鏄紝褰撲慨鏀瑰瓙閰嶇疆鏂囦欢涓殑涓棿鍙橀噺鏃讹紝鐢ㄦ埛闇€瑕佸啀娆″皢涓棿鍙橀噺浼犻€掑埌瀵瑰簲鐨勫瓧娈典腑銆備緥濡傦紝鎴戜滑鎯充娇鐢ㄥ灏哄害绛栫暐璁粌骞舵祴璇?PointPillars锛宍train_pipeline`/`test_pipeline` 鏄垜浠兂瑕佷慨鏀圭殑涓棿鍙橀噺銆?
```python
_base_ = './nus-3d.py'
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=[0.95, 1.0, 1.05],
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
```

鎴戜滑棣栧厛瀹氫箟鏂扮殑 `train_pipeline`/`test_pipeline`锛岀劧鍚庝紶閫掑埌鏁版嵁鍔犺浇鍣ㄥ瓧娈典腑銆?
### 澶嶇敤 \_base\_ 鏂囦欢涓殑鍙橀噺

濡傛灉鐢ㄦ埛甯屾湜澶嶇敤 base 鏂囦欢涓殑鍙橀噺锛屽垯鍙互閫氳繃浣跨敤 `{{_base_.xxx}}` 鑾峰彇瀵瑰簲鍙橀噺鐨勬嫹璐濄€備緥濡傦細

```python
_base_ = './pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'

a = {{_base_.model}}  # 鍙橀噺 `a` 绛変簬 `_base_` 涓畾涔夌殑 `model`
```

## 閫氳繃鑴氭湰鍙傛暟淇敼閰嶇疆

褰撲娇鐢?`tools/train.py` 鎴栬€?`tools/test.py` 鎻愪氦宸ヤ綔鏃讹紝鎮ㄥ彲浠ラ€氳繃鎸囧畾 `--cfg-options` 鏉ヤ慨鏀归厤缃枃浠躲€?
- 鏇存柊閰嶇疆瀛楀吀鐨勯敭鍊?
  鍙互鎸夌収鍘熷閰嶇疆鏂囦欢涓瓧鍏哥殑閿€奸『搴忔寚瀹氶厤缃€夐」銆備緥濡傦紝浣跨敤 `--cfg-options model.backbone.norm_eval=False` 灏嗘ā鍨嬩富骞茬綉缁滀腑鐨勬墍鏈?BN 妯″潡閮芥敼涓?`train` 妯″紡銆?
- 鏇存柊閰嶇疆鍒楄〃涓殑閿€?
  鍦ㄩ厤缃枃浠堕噷锛屼竴浜涢厤缃瓧鍏歌鍖呭惈鍦ㄥ垪琛ㄤ腑锛屼緥濡傦紝璁粌娴佺▼ `train_dataloader.dataset.pipeline` 閫氬父鏄竴涓垪琛紝渚嬪 `[dict(type='LoadPointsFromFile'), ...]`銆傚鏋滄偍鎯宠灏嗚缁冩祦绋嬩腑鐨?`'LoadPointsFromFile'` 鏀规垚 `'LoadPointsFromDict'`锛屾偍闇€瑕佹寚瀹?`--cfg-options data.train.pipeline.0.type=LoadPointsFromDict`銆?
- 鏇存柊鍒楄〃/鍏冪粍鐨勫€?
  濡傛灉瑕佹洿鏂扮殑鍊兼槸鍒楄〃鎴栧厓缁勩€備緥濡傦紝閰嶇疆鏂囦欢閫氬父璁剧疆 `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`銆傚鏋滄偍鎯宠鏀瑰彉杩欎釜鍧囧€硷紝鎮ㄩ渶瑕佹寚瀹?`--cfg-options model.data_preprocessor.mean="[127,127,127]"`銆傛敞鎰忥紝寮曞彿 `"` 鏄敮鎸佸垪琛?鍏冪粍鏁版嵁绫诲瀷鎵€蹇呴渶鐨勶紝骞朵笖鍦ㄦ寚瀹氬€肩殑寮曞彿鍐?*涓嶅厑璁?*鏈夌┖鏍笺€?
## 閰嶇疆鏂囦欢鍚嶇О椋庢牸

鎴戜滑閬靛惊浠ヤ笅鏍峰紡鏉ュ懡鍚嶉厤缃枃浠躲€傚缓璁础鐚€呴伒寰浉鍚岀殑椋庢牸銆?
```
{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{training dataset information}_{testing dataset information}.py
```

鏂囦欢鍚嶅垎涓轰簲涓儴鍒嗐€傛墍鏈夐儴鍒嗗拰缁勪欢鐢?`_` 杩炴帴锛屾瘡涓儴鍒嗘垨缁勪欢鍐呯殑鍗曡瘝搴旇鐢?`-` 杩炴帴銆?
- `{algorithm name}`锛氱畻娉曠殑鍚嶇О銆傚畠鍙互鏄娴嬪櫒鐨勫悕绉帮紝渚嬪 `pointpillars`銆乣fcos3d` 绛夈€?- `{model component names}`锛氱畻娉曚腑浣跨敤鐨勭粍浠跺悕绉帮紝濡?voxel_encoder銆乥ackbone銆乶eck 绛夈€備緥濡?`second_secfpn_head-dcn-circlenms` 琛ㄧず浣跨敤 SECOND 鐨?SparseEncoder锛孲ECONDFPN锛屼互鍙婂甫鏈?DCN 鍜?circle NMS 鐨勬娴嬪ご銆?- `{training settings}`锛氳缁冭缃殑淇℃伅锛屼緥濡傛壒閲忓ぇ灏忥紝鏁版嵁澧炲己锛屾崯澶卞嚱鏁扮瓥鐣ワ紝璋冨害鍣ㄤ互鍙婅缁冭疆娆?杩唬銆備緥濡?`8xb4-tta-cyclic-20e` 琛ㄧず浣跨敤 8 涓?gpu锛屾瘡涓?gpu 鏈?4 涓暟鎹牱鏈紝娴嬭瘯澧炲己锛屼綑寮﹂€€鐏涔犵巼锛岃缁?20 涓?epoch銆傜缉鍐欎粙缁嶏細
  - `{gpu x batch_per_gpu}`锛欸PU 鏁板拰姣忎釜 GPU 鐨勬牱鏈暟銆俙bN` 琛ㄧず姣忎釜 GPU 涓婄殑鎵归噺澶у皬涓?N銆備緥濡?`4xb4` 鏄?4 涓?GPU锛屾瘡涓?GPU 鏈?4 涓牱鏈暟鐨勭缉鍐欍€?  - `{schedule}`锛氳缁冩柟妗堬紝鍙€夐」涓?`schedule-2x`銆乣schedule-3x`銆乣cyclic-20e` 绛夈€俙schedule-2x` 鍜?`schedule-3x` 鍒嗗埆浠ｈ〃 24 epoch 鍜?36 epoch銆俙cyclic-20e` 琛ㄧず 20 epoch銆?- `{training dataset information}`锛氳缁冩暟鎹泦鍚嶏紝渚嬪 `kitti-3d-3class`锛宍nus-3d`锛宍s3dis-seg`锛宍scannet-seg`锛宍waymoD5-3d-car`銆傝繖閲?`3d` 琛ㄧず鏁版嵁闆嗙敤浜?3D 鐩爣妫€娴嬶紝`seg` 琛ㄧず鏁版嵁闆嗙敤浜庣偣浜戝垎鍓层€?- `{testing dataset information}`锛堝彲閫夛級锛氬綋妯″瀷鍦ㄤ竴涓暟鎹泦涓婅缁冿紝鍦ㄥ彟涓€涓暟鎹泦涓婃祴璇曟椂鐨勬祴璇曟暟鎹泦鍚嶃€傚鏋滄病鏈夋敞鏄庯紝鍒欒〃绀鸿缁冨拰娴嬭瘯鐨勬暟鎹泦绫诲瀷鐩稿悓銆?
