# 鑷畾涔夋暟鎹澶勭悊娴佺▼

## 鏁版嵁棰勫鐞嗘祦绋嬬殑璁捐

閬靛惊涓€鑸儻渚嬶紝鎴戜滑浣跨敤 `Dataset` 鍜?`DataLoader` 鏉ヨ皟鐢ㄥ涓繘绋嬭繘琛屾暟鎹殑鍔犺浇銆俙Dataset` 灏嗕細杩斿洖涓庢ā鍨嬪墠鍚戜紶鎾殑鍙傛暟鎵€瀵瑰簲鐨勬暟鎹」鏋勬垚鐨勫瓧鍏搞€傚洜涓虹洰鏍囨娴嬩腑鐨勬暟鎹殑灏哄鍙兘鏃犳硶淇濇寔涓€鑷达紙濡傜偣浜戜腑鐐圭殑鏁伴噺銆佺湡瀹炴爣娉ㄦ鐨勫昂瀵哥瓑锛夛紝鎴戜滑鍦?MMCV 涓紩鍏ヤ竴涓?`DataContainer` 绫诲瀷锛屾潵甯姪鏀堕泦鍜屽垎鍙戜笉鍚屽昂瀵哥殑鏁版嵁銆傝鍙傝€僛姝ゅ](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py)鑾峰彇鏇村缁嗚妭銆?
鏁版嵁棰勫鐞嗘祦绋嬪拰鏁版嵁闆嗕箣闂存槸浜掔浉鍒嗙鐨勪袱涓儴鍒嗭紝閫氬父鏁版嵁闆嗗畾涔変簡濡備綍澶勭悊鏍囨敞淇℃伅锛岃€屾暟鎹澶勭悊娴佺▼瀹氫箟浜嗗噯澶囨暟鎹」瀛楀吀鐨勬墍鏈夋楠ゃ€傛暟鎹泦棰勫鐞嗘祦绋嬪寘鍚竴绯诲垪鐨勬搷浣滐紝姣忎釜鎿嶄綔灏嗕竴涓瓧鍏镐綔涓鸿緭鍏ワ紝骞惰緭鍑哄簲鐢ㄤ簬涓嬩竴涓浆鎹㈢殑涓€涓柊鐨勫瓧鍏搞€?
鎴戜滑灏嗗湪涓嬪浘涓睍绀轰竴涓渶缁忓吀鐨勬暟鎹泦棰勫鐞嗘祦绋嬶紝鍏朵腑钃濊壊妗嗚〃绀洪澶勭悊娴佺▼涓殑鍚勯」鎿嶄綔銆傞殢鐫€棰勫鐞嗙殑杩涜锛屾瘡涓€涓搷浣滈兘浼氭坊鍔犳柊鐨勯敭鍊硷紙鍥句腑鏍囪涓虹豢鑹诧級鍒拌緭鍑哄瓧鍏镐腑锛屾垨鑰呮洿鏂板綋鍓嶅瓨鍦ㄧ殑閿€硷紙鍥句腑鏍囪涓烘鑹诧級銆?
![](../../../resources/data_pipeline.png)

棰勫鐞嗘祦绋嬩腑鐨勫悇椤规搷浣滀富瑕佸垎涓烘暟鎹姞杞姐€侀澶勭悊銆佹牸寮忓寲銆佹祴璇曟椂鐨勬暟鎹寮恒€?
鎺ヤ笅鏉ュ皢灞曠ず涓€涓敤浜?PointPillars 妯″瀷鐨勬暟鎹泦棰勫鐞嗘祦绋嬬殑渚嬪瓙銆?
```python
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
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
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        pts_scale_ratio=1.0,
        flip=False,
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
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
```

瀵逛簬姣忛」鎿嶄綔锛屾垜浠皢鍒楀嚭鐩稿叧鐨勮娣诲姞/鏇存柊/绉婚櫎鐨勫瓧鍏搁」銆?
### 鏁版嵁鍔犺浇

`LoadPointsFromFile`

- 娣诲姞锛歱oints

`LoadPointsFromMultiSweeps`

- 鏇存柊锛歱oints

`LoadAnnotations3D`

- 娣诲姞锛歡t_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, pts_instance_mask, pts_semantic_mask, bbox3d_fields, pts_mask_fields, pts_seg_fields

### 棰勫鐞?
`GlobalRotScaleTrans`

- 娣诲姞锛歱cd_trans, pcd_rotation, pcd_scale_factor
- 鏇存柊锛歱oints, \*bbox3d_fields

`RandomFlip3D`

- 娣诲姞锛歠lip, pcd_horizontal_flip, pcd_vertical_flip
- 鏇存柊锛歱oints, \*bbox3d_fields

`PointsRangeFilter`

- 鏇存柊锛歱oints

`ObjectRangeFilter`

- 鏇存柊锛歡t_bboxes_3d, gt_labels_3d

`ObjectNameFilter`

- 鏇存柊锛歡t_bboxes_3d, gt_labels_3d

`PointShuffle`

- 鏇存柊锛歱oints

`PointsRangeFilter`

- 鏇存柊锛歱oints

### 鏍煎紡鍖?
`DefaultFormatBundle3D`

- 鏇存柊锛歱oints, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels

`Collect3D`

- 娣诲姞锛歩mg_meta 锛堢敱 `meta_keys` 鎸囧畾鐨勯敭鍊兼瀯鎴愮殑 img_meta锛?- 绉婚櫎锛氭墍鏈夐櫎 `keys` 鎸囧畾鐨勯敭鍊间互澶栫殑鍏朵粬閿€?
### 娴嬭瘯鏃剁殑鏁版嵁澧炲己

`MultiScaleFlipAug`

- 鏇存柊: scale, pcd_scale_factor, flip, flip_direction, pcd_horizontal_flip, pcd_vertical_flip 锛堜笌杩欎簺鎸囧畾鐨勫弬鏁板搴旂殑澧炲己鍚庣殑鏁版嵁鍒楄〃锛?
## 鎵╁睍骞朵娇鐢ㄨ嚜瀹氫箟鏁版嵁闆嗛澶勭悊鏂规硶

1. 鍦ㄤ换鎰忔枃浠朵腑鍐欏叆鏂扮殑鏁版嵁闆嗛澶勭悊鏂规硶锛屽 `my_pipeline.py`锛岃棰勫鐞嗘柟娉曠殑杈撳叆鍜岃緭鍑哄潎涓哄瓧鍏?
   ```python
   from mmdet.datasets import PIPELINES

   @PIPELINES.register_module()
   class MyTransform:

       def __call__(self, results):
           results['dummy'] = True
           return results
   ```

2. 瀵煎叆鏂扮殑棰勫鐞嗘柟娉曠被

   ```python
   from .my_pipeline import MyTransform
   ```

3. 鍦ㄩ厤缃枃浠朵腑浣跨敤璇ユ暟鎹泦棰勫鐞嗘柟娉?
   ```python
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
       dict(type='MyTransform'),
       dict(type='PointShuffle'),
       dict(type='DefaultFormatBundle3D', class_names=class_names),
       dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
   ]
   ```

