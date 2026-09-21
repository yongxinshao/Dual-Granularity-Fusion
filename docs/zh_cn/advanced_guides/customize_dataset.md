# 鑷畾涔夋暟鎹泦

鍦ㄦ湰鑺備腑锛屾偍灏嗕簡瑙ｅ浣曚娇鐢ㄨ嚜瀹氫箟鏁版嵁闆嗚缁冨拰娴嬭瘯棰勫畾涔夋ā鍨嬨€?
鍩烘湰姝ラ濡備笅锛?
1. 鍑嗗鏁版嵁
2. 鍑嗗閰嶇疆鏂囦欢
3. 鍦ㄨ嚜瀹氫箟鏁版嵁闆嗕笂璁粌锛屾祴璇曞拰鎺ㄧ悊妯″瀷

## 鏁版嵁鍑嗗

鐞嗘兂鎯呭喌涓嬫垜浠彲浠ラ噸鏂扮粍缁囪嚜瀹氫箟鐨勫師濮嬫暟鎹苟灏嗘爣娉ㄦ牸寮忚浆鎹㈡垚 KITTI 椋庢牸銆備絾鏄紝鑰冭檻鍒板浜庤嚜瀹氫箟鏁版嵁闆嗚€岃█锛孠ITTI 鏍煎紡鐨勬牎鍑嗘枃浠跺拰 3D 鏍囨敞闅句互鑾峰緱锛屽洜姝ゆ垜浠湪鏂囨。涓粙缁嶅熀鏈殑鏁版嵁鏍煎紡銆?
### 鍩烘湰鏁版嵁鏍煎紡

#### 鐐逛簯鏍煎紡

鐩墠锛屾垜浠彧鏀寔 `.bin` 鏍煎紡鐨勭偣浜戠敤浜庤缁冨拰鎺ㄧ悊銆傚湪璁粌鑷繁鐨勬暟鎹泦涔嬪墠锛岄渶瑕佸皢鍏跺畠鏍煎紡鐨勭偣浜戞枃浠惰浆鎹㈡垚 `.bin` 鏂囦欢銆傚父瑙佺殑鐐逛簯鏁版嵁鏍煎紡鍖呮嫭 `.pcd` 鍜?`.las`锛屾垜浠垪涓句簡涓€浜涘紑婧愬伐鍏蜂綔涓哄弬鑰冦€?
1. `.pcd` 杞崲鎴?`.bin`锛歨ttps://github.com/DanielPollithy/pypcd

- 鎮ㄥ彲浠ラ€氳繃浠ヤ笅鎸囦护瀹夎 `pypcd`锛?
  ```bash
  pip install git+https://github.com/DanielPollithy/pypcd.git
  ```

- 鎮ㄥ彲浠ヤ娇鐢ㄤ互涓嬭剼鏈鍙?`.pcd` 鏂囦欢锛屽苟灏嗗叾杞崲鎴?`.bin` 鏍煎紡鏉ヤ繚瀛橈細

  ```python
  import numpy as np
  from pypcd import pypcd

  pcd_data = pypcd.PointCloud.from_path('point_cloud_data.pcd')
  points = np.zeros([pcd_data.width, 4], dtype=np.float32)
  points[:, 0] = pcd_data.pc_data['x'].copy()
  points[:, 1] = pcd_data.pc_data['y'].copy()
  points[:, 2] = pcd_data.pc_data['z'].copy()
  points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
  with open('point_cloud_data.bin', 'wb') as f:
      f.write(points.tobytes())
  ```

2. `.las` 杞崲鎴?`.bin`锛氬父瑙佺殑杞崲娴佺▼涓?`.las -> .pcd -> .bin`锛宍.las -> .pcd` 鐨勮浆鎹㈠彲浠ョ敤璇宸ュ叿](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor)瀹炵幇銆?
#### 鏍囩鏍煎紡

鏈€鍩烘湰鐨勪俊鎭細姣忎釜鍦烘櫙鐨?3D 杈圭晫妗嗗拰绫诲埆鏍囩搴旇鍖呭惈鍦?`.txt` 鏍囨敞鏂囦欢涓€傛瘡涓€琛屼唬琛ㄧ壒瀹氬満鏅殑涓€涓?3D 妗嗭紝濡備笅鎵€绀猴細

```
# 鏍煎紡锛歔x, y, z, dx, dy, dz, yaw, category_name]
1.23 1.42 0.23 3.96 1.65 1.55 1.56 Car
3.51 2.15 0.42 1.05 0.87 1.86 1.23 Pedestrian
...
```

**娉ㄦ剰**锛氬浜庤嚜瀹氫箟鏁版嵁闆嗙殑璇勪及鎴戜滑鐩墠鍙敮鎸?KITTI 璇勪及鏂规硶銆?
3D 妗嗗簲瀛樺偍鍦ㄧ粺涓€鐨?3D 鍧愭爣绯讳腑銆?
#### 鏍″噯鏍煎紡

瀵逛簬姣忎釜婵€鍏夐浄杈炬敹闆嗙殑鐐逛簯鏁版嵁锛岄€氬父浼氳繘琛岃瀺鍚堝苟杞崲鍒扮壒瀹氱殑婵€鍏夐浄杈惧潗鏍囩郴銆傚洜姝わ紝鏍″噯淇℃伅鏂囦欢涓€氬父搴旇鍖呭惈姣忎釜鐩告満鐨勫唴鍙傜煩闃靛拰婵€鍏夐浄杈惧埌姣忎釜鐩告満鐨勫鍙傝浆鎹㈢煩闃碉紝骞朵繚瀛樺湪 `.txt` 鏍″噯鏂囦欢涓紝鍏朵腑 `Px` 琛ㄧず `camera_x` 鐨勫唴鍙傜煩闃碉紝`lidar2camx` 琛ㄧず `lidar` 鍒?`camera_x` 鐨勫鍙傝浆鎹㈢煩闃点€?
```
P0
P1
P2
P3
P4
...
lidar2cam0
lidar2cam1
lidar2cam2
lidar2cam3
lidar2cam4
...
```

### 鍘熷鏁版嵁缁撴瀯

#### 鍩轰簬婵€鍏夐浄杈剧殑 3D 妫€娴?
鍩轰簬婵€鍏夐浄杈剧殑 3D 鐩爣妫€娴嬪師濮嬫暟鎹€氬父缁勭粐鎴愬涓嬫牸寮忥紝鍏朵腑 `ImageSets` 鍖呭惈鍒掑垎鏂囦欢锛屾寚鏄庡摢浜涙枃浠跺睘浜庤缁?楠岃瘉闆嗭紝`points` 鍖呭惈瀛樺偍鎴?`.bin` 鏍煎紡鐨勭偣浜戞暟鎹紝`labels` 鍖呭惈 3D 妫€娴嬬殑鏍囩鏂囦欢銆?
```
mmdetection3d
鈹溾攢鈹€ mmdet3d
鈹溾攢鈹€ tools
鈹溾攢鈹€ configs
鈹溾攢鈹€ data
鈹?  鈹溾攢鈹€ custom
鈹?  鈹?  鈹溾攢鈹€ ImageSets
鈹?  鈹?  鈹?  鈹溾攢鈹€ train.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ val.txt
鈹?  鈹?  鈹溾攢鈹€ points
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.bin
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.bin
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹?  鈹溾攢鈹€ labels
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
```

#### 鍩轰簬瑙嗚鐨?3D 妫€娴?
鍩轰簬瑙嗚鐨?3D 鐩爣妫€娴嬪師濮嬫暟鎹€氬父缁勭粐鎴愬涓嬫牸寮忥紝鍏朵腑 `ImageSets` 鍖呭惈鍒掑垎鏂囦欢锛屾寚鏄庡摢浜涙枃浠跺睘浜庤缁?楠岃瘉闆嗭紝`images` 鍖呭惈鏉ヨ嚜涓嶅悓鐩告満鐨勫浘鍍忥紝渚嬪 `camera_x` 鑾峰緱鐨勫浘鍍忓簲鏀惧湪 `images/images_x` 涓嬶紝`calibs` 鍖呭惈鏍″噯淇℃伅鏂囦欢锛屽叾涓瓨鍌ㄤ簡姣忎釜鐩告満鐨勫唴鍙傜煩闃碉紝`labels` 鍖呭惈 3D 妫€娴嬬殑鏍囩鏂囦欢銆?
```
mmdetection3d
鈹溾攢鈹€ mmdet3d
鈹溾攢鈹€ tools
鈹溾攢鈹€ configs
鈹溾攢鈹€ data
鈹?  鈹溾攢鈹€ custom
鈹?  鈹?  鈹溾攢鈹€ ImageSets
鈹?  鈹?  鈹?  鈹溾攢鈹€ train.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ val.txt
鈹?  鈹?  鈹溾攢鈹€ calibs
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹?  鈹溾攢鈹€ images
鈹?  鈹?  鈹?  鈹溾攢鈹€ images_0
鈹?  鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.png
鈹?  鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.png
鈹?  鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹?  鈹?  鈹溾攢鈹€ images_1
鈹?  鈹?  鈹?  鈹溾攢鈹€ images_2
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹?  鈹溾攢鈹€ labels
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
```

#### 澶氭ā鎬?3D 妫€娴?
澶氭ā鎬?3D 鐩爣妫€娴嬪師濮嬫暟鎹€氬父缁勭粐鎴愬涓嬫牸寮忋€備笉鍚屼簬鍩轰簬瑙嗚鐨?3D 鐩爣妫€娴嬶紝`calibs` 閲岀殑鏍″噯淇℃伅鏂囦欢瀛樺偍浜嗘瘡涓浉鏈虹殑鍐呭弬鐭╅樀鍜屽鍙傜煩闃点€?
```
mmdetection3d
鈹溾攢鈹€ mmdet3d
鈹溾攢鈹€ tools
鈹溾攢鈹€ configs
鈹溾攢鈹€ data
鈹?  鈹溾攢鈹€ custom
鈹?  鈹?  鈹溾攢鈹€ ImageSets
鈹?  鈹?  鈹?  鈹溾攢鈹€ train.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ val.txt
鈹?  鈹?  鈹溾攢鈹€ calibs
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹?  鈹溾攢鈹€ points
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.bin
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.bin
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹?  鈹溾攢鈹€ images
鈹?  鈹?  鈹?  鈹溾攢鈹€ images_0
鈹?  鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.png
鈹?  鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.png
鈹?  鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹?  鈹?  鈹溾攢鈹€ images_1
鈹?  鈹?  鈹?  鈹溾攢鈹€ images_2
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹?  鈹溾攢鈹€ labels
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
```

#### 鍩轰簬婵€鍏夐浄杈剧殑 3D 璇箟鍒嗗壊

鍩轰簬婵€鍏夐浄杈剧殑 3D 璇箟鍒嗗壊鍘熷鏁版嵁閫氬父缁勭粐鎴愬涓嬫牸寮忥紝鍏朵腑 `ImageSets` 鍖呭惈鍒掑垎鏂囦欢锛屾寚鏄庡摢浜涙枃浠跺睘浜庤缁?楠岃瘉闆嗭紝`points` 鍖呭惈鐐逛簯鏁版嵁锛宍semantic_mask` 鍖呭惈閫愮偣绾ф爣绛俱€?
```
mmdetection3d
鈹溾攢鈹€ mmdet3d
鈹溾攢鈹€ tools
鈹溾攢鈹€ configs
鈹溾攢鈹€ data
鈹?  鈹溾攢鈹€ custom
鈹?  鈹?  鈹溾攢鈹€ ImageSets
鈹?  鈹?  鈹?  鈹溾攢鈹€ train.txt
鈹?  鈹?  鈹?  鈹溾攢鈹€ val.txt
鈹?  鈹?  鈹溾攢鈹€ points
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.bin
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.bin
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹?  鈹溾攢鈹€ semantic_mask
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000000.bin
鈹?  鈹?  鈹?  鈹溾攢鈹€ 000001.bin
鈹?  鈹?  鈹?  鈹溾攢鈹€ ...
```

### 鏁版嵁杞崲

鎸夌収鎴戜滑鐨勮鏄庡噯澶囧ソ鍘熷鏁版嵁鍚庯紝鎮ㄥ彲浠ョ洿鎺ヤ娇鐢ㄤ互涓嬪懡浠ょ敓鎴愯缁?楠岃瘉淇℃伅鏂囦欢銆?
```bash
python tools/create_data.py custom --root-path ./data/custom --out-dir ./data/custom --extra-tag custom
```

## 鑷畾涔夋暟鎹泦绀轰緥

鍦ㄥ畬鎴愭暟鎹噯澶囧悗锛屾垜浠彲浠ュ湪 `mmdet3d/datasets/my_dataset.py` 涓垱寤轰竴涓柊鐨勬暟鎹泦鏉ュ姞杞芥暟鎹€?
```python
import mmengine

from mmdet3d.registry import DATASETS
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class MyDataset(Det3DDataset):

    # 鏇挎崲鎴愯嚜瀹氫箟 pkl 淇℃伅鏂囦欢閲岀殑鎵€鏈夌被鍒?    METAINFO = {
        'classes': ('Pedestrian', 'Cyclist', 'Car')
    }

    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # 绌哄疄渚?            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # 杩囨护鎺夋病鏈夊湪璁粌涓娇鐢ㄧ殑绫诲埆
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
```

鏁版嵁棰勫鐞嗗悗锛岀敤鎴峰彲浠ラ€氳繃浠ヤ笅涓や釜姝ラ鏉ヨ缁冭嚜瀹氫箟鏁版嵁闆嗭細

1. 淇敼閰嶇疆鏂囦欢鏉ヤ娇鐢ㄨ嚜瀹氫箟鏁版嵁闆嗐€?2. 楠岃瘉鑷畾涔夋暟鎹泦鏍囨敞鐨勬纭€с€?
杩欓噷鎴戜滑浠ュ湪鑷畾涔夋暟鎹泦涓婅缁?PointPillars 涓轰緥锛?
### 鍑嗗閰嶇疆

杩欓噷鎴戜滑婕旂ず涓€涓函鐐逛簯璁粌鐨勯厤缃ず渚嬶細

#### 鍑嗗鏁版嵁闆嗛厤缃?
鍦?`configs/_base_/datasets/custom.py` 涓細

```python
# 鏁版嵁闆嗚缃?dataset_type = 'MyDataset'
data_root = 'data/custom/'
class_names = ['Pedestrian', 'Cyclist', 'Car']  # 鏇挎崲鎴愭偍鐨勬暟鎹泦绫诲埆
point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # 鏍规嵁鎮ㄧ殑鏁版嵁闆嗚繘琛岃皟鏁?input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # 鏇挎崲鎴愭偍鐨勭偣浜戞暟鎹淮搴?        use_dim=4),  # 鏇挎崲鎴愬湪璁粌鍜屾帹鐞嗘椂瀹為檯浣跨敤鐨勭淮搴?    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
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
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # 鏇挎崲鎴愭偍鐨勭偣浜戞暟鎹淮搴?        use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# 涓哄彲瑙嗗寲闃舵鐨勬暟鎹拰 GT 鍔犺浇鏋勯€犳祦姘寸嚎
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points']),
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
            ann_file='custom_infos_train.pkl',  # 鎸囧畾鎮ㄧ殑璁粌 pkl 淇℃伅
            data_prefix=dict(pts='points'),
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
        data_prefix=dict(pts='points'),
        ann_file='custom_infos_val.pkl',  # 鎸囧畾鎮ㄧ殑楠岃瘉 pkl 淇℃伅
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'custom_infos_val.pkl',  # 鎸囧畾鎮ㄧ殑楠岃瘉 pkl 淇℃伅
    metric='bbox')
```

#### 鍑嗗妯″瀷閰嶇疆

瀵逛簬鍩轰簬浣撶礌鍖栫殑妫€娴嬪櫒濡?SECOND锛孭ointPillars 鍙?CenterPoint锛岀偣浜戣寖鍥达紙point cloud range锛夊拰浣撶礌澶у皬锛坴oxel size锛夊簲璇ユ牴鎹偍鐨勬暟鎹泦鍋氳皟鏁淬€傜悊璁轰笂锛宍voxel_size` 鍜?`point_cloud_range` 鐨勮缃槸鐩稿叧鑱旂殑銆傝缃緝灏忕殑 `voxel_size` 灏嗗鍔犱綋绱犳暟浠ュ強鐩稿簲鐨勫唴瀛樻秷鑰椼€傛澶栵紝闇€瑕佹敞鎰忎互涓嬮棶棰橈細

濡傛灉灏?`point_cloud_range` 鍜?`voxel_size` 鍒嗗埆璁剧疆鎴?`[0, -40, -3, 70.4, 40, 1]` 鍜?`[0.05, 0.05, 0.1]`锛岄偅涔堜腑闂寸壒寰佸浘鐨勫舰鐘跺簲璇ヤ负 `[(1-(-3))/0.1+1, (40-(-40))/0.05, (70.4-0)/0.05]=[41, 1600, 1408]`銆傛洿鏀?`point_cloud_range` 鏃讹紝璇疯寰椾緷鎹?`voxel_size` 鏇存敼 `middle_encoder` 閲屼腑闂寸壒寰佸浘鐨勫舰鐘躲€?
鍏充簬 `anchor_range` 鐨勮缃紝涓€鑸渶瑕佹牴鎹暟鎹泦鍋氳皟鏁淬€傞渶瑕佹敞鎰忕殑鏄紝`z` 鍊奸渶瑕佹牴鎹偣浜戠殑浣嶇疆鍋氱浉搴旇皟鏁达紝鍏蜂綋璇峰弬鑰冩 [issue](https://github.com/open-mmlab/mmdetection3d/issues/986)銆?
鍏充簬 `anchor_size` 鐨勮缃紝閫氬父闇€瑕佽绠楁暣涓缁冮泦涓洰鏍囩殑闀裤€佸銆侀珮鐨勫钩鍧囧€间綔涓?`anchor_size`锛屼互鑾峰緱鏈€濂界殑缁撴灉銆?
鍦?`configs/_base_/models/pointpillars_hv_secfpn_custom.py` 涓細

```python
voxel_size = [0.16, 0.16, 4]  # 鏍规嵁鎮ㄧ殑鏁版嵁闆嗗仛璋冩暣
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # 鏍规嵁鎮ㄧ殑鏁版嵁闆嗗仛璋冩暣
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    # `output_shape` 闇€瑕佹牴鎹?`point_cloud_range` 鍜?`voxel_size` 鍋氱浉搴旇皟鏁?    middle_encoder=dict(
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
        # 鏍规嵁鎮ㄧ殑鏁版嵁闆嗚皟鏁?`ranges` 鍜?`sizes`
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -1.78, 69.12, 39.68, -1.78],
            ],
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
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # 妯″瀷璁粌鍜屾祴璇曡缃?    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
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

#### 鍑嗗鏁翠綋閰嶇疆

鎴戜滑灏嗕笂杩扮殑鎵€鏈夐厤缃粍鍚堝湪 `configs/pointpillars/pointpillars_hv_secfpn_8xb6_custom.py` 鏂囦欢涓細

```python
_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_custom.py',
    '../_base_/datasets/custom.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]
```

#### 鍙鍖栨暟鎹泦锛堝彲閫夛級

涓轰簡楠岃瘉鍑嗗鐨勬暟鎹拰閰嶇疆鏄惁姝ｇ‘锛屾垜浠缓璁湪璁粌鍜岄獙璇佸墠浣跨敤 `tools/misc/browse_dataset.py` 鑴氭湰鍙鍖栨暟鎹泦鍜屾爣娉ㄣ€傛洿澶氱粏鑺傝鍙傝€僛鍙鍖栨枃妗(https://mmdetection3d.readthedocs.io/zh_CN/dev-1.x/user_guides/visualization.html)銆?
## 璇勪及

鍑嗗濂芥暟鎹拰閰嶇疆涔嬪悗锛屾偍鍙互閬靛惊鎴戜滑鐨勬枃妗ｇ洿鎺ヨ繍琛岃缁?娴嬭瘯鑴氭湰銆?
**娉ㄦ剰**锛氭垜浠负鑷畾涔夋暟鎹泦鎻愪緵浜?KITTI 椋庢牸鐨勮瘎浼板疄鐜版柟娉曘€傚湪鏁版嵁闆嗛厤缃腑闇€瑕佸寘鍚涓嬪唴瀹癸細

```python
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'custom_infos_val.pkl',  # 鎸囧畾鎮ㄧ殑楠岃瘉 pkl 淇℃伅
    metric='bbox')
```

