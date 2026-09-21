# 妯″瀷搴?
## 閫氱敤璁剧疆

- 浣跨敤鍒嗗竷寮忚缁冿紱
- 涓轰簡鍜屽叾浠栦唬鐮佸簱鍋氬叕骞冲姣旓紝鏈枃灞曠ず鐨勬槸浣跨敤 `torch.cuda.max_memory_allocated()` 鍦?8 涓?GPUs 涓婂緱鍒扮殑鏈€澶?GPU 鏄惧瓨鍗犵敤鍊硷紝闇€瑕佹敞鎰忕殑鏄紝杩欎簺鏄惧瓨鍗犵敤鍊奸€氬父灏忎簬 `nvidia-smi` 鏄剧ず鍑烘潵鐨勬樉瀛樺崰鐢ㄥ€硷紱
- 鍦ㄦā鍨嬪簱涓墍灞曠ず鐨勬帹鐞嗘椂闂存槸鍖呮嫭缃戠粶鍓嶅悜浼犳挱鍜屽悗澶勭悊鎵€闇€鐨勬€绘椂闂达紝涓嶅寘鎷暟鎹姞杞芥墍闇€鐨勬椂闂达紝妯″瀷搴撲腑鎵€灞曠ず鐨勭粨鏋滃潎鐢?[benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py) 鑴氭湰鏂囦欢鍦?2000 寮犲浘鍍忎笂鎵€璁＄畻鐨勫钩鍧囨椂闂淬€?
## 鍩哄噯缁撴灉

### SECOND

璇峰弬鑰?[SECOND](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/second) 鑾峰彇鏇村鐨勭粏鑺傦紝鎴戜滑鍦?KITTI 鍜?Waymo 鏁版嵁闆嗕笂閮界粰鍑轰簡鐩稿簲鐨勫熀鍑嗙粨鏋溿€?
### PointPillars

璇峰弬鑰?[PointPillars](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 KITTI 銆乶uScenes 銆丩yft 銆乄aymo 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### Part-A2

璇峰弬鑰?[Part-A2](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/parta2) 鑾峰彇鏇村缁嗚妭銆?
### VoteNet

璇峰弬鑰?[VoteNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/votenet) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 ScanNet 鍜?SUNRGBD 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### Dynamic Voxelization

璇峰弬鑰?[Dynamic Voxelization](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/dynamic_voxelization) 鑾峰彇鏇村缁嗚妭銆?
### MVXNet

璇峰弬鑰?[MVXNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/mvxnet) 鑾峰彇鏇村缁嗚妭銆?
### RegNetX

璇峰弬鑰?[RegNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/regnet) 鑾峰彇鏇村缁嗚妭锛屾垜浠皢 pointpillars 鐨勪富骞茬綉缁滄浛鎹㈡垚 RegNetX锛屽苟鍦?nuScenes 鍜?Lyft 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### nuImages

鎴戜滑鍦?[nuImages 鏁版嵁闆哴(https://www.nuscenes.org/nuimages) 涓婁篃鎻愪緵鍩哄噯妯″瀷锛岃鍙傝€?[nuImages](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/nuimages) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪璇ユ暟鎹泦涓婃彁渚?Mask R-CNN 锛?Cascade Mask R-CNN 鍜?HTC 鐨勭粨鏋溿€?
### H3DNet

璇峰弬鑰?[H3DNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/h3dnet) 鑾峰彇鏇村缁嗚妭銆?
### 3DSSD

璇峰弬鑰?[3DSSD](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/3dssd) 鑾峰彇鏇村缁嗚妭銆?
### CenterPoint

璇峰弬鑰?[CenterPoint](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/centerpoint) 鑾峰彇鏇村缁嗚妭銆?
### SSN

璇峰弬鑰?[SSN](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/ssn) 鑾峰彇鏇村缁嗚妭锛屾垜浠皢 pointpillars 涓殑妫€娴嬪ご鏇挎崲鎴?SSN 妯″瀷涓墍浣跨敤鐨?鈥榮hape-aware grouping heads鈥欙紝骞跺湪 nuScenes 鍜?Lyft 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### ImVoteNet

璇峰弬鑰?[ImVoteNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/imvotenet) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 SUNRGBD 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### FCOS3D

璇峰弬鑰?[FCOS3D](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/fcos3d) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 nuScenes 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### PointNet++

璇峰弬鑰?[PointNet++](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointnet2) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 ScanNet 鍜?S3DIS 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### Group-Free-3D

璇峰弬鑰?[Group-Free-3D](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/groupfree3d) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 ScanNet 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### ImVoxelNet

璇峰弬鑰?[ImVoxelNet](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/imvoxelnet) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 KITTI 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### PAConv

璇峰弬鑰?[PAConv](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/paconv) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 S3DIS 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### DGCNN

璇峰弬鑰?[DGCNN](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/dgcnn) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 S3DIS 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### SMOKE

璇峰弬鑰?[SMOKE](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/smoke) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 KITTI 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### PGD

璇峰弬鑰?[PGD](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pgd) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 KITTI 鍜?nuScenes 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### PointRCNN

璇峰弬鑰?[PointRCNN](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/point_rcnn) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 KITTI 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### MonoFlex

璇峰弬鑰?[MonoFlex](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/monoflex) 鑾峰彇鏇村缁嗚妭锛屾垜浠湪 KITTI 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑缁撴灉銆?
### SA-SSD

璇峰弬鑰?[SA-SSD](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/sassd) 鑾峰彇鏇村鐨勭粏鑺傦紝鎴戜滑鍦?KITTI 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### FCAF3D

璇峰弬鑰?[FCAF3D](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/fcaf3d) 鑾峰彇鏇村鐨勭粏鑺傦紝鎴戜滑鍦?ScanNet, S3DIS 鍜?SUN RGB-D 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### PV-RCNN

璇峰弬鑰?[PV-RCNN](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pv_rcnn) 鑾峰彇鏇村鐨勭粏鑺傦紝鎴戜滑鍦?KITTI 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### BEVFusion

璇峰弬鑰?[BEVFusion](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/BEVFusion) 鑾峰彇鏇村鐨勭粏鑺? 鎴戜滑鍦?NuScenes 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### CenterFormer

璇峰弬鑰?[CenterFormer](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/CenterFormer) 鑾峰彇鏇村鐨勭粏鑺? 鎴戜滑鍦?Waymo 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### TR3D

璇峰弬鑰?[TR3D](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/TR3D) 鑾峰彇鏇村鐨勭粏鑺? 鎴戜滑鍦?ScanNet, SUN RGB-D 鍜?S3DIS 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### DETR3D

璇峰弬鑰?[DETR3D](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/DETR3D) 鑾峰彇鏇村鐨勭粏鑺? 鎴戜滑鍦?NuScenes 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### PETR

璇峰弬鑰?[PETR](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/PETR) 鑾峰彇鏇村鐨勭粏鑺? 鎴戜滑鍦?NuScenes 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### TPVFormer

璇峰弬鑰?[TPVFormer](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/projects/TPVFormer) 鑾峰彇鏇村鐨勭粏鑺? 鎴戜滑鍦?NuScenes 鏁版嵁闆嗕笂缁欏嚭浜嗙浉搴旂殑鍩哄噯缁撴灉銆?
### Mixed Precision (FP16) Training

缁嗚妭璇峰弬鑰?[Mixed Precision (FP16) Training 鍦?PointPillars 璁粌鐨勬牱渚媇(https://github.com/open-mmlab/mmdetection3d/blob/main/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-amp-2x_nus-3d.py)銆?
