# 鎺ㄧ悊

## 浠嬬粛

鎴戜滑鎻愪緵浜嗗妯℃€?鍗曟ā鎬侊紙鍩轰簬婵€鍏夐浄杈?鍥惧儚锛夈€佸鍐?瀹ゅ鍦烘櫙鐨?3D 妫€娴嬪拰 3D 璇箟鍒嗗壊鏍蜂緥鐨勮剼鏈紝棰勮缁冩ā鍨嬪彲浠ヤ粠 [Model Zoo](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/zh_cn/model_zoo.md) 涓嬭浇銆傛垜浠篃鎻愪緵浜?KITTI銆丼UN RGB-D銆乶uScenes 鍜?ScanNet 鏁版嵁闆嗙殑棰勫鐞嗘牱鏈暟鎹紝浣犲彲浠ユ牴鎹垜浠殑棰勫鐞嗘楠や娇鐢ㄤ换浣曞叾瀹冩暟鎹€?
## 娴嬭瘯

### 3D 妫€娴?
#### 鐐逛簯鏍蜂緥

鍦ㄧ偣浜戞暟鎹笂娴嬭瘯 3D 妫€娴嬪櫒锛岃繍琛岋細

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

鐐逛簯鍜岄娴?3D 妗嗙殑鍙鍖栫粨鏋滀細琚繚瀛樺湪 `${OUT_DIR}/PCD_NAME`锛屽畠鍙互浣跨敤 [MeshLab](http://www.meshlab.net/) 鎵撳紑銆傛敞鎰忓鏋滀綘璁剧疆浜?`--show`锛岄€氳繃 [Open3D](http://www.open3d.org/) 鍙互鍦ㄧ嚎鏄剧ず棰勬祴缁撴灉銆?
鍦?KITTI 鏁版嵁涓婃祴璇?[PointPillars 妯″瀷](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth)锛?
```shell
python demo/pcd_demo.py demo/data/kitti/000008.bin configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py ${CHECKPOINT_FILE} --show
```

鍦?SUN RGB-D 鏁版嵁涓婃祴璇?[VoteNet 妯″瀷](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth)锛?
```shell
python demo/pcd_demo.py demo/data/sunrgbd/sunrgbd_000017.bin configs/votenet/votenet_8xb16_sunrgbd-3d.py ${CHECKPOINT_FILE} --show
```

#### 鍗曠洰 3D 鏍蜂緥

鍦ㄥ浘鍍忔暟鎹笂娴嬭瘯鍗曠洰 3D 妫€娴嬪櫒锛岃繍琛岋細

```shell
python demo/mono_det_demo.py ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out-dir ${OUT_DIR}] [--show]
```

`ANNOTATION_FILE` 闇€瑕佹彁渚?3D 鍒?2D 鐨勪豢灏勭煩闃碉紙鐩告満鍐呭弬鐭╅樀锛夛紝鍙鍖栫粨鏋滀細琚繚瀛樺湪 `${OUT_DIR}/PCD_NAME`锛屽叾涓寘鎷浘鍍忎互鍙婇娴?3D 妗嗗湪鍥惧儚涓婄殑鎶曞奖銆?
鍦?KITTI 鏁版嵁涓婃祴璇?[PGD 妯″瀷](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pgd/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth)锛?
```shell
python demo/mono_det_demo.py demo/data/kitti/000008.png demo/data/kitti/000008.pkl  configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py ${CHECKPOINT_FILE}  --show --cam-type CAM2 --score-thr 8
```

**娉ㄦ剰**锛?PGD 鏂规硶鐨勯娴嬫鍒嗘暟骞朵笉鏄湪 (0, 1) 涔嬮棿

鍦?nuScenes 鏁版嵁涓婃祴璇?[FCOS3D 妯″瀷](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth)锛?
```shell
python demo/mono_det_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl  configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py ${CHECKPOINT_FILE}  --show --cam-type CAM_BACK
```

**娉ㄦ剰**锛?褰撳缈昏浆鍥惧儚鍙鍖栧崟鐩?3D 妫€娴嬬粨鏋滄槸锛岀浉鏈哄唴鍙傜煩闃典篃搴旇鐩稿簲淇敼銆傚湪 PR [#744](https://github.com/open-mmlab/mmdetection3d/pull/744) 涓彲浠ヤ簡瑙ｆ洿澶氱粏鑺傚拰绀轰緥銆?
#### 澶氭ā鎬佹牱渚?
鍦ㄥ妯℃€佹暟鎹紙閫氬父鏄偣浜戝拰鍥惧儚锛変笂娴嬭瘯 3D 妫€娴嬪櫒锛岃繍琛岋細

```shell
python demo/multi_modality_demo.py ${PCD_FILE} ${IMAGE_FILE} ${ANNOTATION_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}] [--show]
```

`ANNOTATION_FILE` 闇€瑕佹彁渚?3D 鍒?2D 鐨勪豢灏勭煩闃碉紝鍙鍖栫粨鏋滀細琚繚瀛樺湪 `${OUT_DIR}/PCD_NAME`锛屽叾涓寘鎷偣浜戙€佸浘鍍忋€侀娴嬬殑 3D 妗嗕互鍙婂畠浠湪鍥惧儚涓婄殑鎶曞奖銆?
鍦?KITTI 鏁版嵁涓婃祴璇?[MVX-Net 妯″瀷](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth)锛?
```shell
python demo/multi_modality_demo.py demo/data/kitti/000008.bin demo/data/kitti/000008.png demo/data/kitti/000008.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py ${CHECKPOINT_FILE} --cam-type CAM2 --show
```

鍦?SUN RGB-D 鏁版嵁涓婃祴璇?[ImVoteNet 妯″瀷](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/imvotenet/imvotenet_stage2_16x8_sunrgbd-3d-10class/imvotenet_stage2_16x8_sunrgbd-3d-10class_20210819_192851-1bcd1b97.pth)锛?
```shell
python demo/multi_modality_demo.py demo/data/sunrgbd/000017.bin demo/data/sunrgbd/000017.jpg demo/data/sunrgbd/sunrgbd_000017_infos.pkl configs/imvotenet/imvotenet_stage2_8xb16_sunrgbd-3d.py ${CHECKPOINT_FILE} --cam-type CAM0 --show --score-thr 0.6
```

鍦?NuScenes 鏁版嵁涓婃祴璇?[BEVFusion 妯″瀷](https://drive.google.com/file/d/1QkvbYDk4G2d6SZoeJqish13qSyXA4lp3/view?usp=share_link)

```shell
python demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${CHECKPOINT_FILE} --cam-type all --score-thr 0.2 --show
```

### 3D 鍒嗗壊

鍦ㄧ偣浜戞暟鎹笂娴嬭瘯 3D 鍒嗗壊鍣紝杩愯锛?
```shell
python demo/pc_seg_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out-dir ${OUT_DIR}] [--show]
```

鍙鍖栫粨鏋滀細琚繚瀛樺湪 `${OUT_DIR}/PCD_NAME`锛屽叾涓寘鎷偣浜戜互鍙婇娴嬬殑 3D 鍒嗗壊鎺╃爜銆?
鍦?ScanNet 鏁版嵁涓婃祴璇?[PointNet++ (SSG) 妯″瀷](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class_20210514_143644-ee73704a.pth)锛?
```shell
python demo/pcd_seg_demo.py demo/data/scannet/scene0000_00.bin configs/pointnet2/pointnet2_ssg_2xb16-cosine-200e_scannet-seg.py ${CHECKPOINT_FILE} --show
```

