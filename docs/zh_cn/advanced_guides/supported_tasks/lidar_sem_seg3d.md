# 鍩轰簬婵€鍏夐浄杈剧殑 3D 璇箟鍒嗗壊

鍩轰簬婵€鍏夐浄杈剧殑 3D 璇箟鍒嗗壊鏄?MMDetection3D 鏀寔鐨勬渶鍩虹鐨勪换鍔′箣涓€銆傚畠鏈熸湜缁欏畾鐨勬ā鍨嬩互婵€鍏夐浄杈鹃噰闆嗙殑浠绘剰鏁伴噺鐨勭壒寰佺偣涓鸿緭鍏ワ紝骞堕娴嬫瘡涓緭鍏ョ偣鐨勮涔夋爣绛俱€傛帴涓嬫潵锛屾垜浠互 ScanNet 鏁版嵁闆嗕笂鐨?PointNet++ (SSG) 涓轰緥锛屽睍绀哄浣曞噯澶囨暟鎹紝鍦ㄦ爣鍑嗙殑 3D 璇箟鍒嗗壊鍩哄噯涓婅缁冨苟娴嬭瘯妯″瀷锛屼互鍙婂彲瑙嗗寲骞堕獙璇佺粨鏋溿€?
## 鏁版嵁鍑嗗

棣栧厛锛屾垜浠渶瑕佷粠 ScanNet [瀹樻柟缃戠珯](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation)涓嬭浇鍘熷鏁版嵁銆?
鐢变簬涓嶅悓鏁版嵁闆嗙殑鍘熷鏁版嵁鏈変笉鍚岀殑缁勭粐鏂瑰紡锛屾垜浠€氬父闇€瑕佺敤 pkl 鎴?json 鏂囦欢鏀堕泦鏈夌敤鐨勬暟鎹俊鎭€?
鍥犳锛屽湪鍑嗗濂芥墍鏈夌殑鍘熷鏁版嵁涔嬪悗锛屾垜浠彲浠ラ伒寰?[ScanNet 鏂囨。](https://github.com/open-mmlab/mmdetection3d/blob/master/data/scannet/README.md/)涓殑璇存槑鐢熸垚鏁版嵁淇℃伅銆?
闅忓悗锛岀浉鍏崇殑鐩綍缁撴瀯灏嗗涓嬫墍绀猴細

```
mmdetection3d
鈹溾攢鈹€ mmdet3d
鈹溾攢鈹€ tools
鈹溾攢鈹€ configs
鈹溾攢鈹€ data
鈹?  鈹溾攢鈹€ scannet
鈹?  鈹?  鈹溾攢鈹€ scannet_utils.py
鈹?  鈹?  鈹溾攢鈹€ batch_load_scannet_data.py
鈹?  鈹?  鈹溾攢鈹€ load_scannet_data.py
鈹?  鈹?  鈹溾攢鈹€ scannet_utils.py
鈹?  鈹?  鈹溾攢鈹€ README.md
鈹?  鈹?  鈹溾攢鈹€ scans
鈹?  鈹?  鈹溾攢鈹€ scans_test
鈹?  鈹?  鈹溾攢鈹€ scannet_instance_data
鈹?  鈹?  鈹溾攢鈹€ points
鈹?  鈹?  鈹溾攢鈹€ instance_mask
鈹?  鈹?  鈹溾攢鈹€ semantic_mask
鈹?  鈹?  鈹溾攢鈹€ seg_info
鈹?  鈹?  鈹?  鈹溾攢鈹€ train_label_weight.npy
鈹?  鈹?  鈹?  鈹溾攢鈹€ train_resampled_scene_idxs.npy
鈹?  鈹?  鈹?  鈹溾攢鈹€ val_label_weight.npy
鈹?  鈹?  鈹?  鈹溾攢鈹€ val_resampled_scene_idxs.npy
鈹?  鈹?  鈹溾攢鈹€ scannet_infos_train.pkl
鈹?  鈹?  鈹溾攢鈹€ scannet_infos_val.pkl
鈹?  鈹?  鈹溾攢鈹€ scannet_infos_test.pkl
```

## 璁粌

鎺ョ潃锛屾垜浠皢浣跨敤鎻愪緵鐨勯厤缃枃浠惰缁?PointNet++ (SSG) 妯″瀷銆傚綋浣犱娇鐢ㄤ笉鍚岀殑 GPU 璁剧疆杩涜璁粌鏃讹紝浣犲熀鏈笂鍙互鎸夌収杩欎釜[鏁欑▼](https://mmdetection3d.readthedocs.io/zh_CN/latest/1_exist_data_model.html#inference-with-existing-models)鐨勭ず渚嬭剼鏈€傚亣璁炬垜浠湪涓€鍙板叿鏈?2 鍧?GPU 鐨勬満鍣ㄤ笂浣跨敤鍒嗗竷寮忚缁冿細

```
./tools/dist_train.sh configs/pointnet2/pointnet2_ssg_2xb16-cosine-200e_scannet-seg.py 2
```

娉ㄦ剰锛岄厤缃枃浠跺悕涓殑 `16x2` 鏄寚璁粌鏃剁敤浜?2 鍧?GPU锛屾瘡鍧?GPU 涓婃湁 16 涓牱鏈€傚鏋滀綘鐨勮嚜瀹氫箟璁剧疆涓嶅悓浜庢锛岄偅涔堟湁鏃跺€欎綘闇€瑕佺浉搴旂殑璋冩暣瀛︿範鐜囥€傚熀鏈鍒欏彲浠ュ弬鑰僛姝ゅ](https://arxiv.org/abs/1706.02677)銆?
## 瀹氶噺璇勪及

鍦ㄨ缁冩湡闂达紝妯″瀷鏉冮噸灏嗕細鏍规嵁閰嶇疆鏂囦欢涓殑 `train_cfg = dict(val_interval=xxx)` 璁剧疆琚懆鏈熸€у湴璇勪及銆傛垜浠敮鎸佷笉鍚屾暟鎹泦鐨勫畼鏂硅瘎浼版柟妗堛€傚浜?ScanNet锛屽皢浣跨敤 20 涓被鍒殑骞冲潎浜ゅ苟姣?(mIoU) 瀵规ā鍨嬭繘琛岃瘎浼般€傝瘎浼扮粨鏋滃皢浼氳鎵撳嵃鍒扮粓绔腑锛屽涓嬫墍绀猴細

```
+---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+-----------+---------+---------+--------+---------+--------------+----------------+--------+--------+---------+----------------+--------+--------+---------+
| classes | wall   | floor  | cabinet | bed    | chair  | sofa   | table  | door   | window | bookshelf | picture | counter | desk   | curtain | refrigerator | showercurtrain | toilet | sink   | bathtub | otherfurniture | miou   | acc    | acc_cls |
+---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+-----------+---------+---------+--------+---------+--------------+----------------+--------+--------+---------+----------------+--------+--------+---------+
| results | 0.7257 | 0.9373 | 0.4625  | 0.6613 | 0.7707 | 0.5562 | 0.5864 | 0.4010 | 0.4558 | 0.7011    | 0.2500  | 0.4645  | 0.4540 | 0.5399  | 0.2802       | 0.3488         | 0.7359 | 0.4971 | 0.6922  | 0.3681         | 0.5444 | 0.8118 | 0.6695  |
+---------+--------+--------+---------+--------+--------+--------+--------+--------+--------+-----------+---------+---------+--------+---------+--------------+----------------+--------+--------+---------+----------------+--------+--------+---------+
```

姝ゅ锛屽湪璁粌瀹屾垚鍚庝綘涔熷彲浠ヨ瘎浼扮壒瀹氱殑妯″瀷鏉冮噸鏂囦欢銆備綘鍙互绠€鍗曞湴鎵ц浠ヤ笅鑴氭湰锛?
```
./tools/dist_test.sh configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet-seg.py work_dirs/pointnet2_ssg/latest.pth 8
```

## 娴嬭瘯涓庢彁浜?
濡傛灉浣犲彧鎯冲湪鍦ㄧ嚎鍩哄噯涓婅繘琛屾帹鐞嗘垨娴嬭瘯妯″瀷鎬ц兘锛屼綘闇€瑕佸湪閰嶇疆鏂囦欢涓殑 `test_evalutor` 瀛楁澧炲姞 `submission_prefix`锛?渚嬪閰嶇疆鏂囦欢澧炲姞 `test_evaluator = dict(type='SegMetric',submission_prefix=work_dirs/pointnet2_ssg/test_submission`)銆?骞跺皢 ScanNet 鏁版嵁闆哰閰嶇疆鏂囦欢](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/_base_/datasets/scannet-seg.py#L129)涓殑 `ann_file=scannet_infos_val.pkl` 鍙樻垚 `ann_file=scannet_infos_test.pkl`銆傚湪鐢熸垚缁撴灉鍚庯紝浣犲彲浠ュ帇缂╂枃浠跺す骞朵笂浼犺嚦 [ScanNet 璇勪及鏈嶅姟鍣╙(http://kaldir.vc.in.tum.de/scannet_benchmark/semantic_label_3d)涓娿€?
## 瀹氭€ц瘎浼?
MMDetection3D 杩樻彁渚涗簡閫氱敤鐨勫彲瑙嗗寲宸ュ叿锛屼互渚夸簬鎴戜滑鍙互瀵硅缁冨ソ鐨勬ā鍨嬮娴嬬殑鍒嗗壊缁撴灉鏈変竴涓洿瑙傜殑鎰熷彈銆備綘涔熷彲浠ュ湪璇勪及闃舵閫氳繃璁剧疆 `--eval-options 'show=True' 'out_dir=${SHOW_DIR}'` 鏉ュ湪绾垮彲瑙嗗寲鍒嗗壊缁撴灉锛屾垨鑰呬娇鐢?`tools/misc/visualize_results.py` 鏉ョ绾垮湴杩涜鍙鍖栥€傛澶栵紝鎴戜滑杩樻彁渚涗簡鑴氭湰 `tools/misc/browse_dataset.py` 鐢ㄤ簬鍙鍖栨暟鎹泦鑰屼笉鍋氭帹鐞嗐€傛洿澶氱殑缁嗚妭璇峰弬鑰僛鍙鍖栨枃妗(https://mmdetection3d.readthedocs.io/zh_CN/latest/useful_tools.html#visualization)銆?
