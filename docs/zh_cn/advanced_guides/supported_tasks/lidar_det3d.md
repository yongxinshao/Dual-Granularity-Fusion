# 鍩轰簬婵€鍏夐浄杈剧殑 3D 妫€娴?
鍩轰簬婵€鍏夐浄杈剧殑 3D 妫€娴嬫槸 MMDetection3D 鏀寔鐨勬渶鍩虹鐨勪换鍔′箣涓€銆傚畠鏈熸湜缁欏畾鐨勬ā鍨嬩互婵€鍏夐浄杈鹃噰闆嗙殑浠绘剰鏁伴噺鐨勭壒寰佺偣涓鸿緭鍏ワ紝骞朵负姣忎竴涓劅鍏磋叮鐨勭洰鏍囬娴?3D 妗嗗強绫诲埆鏍囩銆傛帴涓嬫潵锛屾垜浠互 KITTI 鏁版嵁闆嗕笂鐨?PointPillars 涓轰緥锛屽睍绀哄浣曞噯澶囨暟鎹紝鍦ㄦ爣鍑嗙殑 3D 妫€娴嬪熀鍑嗕笂璁粌骞舵祴璇曟ā鍨嬶紝浠ュ強鍙鍖栧苟楠岃瘉缁撴灉銆?
## 鏁版嵁鍑嗗

棣栧厛锛屾垜浠渶瑕佷笅杞藉師濮嬫暟鎹苟鎸夌収[鏁版嵁鍑嗗鏂囨。](https://mmdetection3d.readthedocs.io/zh_CN/dev-1.x/user_guides/dataset_prepare.html)涓彁渚涚殑鏍囧噯鏂瑰紡閲嶆柊缁勭粐鏁版嵁銆?
鐢变簬涓嶅悓鏁版嵁闆嗙殑鍘熷鏁版嵁鏈変笉鍚岀殑缁勭粐鏂瑰紡锛屾垜浠€氬父闇€瑕佺敤 `.pkl` 鏂囦欢鏀堕泦鏈夌敤鐨勬暟鎹俊鎭€傚洜姝わ紝鍦ㄥ噯澶囧ソ鎵€鏈夌殑鍘熷鏁版嵁涔嬪悗锛屾垜浠渶瑕佽繍琛?`create_data.py` 涓彁渚涚殑鑴氭湰鏉ヤ负涓嶅悓鐨勬暟鎹泦鐢熸垚鏁版嵁闆嗕俊鎭€備緥濡傦紝瀵逛簬 KITTI锛屾垜浠渶瑕佽繍琛屽涓嬪懡浠わ細

```shell
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

闅忓悗锛岀浉鍏崇殑鐩綍缁撴瀯灏嗗涓嬫墍绀猴細

```
mmdetection3d
鈹溾攢鈹€ mmdet3d
鈹溾攢鈹€ tools
鈹溾攢鈹€ configs
鈹溾攢鈹€ data
鈹?  鈹溾攢鈹€ kitti
鈹?  鈹?  鈹溾攢鈹€ ImageSets
鈹?  鈹?  鈹溾攢鈹€ testing
鈹?  鈹?  鈹?  鈹溾攢鈹€ calib
鈹?  鈹?  鈹?  鈹溾攢鈹€ image_2
鈹?  鈹?  鈹?  鈹溾攢鈹€ velodyne
鈹?  鈹?  鈹?  鈹溾攢鈹€ velodyne_reduced
鈹?  鈹?  鈹溾攢鈹€ training
鈹?  鈹?  鈹?  鈹溾攢鈹€ calib
鈹?  鈹?  鈹?  鈹溾攢鈹€ image_2
鈹?  鈹?  鈹?  鈹溾攢鈹€ label_2
鈹?  鈹?  鈹?  鈹溾攢鈹€ velodyne
鈹?  鈹?  鈹?  鈹溾攢鈹€ velodyne_reduced
鈹?  鈹?  鈹溾攢鈹€ kitti_gt_database
鈹?  鈹?  鈹溾攢鈹€ kitti_infos_train.pkl
鈹?  鈹?  鈹溾攢鈹€ kitti_infos_trainval.pkl
鈹?  鈹?  鈹溾攢鈹€ kitti_infos_val.pkl
鈹?  鈹?  鈹溾攢鈹€ kitti_infos_test.pkl
鈹?  鈹?  鈹溾攢鈹€ kitti_dbinfos_train.pkl
```

## 璁粌

鎺ョ潃锛屾垜浠皢浣跨敤鎻愪緵鐨勯厤缃枃浠惰缁?PointPillars銆傚綋鎮ㄤ娇鐢ㄤ笉鍚岀殑 GPU 璁剧疆杩涜璁粌鏃讹紝鎮ㄥ彲浠ユ寜鐓ц繖涓猍鏁欑▼](https://mmdetection3d.readthedocs.io/en/dev-1.x/user_guides/train_test.html)鐨勭ず渚嬨€傚亣璁炬垜浠湪涓€鍙板叿鏈?8 鍧?GPU 鐨勬満鍣ㄤ笂浣跨敤鍒嗗竷寮忚缁冿細

```shell
./tools/dist_train.sh configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py 8
```

娉ㄦ剰锛岄厤缃枃浠跺悕涓殑 `8xb6` 鏄寚璁粌鐢ㄤ簡 8 鍧?GPU锛屾瘡鍧?GPU 涓婃湁 6 涓暟鎹牱鏈€傚鏋滄偍鐨勮嚜瀹氫箟璁剧疆涓嶅悓浜庢锛岄偅涔堟湁鏃跺€欐偍闇€瑕佺浉搴斿湴璋冩暣瀛︿範鐜囥€傚熀鏈鍒欏彲浠ュ弬鑰僛姝ゅ](https://arxiv.org/abs/1706.02677)銆傛垜浠凡缁忔敮鎸佷簡浣跨敤 `--auto-scale-lr` 鏉ヨ嚜鍔ㄧ缉鏀惧涔犵巼銆?
## 瀹氶噺璇勪及

鍦ㄨ缁冩湡闂达紝妯″瀷鏉冮噸鏂囦欢灏嗕細鏍规嵁閰嶇疆鏂囦欢涓殑 `train_cfg = dict(val_interval=xxx)` 璁剧疆琚懆鏈熸€у湴璇勪及銆傛垜浠敮鎸佷笉鍚屾暟鎹泦鐨勫畼鏂硅瘎浼版柟妗堛€傚浜?KITTI锛屽皢瀵?3 涓被鍒娇鐢ㄤ氦骞舵瘮锛圛oU锛夐槇鍊煎垎鍒负 0.5/0.7 鐨勫钩鍧囩簿搴︼紙mAP锛夋潵璇勪及妯″瀷銆傝瘎浼扮粨鏋滃皢浼氳鎵撳嵃鍒扮粓绔腑锛屽涓嬫墍绀猴細

```
Car AP@0.70, 0.70, 0.70:
bbox AP:98.1839, 89.7606, 88.7837
bev AP:89.6905, 87.4570, 85.4865
3d AP:87.4561, 76.7569, 74.1302
aos AP:97.70, 88.73, 87.34
Car AP@0.70, 0.50, 0.50:
bbox AP:98.1839, 89.7606, 88.7837
bev AP:98.4400, 90.1218, 89.6270
3d AP:98.3329, 90.0209, 89.4035
aos AP:97.70, 88.73, 87.34
```

姝ゅ锛屽湪璁粌瀹屾垚鍚庢偍涔熷彲浠ヨ瘎浼扮壒瀹氱殑妯″瀷鏉冮噸鏂囦欢銆傛偍鍙互绠€鍗曞湴鎵ц浠ヤ笅鑴氭湰锛?
```shell
./tools/dist_test.sh configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py work_dirs/pointpillars/latest.pth 8
```

## 娴嬭瘯涓庢彁浜?
濡傛灉鎮ㄥ彧鎯冲湪鍦ㄧ嚎鍩哄噯涓婅繘琛屾帹鐞嗘垨娴嬭瘯妯″瀷鎬ц兘锛屾偍闇€瑕佸湪鐩稿簲鐨勮瘎浼板櫒涓寚瀹?`submission_prefix`锛屼緥濡傦紝鍦ㄩ厤缃枃浠朵腑娣诲姞 `test_evaluator = dict(type='KittiMetric', ann_file=data_root + 'kitti_infos_test.pkl', format_only=True, pklfile_prefix='results/kitti-3class/kitti_results', submission_prefix='results/kitti-3class/kitti_results')`锛岀劧鍚庡彲浠ュ緱鍒扮粨鏋滄枃浠躲€傝纭繚閰嶇疆鏂囦欢涓殑[娴嬭瘯淇℃伅](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/datasets/kitti-3d-3class.py#L117)鐨?`data_prefix` 鍜?`ann_file` 鐢遍獙璇侀泦鐩稿簲鍦版敼涓烘祴璇曢泦銆傚湪鐢熸垚缁撴灉鍚庯紝鎮ㄥ彲浠ュ帇缂╂枃浠跺す骞朵笂浼犺嚦 KITTI 璇勪及鏈嶅姟鍣ㄤ笂銆?
## 瀹氭€ц瘎浼?
MMDetection3D 杩樻彁渚涗簡閫氱敤鐨勫彲瑙嗗寲宸ュ叿锛屼互渚夸簬鎴戜滑鍙互瀵硅缁冨ソ鐨勬ā鍨嬮娴嬬殑妫€娴嬬粨鏋滄湁涓€涓洿瑙傜殑鎰熷彈銆傛偍涔熷彲浠ュ湪璇勪及闃舵閫氳繃璁剧疆 `--show` 鏉ュ湪绾垮彲瑙嗗寲妫€娴嬬粨鏋滐紝鎴栬€呬娇鐢?`tools/misc/visualize_results.py` 鏉ョ绾垮湴杩涜鍙鍖栥€傛澶栵紝鎴戜滑杩樻彁渚涗簡鑴氭湰 `tools/misc/browse_dataset.py` 鐢ㄤ簬鍙鍖栨暟鎹泦鑰屼笉鍋氭帹鐞嗐€傛洿澶氱殑缁嗚妭璇峰弬鑰僛鍙鍖栨枃妗(https://mmdetection3d.readthedocs.io/zh_CN/dev-1.x/user_guides/visualization.html)銆?
