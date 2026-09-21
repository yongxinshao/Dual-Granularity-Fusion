# 鍩轰簬瑙嗚鐨?3D 妫€娴?
鍩轰簬瑙嗚鐨?3D 妫€娴嬫槸鎸囧熀浜庣函瑙嗚杈撳叆鐨?3D 妫€娴嬫柟娉曪紝渚嬪鍩轰簬鍗曠洰銆佸弻鐩拰澶氳鍥惧浘鍍忕殑 3D 妫€娴嬨€傜洰鍓嶏紝鎴戜滑鍙敮鎸佸崟鐩拰澶氳鍥剧殑 3D 妫€娴嬫柟娉曘€傚叾浠栨柟娉曚篃搴旇涓庢垜浠殑妗嗘灦鍏煎锛屽苟鍦ㄥ皢鏉ュ緱鍒版敮鎸併€?
瀹冩湡鏈涚粰瀹氱殑妯″瀷浠ヤ换鎰忔暟閲忕殑鍥惧儚浣滀负杈撳叆锛屽苟涓烘瘡涓€涓劅鍏磋叮鐨勭洰鏍囬娴?3D 妗嗗強绫诲埆鏍囩銆備互 nuScenes 鏁版嵁闆?FCOS3D 涓轰緥锛屾垜浠皢灞曠ず濡備綍鍑嗗鏁版嵁锛屽湪鏍囧噯鐨?3D 妫€娴嬪熀鍑嗕笂璁粌骞舵祴璇曟ā鍨嬶紝浠ュ強鍙鍖栧苟楠岃瘉缁撴灉銆?
## 鏁版嵁鍑嗗

棣栧厛锛屾垜浠渶瑕佷笅杞藉師濮嬫暟鎹苟鎸夌収[鏁版嵁鍑嗗鏂囨。](https://mmdetection3d.readthedocs.io/zh_CN/latest/data_preparation.html)涓彁渚涚殑鏍囧噯鏂瑰紡閲嶆柊缁勭粐鏁版嵁銆?
鐢变簬涓嶅悓鏁版嵁闆嗙殑鍘熷鏁版嵁鏈変笉鍚岀殑缁勭粐鏂瑰紡锛屾垜浠€氬父闇€瑕佺敤 pkl 鎴?json 鏂囦欢鏀堕泦鏈夌敤鐨勬暟鎹俊鎭€傚洜姝わ紝鍦ㄥ噯澶囧ソ鎵€鏈夌殑鍘熷鏁版嵁涔嬪悗锛屾垜浠渶瑕佽繍琛?`create_data.py` 涓彁渚涚殑鑴氭湰鏉ヤ负涓嶅悓鐨勬暟鎹泦鐢熸垚鏁版嵁淇℃伅銆備緥濡傦紝瀵逛簬 nuScenes锛屾垜浠渶瑕佽繍琛屽涓嬪懡浠わ細

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

闅忓悗锛岀浉鍏崇殑鐩綍缁撴瀯灏嗗涓嬫墍绀猴細

```
mmdetection3d
鈹溾攢鈹€ mmdet3d
鈹溾攢鈹€ tools
鈹溾攢鈹€ configs
鈹溾攢鈹€ data
鈹?  鈹溾攢鈹€ nuscenes
鈹?  鈹?  鈹溾攢鈹€ maps
鈹?  鈹?  鈹溾攢鈹€ samples
鈹?  鈹?  鈹溾攢鈹€ sweeps
鈹?  鈹?  鈹溾攢鈹€ v1.0-test
|   |   鈹溾攢鈹€ v1.0-trainval
鈹?  鈹?  鈹溾攢鈹€ nuscenes_database
鈹?  鈹?  鈹溾攢鈹€ nuscenes_infos_train.pkl
鈹?  鈹?  鈹溾攢鈹€ nuscenes_infos_trainval.pkl
鈹?  鈹?  鈹溾攢鈹€ nuscenes_infos_val.pkl
鈹?  鈹?  鈹溾攢鈹€ nuscenes_infos_test.pkl
鈹?  鈹?  鈹溾攢鈹€ nuscenes_dbinfos_train.pkl
鈹?  鈹?  鈹溾攢鈹€ nuscenes_infos_train_mono3d.coco.json
鈹?  鈹?  鈹溾攢鈹€ nuscenes_infos_trainval_mono3d.coco.json
鈹?  鈹?  鈹溾攢鈹€ nuscenes_infos_val_mono3d.coco.json
鈹?  鈹?  鈹溾攢鈹€ nuscenes_infos_test_mono3d.coco.json
```

娉ㄦ剰锛屾澶勭殑 pkl 鏂囦欢涓昏鐢ㄤ簬浣跨敤 LiDAR 鏁版嵁鐨勬柟娉曪紝json 鏂囦欢鐢ㄤ簬 2D 妫€娴?绾瑙夌殑 3D 妫€娴嬨€傚湪 v0.13.0 鏀寔鍗曠洰 3D 妫€娴嬩箣鍓嶏紝json 鏂囦欢鍙寘鍚?2D 妫€娴嬬殑淇℃伅锛屽洜姝ゅ鏋滀綘闇€瑕佹渶鏂扮殑淇℃伅锛岃鍒囨崲鍒?v0.13.0 涔嬪悗鐨勫垎鏀€?
## 璁粌

鎺ョ潃锛屾垜浠皢浣跨敤鎻愪緵鐨勯厤缃枃浠惰缁?FCOS3D銆傚熀鏈殑鑴氭湰涓庡叾浠栨ā鍨嬩竴鏍枫€傚綋浣犱娇鐢ㄤ笉鍚岀殑 GPU 璁剧疆杩涜璁粌鏃讹紝浣犲熀鏈笂鍙互鎸夌収杩欎釜[鏁欑▼](https://mmdetection3d.readthedocs.io/zh_CN/latest/1_exist_data_model.html#inference-with-existing-models)鐨勭ず渚嬨€傚亣璁炬垜浠湪涓€鍙板叿鏈?8 鍧?GPU 鐨勬満鍣ㄤ笂浣跨敤鍒嗗竷寮忚缁冿細

```
./tools/dist_train.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py 8
```

娉ㄦ剰锛岄厤缃枃浠跺悕涓殑 `2x8` 鏄寚璁粌鏃剁敤浜?8 鍧?GPU锛屾瘡鍧?GPU 涓婃湁 2 涓暟鎹牱鏈€傚鏋滀綘鐨勮嚜瀹氫箟璁剧疆涓嶅悓浜庢锛岄偅涔堟湁鏃跺€欎綘闇€瑕佺浉搴旂殑璋冩暣瀛︿範鐜囥€傚熀鏈鍒欏彲浠ュ弬鑰僛姝ゅ](https://arxiv.org/abs/1706.02677)銆?
鎴戜滑涔熷彲浠ラ€氳繃杩愯浠ヤ笅鍛戒护寰皟 FCOS3D锛屼粠鑰岃揪鍒版洿濂界殑鎬ц兘锛?
```
./tools/dist_train.sh fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py 8
```

閫氳繃鍏堝墠鐨勮剼鏈缁冨ソ涓€涓熀鍑嗘ā鍨嬪悗锛岃璁板緱鐩稿簲鐨勪慨鏀筟姝ゅ](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py#L8)鐨勮矾寰勩€?
## 瀹氶噺璇勪及

鍦ㄨ缁冩湡闂达紝妯″瀷鏉冮噸鏂囦欢灏嗕細鏍规嵁閰嶇疆鏂囦欢涓殑 `evaluation = dict(interval=xxx)` 璁剧疆琚懆鏈熸€у湴璇勪及銆?
鎴戜滑鏀寔涓嶅悓鏁版嵁闆嗙殑瀹樻柟璇勪及鏂规銆傜敱浜庤緭鍑烘牸寮忎笌鍩轰簬鍏朵粬妯℃€佺殑 3D 妫€娴嬬浉鍚岋紝鍥犳璇勪及鏂规硶涔熸槸涓€鏍风殑銆?
瀵逛簬 nuScenes锛屽皢浣跨敤鍩轰簬璺濈鐨勫钩鍧囩簿搴︼紙mAP锛変互鍙?nuScenes 妫€娴嬪垎鏁帮紙NDS锛夊垎鍒 10 涓被鍒繘琛岃瘎浼般€傝瘎浼扮粨鏋滃皢浼氳鎵撳嵃鍒扮粓绔腑锛屽涓嬫墍绀猴細

```
mAP: 0.3197
mATE: 0.7595
mASE: 0.2700
mAOE: 0.4918
mAVE: 1.3307
mAAE: 0.1724
NDS: 0.3905
Eval time: 170.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.503   0.577   0.152   0.111   2.096   0.136
truck   0.223   0.857   0.224   0.220   1.389   0.179
bus     0.294   0.855   0.204   0.190   2.689   0.283
trailer 0.081   1.094   0.243   0.553   0.742   0.167
construction_vehicle    0.058   1.017   0.450   1.019   0.137   0.341
pedestrian      0.392   0.687   0.284   0.694   0.876   0.158
motorcycle      0.317   0.737   0.265   0.580   2.033   0.104
bicycle 0.308   0.704   0.299   0.892   0.683   0.010
traffic_cone    0.555   0.486   0.309   nan     nan     nan
barrier 0.466   0.581   0.269   0.169   nan     nan
```

姝ゅ锛屽湪璁粌瀹屾垚鍚庝綘涔熷彲浠ヨ瘎浼扮壒瀹氱殑妯″瀷鏉冮噸鏂囦欢銆備綘鍙互绠€鍗曞湴鎵ц浠ヤ笅鑴氭湰锛?
```
./tools/dist_test.sh configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py \
    work_dirs/fcos3d/latest.pth --eval mAP
```

## 娴嬭瘯涓庢彁浜?
濡傛灉浣犲彧鎯冲湪鍦ㄧ嚎鍩哄噯涓婅繘琛屾帹鐞嗘垨娴嬭瘯妯″瀷鎬ц兘锛屼綘闇€瑕佸皢涔嬪墠璇勪及鑴氭湰涓殑 `--eval mAP` 鏇挎崲鎴?`--format-only`锛屽苟鍦ㄩ渶瑕佺殑鎯呭喌涓嬫寚瀹?`jsonfile_prefix`锛屼緥濡傦紝娣诲姞閫夐」 `--eval-options jsonfile_prefix=work_dirs/fcos3d/test_submission`銆傝纭繚閰嶇疆鏂囦欢涓殑[娴嬭瘯淇℃伅](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/_base_/datasets/nus-mono3d.py#L93)鐢遍獙璇侀泦鐩稿簲鍦版敼涓烘祴璇曢泦銆?
鍦ㄧ敓鎴愮粨鏋滃悗锛屼綘鍙互鍘嬬缉鏂囦欢澶瑰苟涓婁紶鑷?nuScenes 3D 妫€娴嬫寫鎴樼殑 evalAI 璇勪及鏈嶅姟鍣ㄤ笂銆?
## 瀹氭€ц瘎浼?
MMDetection3D 杩樻彁渚涗簡閫氱敤鐨勫彲瑙嗗寲宸ュ叿锛屼互渚夸簬鎴戜滑鍙互瀵硅缁冨ソ鐨勬ā鍨嬮娴嬬殑妫€娴嬬粨鏋滄湁涓€涓洿瑙傜殑鎰熷彈銆備綘涔熷彲浠ュ湪璇勪及闃舵閫氳繃璁剧疆 `--eval-options 'show=True' 'out_dir=${SHOW_DIR}'` 鏉ュ湪绾垮彲瑙嗗寲妫€娴嬬粨鏋滐紝鎴栬€呬娇鐢?`tools/misc/visualize_results.py` 鏉ョ绾垮湴杩涜鍙鍖栥€?
姝ゅ锛屾垜浠繕鎻愪緵浜嗚剼鏈?`tools/misc/browse_dataset.py` 鐢ㄤ簬鍙鍖栨暟鎹泦鑰屼笉鍋氭帹鐞嗐€傛洿澶氱殑缁嗚妭璇峰弬鑰僛鍙鍖栨枃妗(https://mmdetection3d.readthedocs.io/zh_CN/latest/useful_tools.html#visualization)銆?
娉ㄦ剰锛岀洰鍓嶆垜浠粎鏀寔绾瑙夋柟娉曞湪鍥惧儚涓婄殑鍙鍖栥€傚皢鏉ユ垜浠皢闆嗘垚鍦ㄥ墠鏅浘浠ュ強楦熺灠鍥撅紙BEV锛変腑鐨勫彲瑙嗗寲銆?
