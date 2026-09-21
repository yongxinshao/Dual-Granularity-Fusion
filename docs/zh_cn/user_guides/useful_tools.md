鎴戜滑鍦?`tools/` 鏂囦欢澶硅矾寰勪笅鎻愪緵浜嗚澶氭湁鐢ㄧ殑宸ュ叿銆?
## 鏃ュ織鍒嗘瀽

缁欏畾涓€涓缁冪殑鏃ュ織鏂囦欢锛屾偍鍙互缁樺埗鍑?loss/mAP 鏇茬嚎銆傞鍏堥渶瑕佽繍琛?`pip install seaborn` 瀹夎渚濊禆鍖呫€?
![loss鏇茬嚎鍥綸(../../../resources/loss_curve.png)

```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}] [--mode ${MODE}] [--interval ${INTERVAL}]
```

**娉ㄦ剰**: 濡傛灉鎮ㄦ兂缁樺埗鐨勬寚鏍囨槸鍦ㄩ獙璇侀樁娈佃绠楀緱鍒扮殑锛屾偍闇€瑕佹坊鍔犱竴涓爣蹇?`--mode eval` 锛屽鏋滄偍姣忕粡杩囦竴涓?`${INTERVAL}` 鐨勯棿闅旇繘琛岃瘎浼帮紝鎮ㄩ渶瑕佸鍔犱竴涓弬鏁?`--interval ${INTERVAL}`銆?
绀轰緥锛?
- 缁樺埗鍑烘煇娆¤繍琛岀殑鍒嗙被 loss銆?
  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
  ```

- 缁樺埗鍑烘煇娆¤繍琛岀殑鍒嗙被鍜屽洖褰?loss锛屽苟涓斾繚瀛樺浘鐗囦负 pdf 鏍煎紡銆?
  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
  ```

- 鍦ㄥ悓涓€寮犲浘鐗囦腑姣旇緝涓ゆ杩愯鐨?bbox mAP銆?
  ```shell
  # 鏍规嵁 Car_3D_moderate_strict 鍦?KITTI 涓婅瘎浼?PartA2 鍜?second銆?  python tools/analysis_tools/analyze_logs.py plot_curve tools/logs/PartA2.log.json tools/logs/second.log.json --keys KITTI/Car_3D_moderate_strict --legend PartA2 second --mode eval --interval 1
  # 鏍规嵁 Car_3D_moderate_strict 鍦?KITTI 涓婂垎鍒杞﹀拰 3 绫昏瘎浼?PointPillars銆?  python tools/analysis_tools/analyze_logs.py plot_curve tools/logs/pp-3class.log.json tools/logs/pp.log.json --keys KITTI/Car_3D_moderate_strict --legend pp-3class pp --mode eval --interval 2
  ```

鎮ㄤ篃鑳借绠楀钩鍧囪缁冮€熷害銆?
```shell
python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
```

棰勬湡杈撳嚭搴旇濡備笅鎵€绀恒€?
```
-----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
slowest epoch 11, average time is 1.2024
fastest epoch 1, average time is 1.1909
time std over epochs is 0.0028
average iter time: 1.1959 s/iter
```

&#8195;

## 妯″瀷閮ㄧ讲

**娉ㄦ剰**锛氭宸ュ叿浠嶇劧澶勪簬璇曢獙闃舵锛岀洰鍓嶅彧鏈?SECOND 鏀寔鐢?[`TorchServe`](https://pytorch.org/serve/) 閮ㄧ讲锛屾垜浠皢浼氬湪鏈潵鏀寔鏇村鐨勬ā鍨嬨€?
涓轰簡浣跨敤 [`TorchServe`](https://pytorch.org/serve/) 閮ㄧ讲 `MMDetection3D` 妯″瀷锛屾偍鍙互閬靛惊浠ヤ笅姝ラ锛?
### 1. 灏嗘ā鍨嬩粠 MMDetection3D 杞崲鍒?TorchServe

```shell
python tools/deployment/mmdet3d2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

**Note**: ${MODEL_STORE} 闇€瑕佷负鏂囦欢澶圭殑缁濆璺緞銆?
### 2. 鏋勫缓 `mmdet3d-serve` 闀滃儚

```shell
docker build -t mmdet3d-serve:latest docker/serve/
```

### 3. 杩愯 `mmdet3d-serve`

鏌ョ湅瀹樼綉鏂囨。鏉?[浣跨敤 docker 杩愯 TorchServe](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment)銆?
涓轰簡鍦?GPU 涓婅繍琛岋紝鎮ㄩ渶瑕佸畨瑁?[nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)銆傛偍鍙互蹇界暐 `--gpus` 鍙傛暟锛屼粠鑰屽湪 CPU 涓婅繍琛屻€?
渚嬪瓙锛?
```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=$MODEL_STORE,target=/home/model-server/model-store \
mmdet3d-serve:latest
```

[闃呰鏂囨。](https://github.com/pytorch/serve/blob/072f5d088cce9bb64b2a18af065886c9b01b317b/docs/rest_api.md/) 鍏充簬 Inference (8080), Management (8081) and Metrics (8082) 鎺ュ彛銆?
### 4. 娴嬭瘯閮ㄧ讲

鎮ㄥ彲浠ヤ娇鐢?`test_torchserver.py` 杩涜閮ㄧ讲锛?鍚屾椂姣旇緝 torchserver 鍜?pytorch 鐨勭粨鏋溿€?
```shell
python tools/deployment/test_torchserver.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MODEL_NAME}
[--inference-addr ${INFERENCE_ADDR}] [--device ${DEVICE}] [--score-thr ${SCORE_THR}]
```

渚嬪瓙:

```shell
python tools/deployment/test_torchserver.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth second
```

&#8195;

# 妯″瀷澶嶆潅搴?
鎮ㄥ彲浠ヤ娇鐢?MMDetection 涓殑 `tools/analysis_tools/get_flops.py` 杩欎釜鑴氭湰鏂囦欢锛屽熀浜?[flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 璁＄畻涓€涓粰瀹氭ā鍨嬬殑璁＄畻閲?(FLOPS) 鍜屽弬鏁伴噺 (params)銆?
```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

鎮ㄥ皢浼氬緱鍒板涓嬬殑缁撴灉锛?
```text
==============================
Input shape: (4000, 4)
Flops: 5.78 GFLOPs
Params: 953.83 k
==============================
```

**娉ㄦ剰**锛氭宸ュ叿浠嶇劧澶勪簬璇曢獙闃舵锛屾垜浠笉鑳戒繚璇佹暟鍊兼槸缁濆姝ｇ‘鐨勩€傛偍鍙互灏嗙粨鏋滅敤浜庣畝鍗曠殑姣旇緝锛屼絾鍦ㄥ啓鎶€鏈枃妗ｆ姤鍛婃垨鑰呰鏂囦箣鍓嶆偍闇€瑕佸啀娆＄‘璁や竴涓嬨€?
1. 璁＄畻閲?(FLOPs) 鍜岃緭鍏ュ舰鐘舵湁鍏筹紝浣嗘槸鍙傛暟閲?(params) 鍒欏拰杈撳叆褰㈢姸鏃犲叧銆傞粯璁ょ殑杈撳叆褰㈢姸涓?(1, 40000, 4)銆?2. 涓€浜涜繍绠楁搷浣滀笉璁″叆璁＄畻閲?(FLOPs)锛屾瘮濡傝鍍廏N鍜屽畾鍒剁殑杩愮畻鎿嶄綔锛岃缁嗙粏鑺傝鍙傝€?[`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py)銆?3. 鎴戜滑鐜板湪浠呬粎鏀寔鍗曟ā鎬佽緭鍏ワ紙鐐逛簯鎴栬€呭浘鐗囷級鐨勫崟闃舵妯″瀷鐨勮绠楅噺 (FLOPs) 璁＄畻锛屾垜浠皢浼氬湪鏈潵鏀寔涓ら樁娈靛拰澶氭ā鎬佹ā鍨嬬殑璁＄畻銆?
&#8195;

## 妯″瀷杞崲

### RegNet 妯″瀷杞崲鍒?MMDetection

`tools/model_converters/regnet2mmdet.py` 灏?pycls 棰勮缁?RegNet 妯″瀷涓殑閿浆鎹负 MMDetection 椋庢牸銆?
```shell
python tools/model_converters/regnet2mmdet.py ${SRC} ${DST} [-h]
```

### Detectron ResNet 杞崲鍒?Pytorch

MMDetection 涓殑 `tools/detectron2pytorch.py` 鑳藉鎶婂師濮嬬殑 detectron 涓璁粌鐨?ResNet 妯″瀷鐨勯敭杞崲涓?PyTorch 椋庢牸銆?
```shell
python tools/detectron2pytorch.py ${SRC} ${DST} ${DEPTH} [-h]
```

### 鍑嗗瑕佸彂甯冪殑妯″瀷

`tools/model_converters/publish_model.py` 甯姪鐢ㄦ埛鍑嗗浠栦滑鐢ㄤ簬鍙戝竷鐨勬ā鍨嬨€?
鍦ㄦ偍涓婁紶涓€涓ā鍨嬪埌浜戞湇鍔″櫒 (AWS) 涔嬪墠锛屾偍闇€瑕佸仛浠ヤ笅鍑犳锛?
1. 灏嗘ā鍨嬫潈閲嶈浆鎹负 CPU 寮犻噺
2. 鍒犻櫎璁板綍浼樺寲鍣ㄧ姸鎬?(optimizer states) 鐨勭浉鍏充俊鎭?3. 璁＄畻妫€鏌ョ偣 (checkpoint) 鏂囦欢鐨勫搱甯岀紪鐮?(hash id) 骞朵笖鎶婂搱甯岀紪鐮佸姞鍒版枃浠跺悕閲?
```shell
python tools/model_converters/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

渚嬪锛?
```shell
python tools/model_converters/publish_model.py work_dirs/faster_rcnn/latest.pth faster_rcnn_r50_fpn_1x_20190801.pth
```

鏈€缁堢殑杈撳嚭鏂囦欢鍚嶅皢浼氭槸 `faster_rcnn_r50_fpn_1x_20190801-{hash id}.pth`銆?
&#8195;

# 鏁版嵁闆嗚浆鎹?
`tools/dataset_converters/` 鍖呭惈杞崲鏁版嵁闆嗕负鍏朵粬鏍煎紡鐨勪竴浜涘伐鍏枫€傚叾涓ぇ澶氭暟杞崲鏁版嵁闆嗕负鍩轰簬 pickle 鐨勪俊鎭枃浠讹紝姣斿 KITTI锛宯uscense 鍜?lyft銆俉aymo 杞崲鍣ㄨ鐢ㄦ潵閲嶆柊缁勭粐 waymo 鍘熷鏁版嵁涓?KITTI 椋庢牸銆傜敤鎴疯兘澶熷弬鑰冨畠浠簡瑙ｆ垜浠浆鎹㈡暟鎹牸寮忕殑鏂规硶銆傚皢瀹冧滑淇敼涓?nuImages 杞崲鍣ㄧ瓑鑴氭湰涔熷緢鏂逛究銆?
涓轰簡杞崲 nuImages 鏁版嵁闆嗕负 COCO 鏍煎紡锛岃浣跨敤涓嬮潰鐨勬寚浠わ細

```shell
python -u tools/dataset_converters/nuimage_converter.py --data-root ${DATA_ROOT} --version ${VERSIONS} \
                                                    --out-dir ${OUT_DIR} --nproc ${NUM_WORKERS} --extra-tag ${TAG}
```

- `--data-root`: 鏁版嵁闆嗙殑鏍圭洰褰曪紝榛樿涓?`./data/nuimages`銆?- `--version`: 鏁版嵁闆嗙殑鐗堟湰锛岄粯璁や负 `v1.0-mini`銆傝鑾峰彇瀹屾暣鏁版嵁闆嗭紝璇蜂娇鐢?`--version v1.0-train v1.0-val v1.0-mini`銆?- `--out-dir`: 娉ㄩ噴鍜岃涔夋帺鐮佺殑杈撳嚭鐩綍锛岄粯璁や负 `./data/nuimages/annotations/`銆?- `--nproc`: 鏁版嵁鍑嗗鐨勮繘绋嬫暟锛岄粯璁や负 `4`銆傜敱浜庡浘鐗囨槸骞惰澶勭悊鐨勶紝鏇村ぇ鐨勮繘绋嬫暟鐩兘澶熷噺灏戝噯澶囨椂闂淬€?- `--extra-tag`: 娉ㄩ噴鐨勯澶栨爣绛撅紝榛樿涓?`nuimages`銆傝繖鍙敤浜庡皢涓嶅悓鏃堕棿澶勭悊鐨勪笉鍚屾敞閲婂垎寮€浠ヤ緵鐮旂┒銆?
鏇村鐨勬暟鎹噯澶囩粏鑺傚弬鑰?[doc](https://mmdetection3d.readthedocs.io/zh_CN/latest/data_preparation.html)锛宯uImages 鏁版嵁闆嗙殑缁嗚妭鍙傝€?[README](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/nuimages/README.md/)銆?
&#8195;

# 鍏朵粬鍐呭

## 鎵撳嵃瀹屾暣鐨勯厤缃枃浠?
`tools/misc/print_config.py` 閫愬瓧鎵撳嵃鏁翠釜閰嶇疆鏂囦欢锛屽睍寮€鎵€鏈夌殑瀵煎叆銆?
```shell
python tools/misc/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

