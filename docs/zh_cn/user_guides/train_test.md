# 鍦ㄦ爣娉ㄦ暟鎹泦涓婃祴璇曞拰璁粌

### 鍦ㄦ爣鍑嗘暟鎹泦涓婃祴璇曞凡鏈夋ā鍨?
- 鍗曟樉鍗?- CPU
- 鍗曡妭鐐瑰鏄惧崱
- 澶氳妭鐐?
浣犲彲浠ラ€氳繃浠ヤ笅鍛戒护鏉ユ祴璇曟暟鎹泦锛?
```shell
# 鍗曞潡鏄惧崱娴嬭瘯
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--show-dir ${SHOW_DIR}]

# CPU锛氱鐢ㄦ樉鍗″苟杩愯鍗曞潡 CPU 娴嬭瘯鑴氭湰锛堝疄楠屾€э級
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--show-dir ${SHOW_DIR}]

# 澶氬潡鏄惧崱娴嬭瘯
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

**娉ㄦ剰**:

鐩墠鎴戜滑鍙敮鎸?SMOKE 鐨?CPU 鎺ㄧ悊娴嬭瘯銆?
鍙€夊弬鏁帮細

- `--show`锛氬鏋滆鎸囧畾锛屾娴嬬粨鏋滀細鍦ㄩ潤榛樻ā寮忎笅琚繚瀛橈紝鐢ㄤ簬璋冭瘯鍜屽彲瑙嗗寲锛屼絾鍙湪鍗曞潡 GPU 娴嬭瘯鐨勬儏鍐典笅鐢熸晥锛屽拰 `--show-dir` 鎼厤浣跨敤銆?- `--show-dir`锛氬鏋滆鎸囧畾锛屾娴嬬粨鏋滀細琚繚瀛樺湪鎸囧畾鏂囦欢澶逛笅鐨?`***_points.obj` 鍜?`***_pred.obj` 鏂囦欢涓紝鐢ㄤ簬璋冭瘯鍜屽彲瑙嗗寲锛屼絾鍙湪鍗曞潡 GPU 娴嬭瘯鐨勬儏鍐典笅鐢熸晥锛屽浜庤繖涓€夐」锛屽浘褰㈠寲鐣岄潰鍦ㄤ綘鐨勭幆澧冧腑涓嶆槸蹇呴渶鐨勩€?
鎵€鏈夊拰璇勪及鐩稿叧鐨勫弬鏁板湪鐩稿簲鐨勬暟鎹泦閰嶇疆鐨?`test_evaluator` 涓缃€備緥濡?`test_evaluator = dict(type='KittiMetric', ann_file=data_root + 'kitti_infos_val.pkl', pklfile_prefix=None, submission_prefix=None)`

鍙傛暟锛?
- `type`锛氱浉瀵瑰簲鐨勮瘎浠锋寚鏍囧悕锛岄€氬父鍜屾暟鎹泦鐩稿叧鑱斻€?- `ann_file`锛氭爣娉ㄦ枃浠惰矾寰勩€?- `pklfile_prefix`锛氬彲閫夊弬鏁般€傝緭鍑虹粨鏋滀繚瀛樻垚 pickle 鏍煎紡鐨勬枃浠跺悕銆傚鏋滄病鏈夋寚瀹氾紝缁撴灉灏嗕笉浼氫繚瀛樻垚鏂囦欢銆?- `submission_prefix`锛氬彲閫夊弬鏁般€傜粨鏋滃皢琚繚瀛樺埌鏂囦欢涓紝鐒跺悗浣犲彲浠ュ皢瀹冧笂浼犲埌瀹樻柟璇勪及鏈嶅姟鍣ㄤ腑銆?
绀轰緥锛?
鍋囧畾浣犲凡缁忔妸妯″瀷鏉冮噸鏂囦欢涓嬭浇鍒?`checkpoints/` 鏂囦欢澶逛笅锛?
1. 鍦?ScanNet 鏁版嵁闆嗕笂娴嬭瘯 VoteNet锛屼繚瀛樻ā鍨嬶紝鍙鍖栭娴嬬粨鏋?
   ```shell
   python tools/test.py configs/votenet/votenet_8xb8_scannet-3d.py \
       checkpoints/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a.pth \
       --show --show-dir ./data/scannet/show_results
   ```

2. 鍦?ScanNet 鏁版嵁闆嗕笂娴嬭瘯 VoteNet锛屼繚瀛樻ā鍨嬶紝鍙鍖栭娴嬬粨鏋滐紝鍙鍖栫湡瀹炴爣绛撅紝璁＄畻 mAP

   ```shell
   python tools/test.py configs/votenet/votenet_8xb8_scannet-3d.py \
       checkpoints/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a.pth \
       --show --show-dir ./data/scannet/show_results
   ```

3. 鍦?ScanNet 鏁版嵁闆嗕笂娴嬭瘯 VoteNet锛堜笉淇濆瓨娴嬭瘯缁撴灉锛夛紝璁＄畻 mAP

   ```shell
   python tools/test.py configs/votenet/votenet_8xb8_scannet-3d.py \
       checkpoints/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a.pth
   ```

4. 浣跨敤 8 鍧楁樉鍗″湪 KITTI 鏁版嵁闆嗕笂娴嬭瘯 SECOND锛岃绠?mAP

   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py \
       checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth
   ```

5. 浣跨敤 8 鍧楁樉鍗″湪 nuScenes 鏁版嵁闆嗕笂娴嬭瘯 PointPillars锛岀敓鎴愭彁浜ょ粰瀹樻柟璇勬祴鏈嶅姟鍣ㄧ殑 json 鏂囦欢

   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py \
       checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth \
      --cfg-options 'test_evaluator.jsonfile_prefix=./pointpillars_nuscenes_results'
   ```

   鐢熸垚鐨勭粨鏋滀細淇濆瓨鍦?`./pointpillars_nuscenes_results` 鐩綍銆?
6. 浣跨敤 8 鍧楁樉鍗″湪 KITTI 鏁版嵁闆嗕笂娴嬭瘯 SECOND锛岀敓鎴愭彁浜ょ粰瀹樻柟璇勬祴鏈嶅姟鍣ㄧ殑 txt 鏂囦欢

   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py \
       checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth \
       --cfg-options 'test_evaluator.pklfile_prefix=./second_kitti_results' 'submission_prefix=./second_kitti_results'
   ```

   鐢熸垚鐨勭粨鏋滀細淇濆瓨鍦?`./second_kitti_results` 鐩綍銆?
7. 浣跨敤 8 鍧楁樉鍗″湪 Lyft 鏁版嵁闆嗕笂娴嬭瘯 PointPillars锛岀敓鎴愭彁浜ょ粰鎺掕姒滅殑 pkl 鏂囦欢

   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/pointpillars/hv_pointpillars_fpn_sbn-2x8_2x_lyft-3d.py \
       checkpoints/hv_pointpillars_fpn_sbn-2x8_2x_lyft-3d_latest.pth \
       --cfg-options 'test_evaluator.jsonfile_prefix=results/pp_lyft/results_challenge' \
       'test_evaluator.csv_savepath=results/pp_lyft/results_challenge.csv' \
       'test_evaluator.pklfile_prefix=results/pp_lyft/results_challenge.pkl'
   ```

   **娉ㄦ剰**锛氫负浜嗙敓鎴?Lyft 鏁版嵁闆嗙殑鎻愪氦缁撴灉锛宍--eval-options` 蹇呴』鎸囧畾 `csv_savepath`銆傜敓鎴?csv 鏂囦欢鍚庯紝浣犲彲浠ヤ娇鐢╗缃戠珯](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/submit)涓婄粰鍑虹殑 kaggle 鍛戒护鎻愪氦缁撴灉銆?
   娉ㄦ剰鍦?[Lyft 鏁版嵁闆嗙殑閰嶇疆鏂囦欢](../../configs/_base_/datasets/lyft-3d.py)锛宍test` 涓殑 `ann_file` 鍊间负 `lyft_infos_test.pkl`锛屾槸娌℃湁鏍囨敞鐨?Lyft 瀹樻柟娴嬭瘯闆嗐€傝鍦ㄩ獙璇佹暟鎹泦涓婃祴璇曪紝璇锋妸瀹冩敼涓?`lyft_infos_val.pkl`銆?
8. 浣跨敤 8 鍧楁樉鍗″湪 waymo 鏁版嵁闆嗕笂娴嬭瘯 PointPillars锛屼娇鐢?waymo 搴﹂噺鏂规硶璁＄畻 mAP

   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car.py  \
       checkpoints/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car_latest.pth \
       --cfg-options 'test_evaluator.pklfile_prefix=results/waymo-car/kitti_results' \
       'test_evaluator.submission_prefix=results/waymo-car/kitti_results'
   ```

   **娉ㄦ剰**锛氬浜?waymo 鏁版嵁闆嗕笂鐨勮瘎浼帮紝璇锋牴鎹甗璇存槑](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md/)鏋勫缓浜岃繘鍒舵枃浠?`compute_detection_metrics_main` 鏉ュ仛搴﹂噺璁＄畻锛屽苟鎶婂畠鏀惧湪 `mmdet3d/core/evaluation/waymo_utils/`銆傦紙鍦ㄤ娇鐢?bazel 鏋勫缓  `compute_detection_metrics_main` 鏃讹紝鏈夋椂浼氬嚭鐜?`'round' is not a member of 'std'` 鐨勯敊璇紝鎴戜滑鍙渶瑕佹妸閭ｄ釜鏂囦欢涓?`round` 鍓嶇殑 `std::` 鍘绘帀銆傦級浜岃繘鍒舵枃浠剁敓鎴愭椂闇€瑕佸湪 `--eval-options` 涓粰瀹?`pklfile_prefix`銆傚浜庡害閲忔柟娉曪紝`waymo` 鏄帹鑽愮殑瀹樻柟璇勪及绛栫暐锛岀洰鍓?`kitti` 璇勪及鏄緷鐓?KITTI 鑰屾潵鐨勶紝姣忎釜闅惧害鐨勭粨鏋滃拰 KITTI 鐨勫畾涔夊苟涓嶅畬鍏ㄤ竴鑷淬€傜洰鍓嶅ぇ澶氭暟鐗╀綋閮借鏍囪涓?闅惧害锛屼細鍦ㄦ湭鏉ヤ慨澶嶃€傚畠鐨勪笉绋冲畾鍘熷洜鍖呮嫭璇勪及鐨勮绠楀ぇ銆佽浆鎹㈠悗鐨勬暟鎹己涔忛伄鎸″拰鎴柇銆侀毦搴︾殑瀹氫箟涓嶅悓浠ュ強骞冲潎绮惧害鐨勮绠楁柟娉曚笉鍚屻€?
9. 浣跨敤 8 鍧楁樉鍗″湪 waymo 鏁版嵁闆嗕笂娴嬭瘯 PointPillars锛岀敓鎴?bin 鏂囦欢骞舵彁浜ゅ埌鎺掕姒?
   ```shell
   ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-car.py  \
       checkpoints/hv_pointpillars_secfpn_sbn-2x16_2x_waymo-3d-car_latest.pth \
       --cfg-options 'test_evaluator.pklfile_prefix=results/waymo-car/kitti_results' \
       'test_evaluator.submission_prefix=results/waymo-car/kitti_results'
   ```

   **娉ㄦ剰**锛氱敓鎴?bin 鏂囦欢鍚庯紝浣犲彲浠ョ畝鍗曞湴鏋勫缓浜岃繘鍒舵枃浠? `create_submission`锛屽苟鏍规嵁[璇存槑](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md/)鍒涘缓鎻愪氦鐨勬枃浠躲€傝鍦ㄩ獙璇佹湇鍔″櫒涓婅瘎娴嬮獙璇佹暟鎹泦锛屼綘涔熷彲浠ョ敤鍚屾牱鐨勬柟寮忕敓鎴愭彁浜ょ殑鏂囦欢銆?
## 鍦ㄦ爣鍑嗘暟鎹泦涓婅缁冮瀹氫箟妯″瀷

MMDetection3D 鍒嗗埆鐢?`MMDistributedDataParallel` and `MMDataParallel` 瀹炵幇浜嗗垎甯冨紡璁粌鍜岄潪鍒嗗竷寮忚缁冦€?
鎵€鏈夌殑杈撳嚭锛堟棩蹇楁枃浠跺拰妯″瀷鏉冮噸鏂囦欢锛夐兘浼氳淇濆瓨鍒板伐浣滅洰褰曚笅锛岄€氳繃閰嶇疆鏂囦欢閲岀殑 `work_dir` 鎸囧畾銆?
榛樿鎴戜滑姣忚繃涓€涓懆鏈熼兘鍦ㄩ獙璇佹暟鎹泦涓婅瘎娴嬫ā鍨嬶紝浣犲彲浠ラ€氳繃鍦ㄨ缁冮厤缃噷娣诲姞闂撮殧鍙傛暟鏉ユ敼鍙樿瘎娴嬬殑鏃堕棿闂撮殧锛?
```python
train_cfg = dict(type='EpochBasedTrainLoop', val_interval=1)  # 姣?2涓懆鏈熻瘎浼颁竴娆℃ā鍨?```

**閲嶈**锛氶厤缃枃浠朵腑鐨勯粯璁ゅ涔犵巼瀵瑰簲 8 鍧楁樉鍗★紝閰嶇疆鏂囦欢鍚嶉噷鏈夊叿浣撶殑鎵归噺澶у皬锛屾瘮濡?'2xb8' 琛ㄧず涓€鍏?8 鍧楁樉鍗★紝姣忓潡鏄惧崱 2 涓牱鏈€?鏍规嵁 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677)锛屽綋浣犱娇鐢ㄤ笉鍚屾暟閲忕殑鏄惧崱鎴栨瘡鍧楁樉鍗℃湁涓嶅悓鏁伴噺鐨勫浘鍍忔椂锛岄渶瑕佷緷鎵归噺澶у皬鎸夋瘮渚嬭皟鏁村涔犵巼銆傚鏋滅敤 4 鍧楁樉鍗°€佹瘡鍧楁樉鍗?2 骞呭浘鍍忔椂瀛︿範鐜囦负 0.01锛岄偅涔堢敤 16 鍧楁樉鍗°€佹瘡鍧楁樉鍗?4 骞呭浘鍍忔椂瀛︿範鐜囧簲璁句负 0.08銆傜劧鑰岋紝鐢变簬澶у鏁版ā鍨嬩娇鐢?ADAM 鑰屼笉鏄?SGD 杩涜浼樺寲锛屼笂杩拌鍒欏彲鑳藉苟涓嶉€傜敤锛岀敤鎴烽渶瑕佽嚜宸辫皟鏁村涔犵巼銆?
### 浣跨敤鍗曞潡鏄惧崱杩涜璁粌

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

濡傛灉浣犳兂鍦ㄥ懡浠や腑鎸囧畾宸ヤ綔鐩綍锛屾坊鍔犲弬鏁?`--work-dir ${YOUR_WORK_DIR}`銆?
### 浣跨敤 CPU 杩涜璁粌 (瀹為獙鎬?

鍦?CPU 涓婅缁冪殑杩囩▼涓庡崟 GPU 璁粌涓€鑷淬€?鎴戜滑鍙渶瑕佸湪璁粌杩囩▼涔嬪墠绂佺敤鏄惧崱銆?
```shell
export CUDA_VISIBLE_DEVICES=-1
```

涔嬪悗杩愯鍗曟樉鍗¤缁冭剼鏈嵆鍙€?
**娉ㄦ剰**锛?
鐩墠锛屽ぇ澶氭暟鐐逛簯鐩稿叧绠楁硶閮戒緷璧栦簬 3D CUDA 绠楀瓙锛屾棤娉曞湪 CPU 涓婅繘琛岃缁冦€?涓€浜涘崟鐩?3D 鐗╀綋妫€娴嬬畻娉曪紝渚嬪 FCOS3D銆丼MOKE 鍙互鍦?CPU 涓婅繘琛岃缁冦€傛垜浠笉鎺ㄨ崘鐢ㄦ埛浣跨敤 CPU 杩涜璁粌锛岃繖澶繃缂撴參銆傛垜浠敮鎸佽繖涓姛鑳芥槸涓轰簡鏂逛究鐢ㄦ埛鍦ㄦ病鏈夋樉鍗＄殑鏈哄櫒涓婅皟璇曟煇浜涚壒瀹氱殑鏂规硶銆?
### 浣跨敤澶氬潡鏄惧崱杩涜璁粌

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

鍙€夊弬鏁帮細

- `--cfg-options 'Key=value'`锛氳鐩栦娇鐢ㄧ殑閰嶇疆涓殑涓€浜涜瀹氥€?
### 浣跨敤澶氫釜鏈哄櫒杩涜璁粌

濡傛灉瑕佸湪 [slurm](https://slurm.schedmd.com/) 绠＄悊鐨勯泦缇や笂杩愯 MMDectection3D锛屼綘鍙互浣跨敤 `slurm_train.sh` 鑴氭湰锛堣鑴氭湰涔熸敮鎸佸崟鏈鸿缁冿級

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

涓嬮潰鏄竴涓娇鐢?16 鍧楁樉鍗″湪 dev 鍒嗗尯涓婅缁?Mask R-CNN 鐨勭ず渚嬶細

```shell
GPUS=16 ./tools/slurm_train.sh dev pp_kitti_3class configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py /nfs/xxxx/pp_kitti_3class
```

浣犲彲浠ユ煡鐪?[slurm_train.sh](https://github.com/open-mmlab/mmdetection/blob/master/tools/slurm_train.sh) 鏉ヨ幏鍙栨墍鏈夌殑鍙傛暟鍜岀幆澧冨彉閲忋€?
濡傛灉鎮ㄦ兂浣跨敤鐢?ethernet 杩炴帴璧锋潵鐨勫鍙版満鍣紝 鎮ㄥ彲浠ヤ娇鐢ㄤ互涓嬪懡浠?

鍦ㄧ涓€鍙版満鍣ㄤ笂:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR ./tools/dist_train.sh $CONFIG $GPUS
```

鍦ㄧ浜屽彴鏈哄櫒涓?

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR ./tools/dist_train.sh $CONFIG $GPUS
```

浣嗘槸锛屽鏋滄偍涓嶄娇鐢ㄩ珮閫熺綉璺繛鎺ヨ繖鍑犲彴鏈哄櫒鐨勮瘽锛岃缁冨皢浼氶潪甯告參銆?
### 鍦ㄥ崟涓満鍣ㄤ笂鍚姩澶氫釜浠诲姟

濡傛灉浣犲湪鍗曚釜鏈哄櫒涓婂惎鍔ㄥ涓换鍔★紝姣斿锛屽湪鍏锋湁8鍧楁樉鍗＄殑鏈哄櫒涓婅繘琛?涓?鍧楁樉鍗¤缁冪殑浠诲姟锛屼綘闇€瑕佷负姣忎釜浠诲姟鎸囧畾涓嶅悓鐨勭鍙ｏ紙榛樿涓?9500锛変互閬垮厤閫氫俊鍐茬獊銆?
濡傛灉浣犱娇鐢?`dist_train.sh` 鍚姩璁粌浠诲姟锛屽彲浠ュ湪鍛戒护涓缃鍙ｏ細

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

濡傛灉浣犱娇鐢?Slurm 鍚姩璁粌浠诲姟锛屾湁涓ょ鏂瑰紡鎸囧畾绔彛锛?
1. 閫氳繃 `--cfg-options` 璁剧疆绔彛锛岃繖鏄洿鎺ㄨ崘鐨勶紝鍥犱负瀹冧笉鏀瑰彉鍘熸潵鐨勯厤缃?
   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --cfg-options 'env_cfg.dist_cfg.port=29500'
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --cfg-options 'env_cfg.dist_cfg.port=29501'
   ```

2. 淇敼閰嶇疆鏂囦欢锛堥€氬父鍦ㄩ厤缃枃浠剁殑鍊掓暟绗?琛岋級鏉ヨ缃笉鍚岀殑閫氫俊绔彛

   鍦?`config1.py` 涓紝

   ```python
   env_cfg = dict(
       dist_cfg=dict(backend='nccl', port=29500)
   )
   ```

   鍦?`config2.py` 涓紝

   ```python
   env_cfg = dict(
       dist_cfg=dict(backend='nccl', port=29501)
   )
   ```

   鐒跺悗锛屼綘鍙互浣跨敤 `config1.py` and `config2.py` 鍚姩涓や釜浠诲姟

   ```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```

