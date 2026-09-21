# 鍦ㄨ嚜瀹氫箟鏁版嵁闆嗕笂杩涜璁粌

鏈枃灏嗕富瑕佷粙缁嶅浣曚娇鐢ㄨ嚜瀹氫箟鏁版嵁闆嗘潵杩涜妯″瀷鐨勮缁冨拰娴嬭瘯锛屼互 Waymo 鏁版嵁闆嗕綔涓虹ず渚嬫潵璇存槑鏁翠釜娴佺▼銆?
鍩烘湰姝ラ濡備笅鎵€绀猴細

1. 鍑嗗鑷畾涔夋暟鎹泦锛?2. 鍑嗗閰嶇疆鏂囦欢锛?3. 鍦ㄨ嚜瀹氫箟鏁版嵁闆嗕笂杩涜妯″瀷鐨勮缁冦€佹祴璇曞拰鎺ㄧ悊銆?
## 鍑嗗鑷畾涔夋暟鎹泦

鍦?MMDetection3D 涓湁涓夌鏂瑰紡鏉ヨ嚜瀹氫箟涓€涓柊鐨勬暟鎹泦锛?
1. 灏嗘柊鏁版嵁闆嗙殑鏁版嵁鏍煎紡閲嶆柊缁勭粐鎴愬凡鏀寔鐨勬暟鎹泦鏍煎紡锛?2. 灏嗘柊鏁版嵁闆嗙殑鏁版嵁鏍煎紡閲嶆柊缁勭粐鎴愬凡鏀寔鐨勪竴绉嶄腑闂存牸寮忥紱
3. 浠庡ご寮€濮嬪垱寤轰竴涓柊鐨勬暟鎹泦銆?
鐢变簬鍓嶄袱绉嶆柟寮忔瘮绗笁绉嶆柟寮忔洿鍔犲鏄擄紝鎴戜滑鏇村姞寤鸿閲囩敤鍓嶄袱绉嶆柟寮忔潵鑷畾涔夋暟鎹泦銆?
鍦ㄦ湰鏂囦腑锛屾垜浠粰鍑虹ず渚嬪皢鏁版嵁杞崲鎴?KITTI 鏁版嵁闆嗙殑鏁版嵁鏍煎紡锛屼綘鍙互鍙傝€冩澶勫皢浣犵殑鏁版嵁闆嗛噸鏂扮粍缁囨垚 KITTI 鏍煎紡銆傚叧浜庢爣鍑嗘牸寮忕殑鏁版嵁闆嗭紝浣犲彲浠ュ弬鑰僛鑷畾涔夋暟鎹泦鏂囨。](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/docs/zh_cn/advanced_guides/customize_dataset.md)銆?
**娉ㄦ剰**锛氳€冭檻鍒?Waymo 鏁版嵁闆嗙殑鏍煎紡涓庣幇鏈夌殑鍏朵粬鏁版嵁闆嗙殑鏍煎紡鐨勫樊鍒緝澶э紝鍥犳鏈枃浠ヨ鏁版嵁闆嗕负渚嬫潵璁茶В濡備綍鑷畾涔夋暟鎹泦锛屼粠鑰屾柟渚跨悊瑙ｆ暟鎹泦鑷畾涔夌殑杩囩▼銆傝嫢闇€瑕佸垱寤虹殑鏂版暟鎹泦涓庣幇鏈夌殑鏁版嵁闆嗙殑缁勭粐鏍煎紡杈冧负鐩镐技锛屽 Lyft 鏁版嵁闆嗗拰 nuScenes 鏁版嵁闆嗭紝閲囩敤瀵规暟鎹泦鐨勪腑闂存牸寮忚繘琛岃浆鎹㈢殑鏂瑰紡锛堢浜岀鏂瑰紡锛夌浉姣斾簬閲囩敤瀵规暟鎹牸寮忚繘琛岃浆鎹㈢殑鏂瑰紡锛堢涓€绉嶆柟寮忥級浼氭洿鍔犵畝鍗曟槗琛屻€?
### KITTI 鏁版嵁闆嗘牸寮?
搴旂敤浜?3D 鐩爣妫€娴嬬殑 KITTI 鍘熷鏁版嵁闆嗙殑缁勭粐鏂瑰紡閫氬父濡備笅鎵€绀猴紝鍏朵腑 `ImageSets` 鍖呭惈鏁版嵁闆嗗垝鍒嗘枃浠讹紝鐢ㄤ互鍒掑垎璁粌闆?楠岃瘉闆?娴嬭瘯闆嗭紝`calib` 鍖呭惈瀵逛簬姣忎釜鏁版嵁鏍锋湰鐨勬爣瀹氫俊鎭紝`image_2` 鍜?`velodyne` 鍒嗗埆鍖呭惈鍥惧儚鏁版嵁鍜岀偣浜戞暟鎹紝`label_2` 鍖呭惈涓?3D 鐩爣妫€娴嬬浉鍏崇殑鏍囨敞鏂囦欢銆?
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
鈹?  鈹?  鈹溾攢鈹€ training
鈹?  鈹?  鈹?  鈹溾攢鈹€ calib
鈹?  鈹?  鈹?  鈹溾攢鈹€ image_2
鈹?  鈹?  鈹?  鈹溾攢鈹€ label_2
鈹?  鈹?  鈹?  鈹溾攢鈹€ velodyne
```

KITTI 瀹樻柟鎻愪緵鐨勭洰鏍囨娴嬪紑鍙慬宸ュ叿鍖匽(https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip)璇︾粏鎻忚堪浜?KITTI 鏁版嵁闆嗙殑鏍囨敞鏍煎紡锛屼緥濡傦紝KITTI 鏍囨敞鏍煎紡鍖呭惈浜嗕互涓嬬殑鏍囨敞淇℃伅锛?
```
#  鍊?   鍚嶇О      鎻忚堪
----------------------------------------------------------------------------
   1    绫诲瀷      鎻忚堪妫€娴嬬洰鏍囩殑绫诲瀷锛?Car'锛?Van'锛?Truck'锛?                  'Pedestrian'锛?Person_sitting'锛?Cyclist'锛?Tram'锛?                  'Misc' 鎴?'DontCare'
   1    鎴柇绋嬪害銆€ 浠?0锛堥潪鎴柇锛夊埌 1锛堟埅鏂級鐨勬诞鐐规暟锛屽叾涓埅鏂寚鐨勬槸绂诲紑妫€娴嬪浘鍍忚竟鐣岀殑妫€娴嬬洰鏍?   1    閬尅绋嬪害銆€ 鐢ㄦ潵琛ㄧず閬尅鐘舵€佺殑鍥涚鏁存暟锛?锛?锛?锛?锛?
                  0 = 鍙锛? = 閮ㄥ垎閬尅
                  2 = 澶ч潰绉伄鎸★紝3 = 鏈煡
   1    瑙傛祴瑙?   瑙傛祴鐩爣鐨勮搴︼紝鍙栧€艰寖鍥翠负 [-pi..pi]
   4    鏍囨敞妗?   妫€娴嬬洰鏍囧湪鍥惧儚涓殑浜岀淮鏍囨敞妗嗭紙浠?涓哄垵濮嬩笅鏍囷級锛氬寘鎷瘡涓娴嬬洰鏍囩殑宸︿笂瑙掑拰鍙充笅瑙掔殑鍧愭爣
   3    缁村害銆€    妫€娴嬬洰鏍囩殑涓夌淮缁村害锛氶珮搴︺€佸搴︺€侀暱搴︼紙浠ョ背涓哄崟浣嶏級
   3    浣嶇疆銆€    鐩告満鍧愭爣绯讳笅鐨勪笁缁翠綅缃?x锛寉锛寊锛堜互绫充负鍗曚綅锛?   1    y 鏃嬭浆銆€  鐩告満鍧愭爣绯讳笅妫€娴嬬洰鏍囩粫鐫€Y杞寸殑鏃嬭浆瑙掞紝鍙栧€艰寖鍥翠负 [-pi..pi]
   1    寰楀垎銆€    浠呭湪璁＄畻缁撴灉鏃朵娇鐢紝妫€娴嬩腑琛ㄧず缃俊搴︾殑娴偣鏁帮紝鐢ㄤ簬鐢熸垚 p/r 鏇茬嚎锛屽湪p/r 鍥句腑锛岃秺楂樼殑鏇茬嚎琛ㄧず缁撴灉瓒婂ソ銆?```

鍋囧畾鎴戜滑浣跨敤 Waymo 鏁版嵁闆嗐€?
鍦ㄤ笅杞藉ソ鏁版嵁闆嗗悗锛屾垜浠渶瑕佸疄鐜颁竴涓嚱鏁扮敤鏉ュ皢杈撳叆鏁版嵁鍜屾爣娉ㄦ枃浠惰浆鎹㈡垚 KITTI 椋庢牸銆傜劧鍚庢垜浠彲浠ラ€氳繃缁ф壙 `KittiDataset` 瀹炵幇 `WaymoDataset`锛岀敤鏉ュ姞杞芥暟鎹互鍙婅缁冩ā鍨嬶紝閫氳繃缁ф壙 `KittiMetric` 瀹炵幇 `WaymoMetric` 鏉ュ仛妯″瀷鐨勮瘎浼般€?
鍏蜂綋鏉ヨ锛岄鍏堜娇鐢╗鏁版嵁杞崲鍣╙(https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/tools/dataset_converters/waymo_converter.py)灏?Waymo 鏁版嵁闆嗚浆鎹㈡垚 KITTI 鏁版嵁闆嗙殑鏍煎紡锛屽苟瀹氫箟 [Waymo 绫籡(https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/datasets/waymo_dataset.py)瀵硅浆鎹㈢殑鏁版嵁杩涜澶勭悊銆傛澶栭渶瑕佹坊鍔?waymo [璇勪及绫籡(https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/mmdet3d/evaluation/metrics/waymo_metric.py)鏉ヨ瘎浼扮粨鏋溿€傚洜涓烘垜浠皢 Waymo 鍘熷鏁版嵁闆嗚繘琛岄澶勭悊骞堕噸鏂扮粍缁囨垚 KITTI 鏁版嵁闆嗙殑鏍煎紡锛屽洜姝ゅ彲浠ユ瘮杈冨鏄撻€氳繃缁ф壙 KittiDataset 绫绘潵瀹炵幇 WaymoDataset 绫汇€傞渶瑕佹敞鎰忕殑鏄紝鐢变簬 Waymo 鏁版嵁闆嗘湁鐩稿簲鐨勫畼鏂硅瘎浼版柟娉曪紝鎴戜滑闇€瑕佽繘涓€姝ュ疄鐜版柊鐨?Waymo 璇勪及鏂规硶锛屾洿澶氬叧浜庤瘎浼版柟娉曞弬鑰僛璇勪及鏂囨。](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/metric_and_evaluator.md)銆傛渶鍚庯紝鐢ㄦ埛鍙互鎴愬姛鍦拌浆鎹㈡暟鎹苟浣跨敤 `WaymoDataset` 璁粌浠ュ強 `WaymoMetric` 璇勪及妯″瀷銆?
鏇村鍏充簬 Waymo 鏁版嵁闆嗛澶勭悊鐨勪腑闂寸粨鏋滅殑缁嗚妭锛岃鍙傜収瀵瑰簲鐨刐璇存槑鏂囨。](https://mmdetection3d.readthedocs.io/zh_CN/latest/datasets/waymo_det.html)銆?
## 鍑嗗閰嶇疆鏂囦欢

绗簩姝ユ槸鍑嗗閰嶇疆鏂囦欢鏉ュ府鍔╂暟鎹泦鐨勮鍙栧拰浣跨敤锛屽彟澶栵紝涓轰簡鍦?3D 妫€娴嬩腑鑾峰緱涓嶉敊鐨勬€ц兘锛岃皟鏁磋秴鍙傛暟閫氬父鏄繀瑕佺殑銆?
鍋囪鎴戜滑鎯宠浣跨敤 PointPillars 妯″瀷鍦?Waymo 鏁版嵁闆嗕笂瀹炵幇涓夌被鐨?3D 鐩爣妫€娴嬶細vehicle銆乧yclist銆乸edestrian锛屽弬鐓?KITTI 鏁版嵁闆哰閰嶇疆鏂囦欢](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/datasets/kitti-3d-3class.py)銆佹ā鍨媅閰嶇疆鏂囦欢](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/models/pointpillars_hv_secfpn_kitti.py)鍜孾鏁翠綋閰嶇疆鏂囦欢](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py)锛屾垜浠渶瑕佸噯澶嘯鏁版嵁闆嗛厤缃枃浠禲(https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/datasets/waymoD5-3d-3class.py)銆乕妯″瀷閰嶇疆鏂囦欢](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/_base_/models/pointpillars_hv_secfpn_waymo.py)锛屽苟灏嗚繖涓ょ鏂囦欢杩涜缁撳悎寰楀埌[鏁翠綋閰嶇疆鏂囦欢](https://github.com/open-mmlab/mmdetection3d/blob/dev-1.x/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py)銆?
## 璁粌涓€涓柊鐨勬ā鍨?
涓轰簡浣跨敤涓€涓柊鐨勯厤缃枃浠舵潵璁粌妯″瀷锛屽彲浠ラ€氳繃涓嬮潰鐨勫懡浠ゆ潵瀹炵幇锛?
```shell
python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py
```

鏇村鐨勪娇鐢ㄧ粏鑺傦紝璇峰弬鑰僛妗堜緥 1](https://mmdetection3d.readthedocs.io/zh_CN/latest/1_exist_data_model.html)銆?
## 娴嬭瘯鍜屾帹鐞?
涓轰簡娴嬭瘯宸茬粡璁粌濂界殑妯″瀷鐨勬€ц兘锛屽彲浠ラ€氳繃涓嬮潰鐨勫懡浠ゆ潵瀹炵幇锛?
```shell
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class.py work_dirs/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymoD5-3d-3class/latest.pth
```

**娉ㄦ剰**锛氫负浜嗕娇鐢?Waymo 鏁版嵁闆嗙殑璇勪及鏂规硶锛岄渶瑕佸弬鑰僛璇存槑鏂囨。](https://mmdetection3d.readthedocs.io/zh_CN/latest/datasets/waymo_det.html)骞舵寜鐓у畼鏂规寚瀵兼潵鍑嗗涓庤瘎浼扮浉鍏宠仈鐨勬枃浠躲€?
鏇村鏈夊叧娴嬭瘯鍜屾帹鐞嗙殑浣跨敤缁嗚妭锛岃鍙傝€僛妗堜緥 1](https://mmdetection3d.readthedocs.io/zh_CN/latest/1_exist_data_model.html) 銆?
