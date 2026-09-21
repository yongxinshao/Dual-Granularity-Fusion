# 甯歌闂瑙ｇ瓟

鎴戜滑鍒楀嚭浜嗕竴浜涚敤鎴峰拰寮€鍙戣€呭湪寮€鍙戣繃绋嬩腑浼氶亣鍒扮殑甯歌闂浠ュ強瀵瑰簲鐨勮В鍐虫柟妗堬紝濡傛灉鎮ㄥ彂鐜颁簡浠讳綍棰戠箒鍑虹幇鐨勯棶棰橈紝璇烽殢鏃舵墿鍏呮湰鍒楄〃锛岄潪甯告杩庢偍鎻愬嚭鐨勪换浣曡В鍐虫柟妗堛€傚鏋滄偍鍦ㄧ幆澧冮厤缃€佹ā鍨嬭缁冪瓑宸ヤ綔涓亣鍒颁换浣曠殑闂锛岃浣跨敤[闂妯℃澘](https://github.com/open-mmlab/mmdetection3d/blob/master/.github/ISSUE_TEMPLATE/error-report.md)鏉ュ垱寤虹浉搴旂殑 issue锛屽苟灏嗘墍闇€鐨勬墍鏈変俊鎭～鍏ュ埌闂妯℃澘涓紝鎴戜滑浼氬敖蹇В鍐虫偍鐨勯棶棰樸€?
## MMEngine/MMCV/MMDet/MMDet3D 瀹夎

- 璺?MMEngine, MMCV, MMDetection 鍜?MMDetection3D 鐩稿叧鐨勭紪璇戦棶棰? "ConvWS is already registered in conv layer"; "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

- MMDetection3D 闇€瑕佺殑 MMEngine, MMCV 鍜?MMDetection 鐨勭増鏈垪鍦ㄤ簡涓嬮潰銆傝瀹夎姝ｇ‘鐗堟湰鐨?MMEngine銆丮MCV 鍜?MMDetection 浠ラ伩鍏嶇浉鍏崇殑瀹夎闂銆?
  | MMDetection3D 鐗堟湰 |      MMEngine 鐗堟湰       |        MMCV 鐗堟湰        |     MMDetection 鐗堟湰     |
  | ------------------ | :----------------------: | :---------------------: | :----------------------: |
  | main               | mmengine>=0.8.0, \<1.0.0 | mmcv>=2.0.0rc4, \<2.2.0 | mmdet>=3.0.0rc5, \<3.4.0 |
  | v1.4.0             | mmengine>=0.8.0, \<1.0.0 | mmcv>=2.0.0rc4, \<2.2.0 | mmdet>=3.0.0rc5, \<3.4.0 |
  | v1.3.0             | mmengine>=0.8.0, \<1.0.0 | mmcv>=2.0.0rc4, \<2.2.0 | mmdet>=3.0.0rc5, \<3.3.0 |
  | v1.2.0             | mmengine>=0.8.0, \<1.0.0 | mmcv>=2.0.0rc4, \<2.1.0 |  mmdet>=3.0.0, \<3.2.0   |
  | v1.1.1             | mmengine>=0.7.1, \<1.0.0 | mmcv>=2.0.0rc4, \<2.1.0 |  mmdet>=3.0.0, \<3.1.0   |

  **娉ㄦ剰**锛氬鏋滀綘鎯冲畨瑁?mmdet3d-v1.0.0rcx锛屽彲浠ュ湪[姝ゅ](https://mmdetection3d.readthedocs.io/en/latest/faq.html#mmcv-mmdet-mmdet3d-installation)鎵惧埌 MMDetection锛孧MSegmentation 鍜?MMCV 鐨勫吋瀹圭増鏈€傝閫夋嫨姝ｇ‘鐗堟湰鐨?MMCV銆丮MDetection 鍜?MMSegmentation 浠ラ伩鍏嶅畨瑁呴棶棰樸€?
- 濡傛灉鎮ㄥ湪 `import open3d` 鏃堕亣鍒颁笅闈㈢殑闂锛?
  `OSError: /lib/x86_64-linux-gnu/libm.so.6: version 'GLIBC_2.27' not found`

  璇峰皢 open3d 鐨勭増鏈檷绾ц嚦 0.9.0.0锛屽洜涓烘渶鏂扮増 open3d 闇€瑕?'GLIBC_2.27' 鏂囦欢鐨勬敮鎸侊紝 Ubuntu 16.04 绯荤粺涓己澶辫鏂囦欢锛屼笖璇ユ枃浠朵粎瀛樺湪浜?Ubuntu 18.04 鍙婁箣鍚庣殑绯荤粺涓€?
- 濡傛灉鎮ㄥ湪 `import pycocotools` 鏃堕亣鍒扮増鏈敊璇殑闂锛岃繖鏄敱浜?nuscenes-devkit 闇€瑕佸畨瑁?pycocotools锛岀劧鑰?mmdet 渚濊禆浜?mmpycocotools锛屽綋鍓嶇殑瑙ｅ喅鏂规濡備笅鎵€绀猴紝鎴戜滑灏嗕細鍦ㄤ箣鍚庡叏闈㈡敮鎸?pycocotools 锛?
  ```shell
  pip uninstall pycocotools mmpycocotools
  pip install mmpycocotools
  ```

  **娉ㄦ剰**锛?鎴戜滑宸茬粡鍦?0.13.0 鍙婁箣鍚庣殑鐗堟湰涓叏闈㈡敮鎸?pycocotools銆?
- 濡傛灉鎮ㄥ湪瀵煎叆 pycocotools 鐩稿叧鍖呮椂閬囧埌涓嬮潰鐨勯棶棰橈細

  `ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`

  璇峰皢 pycocotools 鐨勭増鏈檷绾ц嚦 2.0.1锛岃繖鏄敱浜庢渶鏂扮増鏈殑 pycocotools 涓?numpy \< 1.20.0 涓嶅吋瀹广€傛垨鑰呴€氳繃涓嬮潰鐨勬柟寮忎粠婧愮爜杩涜缂栬瘧鏉ュ畨瑁呮渶鏂扮増鏈殑 pycocotools 锛?
  `pip install -e "git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI"`

  鎴栬€?
  `pip install -e "git+https://github.com/ppwwyyxx/cocoapi#egg=pycocotools&subdirectory=PythonAPI"`

- 濡傛灉鎮ㄤ娇鐢?cuda-9.0 鐨勭幆澧冨苟閬囧埌鍏充簬 numba 鐨勯敊璇紝 鎮ㄥ簲璇ユ鏌ヤ笅 numba 鐨勭増鏈€傚湪 cuda-9.0 鐜涓紝楂樼増鏈殑 numba 鏄笉鏀寔鐨勶紝鎴戜滑寤鸿瀹夎 numba==0.53.0.

## 濡備綍鏍囨敞鐐逛簯锛?
MMDetection3D 涓嶆敮鎸佺偣浜戞爣娉ㄣ€傛垜浠彁渚涗竴浜涘紑婧愮殑鏍囨敞宸ュ叿渚涘弬鑰冿細

- [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS)
- [LATTE](https://github.com/bernwang/latte)

姝ゅ锛屾垜浠敼杩涗簡 [LATTE](https://github.com/bernwang/latte) 浠ヤ究鏇存柟渚跨殑鏍囨敞銆傛洿澶氱殑缁嗚妭璇峰弬鑰僛杩欓噷](https://arxiv.org/abs/2011.10174)銆?
